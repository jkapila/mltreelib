//! Model tree: growth and routing only. Node models and numerics live in
//! [`crate::node_stats`]; column screening lives in [`crate::preselect`].
//! Every function here does one job.
//!
//! Split scanning follows the reference loop shapes:
//! - mean leaves: rpart's running-sum scan (`rpart/src/anova.c`) — O(1) per
//!   value, split only where the value changes;
//! - linear leaves: LightGBM's grouped accumulators
//!   (`src/treelearner/feature_histogram.cpp`) over per-value Gram groups,
//!   with per-candidate solves on preallocated matrices
//!   (`src/treelearner/linear_tree_learner.cpp`).
//! Everything is row-weighted; missing values are scored in both default
//! directions (LightGBM's `default_left` shape) and the gain picks.
//!
//! The column scan is rayon-parallel above a work threshold (Level-1
//! parallelism) with a **deterministic** tie-break: first-found wins
//! (lowest column index, then lowest threshold), independent of thread
//! completion order.

use std::time::Instant;

use crate::node_stats::{
    design_row, fit_linear, solve_normal_sum, FrameStats, Gram, LeafModel, LeafStats, SolveScratch,
};
use crate::preselect;

pub use crate::node_stats::{LeafModel as NodeModel, LeafStats as NodeStats, LinearModel};

/// Below this much work (rows x columns) the scan stays single-threaded —
/// thread overhead would lose.
const PAR_MIN_WORK: usize = 20_000;

/// Near-tie test: candidates within one part per trillion of the standing
/// best do not replace it (first-found wins). Mirrors rpart's rounding guard
/// (`improve > iscale * 1e-10` in `bsplit.c`) — ulp-level arithmetic noise
/// must not flip equally-good splits, or memberships drift between runs.
fn beats(gain: f64, best: f64) -> bool {
    gain > best * (1.0 + 1e-12)
}

/// Weighted Shannon entropy (bits) over y-as-class-labels (weighted class
/// masses; distinct values are the classes).
fn class_entropy_w(p: &[(f64, f64)]) -> f64 {
    let total: f64 = p.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for &(y, w) in p {
        *counts.entry(y.to_bits()).or_insert(0.0) += w;
    }
    counts
        .values()
        .map(|&c| {
            let q = c / total;
            if q > 0.0 {
                -q * q.log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Weighted Gini impurity over y-as-class-labels.
fn class_gini_w(p: &[(f64, f64)]) -> f64 {
    let total: f64 = p.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for &(y, w) in p {
        *counts.entry(y.to_bits()).or_insert(0.0) += w;
    }
    1.0 - counts
        .values()
        .map(|&c| {
            let q = c / total;
            q * q
        })
        .sum::<f64>()
}

/// Weighted class-frequency total variation `Σ_j |p_L(j) - p_R(j)|`
/// (twoing rule's contrast term).
fn class_total_variation(left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
    let mut lm: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut rm: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let (mut lw, mut rw) = (0.0, 0.0);
    for &(y, w) in left {
        *lm.entry(y.to_bits()).or_insert(0.0) += w;
        lw += w;
    }
    for &(y, w) in right {
        *rm.entry(y.to_bits()).or_insert(0.0) += w;
        rw += w;
    }
    if lw <= 0.0 || rw <= 0.0 {
        return 0.0;
    }
    let mut keys: Vec<u64> = lm.keys().copied().collect();
    keys.extend(rm.keys().copied());
    keys.sort_unstable();
    keys.dedup();
    keys.iter()
        .map(|k| {
            (lm.get(k).copied().unwrap_or(0.0) / lw - rm.get(k).copied().unwrap_or(0.0) / rw).abs()
        })
        .sum()
}

/// Weighted sample standard deviation (M5's SDR ingredient).
fn w_sd(p: &[(f64, f64)]) -> f64 {
    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
    if sw <= 0.0 {
        return 0.0;
    }
    let mean = p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw;
    let var = p.iter().map(|&(y, w)| w * (y - mean) * (y - mean)).sum::<f64>() / sw;
    var.sqrt()
}

/// Total rows across level buckets (categorical scan bookkeeping).
fn levels_total(levels: &[(f64, Vec<(f64, f64)>)]) -> usize {
    levels.iter().map(|(_, rows)| rows.len()).sum()
}

/// Which columns a leaf's linear model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafCols {
    /// Every usable predictor column.
    All,
    /// Only the columns used in splits along the root→leaf path.
    PathUsed,
}

/// What the leaf model is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Linear model over the chosen columns (the model tree).
    Linear,
    /// Plain mean — the CART/rpart leaf. The split criterion then *is*
    /// rpart's deviance criterion.
    Mean,
}

/// Per-column type for the split search. Values are `f64` level codes;
/// a distinct value IS a level. Missing (`None`) is a first-class state,
/// routed by `missing_left` — never imputed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColType {
    /// Threshold splits (on the `split_precision` grid when set).
    Numeric,
    /// Level-partition splits per `cat_order` / `cat_search`.
    Categorical,
}

/// Order in which a categorical column's levels are visited when building
/// the candidate partitions.
#[derive(Clone, Debug, PartialEq)]
pub enum CatOrder {
    /// Most frequent level first (counted at the node).
    FrequencyDesc,
    /// Highest target mean first (CART-style ordering).
    TargetMean,
    /// Explicit level order (codes); levels not listed are appended after,
    /// by descending frequency.
    Explicit(Vec<f64>),
}

/// Which candidate partitions a categorical column contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatSearch {
    /// Cumulative sweep along the order: {top-1}|rest, {top-2}|rest, ...
    /// `k-1` candidates. Default.
    Cumulative,
    /// Each level alone vs the rest, visited in order; stops early at
    /// `gain >= min_gain` on the first acceptable level.
    SingleVsRest,
    /// The union of both candidate sets; best gain wins.
    Both,
}

/// Split-quality criterion. Built-ins run the O(1) stats scan; a custom
/// criterion sees lazy (y, w) slices per candidate. All criteria are
/// sample-weight aware.
pub enum GainFn {
    /// Variance reduction (weighted sum of squares). The regression default.
    Sse,
    /// Information gain: weighted entropy reduction over y-as-class-labels
    /// (distinct y values are the classes).
    Entropy,
    /// Gini impurity reduction over y-as-class-labels.
    Gini,
    /// C4.5 gain ratio: information gain divided by the split information
    /// of the two children (penalizes lopsided splits).
    GainRatio,
    /// Model-tree criterion (Wang/Quinlan M5): weighted standard-deviation
    /// reduction `sd(parent) - Σ (w_k/w)·sd(child_k)`.
    Sdr,
    /// Breiman twoing rule: `2·P_L·P_R·(Σ_j |p_L(j) - p_R(j)|)^2` over
    /// class frequencies — favors balanced splits with contrasting sides.
    Twoing,
    /// User criterion. Receives lazy `(y, w)` slices of the two sides per
    /// candidate split and returns the split quality (higher = better).
    Custom(std::sync::Arc<dyn CustomGain>),
}

/// User-supplied split criterion over lazy (y, w) slices.
pub trait CustomGain: Send + Sync {
    /// Quality of splitting `left` / `right` (higher = better). Each slice
    /// holds the side's `(target, weight)` rows.
    fn gain(&self, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64;
}

/// Optional post-growth pruning cost (bottom-up: a subtree is replaced by
/// its parent leaf when the leaf's encoding cost is lower).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneCost {
    /// No cost pruning (min_gain only).
    None,
    /// Bayesian information criterion: `n·ln(SSE/n) + ln(n)` per leaf,
    /// `ln(n)` per split — universal (regression and class targets).
    Bic,
    /// Mehta/Rissanen/Agrawal MDL node cost over y-as-class-labels.
    Mdl,
}

impl Clone for GainFn {
    fn clone(&self) -> Self {
        match self {
            GainFn::Sse => GainFn::Sse,
            GainFn::Entropy => GainFn::Entropy,
            GainFn::Gini => GainFn::Gini,
            GainFn::GainRatio => GainFn::GainRatio,
            GainFn::Sdr => GainFn::Sdr,
            GainFn::Twoing => GainFn::Twoing,
            GainFn::Custom(f) => GainFn::Custom(f.clone()),
        }
    }
}

impl std::fmt::Debug for GainFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GainFn::Sse => write!(f, "Sse"),
            GainFn::Entropy => write!(f, "Entropy"),
            GainFn::Gini => write!(f, "Gini"),
            GainFn::GainRatio => write!(f, "GainRatio"),
            GainFn::Sdr => write!(f, "Sdr"),
            GainFn::Twoing => write!(f, "Twoing"),
            GainFn::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

/// Growth and leaf parameters.
#[derive(Clone, Debug)]
pub struct ModelTreeParams {
    /// Maximum tree depth (0 = a single leaf).
    pub max_depth: usize,
    /// Minimum rows per child side at a split (and per leaf).
    pub min_rows: usize,
    /// Minimum SSE reduction a split must achieve (noise floor as much as a
    /// tuning knob).
    pub min_gain: f64,
    /// Leaf column policy.
    pub leaf_cols: LeafCols,
    /// Leaf model kind.
    pub leaf: LeafKind,
    /// Ridge penalty on leaf coefficients (intercept unpenalized).
    pub l2: f64,
    /// Lasso penalty on leaf coefficients (intercept unpenalized). With
    /// `l2` this is the elastic net. 0 = off.
    pub l1: f64,
    /// Per-column monotone direction: -1 / 0 / +1 (indexed like `x`'s
    /// columns). A split on a constrained column must order the children's
    /// target means accordingly (XGBoost-style leaf-output constraint).
    pub monotone: Vec<i8>,
    /// Interaction groups: features that may interact with each other
    /// (LightGBM-style). Empty = unconstrained. Groups are disjoint; a
    /// path may only use columns from one group.
    pub interaction_groups: Vec<Vec<usize>>,
    /// Pre-selection: keep only columns whose winnowing-style relevance
    /// (|weighted correlation with y|) reaches this value. `None` = off.
    /// Dropped columns cannot split or enter leaves.
    pub min_relevance: Option<f64>,
    /// Per-column type for the split search (indexed like `x`'s columns).
    /// Empty = all numeric. Categorical columns contribute level-partition
    /// candidates instead of thresholds — no encoding on the growth path.
    pub dtypes: Vec<ColType>,
    /// Split-quality criterion (higher gain = better split).
    pub gain: GainFn,
    /// Candidate-threshold grid for numeric columns: values are rounded to
    /// this grid before the scan (0.1 = loop over tenths), thresholds are
    /// grid values. `None` = exact unique-value scan. Prediction always
    /// compares raw values.
    pub split_precision: Option<f64>,
    /// Order categorical levels are visited in.
    pub cat_order: CatOrder,
    /// Which candidate partitions categorical columns contribute.
    pub cat_search: CatSearch,
    /// Keep each leaf's t-digest (see `ModelTree::quantile` — any tau
    /// post-fit). `false` drops the digest after fit (leaner leaves);
    /// quantile queries then return NaN.
    pub leaf_digest: bool,
    /// Training log verbosity: 0 = silent, 1 = per depth level, 2 = per
    /// node. Populates `Tree::trace` (candidate counts, dtype search,
    /// precision grid, gain taken, timings).
    pub verbose: u8,
    /// Minimum child weight fraction of the node's total weight (grf's
    /// `alpha`): a split's children must each carry at least this share.
    /// 0.0 = off (the `min_rows` count gate alone).
    pub alpha: f64,
    /// grf-style imbalance penalty subtracted from the gain:
    /// `imbalance_penalty * (1/n_left + 1/n_right)`. 0.0 = off.
    pub imbalance_penalty: f64,
    /// Per-node random column subset (random-forest spirit): fraction of
    /// the active columns considered at each node's scan. `None` = all.
    pub col_sample: Option<f64>,
    /// Seed for the per-node column subset (deterministic given the seed).
    pub seed: u64,
    /// Optional post-growth pruning cost (MDL/BIC).
    pub prune: PruneCost,
}

impl Default for ModelTreeParams {
    fn default() -> Self {
        Self {
            // Defaults below are the bake-off winner on the arena datasets
            // (examples/default_search.py): linear leaves at shallow depth
            // beat plain OLS and every baseline bar. Ridge (l2) keeps leaf
            // coefficients sane in closed form; l1 is OFF by default -
            // coordinate descent is slow in the split scan and the gain over
            // ridge is marginal (see docs/benchmarks/elo.md).
            max_depth: 4,
            min_rows: 10,
            // Noise floor as much as a tuning knob: fitted-leaf gains below
            // this are numerical, not structural.
            min_gain: 1e-9,
            leaf_cols: LeafCols::All,
            leaf: LeafKind::Linear,
            l2: 0.1,
            l1: 0.0,
            monotone: Vec::new(),
            interaction_groups: Vec::new(),
            min_relevance: None,
            dtypes: Vec::new(),
            gain: GainFn::Sse,
            split_precision: Some(0.001),
            cat_order: CatOrder::FrequencyDesc,
            cat_search: CatSearch::Cumulative,
            leaf_digest: true,
            verbose: 0,
            alpha: 0.0,
            imbalance_penalty: 0.0,
            col_sample: None,
            seed: 42,
            prune: PruneCost::None,
        }
    }
}

/// One binary split: values `<= threshold` go left (numeric), or
/// `left_levels` membership goes left (categorical, `cat = Some(i)` indexes
/// `Tree::cat_parts`).
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub col: usize,
    pub threshold: f64,
    pub missing_left: bool,
    /// Goodness of split (parent SSE - children SSE) — the raw material of
    /// feature importance.
    pub gain: f64,
    /// Categorical partition handle: `Some(i)` = `Tree::cat_parts[i]` lists
    /// the level codes routed left. `None` = threshold split.
    pub cat: Option<usize>,
}

/// A prediction decomposed along its decision path (DALEX-style break-down).
#[derive(Clone, Debug)]
pub struct BreakDown {
    /// Training-target mean — the starting level.
    pub baseline: f64,
    /// `(column, contribution)` in decision-path order.
    pub steps: Vec<(usize, f64)>,
    /// Within-leaf model adjustment (identically 0 for mean leaves).
    pub residual: f64,
    pub prediction: f64,
}

#[derive(Clone, Debug)]
struct Node {
    split: Option<Split>,
    left: Option<usize>,
    right: Option<usize>,
    leaf: Option<LeafModel>,
    n: usize,
    stats: FrameStats,
}

/// Per-node timing breakdown — the debug-on-every-level instrumentation.
#[derive(Clone, Debug, Default)]
pub struct NodeProfile {
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
    pub candidates: usize,
    pub group_ns: u64,
    pub scan_ns: u64,
    pub leaf_ns: u64,
}

/// Timing totals per depth level.
#[derive(Clone, Debug, Default)]
pub struct DepthProfile {
    pub depth: usize,
    pub nodes: usize,
    pub rows: usize,
    pub cols: usize,
    pub candidates: usize,
    pub group_ns: u64,
    pub scan_ns: u64,
    pub leaf_ns: u64,
}

/// A fitted model tree.
#[derive(Clone, Debug)]
pub struct ModelTree {
    pub col_names: Vec<String>,
    pub params: ModelTreeParams,
    /// Columns that survived pre-selection (original indices).
    pub active_cols: Vec<usize>,
    nodes: Vec<Node>,
    root: usize,
    profiles: Vec<NodeProfile>,
    /// Categorical partitions referenced by `Split::cat`.
    cat_parts: Vec<Vec<f64>>,
    /// Training log (populated when `params.verbose > 0`).
    pub trace: Vec<String>,
}

impl ModelTree {
    /// Grid-aware left test shared by fit partitioning, prediction and
    /// break_down: numeric values compare on the `split_precision` grid
    /// (raw when unset); categorical splits test partition membership.
    fn goes_left(&self, sp: &Split, v: f64) -> bool {
        if let Some(i) = sp.cat {
            return self.cat_parts[i].contains(&v);
        }
        match self.params.split_precision {
            Some(p) if p > 0.0 => {
                let q = 1.0 / p;
                (v * q).round() / q <= sp.threshold
            }
            _ => v <= sp.threshold,
        }
    }
}

/// Work of one column scan, merged into the node profile.
#[derive(Clone, Copy, Debug, Default)]
struct ScanStats {
    group_ns: u64,
    scan_ns: u64,
    candidates: usize,
}

impl ModelTree {
    /// Fit on row-major data (`None` = missing). Column names default to
    /// `x0`, `x1`…
    pub fn fit(
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        params: ModelTreeParams,
    ) -> Result<Self, String> {
        let d = x.first().map(|r| r.len()).unwrap_or(0);
        let names: Vec<String> = (0..d).map(|j| format!("x{j}")).collect();
        Self::fit_named(names, x, y, params)
    }

    /// Fit with explicit column names.
    pub fn fit_named(
        col_names: Vec<String>,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        params: ModelTreeParams,
    ) -> Result<Self, String> {
        let n = y.len();
        Self::fit_weighted_named(col_names, x, y, &vec![1.0; n], params)
    }

    /// Fit with sample weights (row weightage). Zero-weight rows are
    /// ignored entirely; negative weights are an error.
    pub fn fit_weighted(
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        w: &[f64],
        params: ModelTreeParams,
    ) -> Result<Self, String> {
        let d = x.first().map(|r| r.len()).unwrap_or(0);
        let names: Vec<String> = (0..d).map(|j| format!("x{j}")).collect();
        Self::fit_weighted_named(names, x, y, w, params)
    }

    /// Weighted fit with explicit column names.
    pub fn fit_weighted_named(
        col_names: Vec<String>,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        w: &[f64],
        params: ModelTreeParams,
    ) -> Result<Self, String> {
        let n_all = y.len();
        if x.len() != n_all || w.len() != n_all {
            return Err("x, y and w length mismatch".to_string());
        }
        let d = col_names.len();
        if d == 0 {
            return Err("no predictor columns".to_string());
        }
        for r in x {
            if r.len() != d {
                return Err("row width != number of column names".to_string());
            }
        }
        if params.min_rows == 0 {
            return Err("min_rows must be >= 1".to_string());
        }
        if params.monotone.iter().any(|&m| m < -1 || m > 1) {
            return Err("monotone entries must be -1, 0 or 1".to_string());
        }
        if !params.monotone.is_empty() && params.monotone.len() != d {
            return Err("monotone must have one entry per column".to_string());
        }

        // Column-major; rows with missing targets or zero weight drop
        // (no signal to learn / explicitly absent).
        let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n_all); d];
        let mut yv: Vec<f64> = Vec::with_capacity(n_all);
        let mut wv: Vec<f64> = Vec::with_capacity(n_all);
        for (r, row) in x.iter().enumerate() {
            if w[r] < 0.0 {
                return Err("weights must be non-negative".to_string());
            }
            let Some(t) = y[r] else { continue };
            if w[r] == 0.0 {
                continue;
            }
            for j in 0..d {
                cols[j].push(row[j]);
            }
            yv.push(t);
            wv.push(w[r]);
        }
        if yv.is_empty() {
            return Err("no rows with an observed target and positive weight".to_string());
        }

        // Pre-selection: keep only informative columns (original indices).
        let active_cols: Vec<usize> = match params.min_relevance {
            Some(thr) => {
                let kept = preselect::select_columns(&cols, &yv, &wv, thr);
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            None => (0..d).collect(),
        };
        // Shrinking a monotone vector would break indices; keep it aligned.

        let rows: Vec<usize> = (0..yv.len()).collect();
        let mut grower = Grower {
            cols: &cols,
            y: &yv,
            w: &wv,
            params: &params,
            active: &active_cols,
            nodes: Vec::new(),
            profiles: Vec::new(),
            cat_parts: Vec::new(),
            trace: Vec::new(),
            costs: Vec::new(),
            rng_counter: 0,
        };
        if params.verbose > 0 {
            grower.trace.push(format!(
                "scan: gain={:?} split_precision={:?} cat_search={:?} ({:?}) leaf={:?} leaf_digest={}",
                params.gain, params.split_precision, params.cat_search, params.cat_order,
                params.leaf, params.leaf_digest
            ));
        }
        let root = grower.grow(&rows, 0, &mut Vec::new());
        // Destructure to release the borrows on params/active_cols before
        // moving them into the tree.
        let Grower {
            nodes, profiles, cat_parts, mut trace, ..
        } = grower;
        if params.verbose > 0 {
            // Per-depth summary from the node profiles.
            let mut by_depth: std::collections::BTreeMap<usize, (usize, usize, usize, u64)> =
                std::collections::BTreeMap::new();
            for p in &profiles {
                let e = by_depth.entry(p.depth).or_insert((0, 0, 0, 0));
                e.0 += 1;
                e.1 += p.rows;
                e.2 += p.candidates;
                e.3 += p.scan_ns / 1_000_000;
            }
            let mut lines: Vec<String> = by_depth
                .iter()
                .map(|(d, (nodes, rows, candidates, scan_ms))| {
                    format!(
                        "depth {d}: nodes={nodes} rows={rows} candidates={candidates} scan_ms={scan_ms}"
                    )
                })
                .collect();
            lines.extend(trace);
            trace = lines;
        }
        Ok(Self {
            col_names,
            params,
            active_cols,
            nodes,
            root,
            profiles,
            cat_parts,
            trace,
        })
    }

    /// Leaf node index a row lands in (shared walk with `predict`).
    fn leaf_index(&self, row: &[Option<f64>]) -> usize {
        let mut node = self.root;
        while let (Some(sp), Some(l), Some(r)) = (
            &self.nodes[node].split,
            self.nodes[node].left,
            self.nodes[node].right,
        ) {
            let go_left = match row.get(sp.col).and_then(|v| *v) {
                Some(v) => self.goes_left(sp, v),
                None => sp.missing_left,
            };
            node = if go_left { l } else { r };
        }
        node
    }

    /// Conditional quantile prediction: `tau` in [0, 1], one value per row.
    /// Reads each leaf's t-digest (see `LeafStats::digest`), so ANY tau is
    /// answerable post-fit without a refit. Returns NaN when the leaf's
    /// digest was dropped (`leaf_digest = false`) or is empty.
    pub fn quantile(&self, x: &[Vec<Option<f64>>], tau: f64) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let idx = self.leaf_index(row);
                let leaf = self.nodes[idx]
                    .leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model");
                leaf.stats.quantile(tau).unwrap_or(f64::NAN)
            })
            .collect()
    }

    /// Predict for row-major `x`. NAs never block a prediction.
    pub fn predict(&self, x: &[Vec<Option<f64>>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut node = &self.nodes[self.root];
                while let (Some(sp), Some(l), Some(r)) = (&node.split, node.left, node.right) {
                    let go_left = match row.get(sp.col).and_then(|v| *v) {
                        Some(v) => self.goes_left(sp, v),
                        None => sp.missing_left,
                    };
                    node = &self.nodes[if go_left { l } else { r }];
                }
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict(row)
            })
            .collect()
    }

    /// All splits in the tree (introspection / structure recovery).
    pub fn splits(&self) -> Vec<Split> {
        self.nodes.iter().filter_map(|n| n.split).collect()
    }

    /// Graphviz DOT for the tree. We only describe the graph - layout and
    /// rendering belong to graphviz (the wheel we do not reinvent):
    /// `pgv.AGraph(string=model.to_dot()).draw("tree.svg", prog="dot")`.
    pub fn to_dot(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let mut s = String::from(
            "digraph tree {\n  rankdir=LR;\n  node [shape=box, style=\"rounded,filled\", fillcolor=\"#fafafa\", fontname=\"monospace\", fontsize=10];\n  edge [fontname=\"monospace\", fontsize=9, color=\"#888888\"];\n",
        );
        // Heap numbering (root 1, children 2i/2i+1) - same ids as the text
        // print, so the two outputs line up.
        let mut num = vec![0usize; self.nodes.len()];
        fn number(t: &ModelTree, i: usize, k: usize, num: &mut Vec<usize>) {
            num[i] = k;
            if let (Some(_), Some(l), Some(r)) = (t.nodes[i].split, t.nodes[i].left, t.nodes[i].right) {
                number(t, l, k * 2, num);
                number(t, r, k * 2 + 1, num);
            }
        }
        number(self, self.root, 1, &mut num);
        for (i, nd) in self.nodes.iter().enumerate() {
            let (l1, l2) = match nd.split {
                Some(sp) => (
                    match sp.cat {
                        Some(ci) => format!(
                            "{} in {{{}}}",
                            self.col_names[sp.col],
                            self.cat_parts[ci]
                                .iter()
                                .map(|v| fmt_cut(*v))
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                        None => format!("{} < {}", self.col_names[sp.col], fmt_cut(sp.threshold)),
                    },
                    format!("n={}  dev={:.3}", nd.stats.n, nd.stats.sse),
                ),
                None => (
                    "leaf".to_string(),
                    format!("n={}  yval={:.3}", nd.stats.n, nd.stats.mean),
                ),
            };
            let stroke = if nd.split.is_some() { "#333333" } else { "#22aa77" };
            s += &format!(
                "  n{} [label=\"{}\\n{}\", color=\"{stroke}\"];\n",
                num[i],
                esc(&l1),
                esc(&l2)
            );
        }
        for (i, nd) in self.nodes.iter().enumerate() {
            if let (Some(sp), Some(l), Some(r)) = (nd.split, nd.left, nd.right) {
                let (el, er) = match sp.cat {
                    Some(ci) => {
                        let set = self.cat_parts[ci]
                            .iter()
                            .map(|v| fmt_cut(*v))
                            .collect::<Vec<_>>()
                            .join(",");
                        (format!("in {{{set}}}"), format!("not in {{{set}}}"))
                    }
                    None => {
                        let cut = fmt_cut(sp.threshold);
                        (format!("< {cut}"), format!(">= {cut}"))
                    }
                };
                s += &format!("  n{} -> n{} [label=\"{}\"];\n", num[i], num[l], esc(&el));
                s += &format!("  n{} -> n{} [label=\"{}\"];\n", num[i], num[r], esc(&er));
            }
        }
        s += "}\n";
        s
    }

    /// Feature importance — each column's share of the total split gain
    /// (rpart's "variable importance": summed goodness of split). Returns
    /// (column name, share) pairs sorted descending; shares sum to 1.
    pub fn feature_importances(&self) -> Vec<(String, f64)> {
        let mut totals = vec![0.0f64; self.col_names.len()];
        let mut sum = 0.0;
        for nd in &self.nodes {
            if let Some(sp) = nd.split {
                totals[sp.col] += sp.gain;
                sum += sp.gain;
            }
        }
        let mut out: Vec<(String, f64)> = self
            .col_names
            .iter()
            .enumerate()
            .map(|(j, name)| (name.clone(), if sum > 0.0 { totals[j] / sum } else { 0.0 }))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Break one prediction into path contributions (DALEX-style break-down
    /// for trees): `baseline` is the training mean; each split on the row's
    /// route contributes (child node mean − parent node mean) to its column,
    /// in path order; `residual` is the within-leaf model adjustment. They
    /// telescope: baseline + Σ steps + residual == prediction.
    pub fn break_down(&self, x: &[Option<f64>]) -> BreakDown {
        let mut node = &self.nodes[self.root];
        let baseline = node.stats.mean;
        let mut steps: Vec<(usize, f64)> = Vec::new();
        while let (Some(sp), Some(l), Some(r)) = (&node.split, node.left, node.right) {
            let go_left = match x.get(sp.col).and_then(|v| *v) {
                Some(v) => v <= sp.threshold,
                None => sp.missing_left,
            };
            let child = &self.nodes[if go_left { l } else { r }];
            steps.push((sp.col, child.stats.mean - node.stats.mean));
            node = child;
        }
        let prediction = node
            .leaf
            .as_ref()
            .expect("tree invariant: leaf holds a model")
            .predict(x);
        BreakDown {
            baseline,
            steps,
            residual: prediction - node.stats.mean,
            prediction,
        }
    }

    /// Split columns along each root→leaf path — introspection for
    /// constraint checking (interaction groups: every path must stay within
    /// one group) and structure recovery.
    pub fn split_paths(&self) -> Vec<Vec<usize>> {
        fn walk(
            nodes: &[Node],
            i: usize,
            path: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            let n = &nodes[i];
            match (&n.split, n.left, n.right) {
                (Some(sp), Some(l), Some(r)) => {
                    path.push(sp.col);
                    walk(nodes, l, path, out);
                    walk(nodes, r, path, out);
                    path.pop();
                }
                _ => out.push(path.clone()),
            }
        }
        let mut out = Vec::new();
        walk(&self.nodes, self.root, &mut Vec::new(), &mut out);
        out
    }

    /// Leaf models with their row counts.
    pub fn leaves(&self) -> Vec<(&LeafModel, usize)> {
        self.nodes
            .iter()
            .filter_map(|n| n.leaf.as_ref().map(|m| (m, n.n)))
            .collect()
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.leaves().len()
    }

    /// Per-node timings (every critical event, every level).
    pub fn profile_nodes(&self) -> &[NodeProfile] {
        &self.profiles
    }

    /// Timings aggregated per depth level.
    pub fn profile(&self) -> Vec<DepthProfile> {
        let mut levels: Vec<DepthProfile> = Vec::new();
        for p in &self.profiles {
            while levels.len() <= p.depth {
                levels.push(DepthProfile {
                    depth: levels.len(),
                    ..Default::default()
                });
            }
            let l = &mut levels[p.depth];
            l.nodes += 1;
            l.rows += p.rows;
            l.cols += p.cols;
            l.candidates += p.candidates;
            l.group_ns += p.group_ns;
            l.scan_ns += p.scan_ns;
            l.leaf_ns += p.leaf_ns;
        }
        levels
    }
}

struct Grower<'a> {
    cols: &'a [Vec<Option<f64>>],
    y: &'a [f64],
    w: &'a [f64],
    params: &'a ModelTreeParams,
    active: &'a [usize],
    nodes: Vec<Node>,
    profiles: Vec<NodeProfile>,
    /// Categorical partitions referenced by `Split::cat` (moved into the
    /// fitted tree).
    cat_parts: Vec<Vec<f64>>,
    /// Training log lines (`params.verbose`).
    trace: Vec<String>,
    /// Prune-cost bookkeeping aligned with node indices (`prune != None`).
    costs: Vec<f64>,
    /// Per-node counter for the random column subset (seeded, deterministic).
    rng_counter: u64,
}

impl<'a> Grower<'a> {
    /// Grid-aware left test (see `ModelTree::goes_left`).
    fn goes_left(&self, sp: &Split, v: f64) -> bool {
        if let Some(i) = sp.cat {
            return self.cat_parts[i].contains(&v);
        }
        match self.params.split_precision {
            Some(p) if p > 0.0 => {
                let q = 1.0 / p;
                (v * q).round() / q <= sp.threshold
            }
            _ => v <= sp.threshold,
        }
    }

    fn add(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn grow(&mut self, rows: &[usize], depth: usize, path: &mut Vec<usize>) -> usize {
        let mut prof = NodeProfile {
            depth,
            rows: rows.len(),
            ..Default::default()
        };
        let best = if depth < self.params.max_depth {
            self.best_split(rows, path, &mut prof)
        } else {
            None
        };
        if self.params.verbose >= 2 {
            self.trace.push(format!(
                "node depth={} rows={} cols={} candidates={} scan_ms={} leaf_ms={}",
                prof.depth,
                prof.rows,
                prof.cols,
                prof.candidates,
                prof.scan_ns / 1_000_000,
                prof.leaf_ns / 1_000_000
            ));
        }

        match best {
            None => {
                let t = Instant::now();
                let leaf = self.make_leaf(rows, path);
                prof.leaf_ns += t.elapsed().as_nanos() as u64;
                self.profiles.push(prof);
                let cost = self.leaf_cost(rows);
                self.add(Node {
                    split: None,
                    left: None,
                    right: None,
                    leaf: Some(leaf),
                    n: rows.len(),
                    stats: FrameStats::from_ty(self.y, self.w, rows),
                });
                self.costs.push(cost);
                self.nodes.len() - 1
            }
            Some((sp, lrows, rrows)) => {
                path.push(sp.col);
                let l = self.grow(&lrows, depth + 1, path);
                let r = self.grow(&rrows, depth + 1, path);
                path.pop();
                if self.params.prune != PruneCost::None {
                    // Bottom-up cost pruning: keep the subtree only when it
                    // codes cheaper than this node as one leaf.
                    let leaf_c = self.leaf_cost(rows);
                    let split_bit = (rows.len() as f64).ln().max(0.0);
                    let sub_c = self.costs[l] + self.costs[r] + split_bit;
                    if leaf_c <= sub_c {
                        let t = Instant::now();
                        let leaf = self.make_leaf(rows, path);
                        prof.leaf_ns += t.elapsed().as_nanos() as u64;
                        self.profiles.push(prof);
                        self.add(Node {
                            split: None,
                            left: None,
                            right: None,
                            leaf: Some(leaf),
                            n: rows.len(),
                            stats: FrameStats::from_ty(self.y, self.w, rows),
                        });
                        self.costs.push(leaf_c);
                        return self.nodes.len() - 1;
                    }
                    self.profiles.push(prof);
                    self.add(Node {
                        split: Some(sp),
                        left: Some(l),
                        right: Some(r),
                        leaf: None,
                        n: rows.len(),
                        stats: FrameStats::from_ty(self.y, self.w, rows),
                    });
                    self.costs.push(sub_c);
                    return self.nodes.len() - 1;
                }
                self.profiles.push(prof);
                self.add(Node {
                    split: Some(sp),
                    left: Some(l),
                    right: Some(r),
                    leaf: None,
                    n: rows.len(),
                    stats: FrameStats::from_ty(self.y, self.w, rows),
                });
                self.costs.push(0.0);
                self.nodes.len() - 1
            }
        }
    }

    /// Encoding cost of treating `rows` as one leaf (prune criterion).
    fn leaf_cost(&self, rows: &[usize]) -> f64 {
        let n = rows.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        match self.params.prune {
            PruneCost::None => 0.0,
            PruneCost::Bic => {
                let sw: f64 = rows.iter().map(|&r| self.w[r]).sum();
                let mean =
                    rows.iter().map(|&r| self.w[r] * self.y[r]).sum::<f64>() / sw.max(1e-300);
                let sse: f64 = rows
                    .iter()
                    .map(|&r| self.w[r] * (self.y[r] - mean) * (self.y[r] - mean))
                    .sum();
                n * (sse / n).max(1e-300).ln() + n.ln()
            }
            PruneCost::Mdl => {
                // Mehta/Rissanen/Agrawal node cost over y-as-class-labels
                // (weighted class masses).
                let mut counts: std::collections::HashMap<u64, f64> =
                    std::collections::HashMap::new();
                for &r in rows {
                    *counts.entry(self.y[r].to_bits()).or_insert(0.0) += self.w[r];
                }
                let k = counts.len() as f64;
                let total: f64 = counts.values().sum();
                let data: f64 = counts
                    .values()
                    .map(|&c| if c > 0.0 { c * (total / c).ln() } else { 0.0 })
                    .sum();
                data + (k - 1.0) / 2.0 * (total / 2.0).ln() + k / 2.0 * (k / 2.0).ln().max(0.0)
            }
        }
    }

    /// Columns a leaf model uses for a given root→leaf path (restricted to
    /// the pre-selected active columns).
    fn chosen_cols(&self, path: &[usize]) -> Vec<usize> {
        match self.params.leaf_cols {
            LeafCols::All => self.active.to_vec(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path
                    .iter()
                    .copied()
                    .filter(|c| self.active.contains(c))
                    .collect();
                p.sort_unstable();
                p.dedup();
                p
            }
        }
    }

    /// Interaction constraint (LightGBM-style): a path may only use columns
    /// from one interaction group; ungrouped columns never interact.
    fn interact_ok(&self, path: &[usize], j: usize) -> bool {
        let g = &self.params.interaction_groups;
        if g.is_empty() {
            return true;
        }
        let group_of = |c: usize| g.iter().position(|grp| grp.contains(&c));
        match group_of(j) {
            None => path.iter().all(|&p| group_of(p).is_none()),
            Some(gj) => path.iter().all(|&p| group_of(p) == Some(gj)),
        }
    }

    /// Monotone check: a split on constrained column j must order the
    /// children's (weighted) target means in the constraint's direction.
    fn mono_ok(&self, j: usize, mean_l: f64, mean_r: f64) -> bool {
        if self.params.monotone.is_empty() {
            return true;
        }
        let m = self.params.monotone[j];
        m == 0 || (mean_l - mean_r) * (m as f64) <= 0.0
    }

    /// Best split across columns. Columns scan in parallel above a work
    /// threshold; the winner is chosen deterministically (gain, then
    /// scan order: lowest column index, lowest threshold, missing_left
    /// first) — never by thread completion order.
    fn best_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        let m = self.params.min_rows;
        if rows.len() < 2 * m {
            return None;
        }

        let mut child_path = path.to_vec();
        let mut candidates: Vec<(usize, Option<(f64, Split, Option<Vec<f64>>)>, ScanStats)> =
            Vec::new();
        let work = rows.len() * self.cols.len();
        let mut cols: Vec<usize> = self
            .active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .collect();
        // Per-node random column subset (random-forest spirit): a seeded
        // xorshift draw, deterministic across runs and threads.
        if let Some(frac) = self.params.col_sample {
            let k = ((cols.len() as f64) * frac).round().max(1.0) as usize;
            if k < cols.len() {
                let mut rng = self.params.seed
                    ^ (self.rng_counter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                for i in 0..k {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let j = i + (rng as usize) % (cols.len() - i);
                    cols.swap(i, j);
                }
                cols.truncate(k);
                cols.sort_unstable(); // deterministic scan order
            }
        }
        self.rng_counter += 1;

        if work >= PAR_MIN_WORK {
            use rayon::prelude::*;
            let mut out: Vec<(usize, Option<(f64, Split, Option<Vec<f64>>)>, ScanStats)> = cols
                .par_iter()
                .map(|&j| {
                    let mut cp = path.to_vec();
                    cp.push(j);
                    let (found, st) = self.scan_column(rows, j, &cp);
                    (j, found, st)
                })
                .collect();
            out.sort_by_key(|(j, _, _)| *j); // deterministic order
            candidates = out;
        } else {
            for &j in &cols {
                child_path.push(j);
                let (found, st) = self.scan_column(rows, j, &child_path);
                child_path.pop();
                candidates.push((j, found, st));
            }
        }

        let mut best: Option<(f64, Split, Option<Vec<f64>>, usize, f64)> = None;
        for (j, found, st) in candidates {
            prof.cols += 1;
            prof.group_ns += st.group_ns;
            prof.scan_ns += st.scan_ns;
            prof.candidates += st.candidates;
            if let Some((gain, sp, part)) = found {
                let better = match &best {
                    None => true,
                    Some((g, _, _, bj, bt)) => {
                        beats(gain, *g)
                            || (!beats(*g, gain) && (j, sp.threshold) < (*bj, *bt))
                    }
                };
                if better {
                    best = Some((gain, sp, part, j, sp.threshold));
                }
            }
        }

        best.map(|(_, mut s, part, ..)| {
            if let Some(p) = part {
                s.cat = Some(self.cat_parts.len());
                self.cat_parts.push(p);
            }
            let (mut lv, mut rv) = (Vec::new(), Vec::new());
            for &r in rows {
                match self.cols[s.col][r] {
                    Some(v) => {
                        if self.goes_left(&s, v) {
                            lv.push(r);
                        } else {
                            rv.push(r);
                        }
                    }
                    None => {
                        if s.missing_left {
                            lv.push(r);
                        } else {
                            rv.push(r);
                        }
                    }
                }
            }
            (s, lv, rv)
        })
    }

    /// Dispatch one column's scan to the dtype/leaf-kind-specific loop.
    /// The hit carries an optional categorical partition (registered by
    /// `best_split` into `cat_parts`).
    fn scan_column(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        if self.col_type(j) == ColType::Categorical {
            return self.scan_column_categorical(rows, j);
        }
        let (hit, st) = match self.params.leaf {
            LeafKind::Mean => self.scan_column_mean(rows, j),
            LeafKind::Linear => self.scan_column_linear(rows, j, path),
        };
        (hit.map(|(g, s)| (g, s, None)), st)
    }

    /// Column type for the split search: `params.dtypes` when set, else
    /// numeric.
    fn col_type(&self, j: usize) -> ColType {
        self.params.dtypes.get(j).copied().unwrap_or(ColType::Numeric)
    }

    /// Split quality of a partition under a non-SSE criterion (higher =
    /// better). All criteria are sample-weight aware; a custom criterion
    /// gets the lazy `(y, w)` slices of the two sides.
    fn gain_of(&self, gfn: &GainFn, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
        let wsum = |p: &[(f64, f64)]| p.iter().map(|&(_, w)| w).sum::<f64>();
        let mut all: Vec<(f64, f64)> = Vec::with_capacity(left.len() + right.len());
        all.extend_from_slice(left);
        all.extend_from_slice(right);
        let (wl, wr) = (wsum(left), wsum(right));
        let w = wl + wr;
        if w <= 0.0 {
            return 0.0;
        }
        match gfn {
            GainFn::Sse => {
                let sse = |p: &[(f64, f64)]| {
                    let sw = wsum(p);
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sy: f64 = p.iter().map(|&(y, wt)| wt * y).sum();
                    let sy2: f64 = p.iter().map(|&(y, wt)| wt * y * y).sum();
                    sy2 - sy * sy / sw
                };
                sse(&all) - sse(left) - sse(right)
            }
            GainFn::Entropy => {
                class_entropy_w(&all) - (wl / w) * class_entropy_w(left) - (wr / w) * class_entropy_w(right)
            }
            GainFn::Gini => {
                class_gini_w(&all) - (wl / w) * class_gini_w(left) - (wr / w) * class_gini_w(right)
            }
            GainFn::GainRatio => {
                let ig =
                    class_entropy_w(&all) - (wl / w) * class_entropy_w(left) - (wr / w) * class_entropy_w(right);
                let info = |x: f64| if x > 0.0 { -x * x.log2() } else { 0.0 };
                let split_info = info(wl / w) + info(wr / w);
                if split_info <= 0.0 {
                    0.0
                } else {
                    ig / split_info
                }
            }
            GainFn::Sdr => w_sd(&all) - (wl / w) * w_sd(left) - (wr / w) * w_sd(right),
            GainFn::Twoing => {
                let tv = class_total_variation(left, right);
                2.0 * (wl / w) * (wr / w) * tv * tv
            }
            GainFn::Custom(f) => f.gain(left, right),
        }
    }

    /// Categorical scan — binary level partitions per `cat_order` /
    /// `cat_search`. Levels are distinct non-missing values; missing rows
    /// are scored in both directions. Monotone constraints are ignored
    /// here (they shape numeric trends). Gain ranking for `LeafKind::Linear`
    /// columns also uses `params.gain` on the y-slices (partition quality),
    /// independent of the leaf fit.
    fn scan_column_categorical(
        &self,
        rows: &[usize],
        j: usize,
    ) -> (Option<(f64, Split, Option<Vec<f64>>)>, ScanStats) {
        let mut st = ScanStats::default();
        let t0 = Instant::now();
        let mut missing_yws: Vec<(f64, f64)> = Vec::new();
        let mut by_level: std::collections::HashMap<u64, Vec<(f64, f64)>> =
            std::collections::HashMap::new();
        for &r in rows {
            let (y, w) = (self.y[r], self.w[r]);
            match self.cols[j][r] {
                Some(v) => {
                    by_level.entry(v.to_bits()).or_default().push((y, w));
                }
                None => missing_yws.push((y, w)),
            }
        }
        let mut levels: Vec<(f64, Vec<(f64, f64)>)> = by_level
            .into_iter()
            .map(|(b, ys)| (f64::from_bits(b), ys))
            .collect();
        if levels.len() < 2 {
            st.group_ns = t0.elapsed().as_nanos() as u64;
            return (None, st);
        }
        // Level order (deterministic: ties by level code ascending).
        match &self.params.cat_order {
            CatOrder::FrequencyDesc => {
                levels.sort_by(|a, b| {
                    b.1.len()
                        .cmp(&a.1.len())
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            CatOrder::TargetMean => {
                let wmean = |p: &[(f64, f64)]| {
                    let sw: f64 = p.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        0.0
                    } else {
                        p.iter().map(|&(y, w)| w * y).sum::<f64>() / sw
                    }
                };
                levels.sort_by(|a, b| {
                    wmean(&b.1)
                        .partial_cmp(&wmean(&a.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            CatOrder::Explicit(order) => {
                let rank = |v: f64| {
                    order
                        .iter()
                        .position(|o| *o == v)
                        .unwrap_or(usize::MAX)
                };
                levels.sort_by(|a, b| {
                    rank(a.0)
                        .cmp(&rank(b.0))
                        .then(b.1.len().cmp(&a.1.len()))
                        .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
        }
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let k = levels.len();
        let m = self.params.min_rows as f64;
        // Candidate partitions as level-index sets (left side).
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        match self.params.cat_search {
            CatSearch::Cumulative => {
                for i in 1..k {
                    candidates.push((0..i).collect());
                }
            }
            CatSearch::SingleVsRest => {
                for i in 0..k {
                    candidates.push(vec![i]);
                }
            }
            CatSearch::Both => {
                for i in 1..k {
                    candidates.push((0..i).collect());
                }
                for i in 0..k {
                    candidates.push(vec![i]);
                }
            }
        }

        let wsum = |p: &[(f64, f64)]| p.iter().map(|&(_, w)| w).sum::<f64>();
        let total_w: f64 = wsum(&missing_yws) + levels.iter().map(|(_, r)| wsum(r)).sum::<f64>();
        let mut best: Option<(f64, Vec<f64>, bool)> = None; // (gain, left levels, missing_left)
        for left_idx in candidates.iter() {
            let in_left: std::collections::HashSet<usize> = left_idx.iter().copied().collect();
            let mut ln = 0usize;
            let mut left_yws: Vec<(f64, f64)> = Vec::new();
            let mut right_yws: Vec<(f64, f64)> = Vec::new();
            for (li, (_, ys)) in levels.iter().enumerate() {
                if in_left.contains(&li) {
                    ln += ys.len();
                    left_yws.extend_from_slice(ys);
                } else {
                    right_yws.extend_from_slice(ys);
                }
            }
            let rn = levels_total(&levels) - ln;
            if (ln as f64) < m || (rn as f64) < m {
                continue;
            }
            st.candidates += 1;
            for missing_left in [true, false] {
                let (aln, arn) = if missing_left {
                    (ln + missing_yws.len(), rn)
                } else {
                    (ln, rn + missing_yws.len())
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                let mut ls = left_yws.clone();
                let mut rs = right_yws.clone();
                if missing_left {
                    ls.extend_from_slice(&missing_yws);
                } else {
                    rs.extend_from_slice(&missing_yws);
                }
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0
                    && (wsum(&ls) < self.params.alpha * total_w
                        || wsum(&rs) < self.params.alpha * total_w)
                {
                    continue;
                }
                let gain = self.gain_of(&self.params.gain, &ls, &rs);
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    let left_levels: Vec<f64> =
                        left_idx.iter().map(|&li| levels[li].0).collect();
                    best = Some((gain, left_levels, missing_left));
                }
                // Single-vs-rest loops in `cat_order` and stops at the
                // first acceptable level (gain > min_gain, checked above).
                if matches!(self.params.cat_search, CatSearch::SingleVsRest) {
                    break;
                }
            }
        }

        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (
            best.map(|(gain, left_levels, missing_left)| {
                (
                    gain,
                    Split {
                        col: j,
                        threshold: 0.0,
                        missing_left,
                        gain,
                        cat: None,
                    },
                    Some(left_levels),
                )
            }),
            st,
        )
    }

    /// Mean-leaf scan — rpart's `anova.c` shape: weighted running sums per
    /// side, O(1) per value, split only where the value changes.
    fn scan_column_mean(&self, rows: &[usize], j: usize) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let m = self.params.min_rows as f64;
        let t0 = Instant::now();
        let mut pairs: Vec<(f64, f64, f64)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, self.y[r], self.w[r])))
            .collect();
        if pairs.len() < 2 {
            return (None, st);
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));
        // split_precision grid: values round to the grid before the scan;
        // candidates are grid boundaries (midpoints when unset).
        if let Some(p) = self.params.split_precision {
            if p > 0.0 {
                let q = 1.0 / p;
                for t in pairs.iter_mut() {
                    t.0 = (t.0 * q).round() / q;
                }
            }
        }
        let yws_sorted: Vec<(f64, f64)> = pairs.iter().map(|&(_, y, w)| (y, w)).collect();

        // Missing rows: one extra group, scored in both directions.
        let mut mn = 0usize;
        let mut missing_yws: Vec<(f64, f64)> = Vec::new();
        let (mut msw, mut mswy, mut mswy2) = (0.0, 0.0, 0.0);
        for &r in rows {
            if self.cols[j][r].is_none() {
                mn += 1;
                let (y, w) = (self.y[r], self.w[r]);
                missing_yws.push((y, w));
                msw += w;
                mswy += w * y;
                mswy2 += w * y * y;
            }
        }
        let mut tn = 0usize;
        let (mut tsw, mut tswy, mut tswy2) = (0.0, 0.0, 0.0);
        for &(_, y, w) in &pairs {
            tn += 1;
            tsw += w;
            tswy += w * y;
            tswy2 += w * y * y;
        }

        fn sse(sw: f64, swy: f64, swy2: f64) -> f64 {
            if sw <= 0.0 {
                0.0
            } else {
                swy2 - swy * swy / sw
            }
        }
        let parent_sse = sse(tsw, tswy, tswy2) + sse(msw, mswy, mswy2);
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let mut ln = 0usize;
        let (mut lsw, mut lswy, mut lswy2) = (0.0, 0.0, 0.0);
        let mut best: Option<(f64, Split)> = None;
        for k in 0..pairs.len() {
            let (v, y, w) = pairs[k];
            ln += 1;
            lsw += w;
            lswy += w * y;
            lswy2 += w * y * y;
            let is_boundary = k + 1 < pairs.len() && pairs[k + 1].0 != v;
            if !is_boundary {
                continue;
            }
            // Grid mode: thresholds are grid values (compare rounds to the
            // same grid, see `goes_left`); exact mode keeps midpoints.
            let threshold = match self.params.split_precision {
                Some(_) => v,
                None => (v + pairs[k + 1].0) / 2.0,
            };
            let rn = tn - ln;
            let (rsw, rswy, rswy2) = (tsw - lsw, tswy - lswy, tswy2 - lswy2);
            st.candidates += 1;
            for missing_left in [true, false] {
                let (aln, alsw, alswy, alswy2) = if missing_left {
                    (ln + mn, lsw + msw, lswy + mswy, lswy2 + mswy2)
                } else {
                    (ln, lsw, lswy, lswy2)
                };
                let (arn, arsw, arswy, arswy2) = if missing_left {
                    (rn, rsw, rswy, rswy2)
                } else {
                    (rn + mn, rsw + msw, rswy + mswy, rswy2 + mswy2)
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0 && (alsw < self.params.alpha * (tsw + msw)
                    || arsw < self.params.alpha * (tsw + msw))
                {
                    continue;
                }
                if !self.mono_ok(j, alswy / alsw.max(1e-300), arswy / arsw.max(1e-300)) {
                    continue;
                }
                let gain = match self.params.gain {
                    GainFn::Sse => {
                        parent_sse - (sse(alsw, alswy, alswy2) + sse(arsw, arswy, arswy2))
                    }
                    ref gfn => {
                        // Lazy (y, w) slices over the value-sorted buffer;
                        // missing rows appended to the side they route to
                        // (allocated only when the column has missing).
                        if missing_yws.is_empty() {
                            self.gain_of(gfn, &yws_sorted[..k + 1], &yws_sorted[k + 1..])
                        } else {
                            let mut ls: Vec<(f64, f64)> = yws_sorted[..k + 1].to_vec();
                            let mut rs: Vec<(f64, f64)> = yws_sorted[k + 1..].to_vec();
                            if missing_left {
                                ls.extend_from_slice(&missing_yws);
                            } else {
                                rs.extend_from_slice(&missing_yws);
                            }
                            self.gain_of(gfn, &ls, &rs)
                        }
                    }
                };
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty * (1.0 / aln as f64 + 1.0 / arn as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    best = Some((
                        gain,
                        Split {
                            col: j,
                            threshold,
                            missing_left,
                            gain,
                            cat: None,
                        },
                    ));
                }
            }
        }
        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    /// Linear-leaf scan — LightGBM's histogram shape: one weighted Gram per
    /// distinct value, running left/right aggregates, per-candidate solve on
    /// preallocated scratch (no allocation in the loop).
    fn scan_column_linear(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
    ) -> (Option<(f64, Split)>, ScanStats) {
        let mut st = ScanStats::default();
        let m = self.params.min_rows;
        let t0 = Instant::now();

        let chosen = self.chosen_cols(path);
        let dim = 1 + 2 * chosen.len();
        let mut row_buf = vec![0.0; dim];

        let mut pairs: Vec<(f64, usize)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, r)))
            .collect();
        if pairs.len() < 2 {
            return (None, st);
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));
        // split_precision grid (see scan_column_mean).
        if let Some(p) = self.params.split_precision {
            if p > 0.0 {
                let q = 1.0 / p;
                for t in pairs.iter_mut() {
                    t.0 = (t.0 * q).round() / q;
                }
            }
        }

        let mut groups: Vec<(f64, Gram)> = Vec::new();
        for &(v, r) in &pairs {
            design_row(&mut row_buf, self.cols, r, &chosen, true);
            match groups.last_mut() {
                Some((gv, g)) if *gv == v => g.add_row(&row_buf, self.y[r], self.w[r]),
                _ => {
                    let mut g = Gram::new(dim);
                    g.add_row(&row_buf, self.y[r], self.w[r]);
                    groups.push((v, g));
                }
            }
        }
        let mut miss = Gram::new(dim);
        for &r in rows {
            if self.cols[j][r].is_none() {
                design_row(&mut row_buf, self.cols, r, &chosen, true);
                miss.add_row(&row_buf, self.y[r], self.w[r]);
            }
        }

        // Parent SSE over the parent's design class (once per node).
        let parent_chosen = self.chosen_cols(path);
        let pdim = 1 + 2 * parent_chosen.len();
        let mut pgram = Gram::new(pdim);
        let mut prow = vec![0.0; pdim];
        for &r in rows {
            design_row(&mut prow, self.cols, r, &parent_chosen, true);
            pgram.add_row(&prow, self.y[r], self.w[r]);
        }
        let mut scratch = SolveScratch::new(dim.max(pdim));
        let parent_sse = solve_normal_sum(
            &pgram,
            None,
            pdim,
            self.params.l1,
            self.params.l2,
            &mut scratch,
        );

        // Monotone columns with linear leaves need one more gate per
        // candidate: the children's fitted outputs must be ordered AT the
        // threshold (their intercepts meet there - child means alone don't
        // order linear responses). The eval point holds other columns at the
        // node's mean design row.
        let mut mean_row = vec![0.0; dim];
        for &r in rows {
            design_row(&mut row_buf, self.cols, r, &chosen, true);
            for (m, v) in mean_row.iter_mut().zip(&row_buf) {
                *m += *v;
            }
        }
        let inv_n = 1.0 / rows.len().max(1) as f64;
        for v in mean_row.iter_mut() {
            *v *= inv_n;
        }
        let jslot = 1 + chosen.iter().position(|&c| c == j).unwrap();
        let mut gbuf0 = Gram::new(dim);
        let mut gbuf1 = Gram::new(dim);
        st.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let mut total = Gram::new(dim);
        for (_, g) in &groups {
            total.add(g);
        }
        let mut left = Gram::new(dim);
        let total_ws = total.ws;
        let mut right = total;
        let mut best: Option<(f64, Split)> = None;

        for gi in 0..groups.len() {
            let (v, g) = &groups[gi];
            left.add(g);
            right.sub(g);
            if gi + 1 >= groups.len() || left.n < m || right.n < m {
                continue;
            }
            let threshold = match self.params.split_precision {
                Some(_) => *v,
                None => (v + groups[gi + 1].0) / 2.0,
            };
            st.candidates += 1;
            for missing_left in [true, false] {
                let left_n = left.n + if missing_left { miss.n } else { 0 };
                let right_n = right.n + if missing_left { 0 } else { miss.n };
                if left_n < m || right_n < m {
                    continue;
                }
                let (wl, wr) = if missing_left {
                    (left.ws + miss.ws, right.ws)
                } else {
                    (left.ws, right.ws + miss.ws)
                };
                // grf-style alpha: each child carries at least alpha of the
                // node's total weight.
                if self.params.alpha > 0.0
                    && (wl < self.params.alpha * total_ws || wr < self.params.alpha * total_ws)
                {
                    continue;
                }
                let (ml, mr) = if missing_left {
                    ((left.wy + miss.wy) / wl.max(1e-300), right.wy / wr.max(1e-300))
                } else {
                    (left.wy / wl.max(1e-300), (right.wy + miss.wy) / wr.max(1e-300))
                };
                if !self.mono_ok(j, ml, mr) {
                    continue;
                }
                if self.params.monotone.get(j).copied().unwrap_or(0) != 0 {
                    gbuf0.clone_from(&left);
                    gbuf1.clone_from(&right);
                    if missing_left {
                        gbuf0.add(&miss);
                    } else {
                        gbuf1.add(&miss);
                    }
                    let ml_mod = fit_linear(
                        &gbuf0,
                        &chosen,
                        true,
                        &self.params.monotone,
                        self.params.l1,
                        self.params.l2,
                    );
                    let mr_mod = fit_linear(
                        &gbuf1,
                        &chosen,
                        true,
                        &self.params.monotone,
                        self.params.l1,
                        self.params.l2,
                    );
                    mean_row[jslot] = threshold;
                    let fl = ml_mod.eval_design(&mean_row);
                    let fr = mr_mod.eval_design(&mean_row);
                    if (fl - fr) * self.params.monotone[j] as f64 > 1e-9 {
                        continue;
                    }
                }
                let (sl, sr) = if missing_left {
                    (
                        solve_normal_sum(&left, Some(&miss), dim, self.params.l1, self.params.l2, &mut scratch),
                        solve_normal_sum(&right, None, dim, self.params.l1, self.params.l2, &mut scratch),
                    )
                } else {
                    (
                        solve_normal_sum(&left, None, dim, self.params.l1, self.params.l2, &mut scratch),
                        solve_normal_sum(&right, Some(&miss), dim, self.params.l1, self.params.l2, &mut scratch),
                    )
                };
                let gain = parent_sse - (sl + sr);
                let gain = if self.params.imbalance_penalty > 0.0 {
                    gain - self.params.imbalance_penalty
                        * (1.0 / left_n as f64 + 1.0 / right_n as f64)
                } else {
                    gain
                };
                if gain <= self.params.min_gain {
                    continue;
                }
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    best = Some((
                        gain,
                        Split {
                            col: j,
                            threshold,
                            missing_left,
                            gain,
                            cat: None,
                        },
                    ));
                }
            }
        }
        st.scan_ns += t1.elapsed().as_nanos() as u64;
        (best, st)
    }

    /// Fit the leaf model for a node's rows.
    fn make_leaf(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        let yv: Vec<f64> = rows.iter().map(|&r| self.y[r]).collect();
        let wv: Vec<f64> = rows.iter().map(|&r| self.w[r]).collect();
        let mut stats = LeafStats::from_targets(&yv, &wv);
        if !self.params.leaf_digest {
            // Lean leaves: drop the t-digest (`quantile` then returns NaN).
            stats.digest = tdigest::TDigest::new_with_size(1);
        }
        let linear = if self.params.leaf == LeafKind::Linear {
            let chosen = self.chosen_cols(path);
            let any_missing = rows
                .iter()
                .any(|&r| chosen.iter().any(|&c| self.cols[c][r].is_none()));
            let dim = 1 + chosen.len() + if any_missing { chosen.len() } else { 0 };
            let mut g = Gram::new(dim);
            let mut row = vec![0.0; dim];
            for &r in rows {
                design_row(&mut row, self.cols, r, &chosen, any_missing);
                g.add_row(&row, self.y[r], self.w[r]);
            }
            Some(fit_linear(
                &g,
                &chosen,
                any_missing,
                &self.params.monotone,
                self.params.l1,
                self.params.l2,
            ))
        } else {
            None
        };
        LeafModel { stats, linear }
    }
}

/// Decimals needed to show `x` with up to 7 significant digits, trailing
/// zeros trimmed (R trims the column's common format to what values need).
fn decs_for(x: f64) -> usize {
    if x == 0.0 {
        return 0;
    }
    let d = (7.0 - 1.0 - x.abs().log10().floor()).clamp(0.0, 20.0) as usize;
    let s = format!("{:.*}", d, x);
    let trimmed = s.trim_end_matches('0');
    let nd = trimmed.split_once('.').map(|(_, frac)| frac.len()).unwrap_or(0);
    nd.min(d)
}

/// R's `format(signif(x, 7))` over a whole COLUMN: one common number of
/// decimals (fixed), or all-scientific `d.ddddddde+NN` when any nonzero
/// magnitude is < 1e-3 or any is >= 1e7. Pinned from rpart's print output.
fn fmt_column(vals: &[f64]) -> Vec<String> {
    let sci = vals
        .iter()
        .any(|v| (*v != 0.0 && v.abs() < 1e-3) || v.abs() >= 1e7);
    if sci {
        vals.iter()
            .map(|v| {
                let s = format!("{:.6e}", v);
                let (m, e) = s.split_once('e').unwrap();
                format!("{}e{:+03}", m, e.parse::<i32>().unwrap())
            })
            .collect()
    } else {
        let d = vals.iter().map(|v| decs_for(*v)).max().unwrap_or(0);
        vals.iter().map(|v| format!("{:.*}", d, v)).collect()
    }
}

/// rpart's `formatg`: 7 significant digits, trailing zeros trimmed —
/// the split cutpoint label.
fn fmt_cut(x: f64) -> String {
    let s = format!("{:.*}", decs_for(x), x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// rpart-style text print (grounded byte-for-byte on the installed
/// `rpart`'s output; see `references/repos/rpart/R/print.rpart.R`):
///
/// ```text
/// n= 20
///
/// node), split, n, deviance, yval
///       * denotes terminal node
///
///  1) root 20 6.900288e+01 2.5190520
///    2) x< 0.5 10 2.254216e+00 0.7237466
/// ```
///
/// Rows are preorder with binary-heap numbering (`format(node)` is
/// right-aligned to the widest number); indent is two spaces per depth;
/// terminal nodes are marked `*`.
impl std::fmt::Display for ModelTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Row {
            depth: usize,
            node: usize,
            label: String,
            n: usize,
            dev: f64,
            yval: f64,
            leaf: bool,
        }
        fn walk(
            t: &ModelTree,
            i: usize,
            num: usize,
            depth: usize,
            label: &str,
            out: &mut Vec<Row>,
        ) {
            let nd = &t.nodes[i];
            out.push(Row {
                depth,
                node: num,
                label: label.to_string(),
                n: nd.stats.n,
                dev: nd.stats.sse,
                yval: nd.stats.mean,
                leaf: nd.split.is_none(),
            });
            if let (Some(sp), Some(l), Some(r)) = (nd.split, nd.left, nd.right) {
                let name = &t.col_names[sp.col];
                let (ll, rl) = match sp.cat {
                    Some(ci) => {
                        let set = t.cat_parts[ci]
                            .iter()
                            .map(|v| fmt_cut(*v))
                            .collect::<Vec<_>>()
                            .join(",");
                        (format!("{name} in {{{set}}}"), format!("{name} not-in {{{set}}}"))
                    }
                    None => {
                        let cut = fmt_cut(sp.threshold);
                        (format!("{name}< {cut}"), format!("{name}>={cut}"))
                    }
                };
                walk(t, l, num * 2, depth + 1, &ll, out);
                walk(t, r, num * 2 + 1, depth + 1, &rl, out);
            }
        }
        let mut rows = Vec::new();
        walk(self, self.root, 1, 0, "root", &mut rows);
        let width = rows.iter().map(|r| r.node).max().unwrap_or(1).to_string().len();
        let devs = fmt_column(&rows.iter().map(|r| r.dev).collect::<Vec<_>>());
        let yvals = fmt_column(&rows.iter().map(|r| r.yval).collect::<Vec<_>>());

        writeln!(f, "n= {} ", rows[0].n)?;
        writeln!(f)?;
        writeln!(f, "node), split, n, deviance, yval")?;
        writeln!(f, "      * denotes terminal node")?;
        writeln!(f)?;
        for (k, r) in rows.iter().enumerate() {
            let term = if r.leaf { "*" } else { " " };
            writeln!(
                f,
                "{}{:>w$}) {} {} {} {} {}",
                "  ".repeat(r.depth),
                r.node,
                r.label,
                r.n,
                devs[k],
                yvals[k],
                term,
                w = width
            )?;
        }
        Ok(())
    }
}
