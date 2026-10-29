//! Model tree skeleton: the tree's vocabulary (params, splits, node
//! profiles) and the growth walk. The machinery is modules by its function
//! in the tree: [`crate::criteria`] (split quality), [`crate::split`] (the
//! candidate scan), [`crate::leaf`] (leaf models), [`crate::prune`] (cost
//! pruning), [`crate::gate`] (significance stopping), [`crate::text_print`]
//! (rendering). Node models and numerics live in [`crate::node_stats`];
//! column screening lives in [`crate::preselect`]. Every function does one
//! job.
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

use crate::criteria::GainFn;
use crate::gate::{GateNote, SplitTest};
use crate::leaf::{LeafCols, LeafKind};
use crate::node_stats::{FrameStats, LeafModel};
use crate::preselect;
use crate::prune::PruneCost;
use crate::split::{CatOrder, CatSearch, ColType};

pub use crate::node_stats::{LeafModel as NodeModel, LeafStats as NodeStats, LinearModel};

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
    /// Split significance gate (statistical stopping) — see [`SplitTest`].
    pub split_test: SplitTest,
    /// Significance level for `split_test` (right-tail p threshold).
    pub split_alpha: f64,
    /// Bonferroni correction: divide `split_alpha` by the number of
    /// candidate splits evaluated at the node (selection multiplicity —
    /// the best-of-many is biased without it).
    pub bonferroni: bool,
    /// Columns eligible for the split search (original indices). `None` =
    /// every pre-selected column. Restricts *splitting only* — leaves still
    /// see all active columns unless `leaf_cols` restricts them.
    pub split_cols: Option<Vec<usize>>,
    /// Honesty (grf's sample-splitting fairness): a seeded partition of the
    /// rows — structure rows drive the split search, gate and pruning;
    /// estimation rows fit the leaf models. Leaf estimates are then honest
    /// (independent of the structure that selected the split). Off by
    /// default. A node whose estimation rows fall under `min_rows` falls
    /// back to its structure rows (training log notes it).
    pub honesty: bool,
    /// Share of rows assigned to the *structure* sample under `honesty`
    /// (grf's `honesty.fraction`).
    pub honesty_fraction: f64,
    /// NA surrogate splits (rpart's `maxsurrogate`): store up to this many
    /// backup splits per node that mimic the primary partition, to route
    /// prediction rows missing the split column. 0 = off (the per-split
    /// `missing_left` alone).
    pub maxsurrogate: usize,
    /// Cost-complexity penalty per leaf (`prune="cp"`; rpart `cp` / sklearn
    /// `ccp_alpha` semantics): a subtree collapses into its parent leaf
    /// when the children's deviance reduction does not earn `cp_alpha`.
    pub cp_alpha: f64,
    /// Leaf cap (sklearn `max_leaf_nodes` / LightGBM `num_leaves`): stop
    /// splitting once this many leaves exist (depth-first budget, exact
    /// bound). `None` = unlimited (bounded by `max_depth`).
    pub max_leaf_nodes: Option<usize>,
    /// ctree trio — minimum rows for a node to attempt a split
    /// (`minsplit`); `None` falls back to `min_rows`.
    pub minsplit: Option<usize>,
    /// ctree trio — minimum rows per leaf (`minbucket`); `None` falls back
    /// to `min_rows`.
    pub minbucket: Option<usize>,
    /// ctree trio — minimum weight fraction of the node per leaf
    /// (`minprob`). 0.0 = off.
    pub minprob: f64,
    /// Exclusive feature bundling (LightGBM EFB): before the scan, bundle
    /// mutually-exclusive zero-heavy numeric columns into one (disjoint
    /// value ranges) to cut the scan's column count. `false` = off.
    /// Preselection strategy family (`winnow` / `efb` / `winnowefb` /
    /// `boruta`) — see [`crate::preselect::Preselect`]. The winnow
    /// threshold is `min_relevance`.
    pub preselect: crate::preselect::Preselect,
    /// Purity gate: a node whose weighted target standard deviation is at
    /// most this stays a leaf (its signal is exhausted). `None` = off.
    pub purity_gate: Option<f64>,
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
            split_test: SplitTest::None,
            split_alpha: 0.05,
            bonferroni: false,
            split_cols: None,
            honesty: false,
            honesty_fraction: 0.5,
            maxsurrogate: 0,
            cp_alpha: 0.0,
            max_leaf_nodes: None,
            minsplit: None,
            minbucket: None,
            minprob: 0.0,
            preselect: crate::preselect::Preselect::None,
            purity_gate: None,
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
    /// Surrogate-split handle: `Some(i)` = `Tree::sur_parts[i]` lists the
    /// backup splits that route rows missing the split column.
    pub sur: Option<usize>,
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
pub(crate) struct Node {
    pub(crate) split: Option<Split>,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
    pub(crate) leaf: Option<LeafModel>,
    pub(crate) n: usize,
    pub(crate) stats: FrameStats,
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
    /// Chow gate result on the node's chosen split (`split_test=Chow`).
    pub gate: Option<GateNote>,
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
    pub(crate) nodes: Vec<Node>,
    pub(crate) root: usize,
    pub(crate) profiles: Vec<NodeProfile>,
    /// Categorical partitions referenced by `Split::cat`.
    pub(crate) cat_parts: Vec<Vec<f64>>,
    /// Surrogate splits referenced by `Split::sur` (aligned by handle).
    pub(crate) sur_parts: Vec<Vec<crate::surrogate::Surrogate>>,
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
        match crate::split::grid_of(&self.params.dtypes, self.params.split_precision, sp.col) {
            Some(p) if p > 0.0 => {
                let q = 1.0 / p;
                (v * q).round() / q <= sp.threshold
            }
            _ => v <= sp.threshold,
        }
    }
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
        if let Some(sc) = &params.split_cols {
            if sc.iter().any(|&j| j >= d) {
                return Err("split_cols entries must be valid column indices".to_string());
            }
        }
        if params.cp_alpha < 0.0 {
            return Err("cp_alpha must be >= 0".to_string());
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

        // Pre-selection family: which columns enter the split search
        // (`min_relevance` implies winnow when the family is unset).
        let active_cols: Vec<usize> = match params.preselect {
            crate::preselect::Preselect::None if params.min_relevance.is_none() => {
                (0..d).collect()
            }
            crate::preselect::Preselect::None | crate::preselect::Preselect::Winnow => {
                let thr = params.min_relevance.unwrap_or(0.0);
                let kept = preselect::select_columns(&cols, &yv, &wv, thr);
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            crate::preselect::Preselect::Boruta => {
                let kept = preselect::boruta(
                    &cols,
                    &yv,
                    &wv,
                    params.seed,
                    params.min_relevance.unwrap_or(0.0),
                );
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            crate::preselect::Preselect::ShadowMax => {
                let kept = preselect::shadowmax(
                    &cols,
                    &yv,
                    &wv,
                    params.seed,
                    5,
                    params.min_relevance.unwrap_or(0.0),
                );
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            crate::preselect::Preselect::Variance => {
                let kept = preselect::variance(&cols, &yv, &wv, params.min_relevance.unwrap_or(0.0));
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            crate::preselect::Preselect::MutualInfo => {
                let kept = preselect::mutual_info(&cols, &yv, &wv, params.min_relevance.unwrap_or(0.0));
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
            crate::preselect::Preselect::UniTree => {
                let (kept, _r2) = preselect::uni_tree_r2(
                    &cols,
                    &yv,
                    &wv,
                    params.min_relevance.unwrap_or(0.0),
                    params.seed,
                );
                if kept.is_empty() {
                    return Err("pre-selection removed every column".to_string());
                }
                kept
            }
        };
        // Shrinking a monotone vector would break indices; keep it aligned.

        // Honesty (grf's sample-splitting fairness): a seeded partition of
        // the rows — structure rows drive the split search, gate and
        // pruning; estimation rows fit the leaf models. Default off (the
        // estimation set then IS the structure set).
        let all_rows: Vec<usize> = (0..yv.len()).collect();
        let (rows, est_rows): (Vec<usize>, Vec<usize>) = if params.honesty {
            let n = all_rows.len();
            let k = (((n as f64) * params.honesty_fraction).round() as usize)
                .clamp(1, n.max(2) - 1);
            let mut idx = all_rows;
            let mut rng = params.seed ^ 0xD1B5_4A32_D192_ED03;
            for i in (1..n).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let j = (rng as usize) % (i + 1);
                idx.swap(i, j);
            }
            let (mut s, mut e) = (idx[..k].to_vec(), idx[k..].to_vec());
            s.sort_unstable();
            e.sort_unstable();
            (s, e)
        } else {
            (all_rows.clone(), all_rows)
        };
        let mut grower = Grower {
            cols: &cols,
            y: &yv,
            w: &wv,
            params: &params,
            active: &active_cols,
            nodes: Vec::new(),
            profiles: Vec::new(),
            cat_parts: Vec::new(),
            sur_parts: Vec::new(),
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
        let cap = params.max_leaf_nodes.unwrap_or(usize::MAX);
        let (root, _) = grower.grow(&rows, &est_rows, 0, &mut Vec::new(), cap);
        // Destructure to release the borrows on params/active_cols before
        // moving them into the tree.
        let Grower {
            nodes,
            profiles,
            cat_parts,
            sur_parts,
            mut trace,
            ..
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
            sur_parts,
            trace,
        })
    }

    /// Route a row missing the split column: the first observed surrogate
    /// decides (mirrored when `flip`), else the split's `missing_left`.
    /// Surrogate comparisons share the grid convention with `goes_left`.
    fn surrogate_left(&self, row: &[Option<f64>], sp: &Split) -> bool {
        if let Some(si) = sp.sur {
            for s in &self.sur_parts[si] {
                if let Some(v) = row.get(s.col).and_then(|v| *v) {
                    let left = match crate::split::grid_of(
                        &self.params.dtypes,
                        self.params.split_precision,
                        s.col,
                    ) {
                        Some(p) if p > 0.0 => {
                            let q = 1.0 / p;
                            (v * q).round() / q <= s.threshold
                        }
                        _ => v <= s.threshold,
                    };
                    return if s.flip { !left } else { left };
                }
            }
        }
        sp.missing_left
    }

    /// Root-reachable membership per node index. After cost pruning the
    /// arena keeps the orphaned subtree nodes; introspection walks the
    /// live tree only (arena order preserved).
    fn reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            if seen[i] {
                continue;
            }
            seen[i] = true;
            if let (Some(l), Some(r)) = (self.nodes[i].left, self.nodes[i].right) {
                stack.push(l);
                stack.push(r);
            }
        }
        seen
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
                None => self.surrogate_left(row, sp),
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
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict(row)
            })
            .collect()
    }

    /// All splits in the tree (introspection / structure recovery).
    pub fn splits(&self) -> Vec<Split> {
        let seen = self.reachable();
        (0..self.nodes.len())
            .filter(|&i| seen[i])
            .filter_map(|i| self.nodes[i].split)
            .collect()
    }

    /// Feature importance — each column's share of the total split gain
    /// (rpart's "variable importance": summed goodness of split). Returns
    /// (column name, share) pairs sorted descending; shares sum to 1.
    pub fn feature_importances(&self) -> Vec<(String, f64)> {
        let mut totals = vec![0.0f64; self.col_names.len()];
        let mut sum = 0.0;
        let seen = self.reachable();
        for (i, nd) in self.nodes.iter().enumerate() {
            if !seen[i] {
                continue;
            }
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
                Some(v) => self.goes_left(sp, v),
                None => self.surrogate_left(x, sp),
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
        let seen = self.reachable();
        (0..self.nodes.len())
            .filter(|&i| seen[i])
            .filter_map(|i| self.nodes[i].leaf.as_ref().map(|m| (m, self.nodes[i].n)))
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

pub(crate) struct Grower<'a> {
    pub(crate) cols: &'a [Vec<Option<f64>>],
    pub(crate) y: &'a [f64],
    pub(crate) w: &'a [f64],
    pub(crate) params: &'a ModelTreeParams,
    pub(crate) active: &'a [usize],
    pub(crate) nodes: Vec<Node>,
    pub(crate) profiles: Vec<NodeProfile>,
    /// Categorical partitions referenced by `Split::cat` (moved into the
    /// fitted tree).
    pub(crate) cat_parts: Vec<Vec<f64>>,
    /// Surrogate splits referenced by `Split::sur` (moved into the tree).
    pub(crate) sur_parts: Vec<Vec<crate::surrogate::Surrogate>>,
    /// Training log lines (`params.verbose`).
    pub(crate) trace: Vec<String>,
    /// Prune-cost bookkeeping aligned with node indices (`prune != None`).
    pub(crate) costs: Vec<f64>,
    /// Per-node counter for the random column subset (seeded, deterministic).
    pub(crate) rng_counter: u64,
}

impl<'a> Grower<'a> {
    /// Grid-aware left test (see `ModelTree::goes_left`).
    pub(crate) fn goes_left(&self, sp: &Split, v: f64) -> bool {
        if let Some(i) = sp.cat {
            return self.cat_parts[i].contains(&v);
        }
        match crate::split::grid_of(&self.params.dtypes, self.params.split_precision, sp.col) {
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

    fn grow(
        &mut self,
        rows: &[usize],
        est: &[usize],
        depth: usize,
        path: &mut Vec<usize>,
        budget: usize,
    ) -> (usize, usize) {
        let mut prof = NodeProfile {
            depth,
            rows: rows.len(),
            ..Default::default()
        };
        let mut best = if depth < self.params.max_depth {
            self.best_split(rows, path, &mut prof)
        } else {
            None
        };
        // Statistical stopping gate (default off): the scan's chosen split
        // must significantly beat the pooled node model — else the node
        // stays a leaf. Bonferroni divides alpha by the number of
        // candidate splits evaluated here (best-of-many multiplicity).
        if self.params.split_test == SplitTest::Chow {
            let gate = best
                .as_ref()
                .and_then(|(sp, lrows, rrows)| self.chow_test(rows, lrows, rrows, path, sp.col));
            if let Some((f, p)) = gate {
                let alpha = if self.params.bonferroni {
                    self.params.split_alpha / prof.candidates.max(1) as f64
                } else {
                    self.params.split_alpha
                };
                let passed = p < alpha;
                prof.gate = Some(GateNote {
                    f,
                    p,
                    alpha,
                    passed,
                });
                if !passed {
                    best = None;
                }
            }
        }
        // Purity gate (default off): a node whose weighted target std is at
        // most `purity_gate` has no signal left — it stays a leaf.
        if let Some(t) = self.params.purity_gate {
            let yws: Vec<(f64, f64)> =
                rows.iter().map(|&r| (self.y[r], self.w[r])).collect();
            if crate::stats::w_sd(&yws) <= t {
                best = None;
            }
        }
        // Leaf-cap budget (default off, budget = usize::MAX): a split
        // needs room for at least two leaves; the left subtree reports its
        // spend and the right gets the remainder -- exact bound.
        if best.is_some() && budget < 2 {
            best = None;
        }
        if self.params.verbose >= 2 {
            let mut line = format!(
                "node depth={} rows={} cols={} candidates={} scan_ms={} leaf_ms={}",
                prof.depth,
                prof.rows,
                prof.cols,
                prof.candidates,
                prof.scan_ns / 1_000_000,
                prof.leaf_ns / 1_000_000
            );
            if let Some(g) = &prof.gate {
                line.push_str(&format!(
                    " chow_f={:.4} chow_p={:.4} chow_alpha={:.4} chow={}",
                    g.f,
                    g.p,
                    g.alpha,
                    if g.passed { "pass" } else { "stop" }
                ));
            }
            self.trace.push(line);
        }

        match best {
            None => {
                let t = Instant::now();
                let leaf = self.make_leaf(self.leaf_rows(rows, est), path);
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
                (self.nodes.len() - 1, 1)
            }
            Some((mut sp, lrows, rrows)) => {
                // NA surrogate splits: backup routes for rows missing the
                // split column (rpart lineage; off unless maxsurrogate > 0).
                if self.params.maxsurrogate > 0 {
                    let found = self.find_surrogates(rows, &sp, &lrows);
                    if !found.is_empty() {
                        sp.sur = Some(self.sur_parts.len());
                        self.sur_parts.push(found);
                    }
                }
                let (est_l, est_r) = self.partition_rows(est, &sp);
                path.push(sp.col);
                let (l, lu) =
                    self.grow(&lrows, &est_l, depth + 1, path, budget.saturating_sub(1));
                let (r, ru) =
                    self.grow(&rrows, &est_r, depth + 1, path, budget.saturating_sub(lu));
                path.pop();
                let used = lu + ru;
                if self.params.prune != PruneCost::None {
                    // Bottom-up cost pruning: keep the subtree only when it
                    // codes cheaper than this node as one leaf.
                    let leaf_c = self.leaf_cost(rows);
                    let split_bit = self.params.prune.split_bit(rows.len() as f64);
                    let sub_c = self.costs[l] + self.costs[r] + split_bit;
                    if leaf_c <= sub_c {
                        let t = Instant::now();
                        let leaf = self.make_leaf(self.leaf_rows(rows, est), path);
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
                        return (self.nodes.len() - 1, 1);
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
                    return (self.nodes.len() - 1, used);
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
                (self.nodes.len() - 1, used)
            }
        }
    }

}

