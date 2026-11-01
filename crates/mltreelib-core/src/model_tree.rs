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
//!   (`src/treelearner/feature_histogram.cpp`) over per-value Gram groups
//!   — with the per-candidate solves of that reference
//!   (`src/treelearner/linear_tree_learner.cpp`) replaced by maintained
//!   Cholesky factors: rows crossing the cut fold into the factors one
//!   rank-1 step at a time and candidates are scored by triangular solves
//!   (issue #7). Groups wider than ~dim/3 rows drop the factors instead —
//!   a re-factor is cheaper than the fold — and any solve without a live
//!   factor is exactly the historical solve.
//! Everything is row-weighted; missing values are scored in both default
//! directions (LightGBM's `default_left` shape) and the gain picks.
//!
//! The column scan is rayon-parallel above a work threshold (Level-1
//! parallelism) with a **deterministic** tie-break: first-found wins
//! (lowest column index, then lowest threshold), independent of thread
//! completion order.

use std::time::Instant;

use crate::criteria::GainFn;
use crate::family::LeafFamily;
use crate::gate::{GateNote, SplitTest};
use crate::leaf::{fit_leaf_model, LeafCols, LeafKind};
pub use crate::leaf::LeafControl;
use crate::node_stats::{FrameStats, LeafModel};
use crate::preselect;
use crate::prune::PruneCost;
use crate::split::{CatOrder, CatSearch, ColType};

pub use crate::node_stats::{LeafModel as NodeModel, LeafStats as NodeStats, LinearModel};
pub use crate::split::SplitEval;

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
    /// Distribution fitted by each leaf; Gaussian preserves least squares.
    /// Binomial (logit) and Poisson (log) support mean and linear shapes.
    pub leaf_family: LeafFamily,
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
    /// Split evaluation strategy (fast scalar variance reduction vs exhaustive linear scan).
    pub split_eval: SplitEval,
    /// Upfront pre-sorting of numeric columns. Default `false` (in-place cache-local PDQsort in SplitScratch).
    pub presort: bool,
}

impl Default for ModelTreeParams {
    fn default() -> Self {
        Self {
            // Defaults rebaked 2026-10-30 (candidate C, the robust
            // winner of the 18-dataset sweep: C 10 - coarse-NEW 4 - old 3).
            // Scale-aware leaves (min_rows floor + minprob share), fine
            // grid, cost-complexity pruning -- see examples/default_bake.py
            // for the evidence tables.
            max_depth: 5,
            min_rows: 50,
            // Noise floor as much as a tuning knob: fitted-leaf gains below
            // this are numerical, not structural.
            min_gain: 1e-9,
            leaf_cols: LeafCols::All,
            leaf: LeafKind::Linear,
            leaf_family: LeafFamily::Gaussian,
            l2: 0.5,
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
            prune: PruneCost::Cp,
            split_test: SplitTest::None,
            split_alpha: 0.05,
            bonferroni: false,
            split_cols: None,
            honesty: false,
            honesty_fraction: 0.5,
            maxsurrogate: 0,
            cp_alpha: 5.0,
            max_leaf_nodes: None,
            minsplit: None,
            minbucket: None,
            minprob: 0.02,
            preselect: crate::preselect::Preselect::None,
            purity_gate: None,
            split_eval: SplitEval::Fast,
            presort: false,
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
    pub(crate) leaf_id: Option<usize>,
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
        if params.leaf_family != LeafFamily::Gaussian {
            if !matches!(params.leaf, LeafKind::Mean | LeafKind::Linear) {
                return Err("binomial and poisson families support only mean or linear leaves".into());
            }
            if params.split_test != SplitTest::None {
                return Err("split_test='chow' requires the gaussian family".into());
            }
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
            if params.leaf_family != LeafFamily::Gaussian && t.is_nan() {
                continue;
            }
            if w[r] == 0.0 {
                continue;
            }
            if params.leaf_family != LeafFamily::Gaussian {
                if !t.is_finite() || !w[r].is_finite() {
                    return Err("distributional targets and weights must be finite".into());
                }
                match params.leaf_family {
                    LeafFamily::Binomial if t != 0.0 && t != 1.0 => {
                        return Err("binomial targets must be 0 or 1".into());
                    }
                    LeafFamily::Poisson if t < 0.0 => {
                        return Err("poisson targets must be non-negative".into());
                    }
                    _ => {}
                }
            }
            for j in 0..d {
                let value = if params.leaf_family != LeafFamily::Gaussian {
                    match row[j] {
                        Some(v) if v.is_nan() => None,
                        Some(v) if !v.is_finite() => {
                            return Err("distributional predictors must be finite or missing".into());
                        }
                        value => value,
                    }
                } else {
                    row[j]
                };
                cols[j].push(value);
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
        let (root, _) = grower.grow_iterative(&rows, &est_rows, cap);
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
        if params.leaf_family != LeafFamily::Gaussian {
            let mut stack = vec![root];
            while let Some(i) = stack.pop() {
                let node = &nodes[i];
                if let (Some(l), Some(r)) = (node.left, node.right) {
                    stack.push(l);
                    stack.push(r);
                } else if let Some(leaf) = &node.leaf {
                    let valid = leaf.center.map(|c| c.is_finite()).unwrap_or(true)
                        && leaf.linear.as_ref().map(|m| {
                            m.intercept.is_finite()
                                && m.coef.iter().chain(&m.na_coef).all(|c| c.is_finite())
                        }).unwrap_or(true);
                    if !valid {
                        return Err("distributional leaf fit failed to converge to finite coefficients".into());
                    }
                }
            }
        }
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
        let mut tree = Self {
            col_names,
            params,
            active_cols,
            nodes,
            root,
            profiles,
            cat_parts,
            sur_parts,
            trace,
        };
        tree.assign_leaf_ids();
        Ok(tree)
    }

    /// Retrain all leaf models on new or existing data without altering tree topology or splits.
    ///
    /// Rows are routed through the existing split decision graph down to their assigned leaves.
    /// Each leaf's model is refitted according to the specified `control` parameters.
    pub fn retrain(
        &mut self,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        control: LeafControl,
    ) -> Result<(), String> {
        let n = y.len();
        self.retrain_weighted(x, y, &vec![1.0; n], control)
    }

    /// Retrain all leaf models with sample weights.
    pub fn retrain_weighted(
        &mut self,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        w: &[f64],
        control: LeafControl,
    ) -> Result<(), String> {
        let n_all = y.len();
        if x.len() != n_all || w.len() != n_all {
            return Err("x, y and w length mismatch".to_string());
        }
        let d = self.col_names.len();
        for r in x {
            if r.len() != d {
                return Err("row width != number of column names".to_string());
            }
        }
        if control.l1 < 0.0 || control.l2 < 0.0 {
            return Err("penalties l1 and l2 must be non-negative".to_string());
        }
        if !control.monotone.is_empty() && control.monotone.len() != d {
            return Err("monotone must have one entry per column".to_string());
        }
        if control.family != LeafFamily::Gaussian {
            if !matches!(control.leaf, LeafKind::Mean | LeafKind::Linear) {
                return Err("binomial and poisson families support only mean or linear leaves".into());
            }
        }

        // 1. Filter rows and transpose to column vectors
        let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n_all); d];
        let mut yv: Vec<f64> = Vec::with_capacity(n_all);
        let mut wv: Vec<f64> = Vec::with_capacity(n_all);
        let mut leaf_assignments: Vec<usize> = Vec::with_capacity(n_all);

        for (r, row) in x.iter().enumerate() {
            if w[r] < 0.0 {
                return Err("weights must be non-negative".to_string());
            }
            let Some(t) = y[r] else { continue };
            if control.family != LeafFamily::Gaussian && t.is_nan() {
                continue;
            }
            if w[r] == 0.0 {
                continue;
            }
            if control.family != LeafFamily::Gaussian {
                if !t.is_finite() || !w[r].is_finite() {
                    return Err("distributional targets and weights must be finite".into());
                }
                match control.family {
                    LeafFamily::Binomial if t != 0.0 && t != 1.0 => {
                        return Err("binomial targets must be 0 or 1".into());
                    }
                    LeafFamily::Poisson if t < 0.0 => {
                        return Err("poisson targets must be non-negative".into());
                    }
                    _ => {}
                }
            }
            for j in 0..d {
                let value = if control.family != LeafFamily::Gaussian {
                    match row[j] {
                        Some(v) if v.is_nan() => None,
                        Some(v) if !v.is_finite() => {
                            return Err("distributional predictors must be finite or missing".into());
                        }
                        value => value,
                    }
                } else {
                    row[j]
                };
                cols[j].push(value);
            }
            yv.push(t);
            wv.push(w[r]);

            let leaf_idx = self.leaf_index(row);
            leaf_assignments.push(leaf_idx);
        }

        // Group valid row indices by leaf node index
        let mut leaf_rows_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &leaf_idx) in leaf_assignments.iter().enumerate() {
            leaf_rows_map.entry(leaf_idx).or_default().push(i);
        }

        // 2. Precompute root-to-node split paths
        let mut node_paths = vec![Vec::new(); self.nodes.len()];
        fn walk_paths(
            nodes: &[Node],
            i: usize,
            cur: &mut Vec<usize>,
            node_paths: &mut [Vec<usize>],
        ) {
            node_paths[i] = cur.clone();
            let n = &nodes[i];
            if let (Some(sp), Some(l), Some(r)) = (&n.split, n.left, n.right) {
                cur.push(sp.col);
                walk_paths(nodes, l, cur, node_paths);
                walk_paths(nodes, r, cur, node_paths);
                cur.pop();
            }
        }
        walk_paths(&self.nodes, self.root, &mut Vec::new(), &mut node_paths);

        // 3. For each active leaf in the tree, refit its LeafModel
        let reachable = self.reachable();
        for i in 0..self.nodes.len() {
            if !reachable[i] || self.nodes[i].leaf.is_none() {
                continue;
            }
            if let Some(rows) = leaf_rows_map.get(&i) {
                let path = &node_paths[i];
                let leaf_model = fit_leaf_model(
                    &cols,
                    &yv,
                    &wv,
                    rows,
                    path,
                    &self.active_cols,
                    &control,
                );
                // Validate finiteness for distributional leaf models
                if control.family != LeafFamily::Gaussian {
                    let valid = leaf_model.center.map(|c| c.is_finite()).unwrap_or(true)
                        && leaf_model.linear.as_ref().map(|m| {
                            m.intercept.is_finite()
                                && m.coef.iter().chain(&m.na_coef).all(|c| c.is_finite())
                        }).unwrap_or(true);
                    if !valid {
                        return Err("distributional leaf fit failed to converge to finite coefficients".into());
                    }
                }
                self.nodes[i].stats = FrameStats::from_ty(&yv, &wv, rows);
                self.nodes[i].n = rows.len();
                self.nodes[i].leaf = Some(leaf_model);
            }
        }

        // 4. Update tree parameters to reflect new leaf controls
        self.params.leaf = control.leaf;
        self.params.leaf_family = control.family;
        self.params.leaf_cols = control.leaf_cols;
        self.params.l2 = control.l2;
        self.params.l1 = control.l1;
        self.params.leaf_digest = control.leaf_digest;
        if !control.monotone.is_empty() {
            self.params.monotone = control.monotone;
        }

        Ok(())
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

    /// One P(class 1) per row. Only binomial leaves define this output.
    pub fn predict_proba(&self, x: &[Vec<Option<f64>>]) -> Result<Vec<f64>, String> {
        if self.params.leaf_family != LeafFamily::Binomial {
            return Err("predict_proba requires leaf_family='binomial'".into());
        }
        Ok(self.predict(x))
    }

    /// Hard binary class labels (0 or 1) per row by thresholding binomial probabilities.
    /// Default threshold is 0.5. Requires `leaf_family='binomial'`.
    pub fn predict_label(&self, x: &[Vec<Option<f64>>], threshold: f64) -> Result<Vec<usize>, String> {
        if self.params.leaf_family != LeafFamily::Binomial {
            return Err("predict_label requires leaf_family='binomial'".into());
        }
        if threshold <= 0.0 || threshold >= 1.0 || threshold.is_nan() {
            return Err("threshold must be in (0, 1)".into());
        }
        let probs = self.predict(x);
        Ok(probs.into_iter()
            .map(|p| if p >= threshold { 1 } else { 0 })
            .collect())
    }

    /// Prediction interval `(lower, upper)` per row at the requested coverage level
    /// (e.g. `coverage = 0.90` for a 90% interval, spanning [0.05, 0.95] quantiles).
    /// Uses each leaf's t-digest. Returns `(NaN, NaN)` if leaf digest was dropped.
    pub fn predict_interval(&self, x: &[Vec<Option<f64>>], coverage: f64) -> Result<Vec<(f64, f64)>, String> {
        if coverage <= 0.0 || coverage >= 1.0 || coverage.is_nan() {
            return Err("coverage must be in (0, 1)".into());
        }
        let alpha = (1.0 - coverage) / 2.0;
        let lo_tau = alpha;
        let hi_tau = 1.0 - alpha;
        Ok(x.iter()
            .map(|row| {
                let idx = self.leaf_index(row);
                let leaf = self.nodes[idx]
                    .leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model");
                let lo = leaf.stats.quantile(lo_tau).unwrap_or(f64::NAN);
                let hi = leaf.stats.quantile(hi_tau).unwrap_or(f64::NAN);
                (lo, hi)
            })
            .collect())
    }

    /// Fitted conditional distribution P(Y <= value), not an empirical digest.
    /// Defined for binomial and Poisson leaves; Gaussian keeps empirical quantiles.
    pub fn cdf(&self, x: &[Vec<Option<f64>>], value: f64) -> Result<Vec<f64>, String> {
        if self.params.leaf_family == LeafFamily::Gaussian {
            return Err("cdf requires a binomial or poisson leaf family".into());
        }
        if value.is_nan() {
            return Err("cdf value must not be NaN".into());
        }
        Ok(self.predict(x).into_iter()
            .map(|mean| self.params.leaf_family.cdf(mean, value))
            .collect())
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

    /// Leaf models with their row counts, leaf IDs, and node indices:
    /// `(leaf_model, n_rows, leaf_id, node_idx)`.
    pub fn leaf_entries(&self) -> Vec<(&LeafModel, usize, usize, usize)> {
        let seen = self.reachable();
        let mut entries: Vec<_> = (0..self.nodes.len())
            .filter(|&i| seen[i] && self.nodes[i].leaf.is_some())
            .map(|i| {
                (
                    self.nodes[i].leaf.as_ref().unwrap(),
                    self.nodes[i].n,
                    self.nodes[i].leaf_id.unwrap_or(0),
                    i,
                )
            })
            .collect();
        entries.sort_by_key(|e| e.2);
        entries
    }

    /// Return the leaf_id (0..n_leaves-1) for each input row.
    pub fn apply(&self, x: &[Vec<Option<f64>>]) -> Vec<usize> {
        x.iter()
            .map(|row| {
                let node_idx = self.leaf_index(row);
                self.nodes[node_idx].leaf_id.unwrap_or(0)
            })
            .collect()
    }

    /// Return row indices grouped by leaf_id: leaf_id -> [row_indices].
    pub fn leaf_samples(&self, x: &[Vec<Option<f64>>]) -> std::collections::BTreeMap<usize, Vec<usize>> {
        let mut map = std::collections::BTreeMap::new();
        for (i, row) in x.iter().enumerate() {
            let node_idx = self.leaf_index(row);
            let leaf_id = self.nodes[node_idx].leaf_id.unwrap_or(0);
            map.entry(leaf_id).or_insert_with(Vec::new).push(i);
        }
        map
    }

    /// Leaf IDs of all active leaves in canonical order.
    pub fn leaf_ids(&self) -> Vec<usize> {
        let seen = self.reachable();
        (0..self.nodes.len())
            .filter(|&i| seen[i] && self.nodes[i].leaf.is_some())
            .map(|i| self.nodes[i].leaf_id.unwrap_or(0))
            .collect()
    }

    /// Assign canonical 0-indexed leaf IDs (0..n_leaves-1) in left-to-right DFS order.
    pub(crate) fn assign_leaf_ids(&mut self) {
        let mut leaf_counter = 0;
        let mut stack = vec![self.root];
        let mut dfs_leaves = Vec::new();
        while let Some(i) = stack.pop() {
            let nd = &self.nodes[i];
            match (nd.left, nd.right) {
                (Some(l), Some(r)) => {
                    stack.push(r);
                    stack.push(l);
                }
                _ => {
                    if nd.leaf.is_some() {
                        dfs_leaves.push(i);
                    }
                }
            }
        }
        for node_idx in dfs_leaves {
            self.nodes[node_idx].leaf_id = Some(leaf_counter);
            leaf_counter += 1;
        }
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

enum GrowOp {
    Grow {
        depth: usize,
        rows: Vec<usize>,
        est: Vec<usize>,
        budget: usize,
        res_idx: usize,
    },
    AfterLeft {
        depth: usize,
        rows: Vec<usize>,
        est: Vec<usize>,
        rrows: Vec<usize>,
        est_r: Vec<usize>,
        sp: Split,
        budget: usize,
        prof: NodeProfile,
        left_res_idx: usize,
        right_res_idx: usize,
        res_idx: usize,
    },
    FinishSplit {
        rows: Vec<usize>,
        est: Vec<usize>,
        sp: Split,
        prof: NodeProfile,
        left_res_idx: usize,
        right_res_idx: usize,
        res_idx: usize,
    },
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

    /// Iterative tree growth via an explicit work stack and bottom-up post-order pruning.
    fn grow_iterative(
        &mut self,
        root_rows: &[usize],
        root_est: &[usize],
        budget: usize,
    ) -> (usize, usize) {
        let mut path: Vec<usize> = Vec::new();
        let mut results: Vec<(usize, usize)> = vec![(0, 0)];
        let mut stack: Vec<GrowOp> = vec![GrowOp::Grow {
            depth: 0,
            rows: root_rows.to_vec(),
            est: root_est.to_vec(),
            budget,
            res_idx: 0,
        }];

        while let Some(op) = stack.pop() {
            match op {
                GrowOp::Grow {
                    depth,
                    rows,
                    est,
                    budget,
                    res_idx,
                } => {
                    let mut prof = NodeProfile {
                        depth,
                        rows: rows.len(),
                        ..Default::default()
                    };
                    let mut best = if depth < self.params.max_depth {
                        self.best_split(&rows, &path, &mut prof)
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
                            .and_then(|(sp, lrows, rrows)| {
                                self.chow_test(&rows, lrows, rrows, &path, sp.col)
                            });
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
                            let leaf = self.make_leaf(self.leaf_rows(&rows, &est), &path);
                            prof.leaf_ns += t.elapsed().as_nanos() as u64;
                            self.profiles.push(prof);
                            let stats = FrameStats::from_ty(self.y, self.w, &rows);
                            let cost = if self.params.prune == PruneCost::None {
                                0.0
                            } else {
                                self.leaf_cost(&rows, &path, stats.sse)
                            };
                            let node_idx = self.add(Node {
                                split: None,
                                left: None,
                                right: None,
                                leaf_id: None,
                                leaf: Some(leaf),
                                n: rows.len(),
                                stats,
                            });
                            self.costs.push(cost);
                            results[res_idx] = (node_idx, 1);
                        }
                        Some((mut sp, lrows, rrows)) => {
                            // NA surrogate splits (Phase 3D: only search when the primary split
                            // column has missing values in this node; off unless maxsurrogate > 0).
                            if self.params.maxsurrogate > 0 && rows.iter().any(|&r| self.cols[sp.col][r].is_none()) {
                                let found = self.find_surrogates(&rows, &sp, &lrows);
                                if !found.is_empty() {
                                    sp.sur = Some(self.sur_parts.len());
                                    self.sur_parts.push(found);
                                }
                            }
                            let (est_l, est_r) = self.partition_rows(&est, &sp);
                            let left_res_idx = results.len();
                            results.push((0, 0));
                            let right_res_idx = results.len();
                            results.push((0, 0));

                            stack.push(GrowOp::AfterLeft {
                                depth,
                                rows,
                                est,
                                rrows,
                                est_r,
                                sp,
                                budget,
                                prof,
                                left_res_idx,
                                right_res_idx,
                                res_idx,
                            });
                            path.push(sp.col);
                            stack.push(GrowOp::Grow {
                                depth: depth + 1,
                                rows: lrows,
                                est: est_l,
                                budget: budget.saturating_sub(1),
                                res_idx: left_res_idx,
                            });
                        }
                    }
                }
                GrowOp::AfterLeft {
                    depth,
                    rows,
                    est,
                    rrows,
                    est_r,
                    sp,
                    budget,
                    prof,
                    left_res_idx,
                    right_res_idx,
                    res_idx,
                } => {
                    let (_l, lu) = results[left_res_idx];
                    stack.push(GrowOp::FinishSplit {
                        rows,
                        est,
                        sp,
                        prof,
                        left_res_idx,
                        right_res_idx,
                        res_idx,
                    });
                    stack.push(GrowOp::Grow {
                        depth: depth + 1,
                        rows: rrows,
                        est: est_r,
                        budget: budget.saturating_sub(lu),
                        res_idx: right_res_idx,
                    });
                }
                GrowOp::FinishSplit {
                    rows,
                    est,
                    sp,
                    mut prof,
                    left_res_idx,
                    right_res_idx,
                    res_idx,
                } => {
                    path.pop();
                    let (l, lu) = results[left_res_idx];
                    let (r, ru) = results[right_res_idx];
                    let used = lu + ru;
                    if self.params.prune != PruneCost::None {
                        // Bottom-up cost pruning: keep the subtree only when it
                        // codes cheaper than this node as one leaf.
                        let stats = FrameStats::from_ty(self.y, self.w, &rows);
                        let leaf_c = self.leaf_cost(&rows, &path, stats.sse);
                        let split_bit = self.params.prune.split_bit(rows.len() as f64);
                        let sub_c = self.costs[l] + self.costs[r] + split_bit;
                        if leaf_c <= sub_c {
                            let t = Instant::now();
                            let leaf = self.make_leaf(self.leaf_rows(&rows, &est), &path);
                            prof.leaf_ns += t.elapsed().as_nanos() as u64;
                            self.profiles.push(prof);
                            let node_idx = self.add(Node {
                                split: None,
                                left: None,
                                right: None,
                                leaf_id: None,
                                leaf: Some(leaf),
                                n: rows.len(),
                                stats,
                            });
                            self.costs.push(leaf_c);
                            results[res_idx] = (node_idx, 1);
                            continue;
                        }
                        self.profiles.push(prof);
                        let node_idx = self.add(Node {
                            split: Some(sp),
                            left: Some(l),
                            right: Some(r),
                            leaf_id: None,
                            leaf: None,
                            n: rows.len(),
                            stats,
                        });
                        self.costs.push(sub_c);
                        results[res_idx] = (node_idx, used);
                        continue;
                    }
                    self.profiles.push(prof);
                    let node_idx = self.add(Node {
                        split: Some(sp),
                        left: Some(l),
                        right: Some(r),
                        leaf_id: None,
                        leaf: None,
                        n: rows.len(),
                        stats: FrameStats::from_ty(self.y, self.w, &rows),
                    });
                    self.costs.push(0.0);
                    results[res_idx] = (node_idx, used);
                }
            }
        }

        results[0]
    }

}

