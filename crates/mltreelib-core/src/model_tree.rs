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
pub use crate::leaf::{LeafControl, LeafNaPolicy, LeafSolver};
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
    /// Leaf model optimization solver: Irls (default), Lbfgs, or Ccd.
    pub leaf_solver: LeafSolver,
    /// Ridge penalty on leaf coefficients (intercept unpenalized).
    pub l2: f64,
    /// Lasso penalty on leaf coefficients (intercept unpenalized). With
    /// `l2` this is the elastic net. 0 = off.
    pub l1: f64,
    /// Per-column monotone direction for tree splits: -1 / 0 / +1. A split on
    /// a constrained column must order children's target means accordingly.
    pub tree_monotone: Vec<i8>,
    /// Per-column monotone direction for leaf linear coefficients: -1 / 0 / +1.
    pub leaf_monotone: Vec<i8>,
    /// Legacy combined monotone direction (defaults both tree_monotone and leaf_monotone).
    pub monotone: Vec<i8>,
    /// Interaction groups: features that may interact with each other
    /// (LightGBM-style). Empty = unconstrained. Groups are disjoint; a
    /// path may only use columns from one group.
    pub interaction_groups: Vec<Vec<usize>>,
    /// Per-column importance / priority weights for the split search:
    /// candidate split gains on column j are multiplied by `col_weights[j]`.
    /// Also weights column selection during random `col_sample`. None = uniform 1.0.
    pub col_weights: Option<Vec<f64>>,
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
    /// Purity gate: a node whose weighted target variation is at
    /// most this stays a leaf (its signal is exhausted). `None` = off.
    pub purity_gate: Option<f64>,
    /// Metric evaluated by the purity gate: "sd" (weighted standard deviation)
    /// or "cv" (coefficient of variation: w_sd / |w_mean|).
    pub purity_gate_metric: PurityMetric,
    /// Tail quantile threshold for extrema protection (e.g. 0.05). If set,
    /// nodes containing targets in the lower (<= tau) or upper (>= 1 - tau)
    /// quantiles bypass the purity gate so extrema are always trained.
    pub tail_quantile: Option<f64>,
    /// Power parameter for Tweedie deviance (default 1.5, with 1 < p < 2).
    pub tweedie_p: f64,
    /// Censoring threshold for Tobit response (default 0.0).
    pub tobit_threshold: f64,
    /// Split evaluation strategy (fast scalar variance reduction vs exhaustive linear scan).
    pub split_eval: SplitEval,
    /// Upfront pre-sorting of numeric columns. Default `false` (in-place cache-local PDQsort in SplitScratch).
    pub presort: bool,
    /// Optional event indicators for survival outcome pairs (T, E).
    pub events: Option<Vec<f64>>,
    /// Optional treatment indicators (1=treated, 0=control) for causal outcome pairs (Y, W/T).
    pub treatments: Option<Vec<f64>>,
    /// Whether group columns are eligible for tree building (default true). If false, group columns are dropped.
    pub train_groups: bool,
    /// Whether row ID columns are eligible for tree building (default false). If false, ID columns are dropped.
    pub train_ids: bool,
    /// Number of distinct classes for softmax targets.
    pub n_classes: Option<usize>,
    /// Maximum number of distinct integer classes supported by fast histogram binning (default 256).
    pub max_classes: usize,
    /// Maximum degree of interaction for MARS leaves (default 1).
    pub mars_degree: usize,
    /// Maximum number of terms for MARS leaves (default 10).
    pub mars_terms: usize,
    /// GCV penalty for MARS leaves (default 2.0).
    pub mars_penalty: f64,
    /// Number of nearest neighbors to query at leaf KNN inference (default 5).
    pub knn_k: usize,
    /// Secondary neighborhood size for precomputing neighbor reliability / compactness (default 5).
    pub knn_k_validity: usize,
    /// KNN weighting variant: Uniform, Distance, Exponential, Dual, Double (DW-KNN).
    pub knn_variant: crate::knn::KnnVariant,
    /// Distance metric for leaf KNN (Euclidean, Manhattan, Chebyshev, Gower, etc.).
    pub knn_metric: crate::knn::DistanceMetric,
    /// Decay scale gamma for exponential / double weighting (None => auto-scale by median distance).
    pub knn_gamma: Option<f64>,
    /// Power parameter p for inverse distance / Minkowski metric (default 2.0).
    pub knn_p: f64,
    /// Adaptive hierarchical leaf shrinkage penalty lambda (default 0.0 = off).
    /// Shrinks leaf models toward their parent node: beta_shrunk = beta_parent + (N / (N + lambda)) * (beta_leaf - beta_parent).
    pub shrinkage: f64,
    /// Missing value strategy at leaf level: DropFallback (default), ImputeLocal, or Indicator.
    pub leaf_na_policy: LeafNaPolicy,
}

/// Metric evaluated by the leaf purity gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PurityMetric {
    /// Weighted standard deviation (absolute scale).
    #[default]
    Sd,
    /// Coefficient of variation (scale-invariant: w_sd / |w_mean|).
    Cv,
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
            leaf_solver: LeafSolver::Irls,
            l2: 0.5,
            l1: 0.0,
            tree_monotone: Vec::new(),
            leaf_monotone: Vec::new(),
            monotone: Vec::new(),
            interaction_groups: Vec::new(),
            col_weights: None,
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
            purity_gate_metric: PurityMetric::Sd,
            tail_quantile: None,
            tweedie_p: 1.5,
            tobit_threshold: 0.0,
            split_eval: SplitEval::Fast,
            presort: false,
            events: None,
            treatments: None,
            train_groups: true,
            train_ids: false,
            n_classes: None,
            max_classes: 256,
            mars_degree: 1,
            mars_terms: 10,
            mars_penalty: 2.0,
            knn_k: 5,
            knn_k_validity: 5,
            knn_variant: crate::knn::KnnVariant::Uniform,
            knn_metric: crate::knn::DistanceMetric::Euclidean,
            knn_gamma: None,
            knn_p: 2.0,
            shrinkage: 0.0,
            leaf_na_policy: LeafNaPolicy::Indicator,
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

/// Exact TreeSHAP feature attributions for a single prediction.
#[derive(Clone, Debug)]
pub struct Shapley {
    /// Expected value across the training data (base value).
    pub base_value: f64,
    /// `(column, attribution)` for all features. Efficiency: base_value + Σ values == prediction.
    pub values: Vec<(usize, f64)>,
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

/// Compact training sample memory for Cubist's signature k-NN neighbor adjustment.
#[derive(Clone, Debug)]
pub struct TrainingMemory {
    pub x: Vec<Vec<Option<f64>>>,
    pub y: Vec<f64>,
    pub base_preds: Vec<f64>,
    pub col_stds: Vec<f64>,
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
    /// Training memory for nearest-neighbor adjustments.
    pub training_memory: Option<TrainingMemory>,
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
        let mut params = params;
        if params.tree_monotone.is_empty() && !params.monotone.is_empty() {
            params.tree_monotone = params.monotone.clone();
        }
        if params.leaf_monotone.is_empty() && !params.monotone.is_empty() {
            params.leaf_monotone = params.monotone.clone();
        }
        if params.tree_monotone.iter().any(|&m| m < -1 || m > 1) {
            return Err("tree_monotone entries must be -1, 0 or 1".to_string());
        }
        if !params.tree_monotone.is_empty() && params.tree_monotone.len() != d {
            return Err("tree_monotone must have one entry per column".to_string());
        }
        if params.leaf_monotone.iter().any(|&m| m < -1 || m > 1) {
            return Err("leaf_monotone entries must be -1, 0 or 1".to_string());
        }
        if !params.leaf_monotone.is_empty() && params.leaf_monotone.len() != d {
            return Err("leaf_monotone must have one entry per column".to_string());
        }
        if let Some(cw) = &params.col_weights {
            if cw.len() != d {
                return Err("col_weights must have one entry per column".to_string());
            }
            if cw.iter().any(|&w| w < 0.0 || !w.is_finite()) {
                return Err("col_weights entries must be non-negative finite floats".to_string());
            }
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
            if !matches!(
                params.leaf,
                LeafKind::Mean
                    | LeafKind::Linear
                    | LeafKind::Mars
                    | LeafKind::Knn
                    | LeafKind::Prob
                    | LeafKind::Odds
                    | LeafKind::LogOdds
            ) {
                return Err("non-gaussian families support mean, linear, mars, prob, odds, or log_odds leaves".into());
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
            if t.is_nan() {
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
                    LeafFamily::Binomial | LeafFamily::Probit if t != 0.0 && t != 1.0 => {
                        return Err("binomial and probit targets must be 0 or 1".into());
                    }
                    LeafFamily::Poisson | LeafFamily::Tweedie if t < 0.0 => {
                        return Err("poisson and tweedie targets must be non-negative".into());
                    }
                    LeafFamily::Gamma if t <= 0.0 => {
                        return Err("gamma targets must be strictly positive".into());
                    }
                    LeafFamily::Softmax if t < 0.0 || t.fract() != 0.0 => {
                        return Err("softmax targets must be non-negative integers (0, 1, 2, ...)".into());
                    }
                    LeafFamily::Survival if t < 0.0 => {
                        return Err("survival durations must be non-negative".into());
                    }
                    _ => {}
                }
            }
            for j in 0..d {
                let value = match row[j] {
                    Some(v) if v.is_nan() => None,
                    Some(v) if !v.is_finite() && params.leaf_family != LeafFamily::Gaussian => {
                        return Err("distributional predictors must be finite or missing".into());
                    }
                    value => value,
                };
                cols[j].push(value);
            }
            yv.push(t);
            wv.push(w[r]);
        }
        if yv.is_empty() {
            return Err("no rows with an observed target and positive weight".to_string());
        }

        if params.leaf_family == LeafFamily::Softmax && params.n_classes.is_none() {
            let max_c = yv.iter().map(|&y| y as usize).max().unwrap_or(1);
            params.n_classes = Some(max_c + 1);
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
                    vec![0]
                } else {
                    kept
                }
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
                    vec![0]
                } else {
                    kept
                }
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
                    vec![0]
                } else {
                    kept
                }
            }
            crate::preselect::Preselect::Variance => {
                let kept = preselect::variance(&cols, &yv, &wv, params.min_relevance.unwrap_or(0.0));
                if kept.is_empty() {
                    vec![0]
                } else {
                    kept
                }
            }
            crate::preselect::Preselect::MutualInfo => {
                let kept = preselect::mutual_info(&cols, &yv, &wv, params.min_relevance.unwrap_or(0.0));
                if kept.is_empty() {
                    vec![0]
                } else {
                    kept
                }
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
                    vec![0]
                } else {
                    kept
                }
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
        let tail_bounds = if let Some(tau) = params.tail_quantile {
            if tau > 0.0 && tau < 0.5 && !yv.is_empty() {
                let mut sorted_y: Vec<f64> = yv.clone();
                sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx_low = ((sorted_y.len() as f64) * tau).floor() as usize;
                let idx_high = ((sorted_y.len() as f64) * (1.0 - tau)).floor() as usize;
                let y_low = sorted_y[idx_low.min(sorted_y.len() - 1)];
                let y_high = sorted_y[idx_high.min(sorted_y.len() - 1)];
                Some((y_low, y_high))
            } else {
                None
            }
        } else {
            None
        };
        let start_time = Instant::now();
        let mut grower = Grower {
            cols: &cols,
            col_names: &col_names,
            y: &yv,
            w: &wv,
            treatments: params.treatments.as_deref(),
            params: &params,
            active: &active_cols,
            nodes: Vec::new(),
            profiles: Vec::new(),
            cat_parts: Vec::new(),
            sur_parts: Vec::new(),
            trace: Vec::new(),
            costs: Vec::new(),
            rng_counter: 0,
            tail_bounds,
            max_depth_seen: 0,
        };
        if params.verbose > 0 {
            grower.log(
                1,
                &format!(
                    "[ModelTree] fit start: rows={} cols={} max_depth={} min_rows={} leaf={:?} family={:?} gain={:?}",
                    rows.len(),
                    active_cols.len(),
                    params.max_depth,
                    params.min_rows,
                    params.leaf,
                    params.leaf_family,
                    params.gain
                ),
            );
            grower.log(
                1,
                &format!(
                    "scan: gain={:?} split_precision={:?} cat_search={:?} ({:?}) leaf={:?} leaf_digest={}",
                    params.gain,
                    params.split_precision,
                    params.cat_search,
                    params.cat_order,
                    params.leaf,
                    params.leaf_digest
                ),
            );
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
            for (d, (nodes, rows, candidates, scan_ms)) in &by_depth {
                let depth_summary = format!(
                    "[ModelTree] depth {d}: nodes={nodes} rows={rows} candidates={candidates} scan_ms={scan_ms}"
                );
                println!("{}", depth_summary);
                trace.push(depth_summary);
            }
            let elapsed_ms = start_time.elapsed().as_millis();
            let n_leaves = nodes.iter().filter(|n| n.split.is_none()).count();
            let max_d = profiles.iter().map(|p| p.depth).max().unwrap_or(0);
            let finish_msg = format!(
                "[ModelTree] fit complete: total_nodes={} leaves={} max_depth={} elapsed_ms={}",
                nodes.len(),
                n_leaves,
                max_d,
                elapsed_ms
            );
            println!("{}", finish_msg);
            trace.push(finish_msg);
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
            training_memory: None,
        };
        tree.assign_leaf_ids();

        let n_train = yv.len();
        let mut col_stds = vec![1.0; d];
        for j in 0..d {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0.0;
            for &val_opt in &cols[j] {
                if let Some(v) = val_opt {
                    if v.is_finite() {
                        sum += v;
                        sum_sq += v * v;
                        count += 1.0;
                    }
                }
            }
            if count > 1.0 {
                let mean = sum / count;
                let var = (sum_sq - count * mean * mean) / (count - 1.0);
                if var > 1e-12 {
                    col_stds[j] = var.sqrt();
                }
            }
        }

        let mut train_rows = vec![vec![None; d]; n_train];
        for j in 0..d {
            for i in 0..n_train {
                train_rows[i][j] = cols[j][i];
            }
        }
        let base_preds = tree.predict(&train_rows);
        tree.training_memory = Some(TrainingMemory {
            x: train_rows,
            y: yv,
            base_preds,
            col_stds,
        });

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

    /// Retrain all leaf models with explicit data column names.
    pub fn retrain_named(
        &mut self,
        data_col_names: &[String],
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        control: LeafControl,
    ) -> Result<(), String> {
        let n = y.len();
        self.retrain_weighted_named(data_col_names, x, y, &vec![1.0; n], control)
    }

    /// Retrain all leaf models with sample weights.
    pub fn retrain_weighted(
        &mut self,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        w: &[f64],
        control: LeafControl,
    ) -> Result<(), String> {
        self.retrain_weighted_named(&self.col_names.clone(), x, y, w, control)
    }

    /// Check whether a column index is used in any split or surrogate in the reachable tree.
    pub fn is_split_column(&self, col: usize) -> bool {
        let reachable = self.reachable();
        for (i, node) in self.nodes.iter().enumerate() {
            if reachable[i] {
                if let Some(sp) = &node.split {
                    if sp.col == col {
                        return true;
                    }
                    if let Some(si) = sp.sur {
                        if self.sur_parts[si].iter().any(|s| s.col == col) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Retrain all leaf models with explicit data column names.
    ///
    /// Validates that all split columns are present in `data_col_names`.
    /// Leaf models fit on `control.leaf_features` if provided, or `control.leaf_cols`.
    pub fn retrain_weighted_named(
        &mut self,
        data_col_names: &[String],
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        w: &[f64],
        control: LeafControl,
    ) -> Result<(), String> {
        let n_all = y.len();
        if x.len() != n_all || w.len() != n_all {
            return Err("x, y and w length mismatch".to_string());
        }
        let d = data_col_names.len();
        for r in x {
            if r.len() != d {
                return Err("row width != number of data column names".to_string());
            }
        }
        if control.l1 < 0.0 || control.l2 < 0.0 {
            return Err("penalties l1 and l2 must be non-negative".to_string());
        }
        if control.family != LeafFamily::Gaussian {
            if !matches!(
                control.leaf,
                LeafKind::Mean
                    | LeafKind::Linear
                    | LeafKind::Mars
                    | LeafKind::Knn
                    | LeafKind::Prob
                    | LeafKind::Odds
                    | LeafKind::LogOdds
            ) {
                return Err("non-gaussian families support mean, linear, mars, prob, odds, or log_odds leaves".into());
            }
        }

        // Validate that all split columns exist in data_col_names
        let mut col_map = vec![usize::MAX; self.col_names.len()];
        for (j, name) in self.col_names.iter().enumerate() {
            if let Some(pos) = data_col_names.iter().position(|c| c == name) {
                col_map[j] = pos;
            } else if self.is_split_column(j) {
                return Err(format!("required split column '{name}' missing from retrain data"));
            }
        }

        // Validate leaf_features if provided
        if let Some(ref feats) = control.leaf_features {
            for &f in feats {
                if f >= d {
                    return Err(format!("leaf_features column index {f} out of bounds for width {d}"));
                }
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
            if t.is_nan() {
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
                    LeafFamily::Binomial | LeafFamily::Probit if t != 0.0 && t != 1.0 => {
                        return Err("binomial and probit targets must be 0 or 1".into());
                    }
                    LeafFamily::Poisson | LeafFamily::Tweedie if t < 0.0 => {
                        return Err("poisson and tweedie targets must be non-negative".into());
                    }
                    LeafFamily::Gamma if t <= 0.0 => {
                        return Err("gamma targets must be strictly positive".into());
                    }
                    LeafFamily::Softmax if t < 0.0 || t.fract() != 0.0 => {
                        return Err("softmax targets must be non-negative integers (0, 1, 2, ...)".into());
                    }
                    LeafFamily::Survival if t < 0.0 => {
                        return Err("survival durations must be non-negative".into());
                    }
                    _ => {}
                }
            }
            for j in 0..d {
                let value = match row[j] {
                    Some(v) if v.is_nan() => None,
                    Some(v) if !v.is_finite() && control.family != LeafFamily::Gaussian => {
                        return Err("distributional predictors must be finite or missing".into());
                    }
                    value => value,
                };
                cols[j].push(value);
            }
            yv.push(t);
            wv.push(w[r]);

            let leaf_idx = self.leaf_index_mapped(row, &col_map);
            leaf_assignments.push(leaf_idx);
        }

        let mut control = control.clone();
        if control.family == LeafFamily::Softmax && control.n_classes.is_none() {
            let max_c = yv.iter().map(|&y| y as usize).max().unwrap_or(1);
            control.n_classes = Some(max_c + 1);
        }

        // Group valid row indices by leaf node index
        let mut leaf_rows_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &leaf_idx) in leaf_assignments.iter().enumerate() {
            leaf_rows_map.entry(leaf_idx).or_default().push(i);
        }

        // 2. Precompute root-to-node split paths (mapped to data column indices)
        let mut node_paths = vec![Vec::new(); self.nodes.len()];
        fn walk_mapped_paths(
            nodes: &[Node],
            i: usize,
            cur: &mut Vec<usize>,
            node_paths: &mut [Vec<usize>],
            col_map: &[usize],
        ) {
            node_paths[i] = cur.clone();
            let n = &nodes[i];
            if let (Some(sp), Some(l), Some(r)) = (&n.split, n.left, n.right) {
                let mapped = col_map[sp.col];
                if mapped != usize::MAX {
                    cur.push(mapped);
                }
                walk_mapped_paths(nodes, l, cur, node_paths, col_map);
                walk_mapped_paths(nodes, r, cur, node_paths, col_map);
                if mapped != usize::MAX {
                    cur.pop();
                }
            }
        }
        walk_mapped_paths(&self.nodes, self.root, &mut Vec::new(), &mut node_paths, &col_map);

        let active_cols: Vec<usize> = (0..d).collect();

        // 3. For each active leaf in the tree, refit its LeafModel
        let reachable = self.reachable();
        for i in 0..self.nodes.len() {
            if !reachable[i] || self.nodes[i].leaf.is_none() {
                continue;
            }
            if let Some(rows) = leaf_rows_map.get(&i) {
                let path = &node_paths[i];
                let mut leaf_model = fit_leaf_model(
                    &cols,
                    &yv,
                    &wv,
                    rows,
                    path,
                    &active_cols,
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
                        leaf_model.linear = None;
                        if leaf_model.center.is_none() || !leaf_model.center.unwrap().is_finite() {
                            leaf_model.center = Some(self.nodes[i].stats.mean);
                        }
                    }
                }
                self.nodes[i].stats = FrameStats::from_ty(&yv, &wv, rows);
                self.nodes[i].n = rows.len();
                self.nodes[i].leaf = Some(leaf_model);
            }
        }

        // 4. Remap split columns in tree and update col_names
        for node in self.nodes.iter_mut() {
            if let Some(sp) = &mut node.split {
                sp.col = col_map[sp.col];
            }
        }
        for sur_list in self.sur_parts.iter_mut() {
            for s in sur_list.iter_mut() {
                if col_map[s.col] != usize::MAX {
                    s.col = col_map[s.col];
                }
            }
        }
        self.col_names = data_col_names.to_vec();
        self.active_cols = active_cols;

        // 5. Update tree parameters to reflect new leaf controls
        self.params.leaf = control.leaf;
        self.params.leaf_family = control.family;
        self.params.leaf_cols = control.leaf_cols;
        self.params.l2 = control.l2;
        self.params.l1 = control.l1;
        self.params.leaf_digest = control.leaf_digest;
        self.params.events = control.events.clone();
        self.params.n_classes = control.n_classes;
        self.params.mars_degree = control.mars_degree;
        self.params.mars_terms = control.mars_terms;
        self.params.mars_penalty = control.mars_penalty;
        self.params.tweedie_p = control.tweedie_p;
        self.params.tobit_threshold = control.tobit_threshold;
        if !control.monotone.is_empty() {
            self.params.monotone = control.monotone;
        }

        Ok(())
    }

    /// Grid-aware test for surrogate split.
    fn goes_left_surrogate(&self, s: &crate::surrogate::Surrogate, v: f64) -> bool {
        match crate::split::grid_of(&self.params.dtypes, self.params.split_precision, s.col) {
            Some(p) if p > 0.0 => {
                let q = 1.0 / p;
                (v * q).round() / q <= s.threshold
            }
            _ => v <= s.threshold,
        }
    }

    /// Walk tree using column mapping from tree indices to row indices.
    pub fn leaf_index_mapped(&self, row: &[Option<f64>], col_map: &[usize]) -> usize {
        let mut node = self.root;
        while let (Some(sp), Some(l), Some(r)) = (
            &self.nodes[node].split,
            self.nodes[node].left,
            self.nodes[node].right,
        ) {
            let mapped_col = col_map[sp.col];
            let go_left = match row.get(mapped_col).and_then(|v| *v) {
                Some(v) => self.goes_left(sp, v),
                None => self.surrogate_left_mapped(row, sp, col_map),
            };
            node = if go_left { l } else { r };
        }
        node
    }

    /// Return integer leaf ID for a row mapped by col_map.
    pub fn leaf_id_mapped(&self, row: &[Option<f64>], col_map: &[usize]) -> usize {
        let node_idx = self.leaf_index_mapped(row, col_map);
        self.nodes[node_idx].leaf_id.unwrap_or(0)
    }

    /// Return row indices grouped by leaf_id: leaf_id -> [row_indices] with mapped columns.
    pub fn leaf_samples_mapped(
        &self,
        x: &[Vec<Option<f64>>],
        col_map: &[usize],
    ) -> std::collections::BTreeMap<usize, Vec<usize>> {
        let mut map = std::collections::BTreeMap::new();
        for (i, row) in x.iter().enumerate() {
            let leaf_id = self.leaf_id_mapped(row, col_map);
            map.entry(leaf_id).or_insert_with(Vec::new).push(i);
        }
        map
    }

    fn surrogate_left_mapped(&self, row: &[Option<f64>], sp: &Split, col_map: &[usize]) -> bool {
        if let Some(si) = sp.sur {
            for s in &self.sur_parts[si] {
                let mapped_col = col_map[s.col];
                if mapped_col != usize::MAX {
                    if let Some(v) = row.get(mapped_col).and_then(|v| *v) {
                        let left = self.goes_left_surrogate(s, v);
                        return if s.flip { !left } else { left };
                    }
                }
            }
        }
        sp.missing_left
    }

    /// Route a row missing the split column: the first observed surrogate
    /// decides (mirrored when `flip`), else the split's `missing_left`.
    /// Surrogate comparisons share the grid convention with `goes_left`.
    fn surrogate_left(&self, row: &[Option<f64>], sp: &Split) -> bool {
        if let Some(si) = sp.sur {
            for s in &self.sur_parts[si] {
                if let Some(v) = row.get(s.col).and_then(|v| *v) {
                    let left = self.goes_left_surrogate(s, v);
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

    /// One P(class 1) per row for binary families (Binomial, Probit).
    pub fn predict_proba(&self, x: &[Vec<Option<f64>>]) -> Result<Vec<f64>, String> {
        if self.params.leaf_family != LeafFamily::Binomial && self.params.leaf_family != LeafFamily::Probit {
            return Err("predict_proba requires leaf_family='binomial' or 'probit'".into());
        }
        Ok(self.predict(x))
    }

    /// Full class probability distribution per row (supports Softmax, Binomial, Probit).
    pub fn predict_proba_matrix(&self, x: &[Vec<Option<f64>>]) -> Result<Vec<Vec<f64>>, String> {
        if !matches!(self.params.leaf_family, LeafFamily::Softmax | LeafFamily::Binomial | LeafFamily::Probit) {
            return Err("predict_proba_matrix requires leaf_family='softmax', 'binomial' or 'probit'".into());
        }
        Ok(x.iter()
            .map(|row| {
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict_proba_row(row)
            })
            .collect())
    }

    /// Survival step probabilities S(t) across a given time grid.
    pub fn predict_survival(&self, x: &[Vec<Option<f64>>], times: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        if self.params.leaf_family != LeafFamily::Survival {
            return Err("predict_survival requires leaf_family='survival'".into());
        }
        Ok(x.iter()
            .map(|row| {
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict_survival_row(times)
            })
            .collect())
    }

    /// Cumulative hazard Lambda(t) across a given time grid.
    pub fn predict_hazard(&self, x: &[Vec<Option<f64>>], times: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        if self.params.leaf_family != LeafFamily::Survival {
            return Err("predict_hazard requires leaf_family='survival'".into());
        }
        Ok(x.iter()
            .map(|row| {
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict_hazard_row(times)
            })
            .collect())
    }

    /// Individual treatment effect (CATE / uplift) tau = E[Y|T=1, X] - E[Y|T=0, X] per row.
    pub fn predict_uplift(&self, x: &[Vec<Option<f64>>]) -> Result<Vec<f64>, String> {
        Ok(x.iter()
            .map(|row| {
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict_uplift()
            })
            .collect())
    }

    /// Predict location mu and dispersion sigma (mean, std) per row for heteroscedastic leaves.
    pub fn predict_dist(&self, x: &[Vec<Option<f64>>]) -> Result<Vec<(f64, f64)>, String> {
        Ok(x.iter()
            .map(|row| {
                let node = &self.nodes[self.leaf_index(row)];
                node.leaf
                    .as_ref()
                    .expect("tree invariant: leaf holds a model")
                    .predict_dist_row(row)
            })
            .collect())
    }

    /// Hard binary class labels (0 or 1) per row by thresholding binomial or probit probabilities.
    /// Default threshold is 0.5. Requires `leaf_family='binomial'` or `'probit'`.
    pub fn predict_label(&self, x: &[Vec<Option<f64>>], threshold: f64) -> Result<Vec<usize>, String> {
        if self.params.leaf_family != LeafFamily::Binomial && self.params.leaf_family != LeafFamily::Probit {
            return Err("predict_label requires leaf_family='binomial' or 'probit'".into());
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

    /// Exact TreeSHAP feature attributions for a single row prediction.
    ///
    /// Computes exact Shapley values $\phi_j$ in $O(L \cdot D^2)$ time:
    /// efficiency invariant: `base_value + \sum \phi_j == prediction`.
    pub fn shapley(&self, x: &[Option<f64>]) -> Shapley {
        let seen = self.reachable();
        let mut ev = 0.0;
        let mut total_w = 0.0;
        for (i, nd) in self.nodes.iter().enumerate() {
            if seen[i] && nd.left.is_none() && nd.right.is_none() {
                if let Some(leaf) = &nd.leaf {
                    let n = nd.n as f64;
                    ev += n * leaf.predict(x);
                    total_w += n;
                }
            }
        }
        let base_value = if total_w > 0.0 { ev / total_w } else { self.nodes[self.root].stats.mean };

        let mut phi = vec![0.0f64; self.col_names.len()];
        let path = TreeShapPath::new();
        tree_shap_recurse(self, self.root, path, x, &mut phi);

        let prediction = self.predict(&[x.to_vec()])[0];
        let values: Vec<(usize, f64)> = (0..self.col_names.len())
            .map(|j| (j, phi[j]))
            .collect();

        Shapley {
            base_value,
            values,
            prediction,
        }
    }

    /// Mean absolute Shapley feature importance across a dataset:
    /// $\frac{1}{N} \sum_{i=1}^N |\phi_{i, j}|$, sorted descending by importance.
    pub fn shapley_importance(&self, x: &[Vec<Option<f64>>]) -> Vec<(String, f64)> {
        if x.is_empty() {
            return self.col_names.iter().map(|n| (n.clone(), 0.0)).collect();
        }
        let mut sum_abs = vec![0.0f64; self.col_names.len()];
        for row in x {
            let sh = self.shapley(row);
            for &(col, val) in &sh.values {
                sum_abs[col] += val.abs();
            }
        }
        let n = x.len() as f64;
        let mut out: Vec<(String, f64)> = self
            .col_names
            .iter()
            .enumerate()
            .map(|(j, name)| (name.clone(), sum_abs[j] / n))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Permutation feature importance: per-column validation loss degradation
    /// when the column is shuffled across rows. Returns `(column_name, degradation)`
    /// sorted descending.
    pub fn permutation_importance(
        &self,
        x: &[Vec<Option<f64>>],
        y: &[Option<f64>],
        metric: &str,
        n_repeats: usize,
        seed: u64,
    ) -> Result<Vec<(String, f64)>, String> {
        let n = x.len();
        if n == 0 || y.len() != n {
            return Err("x and y must have equal non-zero length".to_string());
        }
        let n_cols = self.col_names.len();
        let base_preds = self.predict(x);

        let loss_fn = |preds: &[f64]| -> Result<f64, String> {
            match metric {
                "rmse" => {
                    let mut sum_sq = 0.0;
                    let mut count = 0.0;
                    for (&t, &p) in y.iter().zip(preds.iter()) {
                        if let Some(act) = t {
                            sum_sq += (act - p).powi(2);
                            count += 1.0;
                        }
                    }
                    if count == 0.0 {
                        return Err("no valid targets".into());
                    }
                    Ok((sum_sq / count).sqrt())
                }
                "mae" => {
                    let mut sum_abs = 0.0;
                    let mut count = 0.0;
                    for (&t, &p) in y.iter().zip(preds.iter()) {
                        if let Some(act) = t {
                            sum_abs += (act - p).abs();
                            count += 1.0;
                        }
                    }
                    if count == 0.0 {
                        return Err("no valid targets".into());
                    }
                    Ok(sum_abs / count)
                }
                "r2" => {
                    let mut sum_y = 0.0;
                    let mut count = 0.0;
                    for &t in y {
                        if let Some(act) = t {
                            sum_y += act;
                            count += 1.0;
                        }
                    }
                    let mean_y = sum_y / count;
                    let mut ss_tot = 0.0;
                    let mut ss_res = 0.0;
                    for (&t, &p) in y.iter().zip(preds.iter()) {
                        if let Some(act) = t {
                            ss_tot += (act - mean_y).powi(2);
                            ss_res += (act - p).powi(2);
                        }
                    }
                    if ss_tot == 0.0 {
                        Ok(1.0)
                    } else {
                        Ok(1.0 - ss_res / ss_tot)
                    }
                }
                "logloss" => {
                    let mut loss = 0.0;
                    let mut count = 0.0;
                    for (&t, &p) in y.iter().zip(preds.iter()) {
                        if let Some(act) = t {
                            let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                            loss += -act * p_clamped.ln() - (1.0 - act) * (1.0 - p_clamped).ln();
                            count += 1.0;
                        }
                    }
                    if count == 0.0 {
                        return Err("no valid targets".into());
                    }
                    Ok(loss / count)
                }
                other => Err(format!("unknown permutation metric: '{other}'")),
            }
        };

        let base_loss = loss_fn(&base_preds)?;
        let mut total_deg = vec![0.0f64; n_cols];
        let repeats = n_repeats.max(1);

        for j in 0..n_cols {
            for r in 0..repeats {
                // Deterministic Fisher-Yates shuffle of column j
                let mut perm_indices: Vec<usize> = (0..n).collect();
                let mut state = seed.wrapping_add((j as u64 + 1) * 31337).wrapping_add(r as u64 * 7919);
                for i in (1..n).rev() {
                    // xorshift64
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let pick = (state as usize) % (i + 1);
                    perm_indices.swap(i, pick);
                }

                // Construct permuted column inputs on-the-fly
                let mut perm_x = x.to_vec();
                for i in 0..n {
                    perm_x[i][j] = x[perm_indices[i]][j];
                }
                let perm_preds = self.predict(&perm_x);
                let perm_loss = loss_fn(&perm_preds)?;

                let deg = if metric == "r2" {
                    base_loss - perm_loss // For R2: degradation is drop in R2
                } else {
                    perm_loss - base_loss // For loss: degradation is increase in loss
                };
                total_deg[j] += deg;
            }
        }

        let mut out: Vec<(String, f64)> = self
            .col_names
            .iter()
            .enumerate()
            .map(|(j, name)| (name.clone(), total_deg[j] / repeats as f64))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Unified feature importance accessor: `method` in `["gain", "shapley", "permutation"]`.
    pub fn feature_importance_by(
        &self,
        method: &str,
        x: Option<&[Vec<Option<f64>>]>,
        y: Option<&[Option<f64>]>,
    ) -> Result<Vec<(String, f64)>, String> {
        match method {
            "gain" => Ok(self.feature_importances()),
            "shapley" => {
                let x_data = x.ok_or_else(|| "feature_importance(method='shapley') requires X".to_string())?;
                Ok(self.shapley_importance(x_data))
            }
            "permutation" => {
                let x_data = x.ok_or_else(|| "feature_importance(method='permutation') requires X".to_string())?;
                let y_data = y.ok_or_else(|| "feature_importance(method='permutation') requires y".to_string())?;
                self.permutation_importance(x_data, y_data, "rmse", 5, 42)
            }
            other => Err(format!(
                "unknown importance method '{other}'; expected 'gain', 'shapley', or 'permutation'"
            )),
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

    /// Extract a C5.0-style rule set from this fitted tree.
    ///
    /// Each path from root to leaf becomes an IF-THEN rule.
    /// When training data `x` and `y` are provided, empirical coverage and error
    /// rates are computed, and preconditions can be greedily pruned using Quinlan's
    /// pessimistic error rate upper bound (binomial confidence interval with `cf`).
    pub fn extract_rules(
        &self,
        x: Option<&[Vec<Option<f64>>]>,
        y: Option<&[f64]>,
        prune: bool,
        cf: f64,
    ) -> crate::rules::RuleSet {
        use crate::rules::{cf_to_z, pessimistic_error_bound, Rule, RuleCondition, RuleSet};

        let mut raw_rules: Vec<Rule> = Vec::new();
        let mut path: Vec<RuleCondition> = Vec::new();

        self.traverse_for_rules(self.root, &mut path, &mut raw_rules);

        let default_prediction = self.nodes[self.root]
            .leaf
            .as_ref()
            .map(|l| l.predict(&[]))
            .unwrap_or(0.0);

        let z = cf_to_z(cf);

        let mut final_rules: Vec<Rule> = Vec::new();
        for (idx, mut rule) in raw_rules.into_iter().enumerate() {
            rule.id = idx + 1;
            rule.simplify_conditions();

            if let (Some(x_data), Some(y_data)) = (x, y) {
                let mut cover = 0;
                let mut errors = 0;
                let is_classifier = self.params.leaf_family == LeafFamily::Binomial
                    || self.params.leaf_family == LeafFamily::Probit;

                for (row, &target) in x_data.iter().zip(y_data.iter()) {
                    if rule.matches(row) {
                        cover += 1;
                        if is_classifier {
                            let pred_class: f64 = if rule.prediction >= 0.5 { 1.0 } else { 0.0 };
                            let true_class: f64 = if target >= 0.5 { 1.0 } else { 0.0 };
                            if (pred_class - true_class).abs() > 1e-9 {
                                errors += 1;
                            }
                        } else {
                            let tol = (rule.prediction * 0.15).abs().max(1.0);
                            if (rule.prediction - target).abs() > tol {
                                errors += 1;
                            }
                        }
                    }
                }

                rule.cover = cover;
                rule.errors = errors;
                let base_pessimistic = pessimistic_error_bound(errors, cover, z);
                rule.confidence = (1.0 - base_pessimistic).max(0.01);

                // Precondition pruning: greedily test dropping individual conditions
                if prune && rule.conditions.len() > 1 {
                    let mut current_conditions = rule.conditions.clone();
                    let mut current_err = base_pessimistic;

                    let mut i = 0;
                    while i < current_conditions.len() {
                        let mut trial_conditions = current_conditions.clone();
                        trial_conditions.remove(i);

                        let mut t_cover = 0;
                        let mut t_errors = 0;
                        for (row, &target) in x_data.iter().zip(y_data.iter()) {
                            let matches = trial_conditions.iter().all(|c| c.matches(row));
                            if matches {
                                t_cover += 1;
                                if is_classifier {
                                    let pred_class: f64 = if rule.prediction >= 0.5 { 1.0 } else { 0.0 };
                                    let true_class: f64 = if target >= 0.5 { 1.0 } else { 0.0 };
                                    if (pred_class - true_class).abs() > 1e-9 {
                                        t_errors += 1;
                                    }
                                } else {
                                    let tol = (rule.prediction * 0.15).abs().max(1.0);
                                    if (rule.prediction - target).abs() > tol {
                                        t_errors += 1;
                                    }
                                }
                            }
                        }

                        let trial_pessimistic = pessimistic_error_bound(t_errors, t_cover, z);
                        if t_cover > 0 && trial_pessimistic <= current_err {
                            current_conditions = trial_conditions;
                            current_err = trial_pessimistic;
                            rule.cover = t_cover;
                            rule.errors = t_errors;
                            rule.confidence = (1.0 - trial_pessimistic).max(0.01);
                        } else {
                            i += 1;
                        }
                    }
                    rule.conditions = current_conditions;
                }
            }

            final_rules.push(rule);
        }

        final_rules.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.cover.cmp(&a.cover))
        });

        for (i, r) in final_rules.iter_mut().enumerate() {
            r.id = i + 1;
        }

        let default_conf = if let Some(y_data) = y {
            let total = y_data.len();
            if total > 0 {
                let matches_default = y_data
                    .iter()
                    .filter(|&&val| (val - default_prediction).abs() < 0.5)
                    .count();
                matches_default as f64 / total as f64
            } else {
                0.5
            }
        } else {
            0.5
        };

        RuleSet {
            rules: final_rules,
            default_prediction,
            default_confidence: default_conf,
            col_names: self.col_names.clone(),
        }
    }

    fn traverse_for_rules(
        &self,
        node_idx: usize,
        path: &mut Vec<crate::rules::RuleCondition>,
        rules: &mut Vec<crate::rules::Rule>,
    ) {
        use crate::rules::{ConditionOp, Rule, RuleCondition};

        let node = &self.nodes[node_idx];
        match (&node.split, node.left, node.right) {
            (Some(sp), Some(left_idx), Some(right_idx)) => {
                let col_name = self
                    .col_names
                    .get(sp.col)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", sp.col));

                let left_cond = if let Some(cat_idx) = sp.cat {
                    let set = self.cat_parts.get(cat_idx).cloned().unwrap_or_default();
                    RuleCondition {
                        col: sp.col,
                        col_name: col_name.clone(),
                        op: ConditionOp::InSet(set),
                        threshold: 0.0,
                    }
                } else {
                    RuleCondition {
                        col: sp.col,
                        col_name: col_name.clone(),
                        op: ConditionOp::LessOrEqual,
                        threshold: sp.threshold,
                    }
                };

                let right_cond = if let Some(cat_idx) = sp.cat {
                    let set = self.cat_parts.get(cat_idx).cloned().unwrap_or_default();
                    RuleCondition {
                        col: sp.col,
                        col_name,
                        op: ConditionOp::NotInSet(set),
                        threshold: 0.0,
                    }
                } else {
                    RuleCondition {
                        col: sp.col,
                        col_name,
                        op: ConditionOp::GreaterThan,
                        threshold: sp.threshold,
                    }
                };

                path.push(left_cond);
                self.traverse_for_rules(left_idx, path, rules);
                path.pop();

                path.push(right_cond);
                self.traverse_for_rules(right_idx, path, rules);
                path.pop();
            }
            _ => {
                let prediction = node.leaf.as_ref().map(|l| l.predict(&[])).unwrap_or(0.0);
                let leaf_id = node.leaf_id.unwrap_or(0);
                rules.push(Rule {
                    id: rules.len() + 1,
                    conditions: path.clone(),
                    prediction,
                    cover: node.n,
                    errors: 0,
                    confidence: 0.95,
                    leaf_id,
                });
            }
        }
    }

    /// Predict with Cubist-style k-NN sample neighbor adjustment.
    ///
    /// For each query row x_0:
    /// 1. Compute tree base prediction y_hat(x_0).
    /// 2. If neighbors > 0 and training memory is available, find the k nearest
    ///    training instances using normalized Manhattan distance.
    /// 3. Compute distance-weighted mean of training residuals:
    ///    w_i = 1 / (d(x_0, x_i) + 1e-5), delta = sum(w_i * (y_i - y_hat(x_i))) / sum(w_i).
    /// 4. Return y_hat(x_0) + delta.
    pub fn predict_adjusted(&self, x: &[Vec<Option<f64>>], neighbors: usize) -> Vec<f64> {
        let base_preds = self.predict(x);
        if neighbors == 0 {
            return base_preds;
        }
        let mem = match &self.training_memory {
            Some(m) if !m.x.is_empty() => m,
            _ => return base_preds,
        };

        let k = neighbors.min(mem.x.len());
        let d_cols = self.col_names.len();
        let mut adjusted = Vec::with_capacity(x.len());

        for (row_idx, row) in x.iter().enumerate() {
            let base_pred = base_preds[row_idx];
            if !base_pred.is_finite() {
                adjusted.push(base_pred);
                continue;
            }

            // Compute distance from `row` to each training sample in `mem`
            let mut dists: Vec<(f64, usize)> = mem
                .x
                .iter()
                .enumerate()
                .map(|(train_idx, train_row)| {
                    let mut dist_sum = 0.0;
                    let mut p_count = 0.0;
                    for j in 0..d_cols {
                        let scale = mem.col_stds.get(j).copied().unwrap_or(1.0);
                        match (row.get(j).and_then(|&v| v), train_row.get(j).and_then(|&v| v)) {
                            (Some(a), Some(b)) => {
                                dist_sum += (a - b).abs() / scale;
                                p_count += 1.0;
                            }
                            (None, None) => {}
                            _ => {
                                dist_sum += 1.0;
                                p_count += 1.0;
                            }
                        }
                    }
                    let avg_dist = if p_count > 0.0 { dist_sum / p_count } else { 1.0 };
                    (avg_dist, train_idx)
                })
                .collect();

            // Select top-k smallest distances
            dists.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut w_sum = 0.0;
            let mut weighted_res_sum = 0.0;
            for i in 0..k {
                let (dist, train_idx) = dists[i];
                let weight = 1.0 / (dist + 1e-5);
                let residual = mem.y[train_idx] - mem.base_preds[train_idx];
                weighted_res_sum += weight * residual;
                w_sum += weight;
            }

            let delta = if w_sum > 0.0 {
                weighted_res_sum / w_sum
            } else {
                0.0
            };

            adjusted.push(base_pred + delta);
        }

        adjusted
    }

    /// Format a leaf's regression model as an R `summary(lm)` / Python `statsmodels` table.
    pub fn leaf_summary(&self, leaf_id: usize) -> Result<String, String> {
        let entries = self.leaf_entries();
        for (leaf_model, n_rows, id, node_idx) in entries {
            if id == leaf_id {
                let title = format!("Leaf Node {} (Leaf ID {}) [n={}]", node_idx, id, n_rows);
                if let Some(lm) = &leaf_model.linear {
                    if let Some(s) = &lm.summary {
                        return Ok(s.to_r_format(&title, &self.col_names));
                    } else {
                        return Ok(format!(
                            "{}\nLinear model: intercept={:.4}, {} coefficients (summary covariance not computed).",
                            title, lm.intercept, lm.coef.len()
                        ));
                    }
                } else {
                    let mean = leaf_model.center.unwrap_or(leaf_model.stats.mean);
                    return Ok(format!(
                        "======================================================================\n\
                         {}\n\
                         ----------------------------------------------------------------------\n\
                         Mean leaf model: {:.4} (n={}, std={:.4})\n\
                         ======================================================================",
                        title, mean, n_rows, leaf_model.stats.std
                    ));
                }
            }
        }
        Err(format!("Leaf ID {} not found", leaf_id))
    }

    /// Print comprehensive summary of all leaf models across the tree.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("======================================================================\n");
        out.push_str("                        mltreelib ModelTree Summary                   \n");
        out.push_str(&format!(
            "Depth: {}, Total Nodes: {}, Active Features: {}\n",
            self.params.max_depth,
            self.nodes.len(),
            self.active_cols.len()
        ));
        out.push_str("======================================================================\n\n");
        let entries = self.leaf_entries();
        for (_, _, leaf_id, _) in entries {
            if let Ok(leaf_str) = self.leaf_summary(leaf_id) {
                out.push_str(&leaf_str);
                out.push_str("\n\n");
            }
        }
        out
    }
}

pub(crate) struct Grower<'a> {
    pub(crate) cols: &'a [Vec<Option<f64>>],
    pub(crate) col_names: &'a [String],
    pub(crate) y: &'a [f64],
    pub(crate) w: &'a [f64],
    pub(crate) treatments: Option<&'a [f64]>,
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
    /// Tail bounds (lower, upper) for extrema protection.
    pub(crate) tail_bounds: Option<(f64, f64)>,
    pub(crate) max_depth_seen: usize,
}

enum GrowOp {
    Grow {
        depth: usize,
        rows: Vec<usize>,
        est: Vec<usize>,
        budget: usize,
        res_idx: usize,
        parent_mean: Option<f64>,
        parent_probs: Option<Vec<f64>>,
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
        node_mean: Option<f64>,
        node_probs: Option<Vec<f64>>,
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

    pub(crate) fn log(&mut self, level: u8, msg: &str) {
        if self.params.verbose >= level {
            println!("{}", msg);
            self.trace.push(msg.to_string());
        }
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
            parent_mean: None,
            parent_probs: None,
        }];

        while let Some(op) = stack.pop() {
            match op {
                GrowOp::Grow {
                    depth,
                    rows,
                    est,
                    budget,
                    res_idx,
                    parent_mean,
                    parent_probs,
                } => {
                    if depth > self.max_depth_seen {
                        self.max_depth_seen = depth;
                        let depth_msg = format!("[ModelTree] reached depth {}", depth);
                        self.log(1, &depth_msg);
                    }
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
                    // Purity gate (default off): a node whose target variation is at
                    // most `purity_gate` has no signal left — it stays a leaf.
                    if let Some(t) = self.params.purity_gate {
                        let mut is_extrema = false;
                        if let Some((y_low, y_high)) = self.tail_bounds {
                            for &r in &rows {
                                if self.y[r] <= y_low || self.y[r] >= y_high {
                                    is_extrema = true;
                                    break;
                                }
                            }
                        }
                        if !is_extrema {
                            let yws: Vec<(f64, f64)> =
                                rows.iter().map(|&r| (self.y[r], self.w[r])).collect();
                            let sd = crate::stats::w_sd(&yws);
                            let score = match self.params.purity_gate_metric {
                                PurityMetric::Sd => sd,
                                PurityMetric::Cv => {
                                    let mean = crate::stats::w_mean(&yws);
                                    if mean.abs() < 1e-12 {
                                        if sd < 1e-12 { 0.0 } else { f64::INFINITY }
                                    } else {
                                        sd / mean.abs()
                                    }
                                }
                            };
                            if score <= t {
                                best = None;
                            }
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
                            "[node depth={}] rows={} cols={} candidates={} scan_ms={} leaf_ms={}",
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
                        self.log(2, &line);
                    }

                    match best {
                        None => {
                            let t = Instant::now();
                            let mut leaf = self.make_leaf(self.leaf_rows(&rows, &est), &path);
                            if self.params.shrinkage > 0.0 {
                                if let Some(p_mean) = parent_mean {
                                    let n_samples = rows.len() as f64;
                                    let gamma = n_samples / (n_samples + self.params.shrinkage);
                                    if let Some(lm) = &mut leaf.linear {
                                        lm.intercept = p_mean + gamma * (lm.intercept - p_mean);
                                        for c in &mut lm.coef {
                                            *c *= gamma;
                                        }
                                        for c in &mut lm.na_coef {
                                            *c *= gamma;
                                        }
                                    }
                                    let cur_center = leaf.center.unwrap_or(leaf.stats.mean);
                                    leaf.center = Some(p_mean + gamma * (cur_center - p_mean));
                                    if let (Some(sm), Some(p_probs)) = (&mut leaf.softmax, parent_probs) {
                                        let n_c = sm.probs.len().min(p_probs.len());
                                        let mut sum_p = 0.0;
                                        for c in 0..n_c {
                                            sm.probs[c] = p_probs[c] + gamma * (sm.probs[c] - p_probs[c]);
                                            sum_p += sm.probs[c];
                                        }
                                        if sum_p > 0.0 {
                                            for c in 0..n_c { sm.probs[c] /= sum_p; }
                                        }
                                    }
                                }
                            }
                            prof.leaf_ns += t.elapsed().as_nanos() as u64;
                            let stats = FrameStats::from_ty(self.y, self.w, &rows);
                            let cost = if self.params.prune == PruneCost::None {
                                0.0
                            } else {
                                self.leaf_cost(&rows, &path, stats.sse)
                            };
                            let reason = if depth >= self.params.max_depth {
                                "max_depth"
                            } else if rows.len() < self.params.minsplit.unwrap_or(2 * self.params.min_rows) {
                                "min_rows"
                            } else if prof.gate.as_ref().map(|g| !g.passed).unwrap_or(false) {
                                "chow_gate"
                            } else {
                                "purity_or_no_gain"
                            };
                            if self.params.verbose >= 2 {
                                let leaf_msg = format!(
                                    "  [leaf] depth={} rows={} cost={:.4} reason={}",
                                    depth,
                                    rows.len(),
                                    cost,
                                    reason
                                );
                                self.log(2, &leaf_msg);
                            }
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
                            self.costs.push(cost);
                            results[res_idx] = (node_idx, 1);
                        }
                        Some((mut sp, lrows, rrows)) => {
                            if self.params.verbose >= 2 {
                                let col_name = self.col_names.get(sp.col).map(|s| s.as_str()).unwrap_or("?");
                                let op_str = if sp.cat.is_some() { "in" } else { "<=" };
                                let split_msg = format!(
                                    "  [split] col='{}' (idx={}) {} threshold={:.4} gain={:.6} left={} right={}",
                                    col_name,
                                    sp.col,
                                    op_str,
                                    sp.threshold,
                                    sp.gain,
                                    lrows.len(),
                                    rrows.len()
                                );
                                self.log(2, &split_msg);
                            }
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

                            let node_mean = if self.params.shrinkage > 0.0 {
                                let (mut swy, mut sw) = (0.0, 0.0);
                                for &r in &rows {
                                    swy += self.w[r] * self.y[r];
                                    sw += self.w[r];
                                }
                                if sw > 0.0 { Some(swy / sw) } else { None }
                            } else {
                                None
                            };
                            let node_probs = if self.params.shrinkage > 0.0 && self.params.leaf_family == LeafFamily::Softmax {
                                let n_c = self.params.n_classes.unwrap_or(2);
                                let mut counts = vec![0.0; n_c];
                                let mut sw = 0.0;
                                for &r in &rows {
                                    let c = self.y[r] as usize;
                                    if c < n_c {
                                        counts[c] += self.w[r];
                                        sw += self.w[r];
                                    }
                                }
                                if sw > 0.0 { for c in 0..n_c { counts[c] /= sw; } }
                                Some(counts)
                            } else {
                                None
                            };

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
                                node_mean,
                                node_probs: node_probs.clone(),
                            });
                            path.push(sp.col);
                            stack.push(GrowOp::Grow {
                                depth: depth + 1,
                                rows: lrows,
                                est: est_l,
                                budget: budget.saturating_sub(1),
                                res_idx: left_res_idx,
                                parent_mean: node_mean,
                                parent_probs: node_probs,
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
                    node_mean,
                    node_probs,
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
                        parent_mean: node_mean,
                        parent_probs: node_probs,
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

// ---------------------------------------------------------------- TreeSHAP helpers

#[derive(Clone, Debug)]
struct TreeShapPath {
    features: Vec<usize>,
    zero_fractions: Vec<f64>,
    one_fractions: Vec<f64>,
    weights: Vec<f64>,
}

impl TreeShapPath {
    fn new() -> Self {
        Self {
            features: Vec::new(),
            zero_fractions: Vec::new(),
            one_fractions: Vec::new(),
            weights: vec![1.0],
        }
    }

    fn extend(&mut self, feature: usize, zero_frac: f64, one_frac: f64) {
        let l = self.features.len();
        self.features.push(feature);
        self.zero_fractions.push(zero_frac);
        self.one_fractions.push(one_frac);

        let mut next_w = vec![0.0; l + 2];
        for i in 0..=l {
            let wi = self.weights[i];
            let p_o_step = (i + 1) as f64 / (l + 1) as f64;
            let p_z_step = (l + 1 - i) as f64 / (l + 1) as f64;
            next_w[i + 1] += one_frac * wi * p_o_step;
            next_w[i] += zero_frac * wi * p_z_step;
        }
        self.weights = next_w;
    }

    fn unwind_feature(&self, idx: usize) -> (Vec<f64>, f64, f64) {
        let n = self.features.len();
        let p_z = self.zero_fractions[idx];
        let p_o = self.one_fractions[idx];
        let mut unwound = vec![0.0; n];

        if p_o != 0.0 {
            unwound[n - 1] = self.weights[n] / p_o;
            for j in (0..n - 1).rev() {
                let t = self.weights[j + 1] - unwound[j + 1] * p_z * ((n - 1 - j) as f64 / n as f64);
                unwound[j] = t * (n as f64) / ((j + 1) as f64 * p_o);
            }
        } else if p_z != 0.0 {
            unwound[0] = self.weights[0] / p_z;
            for j in 1..n {
                let t = self.weights[j] - unwound[j - 1] * p_o * (j as f64 / n as f64);
                unwound[j] = t * (n as f64) / ((n - j) as f64 * p_z);
            }
        }
        (unwound, p_z, p_o)
    }
}

fn tree_shap_recurse(
    tree: &ModelTree,
    node_idx: usize,
    path: TreeShapPath,
    x: &[Option<f64>],
    phi: &mut [f64],
) {
    let node = &tree.nodes[node_idx];
    if let (Some(sp), Some(l), Some(r)) = (&node.split, node.left, node.right) {
        let left_child = &tree.nodes[l];
        let right_child = &tree.nodes[r];
        let n_parent = node.n as f64;
        let p_z_left = if n_parent > 0.0 { left_child.n as f64 / n_parent } else { 0.5 };
        let p_z_right = if n_parent > 0.0 { right_child.n as f64 / n_parent } else { 0.5 };

        let goes_left = match x.get(sp.col).and_then(|v| *v) {
            Some(v) => tree.goes_left(sp, v),
            None => tree.surrogate_left(x, sp),
        };
        let p_o_left = if goes_left { 1.0 } else { 0.0 };
        let p_o_right = if goes_left { 0.0 } else { 1.0 };

        if let Some(existing_idx) = path.features.iter().position(|&f| f == sp.col) {
            let (unwound_w, old_pz, old_po) = path.unwind_feature(existing_idx);
            let mut base_path = path.clone();
            base_path.features.remove(existing_idx);
            base_path.zero_fractions.remove(existing_idx);
            base_path.one_fractions.remove(existing_idx);
            base_path.weights = unwound_w;

            let new_po_left = old_po * p_o_left;
            let new_po_right = old_po * p_o_right;

            if p_z_left > 0.0 || new_po_left > 0.0 {
                let mut left_path = base_path.clone();
                left_path.extend(sp.col, old_pz * p_z_left, new_po_left);
                tree_shap_recurse(tree, l, left_path, x, phi);
            }
            if p_z_right > 0.0 || new_po_right > 0.0 {
                let mut right_path = base_path;
                right_path.extend(sp.col, old_pz * p_z_right, new_po_right);
                tree_shap_recurse(tree, r, right_path, x, phi);
            }
        } else {
            if p_z_left > 0.0 || p_o_left > 0.0 {
                let mut left_path = path.clone();
                left_path.extend(sp.col, p_z_left, p_o_left);
                tree_shap_recurse(tree, l, left_path, x, phi);
            }
            if p_z_right > 0.0 || p_o_right > 0.0 {
                let mut right_path = path;
                right_path.extend(sp.col, p_z_right, p_o_right);
                tree_shap_recurse(tree, r, right_path, x, phi);
            }
        }
    } else {
        let val = node.leaf.as_ref().map(|m| m.predict(x)).unwrap_or(node.stats.mean);
        let m = path.features.len() as f64;
        if m > 0.0 {
            for k in 0..path.features.len() {
                let feat = path.features[k];
                let (unwound_w, p_z, p_o) = path.unwind_feature(k);
                let sum_w: f64 = unwound_w.iter().sum();
                phi[feat] += (sum_w / m) * (p_o - p_z) * val;
            }
        }
    }
}


