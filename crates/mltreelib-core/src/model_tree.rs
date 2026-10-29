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

}

