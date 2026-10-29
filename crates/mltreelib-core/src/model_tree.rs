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
        }
    }
}

/// One binary split: values `<= threshold` go left.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub col: usize,
    pub threshold: f64,
    pub missing_left: bool,
    /// Goodness of split (parent SSE - children SSE) — the raw material of
    /// feature importance.
    pub gain: f64,
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
        };
        let root = grower.grow(&rows, 0, &mut Vec::new());
        // Destructure to release the borrows on params/active_cols before
        // moving them into the tree.
        let Grower {
            nodes, profiles, ..
        } = grower;
        Ok(Self {
            col_names,
            params,
            active_cols,
            nodes,
            root,
            profiles,
        })
    }

    /// Predict for row-major `x`. NAs never block a prediction.
    pub fn predict(&self, x: &[Vec<Option<f64>>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut node = &self.nodes[self.root];
                while let (Some(sp), Some(l), Some(r)) = (&node.split, node.left, node.right) {
                    let go_left = match row.get(sp.col).and_then(|v| *v) {
                        Some(v) => v <= sp.threshold,
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
                    format!("{} < {}", self.col_names[sp.col], fmt_cut(sp.threshold)),
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
                let cut = fmt_cut(sp.threshold);
                s += &format!("  n{} -> n{} [label=\"< {cut}\"];\n", num[i], num[l]);
                s += &format!("  n{} -> n{} [label=\">= {cut}\"];\n", num[i], num[r]);
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
}

impl<'a> Grower<'a> {
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

        match best {
            None => {
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
                })
            }
            Some((sp, lrows, rrows)) => {
                path.push(sp.col);
                let l = self.grow(&lrows, depth + 1, path);
                let r = self.grow(&rrows, depth + 1, path);
                path.pop();
                self.profiles.push(prof);
                self.add(Node {
                    split: Some(sp),
                    left: Some(l),
                    right: Some(r),
                    leaf: None,
                    n: rows.len(),
                    stats: FrameStats::from_ty(self.y, self.w, rows),
                })
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
        &self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        let m = self.params.min_rows;
        if rows.len() < 2 * m {
            return None;
        }

        let mut child_path = path.to_vec();
        let mut candidates: Vec<(usize, Option<(f64, Split)>, ScanStats)> = Vec::new();
        let work = rows.len() * self.cols.len();
        let cols: Vec<usize> = self
            .active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .collect();

        if work >= PAR_MIN_WORK {
            use rayon::prelude::*;
            let mut out: Vec<(usize, Option<(f64, Split)>, ScanStats)> = cols
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

        let mut best: Option<(f64, Split, usize, f64)> = None; // (gain, split, j, threshold)
        for (j, found, st) in candidates {
            prof.cols += 1;
            prof.group_ns += st.group_ns;
            prof.scan_ns += st.scan_ns;
            prof.candidates += st.candidates;
            if let Some((gain, sp)) = found {
                let better = match &best {
                    None => true,
                    Some((g, _, bj, bt)) => {
                        beats(gain, *g)
                            || (!beats(*g, gain) && (j, sp.threshold) < (*bj, *bt))
                    }
                };
                if better {
                    best = Some((gain, sp, j, sp.threshold));
                }
            }
        }

        best.map(|(_, s, ..)| {
            let (mut lv, mut rv) = (Vec::new(), Vec::new());
            for &r in rows {
                match self.cols[s.col][r] {
                    Some(v) => {
                        if v <= s.threshold {
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

    /// Dispatch one column's scan to the leaf-kind-specific loop.
    fn scan_column(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
    ) -> (Option<(f64, Split)>, ScanStats) {
        match self.params.leaf {
            LeafKind::Mean => self.scan_column_mean(rows, j),
            LeafKind::Linear => self.scan_column_linear(rows, j, path),
        }
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

        // Missing rows: one extra group, scored in both directions.
        let mut mn = 0usize;
        let (mut msw, mut mswy, mut mswy2) = (0.0, 0.0, 0.0);
        for &r in rows {
            if self.cols[j][r].is_none() {
                mn += 1;
                let (y, w) = (self.y[r], self.w[r]);
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
            let threshold = (v + pairs[k + 1].0) / 2.0;
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
                if !self.mono_ok(j, alswy / alsw.max(1e-300), arswy / arsw.max(1e-300)) {
                    continue;
                }
                let gain = parent_sse
                    - (sse(alsw, alswy, alswy2) + sse(arsw, arswy, arswy2));
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
        let mut right = total;
        let mut best: Option<(f64, Split)> = None;

        for gi in 0..groups.len() {
            let (v, g) = &groups[gi];
            left.add(g);
            right.sub(g);
            if gi + 1 >= groups.len() || left.n < m || right.n < m {
                continue;
            }
            let threshold = (v + groups[gi + 1].0) / 2.0;
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
        let stats = LeafStats::from_targets(&yv, &wv);
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
                let cut = fmt_cut(sp.threshold);
                walk(t, l, num * 2, depth + 1, &format!("{name}< {cut}"), out);
                walk(t, r, num * 2 + 1, depth + 1, &format!("{name}>={cut}"), out);
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
