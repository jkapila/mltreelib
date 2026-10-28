//! Model tree: growth and routing only. Node models and numerics live in
//! [`crate::node_stats`]; every function here does one job.
//!
//! Split scanning follows the reference loop shapes instead of inventing
//! new ones:
//! - mean leaves: rpart's running-sum scan (`rpart/src/anova.c`) — O(1) per
//!   value, split only where the value changes;
//! - linear leaves: LightGBM's grouped accumulators
//!   (`src/treelearner/feature_histogram.cpp`) over per-value Gram groups,
//!   with per-candidate solves on preallocated matrices
//!   (`src/treelearner/linear_tree_learner.cpp`: "preallocate the matrix").
//! Missing values are scored in both default directions (LightGBM's
//! `default_left` shape) and the gain picks.

use std::time::Instant;

use crate::node_stats::{
    design_row, fit_linear, solve_normal_sum, Gram, LeafModel, LeafStats, SolveScratch,
};

pub use crate::node_stats::{LeafModel as NodeModel, LeafStats as NodeStats, LinearModel};

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
#[derive(Clone, Copy, Debug)]
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
}

impl Default for ModelTreeParams {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_rows: 20,
            min_gain: 1e-9,
            leaf_cols: LeafCols::All,
            leaf: LeafKind::Linear,
        }
    }
}

/// One binary split: values `<= threshold` go left.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub col: usize,
    pub threshold: f64,
    pub missing_left: bool,
}

#[derive(Clone, Debug)]
struct Node {
    split: Option<Split>,
    left: Option<usize>,
    right: Option<usize>,
    leaf: Option<LeafModel>,
    n: usize,
}

/// Per-node timing breakdown — the debug-on-every-level instrumentation.
/// `group_ns` = extract/sort/value-group aggregation, `scan_ns` = the
/// candidate scan, `leaf_ns` = fitting the leaf model.
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
    nodes: Vec<Node>,
    root: usize,
    profiles: Vec<NodeProfile>,
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
        let n_all = y.len();
        if x.len() != n_all {
            return Err("x and y length mismatch".to_string());
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

        // Column-major; rows with a missing target drop (no signal to learn).
        let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(n_all); d];
        let mut yv: Vec<f64> = Vec::with_capacity(n_all);
        for (r, row) in x.iter().enumerate() {
            let Some(t) = y[r] else { continue };
            for j in 0..d {
                cols[j].push(row[j]);
            }
            yv.push(t);
        }
        if yv.is_empty() {
            return Err("no rows with an observed target".to_string());
        }

        let rows: Vec<usize> = (0..yv.len()).collect();
        let mut grower = Grower {
            cols: &cols,
            y: &yv,
            params,
            nodes: Vec::new(),
            profiles: Vec::new(),
        };
        let root = grower.grow(&rows, 0, &mut Vec::new());
        Ok(Self {
            col_names,
            params,
            nodes: grower.nodes,
            root,
            profiles: grower.profiles,
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
    params: ModelTreeParams,
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
                })
            }
        }
    }

    /// Columns a leaf model uses for a given root→leaf path.
    fn chosen_cols(&self, path: &[usize]) -> Vec<usize> {
        match self.params.leaf_cols {
            LeafCols::All => (0..self.cols.len()).collect(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path.to_vec();
                p.sort_unstable();
                p.dedup();
                p
            }
        }
    }

    /// Best split across columns; candidates are scored from running
    /// aggregates, row sets are built only for the winner.
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

        let mut best: Option<(f64, Split)> = None;
        let mut child_path = path.to_vec();
        for j in 0..self.cols.len() {
            child_path.push(j);
            let found = match self.params.leaf {
                LeafKind::Mean => self.scan_column_mean(rows, j, prof),
                LeafKind::Linear => self.scan_column_linear(rows, j, &child_path, prof),
            };
            child_path.pop();
            if let Some((gain, sp)) = found {
                // Deterministic first-found wins on near-ties (see `beats`).
                if best.as_ref().map(|(g, ..)| beats(gain, *g)).unwrap_or(true) {
                    best = Some((gain, sp));
                }
            }
        }

        best.map(|(gain, s)| {
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
            let _ = gain;
            (s, lv, rv)
        })
    }

    /// Mean-leaf scan — rpart's `anova.c` shape: running sums per side, O(1)
    /// per value, split only where the value changes.
    fn scan_column_mean(
        &self,
        rows: &[usize],
        j: usize,
        prof: &mut NodeProfile,
    ) -> Option<(f64, Split)> {
        prof.cols += 1;
        let m = self.params.min_rows as f64;
        let t0 = Instant::now();
        let mut pairs: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|&r| self.cols[j][r].map(|v| (v, self.y[r])))
            .collect();
        if pairs.len() < 2 {
            return None;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));

        // Missing rows: one extra group, scored in both directions.
        let mut mn = 0usize;
        let (mut ms, mut mss) = (0.0, 0.0);
        for &r in rows {
            if self.cols[j][r].is_none() {
                mn += 1;
                ms += self.y[r];
                mss += self.y[r] * self.y[r];
            }
        }
        let mut tn = 0usize;
        let (mut ts, mut tss) = (0.0, 0.0);
        for &(_, yv) in &pairs {
            tn += 1;
            ts += yv;
            tss += yv * yv;
        }

        fn sse(n: f64, s: f64, ss: f64) -> f64 {
            if n == 0.0 {
                0.0
            } else {
                ss - s * s / n
            }
        }
        let parent_sse = sse(tn as f64, ts, tss) + sse(mn as f64, ms, mss);

        prof.group_ns += t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();

        let mut ln = 0usize;
        let (mut ls, mut lss) = (0.0, 0.0);
        let mut best: Option<(f64, Split)> = None;
        for k in 0..pairs.len() {
            let (v, yv) = pairs[k];
            ln += 1;
            ls += yv;
            lss += yv * yv;
            let is_boundary = k + 1 < pairs.len() && pairs[k + 1].0 != v;
            if !is_boundary {
                continue;
            }
            let threshold = (v + pairs[k + 1].0) / 2.0;
            let rn = tn - ln;
            let (rs, rss) = (ts - ls, tss - lss);
            prof.candidates += 1;
            for missing_left in [true, false] {
                let (aln, als, alss) = if missing_left {
                    (ln + mn, ls + ms, lss + mss)
                } else {
                    (ln, ls, lss)
                };
                let (arn, ars, arss) = if missing_left {
                    (rn, rs, rss)
                } else {
                    (rn + mn, rs + ms, rss + mss)
                };
                if (aln as f64) < m || (arn as f64) < m {
                    continue;
                }
                let gain = parent_sse - (sse(aln as f64, als, alss) + sse(arn as f64, ars, arss));
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
                        },
                    ));
                }
            }
        }
        prof.scan_ns += t1.elapsed().as_nanos() as u64;
        best
    }

    /// Linear-leaf scan — LightGBM's histogram shape: one Gram per distinct
    /// value, running left/right aggregates, per-candidate solve on
    /// preallocated scratch (no allocation in the loop).
    fn scan_column_linear(
        &self,
        rows: &[usize],
        j: usize,
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(f64, Split)> {
        prof.cols += 1;
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
            return None;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));

        // Value groups: one Gram per distinct value ("loop over each element
        // within the variable", not over each row).
        let mut groups: Vec<(f64, Gram)> = Vec::new();
        for &(v, r) in &pairs {
            design_row(&mut row_buf, self.cols, r, &chosen, true);
            match groups.last_mut() {
                Some((gv, g)) if *gv == v => g.add_row(&row_buf, self.y[r]),
                _ => {
                    let mut g = Gram::new(dim);
                    g.add_row(&row_buf, self.y[r]);
                    groups.push((v, g));
                }
            }
        }
        let mut miss = Gram::new(dim);
        for &r in rows {
            if self.cols[j][r].is_none() {
                design_row(&mut row_buf, self.cols, r, &chosen, true);
                miss.add_row(&row_buf, self.y[r]);
            }
        }

        // Parent SSE over the parent's design class (once per node).
        let parent_chosen = self.chosen_cols(path);
        let pdim = 1 + 2 * parent_chosen.len();
        let mut pgram = Gram::new(pdim);
        let mut prow = vec![0.0; pdim];
        for &r in rows {
            design_row(&mut prow, self.cols, r, &parent_chosen, true);
            pgram.add_row(&prow, self.y[r]);
        }
        let mut scratch = SolveScratch::new(dim.max(pdim));
        let parent_sse = solve_normal_sum(&pgram, None, pdim, &mut scratch);

        prof.group_ns += t0.elapsed().as_nanos() as u64;
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
            prof.candidates += 1;
            for missing_left in [true, false] {
                let left_n = left.n + if missing_left { miss.n } else { 0 };
                let right_n = right.n + if missing_left { 0 } else { miss.n };
                if left_n < m || right_n < m {
                    continue;
                }
                // sse(left + miss) + sse(right) or the mirror — two solves,
                // no materialized sums, no clones.
                let (sl, sr) = if missing_left {
                    (
                        solve_normal_sum(&left, Some(&miss), dim, &mut scratch),
                        solve_normal_sum(&right, None, dim, &mut scratch),
                    )
                } else {
                    (
                        solve_normal_sum(&left, None, dim, &mut scratch),
                        solve_normal_sum(&right, Some(&miss), dim, &mut scratch),
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
                        },
                    ));
                }
            }
        }
        prof.scan_ns += t1.elapsed().as_nanos() as u64;
        best
    }

    /// Fit the leaf model for a node's rows.
    fn make_leaf(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        let yv: Vec<f64> = rows.iter().map(|&r| self.y[r]).collect();
        let stats = LeafStats::from_targets(&yv);
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
                g.add_row(&row, self.y[r]);
            }
            Some(fit_linear(&g, &chosen, any_missing))
        } else {
            None
        };
        LeafModel { stats, linear }
    }
}
