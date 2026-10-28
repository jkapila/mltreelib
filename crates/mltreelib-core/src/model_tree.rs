//! Model tree: CART-style growth with least-squares (OLS) leaves.
//!
//! v1 of the "single tree that extrapolates" (see
//! `references/research/preselection-and-expert-thesis.md`, starting point A):
//! greedy splits by SSE reduction on the target, leaves fit ordinary least
//! squares. The leaf-column policy answers the thesis's open choice:
//! - [`LeafCols::All`] — regress on every usable column (model-tree classic).
//! - [`LeafCols::PathUsed`] — regress only on the columns used in splits along
//!   the root→leaf path (the tree's own variable selection; segmented /
//!   piecewise-linear flavor).
//!
//! Missing values are first-class ("NAs are part of the problem"):
//! - rows with a missing **target** are dropped at fit (no signal to learn);
//! - a missing **split**-column value is routed to whichever side improves
//!   the split gain most (learned default direction, XGBoost-style), and the
//!   choice is recorded in the [`Split`];
//! - missing **leaf** predictors are mean-imputed *and* carry an NA-indicator
//!   term in the leaf model, so missingness itself predicts;
//! - for **categorical** columns the caller keeps NA as its own level (the
//!   encoders' codes already do this), i.e. "NA as a category".
//! Predictions are always produced — a row is never lost to an NA.

use nalgebra::{DMatrix, DVector};

/// What the leaf model is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Linear model over the chosen columns (the model tree).
    Linear,
    /// Plain mean — the CART/rpart leaf. With this setting the split
    /// criterion (fitted-leaf SSE reduction) is exactly rpart's deviance
    /// criterion, making trees comparable to R's `rpart`.
    Mean,
}

/// Which columns a leaf's linear model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafCols {
    /// Every usable predictor column.
    All,
    /// Only the columns used in splits along the root→leaf path.
    PathUsed,
}

/// Growth and leaf parameters.
#[derive(Clone, Copy, Debug)]
pub struct ModelTreeParams {
    /// Maximum tree depth (0 = a single leaf: global OLS).
    pub max_depth: usize,
    /// Minimum rows per child side at a split (and per leaf).
    pub min_rows: usize,
    /// Minimum SSE reduction a split must achieve.
    pub min_gain: f64,
    /// Leaf column policy.
    pub leaf_cols: LeafCols,
    /// Leaf model kind (linear = model tree, mean = CART/rpart).
    pub leaf: LeafKind,
}

impl Default for ModelTreeParams {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_rows: 20,
            // Noise floor as much as a tuning knob: fitted-leaf gains below
            // this are numerical, not structural (exact-fit children would
            // otherwise "improve" by ~1e-15).
            min_gain: 1e-9,
            leaf_cols: LeafCols::All,
            leaf: LeafKind::Linear,
        }
    }
}

/// A fitted linear model: `intercept + Σ coef[j] * x[cols[j]] +
/// Σ na_coef[j] * 1{ x[cols[j]] missing }`. Missing predictors are imputed
/// with `fill[j]` and flagged by their indicator term — missingness is
/// treated as information ("NA as a value"), never as a lost row.
#[derive(Clone, Debug)]
pub struct LinearModel {
    /// Predictor column indices (into the fit-time column order).
    pub cols: Vec<usize>,
    /// Value coefficient per entry of `cols`.
    pub coef: Vec<f64>,
    /// NA-indicator coefficient per entry of `cols` (the "missing is a
    /// category" signal for numeric columns).
    pub na_coef: Vec<f64>,
    /// Imputation value (fit-time observed mean) per entry of `cols`.
    pub fill: Vec<f64>,
    /// Intercept term.
    pub intercept: f64,
}

impl LinearModel {
    /// Predict for one row (row-major `x`, one entry per column). Always
    /// returns a value: missing predictors are filled and flagged.
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        let mut y = self.intercept;
        for (k, &j) in self.cols.iter().enumerate() {
            match x[j] {
                Some(v) => y += self.coef[k] * v,
                None => y += self.coef[k] * self.fill[k] + self.na_coef[k],
            }
        }
        y
    }
}

/// One binary split: values `<= threshold` go left.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    /// Split column index.
    pub col: usize,
    /// Split point (midpoint between observed values).
    pub threshold: f64,
    /// Where rows missing the split column go (true = left).
    pub missing_left: bool,
}

#[derive(Clone, Debug)]
struct Node {
    split: Option<Split>,
    left: Option<usize>,
    right: Option<usize>,
    leaf: Option<LinearModel>,
    n: usize,
}

/// A fitted model tree.
#[derive(Clone, Debug)]
pub struct ModelTree {
    /// Column names in fit-time order.
    pub col_names: Vec<String>,
    /// The parameters used.
    pub params: ModelTreeParams,
    nodes: Vec<Node>,
    root: usize,
}

impl ModelTree {
    /// Fit on row-major data: `x` has one inner vec per row (one entry per
    /// column), `y` one entry per row. Column names default to `x0`, `x1`…
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

        // Transpose to column-major; drop rows with a missing target.
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
        };
        let root = grower.grow(&rows, 0, &mut Vec::new());
        Ok(Self {
            col_names,
            params,
            nodes: grower.nodes,
            root,
        })
    }

    /// Predict for row-major `x`; one value per row. NAs never block a
    /// prediction: split routing uses the learned default direction and leaf
    /// models fill + flag missing predictors.
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
                    .expect("tree invariant: leaf node holds a model")
                    .predict(row)
            })
            .collect()
    }

    /// All splits in the tree (for introspection / structure recovery).
    pub fn splits(&self) -> Vec<Split> {
        self.nodes
            .iter()
            .filter_map(|n| n.split)
            .collect()
    }

    /// Leaf models with their column names, coefficients, and row counts.
    pub fn leaves(&self) -> Vec<(&LinearModel, usize)> {
        self.nodes
            .iter()
            .filter_map(|n| n.leaf.as_ref().map(|m| (m, n.n)))
            .collect()
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.leaves().len()
    }
}

struct Grower<'a> {
    cols: &'a [Vec<Option<f64>>],
    y: &'a [f64],
    params: ModelTreeParams,
    nodes: Vec<Node>,
}

impl<'a> Grower<'a> {
    fn add(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn grow(&mut self, rows: &[usize], depth: usize, path: &mut Vec<usize>) -> usize {
        let best = if depth < self.params.max_depth {
            self.best_split(rows, path)
        } else {
            None
        };

        let Some((sp, lrows, rrows)) = best else {
            let leaf = self.fit_leaf(rows, path);
            return self.add(Node {
                split: None,
                left: None,
                right: None,
                leaf: Some(leaf),
                n: rows.len(),
            });
        };

        path.push(sp.col);
        let l = self.grow(&lrows, depth + 1, path);
        let r = self.grow(&rrows, depth + 1, path);
        path.pop();
        self.add(Node {
            split: Some(sp),
            left: Some(l),
            right: Some(r),
            leaf: None,
            n: rows.len(),
        })
    }

    /// Best split by **leaf-model SSE reduction** (the model-tree criterion:
    /// a split is worth it only if the fitted child models improve on the
    /// fitted parent model — variance reduction alone over-splits when leaves
    /// are linear). Honors min_rows/min_gain; missing values learn their
    /// default direction from the gain.
    fn best_split(&self, rows: &[usize], path: &[usize]) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        let d = self.cols.len();
        let m = self.params.min_rows;
        if rows.len() < 2 * m {
            return None;
        }
        let parent_sse = self.leaf_sse(rows, path);
        let mut child_path = path.to_vec();
        let mut best: Option<(f64, Split, Vec<usize>, Vec<usize>)> = None;

        for j in 0..d {
            let mut pairs: Vec<(f64, usize)> = rows
                .iter()
                .filter_map(|&r| self.cols[j][r].map(|v| (v, r)))
                .collect();
            let miss: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&r| self.cols[j][r].is_none())
                .collect();
            if pairs.len() < 2 {
                continue;
            }
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN in predictors"));
            child_path.push(j);
            let cp = child_path.as_slice();

            for k in 0..pairs.len() - 1 {
                if pairs[k + 1].0 == pairs[k].0 {
                    continue; // not a boundary
                }
                let threshold = (pairs[k].0 + pairs[k + 1].0) / 2.0;
                // Missing rows score both default directions; the gain picks.
                for missing_left in [true, false] {
                    let (mut lv, mut rv) = (
                        Vec::with_capacity(k + 1),
                        Vec::with_capacity(pairs.len() - k - 1 + miss.len()),
                    );
                    for &(pv, pr) in &pairs {
                        if pv <= threshold {
                            lv.push(pr);
                        } else {
                            rv.push(pr);
                        }
                    }
                    for &mr in &miss {
                        if missing_left {
                            lv.push(mr);
                        } else {
                            rv.push(mr);
                        }
                    }
                    if lv.len() < m || rv.len() < m {
                        continue;
                    }
                    let gain = parent_sse - (self.leaf_sse(&lv, cp) + self.leaf_sse(&rv, cp));
                    if gain <= self.params.min_gain {
                        continue;
                    }
                    // Deterministic first-found wins on ties. (rpart's own
                    // tie resolution is float-noise-dependent: same-partition
                    // splits tie bit-for-bit and summation order decides.
                    // Verified empirically against rpart 4.1.23 on mtcars —
                    // see tests/rpart_repro.rs for the tie documentation.)
                    if best.as_ref().map(|(g, ..)| gain > *g).unwrap_or(true) {
                        best = Some((
                            gain,
                            Split {
                                col: j,
                                threshold,
                                missing_left,
                            },
                            lv,
                            rv,
                        ));
                    }
                }
            }
            child_path.pop();
        }
        best.map(|(_, s, l, r)| (s, l, r))
    }

    /// Residual SSE of the leaf model that would be fit on these rows.
    fn leaf_sse(&self, rows: &[usize], path: &[usize]) -> f64 {
        let model = self.fit_leaf(rows, path);
        let d = self.cols.len();
        let mut row = vec![None; d];
        let mut sse = 0.0;
        for &r in rows {
            for j in 0..d {
                row[j] = self.cols[j][r];
            }
            let e = self.y[r] - model.predict(&row);
            sse += e * e;
        }
        sse
    }

    /// OLS over the chosen columns for this leaf's rows. Missing predictors
    /// are mean-imputed and flagged with NA-indicator terms ("NA as a value");
    /// every row with an observed target participates. Falls back to the row
    /// mean on rank deficiency or too few rows.
    fn fit_leaf(&self, rows: &[usize], path: &[usize]) -> LinearModel {
        if self.params.leaf == LeafKind::Mean {
            return mean_model(self.y, rows);
        }
        let d = self.cols.len();
        let chosen: Vec<usize> = match self.params.leaf_cols {
            LeafCols::All => (0..d).collect(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path.to_vec();
                p.sort_unstable();
                p.dedup();
                p
            }
        };

        // Per chosen column: observed mean (the imputation fill) and whether
        // it actually varies among observed rows (rank safety).
        let mut fills = Vec::with_capacity(chosen.len());
        let mut used: Vec<usize> = Vec::new();
        for &j in &chosen {
            let mut n_obs = 0usize;
            let mut sum = 0.0;
            let mut first: Option<f64> = None;
            let mut varies = false;
            for &r in rows {
                if let Some(v) = self.cols[j][r] {
                    n_obs += 1;
                    sum += v;
                    match first {
                        None => first = Some(v),
                        Some(f) if f != v => varies = true,
                        _ => {}
                    }
                }
            }
            let fill = if n_obs > 0 { sum / n_obs as f64 } else { 0.0 };
            fills.push(fill);
            // A column is usable when it varies and has at least one observed
            // value; an all-missing column contributes nothing (its indicator
            // would be constant).
            if varies && n_obs > 0 {
                used.push(j);
            }
        }

        let n = rows.len();
        if used.is_empty() || n < 1 + 2 * used.len() {
            return mean_model(self.y, rows);
        }
        // Indicator block only when a row actually misses a used column —
        // no NAs means no indicator terms (and no wasted parameters).
        let any_missing = rows
            .iter()
            .any(|&r| used.iter().any(|&j| self.cols[j][r].is_none()));
        let n_extra = if any_missing { used.len() } else { 0 };

        // Design: [1, imputed values..., NA indicators...].
        let p = 1 + used.len() + n_extra;
        let mut a_flat = vec![0.0; n * p];
        let mut y_flat = vec![0.0; n];
        for (i, &r) in rows.iter().enumerate() {
            a_flat[i * p] = 1.0;
            for (k, &j) in used.iter().enumerate() {
                let pos = chosen.iter().position(|&c| c == j).unwrap();
                match self.cols[j][r] {
                    Some(v) => a_flat[i * p + 1 + k] = v,
                    None => {
                        a_flat[i * p + 1 + k] = fills[pos];
                        if any_missing {
                            a_flat[i * p + 1 + used.len() + k] = 1.0;
                        }
                    }
                }
            }
            y_flat[i] = self.y[r];
        }
        let a = DMatrix::from_row_slice(n, p, &a_flat);
        let yv = DVector::from_vec(y_flat);
        // Least squares via truncated SVD (nalgebra's QR::solve is square-
        // only; SVD gives the minimum-norm solution and degrades gracefully
        // on collinear columns).
        match a.svd(true, true).solve(&yv, 1e-10) {
            Ok(sol) => {
                let nu = used.len();
                let used_fills: Vec<f64> = used
                    .iter()
                    .map(|&j| fills[chosen.iter().position(|&c| c == j).unwrap()])
                    .collect();
                LinearModel {
                    cols: used,
                    coef: sol.iter().skip(1).take(nu).copied().collect(),
                    na_coef: if any_missing {
                        sol.iter().skip(1 + nu).take(nu).copied().collect()
                    } else {
                        Vec::new()
                    },
                    fill: used_fills,
                    intercept: sol[0],
                }
            }
            Err(_) => mean_model(self.y, rows),
        }
    }
}

fn mean_model(y: &[f64], rows: &[usize]) -> LinearModel {
    let n = rows.len() as f64;
    let mean = rows.iter().map(|&r| y[r]).sum::<f64>() / n;
    LinearModel {
        cols: Vec::new(),
        coef: Vec::new(),
        na_coef: Vec::new(),
        fill: Vec::new(),
        intercept: mean,
    }
}
