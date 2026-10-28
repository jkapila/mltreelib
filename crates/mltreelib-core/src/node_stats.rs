//! Node/leaf statistics and models — standalone functions and value types.
//! `model_tree.rs` routes rows and scans split candidates; everything about
//! what a node *stores* (moments, t-digest, linear model) and the small
//! numerics of split scoring live here, one concern per function.

use tdigest::TDigest;

/// Target summary stored at every leaf — mean nodes and model nodes alike.
/// `std` is the sample standard deviation (ddof = 1), kept because quantile
/// work downstream needs the spread, not just the mean. The t-digest (Cam
/// Davidson-Pilon semantics via the `tdigest` crate) keeps the node's
/// quantiles queryable without storing every target.
#[derive(Clone, Debug)]
pub struct LeafStats {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub digest: TDigest,
}

impl LeafStats {
    /// Moments + t-digest of one leaf's target values.
    pub fn from_targets(y: &[f64]) -> Self {
        let n = y.len();
        assert!(n > 0, "leaf must have rows");
        let mean = y.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let mut digest = TDigest::new_with_size(100);
        digest.extend_values(y.iter().copied());
        digest.flush();
        Self {
            n,
            mean,
            std: var.sqrt(),
            min: y.iter().copied().fold(f64::INFINITY, f64::min),
            max: y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            digest,
        }
    }

    /// Quantile estimate from the leaf's t-digest (q in 0..=1).
    pub fn quantile(&self, q: f64) -> Option<f64> {
        self.digest.estimate_quantile(q)
    }
}

/// Fitted linear model: `intercept + Σ coef·x + Σ na_coef·1{missing}`.
/// Missing predictors contribute nothing to the value terms and carry their
/// own indicator term — missingness is an effect, never a lost row.
#[derive(Clone, Debug)]
pub struct LinearModel {
    pub cols: Vec<usize>,
    pub coef: Vec<f64>,
    pub na_coef: Vec<f64>,
    pub intercept: f64,
}

impl LinearModel {
    /// Predict for one row (`None` = missing). Always returns a value; a row
    /// shorter than the fit-time width treats absent cells as missing (an NA
    /// never blocks a prediction, and never panics).
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        let mut y = self.intercept;
        for (k, &j) in self.cols.iter().enumerate() {
            match x.get(j).copied().flatten() {
                Some(v) => y += self.coef[k] * v,
                // A leaf fitted on fully-observed rows has no indicator
                // terms; missingness then contributes its honest estimate,
                // zero (nothing learned), never a panic.
                None => y += self.na_coef.get(k).copied().unwrap_or(0.0),
            }
        }
        y
    }
}

/// Leaf payload: distribution stats always; a linear model for model-tree
/// leaves (`None` for mean/CART leaves).
#[derive(Clone, Debug)]
pub struct LeafModel {
    pub stats: LeafStats,
    pub linear: Option<LinearModel>,
}

impl LeafModel {
    /// Linear model when present, the leaf mean otherwise.
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        self.linear
            .as_ref()
            .map(|m| m.predict(x))
            .unwrap_or(self.stats.mean)
    }
}

/// Sufficient statistics of a least-squares design: `xtx` (dim×dim row
/// major), `xty`, `yty`, `n`. Additive per row or per value group — the
/// split scan keeps running left/right aggregates and scores candidates
/// without ever refitting (the rpart/LightGBM loop shape).
#[derive(Clone, Debug)]
pub struct Gram {
    pub xtx: Vec<f64>,
    pub xty: Vec<f64>,
    pub yty: f64,
    pub n: usize,
    dim: usize,
}

impl Gram {
    pub fn new(dim: usize) -> Self {
        Self {
            xtx: vec![0.0; dim * dim],
            xty: vec![0.0; dim],
            yty: 0.0,
            n: 0,
            dim,
        }
    }

    /// Rank-1 update: add one design row with target `y`. Allocation-free.
    pub fn add_row(&mut self, x: &[f64], y: f64) {
        let d = self.dim;
        for i in 0..d {
            let xi = x[i];
            if xi != 0.0 {
                self.xty[i] += xi * y;
                let row = i * d;
                for j in 0..d {
                    self.xtx[row + j] += xi * x[j];
                }
            }
        }
        self.yty += y * y;
        self.n += 1;
    }

    /// Add a whole aggregate (one value group).
    pub fn add(&mut self, other: &Gram) {
        for (a, b) in self.xtx.iter_mut().zip(&other.xtx) {
            *a += b;
        }
        for (a, b) in self.xty.iter_mut().zip(&other.xty) {
            *a += b;
        }
        self.yty += other.yty;
        self.n += other.n;
    }

    /// Subtract a whole aggregate (scan step the other way).
    pub fn sub(&mut self, other: &Gram) {
        for (a, b) in self.xtx.iter_mut().zip(&other.xtx) {
            *a -= b;
        }
        for (a, b) in self.xty.iter_mut().zip(&other.xty) {
            *a -= b;
        }
        self.yty -= other.yty;
        self.n -= other.n;
    }
}

/// Working memory for solving the normal equations — allocated once per
/// node, reused for every candidate (the "preallocate the matrix" lesson
/// from LightGBM's `linear_tree_learner.cpp`).
pub struct SolveScratch {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl SolveScratch {
    pub fn new(dim: usize) -> Self {
        Self {
            a: vec![0.0; dim * dim],
            b: vec![0.0; dim],
            c: vec![0.0; dim],
        }
    }
}

/// Residual SSE of `xtx·b = xty` (least squares over the design), solved by
/// Gaussian elimination with partial pivoting into `scratch`. `l2` adds ridge
/// penalty to every non-intercept diagonal entry (LightGBM's `linear_lambda`
/// shape) — the ridge objective keeps the `yty - b·xty` identity, so this
/// stays exact for the penalized fit. Rank-deficient directions are dropped
/// (coefficient 0). The solution is left in `scratch.b`.
pub fn solve_normal(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
    scratch: &mut SolveScratch,
) -> f64 {
    scratch.a[..dim * dim].copy_from_slice(&xtx[..dim * dim]);
    scratch.b[..dim].copy_from_slice(&xty[..dim]);
    solve_prepared(scratch, yty, dim, l1, l2)
}

/// Solve the normal equations already copied into `scratch.a`/`scratch.b`.
/// `l1`/`l2` are the elastic-net penalties (ridge = l2 only, lasso = l1
/// only, OLS = both 0); the intercept is unpenalized. At convergence the
/// penalized objective still equals `yty - b·xty`, so the SSE stays exact.
pub fn solve_prepared(
    scratch: &mut SolveScratch,
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
) -> f64 {
    scratch.c[..dim].copy_from_slice(&scratch.b[..dim]); // keep xty
    if l1 > 0.0 {
        return solve_elastic(scratch, yty, dim, l1, l2);
    }
    if l2 > 0.0 {
        for i in 1..dim {
            scratch.a[i * dim + i] += l2; // ridge, intercept unpenalized
        }
    }
    let scale = (0..dim)
        .map(|i| scratch.a[i * dim + i].abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let eps = 1e-10 * scale;

    for col in 0..dim {
        // partial pivot
        let mut piv_row = col;
        let mut piv_val = scratch.a[col * dim + col].abs();
        for r in col + 1..dim {
            let v = scratch.a[r * dim + col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = r;
            }
        }
        if piv_val <= eps {
            // rank-deficient direction: free variable, coefficient 0
            scratch.b[col] = 0.0;
            for r in 0..dim {
                scratch.a[r * dim + col] = 0.0;
            }
            continue;
        }
        if piv_row != col {
            for c in 0..dim {
                scratch.a.swap(col * dim + c, piv_row * dim + c);
            }
            scratch.b.swap(col, piv_row);
        }
        let diag = scratch.a[col * dim + col];
        for r in col + 1..dim {
            let f = scratch.a[r * dim + col] / diag;
            if f != 0.0 {
                for c in col..dim {
                    scratch.a[r * dim + c] -= f * scratch.a[col * dim + c];
                }
                scratch.b[r] -= f * scratch.b[col];
            }
        }
    }

    // back substitution (dropped columns keep coefficient 0)
    for col in (0..dim).rev() {
        let mut s = scratch.b[col];
        for c in col + 1..dim {
            s -= scratch.a[col * dim + c] * scratch.b[c];
        }
        let d = scratch.a[col * dim + col];
        scratch.b[col] = if d.abs() <= eps { 0.0 } else { s / d };
    }

    let explained: f64 = scratch.c[..dim]
        .iter()
        .zip(&scratch.b[..dim])
        .map(|(t, s)| t * s)
        .sum();
    (yty - explained).max(0.0)
}

/// Extract the fitted `LinearModel` from a design Gram. Layout: `[1, values
/// (0 where missing), NA indicators (only when `with_indicators`)]`. `l1`/`l2`
/// are the elastic-net penalties (intercept unpenalized).
pub fn fit_linear(
    g: &Gram,
    chosen: &[usize],
    with_indicators: bool,
    l1: f64,
    l2: f64,
) -> LinearModel {
    let nc = chosen.len();
    let dim = 1 + nc + if with_indicators { nc } else { 0 };
    let mut scratch = SolveScratch::new(dim);
    solve_normal(&g.xtx, &g.xty, g.yty, dim, l1, l2, &mut scratch);
    let sol = scratch.b.clone();
    LinearModel {
        cols: chosen.to_vec(),
        coef: sol[1..1 + nc].to_vec(),
        na_coef: if with_indicators {
            sol[1 + nc..1 + 2 * nc].to_vec()
        } else {
            Vec::new()
        },
        intercept: sol[0],
    }
}

/// One design row from column-major storage: `[1, values (0 where missing),
/// NA indicators]`. Allocation-free; the indicator block is included only
/// when `with_indicators`.
pub fn design_row(
    out: &mut [f64],
    cols: &[Vec<Option<f64>>],
    r: usize,
    chosen: &[usize],
    with_indicators: bool,
) {
    let nc = chosen.len();
    for slot in out.iter_mut() {
        *slot = 0.0;
    }
    out[0] = 1.0;
    for (k, &c) in chosen.iter().enumerate() {
        match cols[c][r] {
            Some(v) => out[1 + k] = v,
            None => {
                if with_indicators {
                    out[1 + nc + k] = 1.0;
                }
            }
        }
    }
}

/// Residual SSE of `(a + b)`'s normal equations without materializing the
/// sum: fills the shared scratch from `a` (+ `b`), then solves in place.
pub fn solve_normal_sum(
    a: &Gram,
    b: Option<&Gram>,
    dim: usize,
    l1: f64,
    l2: f64,
    scratch: &mut SolveScratch,
) -> f64 {
    scratch.a[..dim * dim].copy_from_slice(&a.xtx[..dim * dim]);
    scratch.b[..dim].copy_from_slice(&a.xty[..dim]);
    let mut yty = a.yty;
    if let Some(e) = b {
        for (dst, src) in scratch.a[..dim * dim].iter_mut().zip(&e.xtx[..dim * dim]) {
            *dst += *src;
        }
        for (dst, src) in scratch.b[..dim].iter_mut().zip(&e.xty[..dim]) {
            *dst += *src;
        }
        yty += e.yty;
    }
    solve_prepared(scratch, yty, dim, l1, l2)
}

/// Elastic net by coordinate descent on the standardised Gram (glmnet's
/// approach): features standardised to unit scale so the penalty is fair
/// across columns, soft-thresholding for L1, coefficients unscaled on the
/// way out. The intercept is never penalized.
fn solve_elastic(
    scratch: &mut SolveScratch,
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
) -> f64 {
    let n = scratch.a[0].max(1.0);
    let sum_y = scratch.c[0];
    // Column means and scales from the Gram (column 0 is all ones).
    let mut mean = vec![0.0; dim];
    let mut scale = vec![1.0; dim];
    for j in 1..dim {
        mean[j] = scratch.a[j] / n; // xtx[0][j] = sum of column j
    }
    for j in 1..dim {
        let var = scratch.a[j * dim + j] - n * mean[j] * mean[j];
        scale[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }
    // Standardised Gram and right-hand side.
    let mut g = vec![0.0; (dim - 1) * (dim - 1)];
    let mut r = vec![0.0; dim - 1];
    for j in 1..dim {
        r[j - 1] = (scratch.c[j] - mean[j] * sum_y) / scale[j];
        for k in 1..dim {
            let gc = scratch.a[j * dim + k] - n * mean[j] * mean[k];
            g[(j - 1) * (dim - 1) + (k - 1)] = gc / (scale[j] * scale[k]);
        }
    }
    // Coordinate descent.
    let mut b = vec![0.0; dim - 1];
    for _ in 0..200 {
        let mut delta = 0.0f64;
        for j in 0..dim - 1 {
            let gjj = g[j * (dim - 1) + j];
            if gjj <= 0.0 {
                continue;
            }
            let mut partial = r[j];
            for k in 0..dim - 1 {
                if k != j {
                    partial -= g[j * (dim - 1) + k] * b[k];
                }
            }
            let raw = partial / (gjj + l2);
            let shrunk = raw.signum() * (raw.abs() - l1 / (gjj + l2)).max(0.0);
            delta = delta.max((shrunk - b[j]).abs());
            b[j] = shrunk;
        }
        if delta < 1e-10 {
            break;
        }
    }
    // Unscale and rebuild the intercept; explained = b^T xty (exact at CD
    // convergence for the elastic-net objective).
    scratch.b[0] = sum_y / n;
    for j in 1..dim {
        let bj = b[j - 1] / scale[j];
        scratch.b[j] = bj;
        scratch.b[0] -= mean[j] * bj;
    }
    let explained: f64 = scratch.c[..dim]
        .iter()
        .zip(&scratch.b[..dim])
        .map(|(t, s)| t * s)
        .sum();
    (yty - explained).max(0.0)
}
