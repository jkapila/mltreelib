//! Node/leaf statistics and models — standalone functions and value types.
//! `model_tree.rs` routes rows and scans split candidates; everything about
//! what a node *stores* (moments, t-digest, linear model) and the small
//! numerics of split scoring live here, one concern per function.
//!
//! Everything is **row-weighted**: sample weights flow through the Gram
//! statistics, the leaf moments, and the gains, so boosting/forest
//! reweighting and case weights work without touching the tree.

use tdigest::TDigest;
use crate::family::LeafFamily;

/// Target summary stored at every leaf — mean nodes and model nodes alike.
/// Weighted by sample weights (`w`); `std` is the weighted sample standard
/// deviation. The t-digest (Cam Davidson-Pilon semantics via the `tdigest`
/// crate) keeps quantiles queryable without storing every target.
#[derive(Clone, Debug)]
pub struct LeafStats {
    pub n: usize,
    /// Sum of the sample weights.
    pub wsum: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub digest: TDigest,
}

impl LeafStats {
    /// Weighted moments + t-digest of one leaf's targets. Rows with weight 0
    /// are ignored entirely (a zero-weight row is an absent row).
    pub fn from_targets(y: &[f64], w: &[f64]) -> Self {
        let mut idx: Vec<usize> = (0..y.len()).filter(|&i| w[i] > 0.0).collect();
        assert!(!idx.is_empty(), "leaf must have positive-weight rows");
        let wsum: f64 = idx.iter().map(|&i| w[i]).sum();
        let mean = idx.iter().map(|&i| w[i] * y[i]).sum::<f64>() / wsum;
        let var = idx
            .iter()
            .map(|&i| w[i] * (y[i] - mean) * (y[i] - mean))
            .sum::<f64>()
            / wsum;
        idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).expect("NaN in target"));
        let mut digest = TDigest::new_with_size(100);
        digest.extend_values(idx.iter().map(|&i| y[i]));
        digest.flush();
        Self {
            n: idx.len(),
            wsum,
            mean,
            std: var.sqrt(),
            min: y[idx[0]],
            max: y[*idx.last().unwrap()],
            digest,
        }
    }

    /// Quantile estimate from the leaf's t-digest (q in 0..=1).
    pub fn quantile(&self, q: f64) -> Option<f64> {
        self.digest.estimate_quantile(q)
    }

    /// Moments for split-candidate fits, without sorting targets or building
    /// a digest. The stored empty digest is never exposed as a final leaf.
    pub(crate) fn from_rows(y: &[f64], w: &[f64], rows: &[usize]) -> Self {
        let frame = FrameStats::from_ty(y, w, rows);
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &r in rows {
            if w[r] > 0.0 {
                min = min.min(y[r]);
                max = max.max(y[r]);
            }
        }
        Self {
            n: frame.n,
            wsum: frame.wsum,
            mean: frame.mean,
            std: (frame.sse / frame.wsum).sqrt(),
            min,
            max,
            digest: TDigest::new_with_size(1),
        }
    }
}

/// Fitted link-scale linear predictor:
/// `intercept + Σ coef·x + Σ na_coef·1{missing}`. Gaussian uses the identity
/// link. Missing predictors contribute nothing to value terms and have their
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
    pub family: LeafFamily,
    pub stats: LeafStats,
    pub linear: Option<LinearModel>,
    /// Fitted response-scale constant (robust Gaussian locations or exact
    /// boundary family means). `None` = the weighted response mean.
    pub center: Option<f64>,
}

impl LeafModel {
    /// Linear model when present, the fitted constant otherwise (weighted
    /// mean for `LeafKind::Mean`).
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        match &self.linear {
            Some(model) => self.family.inverse_link(model.predict(x)),
            None => self.center.unwrap_or(self.stats.mean),
        }
    }

    /// Column-major prediction for training-row scoring, without allocating
    /// or copying a row for every candidate leaf.
    pub(crate) fn predict_columns(&self, cols: &[Vec<Option<f64>>], r: usize) -> f64 {
        let Some(model) = &self.linear else {
            return self.center.unwrap_or(self.stats.mean);
        };
        let mut eta = model.intercept;
        for (k, &c) in model.cols.iter().enumerate() {
            match cols[c][r] {
                Some(value) => eta += model.coef[k] * value,
                None => eta += model.na_coef.get(k).copied().unwrap_or(0.0),
            }
        }
        self.family.inverse_link(eta)
    }
}

/// Weighted sufficient statistics of a least-squares design: `xtx`
/// (dim×dim row major), `xty`, `yty`, `n` (rows), `ws` (weight sum).
/// Additive per row or per value group — the split scan keeps running
/// left/right aggregates and scores candidates without refitting.
#[derive(Clone, Debug)]
pub struct Gram {
    pub xtx: Vec<f64>,
    pub xty: Vec<f64>,
    pub yty: f64,
    pub n: usize,
    pub ws: f64,
    /// Σ w·y (available as xty[0] too; kept for cheap leaf means).
    pub wy: f64,
    dim: usize,
}

impl Gram {
    pub fn new(dim: usize) -> Self {
        Self {
            xtx: vec![0.0; dim * dim],
            xty: vec![0.0; dim],
            yty: 0.0,
            n: 0,
            ws: 0.0,
            wy: 0.0,
            dim,
        }
    }

    /// Rank-1 update: add one design row with target `y` and weight `w`.
    /// Allocation-free.
    pub fn add_row(&mut self, x: &[f64], y: f64, w: f64) {
        let d = self.dim;
        for i in 0..d {
            let xi = x[i];
            if xi != 0.0 {
                self.xty[i] += w * xi * y;
                let row = i * d;
                for j in 0..d {
                    self.xtx[row + j] += w * xi * x[j];
                }
            }
        }
        self.yty += w * y * y;
        self.wy += w * y;
        self.ws += w;
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
        self.wy += other.wy;
        self.ws += other.ws;
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
        self.wy -= other.wy;
        self.ws -= other.ws;
        self.n -= other.n;
    }

    /// Design dimension of this aggregate (intercept included).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Weighted mean of the target over this aggregate.
    pub fn mean(&self) -> f64 {
        if self.ws > 0.0 {
            self.wy / self.ws
        } else {
            0.0
        }
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
/// Gaussian elimination with partial pivoting into `scratch`. `l1`/`l2` are
/// the elastic-net penalties (ridge = l2 only, lasso = l1 only, OLS = both
/// 0); the intercept is unpenalized. At convergence the penalized objective
/// still equals `yty - b·xty`, so the SSE stays exact. Rank-deficient
/// directions are dropped (coefficient 0). Solution lands in `scratch.b`.
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
    solve_prepared(scratch, yty, dim, l1, l2, None)
}

/// Dense SPD solve via faer's blocked Cholesky — the split scan's hot
/// path (thousands of small/medium Gram solves per fit). Returns `None`
/// for rank-deficient Grams so the caller falls back to the rank-revealing
/// elimination (singular directions keep coefficient 0).
fn faer_cholesky_solve(a: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    use faer::prelude::*;
    let mut m = faer::Mat::<f64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            m[(i, j)] = a[i * dim + j];
        }
    }
    let llt = m.as_ref().llt(faer::Side::Lower).ok()?;
    let mut rhs = faer::Mat::<f64>::zeros(dim, 1);
    for i in 0..dim {
        rhs[(i, 0)] = b[i];
    }
    let x = llt.solve(rhs.as_ref());
    Some((0..dim).map(|i| x[(i, 0)]).collect())
}

/// Solve the normal equations already copied into `scratch.a`/`scratch.b`.
pub fn solve_prepared(
    scratch: &mut SolveScratch,
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
    warm: Option<&mut [f64]>,
) -> f64 {
    scratch.c[..dim].copy_from_slice(&scratch.b[..dim]); // keep xty
    if l1 > 0.0 {
        return solve_elastic(scratch, yty, dim, l1, l2, warm);
    }
    if l2 > 0.0 {
        for i in 1..dim {
            scratch.a[i * dim + i] += l2; // ridge, intercept unpenalized
        }
    }
    // Fast path: faer Cholesky on the SPD Gram (the common case —
    // thousands of solves per split scan). Falls through to the
    // rank-revealing elimination below when the Gram is singular.
    if let Some(x) = faer_cholesky_solve(&scratch.a[..dim * dim], &scratch.b[..dim], dim) {
        scratch.b[..dim].copy_from_slice(&x);
        let explained: f64 = scratch.c[..dim]
            .iter()
            .zip(&scratch.b[..dim])
            .map(|(t, s)| t * s)
            .sum();
        return (yty - explained).max(0.0);
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

/// Residual SSE of `(a + b)`'s normal equations without materializing the
/// sum: fills the shared scratch from `a` (+ `b`), then solves in place.
pub fn solve_normal_sum(
    a: &Gram,
    b: Option<&Gram>,
    dim: usize,
    l1: f64,
    l2: f64,
    scratch: &mut SolveScratch,
    warm: Option<&mut [f64]>,
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
    solve_prepared(scratch, yty, dim, l1, l2, warm)
}

/// Lower-triangular Cholesky factor of a moving design's normal matrix
/// (`xtx` + the `l2` ridge, intercept unpenalized), maintained across the
/// split sweep: rows entering a side apply a rank-1 update, rows leaving a
/// guarded rank-1 downdate. The Gram aggregate stays the source of truth —
/// the factor is a disposable cache around it, rebuilt on demand, so
/// correctness never depends on the incremental path. Storage is the
/// lower triangle, column-major (`L[i][j]` at `l[j*dim + i]`) so the
/// recurrence works contiguous slices; the recurrences are the classic
/// LINPACK `dchud`/`dchdd` shapes.
pub struct Chol {
    l: Vec<f64>,
    v: Vec<f64>, // rank-1 work buffer (allocation-free updates)
    dim: usize,
}

impl Chol {
    /// Factor `(a + b).xtx + l2·I*` (intercept unpenalized) — `None` when
    /// the matrix is not positive definite, in which case the caller keeps
    /// solving the exact historical way.
    pub fn factor_grams(a: &Gram, b: Option<&Gram>, l2: f64) -> Option<Chol> {
        let dim = a.dim();
        let mut xtx = a.xtx[..dim * dim].to_vec();
        if let Some(e) = b {
            for (dst, src) in xtx.iter_mut().zip(&e.xtx[..dim * dim]) {
                *dst += *src;
            }
        }
        let mut l = vec![0.0; dim * dim];
        for j in 0..dim {
            for i in j..dim {
                let mut s = xtx[i * dim + j] + if i == j && j > 0 { l2 } else { 0.0 };
                for k in 0..j {
                    s -= l[k * dim + i] * l[k * dim + j];
                }
                if i == j {
                    if !(s > 0.0) {
                        return None;
                    }
                    l[j * dim + j] = s.sqrt();
                } else {
                    l[j * dim + i] = s / l[j * dim + j];
                }
            }
        }
        Some(Chol {
            l,
            v: vec![0.0; dim],
            dim,
        })
    }

    /// `A + w·φφᵀ` — never fails: a positive-semidefinite addition keeps
    /// the factor valid.
    pub fn rank1_add(&mut self, phi: &[f64], w: f64) {
        self.rank1(phi, w, true);
    }

    /// `A − w·φφᵀ` — `false` when the downdate would leave the matrix
    /// non-positive-definite; the factor is then unusable and the caller
    /// drops it. (A refused downdate may leave `l` half-updated — dropping
    /// is the contract.)
    pub fn rank1_sub(&mut self, phi: &[f64], w: f64) -> bool {
        self.rank1(phi, w, false)
    }

    /// Shared recurrence for both directions: the L column update differs
    /// only in sign, and the work vector's update collapses to
    /// `(v − s·l_old)/c` in both (the rotation identities make the two
    /// textbook formulas coincide).
    fn rank1(&mut self, phi: &[f64], w: f64, add: bool) -> bool {
        if w <= 0.0 {
            return true;
        }
        let d = self.dim;
        let sw = w.sqrt();
        for (t, &s) in self.v[..d].iter_mut().zip(&phi[..d]) {
            *t = s * sw;
        }
        for j in 0..d {
            let x = self.l[j * d + j];
            let y = self.v[j];
            let r2 = if add { x * x + y * y } else { x * x - y * y };
            if !add && !(r2 > 1e-12 * x * x) {
                return false;
            }
            let r = r2.sqrt();
            let c = r / x;
            let s = y / x;
            let inv_c = 1.0 / c;
            self.l[j * d + j] = r;
            for (lp, vp) in self.l[j * d + j + 1..j * d + d]
                .iter_mut()
                .zip(&mut self.v[j + 1..d])
            {
                let lold = *lp;
                let sv = s * *vp;
                *lp = (lold + if add { sv } else { -sv }) * inv_c;
                *vp = (*vp - s * lold) * inv_c;
            }
        }
        true
    }

    /// Solve `(xtx + ridge)·b = xty` through the factor — O(dim²).
    pub fn solve_into(&self, xty: &[f64], out: &mut [f64]) {
        let d = self.dim;
        for i in 0..d {
            let mut s = xty[i];
            for k in 0..i {
                s -= self.l[k * d + i] * out[k];
            }
            out[i] = s / self.l[i * d + i];
        }
        for i in (0..d).rev() {
            let mut s = out[i];
            for k in i + 1..d {
                s -= self.l[i * d + k] * out[k];
            }
            out[i] = s / self.l[i * d + i];
        }
    }
}

/// One side of the linear scan's sweep. The moving Gram aggregate crosses
/// the cut in whole value groups (exact historical arithmetic); on top of
/// it, Cholesky factors of the two solve systems the side takes part in —
/// `gram` and `gram + extra` (`extra` = the column's fixed missing
/// aggregate) — follow the same rows one rank-1 step at a time and let
/// candidates be scored in O(dim²) instead of a fresh O(dim³) fit. The
/// folding is economic, not unconditional: a group wider than ~dim/3 rows
/// costs more to fold than to re-factor, so the sweep drops the factors
/// there and the solves take the exact historical path until a finer
/// group's [`SideTrack::ensure`] re-materializes them. Factors are
/// disposable (failed downdate, periodic refresh); correctness never
/// depends on the incremental path.
pub struct SideTrack {
    pub gram: Gram,
    pub extra: Option<Gram>,
    dim: usize,
    l2: f64,
    chols: [Option<Chol>; 2], // [plain, +extra]
    ops: u32,
    warm: [Vec<f64>; 2], // elastic-net warm starts (raw coefficients)
    xty: Vec<f64>,
    coef: Vec<f64>,
}

impl SideTrack {
    pub fn new(gram: Gram, extra: Option<Gram>, l2: f64) -> Self {
        let dim = gram.dim();
        Self {
            gram,
            extra,
            dim,
            l2,
            chols: [None, None],
            ops: 0,
            warm: [vec![0.0; dim], vec![0.0; dim]],
            xty: vec![0.0; dim],
            coef: vec![0.0; dim],
        }
    }

    /// Materialize the maintained factors from the current aggregate —
    /// called at groups small enough for the row-by-row folding to pay.
    /// Must run before the next group is folded into the Gram so the
    /// factor and the rank-1 steps stay consistent.
    pub fn ensure(&mut self) {
        if self.chols[0].is_none() {
            self.chols[0] = Chol::factor_grams(&self.gram, None, self.l2);
        }
        if self.chols[1].is_none() && self.extra.is_some() {
            self.chols[1] = Chol::factor_grams(&self.gram, self.extra.as_ref(), self.l2);
        }
    }

    /// Drop the maintained factors — the caller is folding a group too
    /// large for row-by-row updates.
    pub fn drop_factors(&mut self) {
        self.chols[0] = None;
        self.chols[1] = None;
    }

    /// One row crossing the cut: `dir > 0` entering this side (rank-1
    /// update), `dir < 0` leaving (guarded rank-1 downdate). The Gram
    /// aggregate is moved by the caller in whole groups; the factors track
    /// the same contributions row by row.
    pub fn move_row(&mut self, phi: &[f64], w: f64, dir: i32) {
        self.ops += 1;
        if self.ops % 2048 == 0 {
            self.chols[0] = None;
            self.chols[1] = None; // periodic refresh bounds drift
        }
        for i in 0..2 {
            if let Some(c) = self.chols[i].as_mut() {
                let ok = if dir > 0 {
                    c.rank1_add(phi, w);
                    true
                } else {
                    c.rank1_sub(phi, w)
                };
                if !ok {
                    self.chols[i] = None;
                }
            }
        }
    }

    /// Residual score of this side's system in the shared `yty − xty·b`
    /// convention (`plus` selects the `gram + extra` system).
    pub fn sse(&mut self, plus: bool, l1: f64, scratch: &mut SolveScratch) -> f64 {
        let idx = if plus && self.extra.is_some() { 1 } else { 0 };
        let extra = if plus { self.extra.as_ref() } else { None };
        if l1 > 0.0 {
            // Elastic net: no maintained factor — the historical descent,
            // warm-started from the previous candidate's coefficients.
            return solve_normal_sum(
                &self.gram,
                extra,
                self.dim,
                l1,
                self.l2,
                scratch,
                Some(&mut self.warm[idx][..]),
            );
        }
        let d = self.dim;
        self.xty[..d].copy_from_slice(&self.gram.xty[..d]);
        let mut yty = self.gram.yty;
        if let Some(e) = extra {
            for (t, s) in self.xty[..d].iter_mut().zip(&e.xty[..d]) {
                *t += *s;
            }
            yty += e.yty;
        }
        if let Some(c) = self.chols[idx].as_ref() {
            c.solve_into(&self.xty[..d], &mut self.coef[..d]);
            let explained: f64 = self.xty[..d]
                .iter()
                .zip(&self.coef[..d])
                .map(|(t, s)| t * s)
                .sum();
            (yty - explained).max(0.0)
        } else {
            solve_normal_sum(&self.gram, extra, d, l1, self.l2, scratch, None)
        }
    }
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
    warm: Option<&mut [f64]>,
) -> f64 {
    let n = scratch.a[0].max(1.0);
    let sum_y = scratch.c[0];
    let mut mean = vec![0.0; dim];
    let mut scale = vec![1.0; dim];
    for j in 1..dim {
        mean[j] = scratch.a[j] / n; // xtx[0][j] = sum of column j
    }
    for j in 1..dim {
        let var = scratch.a[j * dim + j] - n * mean[j] * mean[j];
        scale[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }
    let mut g = vec![0.0; (dim - 1) * (dim - 1)];
    let mut r = vec![0.0; dim - 1];
    for j in 1..dim {
        r[j - 1] = (scratch.c[j] - mean[j] * sum_y) / scale[j];
        for k in 1..dim {
            let gc = scratch.a[j * dim + k] - n * mean[j] * mean[k];
            g[(j - 1) * (dim - 1) + (k - 1)] = gc / (scale[j] * scale[k]);
        }
    }
    let mut b = vec![0.0; dim - 1];
    // Warm start from the previous candidate's coefficients (scaled into
    // this candidate's standardization): adjacent candidates differ by one
    // value group, so the descent usually converges in a few sweeps.
    if let Some(w) = warm.as_deref() {
        for j in 1..dim {
            b[j - 1] = w[j] * scale[j];
        }
    }
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
    scratch.b[0] = sum_y / n;
    for j in 1..dim {
        let bj = b[j - 1] / scale[j];
        scratch.b[j] = bj;
        scratch.b[0] -= mean[j] * bj;
    }
    if let Some(w) = warm {
        w[..dim].copy_from_slice(&scratch.b[..dim]);
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
/// are the elastic-net penalties (intercept unpenalized). `mono` is indexed by
/// ORIGINAL column: a nonzero entry constrains the feature's direction, and
/// linear leaves carry feature effects in their coefficients - so the
/// coefficient is projected onto the allowed halfline (`+1` ⇒ coef ≥ 0,
/// `-1` ⇒ coef ≤ 0) to keep predictions monotone. Indicators and the
/// intercept are never constrained.
pub fn fit_linear(
    g: &Gram,
    chosen: &[usize],
    with_indicators: bool,
    mono: &[i8],
    l1: f64,
    l2: f64,
) -> LinearModel {
    let nc = chosen.len();
    let dim = 1 + nc + if with_indicators { nc } else { 0 };
    let mut scratch = SolveScratch::new(dim);
    solve_normal(&g.xtx, &g.xty, g.yty, dim, l1, l2, &mut scratch);
    let mut sol = scratch.b.clone();
    for (k, &c) in chosen.iter().enumerate() {
        match mono.get(c).copied().unwrap_or(0) {
            1 => sol[k + 1] = sol[k + 1].max(0.0),
            -1 => sol[k + 1] = sol[k + 1].min(0.0),
            _ => {}
        }
    }
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

impl LinearModel {
    /// Evaluate on a design row `[1, values (0 where missing), NA
    /// indicators]` — the raw model output, used for monotone boundary
    /// ordering (comparing children's fitted outputs AT the split threshold).
    pub fn eval_design(&self, row: &[f64]) -> f64 {
        let nc = self.coef.len();
        let mut y = self.intercept;
        for k in 0..nc {
            y += self.coef[k] * row[1 + k];
        }
        for k in 0..self.na_coef.len() {
            y += self.na_coef[k] * row[1 + nc + k];
        }
        y
    }
}

/// Penalized canonical-link GLM by damped IRLS. The weighted least-squares
/// step reuses the Gaussian Gram/faer solver; constrained/lasso steps use
/// coordinate descent on the same Gram. L1 standardization is fixed from
/// the original case-weighted design, not changed by IRLS working weights.
///
/// Fisher variances are floored at 1e-12, steps are backtracked against
/// penalized deviance, and convergence requires relative objective change
/// below 1e-11. Separation therefore approaches a finite numerical boundary;
/// exact pure/zero leaves are handled by the caller. A failed or exhausted
/// fit returns None, never a successful-looking constant fallback.
pub(crate) fn fit_glm(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    chosen: &[usize],
    with_indicators: bool,
    mono: &[i8],
    l1: f64,
    l2: f64,
    family: LeafFamily,
    mean: f64,
) -> Option<LinearModel> {
    let nc = chosen.len();
    let dim = 1 + nc + if with_indicators { nc } else { 0 };
    let mut current = LinearModel {
        cols: chosen.to_vec(),
        coef: vec![0.0; nc],
        na_coef: if with_indicators { vec![0.0; nc] } else { Vec::new() },
        intercept: match family {
            LeafFamily::Binomial => mean.ln() - (-mean).ln_1p(),
            LeafFamily::Poisson => mean.ln(),
            LeafFamily::Gaussian => mean,
        },
    };
    let mut gram = Gram::new(dim);
    let mut row = vec![0.0; dim];
    let mut scales = vec![1.0; dim];
    if l1 > 0.0 {
        for &r in rows {
            design_row(&mut row, cols, r, chosen, with_indicators);
            gram.add_row(&row, 0.0, w[r]);
        }
        for j in 1..dim {
            let variance = gram.xtx[j * dim + j]
                - gram.xtx[j] * gram.xtx[j] / gram.ws;
            if variance > 1e-12 {
                scales[j] = variance.sqrt();
            }
        }
    }
    let penalty = |model: &LinearModel| -> f64 {
        model.coef.iter().chain(&model.na_coef).enumerate().map(|(j, &b)| {
            let standardized = b * scales[j + 1];
            l1 * standardized.abs() + 0.5 * l2 * standardized * standardized
        }).sum()
    };
    let objective = |model: &LinearModel, row: &mut [f64]| -> f64 {
        let mut loss = 0.0;
        for &r in rows {
            if w[r] <= 0.0 {
                continue;
            }
            design_row(row, cols, r, chosen, with_indicators);
            let mean = family.inverse_link(model.eval_design(row));
            loss += 0.5 * w[r] * family.deviance(y[r], mean);
        }
        loss + penalty(model)
    };
    let constrained = chosen.iter().any(|&c| mono.get(c).copied().unwrap_or(0) != 0);
    let mut old_objective = objective(&current, &mut row);
    if !old_objective.is_finite() {
        return None;
    }
    let mut beta = vec![0.0; dim];
    for _ in 0..200 {
        gram.xtx.fill(0.0);
        gram.xty.fill(0.0);
        gram.yty = 0.0;
        gram.wy = 0.0;
        gram.ws = 0.0;
        gram.n = 0;
        for &r in rows {
            if w[r] <= 0.0 {
                continue;
            }
            design_row(&mut row, cols, r, chosen, with_indicators);
            let eta = current.eval_design(&row);
            let mean = family.inverse_link(eta);
            let variance = match family {
                LeafFamily::Binomial => (mean * (1.0 - mean)).max(1e-12),
                LeafFamily::Poisson => mean.max(1e-12),
                LeafFamily::Gaussian => 1.0,
            };
            let working = eta + (y[r] - mean) / variance;
            gram.add_row(&row, working, w[r] * variance);
        }
        if gram.xtx.iter().chain(&gram.xty).any(|value| !value.is_finite()) {
            return None;
        }
        let mut candidate = if l1 == 0.0 && !constrained {
            // Normalize working mass so the shared solver's rank threshold
            // stays relative even when separation makes Fisher weights tiny.
            let mass = gram.ws;
            if !(mass > 0.0 && mass.is_finite()) {
                return None;
            }
            for value in gram.xtx.iter_mut().chain(&mut gram.xty) {
                *value /= mass;
            }
            gram.yty /= mass;
            fit_linear(&gram, chosen, with_indicators, mono, 0.0, l2 / mass)
        } else {
            beta[0] = current.intercept;
            beta[1..1 + nc].copy_from_slice(&current.coef);
            if with_indicators {
                beta[1 + nc..].copy_from_slice(&current.na_coef);
            }
            let mut converged = false;
            for _ in 0..2000 {
                let mut delta = 0.0_f64;
                let mut magnitude = 1.0_f64;
                for j in 0..dim {
                    let ridge = if j == 0 { 0.0 } else { l2 * scales[j] * scales[j] };
                    let diagonal = gram.xtx[j * dim + j] + ridge;
                    if diagonal <= 0.0 {
                        beta[j] = 0.0;
                        continue;
                    }
                    let mut partial = gram.xty[j];
                    for k in 0..dim {
                        if j != k {
                            partial -= gram.xtx[j * dim + k] * beta[k];
                        }
                    }
                    let threshold = if j == 0 { 0.0 } else { l1 * scales[j] };
                    let mut next = partial.signum() * (partial.abs() - threshold).max(0.0) / diagonal;
                    if j > 0 && j <= nc {
                        match mono.get(chosen[j - 1]).copied().unwrap_or(0) {
                            1 => next = next.max(0.0),
                            -1 => next = next.min(0.0),
                            _ => {}
                        }
                    }
                    delta = delta.max((next - beta[j]).abs() * scales[j]);
                    magnitude = magnitude.max(next.abs() * scales[j]);
                    beta[j] = next;
                }
                if delta <= 1e-10 * magnitude {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return None;
            }
            LinearModel {
                cols: chosen.to_vec(),
                coef: beta[1..1 + nc].to_vec(),
                na_coef: if with_indicators { beta[1 + nc..].to_vec() } else { Vec::new() },
                intercept: beta[0],
            }
        };
        if !candidate.intercept.is_finite()
            || candidate.coef.iter().chain(&candidate.na_coef).any(|b| !b.is_finite())
        {
            return None;
        }
        let mut next_objective = objective(&candidate, &mut row);
        for _ in 0..50 {
            if next_objective.is_finite() && next_objective <= old_objective {
                break;
            }
            candidate.intercept = 0.5 * (candidate.intercept + current.intercept);
            for (new, old) in candidate.coef.iter_mut().zip(&current.coef) {
                *new = 0.5 * (*new + old);
            }
            for (new, old) in candidate.na_coef.iter_mut().zip(&current.na_coef) {
                *new = 0.5 * (*new + old);
            }
            next_objective = objective(&candidate, &mut row);
        }
        if !next_objective.is_finite() || next_objective > old_objective {
            return None;
        }
        let improvement = old_objective - next_objective;
        current = candidate;
        if improvement <= 1e-11 * (1.0 + old_objective) {
            return Some(current);
        }
        old_objective = next_objective;
    }
    None
}

/// Per-node frame statistics for printing (rpart's `frame`): count, weighted
/// mean and residual SSE. O(rows) to accumulate - no digest.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub n: usize,
    pub wsum: f64,
    pub mean: f64,
    pub sse: f64,
}

impl FrameStats {
    pub fn from_ty(y: &[f64], w: &[f64], rows: &[usize]) -> Self {
        let mut wsum = 0.0;
        let mut wy = 0.0;
        for &r in rows {
            if w[r] > 0.0 {
                wsum += w[r];
                wy += w[r] * y[r];
            }
        }
        let mean = wy / wsum.max(1e-300);
        let sse: f64 = rows
            .iter()
            .filter(|&&r| w[r] > 0.0)
            .map(|&r| w[r] * (y[r] - mean) * (y[r] - mean))
            .sum();
        Self {
            n: rows.iter().filter(|&&r| w[r] > 0.0).count(),
            wsum,
            mean,
            sse,
        }
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
