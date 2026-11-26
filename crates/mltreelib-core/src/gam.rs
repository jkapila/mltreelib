//! Generalized Additive Models (GAM) with Penalized B-Splines (P-Splines).
//!
//! Features:
//! - Exact cubic B-spline basis expansion using Cox-de Boor recurrence.
//! - Second-difference roughness penalty ($S = D_2^\top D_2$) penalizing excessive spline curvature.
//! - Centered basis functions ensuring strict identifiability with the intercept.
//! - Closed-form penalized weighted least squares solved via Cholesky decomposition (LLT).
//! - Automatic smoothing parameter ($\lambda$) selection via Generalized Cross-Validation (GCV).
//! - Out-of-bounds boundary knot clamping for stable out-of-sample prediction.

use faer::prelude::*;

/// Parameters controlling Penalized B-Spline GAM estimation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GamParams {
    /// Number of spline basis functions per predictor (default 8).
    pub n_knots: usize,
    /// Polynomial degree of the B-splines (default 3 = cubic).
    pub degree: usize,
    /// Smoothing penalty parameter lambda. If None, chosen automatically via GCV grid search.
    #[serde(with = "crate::persistence::float")]
    pub lambda: Option<f64>,
    /// Minimum lambda exponent for GCV grid search (default 1e-4).
    #[serde(with = "crate::persistence::float")]
    pub lambda_min: f64,
    /// Maximum lambda exponent for GCV grid search (default 1e4).
    #[serde(with = "crate::persistence::float")]
    pub lambda_max: f64,
    /// Number of grid evaluation points for GCV search (default 25).
    pub n_grid: usize,
}

impl Default for GamParams {
    fn default() -> Self {
        Self {
            n_knots: 8,
            degree: 3,
            lambda: None,
            lambda_min: 1e-4,
            lambda_max: 1e4,
            n_grid: 25,
        }
    }
}

/// Evaluates a single set of B-spline basis values using Cox-de Boor recurrence.
pub fn eval_bspline_basis(x: f64, knots: &[f64], n_knots: usize, degree: usize) -> Vec<f64> {
    let m = knots.len();
    if m < n_knots + degree + 1 {
        return vec![0.0; n_knots];
    }

    // Clamp input within knot boundaries
    let v = x.clamp(knots[0], knots[m - 1]);

    // Degree 0 base case
    let mut n_prev = vec![0.0; m - 1];
    for i in 0..(m - 1) {
        if knots[i] <= v && (v < knots[i + 1] || (v == knots[m - 1] && v <= knots[i + 1])) {
            n_prev[i] = 1.0;
        }
    }

    // Higher degrees 1..=degree
    for p in 1..=degree {
        let mut n_curr = vec![0.0; m - 1 - p];
        for i in 0..(m - 1 - p) {
            let d1 = knots[i + p] - knots[i];
            let t1 = if d1 > 1e-14 {
                (v - knots[i]) / d1 * n_prev[i]
            } else {
                0.0
            };

            let d2 = knots[i + p + 1] - knots[i + 1];
            let t2 = if d2 > 1e-14 {
                (knots[i + p + 1] - v) / d2 * n_prev[i + 1]
            } else {
                0.0
            };

            n_curr[i] = t1 + t2;
        }
        n_prev = n_curr;
    }

    // Return the first n_knots basis functions
    if n_prev.len() >= n_knots {
        n_prev[..n_knots].to_vec()
    } else {
        let mut res = n_prev;
        res.resize(n_knots, 0.0);
        res
    }
}

/// Generates a clamped knot vector for a predictor given its empirical values.
pub fn generate_knots(values: &[f64], n_knots: usize, degree: usize) -> Vec<f64> {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
        min_val = 0.0;
        max_val = 1.0;
    }

    let n_interior = n_knots.saturating_sub(degree);
    let mut interior = Vec::with_capacity(n_interior + 1);
    for i in 0..=n_interior {
        let frac = i as f64 / (n_interior as f64).max(1.0);
        interior.push(min_val + frac * (max_val - min_val));
    }

    // Clamped knot vector: repeat min and max 'degree' times
    let mut knots = Vec::with_capacity(n_knots + degree + 1);
    for _ in 0..degree {
        knots.push(min_val);
    }
    knots.extend_from_slice(&interior);
    for _ in 0..degree {
        knots.push(max_val);
    }
    knots
}

/// Constructs second-difference penalty matrix $S = D_2^\top D_2$ of size $K \times K$.
pub fn compute_second_difference_penalty(k: usize) -> Vec<f64> {
    let mut s = vec![0.0; k * k];
    if k < 3 {
        return s;
    }

    // D_2 has size (k - 2) x k with stencil [1, -2, 1]
    let n_diff = k - 2;
    for r in 0..n_diff {
        let row_indices = [r, r + 1, r + 2];
        let weights = [1.0, -2.0, 1.0];
        for (i_idx, &i) in row_indices.iter().enumerate() {
            let wi = weights[i_idx];
            for (j_idx, &j) in row_indices.iter().enumerate() {
                let wj = weights[j_idx];
                s[i * k + j] += wi * wj;
            }
        }
    }
    s
}

/// Fitted Penalized-Spline Generalized Additive Model.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GamModel {
    pub params: GamParams,
    pub intercept: f64,
    /// Per-feature spline coefficients: [p][n_knots].
    pub coefs: Vec<Vec<f64>>,
    /// Per-feature clamped knot vectors: [p][M].
    pub feature_knots: Vec<Vec<f64>>,
    /// Per-feature basis column means for identifiability centering: [p][n_knots].
    pub feature_means: Vec<Vec<f64>>,
    /// Effective degrees of freedom (trace of hat matrix).
    #[serde(with = "crate::persistence::float")]
    pub edf: f64,
    /// Generalized Cross-Validation score at optimal lambda.
    #[serde(with = "crate::persistence::float")]
    pub gcv: f64,
    /// Selected or specified smoothing penalty lambda.
    #[serde(with = "crate::persistence::float")]
    pub lambda: f64,
    pub n_train: usize,
    pub n_features: usize,
}

impl GamModel {
    /// Predicts target value for a single feature row.
    #[inline]
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let mut pred = self.intercept;
        let k = self.params.n_knots;
        let deg = self.params.degree;

        for (j, &val) in row.iter().enumerate() {
            if j >= self.coefs.len() {
                break;
            }
            let knots = &self.feature_knots[j];
            let means = &self.feature_means[j];
            let coefs = &self.coefs[j];

            let basis = eval_bspline_basis(val, knots, k, deg);
            for b in 0..k {
                let centered = basis[b] - means[b];
                pred += coefs[b] * centered;
            }
        }
        pred
    }

    /// Predicts target values for a matrix of rows.
    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|r| self.predict_row(r)).collect()
    }

    /// Total number of basis parameters including intercept.
    pub fn n_terms(&self) -> usize {
        1 + self.coefs.len() * self.params.n_knots
    }
}

/// Standalone Penalized B-Spline GAM Regressor.
#[derive(Clone, Debug, Default)]
pub struct GamRegressor {
    pub params: GamParams,
    pub model: Option<GamModel>,
}

impl GamRegressor {
    /// Creates a new `GamRegressor`.
    pub fn new(n_knots: usize, lambda: Option<f64>) -> Self {
        Self {
            params: GamParams {
                n_knots: n_knots.max(4),
                lambda,
                ..Default::default()
            },
            model: None,
        }
    }

    /// Fits GAM on unweighted features and target.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
        self.fit_weighted(x, y, None)
    }

    /// Fits GAM with optional observation sample weights.
    pub fn fit_weighted(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<(), String> {
        let n = x.len();
        if n == 0 {
            return Err("Empty training data".to_string());
        }
        if n != y.len() {
            return Err(format!("Feature rows {} != target length {}", n, y.len()));
        }
        let p = x[0].len();
        if p == 0 {
            return Err("Zero features provided".to_string());
        }

        let k = self.params.n_knots;
        let deg = self.params.degree;
        let total_w: f64 = weights.map_or(n as f64, |w| w.iter().sum());
        if total_w <= 0.0 {
            return Err("Total sample weight must be positive".to_string());
        }

        // 1. Generate knot vectors for each feature
        let mut feature_knots = Vec::with_capacity(p);
        for j in 0..p {
            let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            let knots = generate_knots(&col, k, deg);
            feature_knots.push(knots);
        }

        // 2. Evaluate uncentered basis for each sample and feature
        // design_raw: [n][p * k]
        let pk = p * k;
        let mut basis_eval = vec![vec![0.0; pk]; n];
        for (i, row) in x.iter().enumerate() {
            for j in 0..p {
                let knots = &feature_knots[j];
                let b_vals = eval_bspline_basis(row[j], knots, k, deg);
                let offset = j * k;
                basis_eval[i][offset..(k + offset)].copy_from_slice(&b_vals[..k]);
            }
        }

        // 3. Compute weighted column means for identifiability centering
        let mut feature_means = vec![vec![0.0; k]; p];
        for j in 0..p {
            let offset = j * k;
            for b in 0..k {
                let mut sum_val = 0.0;
                for i in 0..n {
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    sum_val += w * basis_eval[i][offset + b];
                }
                feature_means[j][b] = sum_val / total_w;
            }
        }

        // 4. Build centered full design matrix Phi: [n][1 + pk] (intercept at col 0)
        let total_cols = 1 + pk;
        let mut phi = vec![vec![0.0; total_cols]; n];
        for i in 0..n {
            phi[i][0] = 1.0;
            for j in 0..p {
                let offset = j * k;
                for b in 0..k {
                    phi[i][1 + offset + b] = basis_eval[i][offset + b] - feature_means[j][b];
                }
            }
        }

        // 5. Precompute Phi^T W Phi and Phi^T W y
        let mut xtx = vec![0.0; total_cols * total_cols];
        let mut xty = vec![0.0; total_cols];

        for i in 0..n {
            let wi = weights.map_or(1.0, |ws| ws[i]);
            if wi <= 0.0 {
                continue;
            }
            let yi = y[i];
            let w_yi = wi * yi;
            let row_i = &phi[i];

            for j1 in 0..total_cols {
                let v1 = row_i[j1];
                xty[j1] += v1 * w_yi;
                let w_v1 = wi * v1;
                for j2 in 0..total_cols {
                    xtx[j1 * total_cols + j2] += w_v1 * row_i[j2];
                }
            }
        }

        // 6. Base roughness penalty matrix S_base: block diagonal of D_2^T D_2 for each feature
        let s_single = compute_second_difference_penalty(k);
        let mut s_block = vec![0.0; total_cols * total_cols];
        for j in 0..p {
            let offset = 1 + j * k;
            for r in 0..k {
                for c in 0..k {
                    s_block[(offset + r) * total_cols + (offset + c)] = s_single[r * k + c];
                }
            }
        }

        // 7. Lambda selection (GCV grid search or user-specified)
        let candidate_lambdas: Vec<f64> = if let Some(lam) = self.params.lambda {
            vec![lam]
        } else {
            let l_min = self.params.lambda_min.max(1e-6);
            let l_max = self.params.lambda_max.max(l_min * 10.0);
            let log_min = l_min.ln();
            let log_max = l_max.ln();
            let n_pts = self.params.n_grid.max(5);
            (0..n_pts)
                .map(|i| {
                    let frac = i as f64 / (n_pts - 1) as f64;
                    (log_min + frac * (log_max - log_min)).exp()
                })
                .collect()
        };

        let mut best_lambda = candidate_lambdas[0];
        let mut best_gcv = f64::INFINITY;
        let mut best_coef = vec![0.0; total_cols];
        let mut best_edf = total_cols as f64;

        for &lam in &candidate_lambdas {
            // Form A = X^T W X + lam * S + ridge * I
            let mut a_eval = xtx.clone();
            for i in 0..(total_cols * total_cols) {
                a_eval[i] += lam * s_block[i];
            }
            // Tiny diagonal ridge for conditioning on basis columns
            for col in 1..total_cols {
                a_eval[col * total_cols + col] += 1e-8;
            }

            let a_mat = faer::Mat::from_fn(total_cols, total_cols, |i, j| a_eval[i * total_cols + j]);
            let b_mat = faer::Mat::from_fn(total_cols, 1, |i, _| xty[i]);

            let coef_cand: Vec<f64> = if let Ok(llt) = a_mat.as_ref().llt(faer::Side::Lower) {
                let sol = llt.solve(b_mat.as_ref());
                (0..total_cols).map(|i| sol[(i, 0)]).collect()
            } else {
                let qr = a_mat.as_ref().col_piv_qr();
                let sol = qr.solve(b_mat.as_ref());
                (0..total_cols).map(|i| sol[(i, 0)]).collect()
            };

            // Compute weighted RSS
            let mut rss = 0.0;
            for i in 0..n {
                let wi = weights.map_or(1.0, |ws| ws[i]);
                if wi <= 0.0 {
                    continue;
                }
                let mut pred_i = 0.0;
                for c in 0..total_cols {
                    pred_i += coef_cand[c] * phi[i][c];
                }
                let diff = y[i] - pred_i;
                rss += wi * diff * diff;
            }

            // Approximate effective degrees of freedom: tr( (X^T W X + lam S)^{-1} X^T W X )
            // Evaluated via trace of product or diagonal approximation
            let mut edf = 1.0; // intercept
            for col in 1..total_cols {
                let diag_xtx = xtx[col * total_cols + col];
                let diag_pen = lam * s_block[col * total_cols + col];
                if diag_xtx + diag_pen > 1e-12 {
                    edf += diag_xtx / (diag_xtx + diag_pen + 1e-8);
                }
            }
            edf = edf.clamp(1.0, (total_cols as f64).min(total_w - 1.0));

            // GCV = (RSS / n) / (1 - edf / n)^2
            let denom = (1.0 - edf / total_w).max(1e-4);
            let gcv = (rss / total_w) / (denom * denom);

            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lam;
                best_coef = coef_cand;
                best_edf = edf;
            }
        }

        // 8. Package final fitted model
        let intercept = best_coef[0];
        let mut coefs = Vec::with_capacity(p);
        for j in 0..p {
            let offset = 1 + j * k;
            coefs.push(best_coef[offset..(offset + k)].to_vec());
        }

        self.model = Some(GamModel {
            params: self.params.clone(),
            intercept,
            coefs,
            feature_knots,
            feature_means,
            edf: best_edf,
            gcv: best_gcv,
            lambda: best_lambda,
            n_train: n,
            n_features: p,
        });

        Ok(())
    }

    /// Predicts target values on input features.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| "Model not fitted".to_string())?;
        Ok(m.predict(x))
    }
}
