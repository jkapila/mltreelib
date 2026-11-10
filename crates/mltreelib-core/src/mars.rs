//! Multivariate Adaptive Regression Splines (MARS / Friedman 1991, `earth` compatible).
//!
//! Features:
//! - Piecewise-linear reflected hinge basis functions: `h(x - t) = max(0, x - t)` and `h(t - x) = max(0, t - x)`.
//! - Pure linear terms (`linpreds`) without hinges.
//! - Multi-variable interaction products up to `degree` with optional `allowed` interaction matrix.
//! - Fast MARS residual screening (Friedman 1993) for 100x-500x speedup in forward pass.
//! - Sample weights across forward pass, OLS fitting, and GCV.
//! - Backward GCV and K-fold CV pruning (`pmethod`).
//! - GLM families: Gaussian (regression), Binomial (logistic classification), and Poisson (count models).

use faer::prelude::*;

/// A single factor: either a hinge function or a pure linear predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct HingeFactor {
    pub feature: usize,
    pub knot: f64,
    /// If true: max(0, x - knot); if false: max(0, knot - x).
    pub is_forward: bool,
    /// If true: pure linear term (eval(x) = x) without knot splitting.
    pub is_linear: bool,
}

impl HingeFactor {
    pub fn hinge(feature: usize, knot: f64, is_forward: bool) -> Self {
        Self {
            feature,
            knot,
            is_forward,
            is_linear: false,
        }
    }

    pub fn linear(feature: usize) -> Self {
        Self {
            feature,
            knot: 0.0,
            is_forward: true,
            is_linear: true,
        }
    }

    #[inline]
    pub fn eval(&self, val: f64) -> f64 {
        if self.is_linear {
            val
        } else if self.is_forward {
            (val - self.knot).max(0.0)
        } else {
            (self.knot - val).max(0.0)
        }
    }

    pub fn to_string_with_names(&self, col_names: Option<&[String]>) -> String {
        let name = match col_names.and_then(|names| names.get(self.feature)) {
            Some(n) => n.clone(),
            None => format!("x{}", self.feature),
        };
        if self.is_linear {
            name
        } else if self.is_forward {
            format!("h({} - {:.4})", name, self.knot)
        } else {
            format!("h({:.4} - {})", self.knot, name)
        }
    }
}

/// A basis function: product of zero or more factors (empty = intercept 1.0).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BasisFunction {
    pub factors: Vec<HingeFactor>,
}

impl BasisFunction {
    pub fn constant() -> Self {
        Self { factors: Vec::new() }
    }

    #[inline]
    pub fn eval(&self, row: &[f64]) -> f64 {
        let mut val = 1.0;
        for f in &self.factors {
            val *= f.eval(row[f.feature]);
            if val == 0.0 {
                return 0.0;
            }
        }
        val
    }

    pub fn to_string_with_names(&self, col_names: Option<&[String]>) -> String {
        if self.factors.is_empty() {
            return "(Intercept)".to_string();
        }
        self.factors
            .iter()
            .map(|f| f.to_string_with_names(col_names))
            .collect::<Vec<_>>()
            .join(" * ")
    }
}

/// GLM family for MARS model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarsGlmFamily {
    Gaussian,
    Binomial,
    Poisson,
}

impl MarsGlmFamily {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "gaussian" | "ols" | "regression" => Ok(Self::Gaussian),
            "binomial" | "logistic" | "classification" => Ok(Self::Binomial),
            "poisson" | "count" => Ok(Self::Poisson),
            other => Err(format!("Unknown MarsGlmFamily '{}'", other)),
        }
    }
}

/// Pruning method for MARS backward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarsPruningMethod {
    Backward,
    None,
    Cv(usize),
}

impl MarsPruningMethod {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "backward" | "gcv" => Ok(Self::Backward),
            "none" | "forward" => Ok(Self::None),
            "cv" | "cv5" => Ok(Self::Cv(5)),
            "cv10" => Ok(Self::Cv(10)),
            other => Err(format!("Unknown MarsPruningMethod '{}'", other)),
        }
    }
}

/// Parameters for training a MARS model.
#[derive(Clone, Debug)]
pub struct MarsParams {
    /// Maximum degree of interaction (1 = additive splines, 2 = pairwise interactions).
    pub degree: usize,
    /// Maximum number of basis terms in forward pass (including intercept).
    pub max_terms: usize,
    /// Generalized Cross Validation (GCV) penalty per knot parameter (typically 2.0 to 3.0).
    pub penalty: f64,
    /// Forward pass stopping threshold: minimum RSS improvement fraction.
    pub thresh: f64,
    /// Minimum observations between candidate knots (0 = automatic).
    pub minspan: usize,
    /// Minimum observations from predictor boundary (0 = automatic).
    pub endspan: usize,
    /// Predictor indices forced to enter only linearly (no knots).
    pub linpreds: Vec<usize>,
    /// Fast MARS screening candidate count (0 = exhaustive OLS evaluation, default = 10).
    pub fast_k: usize,
    /// Pruning method: Backward (GCV), None, or Cv(k).
    pub pmethod: MarsPruningMethod,
    /// GLM link family: Gaussian (default), Binomial (logistic), or Poisson.
    pub glm: MarsGlmFamily,
    /// Allowed interaction pairs (p x p boolean matrix, None = all allowed).
    pub allowed: Option<Vec<Vec<bool>>>,
}

impl Default for MarsParams {
    fn default() -> Self {
        Self {
            degree: 1,
            max_terms: 21,
            penalty: 2.0,
            thresh: 0.001,
            minspan: 0,
            endspan: 0,
            linpreds: Vec::new(),
            fast_k: 20,
            pmethod: MarsPruningMethod::Backward,
            glm: MarsGlmFamily::Gaussian,
            allowed: None,
        }
    }
}

/// Fitted Multivariate Adaptive Regression Splines model.
#[derive(Clone, Debug)]
pub struct MarsModel {
    pub params: MarsParams,
    pub basis: Vec<BasisFunction>,
    pub coef: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
    pub n_train: usize,
    pub n_features: usize,
}

impl MarsModel {
    #[inline]
    pub fn linear_predictor_row(&self, row: &[f64]) -> f64 {
        let mut pred = 0.0;
        for (b, &c) in self.basis.iter().zip(self.coef.iter()) {
            pred += c * b.eval(row);
        }
        pred
    }

    #[inline]
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let eta = self.linear_predictor_row(row);
        match self.params.glm {
            MarsGlmFamily::Gaussian => eta,
            MarsGlmFamily::Binomial => 1.0 / (1.0 + (-eta).exp()),
            MarsGlmFamily::Poisson => eta.exp(),
        }
    }

    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|r| self.predict_row(r)).collect()
    }

    pub fn predict_proba(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if self.params.glm != MarsGlmFamily::Binomial {
            return Err("predict_proba is only supported for Binomial GLM family".to_string());
        }
        Ok(data.iter().map(|r| self.predict_row(r)).collect())
    }

    pub fn n_terms(&self) -> usize {
        self.basis.len()
    }

    pub fn equation(&self, col_names: Option<&[String]>) -> String {
        let mut terms = Vec::new();
        for (b, &c) in self.basis.iter().zip(self.coef.iter()) {
            if b.factors.is_empty() {
                terms.push(format!("{:.4}", c));
            } else {
                let sign = if c >= 0.0 { "+" } else { "-" };
                terms.push(format!("{} {:.4} * {}", sign, c.abs(), b.to_string_with_names(col_names)));
            }
        }
        let expr = terms.join(" ");
        match self.params.glm {
            MarsGlmFamily::Gaussian => expr,
            MarsGlmFamily::Binomial => format!("logit(p) = {}", expr),
            MarsGlmFamily::Poisson => format!("log(mu) = {}", expr),
        }
    }
}

/// Standalone MARS regressor / classifier.
#[derive(Clone, Debug, Default)]
pub struct MarsRegressor {
    pub params: MarsParams,
    pub model: Option<MarsModel>,
}

impl MarsRegressor {
    pub fn new(degree: usize, max_terms: usize, penalty: f64) -> Self {
        Self {
            params: MarsParams {
                degree: degree.max(1),
                max_terms: max_terms.max(3),
                penalty: penalty.max(0.0),
                ..Default::default()
            },
            model: None,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
        self.fit_weighted(x, y, None)
    }

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
        for (i, row) in x.iter().enumerate() {
            if row.len() != p {
                return Err(format!("Inconsistent feature row length at row {}", i));
            }
            for (j, &val) in row.iter().enumerate() {
                if val.is_nan() {
                    return Err(format!("NaN found in feature {} at row {}", j, i));
                }
            }
        }
        for (i, &val) in y.iter().enumerate() {
            if val.is_nan() {
                return Err(format!("NaN found in target at row {}", i));
            }
        }

        if let Some(w) = weights {
            if w.len() != n {
                return Err(format!("Weights length {} != sample count {}", w.len(), n));
            }
        }

        let total_weight: f64 = match weights {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };
        if total_weight <= 0.0 {
            return Err("Sum of sample weights must be positive".to_string());
        }

        let minspan = if self.params.minspan > 0 {
            self.params.minspan
        } else {
            (1 + (n as f64).log10().round() as usize).max(2).min(n / 4).max(1)
        };
        let endspan = if self.params.endspan > 0 {
            self.params.endspan
        } else {
            minspan
        };

        // 1. Initial basis B_0(x) = 1
        let mut active_basis: Vec<BasisFunction> = vec![BasisFunction::constant()];
        let (_initial_coef, initial_rss) = fit_basis(&active_basis, x, y, weights, self.params.glm)?;
        let initial_total_rss = initial_rss.max(1e-12);

        let mut current_rss = initial_rss;

        // 2. Forward Pass: expand basis with reflected pairs
        while active_basis.len() + 2 <= self.params.max_terms {
            // Compute current residuals r_i for Fast MARS screening
            let (_c_curr, _) = fit_basis(&active_basis, x, y, weights, self.params.glm)?;
            let mut residuals = vec![0.0; n];
            for i in 0..n {
                let mut pred = 0.0;
                for (j, b) in active_basis.iter().enumerate() {
                    pred += _c_curr[j] * b.eval(&x[i]);
                }
                residuals[i] = y[i] - pred;
            }

            // Candidates list: (cand1, Option<cand2>, screening_score)
            let mut candidate_pool: Vec<(BasisFunction, Option<BasisFunction>, f64)> = Vec::new();

            // Search over all parent bases B_m
            for (_m, parent) in active_basis.iter().enumerate() {
                if parent.factors.len() >= self.params.degree {
                    continue;
                }
                let parent_features: Vec<usize> = parent.factors.iter().map(|f| f.feature).collect();

                // Search over all predictors j
                for j in 0..p {
                    if parent_features.contains(&j) {
                        continue;
                    }

                    // Check allowed interactions
                    if let Some(ref allowed_matrix) = self.params.allowed {
                        let mut ok = true;
                        for &feat_prev in &parent_features {
                            if !allowed_matrix[feat_prev][j] || !allowed_matrix[j][feat_prev] {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            continue;
                        }
                    }

                    // Linear-only predictor check
                    if self.params.linpreds.contains(&j) {
                        let mut cand = parent.clone();
                        cand.factors.push(HingeFactor::linear(j));

                        let score = eval_pair_fast_mars_score(&cand, None, x, &residuals, weights);
                        candidate_pool.push((cand, None, score));
                        continue;
                    }

                    // Extract and sort unique values for predictor j
                    let mut col_vals: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    col_vals.dedup();

                    let n_unique = col_vals.len();
                    if n_unique <= endspan * 2 {
                        continue;
                    }

                    let start_k = endspan;
                    let end_k = n_unique - endspan;
                    let max_candidates = 35;
                    let step = if n_unique > 100 {
                        (n_unique / max_candidates).max(minspan)
                    } else {
                        1
                    };

                    for k in (start_k..end_k).step_by(step) {
                        let knot = col_vals[k];
                        let mut cand1 = parent.clone();
                        cand1.factors.push(HingeFactor::hinge(j, knot, true));
                        let mut cand2 = parent.clone();
                        cand2.factors.push(HingeFactor::hinge(j, knot, false));

                        let total_score = eval_pair_fast_mars_score(&cand1, Some(&cand2), x, &residuals, weights);
                        candidate_pool.push((cand1, Some(cand2), total_score));
                    }
                }
            }

            if candidate_pool.is_empty() {
                break;
            }

            // Fast MARS: Sort candidates by screening score descending
            candidate_pool.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            let eval_candidates = if self.params.fast_k > 0 && candidate_pool.len() > self.params.fast_k {
                &candidate_pool[..self.params.fast_k]
            } else {
                &candidate_pool[..]
            };

            let mut best_gain = 0.0;
            let mut best_pair = None;

            for (c1, c2_opt, _) in eval_candidates {
                let mut test_basis = active_basis.clone();
                test_basis.push(c1.clone());
                if let Some(c2) = c2_opt {
                    test_basis.push(c2.clone());
                }

                if let Ok((_coef, rss_test)) = fit_basis(&test_basis, x, y, weights, self.params.glm) {
                    let gain = current_rss - rss_test;
                    if gain > best_gain {
                        best_gain = gain;
                        best_pair = Some((c1.clone(), c2_opt.clone(), rss_test));
                    }
                }
            }

            // Stopping threshold
            if best_gain < self.params.thresh * initial_total_rss || best_pair.is_none() {
                break;
            }

            if let Some((b1, b2_opt, new_rss)) = best_pair {
                active_basis.push(b1);
                if let Some(b2) = b2_opt {
                    active_basis.push(b2);
                }
                current_rss = new_rss;
            } else {
                break;
            }
        }

        // 3. Backward Pass: Pruning
        let (final_basis, final_coef, final_rss, final_gcv) = match self.params.pmethod {
            MarsPruningMethod::None => {
                let (c, r) = fit_basis(&active_basis, x, y, weights, self.params.glm)?;
                let g = compute_gcv(r, total_weight, active_basis.len(), self.params.penalty);
                (active_basis, c, r, g)
            }
            MarsPruningMethod::Backward => {
                let mut trajectory: Vec<(Vec<BasisFunction>, Vec<f64>, f64, f64)> = Vec::new();
                let (full_coef, full_rss) = fit_basis(&active_basis, x, y, weights, self.params.glm)?;
                let full_gcv = compute_gcv(full_rss, total_weight, active_basis.len(), self.params.penalty);
                trajectory.push((active_basis.clone(), full_coef, full_rss, full_gcv));

                let mut curr_basis = active_basis;

                while curr_basis.len() > 1 {
                    let mut best_del_idx = 1;
                    let mut best_del_rss = f64::INFINITY;
                    let mut best_del_coef = Vec::new();

                    for del_idx in 1..curr_basis.len() {
                        let mut candidate_basis = curr_basis.clone();
                        candidate_basis.remove(del_idx);

                        if let Ok((c, rss)) = fit_basis(&candidate_basis, x, y, weights, self.params.glm) {
                            if rss < best_del_rss {
                                best_del_rss = rss;
                                best_del_idx = del_idx;
                                best_del_coef = c;
                            }
                        }
                    }

                    if best_del_rss.is_infinite() {
                        break;
                    }

                    curr_basis.remove(best_del_idx);
                    let gcv = compute_gcv(best_del_rss, total_weight, curr_basis.len(), self.params.penalty);
                    trajectory.push((curr_basis.clone(), best_del_coef, best_del_rss, gcv));
                }

                trajectory
                    .into_iter()
                    .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| "Backward pruning trajectory empty".to_string())?
            }
            MarsPruningMethod::Cv(k_folds) => {
                // K-fold cross-validation pruning
                prune_by_cv(&active_basis, x, y, weights, self.params.glm, k_folds, self.params.penalty)?
            }
        };

        self.model = Some(MarsModel {
            params: self.params.clone(),
            basis: final_basis,
            coef: final_coef,
            rss: final_rss,
            gcv: final_gcv,
            n_train: n,
            n_features: p,
        });

        Ok(())
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| "MarsRegressor is not fitted".to_string())?;
        Ok(m.predict(x))
    }

    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let m = self
            .model
            .as_ref()
            .ok_or_else(|| "MarsRegressor is not fitted".to_string())?;
        m.predict_proba(x)
    }
}

/// Compute joint Fast MARS screening score for a reflected hinge pair (cand1, cand2)
/// adjusting for intercept shift: score = v^T G^{-1} v.
#[inline]
fn eval_pair_fast_mars_score(
    cand1: &BasisFunction,
    cand2_opt: Option<&BasisFunction>,
    x: &[Vec<f64>],
    residuals: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    let mut sum_w = 0.0;
    let mut sum_wb1 = 0.0;
    let mut sum_wb2 = 0.0;
    let mut sum_wb1_sq = 0.0;
    let mut sum_wb2_sq = 0.0;
    let mut sum_wb1b2 = 0.0;
    let mut dot1 = 0.0;
    let mut dot2 = 0.0;

    for (i, row) in x.iter().enumerate() {
        let w = weights.map_or(1.0, |ws| ws[i]);
        if w <= 0.0 {
            continue;
        }
        let b1 = cand1.eval(row);
        let b2 = cand2_opt.map_or(0.0, |c2| c2.eval(row));
        let r = residuals[i];

        sum_w += w;
        sum_wb1 += w * b1;
        sum_wb2 += w * b2;
        sum_wb1_sq += w * b1 * b1;
        sum_wb2_sq += w * b2 * b2;
        sum_wb1b2 += w * b1 * b2;
        dot1 += w * b1 * r;
        dot2 += w * b2 * r;
    }

    if sum_w <= 0.0 {
        return 0.0;
    }

    let g11 = (sum_wb1_sq - (sum_wb1 * sum_wb1) / sum_w).max(0.0);

    if cand2_opt.is_none() {
        // Single linear term
        if g11 > 1e-12 {
            return (dot1 * dot1) / g11;
        } else {
            return 0.0;
        }
    }

    let g22 = (sum_wb2_sq - (sum_wb2 * sum_wb2) / sum_w).max(0.0);
    let g12 = sum_wb1b2 - (sum_wb1 * sum_wb2) / sum_w;

    let det = g11 * g22 - g12 * g12;
    if det > 1e-12 {
        // Analytic inverse of 2x2 Gram matrix: [v1, v2] * G^-1 * [v1, v2]^T
        (g22 * dot1 * dot1 + g11 * dot2 * dot2 - 2.0 * g12 * dot1 * dot2) / det
    } else if g11 > 1e-12 {
        (dot1 * dot1) / g11
    } else {
        0.0
    }
}

/// Compute GCV(M) = (RSS / N_w) / (1 - C(M) / N_w)^2
/// where C(M) = M + penalty * (M - 1) / 2
#[inline]
fn compute_gcv(rss: f64, total_weight: f64, m: usize, penalty: f64) -> f64 {
    let n_f = total_weight;
    let m_f = m as f64;
    let c = m_f + penalty * (m_f - 1.0).max(0.0) / 2.0;
    let denom = (1.0 - c / n_f).max(1e-4);
    (rss / n_f) / (denom * denom)
}

/// Fit basis functions under given GLM family (Gaussian via OLS, Binomial/Poisson via IRLS).
fn fit_basis(
    basis: &[BasisFunction],
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
    glm: MarsGlmFamily,
) -> Result<(Vec<f64>, f64), String> {
    match glm {
        MarsGlmFamily::Gaussian => fit_weighted_ols(basis, x, y, weights),
        MarsGlmFamily::Binomial => fit_irls_binomial(basis, x, y, weights),
        MarsGlmFamily::Poisson => fit_irls_poisson(basis, x, y, weights),
    }
}

/// Fit Ordinary Least Squares with sample weights on evaluated basis functions.
fn fit_weighted_ols(
    basis: &[BasisFunction],
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, f64), String> {
    let k = basis.len();
    if k == 0 {
        return Err("No basis functions to fit".to_string());
    }

    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    let mut row_buf = vec![0.0; k];

    for (i, row) in x.iter().enumerate() {
        let yi = y[i];
        let wi = weights.map_or(1.0, |w| w[i]);
        if wi <= 0.0 {
            continue;
        }

        for (j, b) in basis.iter().enumerate() {
            row_buf[j] = b.eval(row);
        }

        let w_yi = wi * yi;
        for j1 in 0..k {
            let val1 = row_buf[j1];
            xty[j1] += val1 * w_yi;
            let w_val1 = wi * val1;
            for j2 in 0..k {
                xtx[j1 * k + j2] += w_val1 * row_buf[j2];
            }
        }
    }

    // Add ridge lambda = 1e-7 on non-intercept diagonal to handle collinear hinge pairs
    for j in 1..k {
        xtx[j * k + j] += 1e-7;
    }

    let a_mat = faer::Mat::from_fn(k, k, |i, j| xtx[i * k + j]);
    let b_mat = faer::Mat::from_fn(k, 1, |i, _| xty[i]);

    let coef: Vec<f64> = if let Ok(llt) = a_mat.as_ref().llt(faer::Side::Lower) {
        let sol = llt.solve(b_mat.as_ref());
        (0..k).map(|i| sol[(i, 0)]).collect()
    } else {
        let qr = a_mat.as_ref().col_piv_qr();
        let sol = qr.solve(b_mat.as_ref());
        (0..k).map(|i| sol[(i, 0)]).collect()
    };

    // Calculate weighted RSS
    let mut rss = 0.0;
    for (i, row) in x.iter().enumerate() {
        let wi = weights.map_or(1.0, |w| w[i]);
        if wi <= 0.0 {
            continue;
        }
        let mut pred = 0.0;
        for (j, b) in basis.iter().enumerate() {
            pred += coef[j] * b.eval(row);
        }
        let diff = y[i] - pred;
        rss += wi * diff * diff;
    }

    Ok((coef, rss))
}

/// Fit Binomial logistic model via IRLS (Fisher scoring).
fn fit_irls_binomial(
    basis: &[BasisFunction],
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, f64), String> {
    let n = x.len();
    let k = basis.len();
    let mut coef = vec![0.0; k];

    // Initial log-odds from weighted mean
    let mut sum_y = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = weights.map_or(1.0, |ws| ws[i]);
        sum_y += w * y[i];
        sum_w += w;
    }
    let p_bar = (sum_y / sum_w).clamp(0.01, 0.99);
    coef[0] = (p_bar / (1.0 - p_bar)).ln();

    // IRLS iterations (3 to 8 typically converge)
    for _iter in 0..8 {
        let mut working_w = vec![0.0; n];
        let mut working_z = vec![0.0; n];

        for i in 0..n {
            let mut eta = 0.0;
            for (j, b) in basis.iter().enumerate() {
                eta += coef[j] * b.eval(&x[i]);
            }
            let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-5, 1.0 - 1e-5);
            let var = mu * (1.0 - mu);
            let base_w = weights.map_or(1.0, |ws| ws[i]);
            let w = base_w * var;
            working_w[i] = w;
            working_z[i] = eta + (y[i] - mu) / var;
        }

        if let Ok((new_c, _)) = fit_weighted_ols(basis, x, &working_z, Some(&working_w)) {
            let mut max_diff: f64 = 0.0;
            for j in 0..k {
                max_diff = max_diff.max((new_c[j] - coef[j]).abs());
            }
            coef = new_c;
            if max_diff < 1e-4 {
                break;
            }
        } else {
            break;
        }
    }

    // Deviance as RSS analog: 2 * sum [ y log(y/mu) + (1-y) log((1-y)/(1-mu)) ]
    let mut deviance = 0.0;
    for i in 0..n {
        let mut eta = 0.0;
        for (j, b) in basis.iter().enumerate() {
            eta += coef[j] * b.eval(&x[i]);
        }
        let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-6, 1.0 - 1e-6);
        let yi = y[i].clamp(0.0, 1.0);
        let base_w = weights.map_or(1.0, |ws| ws[i]);
        let term1 = if yi > 1e-6 { yi * (yi / mu).ln() } else { 0.0 };
        let term2 = if (1.0 - yi) > 1e-6 { (1.0 - yi) * ((1.0 - yi) / (1.0 - mu)).ln() } else { 0.0 };
        deviance += 2.0 * base_w * (term1 + term2);
    }

    Ok((coef, deviance))
}

/// Fit Poisson model via IRLS (log-link).
fn fit_irls_poisson(
    basis: &[BasisFunction],
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, f64), String> {
    let n = x.len();
    let k = basis.len();
    let mut coef = vec![0.0; k];

    let mut sum_y = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = weights.map_or(1.0, |ws| ws[i]);
        sum_y += w * y[i];
        sum_w += w;
    }
    let y_bar = (sum_y / sum_w).max(0.1);
    coef[0] = y_bar.ln();

    for _iter in 0..8 {
        let mut working_w = vec![0.0; n];
        let mut working_z = vec![0.0; n];

        for i in 0..n {
            let mut eta = 0.0;
            for (j, b) in basis.iter().enumerate() {
                eta += coef[j] * b.eval(&x[i]);
            }
            let mu = eta.clamp(-15.0, 15.0).exp().max(1e-5);
            let base_w = weights.map_or(1.0, |ws| ws[i]);
            working_w[i] = base_w * mu;
            working_z[i] = eta + (y[i] - mu) / mu;
        }

        if let Ok((new_c, _)) = fit_weighted_ols(basis, x, &working_z, Some(&working_w)) {
            let mut max_diff: f64 = 0.0;
            for j in 0..k {
                max_diff = max_diff.max((new_c[j] - coef[j]).abs());
            }
            coef = new_c;
            if max_diff < 1e-4 {
                break;
            }
        } else {
            break;
        }
    }

    let mut deviance = 0.0;
    for i in 0..n {
        let mut eta = 0.0;
        for (j, b) in basis.iter().enumerate() {
            eta += coef[j] * b.eval(&x[i]);
        }
        let mu = eta.clamp(-15.0, 15.0).exp().max(1e-6);
        let yi = y[i].max(0.0);
        let base_w = weights.map_or(1.0, |ws| ws[i]);
        let term = if yi > 1e-6 { yi * (yi / mu).ln() - (yi - mu) } else { mu };
        deviance += 2.0 * base_w * term;
    }

    Ok((coef, deviance))
}

/// Prune basis by K-fold cross-validation.
fn prune_by_cv(
    active_basis: &[BasisFunction],
    x: &[Vec<f64>],
    y: &[f64],
    weights: Option<&[f64]>,
    glm: MarsGlmFamily,
    k_folds: usize,
    penalty: f64,
) -> Result<(Vec<BasisFunction>, Vec<f64>, f64, f64), String> {
    let n = x.len();
    let folds = k_folds.clamp(2, 20);

    // Backward elimination sequence on full data
    let mut seq: Vec<Vec<BasisFunction>> = Vec::new();
    let mut curr = active_basis.to_vec();
    seq.push(curr.clone());

    while curr.len() > 1 {
        let mut best_del = 1;
        let mut best_del_rss = f64::INFINITY;

        for del_idx in 1..curr.len() {
            let mut cand = curr.clone();
            cand.remove(del_idx);
            if let Ok((_, rss)) = fit_basis(&cand, x, y, weights, glm) {
                if rss < best_del_rss {
                    best_del_rss = rss;
                    best_del = del_idx;
                }
            }
        }
        if best_del_rss.is_infinite() {
            break;
        }
        curr.remove(best_del);
        seq.push(curr.clone());
    }

    // Evaluate CV error for each model size in sequence
    let mut cv_errors = vec![0.0; seq.len()];
    for f in 0..folds {
        let val_start = f * n / folds;
        let val_end = (f + 1) * n / folds;

        let mut x_train = Vec::with_capacity(n - (val_end - val_start));
        let mut y_train = Vec::with_capacity(n - (val_end - val_start));
        let mut w_train_buf = Vec::new();

        let mut x_val = Vec::with_capacity(val_end - val_start);
        let mut y_val = Vec::with_capacity(val_end - val_start);
        let mut w_val_buf = Vec::new();

        for i in 0..n {
            let is_val = i >= val_start && i < val_end;
            let w = weights.map_or(1.0, |ws| ws[i]);
            if is_val {
                x_val.push(x[i].clone());
                y_val.push(y[i]);
                w_val_buf.push(w);
            } else {
                x_train.push(x[i].clone());
                y_train.push(y[i]);
                w_train_buf.push(w);
            }
        }

        let w_train_opt = if weights.is_some() { Some(&w_train_buf[..]) } else { None };

        for (m_idx, basis_sub) in seq.iter().enumerate() {
            if let Ok((coef, _)) = fit_basis(basis_sub, &x_train, &y_train, w_train_opt, glm) {
                for (vi, vrow) in x_val.iter().enumerate() {
                    let mut pred = 0.0;
                    for (bj, b) in basis_sub.iter().enumerate() {
                        pred += coef[bj] * b.eval(vrow);
                    }
                    let diff = y_val[vi] - pred;
                    cv_errors[m_idx] += w_val_buf[vi] * diff * diff;
                }
            } else {
                cv_errors[m_idx] += 1e9;
            }
        }
    }

    let best_m_idx = cv_errors
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_basis = seq[best_m_idx].clone();
    let (best_coef, best_rss) = fit_basis(&best_basis, x, y, weights, glm)?;
    let total_w: f64 = match weights {
        Some(w) => w.iter().sum(),
        None => n as f64,
    };
    let best_gcv = compute_gcv(best_rss, total_w, best_basis.len(), penalty);

    Ok((best_basis, best_coef, best_rss, best_gcv))
}
