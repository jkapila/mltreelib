//! Multivariate Adaptive Regression Splines (MARS / Friedman 1991, `earth` compatible).
//!
//! Replaces discontinuous step functions with reflected hinge basis pairs:
//! `h(x - t) = max(0, x - t)` and `h(t - x) = max(0, t - x)`.
//! Supports multi-variable interactions up to `degree`, forward basis expansion,
//! and backward Generalized Cross Validation (GCV) pruning.

use faer::prelude::*;

/// A single piecewise-linear hinge factor: max(0, x_j - knot) or max(0, knot - x_j).
#[derive(Clone, Debug, PartialEq)]
pub struct HingeFactor {
    pub feature: usize,
    pub knot: f64,
    /// If true: max(0, x - knot); if false: max(0, knot - x).
    pub is_forward: bool,
}

impl HingeFactor {
    #[inline]
    pub fn eval(&self, val: f64) -> f64 {
        if self.is_forward {
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
        if self.is_forward {
            format!("h({} - {:.4})", name, self.knot)
        } else {
            format!("h({:.4} - {})", self.knot, name)
        }
    }
}

/// A basis function: product of zero or more hinge factors (empty = intercept 1.0).
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
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let mut pred = 0.0;
        for (b, &c) in self.basis.iter().zip(self.coef.iter()) {
            pred += c * b.eval(row);
        }
        pred
    }

    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|r| self.predict_row(r)).collect()
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
        terms.join(" ")
    }
}

/// Standalone MARS regressor.
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
        let (_initial_coef, initial_rss) = fit_ols(&active_basis, x, y)?;
        let initial_total_rss = initial_rss.max(1e-12);

        let mut current_rss = initial_rss;

        // 2. Forward Pass: expand basis with reflected pairs
        while active_basis.len() + 2 <= self.params.max_terms {
            let mut best_gain = 0.0;
            let mut best_pair = None;

            // Search over all parent bases B_m
            for (_m, parent) in active_basis.iter().enumerate() {
                if parent.factors.len() >= self.params.degree {
                    continue;
                }
                // Existing features in parent
                let parent_features: Vec<usize> = parent.factors.iter().map(|f| f.feature).collect();

                // Search over all predictors j
                for j in 0..p {
                    if parent_features.contains(&j) {
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

                    // Scan candidate knots (cap to ~35 candidate knots per feature for scalable forward search,
                    // while retaining step=1 on small distinct grids for exact recovery).
                    let start_k = endspan;
                    let end_k = n_unique - endspan;
                    let max_candidates = 35;
                    let step = if n_unique > max_candidates {
                        (n_unique / max_candidates).max(minspan)
                    } else {
                        1
                    };

                    for k in (start_k..end_k).step_by(step) {
                        let knot = col_vals[k];
                        let mut cand1 = parent.clone();
                        cand1.factors.push(HingeFactor {
                            feature: j,
                            knot,
                            is_forward: true,
                        });
                        let mut cand2 = parent.clone();
                        cand2.factors.push(HingeFactor {
                            feature: j,
                            knot,
                            is_forward: false,
                        });

                        // Candidate basis set
                        let mut test_basis = active_basis.clone();
                        test_basis.push(cand1.clone());
                        test_basis.push(cand2.clone());

                        if let Ok((_coef, rss_test)) = fit_ols(&test_basis, x, y) {
                            let gain = current_rss - rss_test;
                            if gain > best_gain {
                                best_gain = gain;
                                best_pair = Some((cand1, cand2, rss_test));
                            }
                        }
                    }
                }
            }

            // Stopping threshold
            if best_gain < self.params.thresh * initial_total_rss || best_pair.is_none() {
                break;
            }

            if let Some((b1, b2, new_rss)) = best_pair {
                active_basis.push(b1);
                active_basis.push(b2);
                current_rss = new_rss;
            } else {
                break;
            }
        }

        // 3. Backward Pass: GCV Pruning
        let mut trajectory: Vec<(Vec<BasisFunction>, Vec<f64>, f64, f64)> = Vec::new();
        let (full_coef, full_rss) = fit_ols(&active_basis, x, y)?;
        let full_gcv = compute_gcv(full_rss, n, active_basis.len(), self.params.penalty);
        trajectory.push((active_basis.clone(), full_coef, full_rss, full_gcv));

        let mut curr_basis = active_basis;

        while curr_basis.len() > 1 {
            let mut best_del_idx = 1;
            let mut best_del_rss = f64::INFINITY;
            let mut best_del_coef = Vec::new();

            // Try removing each basis function except intercept (index 0)
            for del_idx in 1..curr_basis.len() {
                let mut candidate_basis = curr_basis.clone();
                candidate_basis.remove(del_idx);

                if let Ok((c, rss)) = fit_ols(&candidate_basis, x, y) {
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
            let gcv = compute_gcv(best_del_rss, n, curr_basis.len(), self.params.penalty);
            trajectory.push((curr_basis.clone(), best_del_coef, best_del_rss, gcv));
        }

        // Select the model minimizing GCV along the trajectory
        let (best_basis, best_coef, best_rss, best_gcv) = trajectory
            .into_iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| "Backward pruning trajectory empty".to_string())?;

        self.model = Some(MarsModel {
            params: self.params.clone(),
            basis: best_basis,
            coef: best_coef,
            rss: best_rss,
            gcv: best_gcv,
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
}

/// Compute GCV(M) = (RSS / N) / (1 - C(M) / N)^2
/// where C(M) = M + penalty * (M - 1) / 2
#[inline]
fn compute_gcv(rss: f64, n: usize, m: usize, penalty: f64) -> f64 {
    let n_f = n as f64;
    let m_f = m as f64;
    let c = m_f + penalty * (m_f - 1.0).max(0.0) / 2.0;
    let denom = (1.0 - c / n_f).max(1e-4);
    (rss / n_f) / (denom * denom)
}

/// Fit Ordinary Least Squares on the evaluated design matrix of basis functions.
fn fit_ols(basis: &[BasisFunction], x: &[Vec<f64>], y: &[f64]) -> Result<(Vec<f64>, f64), String> {
    let _n = x.len();
    let k = basis.len();
    if k == 0 {
        return Err("No basis functions to fit".to_string());
    }

    // Evaluate design matrix X [N x K]
    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];

    let mut row_buf = vec![0.0; k];
    for (i, row) in x.iter().enumerate() {
        let yi = y[i];
        for (j, b) in basis.iter().enumerate() {
            row_buf[j] = b.eval(row);
        }
        for j1 in 0..k {
            let val1 = row_buf[j1];
            xty[j1] += val1 * yi;
            for j2 in 0..k {
                xtx[j1 * k + j2] += val1 * row_buf[j2];
            }
        }
    }

    // Add small ridge lambda = 1e-7 on non-intercept diagonal to handle near-collinearity
    for j in 1..k {
        xtx[j * k + j] += 1e-7;
    }

    // Solve via faer Cholesky or ColPivQr
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

    // Calculate exact Residual Sum of Squares
    let mut rss = 0.0;
    for (i, row) in x.iter().enumerate() {
        let mut pred = 0.0;
        for (j, b) in basis.iter().enumerate() {
            pred += coef[j] * b.eval(row);
        }
        let diff = y[i] - pred;
        rss += diff * diff;
    }

    Ok((coef, rss))
}
