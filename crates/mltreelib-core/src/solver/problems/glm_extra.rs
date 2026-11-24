//! Generalized linear model extensions: Negative Binomial (NB2) and Beta regression (Spec 27 Phase 7).

use crate::family::std_norm_cdf;
use crate::solver::design::DesignMatrix;
use crate::solver::problems::nnls::solve_cholesky;

/// Result of Negative Binomial regression with dispersion estimation.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NegativeBinomialFitResult {
    /// Fitted coefficient vector (on log link scale).
    pub coefficients: Vec<f64>,
    /// Estimated dispersion parameter $\alpha \ge 0$ ($V(\mu) = \mu + \alpha \mu^2$).
    pub alpha: f64,
    /// Cameron-Trivedi test p-value testing $H_0: \alpha = 0$ (Poisson) vs $H_1: \alpha > 0$.
    pub dispersion_p_value: f64,
    /// Final negative binomial deviance.
    pub deviance: f64,
    /// Total outer iterations performed.
    pub iterations: usize,
}

/// Result of Beta regression with precision parameter estimation.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BetaFitResult {
    /// Fitted coefficient vector (on logit link scale).
    pub coefficients: Vec<f64>,
    /// Estimated precision parameter $\phi > 0$ ($V(\mu) = \frac{\mu(1-\mu)}{1+\phi}$).
    pub phi: f64,
    /// Final beta deviance / quasi-deviance.
    pub deviance: f64,
    /// Total outer iterations performed.
    pub iterations: usize,
}

/// Evaluates unit negative binomial (NB2) deviance for target $y$, mean $\mu > 0$, and dispersion $\alpha \ge 0$:
///
/// $$d(y, \mu; \alpha) = 2 \left[ y \ln\left(\frac{\max(y, 10^{-10})}{\mu}\right) - \left(y + \frac{1}{\alpha}\right) \ln\left(\frac{1 + \alpha y}{1 + \alpha \mu}\right) \right]$$
///
/// When $\alpha \to 0$ (boundary $\alpha \le 10^{-6}$), smoothly transitions to Poisson deviance.
#[inline]
pub fn nb2_unit_deviance(y: f64, mean: f64, alpha: f64) -> f64 {
    let mean = mean.max(1e-12);
    if alpha <= 1e-6 {
        // Poisson boundary
        if y <= 0.0 {
            2.0 * mean
        } else {
            2.0 * (y * (y.ln() - mean.ln()) - (y - mean)).max(0.0)
        }
    } else {
        let inv_a = 1.0 / alpha;
        if y <= 0.0 {
            2.0 * inv_a * (1.0 + alpha * mean).ln()
        } else {
            let t1 = y * (y.ln() - mean.ln());
            let t2 = (y + inv_a) * ((1.0 + alpha * y).ln() - (1.0 + alpha * mean).ln());
            2.0 * (t1 - t2).max(0.0)
        }
    }
}

/// Evaluates total negative binomial deviance over a [`DesignMatrix`].
pub fn eval_nb2_deviance<D: DesignMatrix>(
    design: &D,
    beta: &[f64],
    alpha: f64,
    l2: f64,
) -> f64 {
    let n = design.n_rows();
    let mut total_dev = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, beta).clamp(-30.0, 30.0);
            let mu = eta.exp();
            let y = design.get_y(i);
            total_dev += w * nb2_unit_deviance(y, mu, alpha);
        }
    }
    if l2 > 0.0 {
        let mut penalty = 0.0;
        for &v in beta.iter().skip(1) {
            penalty += v * v;
        }
        total_dev += l2 * penalty;
    }
    total_dev
}

/// Fits Negative Binomial (NB2) regression via alternating IRLS and method-of-moments dispersion estimation.
pub fn solve_negative_binomial_regression<D: DesignMatrix>(
    design: &D,
    initial_alpha: Option<f64>,
    l2: f64,
    max_outer_iter: usize,
    tol: f64,
) -> Result<NegativeBinomialFitResult, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for Negative Binomial regression".into());
    }

    let tol = if tol > 0.0 { tol } else { 1e-5 };
    let max_outer = if max_outer_iter > 0 { max_outer_iter } else { 15 };

    // Initial estimate of intercept: weighted log mean
    let mut sum_wy = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        let y = design.get_y(i);
        if w > 0.0 && y >= 0.0 {
            sum_wy += w * y;
            sum_w += w;
        }
    }
    if sum_w <= 0.0 {
        return Err("zero total weight in design".into());
    }
    let mean_y = (sum_wy / sum_w).max(1e-4);

    let mut beta = vec![0.0; p];
    beta[0] = mean_y.ln();

    // Initial alpha: from parameter or estimated from Poisson variance
    let mut alpha = initial_alpha.unwrap_or_else(|| {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            let y = design.get_y(i);
            if w > 0.0 {
                let diff = y - mean_y;
                num += w * (diff * diff - mean_y);
                den += w * (mean_y * mean_y);
            }
        }
        if den > 0.0 {
            (num / den).clamp(0.0, 10.0)
        } else {
            0.0
        }
    });

    let mut row_buf = vec![0.0; p];
    let mut total_iter = 0;

    // Feature standardization for numerical stability across unnormalized tabular columns
    let mut means = vec![0.0; p];
    let mut stds = vec![1.0; p];
    if p > 1 {
        let mut row_tmp = vec![0.0; p];
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                design.copy_design_row(i, &mut row_tmp);
                for j in 1..p {
                    means[j] += w * row_tmp[j];
                }
            }
        }
        for j in 1..p {
            means[j] /= sum_w;
        }
        let mut var_accum = vec![0.0; p];
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                design.copy_design_row(i, &mut row_tmp);
                for j in 1..p {
                    let d = row_tmp[j] - means[j];
                    var_accum[j] += w * d * d;
                }
            }
        }
        for j in 1..p {
            let s = (var_accum[j] / sum_w).sqrt();
            stds[j] = if s > 1e-6 { s } else { 1.0 };
        }
    }

    for outer in 0..max_outer {
        total_iter = outer + 1;

        // IRLS inner loop for beta given current alpha
        for _irls_iter in 0..20 {
            let mut g = vec![vec![0.0; p]; p];
            let mut c = vec![0.0; p];

            for i in 0..n {
                let w = design.get_weight(i);
                if w > 0.0 {
                    design.copy_design_row(i, &mut row_buf);
                    for j in 1..p {
                        row_buf[j] = (row_buf[j] - means[j]) / stds[j];
                    }
                    let mut eta = beta[0];
                    for j in 1..p {
                        eta += beta[j] * row_buf[j];
                    }
                    let eta = eta.clamp(-25.0, 25.0);
                    let mu = eta.exp();
                    let y = design.get_y(i);

                    // Working weight W_i = w_i * mu / (1 + alpha * mu)
                    let w_work = (w * mu) / (1.0 + alpha * mu).max(1e-12);
                    // Working response z_i = eta + (y - mu) / mu
                    let z_work = eta + (y - mu) / mu.max(1e-12);

                    for j in 0..p {
                        let w_xj = w_work * row_buf[j];
                        c[j] += w_xj * z_work;
                        for k in 0..=j {
                            let val = w_xj * row_buf[k];
                            g[j][k] += val;
                            if j != k {
                                g[k][j] += val;
                            }
                        }
                    }
                }
            }

            // Ridge stabilization: always small ridge on slopes for small/collinear tree leaves
            let slope_ridge = (l2 + 1e-4).max(1e-4);
            for j in 1..p {
                g[j][j] += slope_ridge;
            }
            g[0][0] += 1e-6;

            let new_beta = match solve_cholesky(&g, &c) {
                Some(b) => b,
                None => break,
            };

            let mut diff = 0.0f64;
            for j in 0..p {
                let step = (new_beta[j] - beta[j]).clamp(-2.5, 2.5);
                let updated = beta[j] + step;
                let d = (updated - beta[j]).abs() / (1.0 + beta[j].abs());
                diff = diff.max(d);
                beta[j] = updated;
            }
            if diff < tol {
                break;
            }
        }

        // Update dispersion alpha using Pearson chi-square / method of moments
        if initial_alpha.is_none() {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let w = design.get_weight(i);
                if w > 0.0 {
                    design.copy_design_row(i, &mut row_buf);
                    for j in 1..p {
                        row_buf[j] = (row_buf[j] - means[j]) / stds[j];
                    }
                    let mut eta = beta[0];
                    for j in 1..p {
                        eta += beta[j] * row_buf[j];
                    }
                    let eta = eta.clamp(-30.0, 30.0);
                    let mu = eta.exp();
                    let y = design.get_y(i);
                    let diff = y - mu;
                    num += w * (diff * diff - mu);
                    den += w * (mu * mu);
                }
            }
            let new_alpha = if den > 0.0 {
                (num / den).clamp(0.0, 50.0)
            } else {
                0.0
            };

            if (new_alpha - alpha).abs() / (1.0 + alpha) < tol {
                alpha = new_alpha;
                break;
            }
            alpha = new_alpha;
        } else {
            break;
        }
    }

    // Cameron-Trivedi test for overdispersion against Poisson
    let mut num_test = 0.0;
    let mut den_test = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, &beta).clamp(-30.0, 30.0);
            let mu = eta.exp();
            let y = design.get_y(i);
            num_test += w * ((y - mu) * (y - mu) - y) * mu;
            den_test += w * 2.0 * mu.powi(4);
        }
    }
    let z_score = if den_test > 1e-12 {
        num_test / den_test.sqrt()
    } else {
        0.0
    };
    let dispersion_p_value = (1.0 - std_norm_cdf(z_score)).clamp(0.0, 1.0);

    // Unstandardize coefficients back to natural design scale
    for j in 1..p {
        let natural_coef = beta[j] / stds[j];
        beta[0] -= natural_coef * means[j];
        beta[j] = natural_coef;
    }

    let deviance = eval_nb2_deviance(design, &beta, alpha, l2);

    Ok(NegativeBinomialFitResult {
        coefficients: beta,
        alpha,
        dispersion_p_value,
        deviance,
        iterations: total_iter,
    })
}

/// Evaluates unit beta regression deviance / quasi-deviance for $y \in (0, 1)$ and $\mu \in (0, 1)$:
///
/// $$d(y, \mu; \phi) = 2 \left[ y \ln\left(\frac{y}{\mu}\right) + (1 - y) \ln\left(\frac{1 - y}{1 - \mu}\right) \right]$$
#[inline]
pub fn beta_unit_deviance(y: f64, mean: f64) -> f64 {
    let y = y.clamp(1e-5, 1.0 - 1e-5);
    let mean = mean.clamp(1e-5, 1.0 - 1e-5);
    let term1 = y * (y.ln() - mean.ln());
    let term2 = (1.0 - y) * ((1.0 - y).ln() - (1.0 - mean).ln());
    2.0 * (term1 + term2).max(0.0)
}

/// Evaluates total beta deviance over a [`DesignMatrix`].
pub fn eval_beta_deviance<D: DesignMatrix>(
    design: &D,
    beta: &[f64],
    l2: f64,
) -> f64 {
    let n = design.n_rows();
    let mut total_dev = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, beta);
            let mu = 1.0 / (1.0 + (-eta).exp());
            let y = design.get_y(i);
            total_dev += w * beta_unit_deviance(y, mu);
        }
    }
    if l2 > 0.0 {
        let mut penalty = 0.0;
        for &v in beta.iter().skip(1) {
            penalty += v * v;
        }
        total_dev += l2 * penalty;
    }
    total_dev
}

/// Fits Beta regression for bounded responses $y \in (0, 1)$ with logit link.
pub fn solve_beta_regression<D: DesignMatrix>(
    design: &D,
    initial_phi: Option<f64>,
    l2: f64,
    max_outer_iter: usize,
    tol: f64,
) -> Result<BetaFitResult, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for Beta regression".into());
    }

    let tol = if tol > 0.0 { tol } else { 1e-5 };
    let max_outer = if max_outer_iter > 0 { max_outer_iter } else { 15 };

    // Initial intercept from clamped mean of y
    let mut sum_wy = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        let y = design.get_y(i).clamp(1e-4, 1.0 - 1e-4);
        if w > 0.0 {
            sum_wy += w * y;
            sum_w += w;
        }
    }
    if sum_w <= 0.0 {
        return Err("zero total weight in design".into());
    }
    let mean_y = (sum_wy / sum_w).clamp(1e-4, 1.0 - 1e-4);

    let mut beta = vec![0.0; p];
    beta[0] = (mean_y / (1.0 - mean_y)).ln();

    // Initial precision phi from method of moments
    let mut phi = initial_phi.unwrap_or_else(|| {
        let mut var_num = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            let y = design.get_y(i).clamp(1e-4, 1.0 - 1e-4);
            if w > 0.0 {
                let diff = y - mean_y;
                var_num += w * diff * diff;
            }
        }
        let var_y = (var_num / sum_w).max(1e-6);
        let est_phi = (mean_y * (1.0 - mean_y) / var_y) - 1.0;
        est_phi.clamp(0.5, 1000.0)
    });

    let mut row_buf = vec![0.0; p];
    let mut total_iter = 0;

    // Feature standardization for numerical stability across unnormalized tabular columns
    let mut means = vec![0.0; p];
    let mut stds = vec![1.0; p];
    if p > 1 {
        let mut row_tmp = vec![0.0; p];
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                design.copy_design_row(i, &mut row_tmp);
                for j in 1..p {
                    means[j] += w * row_tmp[j];
                }
            }
        }
        for j in 1..p {
            means[j] /= sum_w;
        }
        let mut var_accum = vec![0.0; p];
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                design.copy_design_row(i, &mut row_tmp);
                for j in 1..p {
                    let d = row_tmp[j] - means[j];
                    var_accum[j] += w * d * d;
                }
            }
        }
        for j in 1..p {
            let s = (var_accum[j] / sum_w).sqrt();
            stds[j] = if s > 1e-6 { s } else { 1.0 };
        }
    }

    for outer in 0..max_outer {
        total_iter = outer + 1;

        // IRLS for beta with logit link
        for _irls_iter in 0..20 {
            let mut g = vec![vec![0.0; p]; p];
            let mut c = vec![0.0; p];

            for i in 0..n {
                let w = design.get_weight(i);
                if w > 0.0 {
                    design.copy_design_row(i, &mut row_buf);
                    for j in 1..p {
                        row_buf[j] = (row_buf[j] - means[j]) / stds[j];
                    }
                    let mut eta = beta[0];
                    for j in 1..p {
                        eta += beta[j] * row_buf[j];
                    }
                    let eta = eta.clamp(-25.0, 25.0);
                    let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-5, 1.0 - 1e-5);
                    let y = design.get_y(i).clamp(1e-5, 1.0 - 1e-5);

                    // Working weight for Beta regression logit link
                    let var_eta = mu * (1.0 - mu);
                    let w_work = (w * var_eta).max(1e-10);
                    // Working response: z_i = eta + (y - mu) / var_eta
                    let z_work = eta + (y - mu) / var_eta.max(1e-10);

                    for j in 0..p {
                        let w_xj = w_work * row_buf[j];
                        c[j] += w_xj * z_work;
                        for k in 0..=j {
                            let val = w_xj * row_buf[k];
                            g[j][k] += val;
                            if j != k {
                                g[k][j] += val;
                            }
                        }
                    }
                }
            }

            // Ridge stabilization: always small ridge on slopes for small/collinear tree leaves
            let slope_ridge = (l2 + 1e-4).max(1e-4);
            for j in 1..p {
                g[j][j] += slope_ridge;
            }
            g[0][0] += 1e-6;

            let new_beta = match solve_cholesky(&g, &c) {
                Some(b) => b,
                None => break,
            };

            let mut diff = 0.0f64;
            for j in 0..p {
                let step = (new_beta[j] - beta[j]).clamp(-2.5, 2.5);
                let updated = beta[j] + step;
                let d = (updated - beta[j]).abs() / (1.0 + beta[j].abs());
                diff = diff.max(d);
                beta[j] = updated;
            }
            if diff < tol {
                break;
            }
        }

        // Precision update from residual variance
        if initial_phi.is_none() {
            let mut var_num = 0.0;
            let mut theo_var_num = 0.0;
            for i in 0..n {
                let w = design.get_weight(i);
                if w > 0.0 {
                    design.copy_design_row(i, &mut row_buf);
                    for j in 1..p {
                        row_buf[j] = (row_buf[j] - means[j]) / stds[j];
                    }
                    let mut eta = beta[0];
                    for j in 1..p {
                        eta += beta[j] * row_buf[j];
                    }
                    let eta = eta.clamp(-25.0, 25.0);
                    let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-5, 1.0 - 1e-5);
                    let y = design.get_y(i).clamp(1e-5, 1.0 - 1e-5);
                    let diff = y - mu;
                    var_num += w * diff * diff;
                    theo_var_num += w * mu * (1.0 - mu);
                }
            }
            let obs_var = (var_num / sum_w).max(1e-6);
            let theo_var = (theo_var_num / sum_w).max(1e-6);
            let new_phi = ((theo_var / obs_var) - 1.0).clamp(0.5, 2000.0);

            if (new_phi - phi).abs() / (1.0 + phi) < tol {
                phi = new_phi;
                break;
            }
            phi = new_phi;
        } else {
            break;
        }
    }

    // Unstandardize coefficients back to natural design scale
    for j in 1..p {
        let natural_coef = beta[j] / stds[j];
        beta[0] -= natural_coef * means[j];
        beta[j] = natural_coef;
    }

    let deviance = eval_beta_deviance(design, &beta, l2);

    Ok(BetaFitResult {
        coefficients: beta,
        phi,
        deviance,
        iterations: total_iter,
    })
}
