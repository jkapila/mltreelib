//! Direct Cholesky and rank-revealing Normal Equations Solver.
//!
//! High-speed exact solver for OLS and Ridge regression ($D \le 100$).
//! Produces exact parameter covariance $(X^T W X)^{-1}$ and hypothesis testing metrics.

use crate::solver::design::DesignMatrix;
use crate::solver::summary::{CoeffStats, GlmSummary};
use faer::prelude::Solve;

/// Fits OLS or Ridge linear regression using exact Cholesky decomposition with rank-revealing fallback.
pub fn solve_normal<D: DesignMatrix>(
    design: &D,
    l2: f64,
    feature_names: Option<&[String]>,
) -> Result<(Vec<f64>, Option<GlmSummary>), String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("Empty design matrix".to_string());
    }

    let gram = design.compute_gram(None);
    let mut m = faer::Mat::<f64>::zeros(p, p);
    let mut rhs = faer::Mat::<f64>::zeros(p, 1);

    for i in 0..p {
        rhs[(i, 0)] = gram.xty[i];
        for j in 0..p {
            let mut val = gram.xtx[i * p + j];
            if i == j && i > 0 && l2 > 0.0 {
                val += l2;
            }
            m[(i, j)] = val;
        }
    }

    // Try Cholesky decomposition (fast path for strictly positive definite Gram)
    let (sol, inv_opt) = match m.as_ref().llt(faer::Side::Lower) {
        Ok(llt) => {
            let x = llt.solve(rhs.as_ref());
            let sol_vec: Vec<f64> = (0..p).map(|i| x[(i, 0)]).collect();

            // Compute exact inverse for standard errors
            let mut identity = faer::Mat::<f64>::zeros(p, p);
            for i in 0..p {
                identity[(i, i)] = 1.0;
            }
            let inv = llt.solve(identity.as_ref());
            (sol_vec, Some(inv))
        }
        Err(_) => {
            let mut scratch = crate::node_stats::SolveScratch::new(p);
            crate::node_stats::solve_normal(
                &gram.xtx,
                &gram.xty,
                gram.yty,
                p,
                0.0,
                l2,
                &mut scratch,
            );
            (scratch.coefficients().to_vec(), None)
        }
    };

    // Hypothesis testing & Goodness-of-fit summary
    let n = gram.n;
    let summary = if n > p {
        if let Some(inv) = inv_opt {
            let df_residuals = n - p;
            let mut beta_xtx_beta = 0.0;
            for i in 0..p {
                let mut row_sum = 0.0;
                for j in 0..p {
                    row_sum += gram.xtx[i * p + j] * sol[j];
                }
                beta_xtx_beta += sol[i] * row_sum;
            }
            let mut beta_xty = 0.0;
            for i in 0..p {
                beta_xty += sol[i] * gram.xty[i];
            }
            let rss = (gram.yty - 2.0 * beta_xty + beta_xtx_beta).max(0.0);
            let sigma2 = (rss / (df_residuals as f64)).max(1e-12);

            let y_mean = gram.wy / gram.ws;
            let tss = (gram.yty - gram.ws * y_mean * y_mean).max(1e-12);
            let r_squared = (1.0 - rss / tss).clamp(0.0, 1.0);
            let adj_r_squared = (1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df_residuals as f64)).max(0.0);

            let p_pred = p.saturating_sub(1);
            let f_statistic = if p_pred > 0 && rss > 1e-12 {
                ((tss - rss) / (p_pred as f64)) / (rss / (df_residuals as f64))
            } else {
                0.0
            };
            let f_p_value = crate::stats::f_sf(f_statistic, p_pred as u32, df_residuals as u32);

            let mut coefficients = Vec::with_capacity(p);
            for i in 0..p {
                let var_beta = (inv[(i, i)] * sigma2).max(1e-12);
                let std_err = var_beta.sqrt();
                let est = sol[i];
                let t_val = if std_err > 1e-12 { est / std_err } else { 0.0 };
                let p_val = GlmSummary::p_value_t(t_val, df_residuals);
                let name = match feature_names.and_then(|names| names.get(i)) {
                    Some(s) => s.clone(),
                    None => if i == 0 { "(Intercept)".to_string() } else { format!("x{}", i) },
                };
                coefficients.push(CoeffStats {
                    name,
                    estimate: est,
                    std_err,
                    stat_value: t_val,
                    p_value: p_val,
                });
            }

            Some(GlmSummary {
                is_gaussian: true,
                n_obs: n,
                df_null: n - 1,
                df_residual: df_residuals,
                null_deviance: tss,
                residual_deviance: rss,
                log_likelihood: -0.5 * (n as f64) * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * rss / sigma2,
                aic: 2.0 * (p as f64) + (n as f64) * ((2.0 * std::f64::consts::PI * rss / (n as f64)).ln() + 1.0),
                bic: (p as f64) * (n as f64).ln() + (n as f64) * ((2.0 * std::f64::consts::PI * rss / (n as f64)).ln() + 1.0),
                dispersion: sigma2,
                r_squared: Some(r_squared),
                adj_r_squared: Some(adj_r_squared),
                f_statistic: Some(f_statistic),
                f_p_value: Some(f_p_value),
                coefficients,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok((sol, summary))
}
