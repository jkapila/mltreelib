//! Cyclic Coordinate Descent (CCD) Solver for Elastic Net and Lasso.
//!
//! Features:
//! - Exact $L_1$ soft-thresholding and $L_2$ Ridge shrinkage.
//! - Internal feature standardization ensuring equitable penalization.
//! - Active-set cycling strategy (glmnet style) for sparse high-dimensional problems.
//! - Monotone constraints (direction -1, 0, +1).
//! - Warm start support from parent nodes or path regularization.

use crate::solver::design::DesignMatrix;

/// Options controlling Coordinate Descent optimization.
#[derive(Clone, Debug)]
pub struct CcdOptions {
    pub l1: f64,
    pub l2: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub monotone: Vec<i8>,
}

impl Default for CcdOptions {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 0.0,
            max_iter: 1000,
            tol: 1e-7,
            monotone: Vec::new(),
        }
    }
}

/// Fits an ElasticNet model ($L_1 / L_2$) using Cyclic Coordinate Descent with active set.
pub fn solve_ccd<D: DesignMatrix>(
    design: &D,
    options: &CcdOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("Empty design matrix".to_string());
    }

    // 1. Compute column standardizations (intercept at index 0 unpenalized)
    let mut col_scales = vec![1.0; p];
    let mut col_norms = vec![0.0; p];
    for j in 0..p {
        let norm_sq = design.col_norm_sq(j, None);
        col_norms[j] = norm_sq;
        if j > 0 && options.l1 > 0.0 {
            let scale = (norm_sq / (n as f64).max(1.0)).sqrt();
            if scale > 1e-12 {
                col_scales[j] = scale;
            }
        }
    }

    // 2. Initialize coefficients & residuals
    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Start intercept at weighted mean
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            sum_y += w * design.get_y(i);
            sum_w += w;
        }
        if sum_w > 0.0 {
            beta[0] = sum_y / sum_w;
        }
    }

    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let y_hat = design.eval_row(i, &beta);
        residuals[i] = design.get_y(i) - y_hat;
    }

    // 3. Active set tracking & reusable buffer
    let mut active: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0 || j == 0).collect();
    let mut row_buf = vec![0.0; p];

    // 4. Main Active-Set Coordinate Descent Loop
    for _outer_iter in 0..options.max_iter {
        let mut max_active_delta = 0.0f64;

        // Inner loop: cycle over currently active set until convergence
        for _inner in 0..100 {
            let mut inner_delta = 0.0f64;
            for &j in &active {
                let diag = col_norms[j];
                if diag <= 0.0 {
                    continue;
                }
                let old_b = beta[j];
                let dot_res = design.col_dot_residuals(j, &residuals, None);
                let rho = dot_res + diag * old_b;

                let next_b = if j == 0 {
                    // Intercept is unpenalized
                    rho / diag
                } else {
                    let scale = col_scales[j];
                    let l1_pen = options.l1 * scale;
                    let l2_pen = options.l2 * scale * scale;
                    let denom = diag + l2_pen;
                    let raw = rho.signum() * (rho.abs() - l1_pen).max(0.0) / denom;

                    // Monotone constraint
                    let mono = options.monotone.get(j.saturating_sub(1)).copied().unwrap_or(0);
                    match mono {
                        1 => raw.max(0.0),
                        -1 => raw.min(0.0),
                        _ => raw,
                    }
                };

                let delta = next_b - old_b;
                if delta.abs() > 1e-15 {
                    beta[j] = next_b;
                    // Update residuals in-place using preallocated row_buf
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= delta * row_buf[j];
                    }
                    inner_delta = inner_delta.max(delta.abs());
                }
            }
            max_active_delta = max_active_delta.max(inner_delta);
            if inner_delta < options.tol {
                break;
            }
        }

        // Outer sweep: check all features to see if new features enter the active set
        let mut new_active_added = false;
        let mut full_delta = 0.0f64;
        for j in 0..p {
            if active.contains(&j) {
                continue;
            }
            let diag = col_norms[j];
            if diag <= 0.0 {
                continue;
            }
            let rho = design.col_dot_residuals(j, &residuals, None);
            let scale = col_scales[j];
            let l1_pen = options.l1 * scale;
            if rho.abs() > l1_pen {
                let l2_pen = options.l2 * scale * scale;
                let denom = diag + l2_pen;
                let raw = rho.signum() * (rho.abs() - l1_pen) / denom;
                let mono = options.monotone.get(j.saturating_sub(1)).copied().unwrap_or(0);
                let next_b = match mono {
                    1 => raw.max(0.0),
                    -1 => raw.min(0.0),
                    _ => raw,
                };

                if next_b != 0.0 {
                    beta[j] = next_b;
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= next_b * row_buf[j];
                    }
                    active.push(j);
                    new_active_added = true;
                    full_delta = full_delta.max(next_b.abs());
                }
            }
        }

        if !new_active_added && max_active_delta < options.tol {
            break;
        }
    }

    Ok(beta)
}
