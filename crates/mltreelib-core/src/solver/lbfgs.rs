//! Quasi-Newton L-BFGS Solver for Generalized Linear Models.
//!
//! Features:
//! - Standard two-loop recursion ($m=6$) approximating inverse Hessian.
//! - Armijo backtracking line search ensuring sufficient descent.
//! - Direct evaluation over `DesignMatrix` in $O(N \cdot D)$ time and $O(m \cdot D)$ memory.
//! - Completely avoids building or inverting $D \times D$ Gram matrices.

use crate::solver::design::DesignMatrix;
use crate::solver::family::GlmFamily;

/// Options controlling L-BFGS optimization.
#[derive(Clone, Debug)]
pub struct LbfgsOptions {
    pub family: GlmFamily,
    pub l2: f64,
    pub max_iter: usize,
    pub m: usize,
    pub tol: f64,
    pub verbose: u8,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            family: GlmFamily::Gaussian,
            l2: 1e-4,
            max_iter: 100,
            m: 6,
            tol: 1e-6,
            verbose: 0,
        }
    }
}

/// Evaluates total GLM loss (deviance + L2 penalty) and gradient vector.
fn eval_objective_and_grad<D: DesignMatrix>(
    design: &D,
    family: &GlmFamily,
    beta: &[f64],
    l2: f64,
    grad: &mut [f64],
    row_buf: &mut [f64],
) -> f64 {
    let n = design.n_rows();
    let p = design.n_features();
    grad.fill(0.0);

    let mut total_loss = 0.0;
    for i in 0..n {
        let wi = design.get_weight(i);
        if wi <= 0.0 {
            continue;
        }
        let yi = design.get_y(i);
        let eta = design.eval_row(i, beta);
        let mu = family.inverse_link(eta);
        let dev = family.deviance(yi, mu);
        total_loss += 0.5 * wi * dev;

        let (g_i, _) = family.gradient_and_weight(yi, eta, wi);
        design.copy_design_row(i, row_buf);
        for j in 0..p {
            grad[j] += g_i * row_buf[j];
        }
    }

    // Add L2 Ridge penalty (intercept j=0 unpenalized)
    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
            grad[j] += l2 * beta[j];
        }
    }

    total_loss
}

/// Evaluates objective value only (used during line search).
fn eval_objective<D: DesignMatrix>(
    design: &D,
    family: &GlmFamily,
    beta: &[f64],
    l2: f64,
) -> f64 {
    let n = design.n_rows();
    let p = design.n_features();
    let mut total_loss = 0.0;
    for i in 0..n {
        let wi = design.get_weight(i);
        if wi <= 0.0 {
            continue;
        }
        let yi = design.get_y(i);
        let eta = design.eval_row(i, beta);
        let mu = family.inverse_link(eta);
        total_loss += 0.5 * wi * family.deviance(yi, mu);
    }
    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
        }
    }
    total_loss
}

/// Fits a GLM model using Quasi-Newton L-BFGS.
pub fn solve_lbfgs<D: DesignMatrix>(
    design: &D,
    options: &LbfgsOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("Empty design matrix".to_string());
    }

    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Initial intercept
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            sum_y += w * design.get_y(i);
            sum_w += w;
        }
        let y_bar = if sum_w > 0.0 { sum_y / sum_w } else { 0.0 };
        beta[0] = options.family.initial_intercept(y_bar);
    }

    let mut grad = vec![0.0; p];
    let mut row_buf = vec![0.0; p];
    let mut current_loss = eval_objective_and_grad(
        design,
        &options.family,
        &beta,
        options.l2,
        &mut grad,
        &mut row_buf,
    );

    if !current_loss.is_finite() {
        return Err("Initial loss is not finite".to_string());
    }

    if options.verbose >= 3 {
        println!(
            "      [L-BFGS start] family={:?} p={} initial_loss={:.6}",
            options.family, p, current_loss
        );
    }

    let m_history = options.m.max(1);
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m_history);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m_history);
    let mut rho_history: Vec<f64> = Vec::with_capacity(m_history);

    let mut dir = vec![0.0; p];
    let mut q = vec![0.0; p];
    let mut alphas = vec![0.0; m_history];
    let mut trial_beta = vec![0.0; p];

    for iter in 0..options.max_iter {
        // Check gradient convergence: max |grad_j| < tol
        let max_grad = grad.iter().fold(0.0f64, |acc, &g| acc.max(g.abs()));
        if max_grad < options.tol {
            if options.verbose >= 3 {
                println!(
                    "      [L-BFGS converged] iters={} final_loss={:.6} max_grad={:.4e}",
                    iter, current_loss, max_grad
                );
            }
            return Ok(beta);
        }

        // Two-Loop Recursion to compute descent direction dir = - H_k grad
        let k = s_history.len();
        q.copy_from_slice(&grad);

        for i in (0..k).rev() {
            let s_i = &s_history[i];
            let y_i = &y_history[i];
            let rho_i = rho_history[i];
            let s_dot_q: f64 = s_i.iter().zip(&q).map(|(s, qi)| s * qi).sum();
            let alpha = rho_i * s_dot_q;
            alphas[i] = alpha;
            for j in 0..p {
                q[j] -= alpha * y_i[j];
            }
        }

        // Initial Hessian scale gamma_0 = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let gamma_0 = if k > 0 {
            let last_s = &s_history[k - 1];
            let last_y = &y_history[k - 1];
            let s_dot_y: f64 = last_s.iter().zip(last_y).map(|(s, y)| s * y).sum();
            let y_dot_y: f64 = last_y.iter().map(|y| y * y).sum();
            if y_dot_y > 1e-12 {
                (s_dot_y / y_dot_y).max(1e-6).min(1e6)
            } else {
                1.0
            }
        } else {
            1.0
        };

        for j in 0..p {
            dir[j] = gamma_0 * q[j];
        }

        for i in 0..k {
            let s_i = &s_history[i];
            let y_i = &y_history[i];
            let rho_i = rho_history[i];
            let y_dot_r: f64 = y_i.iter().zip(&dir).map(|(y, r)| y * r).sum();
            let beta_val = rho_i * y_dot_r;
            let alpha = alphas[i];
            for j in 0..p {
                dir[j] += s_i[j] * (alpha - beta_val);
            }
        }

        // Descent direction: negate
        for j in 0..p {
            dir[j] = -dir[j];
        }

        // Directional derivative: grad^T dir (must be negative for descent)
        let dir_deriv: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
        if dir_deriv >= 0.0 {
            // Not a descent direction: reset to steepest descent
            for j in 0..p {
                dir[j] = -grad[j];
            }
            s_history.clear();
            y_history.clear();
            rho_history.clear();
        }

        let dir_deriv_final: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();

        // Armijo Backtracking Line Search
        let c1 = 1e-4;
        let mut step = 1.0;
        let mut line_search_converged = false;
        let mut trial_loss = current_loss;

        for _ls in 0..25 {
            for j in 0..p {
                trial_beta[j] = beta[j] + step * dir[j];
            }
            trial_loss = eval_objective(design, &options.family, &trial_beta, options.l2);

            if trial_loss.is_finite() && trial_loss <= current_loss + c1 * step * dir_deriv_final {
                line_search_converged = true;
                break;
            }
            step *= 0.5;
        }

        if !line_search_converged || !trial_loss.is_finite() {
            // Line search failed: return current best
            if options.verbose >= 3 {
                println!("      [L-BFGS line search terminated] iter={}", iter);
            }
            return Ok(beta);
        }

        let loss_diff = current_loss - trial_loss;
        let mut next_grad = vec![0.0; p];
        let _ = eval_objective_and_grad(
            design,
            &options.family,
            &trial_beta,
            options.l2,
            &mut next_grad,
            &mut row_buf,
        );

        if options.verbose >= 3 {
            println!(
                "      [L-BFGS iter {}] loss={:.6} step={:.4} loss_diff={:.6e}",
                iter, trial_loss, step, loss_diff
            );
        }

        // Store history: s_k = trial_beta - beta, y_k = next_grad - grad
        let s_k: Vec<f64> = trial_beta.iter().zip(&beta).map(|(tb, b)| tb - b).collect();
        let y_k: Vec<f64> = next_grad.iter().zip(&grad).map(|(ng, g)| ng - g).collect();
        let s_dot_y: f64 = s_k.iter().zip(&y_k).map(|(s, y)| s * y).sum();

        if s_dot_y > 1e-12 {
            let rho_k = 1.0 / s_dot_y;
            if s_history.len() == m_history {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s_k);
            y_history.push(y_k);
            rho_history.push(rho_k);
        }

        beta.copy_from_slice(&trial_beta);
        grad.copy_from_slice(&next_grad);
        current_loss = trial_loss;

        if loss_diff < options.tol * (1.0 + current_loss.abs()) {
            if options.verbose >= 3 {
                println!(
                    "      [L-BFGS converged] iters={} final_loss={:.6}",
                    iter + 1, current_loss
                );
            }
            return Ok(beta);
        }
    }

    Ok(beta)
}
