//! Quantile regression problem formulations, smoothed pinball surrogate,
//! L-BFGS solver, and quantile crossing diagnostics (Spec 27 Phase 2).

use crate::solver::design::DesignMatrix;
use crate::solver::generic::{require_dimension, require_finite, Bounds, Objective, ObjectiveError, SmoothObjective};
use crate::stats::w_quantile;

/// Computes the asymmetric quantile weight for a given residual:
///
/// $$w_\tau(r) = \begin{cases} \tau & \text{if } r \ge 0 \\ 1 - \tau & \text{if } r < 0 \end{cases}$$
#[inline]
pub fn asymmetric_weight(r: f64, tau: f64) -> f64 {
    if r >= 0.0 {
        tau
    } else {
        1.0 - tau
    }
}

/// Evaluates exact check (pinball) loss:
///
/// $$\rho_\tau(r) = r \cdot (\tau - \mathbb{1}(r < 0)) = \begin{cases} \tau r & \text{if } r \ge 0 \\ (1 - \tau)(-r) & \text{if } r < 0 \end{cases}$$
#[inline]
pub fn pinball_loss(r: f64, tau: f64) -> f64 {
    r * (tau - if r < 0.0 { 1.0 } else { 0.0 })
}

/// Evaluates smoothed surrogate pinball loss and its first derivative with respect to residual $r$
/// using a quadratic splice within $[-\epsilon, \epsilon]$:
///
/// - For $r > \epsilon$: $q_\tau(r) = \tau r, \quad q'_\tau(r) = \tau$
/// - For $r < -\epsilon$: $q_\tau(r) = (1 - \tau)(-r), \quad q'_\tau(r) = \tau - 1$
/// - For $|r| \le \epsilon$: $q_\tau(r) = \frac{r^2}{4\epsilon} + (\tau - 0.5)r + \frac{\epsilon}{4}, \quad q'_\tau(r) = \frac{r}{2\epsilon} + (\tau - 0.5)$
///
/// This surrogate is $C^1$ smooth everywhere, continuous in value and gradient across boundaries $\pm \epsilon$.
#[inline]
pub fn smoothed_pinball_loss(r: f64, tau: f64, eps: f64) -> (f64, f64) {
    let eps = if eps > 0.0 { eps } else { 1e-4 };
    if r > eps {
        (tau * r, tau)
    } else if r < -eps {
        ((1.0 - tau) * (-r), tau - 1.0)
    } else {
        let q = (r * r) / (4.0 * eps) + (tau - 0.5) * r + 0.25 * eps;
        let dq = r / (2.0 * eps) + (tau - 0.5);
        (q, dq)
    }
}

/// Evaluates total smoothed quantile loss and gradient vector over a `DesignMatrix`.
pub fn eval_quantile_loss_and_grad<D: DesignMatrix>(
    design: &D,
    tau: f64,
    eps: f64,
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
        let residual = yi - eta;
        let (q_loss, dq_dr) = smoothed_pinball_loss(residual, tau, eps);
        total_loss += wi * q_loss;

        // Gradient w.r.t beta: d(loss)/d(beta_j) = wi * dq/dr * (-x_ij) = -wi * dq/dr * x_ij
        let factor = -wi * dq_dr;
        design.copy_design_row(i, row_buf);
        for j in 0..p {
            grad[j] += factor * row_buf[j];
        }
    }

    // L2 Ridge penalty on non-intercept coefficients
    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
            grad[j] += l2 * beta[j];
        }
    }

    total_loss
}

/// Evaluates total smoothed quantile loss only (used in backtracking line search).
pub fn eval_quantile_loss<D: DesignMatrix>(
    design: &D,
    tau: f64,
    eps: f64,
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
        let residual = yi - eta;
        let (q_loss, _) = smoothed_pinball_loss(residual, tau, eps);
        total_loss += wi * q_loss;
    }

    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
        }
    }

    total_loss
}

/// Smoothed quantile regression objective implementing the generic optimization traits
/// [`Objective`] and [`SmoothObjective`].
pub struct SmoothedQuantileObjective<'a, D: DesignMatrix> {
    pub design: &'a D,
    pub tau: f64,
    pub eps: f64,
    pub l2: f64,
    bounds: Bounds,
}

impl<'a, D: DesignMatrix> SmoothedQuantileObjective<'a, D> {
    /// Creates a new smoothed quantile regression objective.
    pub fn new(design: &'a D, tau: f64, eps: f64, l2: f64) -> Self {
        let p = design.n_features();
        Self {
            design,
            tau,
            eps: if eps > 0.0 { eps } else { 1e-4 },
            l2: l2.max(0.0),
            bounds: Bounds::unbounded(p),
        }
    }
}

impl<D: DesignMatrix> Objective for SmoothedQuantileObjective<'_, D> {
    fn dim(&self) -> usize {
        self.design.n_features()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let loss = eval_quantile_loss(self.design, self.tau, self.eps, x, self.l2);
        require_finite(loss)
    }
}

impl<D: DesignMatrix> SmoothObjective for SmoothedQuantileObjective<'_, D> {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let p = self.dim();
        let mut grad = vec![0.0; p];
        let mut row_buf = vec![0.0; p];
        let loss = eval_quantile_loss_and_grad(
            self.design,
            self.tau,
            self.eps,
            x,
            self.l2,
            &mut grad,
            &mut row_buf,
        );
        require_finite(loss)?;
        for &g in &grad {
            require_finite(g)?;
        }
        Ok(grad)
    }
}

/// Quasi-Newton L-BFGS solver with Armijo backtracking line search specialized for
/// smoothed quantile regression over a [`DesignMatrix`].
pub fn solve_quantile_lbfgs<D: DesignMatrix>(
    design: &D,
    tau: f64,
    eps: f64,
    l2: f64,
    max_iter: usize,
    tol: f64,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for quantile regression".into());
    }
    if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
        return Err(format!("quantile tau must be in [0, 1], got {tau}"));
    }

    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Initial estimate: 0D weighted quantile for the intercept, zero slopes
        let mut yws = Vec::with_capacity(n);
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                yws.push((design.get_y(i), w));
            }
        }
        if !yws.is_empty() {
            beta[0] = w_quantile(&yws, tau);
        }
    }

    let mut grad = vec![0.0; p];
    let mut row_buf = vec![0.0; p];
    let mut current_loss =
        eval_quantile_loss_and_grad(design, tau, eps, &beta, l2, &mut grad, &mut row_buf);

    if !current_loss.is_finite() {
        return Err("initial quantile loss is not finite".into());
    }

    let m_history = 6;
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m_history);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m_history);
    let mut rho_history: Vec<f64> = Vec::with_capacity(m_history);

    let mut dir = vec![0.0; p];
    let mut q = vec![0.0; p];
    let mut alphas = vec![0.0; m_history];
    let mut trial_beta = vec![0.0; p];

    for _iter in 0..max_iter {
        let max_grad = grad.iter().fold(0.0f64, |acc, &g| acc.max(g.abs()));
        if max_grad < tol {
            return Ok(beta);
        }

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

        let gamma_0 = if k > 0 {
            let last_s = &s_history[k - 1];
            let last_y = &y_history[k - 1];
            let s_dot_y: f64 = last_s.iter().zip(last_y).map(|(s, y)| s * y).sum();
            let y_dot_y: f64 = last_y.iter().map(|y| y * y).sum();
            if y_dot_y > 1e-12 {
                (s_dot_y / y_dot_y).clamp(1e-6, 1e6)
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

        for j in 0..p {
            dir[j] = -dir[j];
        }

        let dir_deriv: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
        if dir_deriv >= 0.0 {
            for j in 0..p {
                dir[j] = -grad[j];
            }
        }

        let mut step = 1.0;
        let c1 = 1e-4;
        let mut line_search_ok = false;
        let armijo_slope = c1 * grad.iter().zip(&dir).map(|(g, d)| g * d).sum::<f64>();

        for _ls in 0..30 {
            for j in 0..p {
                trial_beta[j] = beta[j] + step * dir[j];
            }
            let trial_loss = eval_quantile_loss(design, tau, eps, &trial_beta, l2);
            if trial_loss <= current_loss + step * armijo_slope {
                line_search_ok = true;
                break;
            }
            step *= 0.5;
        }

        if !line_search_ok {
            step = 1e-4;
            for j in 0..p {
                trial_beta[j] = beta[j] + step * dir[j];
            }
        }

        let mut new_grad = vec![0.0; p];
        let new_loss = eval_quantile_loss_and_grad(
            design,
            tau,
            eps,
            &trial_beta,
            l2,
            &mut new_grad,
            &mut row_buf,
        );

        let mut s_k = vec![0.0; p];
        let mut y_k = vec![0.0; p];
        for j in 0..p {
            s_k[j] = trial_beta[j] - beta[j];
            y_k[j] = new_grad[j] - grad[j];
        }

        let s_dot_y: f64 = s_k.iter().zip(&y_k).map(|(s, y)| s * y).sum();
        if s_dot_y > 1e-14 {
            let rho = 1.0 / s_dot_y;
            if s_history.len() == m_history {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s_k);
            y_history.push(y_k);
            rho_history.push(rho);
        }

        let delta_norm = trial_beta
            .iter()
            .zip(&beta)
            .fold(0.0f64, |acc, (b_new, b_old)| acc.max((b_new - b_old).abs()));

        beta.copy_from_slice(&trial_beta);
        grad.copy_from_slice(&new_grad);
        current_loss = new_loss;

        if delta_norm < tol {
            return Ok(beta);
        }
    }

    Ok(beta)
}

/// Summary report of row-level quantile crossings across multiple evaluated quantiles.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QuantileCrossingReport {
    /// Total number of inverted pairs detected: $\hat{Q}_{\tau_1} > \hat{Q}_{\tau_2}$ for $\tau_1 < \tau_2$.
    pub n_crossings: usize,
    /// Number of rows exhibiting at least one crossing.
    pub n_violating_rows: usize,
    /// Rate of violating pairs over all evaluated pairs across all rows: $\frac{\text{crossings}}{N \cdot \binom{T}{2}}$.
    pub crossing_rate: f64,
    /// Maximum crossing violation magnitude: $\max(\hat{Q}_{\tau_1} - \hat{Q}_{\tau_2})_+$.
    pub max_violation: f64,
    /// Detailed record of violations: `(row_idx, tau_lo_idx, tau_hi_idx)`.
    pub crossings: Vec<(usize, usize, usize)>,
}

/// Detects row-level quantile crossing violations ($\hat{Q}_{\tau_1} > \hat{Q}_{\tau_2}$ when $\tau_1 < \tau_2$)
/// across predictions for multiple quantiles.
///
/// Accepts `preds_matrix` formatted either as row-major ($N \times T$) or column-major ($T \times N$).
pub fn detect_quantile_crossings(
    preds_matrix: &[Vec<f64>],
    taus: &[f64],
) -> QuantileCrossingReport {
    let t = taus.len();
    if t < 2 || preds_matrix.is_empty() {
        return QuantileCrossingReport {
            n_crossings: 0,
            n_violating_rows: 0,
            crossing_rate: 0.0,
            max_violation: 0.0,
            crossings: Vec::new(),
        };
    }

    // Determine orientation: column-major if outer length matches taus and inner != taus
    let col_major = preds_matrix.len() == t && preds_matrix[0].len() != t;
    let n_rows = if col_major {
        preds_matrix[0].len()
    } else {
        preds_matrix.len()
    };

    // Sort taus to establish canonical ordering
    let mut tau_order: Vec<usize> = (0..t).collect();
    tau_order.sort_by(|&a, &b| taus[a].partial_cmp(&taus[b]).unwrap());

    let mut crossings = Vec::new();
    let mut violating_rows = 0;
    let mut max_violation = 0.0f64;

    for i in 0..n_rows {
        let mut row_has_violation = false;
        for a_idx in 0..t {
            for b_idx in (a_idx + 1)..t {
                let j = tau_order[a_idx];
                let k = tau_order[b_idx];
                if (taus[k] - taus[j]).abs() < 1e-12 {
                    continue;
                }
                let y_j = if col_major {
                    preds_matrix[j][i]
                } else {
                    preds_matrix[i][j]
                };
                let y_k = if col_major {
                    preds_matrix[k][i]
                } else {
                    preds_matrix[i][k]
                };

                if y_j > y_k + 1e-9 {
                    let diff = y_j - y_k;
                    if diff > max_violation {
                        max_violation = diff;
                    }
                    crossings.push((i, j, k));
                    row_has_violation = true;
                }
            }
        }
        if row_has_violation {
            violating_rows += 1;
        }
    }

    let num_pairs = t * (t - 1) / 2;
    let total_evaluations = (n_rows * num_pairs).max(1);
    let crossing_rate = crossings.len() as f64 / total_evaluations as f64;

    QuantileCrossingReport {
        n_crossings: crossings.len(),
        n_violating_rows: violating_rows,
        crossing_rate,
        max_violation,
        crossings,
    }
}
