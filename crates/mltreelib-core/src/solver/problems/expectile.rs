//! Expectile regression problem formulations, asymmetric squared loss,
//! L-BFGS solver, and central expectile mean recovery (Spec 27 Phase 4).

use crate::solver::design::DesignMatrix;
use crate::solver::generic::{
    require_dimension, require_finite, Bounds, Objective, ObjectiveError, SmoothObjective,
};
pub use super::quantile::asymmetric_weight;

/// Evaluates exact expectile asymmetric squared loss for residual $r$:
///
/// $$\rho_\tau^{\exp}(r) = |\tau - \mathbb{1}(r < 0)| \cdot r^2 = \begin{cases} \tau r^2 & \text{if } r \ge 0 \\ (1 - \tau) r^2 & \text{if } r < 0 \end{cases}$$
#[inline]
pub fn expectile_loss(r: f64, tau: f64) -> f64 {
    asymmetric_weight(r, tau) * r * r
}

/// Derivative of expectile loss with respect to residual $r$:
///
/// $$\frac{d}{dr} \rho_\tau^{\exp}(r) = 2 \cdot |\tau - \mathbb{1}(r < 0)| \cdot r$$
#[inline]
pub fn expectile_deriv(r: f64, tau: f64) -> f64 {
    2.0 * asymmetric_weight(r, tau) * r
}

/// Evaluates total expectile loss over a [`DesignMatrix`]:
///
/// $$L(\beta) = \sum_{i=1}^n w_i \rho_\tau^{\exp}(y_i - \mathbf{x}_i^\top \beta) + \frac{\lambda}{2} \sum_{j=1}^{p-1} \beta_j^2$$
pub fn eval_expectile_loss<D: DesignMatrix>(
    design: &D,
    tau: f64,
    beta: &[f64],
    l2: f64,
) -> f64 {
    let n = design.n_rows();
    let mut total_loss = 0.0;
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, beta);
            let r = design.get_y(i) - eta;
            total_loss += w * expectile_loss(r, tau);
        }
    }
    if l2 > 0.0 {
        let mut penalty = 0.0;
        for &v in beta.iter().skip(1) {
            penalty += v * v;
        }
        total_loss += 0.5 * l2 * penalty;
    }
    total_loss
}

/// Evaluates total expectile loss and exact gradient vector w.r.t coefficients $\beta$:
///
/// $$\nabla_\beta L(\beta) = -2 \sum_{i=1}^n w_i |\tau - \mathbb{1}(r_i < 0)| r_i \mathbf{x}_i + \lambda [0, \beta_1, \dots, \beta_{p-1}]^\top$$
pub fn eval_expectile_loss_and_grad<D: DesignMatrix>(
    design: &D,
    tau: f64,
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
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, beta);
            let r = design.get_y(i) - eta;
            total_loss += w * expectile_loss(r, tau);

            let d_loss_dr = expectile_deriv(r, tau);
            // d_loss_dbeta = d_loss_dr * (-x_i) = -w * d_loss_dr * x_i
            let factor = -w * d_loss_dr;
            design.copy_design_row(i, row_buf);
            for j in 0..p {
                grad[j] += factor * row_buf[j];
            }
        }
    }

    if l2 > 0.0 {
        let mut penalty = 0.0;
        for j in 1..p {
            let v = beta[j];
            penalty += v * v;
            grad[j] += l2 * v;
        }
        total_loss += 0.5 * l2 * penalty;
    }

    total_loss
}

/// Calculates 1D weighted expectile of value-weight pairs by iteratively reweighted least squares.
///
/// When $\tau = 0.5$, this exactly reduces to the weighted arithmetic mean in 0 iterations.
pub fn weighted_expectile(yws: &[(f64, f64)], tau: f64) -> f64 {
    if yws.is_empty() {
        return f64::NAN;
    }
    let tau = tau.clamp(1e-4, 1.0 - 1e-4);

    // Initial estimate: weighted arithmetic mean
    let mut sum_wy = 0.0;
    let mut sum_w = 0.0;
    for &(y, w) in yws {
        if w > 0.0 && y.is_finite() {
            sum_wy += w * y;
            sum_w += w;
        }
    }
    if sum_w <= 0.0 {
        return f64::NAN;
    }
    let mut theta = sum_wy / sum_w;
    if (tau - 0.5).abs() < 1e-12 {
        return theta;
    }

    // Iteratively reweighted mean iterations
    for _ in 0..25 {
        let mut num = 0.0;
        let mut den = 0.0;
        for &(y, w) in yws {
            if w > 0.0 && y.is_finite() {
                let r = y - theta;
                let aw = asymmetric_weight(r, tau);
                num += w * aw * y;
                den += w * aw;
            }
        }
        if den <= 0.0 {
            break;
        }
        let next_theta = num / den;
        if (next_theta - theta).abs() <= 1e-9 * (1.0 + theta.abs()) {
            theta = next_theta;
            break;
        }
        theta = next_theta;
    }
    theta
}

/// Objective conforming to the generic solver seam for multivariate expectile regression.
#[derive(Clone, Debug)]
pub struct ExpectileObjective<'a, D: DesignMatrix> {
    design: &'a D,
    tau: f64,
    l2: f64,
    bounds: Bounds,
}

impl<'a, D: DesignMatrix> ExpectileObjective<'a, D> {
    /// Creates a new `ExpectileObjective`.
    pub fn new(design: &'a D, tau: f64, l2: f64) -> Result<Self, String> {
        let p = design.n_features();
        if p == 0 || design.n_rows() == 0 {
            return Err("empty design matrix for ExpectileObjective".into());
        }
        if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
            return Err(format!("tau must be in (0, 1), got {tau}"));
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err(format!("l2 penalty must be non-negative, got {l2}"));
        }
        Ok(Self {
            design,
            tau,
            l2,
            bounds: Bounds::unbounded(p),
        })
    }
}

impl<D: DesignMatrix> Objective for ExpectileObjective<'_, D> {
    fn dim(&self) -> usize {
        self.design.n_features()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let loss = eval_expectile_loss(self.design, self.tau, x, self.l2);
        require_finite(loss)
    }
}

impl<D: DesignMatrix> SmoothObjective for ExpectileObjective<'_, D> {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let p = self.dim();
        let mut grad = vec![0.0; p];
        let mut row_buf = vec![0.0; p];
        let loss = eval_expectile_loss_and_grad(
            self.design,
            self.tau,
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

/// L-BFGS Quasi-Newton solver for multivariate expectile regression over a [`DesignMatrix`].
pub fn solve_expectile_lbfgs<D: DesignMatrix>(
    design: &D,
    tau: f64,
    l2: f64,
    max_iter: usize,
    tol: f64,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for expectile regression".into());
    }
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(format!("tau must be in (0, 1), got {tau}"));
    }

    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Initial estimate: 1D weighted expectile for intercept
        let mut yws = Vec::with_capacity(n);
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                yws.push((design.get_y(i), w));
            }
        }
        if !yws.is_empty() {
            beta[0] = weighted_expectile(&yws, tau);
        }
    }

    let mut grad = vec![0.0; p];
    let mut row_buf = vec![0.0; p];
    let mut current_loss =
        eval_expectile_loss_and_grad(design, tau, &beta, l2, &mut grad, &mut row_buf);

    if !current_loss.is_finite() {
        return Err("initial expectile loss is not finite".into());
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
        if max_grad / (n as f64).max(1.0) < tol {
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
            let g_norm = grad.iter().fold(0.0f64, |acc, &g| acc.max(g.abs()));
            if g_norm > 1.0 {
                1.0 / g_norm
            } else {
                1.0
            }
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
            let trial_loss = eval_expectile_loss(design, tau, &trial_beta, l2);
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
        let new_loss = eval_expectile_loss_and_grad(
            design,
            tau,
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
        grad.copy_from_slice(&new_grad);
        current_loss = new_loss;
    }

    Ok(beta)
}
