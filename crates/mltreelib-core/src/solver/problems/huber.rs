//! Huber regression problem formulation, exact gradient, robust scale iteration,
//! and L-BFGS solver (Spec 27 Phase 3).

use crate::solver::design::DesignMatrix;
use crate::solver::generic::{require_dimension, require_finite, Bounds, Objective, ObjectiveError, SmoothObjective};
use crate::stats::{huber_location, w_median, w_sd};

/// Evaluates standard Huber loss for a scalar residual $r$ and threshold $\delta > 0$:
///
/// $$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta |r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta \end{cases}$$
#[inline]
pub fn huber_loss(r: f64, delta: f64) -> f64 {
    let d = if delta > 0.0 { delta } else { 1.345 };
    let u = r.abs();
    if u <= d {
        0.5 * u * u
    } else {
        d * (u - 0.5 * d)
    }
}

/// Evaluates Huber influence function $\psi_\delta(r) = \frac{d\rho_\delta}{dr}$:
///
/// $$\psi_\delta(r) = \text{clamp}(r, -\delta, \delta)$$
#[inline]
pub fn huber_psi(r: f64, delta: f64) -> f64 {
    let d = if delta > 0.0 { delta } else { 1.345 };
    r.clamp(-d, d)
}

/// Computes weighted Median Absolute Deviation (MAD) of residuals with consistency
/// scaling for normality:
///
/// $$\hat{\sigma} = 1.4826022185 \cdot \text{MAD}(r)$$
///
/// If $\text{MAD} = 0$, falls back to weighted standard deviation or $1.0$.
pub fn weighted_mad_scale(residuals: &[(f64, f64)]) -> f64 {
    if residuals.is_empty() {
        return 1.0;
    }
    let med = w_median(residuals);
    let devs: Vec<(f64, f64)> = residuals
        .iter()
        .map(|&(r, w)| ((r - med).abs(), w))
        .collect();
    let mad = w_median(&devs);
    let consistency = 1.482602218505602;
    let scale = consistency * mad;
    if scale > 1e-7 && scale.is_finite() {
        scale
    } else {
        let sd = w_sd(residuals);
        if sd > 1e-7 && sd.is_finite() {
            sd
        } else {
            1.0
        }
    }
}

/// Evaluates total weighted Huber loss and gradient vector over a `DesignMatrix`.
///
/// Effective threshold is $\Delta = \delta \cdot \hat{\sigma}$.
pub fn eval_huber_loss_and_grad<D: DesignMatrix>(
    design: &D,
    delta: f64,
    scale: f64,
    beta: &[f64],
    l2: f64,
    grad: &mut [f64],
    row_buf: &mut [f64],
) -> f64 {
    let n = design.n_rows();
    let p = design.n_features();
    let eff_delta = (delta * scale).max(1e-7);
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
        total_loss += wi * huber_loss(residual, eff_delta);

        let psi = huber_psi(residual, eff_delta);
        // Gradient w.r.t beta: d(loss)/d(beta_j) = -wi * psi * x_ij
        let factor = -wi * psi;
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

/// Evaluates total weighted Huber loss only (used in backtracking line search).
pub fn eval_huber_loss<D: DesignMatrix>(
    design: &D,
    delta: f64,
    scale: f64,
    beta: &[f64],
    l2: f64,
) -> f64 {
    let n = design.n_rows();
    let p = design.n_features();
    let eff_delta = (delta * scale).max(1e-7);
    let mut total_loss = 0.0;

    for i in 0..n {
        let wi = design.get_weight(i);
        if wi <= 0.0 {
            continue;
        }
        let yi = design.get_y(i);
        let eta = design.eval_row(i, beta);
        let residual = yi - eta;
        total_loss += wi * huber_loss(residual, eff_delta);
    }

    if l2 > 0.0 {
        for j in 1..p {
            total_loss += 0.5 * l2 * beta[j] * beta[j];
        }
    }

    total_loss
}

/// Huber regression objective implementing the generic optimization traits
/// [`Objective`] and [`SmoothObjective`].
pub struct HuberObjective<'a, D: DesignMatrix> {
    pub design: &'a D,
    pub delta: f64,
    pub scale: f64,
    pub l2: f64,
    bounds: Bounds,
}

impl<'a, D: DesignMatrix> HuberObjective<'a, D> {
    /// Creates a new Huber regression objective for a fixed scale $\hat{\sigma}$.
    pub fn new(design: &'a D, delta: f64, scale: f64, l2: f64) -> Self {
        let p = design.n_features();
        Self {
            design,
            delta: if delta > 0.0 { delta } else { 1.345 },
            scale: if scale > 0.0 { scale } else { 1.0 },
            l2: l2.max(0.0),
            bounds: Bounds::unbounded(p),
        }
    }
}

impl<D: DesignMatrix> Objective for HuberObjective<'_, D> {
    fn dim(&self) -> usize {
        self.design.n_features()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let loss = eval_huber_loss(self.design, self.delta, self.scale, x, self.l2);
        require_finite(loss)
    }
}

impl<D: DesignMatrix> SmoothObjective for HuberObjective<'_, D> {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        require_dimension(self.dim(), x)?;
        let p = self.dim();
        let mut grad = vec![0.0; p];
        let mut row_buf = vec![0.0; p];
        let loss = eval_huber_loss_and_grad(
            self.design,
            self.delta,
            self.scale,
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

/// Quasi-Newton L-BFGS solver with Armijo backtracking line search for fixed-scale
/// Huber location regression over a [`DesignMatrix`].
pub fn solve_huber_location_lbfgs<D: DesignMatrix>(
    design: &D,
    delta: f64,
    scale: f64,
    l2: f64,
    max_iter: usize,
    tol: f64,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for Huber regression".into());
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(format!("Huber delta must be positive, got {delta}"));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!("Huber scale must be positive, got {scale}"));
    }

    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Initial estimate: 0D Huber location for intercept
        let mut yws = Vec::with_capacity(n);
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                yws.push((design.get_y(i), w));
            }
        }
        if !yws.is_empty() {
            beta[0] = huber_location(&yws);
        }
    }

    let mut grad = vec![0.0; p];
    let mut row_buf = vec![0.0; p];
    let mut current_loss =
        eval_huber_loss_and_grad(design, delta, scale, &beta, l2, &mut grad, &mut row_buf);

    if !current_loss.is_finite() {
        return Err("initial Huber loss is not finite".into());
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
            let trial_loss = eval_huber_loss(design, delta, scale, &trial_beta, l2);
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
        let new_loss = eval_huber_loss_and_grad(
            design,
            delta,
            scale,
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

/// Output of multivariate Huber regression with robust scale estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct HuberFitResult {
    /// Fitted coefficient vector `[intercept, beta_1, ..., beta_p]`.
    pub coefficients: Vec<f64>,
    /// Huber tuning constant $\delta$ (default 1.345).
    pub delta: f64,
    /// Estimated robust scale $\hat{\sigma}$ from weighted MAD.
    pub scale: f64,
    /// Final total Huber loss value.
    pub loss: f64,
    /// Number of outer scale-location iterations.
    pub n_iter: usize,
    /// Whether outer scale-location iterations converged within tolerance.
    pub converged: bool,
}

/// Alternating outer loop for multivariate Huber M-regression with robust scale estimation:
///
/// 1. Initialize location $\beta^{(0)}$ (0D Huber location for intercept, 0 slopes).
/// 2. Compute residuals $r_i^{(k)} = y_i - x_i^\top \beta^{(k)}$ and weighted MAD scale $\hat{\sigma}^{(k)}$.
/// 3. Update coefficients $\beta^{(k+1)}$ via L-BFGS with cutoff $\Delta = \delta \cdot \hat{\sigma}^{(k)}$.
/// 4. Repeat until both scale and location converge, or detect divergence.
pub fn solve_huber_regression<D: DesignMatrix>(
    design: &D,
    delta: f64,
    l2: f64,
    max_outer_iter: usize,
    max_inner_iter: usize,
    tol: f64,
) -> Result<HuberFitResult, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for Huber regression".into());
    }
    let delta = if delta > 0.0 { delta } else { 1.345 };

    // Initial coefficients: 0D Huber location for intercept, zero slopes
    let mut beta = vec![0.0; p];
    let mut yws = Vec::with_capacity(n);
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            yws.push((design.get_y(i), w));
        }
    }
    if !yws.is_empty() {
        beta[0] = huber_location(&yws);
    }

    // Initial residuals and scale
    let mut resids = Vec::with_capacity(n);
    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let eta = design.eval_row(i, &beta);
            resids.push((design.get_y(i) - eta, w));
        }
    }
    let mut scale = weighted_mad_scale(&resids);

    let mut converged = false;
    let mut total_iter = 0;
    let outer_iters = max_outer_iter.max(1);

    for iter in 0..outer_iters {
        total_iter = iter + 1;

        // Divergence detection
        if !scale.is_finite() || scale > 1e12 || scale <= 0.0 {
            return Err("Huber regression diverged: robust scale is non-finite or exploded".into());
        }

        // Solve inner location problem for fixed scale
        let new_beta = solve_huber_location_lbfgs(
            design,
            delta,
            scale,
            l2,
            max_inner_iter,
            tol,
            Some(&beta),
        )?;

        // Check finite coefficients
        if !new_beta.iter().all(|v| v.is_finite()) {
            return Err("Huber regression diverged: non-finite coefficients encountered".into());
        }

        // Compute updated residuals
        resids.clear();
        for i in 0..n {
            let w = design.get_weight(i);
            if w > 0.0 {
                let eta = design.eval_row(i, &new_beta);
                resids.push((design.get_y(i) - eta, w));
            }
        }
        let new_scale = weighted_mad_scale(&resids);

        // Relative differences
        let scale_diff = (new_scale - scale).abs() / scale.max(1.0);
        let mut coef_diff = 0.0f64;
        for j in 0..p {
            let diff = (new_beta[j] - beta[j]).abs() / (1.0 + beta[j].abs());
            coef_diff = coef_diff.max(diff);
        }

        beta = new_beta;
        scale = new_scale;

        if scale_diff < tol && coef_diff < tol {
            converged = true;
            break;
        }
    }

    if !converged && scale > 1e6 {
        return Err(format!(
            "Huber regression diverged: scale reached {scale:.2e} after {total_iter} iterations"
        ));
    }

    let loss = eval_huber_loss(design, delta, scale, &beta, l2);
    Ok(HuberFitResult {
        coefficients: beta,
        delta,
        scale,
        loss,
        n_iter: total_iter,
        converged,
    })
}
