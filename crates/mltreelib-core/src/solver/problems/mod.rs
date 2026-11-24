//! Problem registry and group-aware penalty calculations for modular regression solvers.
//!
//! Exposes:
//! - Group-aware penalty norms ($\ell_2$ group norms, group soft-thresholding)
//! - Group Lasso and Sparse-Group Lasso loss evaluations
//! - Problem declarations for modular regression dispatch (Spec 27)
//! - Quantile regression loss, surrogate, solver, and crossing diagnostics (Phase 2)

pub mod expectile;
pub mod glm_extra;
pub mod huber;
pub mod nnls;
pub mod quantile;

pub use expectile::{
    eval_expectile_loss, eval_expectile_loss_and_grad, expectile_deriv, expectile_loss,
    solve_expectile_lbfgs, weighted_expectile, ExpectileObjective,
};
pub use glm_extra::{
    beta_unit_deviance, eval_beta_deviance, eval_nb2_deviance, nb2_unit_deviance,
    solve_beta_regression, solve_negative_binomial_regression, BetaFitResult,
    NegativeBinomialFitResult,
};
pub use huber::{
    eval_huber_loss, eval_huber_loss_and_grad, huber_loss, huber_psi, solve_huber_location_lbfgs,
    solve_huber_regression, weighted_mad_scale, HuberFitResult, HuberObjective,
};
pub use nnls::{solve_cholesky, solve_nnls, NnlsFitResult};
pub use quantile::{
    asymmetric_weight, detect_quantile_crossings, pinball_loss, smoothed_pinball_loss,
    solve_quantile_lbfgs, QuantileCrossingReport, SmoothedQuantileObjective,
};

use crate::solver::groups::{GroupKind, GroupSpec};

/// Computes Euclidean norm ($\ell_2$) of coefficients belonging to a group: $\sqrt{\sum_{j \in \text{members}} \beta_j^2}$.
#[inline]
pub fn group_l2_norm(coef: &[f64], members: &[usize]) -> f64 {
    let mut sum_sq = 0.0;
    for &j in members {
        if j < coef.len() {
            let v = coef[j];
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

/// Applies group soft-thresholding operator in-place to coefficient slice:
///
/// $$\beta_g \leftarrow \left(1 - \frac{\text{threshold}}{\|\beta_g\|_2}\right)_+ \beta_g$$
///
/// If $\|\beta_g\|_2 \le \text{threshold}$, all members are set to zero.
pub fn group_soft_threshold(coef: &mut [f64], members: &[usize], threshold: f64) {
    if threshold <= 0.0 {
        return;
    }
    let norm = group_l2_norm(coef, members);
    if norm <= threshold || norm <= 1e-15 {
        for &j in members {
            if j < coef.len() {
                coef[j] = 0.0;
            }
        }
    } else {
        let scale = 1.0 - threshold / norm;
        for &j in members {
            if j < coef.len() {
                coef[j] *= scale;
            }
        }
    }
}

/// Computes Group Lasso penalty value:
///
/// $$\Omega_{\text{group}}(\beta) = \alpha \sum_{g} w_g \|\beta_g\|_2$$
///
/// where $w_g$ is `group.effective_weight()`.
pub fn group_lasso_penalty(coef: &[f64], groups: &[GroupSpec], alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let mut penalty = 0.0;
    for g in groups {
        if g.kind == GroupKind::CoefficientGroup {
            let norm = group_l2_norm(coef, &g.members);
            penalty += g.effective_weight() * norm;
        }
    }
    alpha * penalty
}

/// Computes Sparse-Group Lasso penalty value:
///
/// $$\Omega_{\text{SGL}}(\beta) = \alpha \left( \lambda_1 \|\beta\|_1 + \lambda_2 \sum_{g} w_g \|\beta_g\|_2 \right)$$
///
/// Intercept (index 0) is excluded from penalty evaluation.
pub fn sparse_group_lasso_penalty(
    coef: &[f64],
    groups: &[GroupSpec],
    alpha: f64,
    lambda1: f64,
    lambda2: f64,
) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let mut l1_val = 0.0;
    if lambda1 > 0.0 && coef.len() > 1 {
        for &c in &coef[1..] {
            l1_val += c.abs();
        }
    }
    let mut grp_val = 0.0;
    if lambda2 > 0.0 {
        for g in groups {
            if g.kind == GroupKind::CoefficientGroup {
                grp_val += g.effective_weight() * group_l2_norm(coef, &g.members);
            }
        }
    }
    alpha * (lambda1 * l1_val + lambda2 * grp_val)
}

/// Identifier for regression problem types implemented across Spec 27 phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RegressionProblemKind {
    /// Standard ordinary / ridge least squares.
    GaussianLeastSquares,
    /// Quantile regression with pinball loss (Phase 2).
    QuantileRegression,
    /// Robust regression with Huber loss and adaptive scale (Phase 3).
    HuberRegression,
    /// Asymmetric squared error expectile regression (Phase 4).
    ExpectileRegression,
    /// Non-negative least squares (Phase 5).
    NonNegativeLeastSquares,
    /// Group and Sparse-Group Lasso penalized regression (Phase 6).
    GroupLasso,
    /// Overdispersed count model (Phase 7).
    NegativeBinomial,
    /// Bounded unit-interval outcome model (Phase 7).
    BetaRegression,
}
