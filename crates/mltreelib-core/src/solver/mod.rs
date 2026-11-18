//! Unified Statistical Modeling and Regression Solver Module.
//!
//! Provides a shared, production-grade optimization engine for:
//! - Exact OLS and Ridge regression via Cholesky decomposition (`normal`)
//! - ElasticNet ($L_1 / L_2$) with active-set Coordinate Descent (`ccd`)
//! - Non-linear GLM deviance minimization via Quasi-Newton L-BFGS (`lbfgs`)
//! - Statistical summaries matching R `summary(glm)` and Python statsmodels (`summary`)
//! - Zero-copy slice views (`design`) for decision tree leaf fits

pub mod ccd;
pub mod design;
pub mod evolutionary;
pub mod family;
pub mod lbfgs;
pub mod normal;
pub mod summary;

pub use ccd::{solve_ccd, CcdOptions};
pub use design::{ColumnSliceView, DenseView, DesignMatrix};
pub use evolutionary::{
    solve_cmaes, solve_evolutionary, solve_rcga, solve_shade, solve_snes, EvolutionaryKind,
    EvolutionaryOptions,
};
pub use family::GlmFamily;
pub use lbfgs::{solve_lbfgs, LbfgsOptions};
pub use normal::solve_normal;
pub use summary::{CoeffStats, GlmSummary};

/// General solver options for fitting a linear or generalized linear model.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    pub family: GlmFamily,
    pub l1: f64,
    pub l2: f64,
    pub monotone: Vec<i8>,
    pub max_iter: usize,
    pub tol: f64,
    pub verbose: u8,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            family: GlmFamily::Gaussian,
            l1: 0.0,
            l2: 0.0,
            monotone: Vec::new(),
            max_iter: 1000,
            tol: 1e-6,
            verbose: 0,
        }
    }
}

/// Fitted linear or GLM model result containing coefficients and optional summary.
#[derive(Clone, Debug)]
pub struct LinearModelResult {
    pub intercept: f64,
    pub coef: Vec<f64>,
    pub na_coef: Vec<f64>,
    pub summary: Option<GlmSummary>,
}

/// Fits a model by automatically routing to the optimal solver:
/// - Exact Cholesky normal equations for unpenalized or Ridge Gaussian.
/// - Cyclic Coordinate Descent with active set for L1 ElasticNet or monotone constraints.
/// - Quasi-Newton L-BFGS for non-linear GLMs (Binomial, Poisson, Gamma, Tweedie, Tobit).
pub fn fit_model<D: DesignMatrix>(
    design: &D,
    options: &SolverOptions,
    warm_start: Option<&[f64]>,
    feature_names: Option<&[String]>,
) -> Result<LinearModelResult, String> {
    let p = design.n_features();
    if p == 0 {
        return Err("Zero features in design matrix".to_string());
    }

    let is_gaussian = options.family == GlmFamily::Gaussian;
    let has_l1 = options.l1 > 0.0;
    let has_monotone = options.monotone.iter().any(|&m| m != 0);

    let (beta, summary) = if is_gaussian {
        if !has_l1 && !has_monotone {
            // Fast path: Exact Cholesky with QR fallback + full hypothesis testing summary
            solve_normal(design, options.l2, feature_names)?
        } else {
            // ElasticNet or monotone constraints for Gaussian: Coordinate Descent
            let ccd_opts = CcdOptions {
                l1: options.l1,
                l2: options.l2,
                max_iter: options.max_iter,
                tol: options.tol,
                monotone: options.monotone.clone(),
            };
            let sol = solve_ccd(design, &ccd_opts, warm_start)?;
            (sol, None)
        }
    } else {
        if has_l1 {
            return Err("L1 regularization currently supported for Gaussian family. Use L2 with GLM families.".to_string());
        }
        // Non-linear GLM: Quasi-Newton L-BFGS
        let lbfgs_opts = LbfgsOptions {
            family: options.family,
            l2: options.l2,
            max_iter: options.max_iter,
            m: 6,
            tol: options.tol,
            verbose: options.verbose,
        };
        let sol = solve_lbfgs(design, &lbfgs_opts, warm_start)?;
        (sol, None)
    };

    let intercept = beta.first().copied().unwrap_or(0.0);
    let coef = if beta.len() > 1 {
        beta[1..].to_vec()
    } else {
        Vec::new()
    };

    Ok(LinearModelResult {
        intercept,
        coef,
        na_coef: Vec::new(),
        summary,
    })
}
