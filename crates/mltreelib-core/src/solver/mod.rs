//! Unified Statistical Modeling and Regression Solver Module.
//!
//! Provides a shared, production-grade optimization engine for:
//! - Exact OLS and Ridge regression via Cholesky decomposition (`normal`)
//! - ElasticNet ($L_1 / L_2$) with active-set Coordinate Descent (`ccd`)
//! - Non-linear GLM deviance minimization via Quasi-Newton L-BFGS (`lbfgs`)
//! - Statistical summaries matching R `summary(glm)` and Python statsmodels (`summary`)
//! - Zero-copy slice views (`design`) for decision tree leaf fits

pub mod ask_tell;
pub mod ccd;
pub mod dense;
pub mod design;
pub mod evolutionary;
pub mod ensemble_prune;
pub mod family;
pub mod generic;
pub mod groups;
pub mod lbfgs;
pub mod normal;
pub mod oct;
pub mod problems;
pub mod sparse_rule;
pub mod summary;

pub use ask_tell::{AskTell, AskTellKind, AskTellOptions, ParetoSolution};
pub use ccd::{solve_ccd, CcdOptions};
pub use dense::{fit_dense, FittedGlm};
pub use design::{ColumnSliceView, DenseView, DesignMatrix};
pub use ensemble_prune::{prune_ensemble_stepwise, PruningResult};
pub use evolutionary::{
    solve_cmaes, solve_evolutionary, solve_rcga, solve_shade, solve_snes, EvolutionaryKind,
    EvolutionaryOptions,
};
pub use family::GlmFamily;
pub use generic::{
    Bounds, Budget, Constraint, ConstraintKind, Objective, ObjectiveError, OptimizationRequest,
    OptimizationResult, OptimizerKind, SmoothObjective, SolveStatus,
};
pub use groups::{
    canonicalize_groups, expand_missing_indicators, validate_groups, GroupKind, GroupSpec,
    GroupValidationError,
};
pub use lbfgs::{solve_lbfgs, LbfgsOptions};
pub use normal::solve_normal;
pub use oct::{fit_optimal_tree, OptimalTreeNode, OptimalTreeResult};
pub use problems::{
    asymmetric_weight, detect_quantile_crossings, group_l2_norm, group_lasso_penalty,
    group_soft_threshold, pinball_loss, smoothed_pinball_loss, solve_quantile_lbfgs,
    sparse_group_lasso_penalty, QuantileCrossingReport, RegressionProblemKind,
    SmoothedQuantileObjective,
};
pub use sparse_rule::{fit_sparse_rule_selection, predict_sparse_rules, SparseRuleResult};
pub use summary::{CoeffStats, GlmSummary};

/// Report summarizing group selection outcomes in Group and Sparse-Group Lasso.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupSelectionReport {
    /// Names of groups that have at least one non-zero coefficient.
    pub active_groups: Vec<String>,
    /// Names of groups where all coefficients were set to zero.
    pub zeroed_groups: Vec<String>,
    /// L2 norm of coefficients for each group: name -> norm.
    pub group_norms: std::collections::HashMap<String, f64>,
    /// Proportion of groups that are active (non-zero).
    pub active_ratio: f64,
}

/// Computes group selection summary report.
pub fn compute_group_selection_report(
    coef: &[f64],
    groups: &[GroupSpec],
) -> GroupSelectionReport {
    let mut active_groups = Vec::new();
    let mut zeroed_groups = Vec::new();
    let mut group_norms = std::collections::HashMap::new();

    for g in groups {
        if g.kind == GroupKind::CoefficientGroup {
            let norm = crate::solver::problems::group_l2_norm(coef, &g.members);
            group_norms.insert(g.name.clone(), norm);
            if norm > 1e-12 {
                active_groups.push(g.name.clone());
            } else {
                zeroed_groups.push(g.name.clone());
            }
        }
    }
    let total = active_groups.len() + zeroed_groups.len();
    let active_ratio = if total > 0 {
        active_groups.len() as f64 / total as f64
    } else {
        0.0
    };
    GroupSelectionReport {
        active_groups,
        zeroed_groups,
        group_norms,
        active_ratio,
    }
}

/// Solves Group Lasso or Sparse-Group Lasso directly over any design matrix.
pub fn solve_group_lasso<D: DesignMatrix>(
    design: &D,
    groups: &[GroupSpec],
    alpha: f64,
    lambda1: f64,
    lambda2: f64,
    ridge_l2: f64,
    monotone: &[i8],
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, GroupSelectionReport), String> {
    let opts = CcdOptions {
        l1: alpha * lambda1,
        l2: ridge_l2,
        max_iter: 1000,
        tol: 1e-7,
        monotone: monotone.to_vec(),
        groups: Some(groups.to_vec()),
        group_l2: alpha * lambda2,
    };
    let beta = solve_ccd(design, &opts, warm_start)?;
    let report = compute_group_selection_report(&beta, groups);
    Ok((beta, report))
}

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
    pub groups: Option<Vec<GroupSpec>>,
    pub group_l2: f64,
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
            groups: None,
            group_l2: 0.0,
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
    pub group_report: Option<GroupSelectionReport>,
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
    let has_groups = options
        .groups
        .as_ref()
        .is_some_and(|g| !g.is_empty())
        || options.group_l2 > 0.0;

    let (beta, summary) = if is_gaussian {
        if !has_l1 && !has_monotone && !has_groups {
            // Fast path: Exact Cholesky with QR fallback + full hypothesis testing summary
            solve_normal(design, options.l2, feature_names)?
        } else {
            // ElasticNet or monotone constraints for Gaussian: Coordinate Descent
            let (sol, _) = solve_iterative(design, LeafSolver::Ccd, options, 42, warm_start)?;
            (sol, None)
        }
    } else {
        if has_l1 || has_groups {
            return Err("L1/Group regularization currently supported for Gaussian family. Use L2 with GLM families.".to_string());
        }
        // Non-linear GLM: Quasi-Newton L-BFGS
        let (sol, _) = solve_iterative(design, LeafSolver::Lbfgs, options, 42, warm_start)?;
        (sol, None)
    };

    let intercept = beta.first().copied().unwrap_or(0.0);
    let coef = if beta.len() > 1 {
        beta[1..].to_vec()
    } else {
        Vec::new()
    };

    let group_report = options
        .groups
        .as_ref()
        .map(|grps| compute_group_selection_report(&beta, grps));

    Ok(LinearModelResult {
        intercept,
        coef,
        na_coef: Vec::new(),
        summary,
        group_report,
    })
}

/// Reason an optimizer stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Exact,
    Tolerance,
    IterationLimit,
    LineSearchFailed,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Tolerance => "tolerance",
            Self::IterationLimit => "iteration_limit",
            Self::LineSearchFailed => "line_search_failed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Convergence {
    pub iterations: usize,
    pub reason: StopReason,
}

impl Convergence {
    pub(crate) fn tolerance(iterations: usize) -> Self {
        Self {
            iterations,
            reason: StopReason::Tolerance,
        }
    }

    pub(crate) fn limit(iterations: usize) -> Self {
        Self {
            iterations,
            reason: StopReason::IterationLimit,
        }
    }

    pub fn converged(self) -> bool {
        matches!(self.reason, StopReason::Exact | StopReason::Tolerance)
    }
}

use crate::family::LeafFamily;
use crate::leaf::{LeafObjective, LeafProblemType, LeafSolver, LeafSolverDiagnostic};

const GLM_FAMILIES: &[LeafFamily] = &[
    LeafFamily::Gaussian,
    LeafFamily::Binomial,
    LeafFamily::Probit,
    LeafFamily::Poisson,
    LeafFamily::Gamma,
    LeafFamily::Tweedie,
    LeafFamily::Tobit,
    LeafFamily::Quantile,
    LeafFamily::Huber,
    LeafFamily::Expectile,
    LeafFamily::NegativeBinomial,
    LeafFamily::Beta,
];

/// Solver memory profile used for leaf compatibility and cost comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverMemoryProfile {
    /// Linear memory in parameter count.
    Linear,
    /// Quadratic memory in parameter count.
    Quadratic,
    /// Population memory proportional to population size and parameter count.
    Population,
}

/// Capabilities declared once for each leaf solver.
#[derive(Clone, Copy, Debug)]
pub struct SolverCapabilities {
    /// Solver consumes exact gradients.
    pub supports_gradients: bool,
    /// Solver can operate without derivatives.
    pub supports_derivative_free: bool,
    /// Solver supports L1 penalties.
    pub supports_l1: bool,
    /// Solver supports monotone coefficient constraints.
    pub supports_monotone: bool,
    /// Solver supports finite box bounds.
    pub supports_finite_bounds: bool,
    /// Solver supports unbounded parameter domains.
    pub supports_unbounded: bool,
    /// Solver supports equality constraints.
    pub supports_equality_constraints: bool,
    /// Solver supports nonlinear inequality constraints.
    pub supports_nonlinear_inequality_constraints: bool,
    /// Response families accepted by the solver.
    pub supported_families: &'static [LeafFamily],
    /// Memory profile of the solver.
    pub memory_profile: SolverMemoryProfile,
}

/// Requirements introduced by leaf controls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeafSolverRequirements {
    /// Leaf requests an L1 penalty.
    pub l1: bool,
    /// Leaf requests monotone coefficient constraints.
    pub monotone: bool,
}

/// Result of one leaf-solver compatibility lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafSolverCompatibility {
    /// Solver selected by the caller.
    pub solver: LeafSolver,
    /// Solver participates in the declared leaf problem.
    pub solver_applies: bool,
}

/// Returns the single capability declaration for one solver.
pub fn solver_capabilities(solver: LeafSolver) -> SolverCapabilities {
    match solver {
        LeafSolver::Irls => SolverCapabilities {
            supports_gradients: true,
            supports_derivative_free: false,
            supports_l1: true,
            supports_monotone: true,
            supports_finite_bounds: false,
            supports_unbounded: true,
            supports_equality_constraints: false,
            supports_nonlinear_inequality_constraints: false,
            supported_families: GLM_FAMILIES,
            memory_profile: SolverMemoryProfile::Quadratic,
        },
        LeafSolver::Lbfgs => SolverCapabilities {
            supports_gradients: true,
            supports_derivative_free: false,
            supports_l1: false,
            supports_monotone: false,
            supports_finite_bounds: false,
            supports_unbounded: true,
            supports_equality_constraints: false,
            supports_nonlinear_inequality_constraints: false,
            supported_families: GLM_FAMILIES,
            memory_profile: SolverMemoryProfile::Linear,
        },
        LeafSolver::Ccd => SolverCapabilities {
            supports_gradients: false,
            supports_derivative_free: true,
            supports_l1: true,
            supports_monotone: true,
            supports_finite_bounds: false,
            supports_unbounded: true,
            supports_equality_constraints: false,
            supports_nonlinear_inequality_constraints: false,
            supported_families: &[LeafFamily::Gaussian],
            memory_profile: SolverMemoryProfile::Linear,
        },
        LeafSolver::CmaEs
        | LeafSolver::Shade
        | LeafSolver::NaturalGradient
        | LeafSolver::GeneticAlgorithm => SolverCapabilities {
            supports_gradients: false,
            supports_derivative_free: true,
            supports_l1: true,
            supports_monotone: true,
            supports_finite_bounds: true,
            supports_unbounded: true,
            supports_equality_constraints: false,
            supports_nonlinear_inequality_constraints: false,
            supported_families: GLM_FAMILIES,
            memory_profile: SolverMemoryProfile::Population,
        },
    }
}

/// Checks one leaf problem, response family, and solver pairing.
///
/// Closed-form and internally fitted leaves accept every solver and report
/// that the solver does not apply. Other leaves return one precise missing
/// capability when pairing is unsupported.
pub fn check_leaf_solver_compatibility(
    leaf: crate::leaf::LeafKind,
    family: LeafFamily,
    solver: LeafSolver,
    requirements: LeafSolverRequirements,
) -> Result<LeafSolverCompatibility, String> {
    let problem = LeafProblemType::for_leaf(leaf, family);
    let capabilities = solver_capabilities(solver);
    match problem {
        LeafProblemType::ClosedFormMean(_)
        | LeafProblemType::ClosedFormMeanTransform(_)
        | LeafProblemType::InternalMars(_)
        | LeafProblemType::Knn(_)
        | LeafProblemType::MulticlassGlm(_)
        | LeafProblemType::Survival(_)
        | LeafProblemType::Heteroscedastic(_)
        | LeafProblemType::Aft(_)
        | LeafProblemType::Cure(_)
        | LeafProblemType::Spline(_) => Ok(LeafSolverCompatibility {
            solver,
            solver_applies: false,
        }),
        LeafProblemType::WeightedMedian(_) | LeafProblemType::WeightedQuantile { .. } => {
            if solver == LeafSolver::Irls {
                return Ok(LeafSolverCompatibility {
                    solver,
                    solver_applies: false,
                });
            }
            if !capabilities.supports_derivative_free {
                return Err(format!(
                    "solver {solver:?} does not support gradients for the nonsmooth leaf objective"
                ));
            }
            Ok(LeafSolverCompatibility {
                solver,
                solver_applies: true,
            })
        }
        LeafProblemType::HuberLocation(_)
        | LeafProblemType::HuberRegression { .. }
        | LeafProblemType::ExpectileLocation { .. }
        | LeafProblemType::ExpectileRegression { .. } => {
            if solver == LeafSolver::Irls {
                return Ok(LeafSolverCompatibility {
                    solver,
                    solver_applies: false,
                });
            }
            if !capabilities.supports_gradients && !capabilities.supports_derivative_free {
                return Err(format!(
                    "solver {solver:?} supports neither gradients nor derivative-free operation"
                ));
            }
            Ok(LeafSolverCompatibility {
                solver,
                solver_applies: true,
            })
        }
        LeafProblemType::NegativeBinomialGlm { .. } | LeafProblemType::BetaGlm { .. } => {
            Ok(LeafSolverCompatibility {
                solver,
                solver_applies: true,
            })
        }
        LeafProblemType::PenalizedGlm(_) => {
            if !capabilities.supported_families.contains(&family) {
                return Err(format!(
                    "solver {solver:?} does not support the {family:?} response family"
                ));
            }
            if requirements.l1 && !capabilities.supports_l1 {
                return Err(format!("solver {solver:?} does not support L1 penalties"));
            }
            if requirements.monotone && !capabilities.supports_monotone {
                return Err(format!(
                    "solver {solver:?} does not support monotone coefficient constraints"
                ));
            }
            if !capabilities.supports_gradients && !capabilities.supports_derivative_free {
                return Err(format!(
                    "solver {solver:?} supports neither gradients nor derivative-free operation"
                ));
            }
            if !capabilities.supports_unbounded {
                return Err(format!(
                    "solver {solver:?} does not support unbounded parameters"
                ));
            }
            Ok(LeafSolverCompatibility {
                solver,
                solver_applies: true,
            })
        }
        LeafProblemType::Unknown => Ok(LeafSolverCompatibility {
            solver,
            solver_applies: false,
        }),
    }
}

/// Builds the persisted solver diagnostic for one leaf pairing.
pub fn leaf_solver_diagnostic(
    leaf: crate::leaf::LeafKind,
    family: LeafFamily,
    solver: LeafSolver,
    requirements: LeafSolverRequirements,
) -> LeafSolverDiagnostic {
    match check_leaf_solver_compatibility(leaf, family, solver, requirements) {
        Ok(compatibility) if compatibility.solver_applies => LeafSolverDiagnostic::Applied(solver),
        Ok(_) => LeafSolverDiagnostic::NotApplicable(solver),
        Err(reason) => LeafSolverDiagnostic::Rejected { solver, reason },
    }
}

pub(crate) fn solve_iterative<D: DesignMatrix>(
    design: &D,
    solver: LeafSolver,
    options: &SolverOptions,
    seed: u64,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, Convergence), String> {
    match solver {
        LeafSolver::Ccd => ccd::solve_ccd_report(
            design,
            &CcdOptions {
                l1: options.l1,
                l2: options.l2,
                max_iter: options.max_iter,
                tol: options.tol,
                monotone: options.monotone.clone(),
                groups: options.groups.clone(),
                group_l2: options.group_l2,
            },
            warm_start,
        ),
        LeafSolver::Lbfgs => lbfgs::solve_lbfgs_report(
            design,
            &LbfgsOptions {
                family: options.family,
                l2: options.l2,
                max_iter: options.max_iter,
                m: 6,
                tol: options.tol,
                verbose: options.verbose,
            },
            warm_start,
        ),
        LeafSolver::Irls => Err("Use the IRLS design solver".into()),
        solver => {
            let kind = EvolutionaryKind::from_leaf_solver(solver)
                .expect("non-evolutionary solvers are handled above");
            evolutionary::solve_evolutionary_report(
                design,
                &EvolutionaryOptions {
                    kind,
                    family: options.family,
                    l1: options.l1,
                    l2: options.l2,
                    monotone: options.monotone.clone(),
                    population_size: None,
                    max_generations: options.max_iter,
                    tol: options.tol,
                    seed,
                    verbose: options.verbose,
                },
                warm_start,
            )
        }
    }
}

/// Linear leaf problem passed to the unified solver entry point.
pub(crate) struct LeafLinearProblem<'a> {
    pub cols: &'a [Vec<Option<f64>>],
    pub y: &'a [f64],
    pub w: &'a [f64],
    pub rows: &'a [usize],
    pub chosen: &'a [usize],
    pub with_indicators: bool,
    pub impute_means: Option<&'a [f64]>,
    pub family: LeafFamily,
    pub solver: LeafSolver,
    pub l1: f64,
    pub l2: f64,
    pub monotone: &'a [i8],
    pub mean: f64,
    pub extra: Option<&'a std::collections::HashMap<String, f64>>,
    pub tweedie_p: f64,
    pub tobit_threshold: f64,
    pub verbose: u8,
    pub groups: Option<Vec<crate::solver::groups::GroupSpec>>,
    pub group_l2: f64,
}

/// One leaf optimization problem routed through the unified solver entry.
pub(crate) enum LeafFitProblem<'a> {
    Linear(LeafLinearProblem<'a>),
    Location {
        solver: LeafSolver,
        objective: LeafObjective,
        max_iter: usize,
        tol: f64,
    },
}

/// Result of one unified leaf solve.
pub(crate) struct LeafSolveOutput {
    pub linear: Option<crate::node_stats::LinearModel>,
    pub center: Option<f64>,
    pub diagnostic: LeafSolverDiagnostic,
}

/// Solves one declared leaf optimization problem.
///
/// Solver identity dispatch stays inside this module. Leaf construction only
/// supplies problem data and maps the returned coefficients or location back
/// into its model.
pub(crate) fn solve_leaf_problem(problem: &LeafFitProblem<'_>) -> Result<LeafSolveOutput, String> {
    match problem {
        LeafFitProblem::Linear(problem) => solve_linear_leaf(problem),
        LeafFitProblem::Location {
            solver,
            objective,
            max_iter,
            tol,
        } => solve_location_leaf(*solver, objective, *max_iter, *tol),
    }
}

fn solve_linear_leaf(problem: &LeafLinearProblem<'_>) -> Result<LeafSolveOutput, String> {
    check_leaf_solver_compatibility(
        crate::leaf::LeafKind::Linear,
        problem.family,
        problem.solver,
        LeafSolverRequirements {
            l1: problem.l1 != 0.0,
            monotone: problem.monotone.iter().any(|&sign| sign != 0),
        },
    )?;
    let view = ColumnSliceView::new_with_imputation(
        problem.cols,
        problem.y,
        problem.w,
        problem.rows,
        problem.chosen,
        problem.with_indicators,
        problem.impute_means,
    );
    let nc = problem.chosen.len();
    if problem.family == LeafFamily::Quantile {
        if problem.rows.len() < nc + 1 {
            return Err(format!(
                "insufficient leaf rows for quantile regression: rows {} < features {} + 1 (minimum support rule)",
                problem.rows.len(),
                nc
            ));
        }
        let tau = problem
            .extra
            .and_then(|e| e.get("tau"))
            .copied()
            .unwrap_or(0.5);
        let solution = crate::solver::problems::quantile::solve_quantile_lbfgs(
            &view,
            tau,
            1e-4,
            problem.l2,
            250,
            1e-7,
            None,
        )?;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("quantile solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    if problem.family == LeafFamily::Huber {
        if problem.rows.len() < nc + 1 {
            return Err(format!(
                "insufficient leaf rows for Huber regression: rows {} < features {} + 1 (minimum support rule)",
                problem.rows.len(),
                nc
            ));
        }
        let delta = problem
            .extra
            .and_then(|e| e.get("huber_delta"))
            .copied()
            .unwrap_or(1.345);
        let result = crate::solver::problems::huber::solve_huber_regression(
            &view,
            delta,
            problem.l2,
            15,
            60,
            1e-4,
        )?;
        let solution = result.coefficients;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("Huber solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    let non_neg = problem
        .extra
        .and_then(|e| e.get("non_negative"))
        .copied()
        .unwrap_or(0.0)
        > 0.5;
    if non_neg {
        let non_neg_intercept = problem
            .extra
            .and_then(|e| e.get("non_negative_intercept"))
            .copied()
            .unwrap_or(0.0)
            > 0.5;
        let result = crate::solver::problems::nnls::solve_nnls(
            &view,
            problem.l2,
            non_neg_intercept,
            3 * view.n_features() + 30,
            1e-8,
        )?;
        let solution = result.coefficients;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("NNLS solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    if problem.family == LeafFamily::Expectile {
        if problem.rows.len() < nc + 1 {
            return Err(format!(
                "insufficient leaf rows for expectile regression: rows {} < features {} + 1 (minimum support rule)",
                problem.rows.len(),
                nc
            ));
        }
        let tau = problem
            .extra
            .and_then(|e| e.get("expectile_tau"))
            .copied()
            .unwrap_or(0.5);
        let solution = crate::solver::problems::expectile::solve_expectile_lbfgs(
            &view,
            tau,
            problem.l2,
            100,
            1e-4,
            None,
        )?;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("expectile solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    if problem.family == LeafFamily::NegativeBinomial {
        if problem.rows.len() < nc + 1 {
            return Err(format!(
                "insufficient leaf rows for negative binomial regression: rows {} < features {} + 1 (minimum support rule)",
                problem.rows.len(),
                nc
            ));
        }
        let initial_alpha = problem
            .extra
            .and_then(|e| e.get("nb_alpha"))
            .copied();
        let result = crate::solver::problems::glm_extra::solve_negative_binomial_regression(
            &view,
            initial_alpha,
            problem.l2,
            15,
            1e-5,
        )?;
        let solution = result.coefficients;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("negative binomial solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    if problem.family == LeafFamily::Beta {
        if problem.rows.len() < nc + 1 {
            return Err(format!(
                "insufficient leaf rows for beta regression: rows {} < features {} + 1 (minimum support rule)",
                problem.rows.len(),
                nc
            ));
        }
        let initial_phi = problem
            .extra
            .and_then(|e| e.get("beta_phi"))
            .copied();
        let result = crate::solver::problems::glm_extra::solve_beta_regression(
            &view,
            initial_phi,
            problem.l2,
            15,
            1e-5,
        )?;
        let solution = result.coefficients;
        if !solution.iter().all(|value| value.is_finite())
            || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
        {
            return Err("beta solver produced a non-finite leaf model".into());
        }
        let mut model = crate::node_stats::LinearModel {
            cols: problem.chosen.to_vec(),
            coef: solution[1..1 + nc].to_vec(),
            na_coef: if problem.with_indicators {
                solution[1 + nc..].to_vec()
            } else {
                Vec::new()
            },
            feature_means: Vec::new(),
            intercept: solution[0],
            summary: None,
        };
        apply_feature_means(&mut model, problem.impute_means);
        return Ok(LeafSolveOutput {
            linear: Some(model),
            center: None,
            diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
        });
    }
    match problem.solver {
        LeafSolver::Irls if problem.family == LeafFamily::Gaussian => {
            let mut gram = crate::node_stats::Gram::new(view.n_features());
            let mut row = vec![0.0; view.n_features()];
            for i in 0..view.n_rows() {
                view.copy_design_row(i, &mut row);
                gram.add_row(&row, view.get_y(i), view.get_weight(i));
            }
            let mut model = crate::node_stats::fit_linear(
                &gram,
                problem.chosen,
                problem.with_indicators,
                problem.monotone,
                problem.l1,
                problem.l2,
            );
            apply_feature_means(&mut model, problem.impute_means);
            Ok(LeafSolveOutput {
                linear: Some(model),
                center: None,
                diagnostic: LeafSolverDiagnostic::Applied(LeafSolver::Irls),
            })
        }
        LeafSolver::Irls => {
            let options = SolverOptions {
                l1: problem.l1,
                l2: problem.l2,
                max_iter: 100,
                tol: 1e-11,
                verbose: problem.verbose,
                ..Default::default()
            };
            let (mut model, convergence) = crate::node_stats::fit_glm_design(
                &view,
                problem.chosen,
                problem.monotone,
                problem.family,
                problem.mean,
                problem.extra,
                &options,
            )?;
            if convergence.reason != StopReason::Tolerance {
                return Err("GLM fit did not reach tolerance".into());
            }
            apply_feature_means(&mut model, problem.impute_means);
            Ok(LeafSolveOutput {
                linear: Some(model),
                center: None,
                diagnostic: LeafSolverDiagnostic::Applied(LeafSolver::Irls),
            })
        }
        LeafSolver::Lbfgs => {
            let family = glm_family(
                problem.family,
                problem.extra,
                problem.tweedie_p,
                problem.tobit_threshold,
            );
            let family = family.ok_or_else(|| {
                format!(
                    "solver {:?} does not support the {:?} response family",
                    LeafSolver::Lbfgs,
                    problem.family
                )
            })?;
            let options = SolverOptions {
                family,
                l2: problem.l2,
                max_iter: 250,
                tol: 1e-9,
                verbose: problem.verbose,
                ..Default::default()
            };
            let (solution, _) = solve_iterative(&view, LeafSolver::Lbfgs, &options, 42, None)?;
            if !solution.iter().all(|value| value.is_finite())
                || (0..view.n_rows()).any(|i| !view.eval_row(i, &solution).is_finite())
            {
                return Err("L-BFGS produced a non-finite leaf model".into());
            }
            let mut model = crate::node_stats::LinearModel {
                cols: problem.chosen.to_vec(),
                coef: solution[1..1 + nc].to_vec(),
                na_coef: if problem.with_indicators {
                    solution[1 + nc..].to_vec()
                } else {
                    Vec::new()
                },
                feature_means: Vec::new(),
                intercept: solution[0],
                summary: None,
            };
            apply_feature_means(&mut model, problem.impute_means);
            Ok(LeafSolveOutput {
                linear: Some(model),
                center: None,
                diagnostic: LeafSolverDiagnostic::Applied(LeafSolver::Lbfgs),
            })
        }
        LeafSolver::Ccd => {
            let options = SolverOptions {
                l1: problem.l1,
                l2: problem.l2,
                max_iter: 1000,
                tol: 1e-7,
                monotone: problem.monotone.to_vec(),
                groups: problem.groups.clone(),
                group_l2: problem.group_l2,
                ..Default::default()
            };
            match solve_iterative(&view, LeafSolver::Ccd, &options, 42, None) {
                Ok((solution, _)) => {
                    let mut model = crate::node_stats::LinearModel {
                        cols: problem.chosen.to_vec(),
                        coef: solution[1..1 + nc].to_vec(),
                        na_coef: if problem.with_indicators {
                            solution[1 + nc..].to_vec()
                        } else {
                            Vec::new()
                        },
                        feature_means: Vec::new(),
                        intercept: solution[0],
                        summary: None,
                    };
                    apply_feature_means(&mut model, problem.impute_means);
                    Ok(LeafSolveOutput {
                        linear: Some(model),
                        center: None,
                        diagnostic: LeafSolverDiagnostic::Applied(LeafSolver::Ccd),
                    })
                }
                Err(_) => Ok(LeafSolveOutput {
                    linear: Some(crate::node_stats::LinearModel {
                        cols: problem.chosen.to_vec(),
                        coef: vec![0.0; nc],
                        na_coef: Vec::new(),
                        feature_means: Vec::new(),
                        intercept: problem.mean,
                        summary: None,
                    }),
                    center: None,
                    diagnostic: LeafSolverDiagnostic::Applied(LeafSolver::Ccd),
                }),
            }
        }
        LeafSolver::CmaEs
        | LeafSolver::Shade
        | LeafSolver::NaturalGradient
        | LeafSolver::GeneticAlgorithm => {
            let family = glm_family(
                problem.family,
                problem.extra,
                problem.tweedie_p,
                problem.tobit_threshold,
            )
            .ok_or_else(|| {
                format!(
                    "solver {:?} does not support the {:?} response family",
                    problem.solver, problem.family
                )
            })?;
            let options = SolverOptions {
                family,
                l1: problem.l1,
                l2: problem.l2,
                monotone: problem.monotone.to_vec(),
                max_iter: 100,
                tol: 1e-6,
                verbose: problem.verbose,
                groups: problem.groups.clone(),
                group_l2: problem.group_l2,
            };
            let (solution, _) = solve_iterative(&view, problem.solver, &options, 42, None)?;
            let mut model = crate::node_stats::LinearModel {
                cols: problem.chosen.to_vec(),
                coef: solution[1..1 + nc].to_vec(),
                na_coef: if problem.with_indicators {
                    solution[1 + nc..].to_vec()
                } else {
                    Vec::new()
                },
                feature_means: Vec::new(),
                intercept: solution[0],
                summary: None,
            };
            apply_feature_means(&mut model, problem.impute_means);
            Ok(LeafSolveOutput {
                linear: Some(model),
                center: None,
                diagnostic: LeafSolverDiagnostic::Applied(problem.solver),
            })
        }
    }
}

fn solve_location_leaf(
    solver: LeafSolver,
    objective: &LeafObjective,
    max_iter: usize,
    tol: f64,
) -> Result<LeafSolveOutput, String> {
    let leaf = match objective {
        LeafObjective::WeightedMedian { .. } => crate::leaf::LeafKind::Median,
        LeafObjective::WeightedQuantile { tau, .. } => crate::leaf::LeafKind::Quantile(*tau),
        LeafObjective::Huber(_) => crate::leaf::LeafKind::Huber,
    };
    check_leaf_solver_compatibility(
        leaf,
        LeafFamily::Gaussian,
        solver,
        LeafSolverRequirements::default(),
    )?;
    if solver == LeafSolver::Irls {
        let center = match objective {
            LeafObjective::WeightedMedian {
                values, weights, ..
            } => crate::stats::w_median(
                &values
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            LeafObjective::WeightedQuantile {
                values,
                weights,
                tau,
                ..
            } => crate::stats::w_quantile(
                &values
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .collect::<Vec<_>>(),
                *tau,
            ),
            LeafObjective::Huber(objective) => crate::stats::huber_location(
                &objective
                    .values()
                    .iter()
                    .copied()
                    .zip(objective.weights().iter().copied())
                    .collect::<Vec<_>>(),
            ),
        };
        return Ok(LeafSolveOutput {
            linear: None,
            center: Some(center),
            diagnostic: LeafSolverDiagnostic::NotApplicable(solver),
        });
    }

    let center = match (solver, objective) {
        (LeafSolver::Lbfgs, LeafObjective::Huber(objective)) => {
            solve_huber_location(objective, max_iter, tol)?
        }
        (
            LeafSolver::Ccd
            | LeafSolver::CmaEs
            | LeafSolver::Shade
            | LeafSolver::NaturalGradient
            | LeafSolver::GeneticAlgorithm,
            _,
        ) => golden_section_location(objective, max_iter, tol)?,
        (LeafSolver::Lbfgs, _) => {
            return Err(
                "solver Lbfgs does not support gradients for the nonsmooth leaf objective".into(),
            )
        }
        _ => {
            return Err(format!(
                "solver {solver:?} does not support the leaf objective"
            ))
        }
    };
    Ok(LeafSolveOutput {
        linear: None,
        center: Some(center),
        diagnostic: LeafSolverDiagnostic::Applied(solver),
    })
}

fn golden_section_location(
    objective: &LeafObjective,
    max_iter: usize,
    tol: f64,
) -> Result<f64, String> {
    let mut lower = objective.bounds().lower()[0];
    let mut upper = objective.bounds().upper()[0];
    if (upper - lower).abs() <= f64::EPSILON {
        return Ok(lower);
    }
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut left = upper - ratio * (upper - lower);
    let mut right = lower + ratio * (upper - lower);
    let mut left_value = objective
        .value(&[left])
        .map_err(|error| error.to_string())?;
    let mut right_value = objective
        .value(&[right])
        .map_err(|error| error.to_string())?;
    let iterations = max_iter.max(64);
    for _ in 0..iterations {
        if (upper - lower).abs() <= tol * (1.0 + upper.abs()) {
            break;
        }
        if left_value <= right_value {
            upper = right;
            right = left;
            right_value = left_value;
            left = upper - ratio * (upper - lower);
            left_value = objective
                .value(&[left])
                .map_err(|error| error.to_string())?;
        } else {
            lower = left;
            left = right;
            left_value = right_value;
            right = lower + ratio * (upper - lower);
            right_value = objective
                .value(&[right])
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(0.5 * (lower + upper))
}

fn solve_huber_location(
    objective: &crate::leaf::LeafHuberObjective,
    max_iter: usize,
    tol: f64,
) -> Result<f64, String> {
    let mut sum_weight = 0.0;
    let mut sum_value = 0.0;
    for (&value, &weight) in objective.values().iter().zip(objective.weights()) {
        sum_weight += weight;
        sum_value += weight * value;
    }
    let mut location = if sum_weight > 0.0 {
        sum_value / sum_weight
    } else {
        0.0
    };
    for _ in 0..max_iter.max(64) {
        let gradient = objective
            .gradient(&[location])
            .map_err(|error| error.to_string())?[0];
        let curvature = objective.curvature(location);
        if curvature <= f64::EPSILON {
            break;
        }
        let step = gradient / curvature;
        let lower = objective.bounds().lower()[0];
        let upper = objective.bounds().upper()[0];
        let next = (location - step).clamp(lower, upper);
        if (next - location).abs() <= tol * (1.0 + location.abs()) {
            location = next;
            break;
        }
        location = next;
    }
    Ok(location)
}

fn apply_feature_means(model: &mut crate::node_stats::LinearModel, impute_means: Option<&[f64]>) {
    if let Some(means) = impute_means {
        model.feature_means = means.iter().copied().map(Some).collect();
    }
}

fn glm_family(
    family: LeafFamily,
    extra: Option<&std::collections::HashMap<String, f64>>,
    tweedie_p: f64,
    tobit_threshold: f64,
) -> Option<GlmFamily> {
    match family {
        LeafFamily::Gaussian => Some(GlmFamily::Gaussian),
        LeafFamily::Binomial => Some(GlmFamily::Binomial),
        LeafFamily::Probit => Some(GlmFamily::Probit),
        LeafFamily::Poisson => Some(GlmFamily::Poisson),
        LeafFamily::Gamma => Some(GlmFamily::Gamma),
        LeafFamily::Tweedie => Some(GlmFamily::Tweedie {
            p: extra
                .and_then(|values| values.get("tweedie_p"))
                .copied()
                .unwrap_or(tweedie_p),
        }),
        LeafFamily::Tobit => Some(GlmFamily::Tobit {
            threshold: extra
                .and_then(|values| values.get("tobit_threshold"))
                .copied()
                .unwrap_or(tobit_threshold),
        }),
        _ => None,
    }
}
