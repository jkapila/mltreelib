//! Shared contracts for the generic bounded optimization seam.
//!
//! Every optimizer in this module uses the same objective, bounds, constraint,
//! budget, request, and result contracts. Methods never carry a private status
//! vocabulary, and budget exhaustion never reports success.

pub mod annealing;
pub mod basin_hopping;
pub mod bfgs;
pub mod bounds;
pub mod budget;
pub mod conjugate_gradient;
pub mod constraint;
pub mod finite_difference;
pub mod lbfgs_b;
pub mod line_search;
pub mod nelder_mead;
pub mod objective;
pub mod powell;
pub mod random;
pub mod request;
pub mod result;

pub use annealing::simulated_annealing;
pub use basin_hopping::basin_hopping;
pub use bfgs::bfgs;
pub use bounds::Bounds;
pub use budget::{Budget, BudgetStatus};
pub use conjugate_gradient::nonlinear_conjugate_gradient;
pub use constraint::{
    is_feasible, max_violation, violation, Constraint, ConstraintEvaluator, ConstraintKind,
};
pub use finite_difference::FiniteDifferenceObjective;
pub use lbfgs_b::solve_lbfgs_b;
pub use line_search::{strong_wolfe, LineSearchOptions, LineSearchResult};
pub use nelder_mead::nelder_mead;
pub use objective::{
    require_dimension, require_finite, CountedObjective, Objective, ObjectiveError, SmoothObjective,
};
pub use powell::powell;
pub use random::{latin_hypercube_samples, latin_hypercube_search, random_search};
pub use request::{
    AnnealingOptions, BasinHoppingOptions, BfgsOptions, BoundaryPolicy, CobylaOptions,
    LatinHypercubeOptions, LbfgsBOptions, MethodOptions, NelderMeadOptions, NonlinearCgOptions,
    OptimizationRequest, OptimizerKind, PowellOptions, RandomSearchOptions, SharedOptions,
    DEFAULT_SEED,
};
pub use result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Step size used by the central-difference gradient fallback.
pub const FINITE_DIFFERENCE_STEP: f64 = 1e-6;

/// Runs one optimization with the shared contracts.
///
/// The method dispatch resolves the method-specific options, evaluates the
/// objective through one shared budget, and returns the shared result type.
/// A method that is not implemented yet returns one error naming the method.
pub fn optimize(request: &OptimizationRequest) -> Result<OptimizationResult, String> {
    request.validate()?;
    let budget = Budget::new(
        request.shared.max_evaluations,
        request.shared.max_iterations,
    );
    optimize_with_budget(request, &budget)
}

/// Runs one optimization against a budget that the caller owns.
///
/// A composition method uses this entry point so the inner local solve spends
/// the outer budget and no objective call escapes the count.
pub fn optimize_with_budget(
    request: &OptimizationRequest,
    budget: &Budget,
) -> Result<OptimizationResult, String> {
    request.validate()?;
    // Smooth methods use the supplied gradient source or the central-difference
    // fallback. The fallback wraps the counted objective, so its perturbation
    // calls land in the same budget as the solver's own calls.
    let counted = CountedObjective::new(request.objective, budget);
    let finite_difference = FiniteDifferenceObjective::new(&counted, FINITE_DIFFERENCE_STEP);
    let smooth: &dyn SmoothObjective = match request.smooth {
        Some(smooth) => smooth,
        None => &finite_difference as &dyn SmoothObjective,
    };

    match request.method {
        OptimizerKind::RandomSearch => {
            let options = match &request.options {
                MethodOptions::Random(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(random::random_search(
                request.objective,
                budget,
                options,
                request.shared.seed,
                request.shared.initial_population.as_deref(),
            ))
        }
        OptimizerKind::LatinHypercube => {
            let options = match &request.options {
                MethodOptions::LatinHypercube(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(random::latin_hypercube_search(
                request.objective,
                budget,
                options,
                request.shared.seed,
            ))
        }
        OptimizerKind::LbfgsB => {
            let options = match &request.options {
                MethodOptions::LbfgsB(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(lbfgs_b::solve_lbfgs_b(
                request.objective,
                budget,
                options,
                request.start.as_deref(),
                smooth,
            ))
        }
        OptimizerKind::Bfgs => {
            let options = match &request.options {
                MethodOptions::Bfgs(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(bfgs::bfgs(
                request.objective,
                smooth,
                budget,
                options,
                request.start.as_deref(),
            ))
        }
        OptimizerKind::NonlinearConjugateGradient => {
            let options = match &request.options {
                MethodOptions::NonlinearCg(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(conjugate_gradient::nonlinear_conjugate_gradient(
                request.objective,
                smooth,
                budget,
                options,
                request.start.as_deref(),
            ))
        }
        OptimizerKind::NelderMead => {
            let options = match &request.options {
                MethodOptions::NelderMead(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(nelder_mead::nelder_mead(
                request.objective,
                budget,
                options,
                request.start.as_deref(),
            ))
        }
        OptimizerKind::Powell => {
            let options = match &request.options {
                MethodOptions::Powell(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(powell::powell(
                request.objective,
                budget,
                options,
                request.start.as_deref(),
            ))
        }
        OptimizerKind::SimulatedAnnealing => {
            let options = match &request.options {
                MethodOptions::Annealing(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(annealing::simulated_annealing(
                request.objective,
                budget,
                options,
                request.shared.seed,
                request.start.as_deref(),
            ))
        }
        OptimizerKind::BasinHopping => {
            let options = match &request.options {
                MethodOptions::BasinHopping(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(basin_hopping::basin_hopping(
                request.objective,
                smooth,
                budget,
                options,
                request.shared.seed,
                request.start.as_deref(),
            ))
        }
        method => Err(format!("method {} is not implemented yet", method.as_str())),
    }
}

/// Builds the inner request of a composition method.
///
/// The inner request spends the remaining outer budget and uses the same
/// gradient source.
pub(crate) fn inner_request<'a>(
    objective: &'a dyn Objective,
    smooth: &'a dyn SmoothObjective,
    method: OptimizerKind,
    start: Option<Vec<f64>>,
    budget: &Budget,
    seed: u64,
) -> OptimizationRequest<'a> {
    let shared = SharedOptions {
        max_evaluations: budget.remaining_evaluations(),
        max_iterations: None,
        tol: 1e-6,
        seed,
        initial_population: None,
        verbose: 0,
    };
    let mut request = OptimizationRequest::new(objective, Some(smooth), method, shared);
    if let Some(start) = start {
        request = request.with_start(start);
    }
    request
}
