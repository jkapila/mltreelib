//! Shared contracts for the generic bounded optimization seam.
//!
//! Every optimizer in this module uses the same objective, bounds, constraint,
//! budget, request, and result contracts. Methods never carry a private status
//! vocabulary, and budget exhaustion never reports success.

pub mod bounds;
pub mod budget;
pub mod constraint;
pub mod finite_difference;
pub mod objective;
pub mod random;
pub mod request;
pub mod result;

pub use bounds::Bounds;
pub use budget::{Budget, BudgetStatus};
pub use constraint::{
    is_feasible, max_violation, violation, Constraint, ConstraintEvaluator, ConstraintKind,
};
pub use finite_difference::FiniteDifferenceObjective;
pub use objective::{
    require_dimension, require_finite, CountedObjective, Objective, ObjectiveError, SmoothObjective,
};
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
    // Smooth methods use the supplied gradient source or the central-difference
    // fallback. The fallback wraps the counted objective, so its perturbation
    // calls land in the same budget as the solver's own calls.
    let counted = CountedObjective::new(request.objective, &budget);
    let finite_difference = FiniteDifferenceObjective::new(&counted, FINITE_DIFFERENCE_STEP);
    let _smooth: Option<&dyn SmoothObjective> = match request.smooth {
        Some(smooth) => Some(smooth),
        None => Some(&finite_difference),
    };

    match request.method {
        OptimizerKind::RandomSearch => {
            let options = match &request.options {
                MethodOptions::Random(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(random::random_search(
                request.objective,
                &budget,
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
                &budget,
                options,
                request.shared.seed,
            ))
        }
        method => Err(format!("method {} is not implemented yet", method.as_str())),
    }
}
