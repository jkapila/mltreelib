//! Shared contracts for the generic bounded optimization seam.
//!
//! Every optimizer in this module uses the same objective, bounds, constraint,
//! budget, request, and result contracts. Methods never carry a private status
//! vocabulary, and budget exhaustion never reports success.

pub mod bounds;
pub mod budget;
pub mod constraint;
pub mod objective;
pub mod random;
pub mod request;
pub mod result;

pub use bounds::Bounds;
pub use budget::{Budget, BudgetStatus};
pub use constraint::{
    is_feasible, max_violation, violation, Constraint, ConstraintEvaluator, ConstraintKind,
};
pub use objective::{
    require_dimension, require_finite, Objective, ObjectiveError, SmoothObjective,
};
pub use random::{latin_hypercube_samples, latin_hypercube_search, random_search};
pub use request::{
    AnnealingOptions, BasinHoppingOptions, BfgsOptions, BoundaryPolicy, CobylaOptions,
    LatinHypercubeOptions, LbfgsBOptions, MethodOptions, NelderMeadOptions, NonlinearCgOptions,
    OptimizationRequest, OptimizerKind, PowellOptions, RandomSearchOptions, SharedOptions,
    DEFAULT_SEED,
};
pub use result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Runs one optimization with the shared contracts.
///
/// The method dispatch resolves the method-specific options, evaluates the
/// objective through one shared budget, and returns the shared result type.
/// A method that is not implemented yet returns one error naming the method.
pub fn optimize(request: &OptimizationRequest) -> Result<OptimizationResult, String> {
    request.validate()?;
    let mut budget = Budget::new(
        request.shared.max_evaluations,
        request.shared.max_iterations,
    );

    match request.method {
        OptimizerKind::RandomSearch => {
            let options = match &request.options {
                MethodOptions::Random(options) => options,
                _ => return Err("method options do not match the selected method".to_string()),
            };
            Ok(random::random_search(
                request.objective,
                &mut budget,
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
                &mut budget,
                options,
                request.shared.seed,
            ))
        }
        method => Err(format!("method {} is not implemented yet", method.as_str())),
    }
}
