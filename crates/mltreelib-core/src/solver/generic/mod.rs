//! Shared contracts for the generic bounded optimization seam.
//!
//! Every optimizer in this module uses the same objective, bounds, constraint,
//! budget, request, and result contracts. Methods never carry a private status
//! vocabulary, and budget exhaustion never reports success.

pub mod bounds;
pub mod budget;
pub mod constraint;
pub mod objective;
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
pub use request::{
    AnnealingOptions, BasinHoppingOptions, BfgsOptions, BoundaryPolicy, CobylaOptions,
    LatinHypercubeOptions, LbfgsBOptions, MethodOptions, NelderMeadOptions, NonlinearCgOptions,
    OptimizationRequest, OptimizerKind, PowellOptions, RandomSearchOptions, SharedOptions,
    DEFAULT_SEED,
};
pub use result::{HistoryEntry, OptimizationResult, SolveStatus};
