//! Objective contracts for the generic optimization seam.

use std::fmt;

use super::bounds::Bounds;
use super::budget::{Budget, BudgetStatus};

/// Failure reported by an objective or gradient evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectiveError {
    /// The point length differs from the objective dimension.
    DimensionMismatch { expected: usize, actual: usize },
    /// The objective returned a NaN or infinite value.
    NotFinite(f64),
    /// The objective callback failed.
    Callback(String),
    /// The evaluation budget ended the run.
    BudgetExhausted(BudgetStatus),
}

impl fmt::Display for ObjectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "objective dimension mismatch: expected {expected} values, got {actual}"
            ),
            Self::NotFinite(value) => write!(f, "objective value is not finite: {value}"),
            Self::Callback(message) => write!(f, "objective callback failed: {message}"),
            Self::BudgetExhausted(status) => write!(f, "budget exhausted: {status:?}"),
        }
    }
}

impl std::error::Error for ObjectiveError {}

/// A bounded scalar objective.
///
/// Implementations return one finite value for one point. Solvers reject
/// non-finite values through [`ObjectiveError::NotFinite`] and report the
/// [`crate::solver::generic::SolveStatus::InvalidObjective`] status.
pub trait Objective {
    /// Number of parameters.
    fn dim(&self) -> usize;

    /// Parameter domain.
    fn bounds(&self) -> &Bounds;

    /// Objective value at one point.
    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError>;
}

/// An objective that also provides an exact gradient.
pub trait SmoothObjective: Objective {
    /// Gradient at one point. The returned vector has one entry per dimension.
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError>;
}

/// Checks that an objective value is finite.
pub fn require_finite(value: f64) -> Result<f64, ObjectiveError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ObjectiveError::NotFinite(value))
    }
}

/// Checks that a point matches the objective dimension.
pub fn require_dimension(expected: usize, x: &[f64]) -> Result<(), ObjectiveError> {
    if x.len() == expected {
        Ok(())
    } else {
        Err(ObjectiveError::DimensionMismatch {
            expected,
            actual: x.len(),
        })
    }
}

/// Objective wrapper that charges the shared budget for every value call.
///
/// Wrapping the user objective keeps one counter for every objective call,
/// including the calls a finite-difference gradient makes.
pub struct CountedObjective<'a> {
    inner: &'a dyn Objective,
    budget: &'a Budget,
}

impl<'a> CountedObjective<'a> {
    /// Wraps one objective with one budget.
    pub fn new(inner: &'a dyn Objective, budget: &'a Budget) -> Self {
        Self { inner, budget }
    }
}

impl Objective for CountedObjective<'_> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds {
        self.inner.bounds()
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.budget
            .charge_evaluation()
            .map_err(ObjectiveError::BudgetExhausted)?;
        self.inner.value(x)
    }
}
