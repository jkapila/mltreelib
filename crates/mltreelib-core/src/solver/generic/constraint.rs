//! Constraint contracts for the generic optimization seam.
//!
//! An inequality constraint is feasible when its value is `>= 0`. An equality
//! constraint is feasible when its value is zero within the feasibility
//! tolerance.

/// Constraint type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Feasible when the value is `>= 0`.
    Inequality,
    /// Feasible when the absolute value is `<= tolerance`.
    Equality,
}

impl ConstraintKind {
    /// Canonical name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inequality => "inequality",
            Self::Equality => "equality",
        }
    }
}

/// Constraint values evaluated at one point.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// Constraint type shared by every value.
    pub kind: ConstraintKind,
    /// Evaluated constraint values.
    pub values: Vec<f64>,
}

impl Constraint {
    /// Builds evaluated constraint values.
    pub fn new(kind: ConstraintKind, values: Vec<f64>) -> Self {
        Self { kind, values }
    }

    /// Largest violation across all values. Feasible points report zero.
    pub fn max_violation(&self) -> f64 {
        max_violation(self.kind, &self.values)
    }

    /// Reports whether every value satisfies the constraint.
    pub fn is_feasible(&self, tolerance: f64) -> bool {
        is_feasible(self.kind, &self.values, tolerance)
    }
}

/// Largest violation of one constraint value.
///
/// An inequality value below zero violates the constraint by its magnitude.
/// An equality value violates the constraint by its absolute distance from zero.
pub fn violation(kind: ConstraintKind, value: f64) -> f64 {
    match kind {
        ConstraintKind::Inequality => {
            if value < 0.0 {
                -value
            } else {
                0.0
            }
        }
        ConstraintKind::Equality => value.abs(),
    }
}

/// Largest violation across all constraint values.
pub fn max_violation(kind: ConstraintKind, values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |worst, &value| worst.max(violation(kind, value)))
}

/// Reports whether constraint values satisfy the constraint.
///
/// Inequality values accept a small negative slack. Equality values accept an
/// absolute distance from zero up to the tolerance.
pub fn is_feasible(kind: ConstraintKind, values: &[f64], tolerance: f64) -> bool {
    match kind {
        ConstraintKind::Inequality => values.iter().all(|&value| value >= -tolerance),
        ConstraintKind::Equality => values.iter().all(|&value| value.abs() <= tolerance),
    }
}

/// Evaluates constraint values at a point.
pub trait ConstraintEvaluator {
    /// Returns one value per constraint.
    fn values(&self, x: &[f64]) -> Result<Vec<f64>, String>;
}
