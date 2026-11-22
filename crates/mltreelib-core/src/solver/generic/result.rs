//! Result and status contracts for the generic optimization seam.

/// Termination status of one optimization run.
///
/// Budget exhaustion never reports success. Only [`SolveStatus::Converged`] and
/// [`SolveStatus::Tolerance`] report success.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// The method reached its convergence test.
    Converged,
    /// The iteration budget ended the run.
    IterationLimit,
    /// The evaluation budget ended the run.
    EvaluationLimit,
    /// The tolerance test ended the run.
    Tolerance,
    /// The objective returned a NaN or infinite value.
    InvalidObjective,
    /// The numerical kernel failed.
    NumericalFailure,
    /// No feasible point was found.
    Infeasible,
    /// The method cannot solve the requested problem.
    Unsupported,
    /// An unrecoverable error ended the run.
    HardError,
}

impl SolveStatus {
    /// Canonical name used by the Python surface and result files.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::IterationLimit => "iteration_limit",
            Self::EvaluationLimit => "evaluation_limit",
            Self::Tolerance => "tolerance",
            Self::InvalidObjective => "invalid_objective",
            Self::NumericalFailure => "numerical_failure",
            Self::Infeasible => "infeasible",
            Self::Unsupported => "unsupported",
            Self::HardError => "hard_error",
        }
    }

    /// Reports whether the run reached an accepted optimum.
    pub fn is_success(self) -> bool {
        matches!(self, Self::Converged | Self::Tolerance)
    }
}

/// One convergence history sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryEntry {
    /// Iteration index of the sample.
    pub iteration: usize,
    /// Objective value at the sample.
    pub value: f64,
    /// Best objective value seen up to the sample.
    pub best_value: f64,
}

/// Result of one optimization run.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Best point found. The point is finite and inside the bounds.
    pub best_x: Vec<f64>,
    /// Objective value at `best_x`.
    pub best_value: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// User objective evaluations performed.
    pub evaluations: usize,
    /// Termination status.
    pub status: SolveStatus,
    /// Optional convergence history.
    pub history: Option<Vec<HistoryEntry>>,
    /// Largest constraint violation at `best_x`.
    pub max_constraint_violation: Option<f64>,
    /// Number of local solves performed by a composition method.
    pub local_solve_count: Option<usize>,
    /// Optional failure detail.
    pub message: Option<String>,
}

impl OptimizationResult {
    /// Builds a result with the given point and value.
    pub fn new(best_x: Vec<f64>, best_value: f64) -> Self {
        Self {
            best_x,
            best_value,
            iterations: 0,
            evaluations: 0,
            status: SolveStatus::Converged,
            history: None,
            max_constraint_violation: None,
            local_solve_count: None,
            message: None,
        }
    }

    /// Sets the termination status.
    pub fn with_status(mut self, status: SolveStatus) -> Self {
        self.status = status;
        self
    }

    /// Sets the iteration and evaluation counts.
    pub fn with_counts(mut self, iterations: usize, evaluations: usize) -> Self {
        self.iterations = iterations;
        self.evaluations = evaluations;
        self
    }

    /// Sets the failure detail.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}
