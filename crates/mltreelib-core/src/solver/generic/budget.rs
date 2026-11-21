//! Shared evaluation and iteration budget for the generic optimization seam.

/// Which budget ended a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetStatus {
    /// The evaluation budget ended the run.
    EvaluationLimit,
    /// The iteration budget ended the run.
    IterationLimit,
}

/// Counts user objective evaluations and solver iterations.
///
/// One counter covers every user objective call, including calls inside local
/// solves, perturbation steps, population initialization, and archive updates.
#[derive(Clone, Debug, Default)]
pub struct Budget {
    max_evaluations: Option<usize>,
    max_iterations: Option<usize>,
    evaluations: usize,
    iterations: usize,
}

impl Budget {
    /// Builds a budget with optional limits. `None` means unlimited.
    pub fn new(max_evaluations: Option<usize>, max_iterations: Option<usize>) -> Self {
        Self {
            max_evaluations,
            max_iterations,
            evaluations: 0,
            iterations: 0,
        }
    }

    /// Builds a budget without limits.
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Charges one objective evaluation.
    ///
    /// Returns [`BudgetStatus::EvaluationLimit`] without incrementing the
    /// counter when the evaluation budget is exhausted.
    pub fn charge_evaluation(&mut self) -> Result<(), BudgetStatus> {
        if let Some(max) = self.max_evaluations {
            if self.evaluations >= max {
                return Err(BudgetStatus::EvaluationLimit);
            }
        }
        self.evaluations += 1;
        Ok(())
    }

    /// Charges one solver iteration.
    ///
    /// Returns [`BudgetStatus::IterationLimit`] without incrementing the
    /// counter when the iteration budget is exhausted.
    pub fn charge_iteration(&mut self) -> Result<(), BudgetStatus> {
        if let Some(max) = self.max_iterations {
            if self.iterations >= max {
                return Err(BudgetStatus::IterationLimit);
            }
        }
        self.iterations += 1;
        Ok(())
    }

    /// Objective evaluations performed so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Solver iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Remaining objective evaluations. `None` means unlimited.
    pub fn remaining_evaluations(&self) -> Option<usize> {
        self.max_evaluations
            .map(|max| max.saturating_sub(self.evaluations))
    }

    /// Remaining solver iterations. `None` means unlimited.
    pub fn remaining_iterations(&self) -> Option<usize> {
        self.max_iterations
            .map(|max| max.saturating_sub(self.iterations))
    }
}
