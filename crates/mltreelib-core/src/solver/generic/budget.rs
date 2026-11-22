//! Shared evaluation and iteration budget for the generic optimization seam.

use std::cell::Cell;

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
/// solves, perturbation steps, population initialization, finite-difference
/// gradients, and archive updates. The counters use interior mutability, so
/// the objective wrapper and the solver share one budget without aliasing
/// borrows.
#[derive(Clone, Debug)]
pub struct Budget {
    max_evaluations: Option<usize>,
    max_iterations: Option<usize>,
    evaluations: Cell<usize>,
    iterations: Cell<usize>,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl Budget {
    /// Builds a budget with optional limits. `None` means unlimited.
    pub fn new(max_evaluations: Option<usize>, max_iterations: Option<usize>) -> Self {
        Self {
            max_evaluations,
            max_iterations,
            evaluations: Cell::new(0),
            iterations: Cell::new(0),
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
    pub fn charge_evaluation(&self) -> Result<(), BudgetStatus> {
        let used = self.evaluations.get();
        if let Some(max) = self.max_evaluations {
            if used >= max {
                return Err(BudgetStatus::EvaluationLimit);
            }
        }
        self.evaluations.set(used + 1);
        Ok(())
    }

    /// Charges one solver iteration.
    ///
    /// Returns [`BudgetStatus::IterationLimit`] without incrementing the
    /// counter when the iteration budget is exhausted.
    pub fn charge_iteration(&self) -> Result<(), BudgetStatus> {
        let used = self.iterations.get();
        if let Some(max) = self.max_iterations {
            if used >= max {
                return Err(BudgetStatus::IterationLimit);
            }
        }
        self.iterations.set(used + 1);
        Ok(())
    }

    /// Objective evaluations performed so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations.get()
    }

    /// Solver iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.iterations.get()
    }

    /// Remaining objective evaluations. `None` means unlimited.
    pub fn remaining_evaluations(&self) -> Option<usize> {
        self.max_evaluations
            .map(|max| max.saturating_sub(self.evaluations.get()))
    }

    /// Remaining solver iterations. `None` means unlimited.
    pub fn remaining_iterations(&self) -> Option<usize> {
        self.max_iterations
            .map(|max| max.saturating_sub(self.iterations.get()))
    }
}
