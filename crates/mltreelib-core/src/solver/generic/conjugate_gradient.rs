//! Nonlinear conjugate gradient for smooth unbounded objectives.
//!
//! The method uses the Polak-Ribiere direction with a restart. It needs only
//! the objective and the gradient, so its memory cost is linear in the
//! parameter count. The restart rule is documented: the search direction resets
//! when the Polak-Ribiere coefficient turns negative, when the configured
//! restart period ends, or when the direction loses its length.

use super::budget::Budget;
use super::line_search::{strong_wolfe, LineSearchOptions};
use super::objective::{CountedObjective, Objective, ObjectiveError, SmoothObjective};
use super::request::NonlinearCgOptions;
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Largest absolute gradient component.
fn gradient_norm(gradient: &[f64]) -> f64 {
    gradient
        .iter()
        .fold(0.0_f64, |worst, value| worst.max(value.abs()))
}

/// Euclidean norm of one vector.
fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Runs nonlinear conjugate gradient on an unbounded smooth objective.
///
/// The starting point defaults to the origin. The run stops when the gradient
/// falls under the tolerance, when the iteration limit is reached, or when a
/// budget ends the run.
pub fn nonlinear_conjugate_gradient(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    budget: &Budget,
    options: &NonlinearCgOptions,
    start: Option<&[f64]>,
) -> OptimizationResult {
    if objective.bounds().has_finite_limit() {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message("nonlinear conjugate gradient needs an unbounded domain");
    }
    let dimension = objective.dim();
    if dimension == 0 {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("nonlinear conjugate gradient needs at least one parameter");
    }
    let mut point = match start {
        Some(start) => start.to_vec(),
        None => vec![0.0; dimension],
    };

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let line_search = LineSearchOptions::default();

    let mut gradient = match smooth.gradient(&point) {
        Ok(gradient) => gradient,
        Err(err) => {
            return OptimizationResult::new(point, f64::NAN)
                .with_status(failure_status(&err))
                .with_message(err.to_string());
        }
    };
    let mut value = match counted.value(&point) {
        Ok(value) => value,
        Err(err) => {
            return OptimizationResult::new(point, f64::NAN)
                .with_status(failure_status(&err))
                .with_message(err.to_string());
        }
    };
    let mut direction: Vec<f64> = gradient.iter().map(|g| -g).collect();

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut status = SolveStatus::Converged;

    while gradient_norm(&gradient) > options.gtol {
        if iterations >= options.max_iter {
            status = SolveStatus::IterationLimit;
            break;
        }
        if budget.charge_iteration().is_err() {
            status = SolveStatus::IterationLimit;
            break;
        }
        iterations += 1;

        let line = match strong_wolfe(&counted, smooth, &point, &direction, &line_search) {
            Ok(line) => line,
            Err(err) => {
                status = failure_status(&err);
                break;
            }
        };

        let previous_gradient = gradient.clone();
        point = (0..dimension)
            .map(|axis| point[axis] + line.step * direction[axis])
            .collect();
        value = line.value;
        gradient = line.gradient;

        // Polak-Ribiere coefficient with a non-negative clamp.
        let denominator: f64 = previous_gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let numerator: f64 = gradient
            .iter()
            .zip(previous_gradient.iter())
            .map(|(new, old)| new * (new - old))
            .sum();
        let coefficient = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Restart rule.
        let period_restart = options
            .restart_period
            .is_some_and(|period| iterations % period == 0);
        let direction_lost = norm(&direction) <= 1e-12;
        if coefficient <= 0.0 || period_restart || direction_lost {
            direction = gradient.iter().map(|g| -g).collect();
        } else {
            direction = (0..dimension)
                .map(|axis| -gradient[axis] + coefficient * direction[axis])
                .collect();
        }

        history.push(HistoryEntry {
            iteration: iterations,
            value,
            best_value: value,
        });
    }

    let mut result = OptimizationResult::new(point, value)
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    if !history.is_empty() {
        result.history = Some(history);
    }
    result
}
