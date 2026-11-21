//! Full-memory BFGS for smooth unbounded objectives.
//!
//! The method keeps a dense inverse-Hessian approximation, so its memory cost
//! is quadratic in the parameter count. It solves the same unconstrained
//! objective as the limited-memory path and rejects finite bounds instead of
//! ignoring them.

use super::budget::Budget;
use super::line_search::{strong_wolfe, LineSearchOptions};
use super::objective::{CountedObjective, Objective, ObjectiveError, SmoothObjective};
use super::request::BfgsOptions;
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

/// Runs BFGS on an unbounded smooth objective.
///
/// The starting point defaults to the origin. The run stops when the gradient
/// falls under the tolerance, when the iteration limit is reached, or when a
/// budget ends the run.
pub fn bfgs(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    budget: &Budget,
    options: &BfgsOptions,
    start: Option<&[f64]>,
) -> OptimizationResult {
    if objective.bounds().has_finite_limit() {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message("bfgs needs an unbounded domain");
    }
    let dimension = objective.dim();
    if dimension == 0 {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("bfgs needs at least one parameter");
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

    // Dense inverse-Hessian approximation, starting from the identity.
    let mut inverse: Vec<Vec<f64>> = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| if row == column { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

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

        // Search direction: -H g.
        let direction: Vec<f64> = (0..dimension)
            .map(|row| {
                -inverse[row]
                    .iter()
                    .zip(gradient.iter())
                    .map(|(h, g)| h * g)
                    .sum::<f64>()
            })
            .collect();

        let line = match strong_wolfe(&counted, smooth, &point, &direction, &line_search) {
            Ok(line) => line,
            Err(err) => {
                status = failure_status(&err);
                break;
            }
        };

        // Broyden update on the inverse Hessian.
        let step: Vec<f64> = (0..dimension)
            .map(|axis| line.step * direction[axis])
            .collect();
        let slope: Vec<f64> = (0..dimension)
            .map(|axis| line.gradient[axis] - gradient[axis])
            .collect();
        let curvature: f64 = step.iter().zip(slope.iter()).map(|(s, y)| s * y).sum();
        if curvature > 0.0 {
            let rho = 1.0 / curvature;
            // Left factor: (I - rho s y^T) H.
            let mut left = vec![vec![0.0; dimension]; dimension];
            for row in 0..dimension {
                for column in 0..dimension {
                    let mut value = inverse[row][column];
                    for k in 0..dimension {
                        value -= rho * step[row] * slope[k] * inverse[k][column];
                    }
                    left[row][column] = value;
                }
            }
            // Right factor: left (I - rho y s^T).
            let mut updated = vec![vec![0.0; dimension]; dimension];
            for row in 0..dimension {
                for column in 0..dimension {
                    let mut value = left[row][column];
                    for k in 0..dimension {
                        value -= rho * left[row][k] * slope[k] * step[column];
                    }
                    updated[row][column] = value;
                }
            }
            // Rank-one term: rho s s^T.
            for row in 0..dimension {
                for column in 0..dimension {
                    updated[row][column] += rho * step[row] * step[column];
                }
            }
            inverse = updated;
        }

        point = (0..dimension)
            .map(|axis| point[axis] + step[axis])
            .collect();
        value = line.value;
        gradient = line.gradient;

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
