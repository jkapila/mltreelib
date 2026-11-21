//! Bounded Powell direction-set search.
//!
//! The method needs objective values only. It runs one line search per
//! direction, then replaces the direction set with the conjugate directions of
//! the completed cycle.
//!
//! The line search is a golden-section search on the step length. Every
//! evaluated point stays inside the box, matching the SciPy behaviour for an
//! in-bounds start. The classic extrapolation step is not implemented.

use super::bounds::Bounds;
use super::budget::Budget;
use super::objective::{CountedObjective, Objective, ObjectiveError};
use super::request::PowellOptions;
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Golden-section ratio.
const GOLDEN: f64 = 0.618_033_988_749_894_9;

/// Largest step length the expanding line search tries.
const MAX_STEP: f64 = 1e6;

/// Cycles without a better incumbent before the run stops.
const STAGNATION_LIMIT: usize = 20;

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Feasible step interval for one direction.
///
/// The interval holds every step length that keeps the point inside the box.
/// A step may be negative, so the search can move against the direction.
fn feasible_interval(bounds: &Bounds, point: &[f64], direction: &[f64]) -> (f64, f64) {
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;
    for (axis, &step) in direction.iter().enumerate() {
        if step == 0.0 {
            continue;
        }
        let lower = bounds.lower()[axis];
        let upper = bounds.upper()[axis];
        let mut axis_low = f64::NEG_INFINITY;
        let mut axis_high = f64::INFINITY;
        if lower.is_finite() {
            axis_low = (lower - point[axis]) / step;
        }
        if upper.is_finite() {
            axis_high = (upper - point[axis]) / step;
        }
        if axis_low > axis_high {
            std::mem::swap(&mut axis_low, &mut axis_high);
        }
        low = low.max(axis_low);
        high = high.min(axis_high);
    }
    (low, high)
}

/// Minimizes the objective along one direction.
///
/// The feasible interval holds every step that keeps the point inside the box.
/// A step may be negative, so the search can move against the direction. When
/// one side of the interval is unbounded, the search expands that side until
/// the objective stops improving, then searches the whole interval.
fn line_search(
    objective: &dyn Objective,
    bounds: &Bounds,
    point: &[f64],
    direction: &[f64],
    options: &PowellOptions,
) -> Result<(f64, f64), (SolveStatus, Option<String>)> {
    let (low, high) = feasible_interval(bounds, point, direction);

    let mut low = low;
    let mut high = high;
    if !low.is_finite() {
        let base = evaluate(objective, bounds, point, direction, 0.0)?;
        let mut step = 1.0;
        low = 0.0;
        while step < MAX_STEP {
            let value = evaluate(objective, bounds, point, direction, -step)?;
            if value > base {
                low = -step;
                break;
            }
            low = -step;
            step *= 2.0;
        }
    }
    if !high.is_finite() {
        let base = evaluate(objective, bounds, point, direction, 0.0)?;
        let mut step = 1.0;
        high = 0.0;
        while step < MAX_STEP {
            let value = evaluate(objective, bounds, point, direction, step)?;
            if value > base {
                high = step;
                break;
            }
            high = step;
            step *= 2.0;
        }
    }
    if high <= low {
        return Ok((0.0, evaluate(objective, bounds, point, direction, 0.0)?));
    }

    // Golden-section search on the feasible interval.
    let mut left = low;
    let mut right = high;
    let mut point_left = left + (1.0 - GOLDEN) * (right - left);
    let mut point_right = left + GOLDEN * (right - left);
    let mut value_left = evaluate(objective, bounds, point, direction, point_left)?;
    let mut value_right = evaluate(objective, bounds, point, direction, point_right)?;

    for _ in 0..80 {
        if (right - left).abs() <= options.xtol {
            break;
        }
        if (value_left - value_right).abs() <= options.ftol * (1.0 + value_left.abs()) {
            break;
        }
        if value_left < value_right {
            right = point_right;
            point_right = point_left;
            value_right = value_left;
            point_left = right - GOLDEN * (right - left);
            value_left = evaluate(objective, bounds, point, direction, point_left)?;
        } else {
            left = point_left;
            point_left = point_right;
            value_left = value_right;
            point_right = left + GOLDEN * (right - left);
            value_right = evaluate(objective, bounds, point, direction, point_right)?;
        }
    }

    let (step, value) = if value_left <= value_right {
        (point_left, value_left)
    } else {
        (point_right, value_right)
    };
    Ok((step, value))
}

/// Evaluates one point along one direction.
fn evaluate(
    objective: &dyn Objective,
    bounds: &Bounds,
    point: &[f64],
    direction: &[f64],
    step: f64,
) -> Result<f64, (SolveStatus, Option<String>)> {
    let candidate: Vec<f64> = (0..point.len())
        .map(|axis| point[axis] + step * direction[axis])
        .collect();
    let candidate = bounds.project(&candidate);
    match objective.value(&candidate) {
        Ok(value) => Ok(value),
        Err(err) => Err((failure_status(&err), Some(err.to_string()))),
    }
}

/// Runs bounded Powell.
///
/// The starting point defaults to the middle of the box. The method stops when
/// the objective and the point stop moving, when the iteration limit is
/// reached, or when a budget ends the run.
pub fn powell(
    objective: &dyn Objective,
    budget: &Budget,
    options: &PowellOptions,
    start: Option<&[f64]>,
) -> OptimizationResult {
    let bounds = objective.bounds();
    let mut point = match start {
        Some(start) => bounds.project(start),
        None if bounds.is_bounded() => bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect(),
        None => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(SolveStatus::Unsupported)
                .with_message("powell needs a starting point for unbounded domains");
        }
    };
    let dimension = point.len();
    if dimension == 0 {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("powell needs at least one parameter");
    }

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let mut value = match counted.value(&point) {
        Ok(value) => value,
        Err(err) => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(failure_status(&err))
                .with_message(err.to_string());
        }
    };

    let mut directions: Vec<Vec<f64>> = (0..dimension)
        .map(|axis| {
            let mut direction = vec![0.0; dimension];
            direction[axis] = 1.0;
            direction
        })
        .collect();
    if let Some(max_directions) = options.max_directions {
        directions.truncate(max_directions.max(1));
    }

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut stagnant = 0usize;
    let mut status = SolveStatus::Converged;
    let mut best_point = point.clone();
    let mut best_value = value;

    'outer: loop {
        if iterations >= options.max_iter {
            status = SolveStatus::IterationLimit;
            break;
        }
        if budget.charge_iteration().is_err() {
            status = SolveStatus::IterationLimit;
            break;
        }
        iterations += 1;

        let cycle_start = point.clone();
        let cycle_value = value;

        for direction in directions.iter() {
            match line_search(&counted, bounds, &point, direction, options) {
                Ok((step, line_value)) => {
                    // Keep the current point when the line search found nothing
                    // better.
                    if line_value < value {
                        let candidate: Vec<f64> = (0..dimension)
                            .map(|axis| point[axis] + step * direction[axis])
                            .collect();
                        point = bounds.project(&candidate);
                        value = line_value;
                        if value < best_value {
                            best_value = value;
                            best_point = point.clone();
                        }
                    }
                }
                Err((failure, message)) => {
                    status = failure;
                    if let Some(message) = message {
                        let mut result = OptimizationResult::new(best_point.clone(), best_value)
                            .with_status(status)
                            .with_counts(iterations, budget.evaluations())
                            .with_message(message);
                        result.history = Some(history);
                        return result;
                    }
                    break 'outer;
                }
            }
        }

        // Powell direction update.
        if directions.len() > 1 {
            for index in 0..directions.len() - 1 {
                let next_norm: f64 = directions[index + 1]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                let norm: f64 = directions[index]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                let next = directions[index + 1].clone();
                let current = directions[index].clone();
                directions[index] = (0..dimension)
                    .map(|axis| next[axis] * next_norm * next_norm - current[axis] * norm * norm)
                    .collect();
            }
            let last: Vec<f64> = (0..dimension)
                .map(|axis| point[axis] - cycle_start[axis])
                .collect();
            if let Some(last_direction) = directions.last_mut() {
                *last_direction = last;
            }
        }

        let movement = (0..dimension)
            .map(|axis| (point[axis] - cycle_start[axis]).abs())
            .fold(0.0_f64, f64::max);
        let improvement = (cycle_value - value).abs();
        history.push(HistoryEntry {
            iteration: iterations,
            value,
            best_value,
        });
        if movement <= options.xtol || improvement <= options.ftol * (1.0 + value.abs()) {
            break;
        }
        // Stop when the incumbent stops improving.
        if improvement <= options.ftol * (1.0 + best_value.abs()) {
            stagnant += 1;
            if stagnant >= STAGNATION_LIMIT {
                break;
            }
        } else {
            stagnant = 0;
        }
    }

    let mut result = OptimizationResult::new(best_point, best_value)
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    if !history.is_empty() {
        result.history = Some(history);
    }
    result
}
