//! Bounded Nelder-Mead simplex search.
//!
//! The method needs objective values only. It replaces the worst vertex of the
//! simplex with a reflection, an expansion, a contraction, or a shrink. Every
//! vertex stays inside the box: a proposed vertex outside the box is clipped
//! onto the nearest bound.

use super::bounds::Bounds;
use super::budget::Budget;
use super::objective::{CountedObjective, Objective, ObjectiveError};
use super::request::NelderMeadOptions;
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Reflection coefficient.
const ALPHA: f64 = 1.0;
/// Expansion coefficient.
const GAMMA: f64 = 2.0;
/// Contraction coefficient.
const BETA: f64 = 0.5;
/// Shrink coefficient.
const SIGMA: f64 = 0.5;

/// Builds the starting simplex around one point.
fn initial_simplex(start: &[f64], bounds: &Bounds, options: &NelderMeadOptions) -> Vec<Vec<f64>> {
    if let Some(simplex) = &options.initial_simplex {
        return simplex
            .iter()
            .map(|vertex| bounds.project(vertex))
            .collect();
    }
    let mut simplex = vec![start.to_vec()];
    for dimension in 0..start.len() {
        let mut vertex = start.to_vec();
        let scale = start[dimension].abs();
        let step = if scale > 0.0 { 0.05 * scale } else { 0.00025 };
        vertex[dimension] += step;
        simplex.push(bounds.project(&vertex));
    }
    simplex
}

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Runs bounded Nelder-Mead.
///
/// The starting point defaults to the middle of the box. The method stops when
/// the objective spread and the simplex spread fall under the configured
/// tolerances, when the iteration limit is reached, or when a budget ends the
/// run.
pub fn nelder_mead(
    objective: &dyn Objective,
    budget: &Budget,
    options: &NelderMeadOptions,
    start: Option<&[f64]>,
) -> OptimizationResult {
    let bounds = objective.bounds();
    let start = match start {
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
                .with_message("nelder-mead needs a starting point for unbounded domains");
        }
    };
    let dimension = start.len();
    if dimension == 0 {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("nelder-mead needs at least one parameter");
    }

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let mut simplex = initial_simplex(&start, bounds, options);
    let mut values: Vec<f64> = Vec::with_capacity(simplex.len());
    for vertex in &simplex {
        match counted.value(vertex) {
            Ok(value) => values.push(value),
            Err(err) => {
                return OptimizationResult::new(Vec::new(), f64::NAN)
                    .with_status(failure_status(&err))
                    .with_message(err.to_string());
            }
        }
    }

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut status = SolveStatus::Converged;

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

        // Sort the vertices by objective value.
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&left, &right| {
            values[left]
                .partial_cmp(&values[right])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&index| simplex[index].clone()).collect();
        let sorted: Vec<f64> = order.iter().map(|&index| values[index]).collect();
        values = sorted;

        let best = 0usize;
        let worst = dimension;
        let second_worst = dimension - 1;

        let value_spread = values[worst] - values[best];
        let point_spread = (0..dimension)
            .map(|axis| {
                let mut low = f64::INFINITY;
                let mut high = f64::NEG_INFINITY;
                for vertex in &simplex {
                    low = low.min(vertex[axis]);
                    high = high.max(vertex[axis]);
                }
                high - low
            })
            .fold(0.0_f64, f64::max);
        if value_spread <= options.fatol && point_spread <= options.xatol {
            break;
        }

        // Centroid of the best vertices.
        let mut centroid = vec![0.0; dimension];
        for vertex in simplex.iter().take(dimension) {
            for (accumulator, component) in centroid.iter_mut().zip(vertex.iter()) {
                *accumulator += component;
            }
        }
        for value in centroid.iter_mut() {
            *value /= dimension as f64;
        }

        // Reflection.
        let reflected: Vec<f64> = (0..dimension)
            .map(|axis| {
                let worst_value = simplex[worst][axis];
                centroid[axis] + ALPHA * (centroid[axis] - worst_value)
            })
            .collect();
        let reflected = bounds.project(&reflected);
        let reflected_value = match counted.value(&reflected) {
            Ok(value) => value,
            Err(err) => {
                status = failure_status(&err);
                break 'outer;
            }
        };

        if reflected_value < values[best] {
            // Expansion.
            let expanded: Vec<f64> = (0..dimension)
                .map(|axis| centroid[axis] + GAMMA * (reflected[axis] - centroid[axis]))
                .collect();
            let expanded = bounds.project(&expanded);
            let expanded_value = match counted.value(&expanded) {
                Ok(value) => value,
                Err(err) => {
                    status = failure_status(&err);
                    break 'outer;
                }
            };
            if expanded_value < reflected_value {
                simplex[worst] = expanded;
                values[worst] = expanded_value;
            } else {
                simplex[worst] = reflected;
                values[worst] = reflected_value;
            }
        } else if reflected_value < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
        } else {
            // Contraction.
            let contracted: Vec<f64> = if reflected_value < values[worst] {
                (0..dimension)
                    .map(|axis| centroid[axis] + BETA * (reflected[axis] - centroid[axis]))
                    .collect()
            } else {
                (0..dimension)
                    .map(|axis| centroid[axis] + BETA * (simplex[worst][axis] - centroid[axis]))
                    .collect()
            };
            let contracted = bounds.project(&contracted);
            let contracted_value = match counted.value(&contracted) {
                Ok(value) => value,
                Err(err) => {
                    status = failure_status(&err);
                    break 'outer;
                }
            };
            if contracted_value < values[worst] {
                simplex[worst] = contracted;
                values[worst] = contracted_value;
            } else {
                // Shrink toward the best vertex.
                for index in 1..simplex.len() {
                    let shrunk: Vec<f64> = (0..dimension)
                        .map(|axis| {
                            simplex[best][axis]
                                + SIGMA * (simplex[index][axis] - simplex[best][axis])
                        })
                        .collect();
                    let shrunk = bounds.project(&shrunk);
                    let shrunk_value = match counted.value(&shrunk) {
                        Ok(value) => value,
                        Err(err) => {
                            status = failure_status(&err);
                            break 'outer;
                        }
                    };
                    simplex[index] = shrunk;
                    values[index] = shrunk_value;
                }
            }
        }

        history.push(HistoryEntry {
            iteration: iterations,
            value: values[worst],
            best_value: values[best],
        });
    }

    let mut best = 0usize;
    for index in 1..simplex.len() {
        if values[index] < values[best] {
            best = index;
        }
    }
    let mut result = OptimizationResult::new(simplex[best].clone(), values[best])
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    if !history.is_empty() {
        result.history = Some(history);
    }
    result
}
