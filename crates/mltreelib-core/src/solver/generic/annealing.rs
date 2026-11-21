//! Bounded simulated annealing.
//!
//! The method is a global trajectory search. It proposes a bounded random
//! displacement per dimension, accepts the proposal with a Metropolis test, and
//! cools the temperature geometrically. The best incumbent is tracked
//! separately from the current point, so the returned result never degrades.
//!
//! The method is heuristic. It carries no optimality claim.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use super::bounds::Bounds;
use super::budget::Budget;
use super::objective::{CountedObjective, Objective, ObjectiveError};
use super::request::{AnnealingOptions, BoundaryPolicy};
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Number of samples used to derive the initial temperature.
const TEMPERATURE_SAMPLES: usize = 16;

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Reflects one point back into the box.
fn reflect(bounds: &Bounds, point: &[f64]) -> Vec<f64> {
    (0..point.len())
        .map(|axis| {
            let mut value = point[axis];
            let low = bounds.lower()[axis];
            let high = bounds.upper()[axis];
            while value < low || value > high {
                if value < low {
                    value = low + (low - value);
                } else {
                    value = high - (value - high);
                }
                if !low.is_finite() || !high.is_finite() {
                    return value.clamp(low, high);
                }
            }
            value
        })
        .collect()
}

/// Draws one proposal inside the box.
///
/// The proposal is a per-dimension displacement scaled by the bound width. The
/// boundary policy decides what happens when the proposal leaves the box.
fn propose(
    rng: &mut ChaCha8Rng,
    normal: &Normal<f64>,
    bounds: &Bounds,
    current: &[f64],
    options: &AnnealingOptions,
) -> Vec<f64> {
    for _ in 0..64 {
        let candidate: Vec<f64> = (0..current.len())
            .map(|axis| {
                let width = bounds.width(axis);
                let scale = if width.is_finite() { width } else { 1.0 };
                current[axis] + options.proposal_scale * scale * normal.sample(rng)
            })
            .collect();
        match options.boundary {
            BoundaryPolicy::Resample => {
                if bounds.contains(&candidate) {
                    return candidate;
                }
            }
            BoundaryPolicy::Reflect => {
                if bounds.contains(&candidate) {
                    return candidate;
                }
                return reflect(bounds, &candidate);
            }
        }
    }
    bounds.project(current)
}

/// Runs bounded simulated annealing.
///
/// The starting point defaults to the middle of the box. The initial
/// temperature comes from the observed objective spread when the caller does
/// not supply one.
pub fn simulated_annealing(
    objective: &dyn Objective,
    budget: &Budget,
    options: &AnnealingOptions,
    seed: u64,
    start: Option<&[f64]>,
) -> OptimizationResult {
    let bounds = objective.bounds();
    if !bounds.is_bounded() {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message("simulated annealing needs finite bounds");
    }
    let mut current = match start {
        Some(start) => bounds.project(start),
        None => bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect(),
    };

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("standard normal");

    let mut current_value = match counted.value(&current) {
        Ok(value) => value,
        Err(err) => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(failure_status(&err))
                .with_message(err.to_string());
        }
    };
    let mut best_point = current.clone();
    let mut best_value = current_value;

    // Derive the initial temperature from the observed objective spread.
    let mut temperature = match options.initial_temperature {
        Some(temperature) => temperature,
        None => {
            let mut samples = Vec::with_capacity(TEMPERATURE_SAMPLES);
            for _ in 0..TEMPERATURE_SAMPLES {
                let candidate: Vec<f64> = (0..current.len())
                    .map(|axis| rng.gen_range(bounds.lower()[axis]..bounds.upper()[axis]))
                    .collect();
                match counted.value(&candidate) {
                    Ok(value) => samples.push(value),
                    Err(err) => {
                        return OptimizationResult::new(best_point.clone(), best_value)
                            .with_status(failure_status(&err))
                            .with_message(err.to_string());
                    }
                }
            }
            let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (high - low).max(f64::MIN_POSITIVE)
        }
    };

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut stagnation = 0usize;
    let mut status = SolveStatus::Converged;

    while temperature > options.min_temperature {
        if budget.charge_iteration().is_err() {
            status = SolveStatus::IterationLimit;
            break;
        }
        iterations += 1;

        let candidate = propose(&mut rng, &normal, bounds, &current, options);
        let candidate_value = match counted.value(&candidate) {
            Ok(value) => value,
            Err(err) => {
                status = failure_status(&err);
                break;
            }
        };

        let delta = candidate_value - current_value;
        let accept = delta <= 0.0 || rng.gen::<f64>() < (-delta / temperature).exp();
        if accept {
            current = candidate;
            current_value = candidate_value;
            if current_value < best_value {
                best_value = current_value;
                best_point = current.clone();
                stagnation = 0;
            } else {
                stagnation += 1;
            }
        } else {
            stagnation += 1;
        }

        history.push(HistoryEntry {
            iteration: iterations,
            value: current_value,
            best_value,
        });

        if stagnation >= options.max_stagnation {
            status = SolveStatus::Converged;
            break;
        }
        temperature *= options.alpha;
    }

    let mut result = OptimizationResult::new(best_point, best_value)
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    if !history.is_empty() {
        result.history = Some(history);
    }
    result
}
