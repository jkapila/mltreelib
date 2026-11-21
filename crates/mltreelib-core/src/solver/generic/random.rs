//! Random search and Latin hypercube sampling.
//!
//! Both methods sample candidates inside the bounds, evaluate them through the
//! shared budget, and report the best candidate with the shared result contract.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::bounds::Bounds;
use super::budget::Budget;
use super::objective::{CountedObjective, Objective, ObjectiveError};
use super::request::{LatinHypercubeOptions, RandomSearchOptions};
use super::result::{OptimizationResult, SolveStatus};

/// Outcome of one candidate evaluation round.
struct Evaluation {
    best_x: Vec<f64>,
    best_value: f64,
    status: SolveStatus,
    message: Option<String>,
}

/// Evaluates candidates in order and keeps the best one.
fn evaluate_candidates(
    objective: &dyn Objective,
    candidates: &[Vec<f64>],
) -> Evaluation {
    let mut best_x = Vec::new();
    let mut best_value = f64::INFINITY;

    for candidate in candidates {
        match objective.value(candidate) {
            Ok(value) if value.is_finite() => {
                if value < best_value {
                    best_value = value;
                    best_x = candidate.clone();
                }
            }
            Ok(value) => {
                return Evaluation {
                    best_x,
                    best_value,
                    status: SolveStatus::InvalidObjective,
                    message: Some(format!("objective value is not finite: {value}")),
                };
            }
            Err(ObjectiveError::BudgetExhausted(_)) => {
                return Evaluation {
                    best_x,
                    best_value,
                    status: SolveStatus::EvaluationLimit,
                    message: None,
                };
            }
            Err(err) => {
                return Evaluation {
                    best_x,
                    best_value,
                    status: SolveStatus::HardError,
                    message: Some(err.to_string()),
                };
            }
        }
    }

    Evaluation {
        best_x,
        best_value,
        status: SolveStatus::Converged,
        message: None,
    }
}

/// Checks that every limit is finite, which uniform sampling requires.
fn require_finite_bounds(bounds: &Bounds) -> Result<(), String> {
    if bounds.is_bounded() {
        Ok(())
    } else {
        Err("random sampling needs finite bounds".to_string())
    }
}

/// Draws one uniform point inside finite bounds.
fn uniform_point(rng: &mut ChaCha8Rng, bounds: &Bounds) -> Vec<f64> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&lo, &hi)| {
            if hi > lo {
                rng.gen_range(lo..hi)
            } else {
                lo
            }
        })
        .collect()
}

/// Builds Latin hypercube samples inside finite bounds.
///
/// Every dimension is stratified independently into `samples` strata. One point
/// takes one stratum per dimension. The stratum order is shuffled per dimension
/// with the seeded generator.
pub fn latin_hypercube_samples(
    bounds: &Bounds,
    samples: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    require_finite_bounds(bounds)?;
    if samples == 0 {
        return Ok(Vec::new());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut strata: Vec<Vec<f64>> = Vec::with_capacity(bounds.dim());

    for dim in 0..bounds.dim() {
        let lo = bounds.lower()[dim];
        let hi = bounds.upper()[dim];
        let mut order: Vec<usize> = (0..samples).collect();
        order.shuffle(&mut rng);
        let width = (hi - lo) / samples as f64;
        let column: Vec<f64> = order
            .iter()
            .map(|&index| lo + (index as f64 + rng.gen::<f64>()) * width)
            .collect();
        strata.push(column);
    }

    Ok((0..samples)
        .map(|row| strata.iter().map(|column| column[row]).collect())
        .collect())
}

/// Runs uniform random search inside bounds.
///
/// Candidates come from `initial_population` first when one is supplied. The
/// remaining candidates are uniform draws.
pub fn random_search(
    objective: &dyn Objective,
    budget: &Budget,
    options: &RandomSearchOptions,
    seed: u64,
    initial_population: Option<&[Vec<f64>]>,
) -> OptimizationResult {
    let bounds = objective.bounds();
    if let Err(message) = require_finite_bounds(bounds) {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message(message);
    }

    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(options.samples);
    if let Some(population) = initial_population {
        for point in population {
            if !bounds.contains(point) {
                return OptimizationResult::new(Vec::new(), f64::NAN)
                    .with_status(SolveStatus::HardError)
                    .with_message("initial population point lies outside the bounds");
            }
            candidates.push(point.clone());
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    while candidates.len() < options.samples {
        candidates.push(uniform_point(&mut rng, bounds));
    }

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let evaluation = evaluate_candidates(&counted, &candidates);
    build_result(evaluation, budget)
}

/// Runs Latin hypercube sampling inside bounds.
pub fn latin_hypercube_search(
    objective: &dyn Objective,
    budget: &Budget,
    options: &LatinHypercubeOptions,
    seed: u64,
) -> OptimizationResult {
    let bounds = objective.bounds();
    let candidates = match latin_hypercube_samples(bounds, options.samples, seed) {
        Ok(candidates) => candidates,
        Err(message) => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(SolveStatus::Unsupported)
                .with_message(message);
        }
    };

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let evaluation = evaluate_candidates(&counted, &candidates);
    build_result(evaluation, budget)
}

/// Builds the shared result from one evaluation round.
fn build_result(evaluation: Evaluation, budget: &Budget) -> OptimizationResult {
    let mut result = OptimizationResult::new(evaluation.best_x, evaluation.best_value)
        .with_status(evaluation.status)
        .with_counts(budget.iterations(), budget.evaluations());
    if let Some(message) = evaluation.message {
        result = result.with_message(message);
    }
    result
}
