//! Basin hopping over the shared seam.
//!
//! The method perturbs the incumbent, runs an inner local solve through the
//! same contracts, and accepts the result with a Metropolis test. The inner
//! solve shares the outer budget, so no objective call escapes the count. A
//! failed inner solve never erases the incumbent.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use super::budget::Budget;
use super::objective::{CountedObjective, Objective, ObjectiveError, SmoothObjective};
use super::request::{BasinHoppingOptions, OptimizerKind};
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Number of iterations between step-size adjustments.
const ADAPTIVE_WINDOW: usize = 10;

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Runs basin hopping.
///
/// The inner local solver comes from the shared seam. The perturbation scales
/// with the bound width, so one step size fits heterogeneous parameter scales.
pub fn basin_hopping(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    budget: &Budget,
    options: &BasinHoppingOptions,
    seed: u64,
    start: Option<&[f64]>,
) -> OptimizationResult {
    let bounds = objective.bounds();
    match options.inner {
        OptimizerKind::Bfgs | OptimizerKind::NonlinearConjugateGradient
            if bounds.has_finite_limit() =>
        {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(SolveStatus::Unsupported)
                .with_message("this basin hopping local method requires unbounded parameters");
        }
        OptimizerKind::LbfgsB
        | OptimizerKind::Bfgs
        | OptimizerKind::NonlinearConjugateGradient
        | OptimizerKind::NelderMead
        | OptimizerKind::Powell => {}
        _ => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(SolveStatus::Unsupported)
                .with_message("basin hopping requires a supported local method");
        }
    }
    let mut current = match start {
        Some(start) => bounds.project(start),
        None => bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(&lo, &hi)| {
                if lo.is_finite() && hi.is_finite() {
                    lo.midpoint(hi)
                } else {
                    0.0_f64.clamp(lo, hi)
                }
            })
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

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut local_solves = 0usize;
    let mut status = SolveStatus::Converged;
    let mut step_size = options.step_size;
    let mut window_accepts = 0usize;
    let mut window_attempts = 0usize;

    'outer: for _ in 0..options.iterations {
        if budget.charge_iteration().is_err() {
            status = SolveStatus::IterationLimit;
            break;
        }
        iterations += 1;

        // Perturb the incumbent.
        let perturbed: Vec<f64> = (0..current.len())
            .map(|axis| {
                let width = bounds.width(axis);
                let width = if width.is_finite() { width } else { 1.0 };
                current[axis] + step_size * width * normal.sample(&mut rng)
            })
            .collect();
        let perturbed = bounds.project(&perturbed);

        // Inner local solve through the shared seam.
        let inner = super::optimize_with_budget(
            &super::inner_request(
                objective,
                smooth,
                options.inner,
                Some(perturbed),
                budget,
                seed,
            ),
            budget,
        );
        let candidate = match inner {
            Ok(inner) => {
                local_solves += 1;
                if inner.status == SolveStatus::EvaluationLimit {
                    if inner.best_value < best_value && bounds.contains(&inner.best_x) {
                        best_value = inner.best_value;
                        best_point = inner.best_x;
                    }
                    status = SolveStatus::EvaluationLimit;
                    break 'outer;
                }
                if !inner.best_value.is_finite() || !bounds.contains(&inner.best_x) {
                    // The inner solve failed. Keep the incumbent and continue.
                    window_attempts += 1;
                    history.push(HistoryEntry {
                        iteration: iterations,
                        value: current_value,
                        best_value,
                    });
                    continue;
                }
                inner
            }
            Err(_) => {
                // A failed inner solve never erases the incumbent.
                window_attempts += 1;
                history.push(HistoryEntry {
                    iteration: iterations,
                    value: current_value,
                    best_value,
                });
                continue;
            }
        };

        let candidate_value = candidate.best_value;
        let delta = candidate_value - current_value;
        let accept = if options.monotonic {
            delta <= 0.0
        } else {
            delta <= 0.0 || rng.gen::<f64>() < (-delta / options.temperature).exp()
        };
        window_attempts += 1;
        if accept {
            window_accepts += 1;
            current = candidate.best_x.clone();
            current_value = candidate_value;
            if current_value < best_value {
                best_value = current_value;
                best_point = current.clone();
            }
        }

        history.push(HistoryEntry {
            iteration: iterations,
            value: current_value,
            best_value,
        });

        // Adaptive step size toward the target acceptance rate.
        if let Some(target) = options.target_acceptance {
            if window_attempts >= ADAPTIVE_WINDOW {
                let rate = window_accepts as f64 / window_attempts as f64;
                if rate < target {
                    step_size *= 0.8;
                } else {
                    step_size *= 1.2;
                }
                step_size = step_size.clamp(1e-6, 1.0);
                window_accepts = 0;
                window_attempts = 0;
            }
        }
    }

    let mut result = OptimizationResult::new(best_point, best_value)
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    result.local_solve_count = Some(local_solves);
    if !history.is_empty() {
        result.history = Some(history);
    }
    result
}
