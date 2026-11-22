//! Adapts the shared population kernels to a bounded scalar objective.

use crate::solver::evolutionary::{run_population, PopulationOptions};
use crate::solver::{solver_capabilities, EvolutionaryKind};

use super::{
    require_finite, Budget, BudgetStatus, CountedObjective, Objective, ObjectiveError,
    OptimizationRequest, OptimizationResult, SolveStatus,
};

pub(super) fn optimize(
    request: &OptimizationRequest<'_>,
    budget: &Budget,
    kind: EvolutionaryKind,
) -> OptimizationResult {
    let objective = CountedObjective::new(request.objective, budget);
    let bounds = objective.bounds();
    let capabilities = solver_capabilities(kind.leaf_solver());
    let bounded = bounds.has_finite_limit();
    if (bounded && !capabilities.supports_finite_bounds)
        || (!bounds.is_bounded() && !capabilities.supports_unbounded)
    {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message(format!(
                "{} does not support the requested bounds",
                kind.as_str()
            ));
    }
    if bounds.dim() == 0 || bounds.dim() != objective.dim() {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("expected one bound pair per parameter and at least one parameter");
    }
    let mut best_x = Vec::new();
    let mut best_value = f64::INFINITY;
    let mut iterations = 0;
    let options = PopulationOptions {
        population_size: None,
        max_generations: request.shared.max_iterations.unwrap_or(100),
        tol: request.shared.tol,
        seed: request.shared.seed,
        verbose: request.shared.verbose,
    };

    let run = (|| {
        let mut start = request.start.clone().unwrap_or_else(|| {
            bounds
                .lower()
                .iter()
                .zip(bounds.upper())
                .map(|(&lo, &hi)| {
                    if lo.is_finite() && hi.is_finite() {
                        lo.midpoint(hi)
                    } else {
                        0.0_f64.clamp(lo, hi)
                    }
                })
                .collect()
        });
        let mut evaluate = |point: &mut [f64]| {
            for ((value, &lo), &hi) in point.iter_mut().zip(bounds.lower()).zip(bounds.upper()) {
                *value = require_finite(*value)?.clamp(lo, hi);
            }
            let value = require_finite(objective.value(point)?)?;
            if value < best_value {
                best_value = value;
                best_x.clear();
                best_x.extend_from_slice(point);
            }
            Ok(value)
        };
        if let Some(points) = &request.shared.initial_population {
            let mut point = Vec::with_capacity(start.len());
            let mut seed_value = f64::INFINITY;
            for seed in points {
                super::require_dimension(start.len(), seed)?;
                point.clear();
                point.extend_from_slice(seed);
                let value = evaluate(&mut point)?;
                if value < seed_value {
                    start.copy_from_slice(&point);
                    seed_value = value;
                }
            }
        }
        run_population(kind, start, &options, evaluate, || {
            budget
                .charge_iteration()
                .map_err(ObjectiveError::BudgetExhausted)?;
            iterations += 1;
            Ok(())
        })
    })();
    let (status, message) = match run {
        Ok((_, report)) if report.converged() => (SolveStatus::Tolerance, None),
        Ok(_) | Err(ObjectiveError::BudgetExhausted(BudgetStatus::IterationLimit)) => {
            (SolveStatus::IterationLimit, None)
        }
        Err(ObjectiveError::BudgetExhausted(BudgetStatus::EvaluationLimit)) => {
            (SolveStatus::EvaluationLimit, None)
        }
        Err(error) => (SolveStatus::InvalidObjective, Some(error.to_string())),
    };
    let mut result = OptimizationResult::new(
        best_x,
        if best_value.is_finite() {
            best_value
        } else {
            f64::NAN
        },
    )
    .with_counts(iterations, budget.evaluations())
    .with_status(status);
    result.message = message;
    result
}
