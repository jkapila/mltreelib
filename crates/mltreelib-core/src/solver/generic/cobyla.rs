//! COBYLA: nonlinear constrained optimization by linear interpolation.
//!
//! The method follows Powell (1994) and the PRIMA formulation of COBYLA. The
//! objective and the constraints are modelled by linear interpolation over an
//! interpolation set, and the model is minimized inside a shrinking trust
//! region. Only function values are used, so the method never needs a
//! gradient.
//!
//! # What the implementation does exactly
//!
//! - **Interpolation model.** Every model is a linear function fitted over the
//!   interpolation set by least squares and anchored at the reference point.
//!   The anchored form `value(x) = value(reference) + gradient · (x -
//!   reference)` reproduces the objective and constraint values of the
//!   reference point exactly.
//! - **Trust-region schedule.** The radius starts at `rhobeg`, doubles after an
//!   accepted step, halves after a rejected step, and the run stops when the
//!   radius falls below `rhobeg * 1e-8`.
//! - **Function values only.** No gradient is ever requested, so the method
//!   works with the plain [`Objective`](super::objective::Objective) trait.
//! - **Bounds.** Every evaluated point is projected into the box. The box is
//!   never relaxed.
//! - **Constraint semantics.** An inequality constraint is feasible when its
//!   value is at least `-feasibility_tol`, matching
//!   [`ConstraintKind::Inequality`](super::constraint::ConstraintKind).
//! - **Violation reporting.** The result reports the largest constraint
//!   violation at the returned point.
//!
//! # What the implementation approximates
//!
//! - **Model subproblem.** The published method solves the model subproblem as
//!   a linear program inside the trust region. This implementation projects
//!   the model objective gradient onto the feasible cone of the model
//!   constraints and takes that direction into the trust region. The
//!   projection is a cyclic projection onto the constraint halfspaces, so the
//!   step satisfies the model constraints up to the projection residual.
//! - **Interpolation set.** The published method interpolates exactly on a
//!   simplex of `n + 1` points. This implementation keeps the most recent
//!   evaluated points and fits the model by least squares, so the model is a
//!   best fit rather than an exact interpolation when the set holds more than
//!   `n + 1` points.
//! - **Trust region.** This implementation uses an infinity-norm ball instead
//!   of the published Euclidean trust region.
//! - **Feasibility repair.** When the reference point is infeasible, the step
//!   follows the model gradient of the worst violated constraint instead of
//!   the objective gradient.
//!
//! The method ships as a robust practical solver. It carries no optimality
//! claim.
//!
//! Equality constraints are outside the method. A request with
//! [`ConstraintKind::Equality`](super::constraint::ConstraintKind) returns
//! [`SolveStatus::Unsupported`] with a message that names the rejection.

use nalgebra::DMatrix;

use super::bounds::Bounds;
use super::budget::Budget;
use super::constraint::{max_violation, violation, ConstraintEvaluator, ConstraintKind};
use super::objective::{require_finite, CountedObjective, Objective, ObjectiveError};
use super::request::CobylaOptions;
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Growth factor of the trust region radius after an accepted step.
const RHO_GROWTH: f64 = 2.0;

/// Shrink factor of the trust region radius after a rejected step.
const RHO_SHRINK: f64 = 0.5;

/// Largest trust region radius as a multiple of the initial radius.
const MAX_RHO_FACTOR: f64 = 1000.0;

/// Smallest trust region radius as a fraction of the initial radius.
const MIN_RHO_FACTOR: f64 = 1e-8;

/// Hard cap on the number of iterations of one run.
const MAX_ITERATIONS: usize = 100_000;

/// Number of cyclic projection passes over the model constraints.
const PROJECTION_PASSES: usize = 20;

/// Number of consecutive model rebuilds before the run stops.
const MAX_REBUILDS: usize = 3;

/// Relative size below which a model direction counts as unusable.
const DIRECTION_EPS: f64 = 1e-14;

/// One evaluated point of the interpolation set.
#[derive(Clone, Debug)]
struct Sample {
    /// Point in parameter space.
    x: Vec<f64>,
    /// Objective value at the point.
    f: f64,
    /// Constraint values at the point.
    constraints: Vec<f64>,
}

/// Linear models of the objective and the constraints around one reference.
struct Model {
    /// Objective gradient of the model.
    objective: Vec<f64>,
    /// One gradient per constraint.
    constraints: Vec<Vec<f64>>,
}

/// Largest absolute value of a vector.
fn inf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |worst, value| worst.max(value.abs()))
}

/// Dot product of two vectors.
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

/// Turns one objective failure into a status.
fn failure_status(error: &ObjectiveError) -> SolveStatus {
    match error {
        ObjectiveError::BudgetExhausted(_) => SolveStatus::EvaluationLimit,
        _ => SolveStatus::InvalidObjective,
    }
}

/// Evaluates the objective and the constraints at one point.
///
/// The objective wrapper charges the shared budget for every call, so the
/// evaluation count of the result matches the number of objective calls.
fn evaluate(
    objective: &CountedObjective<'_>,
    constraints: Option<&dyn ConstraintEvaluator>,
    x: &[f64],
    expected_constraints: Option<usize>,
) -> Result<(f64, Vec<f64>), (SolveStatus, String)> {
    let value = objective
        .value(x)
        .and_then(require_finite)
        .map_err(|error| (failure_status(&error), error.to_string()))?;
    let values = match constraints {
        Some(evaluator) => evaluator.values(x).map_err(|message| {
            (
                SolveStatus::HardError,
                format!("constraint callback failed: {message}"),
            )
        })?,
        None => Vec::new(),
    };
    if expected_constraints.is_some_and(|count| count != values.len()) {
        return Err((
            SolveStatus::HardError,
            "constraint callback changed the number of constraints".into(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err((
            SolveStatus::HardError,
            "constraint callback returned a non-finite value".into(),
        ));
    }
    Ok((value, values))
}

/// Builds one point of a simplex around a point.
///
/// The point moves along the axis toward the side with more room if the
/// positive step does not fit. Fixed coordinates contribute no point.
fn simplex_point(point: &[f64], bounds: &Bounds, axis: usize, radius: f64) -> Option<Vec<f64>> {
    let center = point[axis];
    let upper = bounds.upper()[axis];
    let lower = bounds.lower()[axis];
    let positive = (center + radius).min(upper);
    let negative = (center - radius).max(lower);
    let next = if positive - center >= center - negative {
        positive
    } else {
        negative
    };
    if next == center || !next.is_finite() {
        return None;
    }
    let mut candidate = point.to_vec();
    candidate[axis] = next;
    Some(candidate)
}

/// Builds the linear model of the objective and the constraints.
///
/// The model is a least-squares fit of a linear function over the
/// interpolation set, anchored at the reference point. The fit uses the
/// singular value decomposition of the difference matrix, so a rank-deficient
/// interpolation set yields the minimum-norm gradient instead of a failure.
fn build_model(samples: &[Sample], reference: &Sample) -> Model {
    let dimension = reference.x.len();
    let count = samples.len();
    let constraint_count = reference.constraints.len();

    let mut matrix = DMatrix::<f64>::zeros(count, dimension);
    let mut objective_rhs = DMatrix::<f64>::zeros(count, 1);
    let mut constraint_rhs = DMatrix::<f64>::zeros(count, constraint_count);
    for (index, sample) in samples.iter().enumerate() {
        for axis in 0..dimension {
            matrix[(index, axis)] = sample.x[axis] - reference.x[axis];
        }
        objective_rhs[(index, 0)] = sample.f - reference.f;
        for (column, &value) in sample.constraints.iter().enumerate() {
            constraint_rhs[(index, column)] = value - reference.constraints[column];
        }
    }

    let svd = matrix.svd(true, true);
    let objective = match svd.solve(&objective_rhs, 1e-12) {
        Ok(solution) => solution.iter().copied().collect(),
        Err(_) => vec![0.0; dimension],
    };
    let constraints = if constraint_count == 0 {
        Vec::new()
    } else {
        match svd.solve(&constraint_rhs, 1e-12) {
            Ok(solution) => (0..constraint_count)
                .map(|column| (0..dimension).map(|row| solution[(row, column)]).collect())
                .collect(),
            Err(_) => vec![vec![0.0; dimension]; constraint_count],
        }
    };
    Model {
        objective,
        constraints,
    }
}

/// Projects a model direction onto the model constraints.
///
/// Every model constraint is a halfspace, so the projection of one point onto
/// one halfspace is a single shift along the constraint gradient. The cyclic
/// projection leaves a direction that satisfies every constraint which holds
/// at the reference point, up to the residual of the last pass.
fn project_onto_constraints(direction: &mut [f64], model: &Model, reference: &Sample) {
    for _ in 0..PROJECTION_PASSES {
        let mut violated = false;
        for (index, gradient) in model.constraints.iter().enumerate() {
            let value = reference.constraints[index];
            if value < 0.0 {
                // The constraint already fails at the reference point, so it
                // holds no projection target.
                continue;
            }
            let current = value + dot(gradient, direction);
            if current >= 0.0 {
                continue;
            }
            let squared = dot(gradient, gradient);
            if squared <= 0.0 {
                continue;
            }
            violated = true;
            let shift = -current / squared;
            for (component, &gradient_value) in direction.iter_mut().zip(gradient.iter()) {
                *component += shift * gradient_value;
            }
        }
        if !violated {
            break;
        }
    }
}

/// Picks the reference point of the interpolation set.
///
/// A feasible point with the smallest objective value wins. Without a feasible
/// point the least violated point wins.
fn select_reference(samples: &[Sample], kind: ConstraintKind, tolerance: f64) -> usize {
    let mut best = 0usize;
    let mut best_feasible = false;
    let mut best_violation = f64::INFINITY;
    for (index, sample) in samples.iter().enumerate() {
        let amount = max_violation(kind, &sample.constraints);
        let feasible = amount <= tolerance;
        match (feasible, best_feasible) {
            (true, false) => {
                best = index;
                best_feasible = true;
                best_violation = amount;
            }
            (true, true) => {
                if sample.f < samples[best].f {
                    best = index;
                }
            }
            (false, true) => {}
            (false, false) => {
                if amount < best_violation {
                    best = index;
                    best_violation = amount;
                }
            }
        }
    }
    best
}

/// Keeps the interpolation set bounded.
///
/// The point farthest from the reference leaves the set first. The reference
/// itself stays in the set.
fn trim_samples(samples: &mut Vec<Sample>, reference: &mut usize, limit: usize) {
    if samples.len() <= limit {
        return;
    }
    let anchor = &samples[*reference].x;
    let mut worst = usize::MAX;
    let mut worst_distance = f64::NEG_INFINITY;
    for (index, sample) in samples.iter().enumerate() {
        if index == *reference {
            continue;
        }
        let distance = sample
            .x
            .iter()
            .zip(anchor)
            .map(|(value, center)| (value - center).abs())
            .fold(0.0, f64::max);
        if distance > worst_distance {
            worst_distance = distance;
            worst = index;
        }
    }
    if worst == usize::MAX {
        return;
    }
    samples.remove(worst);
    if worst < *reference {
        *reference -= 1;
    }
}

/// Rebuilds the interpolation set as a fresh simplex around the reference.
///
/// The rebuild gives the model new points when the fitted gradient carries no
/// usable direction.
fn rebuild_simplex(
    samples: &mut Vec<Sample>,
    reference: &mut usize,
    objective: &CountedObjective<'_>,
    constraints: Option<&dyn ConstraintEvaluator>,
    bounds: &Bounds,
    radius: f64,
) -> Result<(), (SolveStatus, String)> {
    let anchor = samples[*reference].x.clone();
    let dimension = anchor.len();
    let mut fresh: Vec<Sample> = Vec::with_capacity(dimension + 1);
    fresh.push(samples[*reference].clone());
    for axis in 0..dimension {
        let Some(candidate) = simplex_point(&anchor, bounds, axis, radius) else {
            continue;
        };
        if candidate == anchor {
            continue;
        }
        let (value, values) =
            evaluate(objective, constraints, &candidate, Some(fresh[0].constraints.len()))?;
        fresh.push(Sample {
            x: candidate,
            f: value,
            constraints: values,
        });
    }
    *samples = fresh;
    *reference = 0;
    Ok(())
}

/// Runs COBYLA on a bounded problem with inequality constraints.
///
/// The starting point defaults to the middle of the box. The method stops
/// when the trust region collapses, when the model step vanishes, when the
/// iteration limit is reached, or when a budget ends the run.
pub fn cobyla(
    objective: &dyn Objective,
    budget: &Budget,
    options: &CobylaOptions,
    start: Option<&[f64]>,
    constraints: Option<&dyn ConstraintEvaluator>,
    kind: ConstraintKind,
) -> OptimizationResult {
    let bounds = objective.bounds();
    let dimension = objective.dim();

    // The starting point resolves before any objective call, so an
    // unsupported request never spends an evaluation.
    let start_point: Vec<f64> = match start {
        Some(start) => {
            if start.len() != dimension {
                return OptimizationResult::new(Vec::new(), f64::NAN)
                    .with_status(SolveStatus::HardError)
                    .with_message(format!(
                        "cobyla starting point has {} values, objective has {} dimensions",
                        start.len(),
                        dimension
                    ));
            }
            bounds.project(start)
        }
        None if bounds.is_bounded() => bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .map(|(&low, &high)| low.midpoint(high))
            .collect(),
        None => {
            return OptimizationResult::new(Vec::new(), f64::NAN)
                .with_status(SolveStatus::Unsupported)
                .with_message("cobyla needs a starting point for unbounded domains");
        }
    };

    // Equality constraints sit outside the method.
    if kind == ConstraintKind::Equality {
        return OptimizationResult::new(start_point, f64::NAN)
            .with_status(SolveStatus::Unsupported)
            .with_message(
                "cobyla rejects equality constraints: only inequality constraints are supported",
            );
    }

    if dimension == 0 {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("cobyla needs at least one parameter");
    }

    // One wrapper charges the budget for every objective call.
    let counted = CountedObjective::new(objective, budget);
    let tolerance = options.feasibility_tol.max(0.0);
    let initial_rho = options.rhobeg.max(1e-12);
    let mut rho = initial_rho;

    let mut samples: Vec<Sample> = Vec::new();
    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut iterations = 0usize;
    let mut status;
    let mut message: Option<String> = None;

    let (start_value, start_constraints) = match evaluate(&counted, constraints, &start_point, None) {
        Ok(values) => values,
        Err((status, message)) => {
            return OptimizationResult::new(start_point, f64::NAN)
                .with_status(status)
                .with_counts(iterations, budget.evaluations())
                .with_message(message);
        }
    };
    samples.push(Sample {
        x: start_point,
        f: start_value,
        constraints: start_constraints,
    });
    let constraint_count = samples[0].constraints.len();

    // The initial simplex gives the model enough points to fit a gradient.
    let mut initialization_failure = None;
    for axis in 0..dimension {
        let Some(candidate) = simplex_point(&samples[0].x, bounds, axis, initial_rho) else {
            continue;
        };
        let (value, values) = match evaluate(&counted, constraints, &candidate, Some(constraint_count)) {
            Ok(values) => values,
            Err(failure) => {
                initialization_failure = Some(failure);
                break;
            }
        };
        samples.push(Sample {
            x: candidate,
            f: value,
            constraints: values,
        });
    }

    let mut reference = select_reference(&samples, kind, tolerance);
    let mut best_feasible: Option<Sample> = None;
    if max_violation(kind, &samples[reference].constraints) <= tolerance {
        best_feasible = Some(samples[reference].clone());
    }
    let limit = dimension + 2;
    let mut rebuilds = 0usize;
    status = SolveStatus::Converged;
    'outer: loop {
        if let Some((failed, detail)) = initialization_failure.take() {
            status = failed;
            message = Some(detail);
            break;
        }
        if iterations >= MAX_ITERATIONS {
            status = SolveStatus::IterationLimit;
            break;
        }
        if budget.charge_iteration().is_err() {
            status = SolveStatus::IterationLimit;
            break;
        }
        iterations += 1;

        let model = build_model(&samples, &samples[reference]);
        let reference_violation = max_violation(kind, &samples[reference].constraints);
        let reference_feasible = reference_violation <= tolerance;

        // The direction follows the objective model, or the model of the worst
        // violated constraint while the reference point is infeasible.
        let mut direction: Vec<f64> = if reference_feasible {
            model.objective.iter().map(|value| -value).collect()
        } else {
            let mut worst = 0usize;
            let mut worst_violation = f64::NEG_INFINITY;
            for (index, &value) in samples[reference].constraints.iter().enumerate() {
                let amount = violation(kind, value);
                if amount > worst_violation {
                    worst_violation = amount;
                    worst = index;
                }
            }
            match model.constraints.get(worst) {
                Some(gradient) => gradient.clone(),
                None => Vec::new(),
            }
        };

        if inf_norm(&direction) <= DIRECTION_EPS {
            // The model carries no usable direction. A fresh simplex around the
            // reference gives the model new points.
            rebuilds += 1;
            if rebuilds > MAX_REBUILDS {
                break 'outer;
            }
            if let Err((failed, detail)) = rebuild_simplex(
                &mut samples,
                &mut reference,
                &counted,
                constraints,
                bounds,
                rho,
            ) {
                status = failed;
                message = Some(detail);
                break 'outer;
            }
            continue;
        }
        rebuilds = 0;

        // The step follows the feasible part of the model direction.
        project_onto_constraints(&mut direction, &model, &samples[reference]);
        let norm = inf_norm(&direction);
        if norm <= DIRECTION_EPS {
            // The feasible model step vanishes, so the model step reaches the
            // constraint boundary and the run has no further progress.
            break 'outer;
        }

        // Scale the step into the trust region.
        let scale = (rho / norm).min(1.0);
        let mut candidate = samples[reference].x.clone();
        for (axis, value) in direction.iter().enumerate() {
            candidate[axis] += scale * value;
        }
        let candidate = bounds.project(&candidate);

        let (candidate_value, candidate_constraints) =
            match evaluate(&counted, constraints, &candidate, Some(constraint_count)) {
                Ok(values) => values,
                Err((failed, detail)) => {
                    status = failed;
                    message = Some(detail);
                    break 'outer;
                }
            };
        let candidate_violation = max_violation(kind, &candidate_constraints);
        let candidate_feasible = candidate_violation <= tolerance;

        let accepted = if reference_feasible {
            candidate_feasible && candidate_value < samples[reference].f
        } else {
            candidate_feasible || candidate_violation < reference_violation - 1e-12
        };

        samples.push(Sample {
            x: candidate,
            f: candidate_value,
            constraints: candidate_constraints,
        });
        if accepted {
            rho = (rho * RHO_GROWTH).min(initial_rho * MAX_RHO_FACTOR);
            reference = samples.len() - 1;
        } else {
            rho *= RHO_SHRINK;
        }
        if candidate_feasible
            && best_feasible
                .as_ref()
                .map(|best| candidate_value < best.f)
                .unwrap_or(true)
        {
            best_feasible = Some(samples[samples.len() - 1].clone());
        }
        trim_samples(&mut samples, &mut reference, limit);

        let current = match &best_feasible {
            Some(best) => best.f,
            None => samples[reference].f,
        };
        history.push(HistoryEntry {
            iteration: iterations,
            value: current,
            best_value: current,
        });

        if rho < initial_rho * MIN_RHO_FACTOR {
            break 'outer;
        }
    }

    let (best_x, best_value, best_violation) = match &best_feasible {
        Some(best) => (best.x.clone(), best.f, max_violation(kind, &best.constraints)),
        None => {
            if status == SolveStatus::Converged {
                status = SolveStatus::Infeasible;
                message = Some("cobyla found no feasible point".to_string());
            }
            let sample = &samples[reference];
            (
                sample.x.clone(),
                sample.f,
                max_violation(kind, &sample.constraints),
            )
        }
    };

    let mut result = OptimizationResult::new(best_x, best_value)
        .with_status(status)
        .with_counts(iterations, budget.evaluations());
    if !history.is_empty() {
        result.history = Some(history);
    }
    if constraints.is_some() {
        result.max_constraint_violation = Some(best_violation);
    }
    if let Some(message) = message {
        result.message = Some(message);
    }
    result
}
