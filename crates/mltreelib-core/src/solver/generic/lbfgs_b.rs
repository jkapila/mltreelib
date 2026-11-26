//! L-BFGS-B bounded local optimizer.
//!
//! This is a compact implementation of the bounded quasi-Newton iteration
//! from Byrd, Lu, Nocedal and Zhu (1995).  Each iteration computes the
//! generalized Cauchy point (GCP), constructs a limited-memory inverse-Hessian
//! direction on the GCP free set, and performs a projected line search.
//!
//! Simplifications relative to the Fortran reference:
//!
//! * The compact inverse-Hessian approximation is used as the quadratic
//!   curvature model for the GCP and the subspace step.  The reference uses a
//!   full reduced-Hessian model in the subspace minimization.
//! * The free-set subspace step is an approximate quasi-Newton minimizer,
//!   rather than an exact reduced-Hessian solve.
//! * The line search uses the projected path and a strong Wolfe test with
//!   deterministic backtracking, not the reference's safeguarded interval
//!   search.
//! * No scaling of the initial inverse-Hessian matrix is performed beyond
//!   the usual `s'y / y'y` scaling used by the two-loop recursion.
//!
//! The implementation still follows the defining pieces: a genuine GCP over
//! the piecewise-linear projected-gradient path, a free-variable subspace
//! direction from the two-loop recursion, projection after every accepted
//! step, and convergence based on the projected gradient over free variables.

use super::bounds::Bounds;
use super::budget::{Budget, BudgetStatus};
use super::objective::{CountedObjective, Objective, ObjectiveError, SmoothObjective};
use super::request::LbfgsBOptions;
use super::result::{HistoryEntry, OptimizationResult, SolveStatus};

/// Limited-memory inverse-Hessian correction pairs.
#[derive(Clone, Debug)]
struct CurvatureMemory {
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,
    limit: usize,
}

impl CurvatureMemory {
    fn new(limit: usize) -> Self {
        Self {
            s: Vec::new(),
            y: Vec::new(),
            rho: Vec::new(),
            limit,
        }
    }

    /// Applies the compact inverse-Hessian approximation with the two-loop
    /// recursion used by the leaf L-BFGS solver.
    fn apply(&self, vector: &[f64]) -> Vec<f64> {
        let dim = vector.len();
        let count = self.s.len();
        if count == 0 {
            return vector.to_vec();
        }
        let mut q = vector.to_vec();
        let mut alpha = vec![0.0; count];
        for i in (0..count).rev() {
            let dot = dot(&self.s[i], &q);
            alpha[i] = self.rho[i] * dot;
            for j in 0..dim {
                q[j] -= alpha[i] * self.y[i][j];
            }
        }
        let last = count - 1;
        let sy = dot(&self.s[last], &self.y[last]);
        let yy = dot(&self.y[last], &self.y[last]);
        let gamma = if yy > 1e-14 {
            (sy / yy).clamp(1e-6, 1e6)
        } else {
            1.0
        };
        for j in 0..dim {
            q[j] *= gamma;
        }
        for i in 0..count {
            let yq = dot(&self.y[i], &q);
            let beta = self.rho[i] * yq;
            let coefficient = alpha[i] - beta;
            for j in 0..dim {
                q[j] += coefficient * self.s[i][j];
            }
        }
        q
    }

    fn update(&mut self, s: Vec<f64>, y: Vec<f64>) {
        let sy = dot(&s, &y);
        if !sy.is_finite() || sy <= 1e-12 {
            return;
        }
        if self.limit == 0 {
            return;
        }
        if self.s.len() == self.limit {
            self.s.remove(0);
            self.y.remove(0);
            self.rho.remove(0);
        }
        self.s.push(s);
        self.y.push(y);
        self.rho.push(1.0 / sy);
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// One breakpoint and the path displacement on the following interval.
#[derive(Clone)]
struct PathSegment {
    start: f64,
    end: f64,
    /// Dimension is active (fixed at a bound) on this interval.
    active: Vec<bool>,
}

/// Computes the generalized Cauchy point on x(t)=P(x-tg).
///
/// The path is piecewise linear because each coordinate either moves with
/// `-t*g_i` or stays at a bound.  We form every constant-active-set interval,
/// minimize the compact quadratic model on each interval, and select the
/// lowest model value.  This is the central GCP construction, not a single
/// projected-gradient step.
fn generalized_cauchy_point(
    x: &[f64],
    gradient: &[f64],
    bounds: &Bounds,
    memory: &CurvatureMemory,
) -> (Vec<f64>, f64) {
    let dim = x.len();
    let mut breakpoints = Vec::new();
    let mut hit_times = vec![f64::INFINITY; dim];
    for i in 0..dim {
        let lo = bounds.lower()[i];
        let hi = bounds.upper()[i];
        let g = gradient[i];
        let hit = if g > 0.0 && lo.is_finite() {
            (x[i] - lo) / g
        } else if g < 0.0 && hi.is_finite() {
            (hi - x[i]) / (-g)
        } else {
            f64::INFINITY
        };
        if hit.is_finite() && hit > 0.0 {
            hit_times[i] = hit;
            breakpoints.push(hit);
        }
    }
    breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    breakpoints.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

    let mut segments = Vec::new();
    let mut start = 0.0;
    for &end in &breakpoints {
        if end > start {
            let active = (0..dim)
                .map(|i| {
                    hit_times[i] <= start + 1e-12 || {
                        let lo = bounds.lower()[i];
                        let hi = bounds.upper()[i];
                        (lo.is_finite() && x[i] <= lo + 1e-12 && gradient[i] > 0.0)
                            || (hi.is_finite() && x[i] >= hi - 1e-12 && gradient[i] < 0.0)
                    }
                })
                .collect();
            segments.push(PathSegment { start, end, active });
            start = end;
        }
    }
    let active = (0..dim)
        .map(|i| {
            let lo = bounds.lower()[i];
            let hi = bounds.upper()[i];
            hit_times[i] <= start + 1e-12
                || (lo.is_finite() && x[i] <= lo + 1e-12 && gradient[i] > 0.0)
                || (hi.is_finite() && x[i] >= hi - 1e-12 && gradient[i] < 0.0)
        })
        .collect();
    segments.push(PathSegment {
        start,
        end: f64::INFINITY,
        active,
    });

    let mut best_t = 0.0;
    let mut best_model = 0.0;
    for segment in &segments {
        let (d0, slope) = path_displacement(x, gradient, bounds, segment);
        let hs = memory.apply(&slope);
        let linear = dot(gradient, &slope) + dot(&d0, &hs);
        let quadratic = 0.5 * dot(&slope, &hs);
        let mut candidates = vec![segment.start];
        if segment.end.is_finite() {
            candidates.push(segment.end);
        }
        if quadratic > 1e-14 {
            let candidate = -linear / (2.0 * quadratic);
            if candidate.is_finite() && candidate >= segment.start {
                candidates.push(candidate.min(segment.end));
            }
        }
        for &candidate in &candidates {
            if candidate < segment.start || (segment.end.is_finite() && candidate > segment.end) {
                continue;
            }
            let displacement = displacement_at(&d0, &slope, candidate);
            let model = dot(gradient, &displacement)
                + 0.5 * dot(&displacement, &memory.apply(&displacement));
            if candidate == 0.0 || model < best_model {
                best_model = model;
                best_t = candidate;
            }
        }
    }
    let mut point = x.to_vec();
    for i in 0..dim {
        point[i] -= best_t * gradient[i];
    }
    (bounds.project(&point), best_t)
}

/// Returns constant and linear path-displacement components.
fn path_displacement(
    x: &[f64],
    gradient: &[f64],
    bounds: &Bounds,
    segment: &PathSegment,
) -> (Vec<f64>, Vec<f64>) {
    let mut d0 = vec![0.0; x.len()];
    let mut slope = vec![0.0; x.len()];
    let projected_at_start = bounds.project(
        &x.iter()
            .zip(gradient)
            .map(|(value, g)| value - segment.start * g)
            .collect::<Vec<_>>(),
    );
    for i in 0..x.len() {
        if segment.active[i] {
            // At the beginning of this interval the coordinate is fixed at a
            // bound.  Its displacement therefore includes the distance already
            // travelled before the breakpoint.
            d0[i] = projected_at_start[i] - x[i];
        } else {
            slope[i] = -gradient[i];
        }
    }
    (d0, slope)
}

fn displacement_at(d0: &[f64], slope: &[f64], t: f64) -> Vec<f64> {
    d0.iter().zip(slope).map(|(a, b)| a + b * t).collect()
}

/// Returns the free set and its quasi-Newton subspace direction.
fn subspace_direction(
    x: &[f64],
    gradient: &[f64],
    bounds: &Bounds,
    memory: &CurvatureMemory,
) -> Vec<f64> {
    let mut direction = vec![0.0; x.len()];
    let mut free = vec![false; x.len()];
    for i in 0..x.len() {
        let lo = bounds.lower()[i];
        let hi = bounds.upper()[i];
        let at_lower = lo.is_finite() && x[i] <= lo + 1e-12;
        let at_upper = hi.is_finite() && x[i] >= hi - 1e-12;
        free[i] = !((at_lower && gradient[i] > 0.0) || (at_upper && gradient[i] < 0.0));
    }
    let h_gradient = memory.apply(gradient);
    for i in 0..x.len() {
        if free[i] {
            direction[i] = -h_gradient[i];
        }
    }
    let derivative = dot(gradient, &direction);
    if derivative >= -1e-14 {
        for i in 0..x.len() {
            if free[i] {
                direction[i] = -gradient[i];
            }
        }
    }
    direction
}

/// Performs a projected line search satisfying a strong Wolfe condition.
fn projected_line_search(
    x: &[f64],
    value: f64,
    gradient: &[f64],
    direction: &[f64],
    bounds: &Bounds,
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
) -> Result<Option<(Vec<f64>, f64, Vec<f64>, f64)>, ObjectiveError> {
    let derivative = dot(gradient, direction);
    if !derivative.is_finite() || derivative >= 0.0 {
        return Ok(None);
    }
    let c1 = 1e-4;
    let c2 = 0.9;
    let mut alpha = 1.0;
    for _ in 0..40 {
        let trial = bounds.project(
            &x.iter()
                .zip(direction)
                .map(|(a, b)| a + alpha * b)
                .collect::<Vec<_>>(),
        );
        let trial_value = objective.value(&trial)?;
        let trial_delta: Vec<f64> = trial.iter().zip(x).map(|(a, b)| a - b).collect();
        if trial_value <= value + c1 * alpha * derivative {
            let trial_gradient = smooth.gradient(&trial)?;
            let trial_projected = dot(&trial_gradient, &trial_delta) / alpha;
            if trial_projected.abs() <= c2 * derivative.abs() {
                return Ok(Some((trial, trial_value, trial_gradient, alpha)));
            }
        }
        // A deterministic backtracking safeguard.  It also handles projected
        // paths whose derivative is not monotone because a coordinate reaches
        // a bound during the search.
        alpha *= 0.5;
        if alpha < 1e-14 {
            break;
        }
    }
    Ok(None)
}

/// Runs L-BFGS-B on one bounded smooth objective.
///
/// The returned point is always projected into the box.  Evaluation budget
/// exhaustion is reported as `EvaluationLimit`; iteration budget exhaustion is
/// reported as `IterationLimit`.
pub fn solve_lbfgs_b(
    objective: &dyn Objective,
    budget: &Budget,
    options: &LbfgsBOptions,
    start: Option<&[f64]>,
    smooth: &dyn SmoothObjective,
) -> OptimizationResult {
    let bounds = objective.bounds();
    let dim = objective.dim();
    let counted = CountedObjective::new(objective, budget);
    let mut x = start
        .map(|value| value.to_vec())
        .unwrap_or_else(|| vec![0.0; dim]);
    if x.len() != dim {
        return OptimizationResult::new(Vec::new(), f64::NAN)
            .with_status(SolveStatus::HardError)
            .with_message("starting point has wrong dimension");
    }
    x = bounds.project(&x);
    let mut value = match counted.value(&x) {
        Ok(value) => value,
        Err(error) => return failure_result(error, Vec::new(), f64::NAN, budget),
    };
    let mut best_x = x.clone();
    let mut best_value = value;
    let mut memory = CurvatureMemory::new(options.m.max(1));
    let mut history = Vec::new();
    let mut iterations = 0usize;

    for _ in 0..options.max_iter {
        let gradient = match smooth.gradient(&x) {
            Ok(gradient) => gradient,
            Err(error) => return failure_result(error, best_x, best_value, budget),
        };
        let projected_gradient = projected_gradient_norm(&x, &gradient, bounds);
        if projected_gradient <= options.gtol {
            return build_result(
                best_x,
                best_value,
                iterations,
                budget,
                SolveStatus::Converged,
                Some(history),
                None,
            );
        }
        if let Err(status) = budget.charge_iteration() {
            let _ = status;
            return build_result(
                best_x,
                best_value,
                iterations,
                budget,
                SolveStatus::IterationLimit,
                Some(history),
                None,
            );
        }
        iterations += 1;

        let (gcp, _) = generalized_cauchy_point(&x, &gradient, bounds, &memory);
        let mut direction = subspace_direction(&x, &gradient, bounds, &memory);
        // The GCP is used to identify the active set.  Free coordinates that
        // moved along the projected path remain free in the subspace.
        for i in 0..dim {
            let lo = bounds.lower()[i];
            let hi = bounds.upper()[i];
            let active = (lo.is_finite() && gcp[i] <= lo + 1e-12 && gradient[i] > 0.0)
                || (hi.is_finite() && gcp[i] >= hi - 1e-12 && gradient[i] < 0.0);
            if active {
                direction[i] = 0.0;
            }
        }
        if dot(&gradient, &direction) >= -1e-14 {
            direction = vec![0.0; dim];
            for i in 0..dim {
                if !(bounds.lower()[i].is_finite()
                    && x[i] <= bounds.lower()[i] + 1e-12
                    && gradient[i] > 0.0
                    || bounds.upper()[i].is_finite()
                        && x[i] >= bounds.upper()[i] - 1e-12
                        && gradient[i] < 0.0)
                {
                    direction[i] = -gradient[i];
                }
            }
        }
        let line =
            match projected_line_search(&x, value, &gradient, &direction, bounds, &counted, smooth)
            {
                Ok(line) => line,
                Err(error) => return failure_result(error, best_x, best_value, budget),
            };
        let Some((new_x, new_value, new_gradient, _alpha)) = line else {
            return build_result(
                best_x,
                best_value,
                iterations,
                budget,
                SolveStatus::NumericalFailure,
                Some(history),
                Some("projected strong-Wolfe line search failed"),
            );
        };
        let s: Vec<f64> = new_x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = new_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        memory.update(s, y);
        let improvement = value - new_value;
        x = new_x;
        value = new_value;
        if value < best_value {
            best_value = value;
            best_x = x.clone();
        }
        history.push(HistoryEntry {
            iteration: iterations,
            value,
            best_value,
        });
        if improvement <= options.ftol * (1.0 + value.abs()) {
            return build_result(
                best_x,
                best_value,
                iterations,
                budget,
                SolveStatus::Converged,
                Some(history),
                None,
            );
        }
    }
    build_result(
        best_x,
        best_value,
        iterations,
        budget,
        SolveStatus::IterationLimit,
        Some(history),
        None,
    )
}

/// Projected gradient infinity norm over free variables only.
fn projected_gradient_norm(x: &[f64], gradient: &[f64], bounds: &Bounds) -> f64 {
    let mut norm = 0.0_f64;
    for i in 0..x.len() {
        let lo = bounds.lower()[i];
        let hi = bounds.upper()[i];
        let active = (lo.is_finite() && x[i] <= lo + 1e-12 && gradient[i] > 0.0)
            || (hi.is_finite() && x[i] >= hi - 1e-12 && gradient[i] < 0.0);
        if !active {
            norm = norm.max(gradient[i].abs());
        }
    }
    norm
}

fn failure_result(
    error: ObjectiveError,
    point: Vec<f64>,
    value: f64,
    budget: &Budget,
) -> OptimizationResult {
    let (status, message) = match error {
        ObjectiveError::BudgetExhausted(BudgetStatus::EvaluationLimit) => (
            SolveStatus::EvaluationLimit,
            Some("evaluation budget exhausted".to_string()),
        ),
        ObjectiveError::BudgetExhausted(BudgetStatus::IterationLimit) => (
            SolveStatus::IterationLimit,
            Some("iteration budget exhausted".to_string()),
        ),
        ObjectiveError::NotFinite(value) => (
            SolveStatus::InvalidObjective,
            Some(format!("objective value is not finite: {value}")),
        ),
        ObjectiveError::DimensionMismatch { expected, actual } => (
            SolveStatus::HardError,
            Some(format!(
                "objective dimension mismatch: expected {expected}, got {actual}"
            )),
        ),
        ObjectiveError::Callback(message) => (SolveStatus::HardError, Some(message)),
    };
    let mut result = OptimizationResult::new(point, value);
    result.status = status;
    result.evaluations = budget.evaluations();
    result.iterations = budget.iterations();
    result.message = message;
    result
}

fn build_result(
    best_x: Vec<f64>,
    best_value: f64,
    iterations: usize,
    budget: &Budget,
    status: SolveStatus,
    history: Option<Vec<HistoryEntry>>,
    message: Option<&str>,
) -> OptimizationResult {
    OptimizationResult {
        best_x,
        best_value,
        iterations,
        evaluations: budget.evaluations(),
        status,
        history,
        max_constraint_violation: None,
        local_solve_count: None,
        message: message.map(str::to_string),
    }
}
