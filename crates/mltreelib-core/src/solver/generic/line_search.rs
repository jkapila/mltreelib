//! Strong Wolfe line search shared by the gradient-based methods.

use super::objective::{Objective, ObjectiveError, SmoothObjective};

/// Line search parameters.
#[derive(Clone, Copy, Debug)]
pub struct LineSearchOptions {
    /// Sufficient increase parameter.
    pub c1: f64,
    /// Curvature parameter.
    pub c2: f64,
    /// Maximum number of expansion or zoom steps.
    pub max_iter: usize,
    /// Largest step length.
    pub max_step: f64,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_iter: 30,
            max_step: 1e3,
        }
    }
}

/// Result of one line search.
#[derive(Clone, Debug)]
pub struct LineSearchResult {
    /// Accepted step length.
    pub step: f64,
    /// Objective value at the accepted point.
    pub value: f64,
    /// Gradient at the accepted point.
    pub gradient: Vec<f64>,
}

/// Evaluates one point along one direction.
fn evaluate(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    point: &[f64],
    direction: &[f64],
    step: f64,
) -> Result<(f64, Vec<f64>), ObjectiveError> {
    let candidate: Vec<f64> = (0..point.len())
        .map(|axis| point[axis] + step * direction[axis])
        .collect();
    let value = objective.value(&candidate)?;
    let gradient = smooth.gradient(&candidate)?;
    Ok((value, gradient))
}

/// Runs a strong Wolfe line search along one direction.
///
/// The search returns the step, the objective value, and the gradient at the
/// accepted point. It falls back to the steepest descent direction when the
/// supplied direction is not a descent direction.
pub fn strong_wolfe(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    point: &[f64],
    direction: &[f64],
    options: &LineSearchOptions,
) -> Result<LineSearchResult, ObjectiveError> {
    let base_value = objective.value(point)?;
    let base_gradient = smooth.gradient(point)?;
    let base_slope: f64 = base_gradient
        .iter()
        .zip(direction.iter())
        .map(|(g, d)| g * d)
        .sum();

    if base_slope >= 0.0 {
        let steepest: Vec<f64> = base_gradient.iter().map(|g| -g).collect();
        return strong_wolfe(objective, smooth, point, &steepest, options);
    }

    let mut previous_step = 0.0;
    let mut previous_value = base_value;
    let mut step = 1.0_f64;

    for _ in 0..options.max_iter {
        let (value, gradient) = evaluate(objective, smooth, point, direction, step)?;
        let slope: f64 = gradient
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();

        if value > base_value + options.c1 * step * base_slope
            || (step > previous_step && value >= previous_value)
        {
            return zoom(
                objective,
                smooth,
                point,
                direction,
                options,
                base_value,
                base_slope,
                previous_step,
                previous_value,
                step,
            );
        }
        if slope.abs() <= -options.c2 * base_slope {
            return Ok(LineSearchResult {
                step,
                value,
                gradient,
            });
        }
        if slope >= 0.0 {
            return zoom(
                objective,
                smooth,
                point,
                direction,
                options,
                base_value,
                base_slope,
                step,
                value,
                previous_step,
            );
        }

        previous_step = step;
        previous_value = value;
        step = (2.0 * step).min(options.max_step);
        if step >= options.max_step {
            break;
        }
    }

    let (value, gradient) = evaluate(objective, smooth, point, direction, previous_step)?;
    Ok(LineSearchResult {
        step: previous_step,
        value,
        gradient,
    })
}

/// Zooms into an interval that holds a point satisfying the Wolfe conditions.
#[allow(clippy::too_many_arguments)]
fn zoom(
    objective: &dyn Objective,
    smooth: &dyn SmoothObjective,
    point: &[f64],
    direction: &[f64],
    options: &LineSearchOptions,
    base_value: f64,
    base_slope: f64,
    low: f64,
    low_value: f64,
    high: f64,
) -> Result<LineSearchResult, ObjectiveError> {
    let mut low = low;
    let mut low_value = low_value;
    let mut high = high;
    let mut best: Option<LineSearchResult> = None;

    for _ in 0..options.max_iter {
        let step = 0.5 * (low + high);
        let (value, gradient) = evaluate(objective, smooth, point, direction, step)?;
        let slope: f64 = gradient
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();

        if value > base_value + options.c1 * step * base_slope || (step > low && value >= low_value)
        {
            high = step;
        } else {
            if slope.abs() <= -options.c2 * base_slope {
                return Ok(LineSearchResult {
                    step,
                    value,
                    gradient,
                });
            }
            if slope * (high - low) >= 0.0 {
                high = low;
            }
            best = Some(LineSearchResult {
                step,
                value,
                gradient,
            });
            low = step;
            low_value = value;
        }
        if (high - low).abs() <= 1e-14 {
            break;
        }
    }

    match best {
        Some(result) => Ok(result),
        None => {
            let (value, gradient) = evaluate(objective, smooth, point, direction, low)?;
            Ok(LineSearchResult {
                step: low,
                value,
                gradient,
            })
        }
    }
}
