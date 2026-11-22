//! Central-difference gradient fallback for smooth methods.

use super::bounds::Bounds;
use super::objective::{Objective, ObjectiveError, SmoothObjective};

/// Objective that supplies a central-difference gradient.
///
/// The perturbation calls the wrapped objective, so the shared budget counts
/// them like every other objective call.
pub struct FiniteDifferenceObjective<'a> {
    inner: &'a dyn Objective,
    step: f64,
}

impl<'a> FiniteDifferenceObjective<'a> {
    /// Wraps one objective with a central-difference step size.
    pub fn new(inner: &'a dyn Objective, step: f64) -> Self {
        Self { inner, step }
    }
}

impl Objective for FiniteDifferenceObjective<'_> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds {
        self.inner.bounds()
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.inner.value(x)
    }
}

impl SmoothObjective for FiniteDifferenceObjective<'_> {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        let mut gradient = vec![0.0; x.len()];
        for (index, value) in x.iter().enumerate() {
            let step = self.step * value.abs().max(1.0);
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            plus[index] += step;
            minus[index] -= step;
            let plus_value = self.inner.value(&plus)?;
            let minus_value = self.inner.value(&minus)?;
            gradient[index] = (plus_value - minus_value) / (2.0 * step);
        }
        Ok(gradient)
    }
}
