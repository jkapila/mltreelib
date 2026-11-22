//! Per-dimension box bounds for the generic optimization seam.

/// Lower and upper limits for each parameter.
///
/// A bound is either finite or infinite. `lower[i] <= upper[i]` holds for every
/// dimension. A fully unbounded domain uses negative and positive infinity.
#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Bounds {
    /// Builds bounds from per-dimension lower and upper limits.
    ///
    /// Returns an error when the vectors differ in length, when a limit is
    /// `NaN`, when a lower limit is positive infinity, when an upper limit is
    /// negative infinity, or when a lower limit exceeds its upper limit.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, String> {
        if lower.len() != upper.len() {
            return Err(format!(
                "bounds length mismatch: lower has {} values, upper has {}",
                lower.len(),
                upper.len()
            ));
        }
        for (i, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
            if lo.is_nan() || hi.is_nan() {
                return Err(format!("bounds at dimension {i} contain NaN"));
            }
            if lo == f64::INFINITY {
                return Err(format!("lower bound at dimension {i} is infinite"));
            }
            if hi == f64::NEG_INFINITY {
                return Err(format!("upper bound at dimension {i} is infinite"));
            }
            if lo > hi {
                return Err(format!(
                    "lower bound {lo} exceeds upper bound {hi} at dimension {i}"
                ));
            }
        }
        Ok(Self { lower, upper })
    }

    /// Builds the same finite limit for every dimension.
    pub fn uniform(dim: usize, lower: f64, upper: f64) -> Result<Self, String> {
        Self::new(vec![lower; dim], vec![upper; dim])
    }

    /// Builds a domain without any finite limit.
    pub fn unbounded(dim: usize) -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY; dim],
            upper: vec![f64::INFINITY; dim],
        }
    }

    /// Number of dimensions.
    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    /// Lower limits.
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Upper limits.
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Reports whether every limit is finite.
    pub fn is_bounded(&self) -> bool {
        self.lower.iter().all(|v| v.is_finite()) && self.upper.iter().all(|v| v.is_finite())
    }

    /// Reports whether any limit is finite.
    pub fn has_finite_limit(&self) -> bool {
        self.lower.iter().any(|v| v.is_finite()) || self.upper.iter().any(|v| v.is_finite())
    }

    /// Width of one dimension. Infinite when either limit is infinite.
    pub fn width(&self, dim: usize) -> f64 {
        self.upper[dim] - self.lower[dim]
    }

    /// Reports whether a point lies inside the box and has the right length.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dim()
            && x.iter()
                .zip(self.lower.iter().zip(self.upper.iter()))
                .all(|(&v, (&lo, &hi))| v.is_finite() && v >= lo && v <= hi)
    }

    /// Clamps a point into the box. Non-finite values stay unchanged.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if !v.is_finite() {
                    return v;
                }
                let lo = self.lower[i];
                let hi = self.upper[i];
                v.max(lo).min(hi)
            })
            .collect()
    }
}
