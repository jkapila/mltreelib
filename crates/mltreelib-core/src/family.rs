//! Response families for model-tree leaves. Linear coefficients live on the
//! link scale; leaf predictions and distribution queries use response means.

use statrs::function::gamma::gamma_ur;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafFamily {
    #[default]
    Gaussian,
    /// Bernoulli observations with a logit link.
    Binomial,
    /// Nonnegative counts/rates with a log link.
    Poisson,
}

impl LeafFamily {
    /// Numerically stable inverse link. Log rates are bounded to ±700 to
    /// prevent exponential overflow/underflow even when extrapolating.
    pub fn inverse_link(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Binomial if eta >= 0.0 => 1.0 / (1.0 + (-eta).exp()),
            Self::Binomial => {
                let e = eta.exp();
                e / (1.0 + e)
            }
            Self::Poisson => eta.clamp(-700.0, 700.0).exp(),
        }
    }

    /// Unit deviance (Gaussian: squared error), including exact boundary
    /// means. Impossible events have infinite deviance, not a clipped loss.
    pub fn deviance(self, y: f64, mean: f64) -> f64 {
        if y.is_nan() || mean.is_nan()
            || (self != Self::Gaussian
                && (!y.is_finite() || !mean.is_finite() || mean < 0.0 || y < 0.0))
            || (self == Self::Binomial && (mean > 1.0 || y > 1.0))
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian => (y - mean).powi(2),
            Self::Binomial => {
                let success = if y == 0.0 { 0.0 } else { y * (y / mean).ln() };
                let failure = if y == 1.0 {
                    0.0
                } else {
                    (1.0 - y) * ((1.0 - y).ln() - (-mean).ln_1p())
                };
                2.0 * (success + failure)
            }
            Self::Poisson if y == 0.0 => 2.0 * mean,
            Self::Poisson => 2.0 * (y * (y.ln() - mean.ln()) - (y - mean)).max(0.0),
        }
    }

    /// Conditional response CDF at a real-valued threshold. Gaussian
    /// requires a variance model and is intentionally unsupported (NaN);
    /// the tree's public query reports that case as an error.
    pub fn cdf(self, mean: f64, value: f64) -> f64 {
        if value.is_nan() || mean.is_nan() || mean < 0.0
            || (self == Self::Binomial && mean > 1.0)
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian => f64::NAN,
            _ if value < 0.0 => 0.0,
            Self::Binomial if value < 1.0 => 1.0 - mean,
            Self::Binomial => 1.0,
            Self::Poisson if mean == 0.0 || value == f64::INFINITY => 1.0,
            Self::Poisson => gamma_ur(value.floor() + 1.0, mean),
        }
    }
}
