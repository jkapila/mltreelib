//! Response families for model-tree leaves. Linear coefficients live on the
//! link scale; leaf predictions and distribution queries use response means.

use statrs::function::gamma::gamma_ur;

pub fn std_norm_cdf(z: f64) -> f64 {
    let clamped_z = z.clamp(-35.0, 35.0);
    0.5 * (1.0 + statrs::function::erf::erf(clamped_z / std::f64::consts::SQRT_2))
}

pub fn std_norm_pdf(z: f64) -> f64 {
    let clamped_z = z.clamp(-35.0, 35.0);
    (-0.5 * clamped_z * clamped_z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

pub fn std_norm_inv_cdf(p: f64) -> f64 {
    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
    std::f64::consts::SQRT_2 * statrs::function::erf::erf_inv(2.0 * p_clamped - 1.0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafFamily {
    #[default]
    Gaussian,
    /// Bernoulli observations with a logit link.
    Binomial,
    /// Nonnegative counts/rates with a log link.
    Poisson,
    /// Bernoulli observations with standard normal probit link Φ.
    Probit,
    /// Positive continuous observations with log link (CV constant).
    Gamma,
    /// Zero-inflated positive continuous observations with compound Poisson-Gamma deviance (p=1.5).
    Tweedie,
    /// Censored continuous observations with lower bound (default 0).
    Tobit,
    /// Multinomial categorical observations with softmax link.
    Softmax,
    /// Right-censored time-to-event pairs (T, E) with Nelson-Aalen / Kaplan-Meier hazard.
    Survival,
    /// Continuous observations with dual location-scale (mu, sigma) Gaussian NLL.
    Heteroscedastic,
}

impl LeafFamily {
    /// Numerically stable inverse link. Log rates are bounded to ±700 to
    /// prevent exponential overflow/underflow even when extrapolating.
    pub fn inverse_link(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Tobit | Self::Softmax | Self::Survival | Self::Heteroscedastic => eta,
            Self::Binomial if eta >= 0.0 => 1.0 / (1.0 + (-eta).exp()),
            Self::Binomial => {
                let e = eta.exp();
                e / (1.0 + e)
            }
            Self::Probit => std_norm_cdf(eta).clamp(1e-15, 1.0 - 1e-15),
            Self::Poisson | Self::Gamma | Self::Tweedie => eta.clamp(-700.0, 700.0).exp(),
        }
    }

    /// Unit deviance (Gaussian: squared error), including exact boundary
    /// means. Impossible events have infinite deviance, not a clipped loss.
    pub fn deviance(self, y: f64, mean: f64) -> f64 {
        if y.is_nan() || mean.is_nan()
            || (self != Self::Gaussian && self != Self::Tobit && self != Self::Heteroscedastic && self != Self::Softmax && self != Self::Survival
                && (!y.is_finite() || !mean.is_finite() || mean < 0.0 || y < 0.0))
            || ((self == Self::Binomial || self == Self::Probit) && (mean > 1.0 || y > 1.0))
            || (self == Self::Gamma && (y <= 0.0 || mean <= 0.0))
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian | Self::Heteroscedastic | Self::Survival => (y - mean).powi(2),
            Self::Softmax => {
                if y < 0.0 || mean <= 0.0 {
                    f64::NAN
                } else {
                    -2.0 * mean.max(1e-15).ln()
                }
            }
            Self::Binomial | Self::Probit => {
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
            Self::Gamma => {
                2.0 * (-(y / mean).ln() + (y - mean) / mean).max(0.0)
            }
            Self::Tweedie => {
                let p = 1.5;
                let term1 = if y == 0.0 { 0.0 } else { y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) };
                let term2 = if y == 0.0 { 0.0 } else { y * mean.powf(1.0 - p) / (1.0 - p) };
                let term3 = mean.powf(2.0 - p) / (2.0 - p);
                2.0 * (term1 - term2 + term3).max(0.0)
            }
            Self::Tobit => {
                if y <= 0.0 {
                    let p_cens = std_norm_cdf(-mean).max(1e-15);
                    -2.0 * p_cens.ln()
                } else {
                    (y - mean).powi(2)
                }
            }
        }
    }

    /// Conditional response CDF at a real-valued threshold. Gaussian
    /// requires a variance model and is intentionally unsupported (NaN);
    /// the tree's public query reports that case as an error.
    pub fn cdf(self, mean: f64, value: f64) -> f64 {
        if value.is_nan() || mean.is_nan()
            || ((self == Self::Binomial || self == Self::Probit) && (mean < 0.0 || mean > 1.0))
            || ((self == Self::Poisson || self == Self::Gamma || self == Self::Tweedie) && mean < 0.0)
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian | Self::Softmax => f64::NAN,
            Self::Heteroscedastic => std_norm_cdf(value - mean),
            Self::Survival => (1.0 - mean).clamp(0.0, 1.0),
            _ if value < 0.0 && self != Self::Tobit => 0.0,
            Self::Binomial | Self::Probit if value < 1.0 => 1.0 - mean,
            Self::Binomial | Self::Probit => 1.0,
            Self::Poisson if mean == 0.0 || value == f64::INFINITY => 1.0,
            Self::Poisson => gamma_ur(value.floor() + 1.0, mean),
            Self::Gamma => 1.0 - (-value / mean.max(1e-12)).exp(),
            Self::Tweedie => {
                if value == 0.0 {
                    (-mean.powf(0.5) / 0.5).exp()
                } else {
                    1.0 - (-value / mean.max(1e-12)).exp()
                }
            }
            Self::Tobit => std_norm_cdf(value - mean),
        }
    }
}
