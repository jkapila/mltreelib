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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// Parametric Accelerated Failure Time (AFT) model (Weibull, Lognormal, Log-logistic).
    Aft,
    /// Mixture cure rate model estimating long-term immune fraction pi(X) and uncured survival.
    Cure,
    /// Royston-Parmar flexible restricted cubic spline model on log cumulative hazard.
    Spline,
    /// Quantile regression with pinball loss (Spec 27 Phase 2).
    Quantile,
    /// Huber M-regression with robust loss (Spec 27 Phase 3).
    Huber,
    /// Expectile regression with asymmetric squared loss (Spec 27 Phase 4).
    Expectile,
    /// Overdispersed count regression with NB2 variance V(mu) = mu + alpha * mu^2 (Spec 27 Phase 7).
    NegativeBinomial,
    /// Beta regression for outcomes strictly inside (0, 1) with logit link (Spec 27 Phase 7).
    Beta,
}

impl LeafFamily {
    /// Numerically stable inverse link. Log rates are bounded to ±700 to
    /// prevent exponential overflow/underflow even when extrapolating.
    pub fn inverse_link(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Tobit | Self::Softmax | Self::Survival | Self::Heteroscedastic | Self::Aft | Self::Cure | Self::Spline | Self::Quantile | Self::Huber | Self::Expectile => eta,
            Self::Binomial | Self::Beta if eta >= 0.0 => 1.0 / (1.0 + (-eta).exp()),
            Self::Binomial | Self::Beta => {
                let e = eta.exp();
                e / (1.0 + e)
            }
            Self::Probit => std_norm_cdf(eta).clamp(1e-15, 1.0 - 1e-15),
            Self::Poisson | Self::Gamma | Self::Tweedie | Self::NegativeBinomial => eta.clamp(-700.0, 700.0).exp(),
        }
    }

    /// Unit deviance (Gaussian: squared error), including exact boundary
    /// means. Impossible events have infinite deviance, not a clipped loss.
    pub fn deviance(self, y: f64, mean: f64) -> f64 {
        self.deviance_with_params(y, mean, 1.5, 0.0)
    }

    /// Unit deviance with configurable Tweedie power parameter p and Tobit threshold.
    pub fn deviance_with_params(self, y: f64, mean: f64, tweedie_p: f64, tobit_threshold: f64) -> f64 {
        if y.is_nan() || mean.is_nan()
            || (self != Self::Gaussian && self != Self::Tobit && self != Self::Heteroscedastic && self != Self::Softmax && self != Self::Survival
                && self != Self::Aft && self != Self::Cure && self != Self::Spline && self != Self::Quantile && self != Self::Huber && self != Self::Expectile
                && (!y.is_finite() || !mean.is_finite() || mean < 0.0 || y < 0.0))
            || ((self == Self::Binomial || self == Self::Probit || self == Self::Beta) && (mean > 1.0 || y > 1.0))
            || (self == Self::Gamma && (y <= 0.0 || mean <= 0.0))
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian | Self::Heteroscedastic | Self::Survival | Self::Aft | Self::Cure | Self::Spline | Self::Quantile => (y - mean).powi(2),
            Self::Expectile => 2.0 * crate::solver::problems::expectile::expectile_loss(y - mean, 0.5),
            Self::Huber => 2.0 * crate::solver::problems::huber::huber_loss(y - mean, 1.345),
            Self::NegativeBinomial => crate::solver::problems::glm_extra::nb2_unit_deviance(y, mean, 0.1),
            Self::Beta => crate::solver::problems::glm_extra::beta_unit_deviance(y, mean),
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
                let p = tweedie_p;
                let term1 = if y == 0.0 { 0.0 } else { y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) };
                let term2 = if y == 0.0 { 0.0 } else { y * mean.powf(1.0 - p) / (1.0 - p) };
                let term3 = mean.powf(2.0 - p) / (2.0 - p);
                2.0 * (term1 - term2 + term3).max(0.0)
            }
            Self::Tobit => {
                if y <= tobit_threshold {
                    let p_cens = std_norm_cdf(tobit_threshold - mean).max(1e-15);
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
            || ((self == Self::Binomial || self == Self::Probit || self == Self::Beta) && (mean < 0.0 || mean > 1.0))
            || ((self == Self::Poisson || self == Self::Gamma || self == Self::Tweedie || self == Self::NegativeBinomial) && mean < 0.0)
        {
            return f64::NAN;
        }
        match self {
            Self::Gaussian | Self::Softmax | Self::Quantile | Self::Huber | Self::Expectile | Self::Beta => f64::NAN,
            Self::Heteroscedastic => std_norm_cdf(value - mean),
            Self::Survival | Self::Aft | Self::Cure | Self::Spline => (1.0 - mean).clamp(0.0, 1.0),
            _ if value < 0.0 && self != Self::Tobit => 0.0,
            Self::Binomial | Self::Probit if value < 1.0 => 1.0 - mean,
            Self::Binomial | Self::Probit => 1.0,
            Self::Poisson | Self::NegativeBinomial if mean == 0.0 || value == f64::INFINITY => 1.0,
            Self::Poisson | Self::NegativeBinomial => gamma_ur(value.floor() + 1.0, mean),
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
