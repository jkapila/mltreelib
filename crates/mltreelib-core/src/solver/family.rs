//! Generalized Linear Model families and link functions.

use crate::family::{std_norm_cdf, std_norm_inv_cdf, std_norm_pdf};

/// GLM exponential dispersion families and link functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlmFamily {
    Gaussian,
    Binomial,
    Probit,
    Poisson,
    Gamma,
    Tweedie { p: f64 },
    Tobit { threshold: f64 },
}

impl GlmFamily {
    /// Maps response mean $\mu$ to linear predictor $\eta = g(\mu)$.
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Tobit { .. } => mu,
            Self::Binomial => {
                let p = mu.clamp(1e-12, 1.0 - 1e-12);
                p.ln() - (1.0 - p).ln()
            }
            Self::Probit => {
                let p = mu.clamp(1e-12, 1.0 - 1e-12);
                std_norm_inv_cdf(p)
            }
            Self::Poisson | Self::Gamma | Self::Tweedie { .. } => mu.max(1e-12).ln(),
        }
    }

    /// Maps linear predictor $\eta$ to expected response $\mu = g^{-1}(\eta)$.
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Binomial => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let z = eta.exp();
                    z / (1.0 + z)
                }
            }
            Self::Probit => std_norm_cdf(eta),
            Self::Poisson | Self::Gamma | Self::Tweedie { .. } => {
                // Safeguard against exponential overflow
                eta.clamp(-30.0, 30.0).exp()
            }
            Self::Tobit { threshold } => {
                if eta < *threshold {
                    *threshold
                } else {
                    eta
                }
            }
        }
    }

    /// Tobit deviance uses the uncensored latent mean.
    pub(super) fn mean_for_loss(self, eta: f64) -> f64 {
        match self {
            Self::Tobit { .. } => eta,
            _ => self.inverse_link(eta),
        }
    }

    /// Variance function $V(\mu)$.
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Tobit { .. } => 1.0,
            Self::Binomial | Self::Probit => (mu * (1.0 - mu)).max(1e-12),
            Self::Poisson => mu.max(1e-12),
            Self::Gamma => (mu * mu).max(1e-12),
            Self::Tweedie { p } => mu.max(1e-12).powf(*p),
        }
    }

    /// Unit deviance $d(y, \mu)$ such that loss $L(y, \mu) = \frac{1}{2} w \cdot d(y, \mu)$.
    pub fn deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            Self::Gaussian => (y - mu) * (y - mu),
            Self::Binomial | Self::Probit => {
                let p = mu.clamp(1e-12, 1.0 - 1e-12);
                let mut dev = 0.0;
                if y > 0.0 {
                    dev += y * (y / p).ln();
                }
                if y < 1.0 {
                    dev += (1.0 - y) * ((1.0 - y) / (1.0 - p)).ln();
                }
                2.0 * dev
            }
            Self::Poisson => {
                let m = mu.max(1e-12);
                if y <= 0.0 {
                    2.0 * m
                } else {
                    2.0 * (y * (y / m).ln() - (y - m))
                }
            }
            Self::Gamma => {
                let m = mu.max(1e-12);
                let y_pos = y.max(1e-12);
                2.0 * (-((y_pos / m).ln()) + (y_pos - m) / m)
            }
            Self::Tweedie { p } => {
                let p_val = *p;
                let m = mu.max(1e-12);
                if (p_val - 1.0).abs() < 1e-6 {
                    return Self::Poisson.deviance(y, m);
                }
                if (p_val - 2.0).abs() < 1e-6 {
                    return Self::Gamma.deviance(y, m);
                }
                let t1 = y * m.powf(1.0 - p_val) / (1.0 - p_val);
                let t2 = m.powf(2.0 - p_val) / (2.0 - p_val);
                let y0 = if y > 0.0 {
                    let y1 = y.powf(2.0 - p_val) / ((1.0 - p_val) * (2.0 - p_val));
                    y1
                } else {
                    0.0
                };
                2.0 * (y0 - t1 + t2)
            }
            Self::Tobit { threshold } => {
                if y <= *threshold {
                    let delta = *threshold - mu;
                    let cdf = std_norm_cdf(delta).max(1e-12);
                    -2.0 * cdf.ln()
                } else {
                    (y - mu) * (y - mu)
                }
            }
        }
    }

    /// Evaluates negative log-likelihood derivative $\frac{\partial L}{\partial \eta}$ and Fisher weight $W$.
    pub fn gradient_and_weight(&self, y: f64, eta: f64, w: f64) -> (f64, f64) {
        let mu = self.inverse_link(eta);
        match self {
            Self::Gaussian => (w * (mu - y), w),
            Self::Binomial => {
                // For logit link, dmu/deta = mu*(1-mu) = V(mu)
                let grad = w * (mu - y);
                let weight = w * (mu * (1.0 - mu)).max(1e-12);
                (grad, weight)
            }
            Self::Probit => {
                let phi = std_norm_pdf(eta).max(1e-12);
                let var = (mu * (1.0 - mu)).max(1e-12);
                let grad = w * (mu - y) * phi / var;
                let weight = w * (phi * phi) / var;
                (grad, weight.max(1e-12))
            }
            Self::Poisson => {
                let grad = w * (mu - y);
                let weight = w * mu.max(1e-12);
                (grad, weight)
            }
            Self::Gamma => {
                // Log link: dmu/deta = mu, V(mu) = mu^2
                let grad = w * (mu - y) / mu.max(1e-12);
                let weight = w;
                (grad, weight)
            }
            Self::Tweedie { p } => {
                let weight = w * mu.max(1e-12).powf(2.0 - *p);
                let grad = weight * (mu - y) / mu.max(1e-12);
                (grad, weight.max(1e-12))
            }
            Self::Tobit { threshold } => {
                if y <= *threshold {
                    let delta = *threshold - eta;
                    let cdf = std_norm_cdf(delta).max(1e-12);
                    let pdf = std_norm_pdf(delta);
                    let grad = w * (pdf / cdf);
                    (grad, w)
                } else {
                    (w * (eta - y), w)
                }
            }
        }
    }

    /// Initial intercept from response mean.
    pub fn initial_intercept(&self, weighted_mean: f64) -> f64 {
        match self {
            Self::Gaussian | Self::Tobit { .. } => weighted_mean,
            Self::Binomial | Self::Probit => self.link(weighted_mean.clamp(0.01, 0.99)),
            Self::Poisson | Self::Gamma | Self::Tweedie { .. } => self.link(weighted_mean.max(1e-6)),
        }
    }
}
