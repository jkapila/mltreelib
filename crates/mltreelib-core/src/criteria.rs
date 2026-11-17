//! Split-quality criteria (the `gain` knob): the [`GainFn`] family, the
//! weighted quality functions behind it, and the deterministic gain
//! comparison the split search uses.

use crate::stats::{
    class_entropy_w, class_gini_w, class_total_variation, huber_cost, l1_cost, poisson_deviance,
    w_sd,
};
/// Near-tie test: candidates within one part per trillion of the standing
/// best do not replace it (first-found wins). Mirrors rpart's rounding guard
/// (`improve > iscale * 1e-10` in `bsplit.c`) — ulp-level arithmetic noise
/// must not flip equally-good splits, or memberships drift between runs.
pub(crate) fn beats(gain: f64, best: f64) -> bool {
    gain > best * (1.0 + 1e-12)
}
/// Split-quality criterion. Built-ins run the O(1) stats scan where the
/// math allows (`Sse`, `FriedmanMse`); the rest see lazy (y, w) slices per
/// candidate. All criteria are sample-weight aware.
pub enum GainFn {
    /// Variance reduction (weighted sum of squares). The regression default.
    Sse,
    /// Fitted family deviance reduction (Gaussian: squared-error reduction).
    /// Unlike entropy/Poisson constant-partition criteria, linear leaves
    /// are refitted on each candidate side.
    Deviance,
    /// Friedman's MSE proxy: `(w_l·w_r/w)·(mean_l − mean_r)^2` — the
    /// squared mean gap weighted by the child masses (sklearn's
    /// `friedman_mse`). Regression.
    FriedmanMse,
    /// Absolute-error reduction (weighted L1 around each side's median) —
    /// robust to outliers where SSE chases them. Regression.
    AbsoluteError,
    /// Huber-loss reduction (δ = 1.345·sd of the node): quadratic near the
    /// median, linear in the tails. Regression.
    Huber,
    /// Poisson deviance reduction — for counts/rate targets (values must be
    /// >= 0). Regression.
    Poisson,
    /// Information gain: weighted entropy reduction over y-as-class-labels
    /// (distinct y values are the classes).
    Entropy,
    /// Gini impurity reduction over y-as-class-labels.
    Gini,
    /// C4.5 gain ratio: information gain divided by the split information
    /// of the two children (penalizes lopsided splits).
    GainRatio,
    /// Model-tree criterion (Wang/Quinlan M5): weighted standard-deviation
    /// reduction `sd(parent) - Σ (w_k/w)·sd(child_k)`. Regression.
    Sdr,
    /// Breiman twoing rule: `2·P_L·P_R·(Σ_j |p_L(j) - p_R(j)|)^2` over
    /// class frequencies — favors balanced splits with contrasting sides.
    Twoing,
    /// Causal / uplift splitting criterion (Athey & Imbens 2016):
    /// maximizes treatment effect heterogeneity between children:
    /// `(w_l·w_r / (w_l + w_r)) · (tau_l - tau_r)^2`.
    Causal,
    /// Log-rank test statistic for survival trees (Leblanc & Crowley 1992):
    /// compares observed vs expected survival events between partitions.
    LogRank,
    /// Tweedie unit quasi-deviance reduction for 1 < p < 2 zero-inflated continuous data.
    Tweedie { p: f64 },
    /// Dual-parameter Gaussian negative log-likelihood deviance reduction:
    /// detects variance breaks even when means are identical.
    Heteroscedastic,
    /// Censored continuous regression hurdle gain combining binary classification
    /// of non-zero/censored indicator with Gaussian deviance on positive slice.
    Tobit { threshold: f64 },
    /// Coefficient of variation reduction: CV(P) - [w_l/w * CV(L) + w_r/w * CV(R)].
    Cv,
    /// Survival martingale residual variance reduction: SSE reduction on M_i.
    Martingale,
    /// User criterion. Receives lazy `(y, w)` slices of the two sides per
    /// candidate split and returns the split quality (higher = better).
    /// Pinball/quantile losses and anything exotic live here.
    Custom(std::sync::Arc<dyn CustomGain>),
}

/// User-supplied split criterion over lazy (y, w) slices.
pub trait CustomGain: Send + Sync {
    /// Quality of splitting `left` / `right` (higher = better). Each slice
    /// holds the side's `(target, weight)` rows.
    fn gain(&self, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64;
}


impl Clone for GainFn {
    fn clone(&self) -> Self {
        match self {
            GainFn::Sse => GainFn::Sse,
            GainFn::Deviance => GainFn::Deviance,
            GainFn::FriedmanMse => GainFn::FriedmanMse,
            GainFn::AbsoluteError => GainFn::AbsoluteError,
            GainFn::Huber => GainFn::Huber,
            GainFn::Poisson => GainFn::Poisson,
            GainFn::Entropy => GainFn::Entropy,
            GainFn::Gini => GainFn::Gini,
            GainFn::GainRatio => GainFn::GainRatio,
            GainFn::Sdr => GainFn::Sdr,
            GainFn::Twoing => GainFn::Twoing,
            GainFn::Causal => GainFn::Causal,
            GainFn::LogRank => GainFn::LogRank,
            GainFn::Tweedie { p } => GainFn::Tweedie { p: *p },
            GainFn::Heteroscedastic => GainFn::Heteroscedastic,
            GainFn::Tobit { threshold } => GainFn::Tobit { threshold: *threshold },
            GainFn::Cv => GainFn::Cv,
            GainFn::Martingale => GainFn::Martingale,
            GainFn::Custom(f) => GainFn::Custom(f.clone()),
        }
    }
}

impl std::fmt::Debug for GainFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GainFn::Sse => write!(f, "Sse"),
            GainFn::Deviance => write!(f, "Deviance"),
            GainFn::FriedmanMse => write!(f, "FriedmanMse"),
            GainFn::AbsoluteError => write!(f, "AbsoluteError"),
            GainFn::Huber => write!(f, "Huber"),
            GainFn::Poisson => write!(f, "Poisson"),
            GainFn::Entropy => write!(f, "Entropy"),
            GainFn::Gini => write!(f, "Gini"),
            GainFn::GainRatio => write!(f, "GainRatio"),
            GainFn::Sdr => write!(f, "Sdr"),
            GainFn::Twoing => write!(f, "Twoing"),
            GainFn::Causal => write!(f, "Causal"),
            GainFn::LogRank => write!(f, "LogRank"),
            GainFn::Tweedie { p } => write!(f, "Tweedie(p={})", p),
            GainFn::Heteroscedastic => write!(f, "Heteroscedastic"),
            GainFn::Tobit { threshold } => write!(f, "Tobit(threshold={})", threshold),
            GainFn::Cv => write!(f, "Cv"),
            GainFn::Martingale => write!(f, "Martingale"),
            GainFn::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}


    /// Split quality of a partition under a non-SSE criterion (higher =
    /// better). All criteria are sample-weight aware; a custom criterion
    /// gets the lazy `(y, w)` slices of the two sides.
    pub(crate) fn gain_of(gfn: &GainFn, left: &[(f64, f64)], right: &[(f64, f64)]) -> f64 {
        let wsum = |p: &[(f64, f64)]| p.iter().map(|&(_, w)| w).sum::<f64>();
        let mut all: Vec<(f64, f64)> = Vec::with_capacity(left.len() + right.len());
        all.extend_from_slice(left);
        all.extend_from_slice(right);
        let (wl, wr) = (wsum(left), wsum(right));
        let w = wl + wr;
        if w <= 0.0 {
            return 0.0;
        }
        match gfn {
            GainFn::Sse | GainFn::Deviance => {
                let sse = |p: &[(f64, f64)]| {
                    let sw = wsum(p);
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sy: f64 = p.iter().map(|&(y, wt)| wt * y).sum();
                    let sy2: f64 = p.iter().map(|&(y, wt)| wt * y * y).sum();
                    sy2 - sy * sy / sw
                };
                sse(&all) - sse(left) - sse(right)
            }
            GainFn::FriedmanMse => {
                let mean = |p: &[(f64, f64)]| {
                    let sw = wsum(p);
                    if sw <= 0.0 {
                        0.0
                    } else {
                        p.iter().map(|&(y, wt)| wt * y).sum::<f64>() / sw
                    }
                };
                (wl * wr / w) * (mean(left) - mean(right)).powi(2)
            }
            GainFn::AbsoluteError => {
                l1_cost(&all) - l1_cost(left) - l1_cost(right)
            }
            GainFn::Huber => {
                let delta = 1.345 * w_sd(&all);
                huber_cost(&all, delta) - huber_cost(left, delta) - huber_cost(right, delta)
            }
            GainFn::Poisson => poisson_deviance(&all) - poisson_deviance(left) - poisson_deviance(right),
            GainFn::Entropy => {
                class_entropy_w(&all) - (wl / w) * class_entropy_w(left) - (wr / w) * class_entropy_w(right)
            }
            GainFn::Gini => {
                class_gini_w(&all) - (wl / w) * class_gini_w(left) - (wr / w) * class_gini_w(right)
            }
            GainFn::GainRatio => {
                let ig =
                    class_entropy_w(&all) - (wl / w) * class_entropy_w(left) - (wr / w) * class_entropy_w(right);
                let info = |x: f64| if x > 0.0 { -x * x.log2() } else { 0.0 };
                let split_info = info(wl / w) + info(wr / w);
                if split_info <= 0.0 {
                    0.0
                } else {
                    ig / split_info
                }
            }
            GainFn::Sdr => w_sd(&all) - (wl / w) * w_sd(left) - (wr / w) * w_sd(right),
            GainFn::Twoing => {
                let tv = class_total_variation(left, right);
                2.0 * (wl / w) * (wr / w) * tv * tv
            }
            GainFn::Causal | GainFn::LogRank => {
                // Specialized fast-scan criteria: auxiliary channels (treatment/event)
                // are scored directly in scan_column. Lazy slice fallback defaults to Friedman MSE.
                let mean = |p: &[(f64, f64)]| {
                    let sw = wsum(p);
                    if sw <= 0.0 { 0.0 } else { p.iter().map(|&(y, wt)| wt * y).sum::<f64>() / sw }
                };
                (wl * wr / w) * (mean(left) - mean(right)).powi(2)
            }
            GainFn::Tweedie { p } => {
                let tweedie_dev = |slice: &[(f64, f64)]| -> f64 {
                    let sw: f64 = slice.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sy: f64 = slice.iter().map(|&(y, w)| w * y).sum();
                    let mu = (sy / sw).max(1e-9);
                    let p_val = *p;
                    let inv_1_p = 1.0 / (1.0 - p_val);
                    let inv_2_p = 1.0 / (2.0 - p_val);
                    let mu_1_p = mu.powf(1.0 - p_val);
                    let mu_2_p = mu.powf(2.0 - p_val);

                    slice.iter().map(|&(y, w)| {
                        let y_clamp = y.max(0.0);
                        let term1 = if y_clamp > 0.0 {
                            y_clamp.powf(2.0 - p_val) * inv_1_p * inv_2_p
                        } else {
                            0.0
                        };
                        let term2 = y_clamp * mu_1_p * inv_1_p;
                        let term3 = mu_2_p * inv_2_p;
                        let d = 2.0 * (term1 - term2 + term3);
                        w * d.max(0.0)
                    }).sum()
                };
                (tweedie_dev(&all) - tweedie_dev(left) - tweedie_dev(right)).max(0.0)
            }
            GainFn::Heteroscedastic => {
                let nll_dev = |slice: &[(f64, f64)]| -> f64 {
                    let sw: f64 = slice.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let mean = slice.iter().map(|&(y, w)| w * y).sum::<f64>() / sw;
                    let var = slice.iter().map(|&(y, w)| w * (y - mean) * (y - mean)).sum::<f64>() / sw;
                    sw * (var + 1e-9).ln()
                };
                (nll_dev(&all) - nll_dev(left) - nll_dev(right)).max(0.0)
            }
            GainFn::Tobit { threshold } => {
                let th = *threshold;
                let bin_dev = |slice: &[(f64, f64)]| -> f64 {
                    let sw: f64 = slice.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sw_pos: f64 = slice.iter().filter(|&&(y, _)| y > th).map(|&(_, w)| w).sum();
                    let p_pos = (sw_pos / sw).clamp(1e-12, 1.0 - 1e-12);
                    -2.0 * (sw_pos * p_pos.ln() + (sw - sw_pos) * (1.0 - p_pos).ln())
                };
                let gain_bin = bin_dev(&all) - bin_dev(left) - bin_dev(right);

                let sse_pos = |slice: &[(f64, f64)]| -> f64 {
                    let pos: Vec<(f64, f64)> = slice.iter().filter(|&&(y, _)| y > th).copied().collect();
                    let sw: f64 = pos.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sy: f64 = pos.iter().map(|&(y, w)| w * y).sum();
                    let sy2: f64 = pos.iter().map(|&(y, w)| w * y * y).sum();
                    sy2 - sy * sy / sw
                };
                let gain_sse = sse_pos(&all) - sse_pos(left) - sse_pos(right);
                gain_bin.max(0.0) + gain_sse.max(0.0)
            }
            GainFn::Cv => {
                let cv = |slice: &[(f64, f64)]| -> f64 {
                    let sw: f64 = slice.iter().map(|&(_, w)| w).sum();
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let mean = slice.iter().map(|&(y, w)| w * y).sum::<f64>() / sw;
                    let var = slice.iter().map(|&(y, w)| w * (y - mean) * (y - mean)).sum::<f64>() / sw;
                    let sd = var.sqrt();
                    sd / (mean.abs() + 1e-6)
                };
                cv(&all) - ((wl / w) * cv(left) + (wr / w) * cv(right))
            }
            GainFn::Martingale => {
                let sse = |p: &[(f64, f64)]| {
                    let sw = wsum(p);
                    if sw <= 0.0 {
                        return 0.0;
                    }
                    let sy: f64 = p.iter().map(|&(y, wt)| wt * y).sum();
                    let sy2: f64 = p.iter().map(|&(y, wt)| wt * y * y).sum();
                    sy2 - sy * sy / sw
                };
                sse(&all) - sse(left) - sse(right)
            }
            GainFn::Custom(f) => f.gain(left, right),
        }
    }

