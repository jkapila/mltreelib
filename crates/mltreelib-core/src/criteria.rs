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
            GainFn::FriedmanMse => GainFn::FriedmanMse,
            GainFn::AbsoluteError => GainFn::AbsoluteError,
            GainFn::Huber => GainFn::Huber,
            GainFn::Poisson => GainFn::Poisson,
            GainFn::Entropy => GainFn::Entropy,
            GainFn::Gini => GainFn::Gini,
            GainFn::GainRatio => GainFn::GainRatio,
            GainFn::Sdr => GainFn::Sdr,
            GainFn::Twoing => GainFn::Twoing,
            GainFn::Custom(f) => GainFn::Custom(f.clone()),
        }
    }
}

impl std::fmt::Debug for GainFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GainFn::Sse => write!(f, "Sse"),
            GainFn::FriedmanMse => write!(f, "FriedmanMse"),
            GainFn::AbsoluteError => write!(f, "AbsoluteError"),
            GainFn::Huber => write!(f, "Huber"),
            GainFn::Poisson => write!(f, "Poisson"),
            GainFn::Entropy => write!(f, "Entropy"),
            GainFn::Gini => write!(f, "Gini"),
            GainFn::GainRatio => write!(f, "GainRatio"),
            GainFn::Sdr => write!(f, "Sdr"),
            GainFn::Twoing => write!(f, "Twoing"),
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
            GainFn::Sse => {
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
            GainFn::Custom(f) => f.gain(left, right),
        }
    }

