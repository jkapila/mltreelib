//! Split-quality criteria (the `gain` knob): the [`GainFn`] family, the
//! weighted quality functions behind it, and the deterministic gain
//! comparison the split search uses.

use std::collections::{HashMap, HashSet};

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
/// Kernel kind for Maximum Mean Discrepancy (MMD) splitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelKind {
    /// Radial Basis Function (Gaussian) kernel: exp(-||x - y||^2 / (2 * sigma^2)).
    Rbf,
    /// Linear kernel: <x, y>.
    Linear,
    /// Polynomial kernel: (<x, y> + c)^degree.
    Polynomial { degree: usize, c: f64 },
}

/// Weighting schemes for two-sample log-rank tests (Spec 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogRankWeight {
    /// Standard Mantel-Haenszel / Peto log-rank test (equal weights w_j = 1.0).
    Standard,
    /// Tarone-Ware weighted log-rank (w_j = sqrt(n_j)): balances early vs late divergence.
    TaroneWare,
    /// Peto-Prentice weighted log-rank (w_j = S(t_j)): emphasizes early divergence.
    PetoPrentice,
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
    /// Weighted log-rank tests (Tarone-Ware or Peto-Prentice).
    LogRankWeighted { scheme: LogRankWeight },
    /// Gray's test for cause-specific cumulative incidence in competing risks (Gray 1988).
    GrayTest { cause: usize },
    /// Survival Brier score / proper separable scoring rule with IPCW (AISTATS 2025 SurvivalBoost).
    SurvivalBrier,
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
    /// Model-based recursive partitioning (MOB / Zeileis et al. 2008):
    /// score fluctuation parameter instability splitting.
    Mob,
    /// Conditional inference framework (CTree / Hothorn, Hornik & Zeileis 2006):
    /// permutation rank hypothesis test independence splitting.
    Ctree,
    /// Energy distance metric splitting (Székely & Rizzo 2004, 2013):
    /// maximizes two-sample energy distance between child target distributions:
    /// (w_l * w_r / (w_l + w_r)) * [2 E||Y_l - Y_r|| - E||Y_l - Y_l'|| - E||Y_r - Y_r'||].
    /// Supports scalar targets and high-dimensional vector embeddings Y in R^D.
    EnergyDistance,
    /// Maximum Mean Discrepancy metric splitting (MMD / Gretton et al. 2012):
    /// maximizes two-sample RKHS distance between child target distributions:
    /// (w_l * w_r / (w_l + w_r)) * [E[k(Y_l, Y_l')] + E[k(Y_r, Y_r')] - 2 E[k(Y_l, Y_r)]].
    /// Distinguishes complex multi-modal and covariance differences with characteristic kernels.
    Mmd { kernel: KernelKind, bandwidth: f64 },
    /// Wasserstein-2 optimal transport metric splitting (Villani 2003, Agueh & Carlier 2011):
    /// maximizes Wasserstein-2 Fréchet dispersion reduction across distribution targets Y in R^D
    /// or scalar distributions: (w_l * w_r / (w_l + w_r)) * W_2^2(barycenter_l, barycenter_r).
    Wasserstein2,
    /// Fairness-constrained splitting (demographic parity penalty):
    /// penalizes base split gain by disparity divergence across protected attribute S:
    /// BaseGain(L, R) - lambda * scale * |P(S=1 | L) - P(S=1 | R)|.
    FairConstrained {
        base_gain: Box<GainFn>,
        lambda: f64,
        protected_col: usize,
    },
    /// Hellinger distance splitting (Cieslak & Chawla 2008):
    /// Skew-invariant affinity divergence for extreme class imbalance:
    /// (w_l * w_r / (w_l + w_r)) * sum_{k=1}^C (sqrt(p_{L, k}) - sqrt(p_{R, k}))^2.
    Hellinger,
    /// Maximally selected rank statistics splitting (Hothorn & Lausen 2003):
    /// Maximizes standardized two-sample linear rank statistic T^2(s) = S_L^2 / Var(S_L)
    /// across candidate cutpoints with exact invariance to monotonic transformations.
    MaxStat,
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
            GainFn::LogRankWeighted { scheme } => GainFn::LogRankWeighted { scheme: *scheme },
            GainFn::GrayTest { cause } => GainFn::GrayTest { cause: *cause },
            GainFn::SurvivalBrier => GainFn::SurvivalBrier,
            GainFn::Tweedie { p } => GainFn::Tweedie { p: *p },
            GainFn::Heteroscedastic => GainFn::Heteroscedastic,
            GainFn::Tobit { threshold } => GainFn::Tobit { threshold: *threshold },
            GainFn::Cv => GainFn::Cv,
            GainFn::Martingale => GainFn::Martingale,
            GainFn::Mob => GainFn::Mob,
            GainFn::Ctree => GainFn::Ctree,
            GainFn::EnergyDistance => GainFn::EnergyDistance,
            GainFn::Mmd { kernel, bandwidth } => GainFn::Mmd { kernel: *kernel, bandwidth: *bandwidth },
            GainFn::Wasserstein2 => GainFn::Wasserstein2,
            GainFn::FairConstrained { base_gain, lambda, protected_col } => GainFn::FairConstrained {
                base_gain: base_gain.clone(),
                lambda: *lambda,
                protected_col: *protected_col,
            },
            GainFn::Hellinger => GainFn::Hellinger,
            GainFn::MaxStat => GainFn::MaxStat,
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
            GainFn::LogRankWeighted { scheme } => write!(f, "LogRankWeighted({:?})", scheme),
            GainFn::GrayTest { cause } => write!(f, "GrayTest(cause={})", cause),
            GainFn::SurvivalBrier => write!(f, "SurvivalBrier"),
            GainFn::Tweedie { p } => write!(f, "Tweedie(p={})", p),
            GainFn::Heteroscedastic => write!(f, "Heteroscedastic"),
            GainFn::Tobit { threshold } => write!(f, "Tobit(threshold={})", threshold),
            GainFn::Cv => write!(f, "Cv"),
            GainFn::Martingale => write!(f, "Martingale"),
            GainFn::Mob => write!(f, "Mob"),
            GainFn::Ctree => write!(f, "Ctree"),
            GainFn::EnergyDistance => write!(f, "EnergyDistance"),
            GainFn::Mmd { kernel, bandwidth } => write!(f, "Mmd({:?}, bw={})", kernel, bandwidth),
            GainFn::Wasserstein2 => write!(f, "Wasserstein2"),
            GainFn::FairConstrained { base_gain, lambda, protected_col } => {
                write!(f, "FairConstrained({:?}, lambda={}, prot_col={})", base_gain, lambda, protected_col)
            }
            GainFn::Hellinger => write!(f, "Hellinger"),
            GainFn::MaxStat => write!(f, "MaxStat"),
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
            GainFn::Causal
            | GainFn::LogRank
            | GainFn::LogRankWeighted { .. }
            | GainFn::GrayTest { .. }
            | GainFn::SurvivalBrier
            | GainFn::Mob
            | GainFn::Ctree => {
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
            GainFn::EnergyDistance => {
                if wl <= 0.0 || wr <= 0.0 {
                    return 0.0;
                }
                let mut d_ll = 0.0;
                for &(y1, w1) in left {
                    for &(y2, w2) in left {
                        d_ll += w1 * w2 * (y1 - y2).abs();
                    }
                }
                d_ll /= wl * wl;

                let mut d_rr = 0.0;
                for &(y1, w1) in right {
                    for &(y2, w2) in right {
                        d_rr += w1 * w2 * (y1 - y2).abs();
                    }
                }
                d_rr /= wr * wr;

                let mut d_lr = 0.0;
                for &(y1, w1) in left {
                    for &(y2, w2) in right {
                        d_lr += w1 * w2 * (y1 - y2).abs();
                    }
                }
                d_lr /= wl * wr;

                let energy = 2.0 * d_lr - d_ll - d_rr;
                ((wl * wr / (wl + wr)) * energy).max(0.0)
            }
            GainFn::Mmd { kernel, bandwidth } => {
                if wl <= 0.0 || wr <= 0.0 {
                    return 0.0;
                }
                let bw = if *bandwidth > 0.0 { *bandwidth } else { 1.0 };
                let keval = |a: f64, b: f64| match kernel {
                    KernelKind::Rbf => (-(a - b).powi(2) / (2.0 * bw * bw)).exp(),
                    KernelKind::Linear => a * b,
                    KernelKind::Polynomial { degree, c } => (a * b + c).powi(*degree as i32),
                };

                let mut k_ll = 0.0;
                for &(y1, w1) in left {
                    for &(y2, w2) in left {
                        k_ll += w1 * w2 * keval(y1, y2);
                    }
                }
                k_ll /= wl * wl;

                let mut k_rr = 0.0;
                for &(y1, w1) in right {
                    for &(y2, w2) in right {
                        k_rr += w1 * w2 * keval(y1, y2);
                    }
                }
                k_rr /= wr * wr;

                let mut k_lr = 0.0;
                for &(y1, w1) in left {
                    for &(y2, w2) in right {
                        k_lr += w1 * w2 * keval(y1, y2);
                    }
                }
                k_lr /= wl * wr;

                let mmd2 = k_ll + k_rr - 2.0 * k_lr;
                ((wl * wr / (wl + wr)) * mmd2).max(0.0)
            }
            GainFn::Wasserstein2 => {
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
            GainFn::FairConstrained { base_gain, .. } => {
                gain_of(base_gain, left, right)
            }
            GainFn::Hellinger => {
                let mut map_l: HashMap<i64, f64> = HashMap::new();
                let mut map_r: HashMap<i64, f64> = HashMap::new();
                for &(y, wt) in left {
                    *map_l.entry(y.round() as i64).or_insert(0.0) += wt;
                }
                for &(y, wt) in right {
                    *map_r.entry(y.round() as i64).or_insert(0.0) += wt;
                }
                let mut h_sq = 0.0;
                let mut all_keys: HashSet<i64> = map_l.keys().copied().collect();
                all_keys.extend(map_r.keys().copied());
                for k in all_keys {
                    let p_l = map_l.get(&k).copied().unwrap_or(0.0) / wl;
                    let p_r = map_r.get(&k).copied().unwrap_or(0.0) / wr;
                    let diff = p_l.sqrt() - p_r.sqrt();
                    h_sq += diff * diff;
                }
                (wl * wr / w) * h_sq
            }
            GainFn::MaxStat => {
                let n = all.len();
                if n < 2 || wl <= 0.0 || wr <= 0.0 {
                    return 0.0;
                }
                let mut indexed: Vec<(f64, f64, bool)> = Vec::with_capacity(n);
                for &(y, wt) in left {
                    indexed.push((y, wt, true));
                }
                for &(y, wt) in right {
                    indexed.push((y, wt, false));
                }
                indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mean_rank = (n as f64 + 1.0) / 2.0;
                let mut sum_a_l = 0.0;
                let mut var_a = 0.0;
                for (rank_idx, item) in indexed.iter().enumerate() {
                    let rank = rank_idx as f64 + 1.0;
                    let a_i = rank - mean_rank;
                    var_a += item.1 * a_i * a_i;
                    if item.2 {
                        sum_a_l += item.1 * a_i;
                    }
                }
                let var_s = (wl * wr / w) * (var_a / w);
                if var_s > 1e-12 {
                    (sum_a_l * sum_a_l) / var_s
                } else {
                    0.0
                }
            }
            GainFn::Custom(f) => f.gain(left, right),
        }
    }

