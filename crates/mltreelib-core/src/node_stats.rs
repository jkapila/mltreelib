//! Node/leaf statistics and models — standalone functions and value types.
//! `model_tree.rs` routes rows and scans split candidates; everything about
//! what a node *stores* (moments, t-digest, linear model) and the small
//! numerics of split scoring live here, one concern per function.
//!
//! Everything is **row-weighted**: sample weights flow through the Gram
//! statistics, the leaf moments, and the gains, so boosting/forest
//! reweighting and case weights work without touching the tree.

use crate::family::{std_norm_cdf, std_norm_inv_cdf, std_norm_pdf, LeafFamily};
use tdigest::TDigest;

/// Target summary stored at every leaf — mean nodes and model nodes alike.
/// Weighted by sample weights (`w`); `std` is the weighted sample standard
/// deviation. The t-digest (Cam Davidson-Pilon semantics via the `tdigest`
/// crate) keeps quantiles queryable without storing every target.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafStats {
    pub n: usize,
    /// Sum of the sample weights.
    #[serde(with = "crate::persistence::float")]
    pub wsum: f64,
    #[serde(with = "crate::persistence::float")]
    pub mean: f64,
    #[serde(with = "crate::persistence::float")]
    pub std: f64,
    #[serde(with = "crate::persistence::float")]
    pub var: f64,
    #[serde(with = "crate::persistence::float")]
    pub skew: f64,
    #[serde(with = "crate::persistence::float")]
    pub kurtosis: f64,
    #[serde(with = "crate::persistence::float")]
    pub median: f64,
    #[serde(with = "crate::persistence::float")]
    pub mad: f64,
    #[serde(with = "crate::persistence::float")]
    pub iqr: f64,
    #[serde(with = "crate::persistence::float")]
    pub min: f64,
    #[serde(with = "crate::persistence::float")]
    pub max: f64,
    #[serde(with = "crate::persistence::float")]
    pub prob: f64,
    #[serde(with = "crate::persistence::float")]
    pub odds: f64,
    #[serde(with = "crate::persistence::float")]
    pub log_odds: f64,
    pub majority_class: usize,
    #[serde(with = "crate::persistence::float")]
    pub entropy: f64,
    #[serde(with = "crate::persistence::float")]
    pub gini: f64,
    #[serde(with = "crate::persistence::float")]
    pub zero_fraction: f64,
    #[serde(with = "crate::persistence::float")]
    pub dispersion: f64,
    #[serde(with = "crate::persistence::digest")]
    pub digest: TDigest,
}

impl LeafStats {
    /// Weighted moments + t-digest of one leaf's targets. Rows with weight 0
    /// are ignored entirely (a zero-weight row is an absent row).
    pub fn from_targets(y: &[f64], w: &[f64]) -> Self {
        let mut idx: Vec<usize> = (0..y.len()).filter(|&i| w[i] > 0.0).collect();
        assert!(!idx.is_empty(), "leaf must have positive-weight rows");
        let wsum: f64 = idx.iter().map(|&i| w[i]).sum();
        let mean = idx.iter().map(|&i| w[i] * y[i]).sum::<f64>() / wsum;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        let mut w_zero = 0.0;
        for &i in &idx {
            let diff = y[i] - mean;
            let diff2 = diff * diff;
            m2 += w[i] * diff2;
            m3 += w[i] * diff2 * diff;
            m4 += w[i] * diff2 * diff2;
            if y[i].abs() < 1e-9 {
                w_zero += w[i];
            }
        }
        let var = m2 / wsum;
        let std = var.sqrt();
        let skew = if std > 1e-9 {
            (m3 / wsum) / (var * std)
        } else {
            0.0
        };
        let kurtosis = if std > 1e-9 {
            (m4 / wsum) / (var * var) - 3.0
        } else {
            0.0
        };
        let zero_fraction = w_zero / wsum;
        let dispersion = if mean.abs() > 1e-9 {
            var / mean.abs()
        } else {
            0.0
        };

        idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).expect("NaN in target"));
        let mut digest = TDigest::new_with_size(100);
        digest.extend_values(idx.iter().map(|&i| y[i]));
        digest.flush();

        let q25 = digest.estimate_quantile(0.25).unwrap_or(mean);
        let q50 = digest.estimate_quantile(0.5).unwrap_or(mean);
        let q75 = digest.estimate_quantile(0.75).unwrap_or(mean);
        let median = q50;
        let iqr = q75 - q25;

        let mut abs_devs: Vec<(f64, f64)> =
            idx.iter().map(|&i| ((y[i] - median).abs(), w[i])).collect();
        abs_devs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum_w = 0.0;
        let half_w = wsum * 0.5;
        let mut mad = 0.0;
        for &(dev, wi) in &abs_devs {
            cum_w += wi;
            if cum_w >= half_w {
                mad = dev;
                break;
            }
        }

        let p_clamped = mean.clamp(0.0, 1.0);
        let p_smooth = mean.clamp(1e-6, 1.0 - 1e-6);
        let odds = p_smooth / (1.0 - p_smooth);
        let log_odds = odds.ln();
        let majority_class = if p_clamped >= 0.5 { 1 } else { 0 };
        let entropy = if p_clamped > 0.0 && p_clamped < 1.0 {
            -p_clamped * p_clamped.ln() - (1.0 - p_clamped) * (1.0 - p_clamped).ln()
        } else {
            0.0
        };
        let gini = 2.0 * p_clamped * (1.0 - p_clamped);

        Self {
            n: idx.len(),
            wsum,
            mean,
            std,
            var,
            skew,
            kurtosis,
            median,
            mad,
            iqr,
            min: y[idx[0]],
            max: y[*idx.last().unwrap()],
            prob: p_clamped,
            odds,
            log_odds,
            majority_class,
            entropy,
            gini,
            zero_fraction,
            dispersion,
            digest,
        }
    }

    /// Quantile estimate from the leaf's t-digest (q in 0..=1).
    pub fn quantile(&self, q: f64) -> Option<f64> {
        self.digest.estimate_quantile(q)
    }

    /// Class probability P(Y=1) for binary targets (clamped to [0, 1]).
    pub fn prob(&self) -> f64 {
        self.prob
    }

    /// Odds ratio p / (1 - p), with epsilon smoothing to prevent division by zero.
    pub fn odds(&self) -> f64 {
        self.odds
    }

    /// Log-odds (logit) ln(p / (1 - p)).
    pub fn log_odds(&self) -> f64 {
        self.log_odds
    }

    /// Majority class label (1 if prob >= 0.5, else 0).
    pub fn majority_class(&self) -> usize {
        self.majority_class
    }

    /// Moments for split-candidate fits, without sorting targets or building
    /// a digest. The stored empty digest is never exposed as a final leaf.
    pub(crate) fn from_rows(y: &[f64], w: &[f64], rows: &[usize]) -> Self {
        let frame = FrameStats::from_ty(y, w, rows);
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &r in rows {
            if w[r] > 0.0 {
                min = min.min(y[r]);
                max = max.max(y[r]);
            }
        }
        let var = if frame.wsum > 0.0 {
            frame.sse / frame.wsum
        } else {
            0.0
        };
        let p_clamped = frame.mean.clamp(0.0, 1.0);
        let p_smooth = frame.mean.clamp(1e-6, 1.0 - 1e-6);
        let odds = p_smooth / (1.0 - p_smooth);
        Self {
            n: frame.n,
            wsum: frame.wsum,
            mean: frame.mean,
            std: var.sqrt(),
            var,
            skew: 0.0,
            kurtosis: 0.0,
            median: frame.mean,
            mad: 0.0,
            iqr: 0.0,
            min,
            max,
            prob: p_clamped,
            odds,
            log_odds: odds.ln(),
            majority_class: if frame.mean >= 0.5 { 1 } else { 0 },
            entropy: 0.0,
            gini: 2.0 * p_clamped * (1.0 - p_clamped),
            zero_fraction: 0.0,
            dispersion: if frame.mean.abs() > 1e-9 {
                var / frame.mean.abs()
            } else {
                0.0
            },
            digest: TDigest::new_with_size(1),
        }
    }
}

/// Single coefficient statistics matching R summary(lm) and statsmodels.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CoeffStats {
    pub col_idx: Option<usize>,
    #[serde(with = "crate::persistence::float")]
    pub estimate: f64,
    #[serde(with = "crate::persistence::float")]
    pub std_err: f64,
    #[serde(with = "crate::persistence::float")]
    pub t_value: f64,
    #[serde(with = "crate::persistence::float")]
    pub p_value: f64,
}

/// Goodness-of-fit and hypothesis testing summary for a leaf regression model.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegressionSummary {
    pub n: usize,
    pub df_residuals: usize,
    #[serde(with = "crate::persistence::float")]
    pub r_squared: f64,
    #[serde(with = "crate::persistence::float")]
    pub adj_r_squared: f64,
    #[serde(with = "crate::persistence::float")]
    pub sigma: f64,
    #[serde(with = "crate::persistence::float")]
    pub f_statistic: f64,
    #[serde(with = "crate::persistence::float")]
    pub f_p_value: f64,
    pub coefficients: Vec<CoeffStats>,
}

impl RegressionSummary {
    /// Format as standard R summary(lm) / Python statsmodels table.
    pub fn to_r_format(&self, title: &str, col_names: &[String]) -> String {
        let mut out = String::new();
        out.push_str("======================================================================\n");
        out.push_str(&format!("{}\n", title));
        out.push_str("----------------------------------------------------------------------\n");
        out.push_str(&format!(
            "Residual standard error: {:.4} on {} degrees of freedom\n",
            self.sigma, self.df_residuals
        ));
        out.push_str(&format!(
            "Multiple R-squared:  {:.4},  Adjusted R-squared:  {:.4}\n",
            self.r_squared, self.adj_r_squared
        ));
        let f_p_str = if self.f_p_value < 2.2e-16 {
            "< 2.2e-16".to_string()
        } else {
            format!("{:.4e}", self.f_p_value)
        };
        let p_df = self.coefficients.len().saturating_sub(1).max(1);
        out.push_str(&format!(
            "F-statistic: {:.2} on {} and {} DF,  p-value: {}\n\n",
            self.f_statistic, p_df, self.df_residuals, f_p_str
        ));
        out.push_str("Coefficients:\n");
        out.push_str(&format!(
            "{:<18} {:>12} {:>12} {:>10} {:>12} \n",
            "", "Estimate", "Std. Error", "t value", "Pr(>|t|)"
        ));

        for c in &self.coefficients {
            let name = match c.col_idx {
                None => "(Intercept)".to_string(),
                Some(idx) => col_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", idx)),
            };
            let p_str = if c.p_value < 2e-16 {
                "< 2e-16".to_string()
            } else if c.p_value < 0.001 {
                format!("{:.3e}", c.p_value)
            } else {
                format!("{:.4}", c.p_value)
            };
            let stars = if c.p_value < 0.001 {
                "***"
            } else if c.p_value < 0.01 {
                "**"
            } else if c.p_value < 0.05 {
                "*"
            } else if c.p_value < 0.10 {
                "."
            } else {
                " "
            };
            out.push_str(&format!(
                "{:<18} {:>12.4} {:>12.4} {:>10.2} {:>12} {:<3}\n",
                name, c.estimate, c.std_err, c.t_value, p_str, stars
            ));
        }
        out.push_str("---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");
        out.push_str("======================================================================\n");
        out
    }
}

/// Fitted link-scale linear predictor:
/// `intercept + Σ coef·x + Σ na_coef·1{missing}`. Gaussian uses the identity
/// link. Missing predictors contribute nothing to value terms and have their
/// own indicator term — missingness is an effect, never a lost row.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinearModel {
    pub cols: Vec<usize>,
    #[serde(with = "crate::persistence::float")]
    pub coef: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub na_coef: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub feature_means: Vec<Option<f64>>,
    #[serde(with = "crate::persistence::float")]
    pub intercept: f64,
    pub summary: Option<RegressionSummary>,
}

impl LinearModel {
    pub fn new(cols: Vec<usize>, coef: Vec<f64>, na_coef: Vec<f64>, intercept: f64) -> Self {
        Self {
            cols,
            coef,
            na_coef,
            feature_means: Vec::new(),
            intercept,
            summary: None,
        }
    }

    pub fn new_with_means(
        cols: Vec<usize>,
        coef: Vec<f64>,
        na_coef: Vec<f64>,
        feature_means: Vec<Option<f64>>,
        intercept: f64,
    ) -> Self {
        Self {
            cols,
            coef,
            na_coef,
            feature_means,
            intercept,
            summary: None,
        }
    }

    /// Predict for one row (`None` = missing). Always returns a value; a row
    /// shorter than the fit-time width treats absent cells as missing (an NA
    /// never blocks a prediction, and never panics).
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        let mut y = self.intercept;
        for (k, &j) in self.cols.iter().enumerate() {
            match x.get(j).copied().flatten() {
                Some(v) => y += self.coef[k] * v,
                None => {
                    if let Some(Some(mean_val)) = self.feature_means.get(k) {
                        y += self.coef[k] * mean_val;
                    } else {
                        y += self.na_coef.get(k).copied().unwrap_or(0.0);
                    }
                }
            }
        }
        y
    }
}

/// Summary statistics for non-parametric survival curves (Nelson-Aalen / Kaplan-Meier)
/// and competing risks Cumulative Incidence Functions (Aalen-Johansen CIF).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurvivalStats {
    #[serde(with = "crate::persistence::float")]
    pub times: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub cum_hazard: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub survival: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub median_time: f64,
    pub causes: Vec<usize>,
    #[serde(with = "crate::persistence::float")]
    pub cif: Vec<Vec<f64>>,
    pub event_counts: Vec<usize>,
}

/// Fitted multinomial softmax parameters for a leaf.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SoftmaxModel {
    pub n_classes: usize,
    #[serde(with = "crate::persistence::float")]
    pub probs: Vec<f64>,
    pub linear_models: Option<Vec<LinearModel>>,
}

/// Fitted heteroscedastic location-scale parameters for a leaf.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeteroscedasticModel {
    #[serde(with = "crate::persistence::float")]
    pub scale: f64,
    pub scale_linear: Option<LinearModel>,
}

/// Fitted causal / treatment effect parameters for a leaf.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CausalStats {
    /// Estimated Conditional Average Treatment Effect (CATE): `mean_treated - mean_control`.
    #[serde(with = "crate::persistence::float")]
    pub tau: f64,
    #[serde(with = "crate::persistence::float")]
    pub mean_treated: f64,
    #[serde(with = "crate::persistence::float")]
    pub mean_control: f64,
    pub n_treated: usize,
    pub n_control: usize,
    #[serde(with = "crate::persistence::float")]
    pub w_treated: f64,
    #[serde(with = "crate::persistence::float")]
    pub w_control: f64,
}

/// Parametric distribution for Accelerated Failure Time (AFT) models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AftDist {
    #[default]
    Weibull,
    LogNormal,
    LogLogistic,
}

/// Fitted parametric Accelerated Failure Time (AFT) model for a leaf.
/// Models ln(T) = intercept + sum(coef_j * x_j) + scale * epsilon.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AftModel {
    pub dist: AftDist,
    #[serde(with = "crate::persistence::float")]
    pub intercept: f64,
    #[serde(with = "crate::persistence::float")]
    pub coef: Vec<f64>,
    pub cols: Vec<usize>,
    #[serde(with = "crate::persistence::float")]
    pub scale: f64,
}

impl AftModel {
    pub fn linear_predictor(&self, x: &[f64]) -> f64 {
        let mut eta = self.intercept;
        for (k, &col) in self.cols.iter().enumerate() {
            if col < x.len() && !x[col].is_nan() {
                eta += self.coef[k] * x[col];
            }
        }
        eta
    }

    pub fn predict_survival(&self, x: &[f64], t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let eta = self.linear_predictor(x);
        let sigma = self.scale.max(1e-4);
        let z = (t.ln() - eta) / sigma;
        match self.dist {
            AftDist::Weibull => (-z.exp()).exp().clamp(0.0, 1.0),
            AftDist::LogNormal => (1.0 - std_norm_cdf(z)).clamp(0.0, 1.0),
            AftDist::LogLogistic => (1.0 / (1.0 + z.exp())).clamp(0.0, 1.0),
        }
    }

    pub fn predict_cumulative_hazard(&self, x: &[f64], t: f64) -> f64 {
        let s = self.predict_survival(x, t).max(1e-15);
        -s.ln()
    }

    pub fn predict_median(&self, x: &[f64]) -> f64 {
        let eta = self.linear_predictor(x);
        let sigma = self.scale.max(1e-4);
        match self.dist {
            AftDist::Weibull => {
                let z = (2.0f64.ln()).ln();
                (eta + sigma * z).exp()
            }
            AftDist::LogNormal => eta.exp(),
            AftDist::LogLogistic => eta.exp(),
        }
    }
}

/// Fitted mixture cure rate model for a leaf.
/// Models S(t | x) = cure_rate + (1 - cure_rate) * S_uncured(t | x).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CureModel {
    #[serde(with = "crate::persistence::float")]
    pub cure_rate: f64,
    pub aft: Option<AftModel>,
    pub survival: Option<SurvivalStats>,
}

impl CureModel {
    pub fn predict_survival(&self, x: &[f64], t: f64) -> f64 {
        let pi = self.cure_rate.clamp(0.0, 1.0);
        let s_u = if let Some(aft) = &self.aft {
            aft.predict_survival(x, t)
        } else if let Some(surv) = &self.survival {
            if surv.times.is_empty() {
                1.0
            } else {
                match surv.times.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
                    Ok(idx) => surv.survival[idx],
                    Err(0) => 1.0,
                    Err(idx) => surv.survival[idx - 1],
                }
            }
        } else {
            1.0
        };
        (pi + (1.0 - pi) * s_u).clamp(0.0, 1.0)
    }

    pub fn predict_cumulative_hazard(&self, x: &[f64], t: f64) -> f64 {
        let s = self.predict_survival(x, t).max(1e-15);
        -s.ln()
    }
}

/// Fitted Royston-Parmar restricted cubic spline model for a leaf.
/// Models ln(Lambda(t | x)) = s(ln t; gamma) + beta^T x.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SplineSurvivalModel {
    #[serde(with = "crate::persistence::float")]
    pub knots: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub gamma: Vec<f64>,
    #[serde(with = "crate::persistence::float")]
    pub coef: Vec<f64>,
    pub cols: Vec<usize>,
}

impl SplineSurvivalModel {
    pub fn predict_survival(&self, x: &[f64], t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let ln_t = t.ln();
        let mut s_val = self.gamma.first().copied().unwrap_or(0.0);
        if self.gamma.len() > 1 {
            s_val += self.gamma[1] * ln_t;
        }
        if self.knots.len() >= 3 && self.gamma.len() >= 3 {
            let k_min = self.knots[0];
            let k_max = *self.knots.last().unwrap();
            for j in 1..(self.knots.len() - 1) {
                if j + 1 < self.gamma.len() {
                    let kj = self.knots[j];
                    let lam = (k_max - kj) / (k_max - k_min);
                    let term1 = (ln_t - kj).max(0.0).powi(3);
                    let term2 = lam * (ln_t - k_min).max(0.0).powi(3);
                    let term3 = (1.0 - lam) * (ln_t - k_max).max(0.0).powi(3);
                    let basis = term1 - term2 - term3;
                    s_val += self.gamma[j + 1] * basis;
                }
            }
        }
        let mut cov_val = 0.0;
        for (k, &col) in self.cols.iter().enumerate() {
            if col < x.len() && !x[col].is_nan() && k < self.coef.len() {
                cov_val += self.coef[k] * x[col];
            }
        }
        let ln_cum_hazard = (s_val + cov_val).clamp(-30.0, 30.0);
        let cum_hazard = ln_cum_hazard.exp();
        (-cum_hazard).exp().clamp(0.0, 1.0)
    }

    pub fn predict_cumulative_hazard(&self, x: &[f64], t: f64) -> f64 {
        let s = self.predict_survival(x, t).max(1e-15);
        -s.ln()
    }
}

/// Leaf payload: distribution stats always; a linear model for model-tree
/// leaves (`None` for mean/CART leaves).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafModel {
    pub family: LeafFamily,
    pub stats: LeafStats,
    pub linear: Option<LinearModel>,
    /// Fitted response-scale constant (robust Gaussian locations or exact
    /// boundary family means). `None` = the weighted response mean.
    #[serde(with = "crate::persistence::float")]
    pub center: Option<f64>,
    pub softmax: Option<SoftmaxModel>,
    pub survival: Option<SurvivalStats>,
    pub heteroscedastic: Option<HeteroscedasticModel>,
    pub causal: Option<CausalStats>,
    pub aft: Option<AftModel>,
    pub cure: Option<CureModel>,
    pub spline: Option<SplineSurvivalModel>,
    pub mars: Option<crate::mars::MarsModel>,
    pub mars_cols: Option<Vec<usize>>,
    pub knn: Option<crate::knn::LeafKnnModel>,
    pub knn_cols: Option<Vec<usize>>,
    #[serde(with = "crate::persistence::float")]
    pub vector_mean: Option<Vec<f64>>,
    /// Solver actually used, or why requested solver did not apply.
    #[serde(default)]
    pub solver_used: crate::leaf::LeafSolverDiagnostic,
    /// Optimization problem declared by this leaf.
    #[serde(default)]
    pub leaf_problem_type: crate::leaf::LeafProblemType,
}

impl LeafModel {
    /// Hierarchical shrinkage of leaf coefficients toward an ancestor/parent leaf model:
    /// beta_shrunk = beta_parent + (N_t / (N_t + lambda)) * (beta_leaf - beta_parent)
    pub fn shrink_toward(&mut self, parent: &LeafModel, lambda: f64, n_samples: f64) {
        if lambda <= 0.0 || n_samples <= 0.0 {
            return;
        }
        let gamma = n_samples / (n_samples + lambda);

        // 1. Shrink LinearModel if present
        if let Some(lm) = &mut self.linear {
            if let Some(plm) = &parent.linear {
                lm.intercept = plm.intercept + gamma * (lm.intercept - plm.intercept);
                for (k, &col) in lm.cols.iter().enumerate() {
                    let p_coef = plm
                        .cols
                        .iter()
                        .position(|&pc| pc == col)
                        .map(|idx| plm.coef[idx])
                        .unwrap_or(0.0);
                    lm.coef[k] = p_coef + gamma * (lm.coef[k] - p_coef);
                    if k < lm.na_coef.len() {
                        let p_na = plm
                            .cols
                            .iter()
                            .position(|&pc| pc == col)
                            .and_then(|idx| plm.na_coef.get(idx))
                            .copied()
                            .unwrap_or(0.0);
                        lm.na_coef[k] = p_na + gamma * (lm.na_coef[k] - p_na);
                    }
                }
            } else {
                let p_mean = parent.center.unwrap_or(parent.stats.mean);
                lm.intercept = p_mean + gamma * (lm.intercept - p_mean);
                for c in &mut lm.coef {
                    *c *= gamma;
                }
                for c in &mut lm.na_coef {
                    *c *= gamma;
                }
            }
        }

        // 2. Shrink response-scale center / mean
        let p_center = parent.center.unwrap_or(parent.stats.mean);
        let cur_center = self.center.unwrap_or(self.stats.mean);
        let shrunk_center = p_center + gamma * (cur_center - p_center);
        self.center = Some(shrunk_center);

        // 3. Shrink SoftmaxModel if present
        if let (Some(sm), Some(psm)) = (&mut self.softmax, &parent.softmax) {
            let n_c = sm.probs.len().min(psm.probs.len());
            let mut sum_p = 0.0;
            for c in 0..n_c {
                sm.probs[c] = psm.probs[c] + gamma * (sm.probs[c] - psm.probs[c]);
                sum_p += sm.probs[c];
            }
            if sum_p > 0.0 {
                for c in 0..n_c {
                    sm.probs[c] /= sum_p;
                }
            }
            if let (Some(lms), Some(plms)) = (&mut sm.linear_models, &psm.linear_models) {
                let m_len = lms.len().min(plms.len());
                for c in 0..m_len {
                    let lm = &mut lms[c];
                    let plm = &plms[c];
                    lm.intercept = plm.intercept + gamma * (lm.intercept - plm.intercept);
                    for (k, &col) in lm.cols.iter().enumerate() {
                        let p_coef = plm
                            .cols
                            .iter()
                            .position(|&pc| pc == col)
                            .map(|idx| plm.coef[idx])
                            .unwrap_or(0.0);
                        lm.coef[k] = p_coef + gamma * (lm.coef[k] - p_coef);
                    }
                }
            }
        }
    }

    /// Linear model when present, the fitted constant otherwise (weighted
    /// mean for `LeafKind::Mean`).
    pub fn predict(&self, x: &[Option<f64>]) -> f64 {
        if let (Some(knn), Some(_chosen)) = (&self.knn, &self.knn_cols) {
            return knn.predict(x, self.family);
        }
        if let (Some(mars), Some(chosen)) = (&self.mars, &self.mars_cols) {
            let row_f64: Vec<f64> = chosen
                .iter()
                .map(|&c| x.get(c).and_then(|&v| v).unwrap_or(0.0))
                .collect();
            return mars.predict_row(&row_f64);
        }
        match self.family {
            LeafFamily::Softmax => {
                if let Some(sm) = &self.softmax {
                    if let Some(models) = &sm.linear_models {
                        let mut best_c = 0;
                        let mut max_eta = f64::NEG_INFINITY;
                        for (c, m) in models.iter().enumerate() {
                            let eta = m.predict(x);
                            if eta > max_eta {
                                max_eta = eta;
                                best_c = c;
                            }
                        }
                        best_c as f64
                    } else {
                        sm.probs
                            .iter()
                            .enumerate()
                            .max_by(|a, b| {
                                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(c, _)| c as f64)
                            .unwrap_or(0.0)
                    }
                } else {
                    self.center.unwrap_or(self.stats.mean)
                }
            }
            LeafFamily::Survival => {
                if let Some(surv) = &self.survival {
                    surv.median_time
                } else {
                    self.center.unwrap_or(self.stats.mean)
                }
            }
            _ => match &self.linear {
                Some(model) => {
                    let has_na = model
                        .cols
                        .iter()
                        .any(|&j| x.get(j).copied().flatten().is_none());
                    if has_na && model.na_coef.is_empty() && model.feature_means.is_empty() {
                        self.center.unwrap_or(self.stats.mean)
                    } else {
                        self.family.inverse_link(model.predict(x))
                    }
                }
                None => self.center.unwrap_or(self.stats.mean),
            },
        }
    }

    /// Conditional Average Treatment Effect (CATE) / uplift prediction at this leaf.
    pub fn predict_uplift(&self) -> f64 {
        self.causal.as_ref().map(|c| c.tau).unwrap_or(0.0)
    }

    /// Column-major prediction for training-row scoring, without allocating
    /// or copying a row for every candidate leaf.
    pub(crate) fn predict_columns(&self, cols: &[Vec<Option<f64>>], r: usize) -> f64 {
        if let (Some(mars), Some(chosen)) = (&self.mars, &self.mars_cols) {
            let row_f64: Vec<f64> = chosen.iter().map(|&c| cols[c][r].unwrap_or(0.0)).collect();
            return mars.predict_row(&row_f64);
        }
        let Some(model) = &self.linear else {
            return self.center.unwrap_or(self.stats.mean);
        };
        let has_na = model.cols.iter().any(|&c| cols[c][r].is_none());
        if has_na && model.na_coef.is_empty() && model.feature_means.is_empty() {
            return self.center.unwrap_or(self.stats.mean);
        }
        let mut eta = model.intercept;
        for (k, &c) in model.cols.iter().enumerate() {
            match cols[c][r] {
                Some(value) => eta += model.coef[k] * value,
                None => {
                    if let Some(Some(mean_val)) = model.feature_means.get(k) {
                        eta += model.coef[k] * mean_val;
                    } else {
                        eta += model.na_coef.get(k).copied().unwrap_or(0.0);
                    }
                }
            }
        }
        self.family.inverse_link(eta)
    }

    /// Predict class probabilities for classification leaves (Softmax, Binomial, Probit, Knn).
    pub fn predict_proba_row(&self, x: &[Option<f64>]) -> Vec<f64> {
        if let Some(knn) = &self.knn {
            return knn.predict_proba(x);
        }
        match self.family {
            LeafFamily::Softmax => {
                if let Some(sm) = &self.softmax {
                    if let Some(models) = &sm.linear_models {
                        let has_na = models
                            .first()
                            .map(|m| {
                                m.cols
                                    .iter()
                                    .any(|&j| x.get(j).copied().flatten().is_none())
                            })
                            .unwrap_or(false);
                        if has_na
                            && models
                                .first()
                                .map(|m| m.na_coef.is_empty() && m.feature_means.is_empty())
                                .unwrap_or(false)
                        {
                            sm.probs.clone()
                        } else {
                            let etas: Vec<f64> = models.iter().map(|m| m.predict(x)).collect();
                            let max_eta = etas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let exps: Vec<f64> =
                                etas.iter().map(|&eta| (eta - max_eta).exp()).collect();
                            let sum_exp: f64 = exps.iter().sum();
                            if sum_exp > 0.0 {
                                exps.iter().map(|&e| e / sum_exp).collect()
                            } else {
                                vec![1.0 / sm.n_classes as f64; sm.n_classes]
                            }
                        }
                    } else {
                        sm.probs.clone()
                    }
                } else {
                    vec![1.0]
                }
            }
            LeafFamily::Binomial | LeafFamily::Probit => {
                let p = self.predict(x);
                vec![1.0 - p, p]
            }
            _ => {
                let p = self.predict(x);
                vec![p]
            }
        }
    }

    /// Predict survival step probabilities S(t) across a given time grid.
    pub fn predict_survival_row(&self, times: &[f64]) -> Vec<f64> {
        self.predict_survival_row_with_x(times, None)
    }

    pub fn predict_survival_row_with_x(&self, times: &[f64], x: Option<&[f64]>) -> Vec<f64> {
        if let (Some(aft), Some(x_row)) = (&self.aft, x) {
            return times
                .iter()
                .map(|&t| aft.predict_survival(x_row, t))
                .collect();
        }
        if let (Some(cure), Some(x_row)) = (&self.cure, x) {
            return times
                .iter()
                .map(|&t| cure.predict_survival(x_row, t))
                .collect();
        }
        if let (Some(spline), Some(x_row)) = (&self.spline, x) {
            return times
                .iter()
                .map(|&t| spline.predict_survival(x_row, t))
                .collect();
        }
        let Some(surv) = &self.survival else {
            return vec![1.0; times.len()];
        };
        times
            .iter()
            .map(|&t| {
                if surv.times.is_empty() || t < surv.times[0] {
                    1.0
                } else {
                    let idx = match surv.times.binary_search_by(|st| {
                        st.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(exact) => exact,
                        Err(ins) => ins.saturating_sub(1),
                    };
                    surv.survival[idx]
                }
            })
            .collect()
    }

    /// Predict cumulative hazard Lambda(t) across a given time grid.
    pub fn predict_hazard_row(&self, times: &[f64]) -> Vec<f64> {
        self.predict_hazard_row_with_x(times, None)
    }

    pub fn predict_hazard_row_with_x(&self, times: &[f64], x: Option<&[f64]>) -> Vec<f64> {
        if let (Some(aft), Some(x_row)) = (&self.aft, x) {
            return times
                .iter()
                .map(|&t| aft.predict_cumulative_hazard(x_row, t))
                .collect();
        }
        if let (Some(cure), Some(x_row)) = (&self.cure, x) {
            return times
                .iter()
                .map(|&t| cure.predict_cumulative_hazard(x_row, t))
                .collect();
        }
        if let (Some(spline), Some(x_row)) = (&self.spline, x) {
            return times
                .iter()
                .map(|&t| spline.predict_cumulative_hazard(x_row, t))
                .collect();
        }
        let Some(surv) = &self.survival else {
            return vec![0.0; times.len()];
        };
        times
            .iter()
            .map(|&t| {
                if surv.times.is_empty() || t < surv.times[0] {
                    0.0
                } else {
                    let idx = match surv.times.binary_search_by(|st| {
                        st.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(exact) => exact,
                        Err(ins) => ins.saturating_sub(1),
                    };
                    surv.cum_hazard[idx]
                }
            })
            .collect()
    }

    /// Predict cause-specific Cumulative Incidence Functions F_j(t) for competing risks.
    pub fn predict_cif_row(&self, times: &[f64], cause: usize) -> Vec<f64> {
        let Some(surv) = &self.survival else {
            return vec![0.0; times.len()];
        };
        let cause_idx = surv.causes.iter().position(|&c| c == cause);
        let cif_curve = cause_idx.and_then(|idx| surv.cif.get(idx));
        times
            .iter()
            .map(|&t| {
                if surv.times.is_empty() || t < surv.times[0] {
                    0.0
                } else if let Some(curve) = cif_curve {
                    let idx = match surv.times.binary_search_by(|st| {
                        st.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(exact) => exact,
                        Err(ins) => ins.saturating_sub(1),
                    };
                    curve.get(idx).copied().unwrap_or(0.0)
                } else {
                    // Fallback to 1 - S(t) if single-event
                    let idx = match surv.times.binary_search_by(|st| {
                        st.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(exact) => exact,
                        Err(ins) => ins.saturating_sub(1),
                    };
                    (1.0 - surv.survival[idx]).max(0.0)
                }
            })
            .collect()
    }

    /// Predict location mu and dispersion sigma for heteroscedastic leaves.
    pub fn predict_dist_row(&self, x: &[Option<f64>]) -> (f64, f64) {
        let mu = self.predict(x);
        let sigma = match &self.heteroscedastic {
            Some(het) => match &het.scale_linear {
                Some(sm) => (0.5 * sm.predict(x)).exp().clamp(1e-6, 1e8),
                None => het.scale.max(1e-6),
            },
            None => self.stats.std.max(1e-6),
        };
        (mu, sigma)
    }
}

/// Weighted sufficient statistics of a least-squares design: `xtx`
/// (dim×dim row major), `xty`, `yty`, `n` (rows), `ws` (weight sum).
/// Additive per row or per value group — the split scan keeps running
/// left/right aggregates and scores candidates without refitting.
#[derive(Clone, Debug)]
pub struct Gram {
    pub xtx: Vec<f64>,
    pub xty: Vec<f64>,
    pub yty: f64,
    pub n: usize,
    pub ws: f64,
    /// Σ w·y (available as xty[0] too; kept for cheap leaf means).
    pub wy: f64,
    dim: usize,
}

impl Gram {
    pub fn new(dim: usize) -> Self {
        Self {
            xtx: vec![0.0; dim * dim],
            xty: vec![0.0; dim],
            yty: 0.0,
            n: 0,
            ws: 0.0,
            wy: 0.0,
            dim,
        }
    }

    /// Rank-1 update: add one design row with target `y` and weight `w`.
    /// Allocation-free.
    pub fn add_row(&mut self, x: &[f64], y: f64, w: f64) {
        let d = self.dim;
        for i in 0..d {
            let xi = x[i];
            if xi != 0.0 {
                self.xty[i] += w * xi * y;
                let row = i * d;
                for j in 0..d {
                    self.xtx[row + j] += w * xi * x[j];
                }
            }
        }
        self.yty += w * y * y;
        self.wy += w * y;
        self.ws += w;
        self.n += 1;
    }

    /// Add a whole aggregate (one value group).
    pub fn add(&mut self, other: &Gram) {
        for (a, b) in self.xtx.iter_mut().zip(&other.xtx) {
            *a += b;
        }
        for (a, b) in self.xty.iter_mut().zip(&other.xty) {
            *a += b;
        }
        self.yty += other.yty;
        self.wy += other.wy;
        self.ws += other.ws;
        self.n += other.n;
    }

    /// Subtract a whole aggregate (scan step the other way).
    pub fn sub(&mut self, other: &Gram) {
        for (a, b) in self.xtx.iter_mut().zip(&other.xtx) {
            *a -= b;
        }
        for (a, b) in self.xty.iter_mut().zip(&other.xty) {
            *a -= b;
        }
        self.yty -= other.yty;
        self.wy -= other.wy;
        self.ws -= other.ws;
        self.n -= other.n;
    }

    /// Design dimension of this aggregate (intercept included).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Weighted mean of the target over this aggregate.
    pub fn mean(&self) -> f64 {
        if self.ws > 0.0 {
            self.wy / self.ws
        } else {
            0.0
        }
    }
}

/// Working memory for solving the normal equations — allocated once per
/// node, reused for every candidate (the "preallocate the matrix" lesson
/// from LightGBM's `linear_tree_learner.cpp`).
pub struct SolveScratch {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl SolveScratch {
    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.b
    }

    pub fn new(dim: usize) -> Self {
        Self {
            a: vec![0.0; dim * dim],
            b: vec![0.0; dim],
            c: vec![0.0; dim],
        }
    }
}

/// Residual SSE of `xtx·b = xty` (least squares over the design), solved by
/// Gaussian elimination with partial pivoting into `scratch`. `l1`/`l2` are
/// the elastic-net penalties (ridge = l2 only, lasso = l1 only, OLS = both
/// 0); the intercept is unpenalized. At convergence the penalized objective
/// still equals `yty - b·xty`, so the SSE stays exact. Rank-deficient
/// directions are dropped (coefficient 0). Solution lands in `scratch.b`.
pub fn solve_normal(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
    scratch: &mut SolveScratch,
) -> f64 {
    scratch.a[..dim * dim].copy_from_slice(&xtx[..dim * dim]);
    scratch.b[..dim].copy_from_slice(&xty[..dim]);
    solve_prepared(scratch, yty, dim, l1, l2, None)
}

/// Dense SPD solve via faer's blocked Cholesky — the split scan's hot
/// path (thousands of small/medium Gram solves per fit). Returns `None`
/// for rank-deficient Grams so the caller falls back to the rank-revealing
/// elimination (singular directions keep coefficient 0).
fn faer_cholesky_solve(a: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    use faer::prelude::*;
    let mut m = faer::Mat::<f64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            m[(i, j)] = a[i * dim + j];
        }
    }
    let llt = m.as_ref().llt(faer::Side::Lower).ok()?;
    let mut rhs = faer::Mat::<f64>::zeros(dim, 1);
    for i in 0..dim {
        rhs[(i, 0)] = b[i];
    }
    let x = llt.solve(rhs.as_ref());
    Some((0..dim).map(|i| x[(i, 0)]).collect())
}

/// Solve the normal equations already copied into `scratch.a`/`scratch.b`.
pub fn solve_prepared(
    scratch: &mut SolveScratch,
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
    warm: Option<&mut [f64]>,
) -> f64 {
    scratch.c[..dim].copy_from_slice(&scratch.b[..dim]); // keep xty
    if l1 > 0.0 {
        return solve_elastic(scratch, yty, dim, l1, l2, warm);
    }
    if l2 > 0.0 {
        for i in 1..dim {
            scratch.a[i * dim + i] += l2; // ridge, intercept unpenalized
        }
    }
    // Fast path: faer Cholesky on the SPD Gram (the common case —
    // thousands of solves per split scan). Falls through to the
    // rank-revealing elimination below when the Gram is singular.
    if let Some(x) = faer_cholesky_solve(&scratch.a[..dim * dim], &scratch.b[..dim], dim) {
        scratch.b[..dim].copy_from_slice(&x);
        let explained: f64 = scratch.c[..dim]
            .iter()
            .zip(&scratch.b[..dim])
            .map(|(t, s)| t * s)
            .sum();
        return (yty - explained).max(0.0);
    }
    let scale = (0..dim)
        .map(|i| scratch.a[i * dim + i].abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let eps = 1e-10 * scale;

    for col in 0..dim {
        // partial pivot
        let mut piv_row = col;
        let mut piv_val = scratch.a[col * dim + col].abs();
        for r in col + 1..dim {
            let v = scratch.a[r * dim + col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = r;
            }
        }
        if piv_val <= eps {
            // rank-deficient direction: free variable, coefficient 0
            scratch.b[col] = 0.0;
            for r in 0..dim {
                scratch.a[r * dim + col] = 0.0;
            }
            continue;
        }
        if piv_row != col {
            for c in 0..dim {
                scratch.a.swap(col * dim + c, piv_row * dim + c);
            }
            scratch.b.swap(col, piv_row);
        }
        let diag = scratch.a[col * dim + col];
        for r in col + 1..dim {
            let f = scratch.a[r * dim + col] / diag;
            if f != 0.0 {
                for c in col..dim {
                    scratch.a[r * dim + c] -= f * scratch.a[col * dim + c];
                }
                scratch.b[r] -= f * scratch.b[col];
            }
        }
    }

    // back substitution (dropped columns keep coefficient 0)
    for col in (0..dim).rev() {
        let mut s = scratch.b[col];
        for c in col + 1..dim {
            s -= scratch.a[col * dim + c] * scratch.b[c];
        }
        let d = scratch.a[col * dim + col];
        scratch.b[col] = if d.abs() <= eps { 0.0 } else { s / d };
    }

    let explained: f64 = scratch.c[..dim]
        .iter()
        .zip(&scratch.b[..dim])
        .map(|(t, s)| t * s)
        .sum();
    (yty - explained).max(0.0)
}

/// Residual SSE of `(a + b)`'s normal equations without materializing the
/// sum: fills the shared scratch from `a` (+ `b`), then solves in place.
pub fn solve_normal_sum(
    a: &Gram,
    b: Option<&Gram>,
    dim: usize,
    l1: f64,
    l2: f64,
    scratch: &mut SolveScratch,
    warm: Option<&mut [f64]>,
) -> f64 {
    scratch.a[..dim * dim].copy_from_slice(&a.xtx[..dim * dim]);
    scratch.b[..dim].copy_from_slice(&a.xty[..dim]);
    let mut yty = a.yty;
    if let Some(e) = b {
        for (dst, src) in scratch.a[..dim * dim].iter_mut().zip(&e.xtx[..dim * dim]) {
            *dst += *src;
        }
        for (dst, src) in scratch.b[..dim].iter_mut().zip(&e.xty[..dim]) {
            *dst += *src;
        }
        yty += e.yty;
    }
    solve_prepared(scratch, yty, dim, l1, l2, warm)
}

/// Lower-triangular Cholesky factor of a moving design's normal matrix
/// (`xtx` + the `l2` ridge, intercept unpenalized), maintained across the
/// split sweep: rows entering a side apply a rank-1 update, rows leaving a
/// guarded rank-1 downdate. The Gram aggregate stays the source of truth —
/// the factor is a disposable cache around it, rebuilt on demand, so
/// correctness never depends on the incremental path. Storage is the
/// lower triangle, column-major (`L[i][j]` at `l[j*dim + i]`) so the
/// recurrence works contiguous slices; the recurrences are the classic
/// LINPACK `dchud`/`dchdd` shapes.
pub struct Chol {
    l: Vec<f64>,
    v: Vec<f64>, // rank-1 work buffer (allocation-free updates)
    dim: usize,
}

impl Chol {
    /// Factor `(a + b).xtx + l2·I*` (intercept unpenalized) — `None` when
    /// the matrix is not positive definite, in which case the caller keeps
    /// solving the exact historical way.
    pub fn factor_grams(a: &Gram, b: Option<&Gram>, l2: f64) -> Option<Chol> {
        let dim = a.dim();
        let mut xtx = a.xtx[..dim * dim].to_vec();
        if let Some(e) = b {
            for (dst, src) in xtx.iter_mut().zip(&e.xtx[..dim * dim]) {
                *dst += *src;
            }
        }
        let mut l = vec![0.0; dim * dim];
        for j in 0..dim {
            for i in j..dim {
                let mut s = xtx[i * dim + j] + if i == j && j > 0 { l2 } else { 0.0 };
                for k in 0..j {
                    s -= l[k * dim + i] * l[k * dim + j];
                }
                if i == j {
                    if !(s > 0.0) {
                        return None;
                    }
                    l[j * dim + j] = s.sqrt();
                } else {
                    l[j * dim + i] = s / l[j * dim + j];
                }
            }
        }
        Some(Chol {
            l,
            v: vec![0.0; dim],
            dim,
        })
    }

    /// `A + w·φφᵀ` — never fails: a positive-semidefinite addition keeps
    /// the factor valid.
    pub fn rank1_add(&mut self, phi: &[f64], w: f64) {
        self.rank1(phi, w, true);
    }

    /// `A − w·φφᵀ` — `false` when the downdate would leave the matrix
    /// non-positive-definite; the factor is then unusable and the caller
    /// drops it. (A refused downdate may leave `l` half-updated — dropping
    /// is the contract.)
    pub fn rank1_sub(&mut self, phi: &[f64], w: f64) -> bool {
        self.rank1(phi, w, false)
    }

    /// Shared recurrence for both directions: the L column update differs
    /// only in sign, and the work vector's update collapses to
    /// `(v − s·l_old)/c` in both (the rotation identities make the two
    /// textbook formulas coincide).
    fn rank1(&mut self, phi: &[f64], w: f64, add: bool) -> bool {
        if w <= 0.0 {
            return true;
        }
        let d = self.dim;
        let sw = w.sqrt();
        for (t, &s) in self.v[..d].iter_mut().zip(&phi[..d]) {
            *t = s * sw;
        }
        for j in 0..d {
            let x = self.l[j * d + j];
            let y = self.v[j];
            let r2 = if add { x * x + y * y } else { x * x - y * y };
            if !add && !(r2 > 1e-12 * x * x) {
                return false;
            }
            let r = r2.sqrt();
            let c = r / x;
            let s = y / x;
            let inv_c = 1.0 / c;
            self.l[j * d + j] = r;
            for (lp, vp) in self.l[j * d + j + 1..j * d + d]
                .iter_mut()
                .zip(&mut self.v[j + 1..d])
            {
                let lold = *lp;
                let sv = s * *vp;
                *lp = (lold + if add { sv } else { -sv }) * inv_c;
                *vp = (*vp - s * lold) * inv_c;
            }
        }
        true
    }

    /// Solve `(xtx + ridge)·b = xty` through the factor — O(dim²).
    pub fn solve_into(&self, xty: &[f64], out: &mut [f64]) {
        let d = self.dim;
        for i in 0..d {
            let mut s = xty[i];
            for k in 0..i {
                s -= self.l[k * d + i] * out[k];
            }
            out[i] = s / self.l[i * d + i];
        }
        for i in (0..d).rev() {
            let mut s = out[i];
            for k in i + 1..d {
                s -= self.l[i * d + k] * out[k];
            }
            out[i] = s / self.l[i * d + i];
        }
    }
}

/// One side of the linear scan's sweep. The moving Gram aggregate crosses
/// the cut in whole value groups (exact historical arithmetic); on top of
/// it, Cholesky factors of the two solve systems the side takes part in —
/// `gram` and `gram + extra` (`extra` = the column's fixed missing
/// aggregate) — follow the same rows one rank-1 step at a time and let
/// candidates be scored in O(dim²) instead of a fresh O(dim³) fit. The
/// folding is economic, not unconditional: a group wider than ~dim/3 rows
/// costs more to fold than to re-factor, so the sweep drops the factors
/// there and the solves take the exact historical path until a finer
/// group's [`SideTrack::ensure`] re-materializes them. Factors are
/// disposable (failed downdate, periodic refresh); correctness never
/// depends on the incremental path.
pub struct SideTrack {
    pub gram: Gram,
    pub extra: Option<Gram>,
    dim: usize,
    l2: f64,
    chols: [Option<Chol>; 2], // [plain, +extra]
    ops: u32,
    warm: [Vec<f64>; 2], // elastic-net warm starts (raw coefficients)
    xty: Vec<f64>,
    coef: Vec<f64>,
}

impl SideTrack {
    pub fn new(gram: Gram, extra: Option<Gram>, l2: f64) -> Self {
        let dim = gram.dim();
        Self {
            gram,
            extra,
            dim,
            l2,
            chols: [None, None],
            ops: 0,
            warm: [vec![0.0; dim], vec![0.0; dim]],
            xty: vec![0.0; dim],
            coef: vec![0.0; dim],
        }
    }

    /// Materialize the maintained factors from the current aggregate —
    /// called at groups small enough for the row-by-row folding to pay.
    /// Must run before the next group is folded into the Gram so the
    /// factor and the rank-1 steps stay consistent.
    pub fn ensure(&mut self) {
        if self.chols[0].is_none() {
            self.chols[0] = Chol::factor_grams(&self.gram, None, self.l2);
        }
        if self.chols[1].is_none() && self.extra.is_some() {
            self.chols[1] = Chol::factor_grams(&self.gram, self.extra.as_ref(), self.l2);
        }
    }

    /// Drop the maintained factors — the caller is folding a group too
    /// large for row-by-row updates.
    pub fn drop_factors(&mut self) {
        self.chols[0] = None;
        self.chols[1] = None;
    }

    /// One row crossing the cut: `dir > 0` entering this side (rank-1
    /// update), `dir < 0` leaving (guarded rank-1 downdate). The Gram
    /// aggregate is moved by the caller in whole groups; the factors track
    /// the same contributions row by row.
    pub fn move_row(&mut self, phi: &[f64], w: f64, dir: i32) {
        self.ops += 1;
        if self.ops % 2048 == 0 {
            self.chols[0] = None;
            self.chols[1] = None; // periodic refresh bounds drift
        }
        for i in 0..2 {
            if let Some(c) = self.chols[i].as_mut() {
                let ok = if dir > 0 {
                    c.rank1_add(phi, w);
                    true
                } else {
                    c.rank1_sub(phi, w)
                };
                if !ok {
                    self.chols[i] = None;
                }
            }
        }
    }

    /// Residual score of this side's system in the shared `yty − xty·b`
    /// convention (`plus` selects the `gram + extra` system).
    pub fn sse(&mut self, plus: bool, l1: f64, scratch: &mut SolveScratch) -> f64 {
        let idx = if plus && self.extra.is_some() { 1 } else { 0 };
        let extra = if plus { self.extra.as_ref() } else { None };
        if l1 > 0.0 {
            // Elastic net: no maintained factor — the historical descent,
            // warm-started from the previous candidate's coefficients.
            return solve_normal_sum(
                &self.gram,
                extra,
                self.dim,
                l1,
                self.l2,
                scratch,
                Some(&mut self.warm[idx][..]),
            );
        }
        let d = self.dim;
        self.xty[..d].copy_from_slice(&self.gram.xty[..d]);
        let mut yty = self.gram.yty;
        if let Some(e) = extra {
            for (t, s) in self.xty[..d].iter_mut().zip(&e.xty[..d]) {
                *t += *s;
            }
            yty += e.yty;
        }
        if let Some(c) = self.chols[idx].as_ref() {
            c.solve_into(&self.xty[..d], &mut self.coef[..d]);
            let explained: f64 = self.xty[..d]
                .iter()
                .zip(&self.coef[..d])
                .map(|(t, s)| t * s)
                .sum();
            (yty - explained).max(0.0)
        } else {
            solve_normal_sum(&self.gram, extra, d, l1, self.l2, scratch, None)
        }
    }
}

/// Elastic net by coordinate descent on the standardised Gram (glmnet's
/// approach): features standardised to unit scale so the penalty is fair
/// across columns, soft-thresholding for L1, coefficients unscaled on the
/// way out. The intercept is never penalized.
fn solve_elastic(
    scratch: &mut SolveScratch,
    yty: f64,
    dim: usize,
    l1: f64,
    l2: f64,
    warm: Option<&mut [f64]>,
) -> f64 {
    let n = scratch.a[0].max(1.0);
    let sum_y = scratch.c[0];
    let mut mean = vec![0.0; dim];
    let mut scale = vec![1.0; dim];
    for j in 1..dim {
        mean[j] = scratch.a[j] / n; // xtx[0][j] = sum of column j
    }
    for j in 1..dim {
        let var = scratch.a[j * dim + j] - n * mean[j] * mean[j];
        scale[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }
    let mut g = vec![0.0; (dim - 1) * (dim - 1)];
    let mut r = vec![0.0; dim - 1];
    for j in 1..dim {
        r[j - 1] = (scratch.c[j] - mean[j] * sum_y) / scale[j];
        for k in 1..dim {
            let gc = scratch.a[j * dim + k] - n * mean[j] * mean[k];
            g[(j - 1) * (dim - 1) + (k - 1)] = gc / (scale[j] * scale[k]);
        }
    }
    let mut b = vec![0.0; dim - 1];
    // Warm start from the previous candidate's coefficients (scaled into
    // this candidate's standardization): adjacent candidates differ by one
    // value group, so the descent usually converges in a few sweeps.
    if let Some(w) = warm.as_deref() {
        for j in 1..dim {
            b[j - 1] = w[j] * scale[j];
        }
    }
    for _ in 0..200 {
        let mut delta = 0.0f64;
        for j in 0..dim - 1 {
            let gjj = g[j * (dim - 1) + j];
            if gjj <= 0.0 {
                continue;
            }
            let mut partial = r[j];
            for k in 0..dim - 1 {
                if k != j {
                    partial -= g[j * (dim - 1) + k] * b[k];
                }
            }
            let raw = partial / (gjj + l2);
            let shrunk = raw.signum() * (raw.abs() - l1 / (gjj + l2)).max(0.0);
            delta = delta.max((shrunk - b[j]).abs());
            b[j] = shrunk;
        }
        if delta < 1e-10 {
            break;
        }
    }
    scratch.b[0] = sum_y / n;
    for j in 1..dim {
        let bj = b[j - 1] / scale[j];
        scratch.b[j] = bj;
        scratch.b[0] -= mean[j] * bj;
    }
    if let Some(w) = warm {
        w[..dim].copy_from_slice(&scratch.b[..dim]);
    }
    let explained: f64 = scratch.c[..dim]
        .iter()
        .zip(&scratch.b[..dim])
        .map(|(t, s)| t * s)
        .sum();
    (yty - explained).max(0.0)
}

/// Extract the fitted `LinearModel` from a design Gram. Layout: `[1, values
/// (0 where missing), NA indicators (only when `with_indicators`)]`. `l1`/`l2`
/// are the elastic-net penalties (intercept unpenalized). `mono` is indexed by
/// ORIGINAL column: a nonzero entry constrains the feature's direction, and
/// linear leaves carry feature effects in their coefficients - so the
/// coefficient is projected onto the allowed halfline (`+1` ⇒ coef ≥ 0,
/// `-1` ⇒ coef ≤ 0) to keep predictions monotone. Indicators and the
/// intercept are never constrained.
pub fn fit_linear(
    g: &Gram,
    chosen: &[usize],
    with_indicators: bool,
    mono: &[i8],
    l1: f64,
    l2: f64,
) -> LinearModel {
    let nc = chosen.len();
    let dim = 1 + nc + if with_indicators { nc } else { 0 };
    let mut scratch = SolveScratch::new(dim);
    solve_normal(&g.xtx, &g.xty, g.yty, dim, l1, l2, &mut scratch);
    let mut sol = scratch.b.clone();
    for (k, &c) in chosen.iter().enumerate() {
        match mono.get(c).copied().unwrap_or(0) {
            1 => sol[k + 1] = sol[k + 1].max(0.0),
            -1 => sol[k + 1] = sol[k + 1].min(0.0),
            _ => {}
        }
    }
    let summary = compute_regression_summary(&g.xtx, &g.xty, g.yty, g.n, &sol[..1 + nc], chosen);
    LinearModel {
        cols: chosen.to_vec(),
        coef: sol[1..1 + nc].to_vec(),
        na_coef: if with_indicators {
            sol[1 + nc..1 + 2 * nc].to_vec()
        } else {
            Vec::new()
        },
        feature_means: Vec::new(),
        intercept: sol[0],
        summary,
    }
}

pub fn compute_regression_summary(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: usize,
    sol: &[f64],
    chosen: &[usize],
) -> Option<RegressionSummary> {
    let p = sol.len();
    if n <= p {
        return None;
    }
    let df_residuals = n - p;

    let mut beta_xtx_beta = 0.0;
    for i in 0..p {
        let mut row_sum = 0.0;
        for j in 0..p {
            row_sum += xtx[i * p + j] * sol[j];
        }
        beta_xtx_beta += sol[i] * row_sum;
    }
    let mut beta_xty = 0.0;
    for i in 0..p {
        beta_xty += sol[i] * xty[i];
    }
    let rss = (yty - 2.0 * beta_xty + beta_xtx_beta).max(0.0);
    let sigma2 = rss / (df_residuals as f64);
    let sigma = sigma2.sqrt();

    let y_mean = xty[0] / (n as f64);
    let tss = (yty - (n as f64) * y_mean * y_mean).max(1e-12);
    let r_squared = (1.0 - rss / tss).max(0.0).min(1.0);
    let adj_r_squared =
        (1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df_residuals as f64)).max(0.0);

    let p_pred = p.saturating_sub(1);
    let f_statistic = if p_pred > 0 && rss > 1e-12 {
        ((tss - rss) / (p_pred as f64)) / (rss / (df_residuals as f64))
    } else {
        0.0
    };
    let f_p_value = crate::stats::f_sf(f_statistic, p_pred as u32, df_residuals as u32);

    let mut m = faer::Mat::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            m[(i, j)] = xtx[i * p + j];
            if i == j && i > 0 {
                m[(i, j)] += 1e-6;
            }
        }
    }
    use faer::linalg::solvers::DenseSolveCore;
    let inv = m.as_ref().partial_piv_lu().inverse();

    let mut coefficients = Vec::new();
    for i in 0..p {
        let col_idx = if i == 0 {
            None
        } else {
            chosen.get(i - 1).copied()
        };
        let var_beta = (inv[(i, i)] * sigma2).max(1e-12);
        let std_err = var_beta.sqrt();
        let est = sol[i];
        let t_val = est / std_err;
        let p_val = 2.0 * (1.0 - crate::family::std_norm_cdf(t_val.abs()));

        coefficients.push(CoeffStats {
            col_idx,
            estimate: est,
            std_err,
            t_value: t_val,
            p_value: p_val,
        });
    }

    Some(RegressionSummary {
        n,
        df_residuals,
        r_squared,
        adj_r_squared,
        sigma,
        f_statistic,
        f_p_value,
        coefficients,
    })
}

impl LinearModel {
    /// Evaluate on a design row `[1, values (0 where missing), NA
    /// indicators]` — the raw model output, used for monotone boundary
    /// ordering (comparing children's fitted outputs AT the split threshold).
    pub fn eval_design(&self, row: &[f64]) -> f64 {
        let nc = self.coef.len();
        let mut y = self.intercept;
        for k in 0..nc {
            y += self.coef[k] * row[1 + k];
        }
        for k in 0..self.na_coef.len() {
            y += self.na_coef[k] * row[1 + nc + k];
        }
        y
    }
}

/// Penalized canonical-link GLM by damped IRLS. The weighted least-squares
/// step reuses the Gaussian Gram/faer solver; constrained/lasso steps use
/// coordinate descent on the same Gram. L1 standardization is fixed from
/// the original case-weighted design, not changed by IRLS working weights.
///
/// Fisher variances are floored at 1e-12, steps are backtracked against
/// penalized deviance, and convergence requires relative objective change
/// below 1e-11. Separation therefore approaches a finite numerical boundary;
/// exact pure/zero leaves are handled by the caller. A failed or exhausted
/// fit returns None, never a successful-looking constant fallback.
#[allow(unused)]
pub(crate) fn fit_glm(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    chosen: &[usize],
    with_indicators: bool,
    mono: &[i8],
    l1: f64,
    l2: f64,
    family: LeafFamily,
    mean: f64,
    extra: Option<&std::collections::HashMap<String, f64>>,
    verbose: u8,
) -> Option<LinearModel> {
    fit_glm_imputed(
        cols,
        y,
        w,
        rows,
        chosen,
        with_indicators,
        None,
        mono,
        l1,
        l2,
        family,
        mean,
        extra,
        verbose,
    )
}

pub(crate) fn fit_glm_imputed(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    chosen: &[usize],
    with_indicators: bool,
    impute_means: Option<&[f64]>,
    mono: &[i8],
    l1: f64,
    l2: f64,
    family: LeafFamily,
    mean: f64,
    extra: Option<&std::collections::HashMap<String, f64>>,
    verbose: u8,
) -> Option<LinearModel> {
    let design = crate::solver::ColumnSliceView::new_with_imputation(
        cols,
        y,
        w,
        rows,
        chosen,
        with_indicators,
        impute_means,
    );
    let options = crate::solver::SolverOptions {
        l1,
        l2,
        max_iter: 100,
        tol: 1e-11,
        verbose,
        ..Default::default()
    };
    let (mut model, convergence) =
        fit_glm_design(&design, chosen, mono, family, mean, extra, &options).ok()?;
    if convergence.reason != crate::solver::StopReason::Tolerance {
        return None;
    }
    model.feature_means = impute_means
        .map(|means| means.iter().copied().map(Some).collect())
        .unwrap_or_default();
    Some(model)
}

pub(crate) fn fit_glm_design<D: crate::solver::DesignMatrix>(
    design: &D,
    chosen: &[usize],
    mono: &[i8],
    family: LeafFamily,
    mean: f64,
    extra: Option<&std::collections::HashMap<String, f64>>,
    options: &crate::solver::SolverOptions,
) -> Result<(LinearModel, crate::solver::Convergence), String> {
    let nc = chosen.len();
    let dim = design.n_features();
    let with_indicators = dim > nc + 1;
    let crate::solver::SolverOptions {
        l1,
        l2,
        max_iter,
        tol,
        verbose,
        ..
    } = *options;
    let mut current = LinearModel {
        cols: chosen.to_vec(),
        coef: vec![0.0; nc],
        na_coef: if with_indicators {
            vec![0.0; nc]
        } else {
            Vec::new()
        },
        feature_means: Vec::new(),
        intercept: match family {
            LeafFamily::Binomial => mean.ln() - (-mean).ln_1p(),
            LeafFamily::Probit => std_norm_inv_cdf(mean),
            LeafFamily::Poisson | LeafFamily::Gamma | LeafFamily::Tweedie => mean.max(1e-12).ln(),
            LeafFamily::Gaussian
            | LeafFamily::Tobit
            | LeafFamily::Softmax
            | LeafFamily::Survival
            | LeafFamily::Heteroscedastic
            | LeafFamily::Aft
            | LeafFamily::Cure
            | LeafFamily::Spline
            | LeafFamily::Quantile
            | LeafFamily::Huber => mean,
        },
        summary: None,
    };
    let mut gram = Gram::new(dim);
    let mut row = vec![0.0; dim];
    let mut scales = vec![1.0; dim];
    if l1 > 0.0 {
        gram = design.compute_gram(None);
        for j in 1..dim {
            let variance = gram.xtx[j * dim + j] - gram.xtx[j] * gram.xtx[j] / gram.ws;
            if variance > 1e-12 {
                scales[j] = variance.sqrt();
            }
        }
    }
    let penalty = |model: &LinearModel| -> f64 {
        model
            .coef
            .iter()
            .chain(&model.na_coef)
            .enumerate()
            .map(|(j, &b)| {
                let standardized = b * scales[j + 1];
                l1 * standardized.abs() + 0.5 * l2 * standardized * standardized
            })
            .sum()
    };
    let tweedie_p = extra
        .and_then(|m| m.get("tweedie_p"))
        .copied()
        .unwrap_or(1.5);
    let tobit_threshold = extra
        .and_then(|m| m.get("tobit_threshold"))
        .copied()
        .unwrap_or(0.0);
    let objective = |model: &LinearModel, row: &mut [f64]| -> f64 {
        let mut loss = 0.0;
        for r in 0..design.n_rows() {
            let w = design.get_weight(r);
            let y = design.get_y(r);
            if w <= 0.0 {
                continue;
            }
            design.copy_design_row(r, row);
            let mean = family.inverse_link(model.eval_design(row));
            loss += 0.5 * w * family.deviance_with_params(y, mean, tweedie_p, tobit_threshold);
        }
        loss + penalty(model)
    };
    let constrained = chosen
        .iter()
        .any(|&c| mono.get(c).copied().unwrap_or(0) != 0);
    let mut old_objective = objective(&current, &mut row);
    if !old_objective.is_finite() {
        return Err("IRLS produced a non-finite objective or weighted design".into());
    }
    if verbose >= 3 {
        println!(
            "      [glm start] family={:?} rows={} dim={} initial_obj={:.6}",
            family,
            design.n_rows(),
            dim,
            old_objective
        );
    }
    let mut beta = vec![0.0; dim];
    for iter in 0..max_iter {
        gram.xtx.fill(0.0);
        gram.xty.fill(0.0);
        gram.yty = 0.0;
        gram.wy = 0.0;
        gram.ws = 0.0;
        gram.n = 0;
        for r in 0..design.n_rows() {
            let w = design.get_weight(r);
            let y = design.get_y(r);
            if w <= 0.0 {
                continue;
            }
            design.copy_design_row(r, &mut row);
            let eta = current.eval_design(&row);
            let mean = family.inverse_link(eta);
            match family {
                LeafFamily::Binomial => {
                    let variance = (mean * (1.0 - mean)).max(1e-12);
                    let working = eta + (y - mean) / variance;
                    gram.add_row(&row, working, w * variance);
                }
                LeafFamily::Probit => {
                    let phi = std_norm_pdf(eta).max(1e-12);
                    let variance = (mean * (1.0 - mean)).max(1e-12);
                    let weight = w * (phi * phi) / variance;
                    let working = eta + (y - mean) / phi;
                    gram.add_row(&row, working, weight.max(1e-12));
                }
                LeafFamily::Poisson => {
                    let variance = mean.max(1e-12);
                    let working = eta + (y - mean) / variance;
                    gram.add_row(&row, working, w * variance);
                }
                LeafFamily::Gamma => {
                    let working = eta + (y - mean) / mean.max(1e-12);
                    gram.add_row(&row, working, w);
                }
                LeafFamily::Tweedie => {
                    let weight = w * mean.max(1e-12).powf(2.0 - tweedie_p);
                    let working = eta + (y - mean) / mean.max(1e-12);
                    gram.add_row(&row, working, weight.max(1e-12));
                }
                LeafFamily::Tobit => {
                    let y_latent = if y <= tobit_threshold {
                        let delta = tobit_threshold - eta;
                        let cdf = std_norm_cdf(delta).max(1e-12);
                        let pdf = std_norm_pdf(delta);
                        eta - pdf / cdf
                    } else {
                        y
                    };
                    gram.add_row(&row, y_latent, w);
                }
                LeafFamily::Gaussian
                | LeafFamily::Softmax
                | LeafFamily::Survival
                | LeafFamily::Heteroscedastic
                | LeafFamily::Aft
                | LeafFamily::Cure
                | LeafFamily::Spline
                | LeafFamily::Quantile
                | LeafFamily::Huber => {
                    gram.add_row(&row, y, w);
                }
            }
        }
        if gram
            .xtx
            .iter()
            .chain(&gram.xty)
            .any(|value| !value.is_finite())
        {
            return Err("IRLS produced a non-finite objective or weighted design".into());
        }
        let mut candidate = if l1 == 0.0 && !constrained {
            // Normalize working mass so the shared solver's rank threshold
            // stays relative even when separation makes Fisher weights tiny.
            let mass = gram.ws;
            if !(mass > 0.0 && mass.is_finite()) {
                return Err("IRLS produced a non-finite objective or weighted design".into());
            }
            for value in gram.xtx.iter_mut().chain(&mut gram.xty) {
                *value /= mass;
            }
            gram.yty /= mass;
            fit_linear(&gram, chosen, with_indicators, mono, 0.0, l2 / mass)
        } else {
            beta[0] = current.intercept;
            beta[1..1 + nc].copy_from_slice(&current.coef);
            if with_indicators {
                beta[1 + nc..].copy_from_slice(&current.na_coef);
            }
            let mut converged = false;
            for _ in 0..2000 {
                let mut delta = 0.0_f64;
                let mut magnitude = 1.0_f64;
                for j in 0..dim {
                    let ridge = if j == 0 {
                        0.0
                    } else {
                        l2 * scales[j] * scales[j]
                    };
                    let diagonal = gram.xtx[j * dim + j] + ridge;
                    if diagonal <= 0.0 {
                        beta[j] = 0.0;
                        continue;
                    }
                    let mut partial = gram.xty[j];
                    for k in 0..dim {
                        if j != k {
                            partial -= gram.xtx[j * dim + k] * beta[k];
                        }
                    }
                    let threshold = if j == 0 { 0.0 } else { l1 * scales[j] };
                    let mut next =
                        partial.signum() * (partial.abs() - threshold).max(0.0) / diagonal;
                    if j > 0 && j <= nc {
                        match mono.get(chosen[j - 1]).copied().unwrap_or(0) {
                            1 => next = next.max(0.0),
                            -1 => next = next.min(0.0),
                            _ => {}
                        }
                    }
                    delta = delta.max((next - beta[j]).abs() * scales[j]);
                    magnitude = magnitude.max(next.abs() * scales[j]);
                    beta[j] = next;
                }
                if delta <= 1e-10 * magnitude {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err("IRLS penalized least-squares step did not converge".into());
            }
            LinearModel {
                cols: chosen.to_vec(),
                coef: beta[1..1 + nc].to_vec(),
                na_coef: if with_indicators {
                    beta[1 + nc..].to_vec()
                } else {
                    Vec::new()
                },
                feature_means: Vec::new(),
                intercept: beta[0],
                summary: None,
            }
        };
        if !candidate.intercept.is_finite()
            || candidate
                .coef
                .iter()
                .chain(&candidate.na_coef)
                .any(|b| !b.is_finite())
        {
            return Err("IRLS produced non-finite coefficients".into());
        }
        let mut next_objective = objective(&candidate, &mut row);
        for _ in 0..20 {
            if next_objective.is_finite() && next_objective <= old_objective {
                break;
            }
            candidate.intercept = 0.5 * (candidate.intercept + current.intercept);
            for (new, old) in candidate.coef.iter_mut().zip(&current.coef) {
                *new = 0.5 * (*new + old);
            }
            for (new, old) in candidate.na_coef.iter_mut().zip(&current.na_coef) {
                *new = 0.5 * (*new + old);
            }
            next_objective = objective(&candidate, &mut row);
        }
        if !next_objective.is_finite() || next_objective > old_objective {
            return Ok((
                current,
                crate::solver::Convergence {
                    iterations: iter,
                    reason: crate::solver::StopReason::LineSearchFailed,
                },
            ));
        }
        let improvement = old_objective - next_objective;
        current = candidate;
        if verbose >= 3 {
            println!(
                "      [glm iter {}] obj={:.6} imp={:.6e}",
                iter, next_objective, improvement
            );
        }
        if improvement <= tol * (1.0 + old_objective) {
            if verbose >= 3 {
                println!(
                    "      [glm converged] iters={} final_obj={:.6}",
                    iter + 1,
                    next_objective
                );
            }
            return Ok((current, crate::solver::Convergence::tolerance(iter + 1)));
        }
        old_objective = next_objective;
    }
    if verbose >= 3 {
        println!("      [glm did not converge after {max_iter} iters]");
    }
    Ok((current, crate::solver::Convergence::limit(max_iter)))
}

/// Per-node frame statistics for printing (rpart's `frame`): count, weighted
/// mean and residual SSE. O(rows) to accumulate - no digest.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FrameStats {
    pub n: usize,
    #[serde(with = "crate::persistence::float")]
    pub wsum: f64,
    #[serde(with = "crate::persistence::float")]
    pub mean: f64,
    #[serde(with = "crate::persistence::float")]
    pub sse: f64,
}

impl FrameStats {
    pub fn from_ty(y: &[f64], w: &[f64], rows: &[usize]) -> Self {
        let mut wsum = 0.0;
        let mut wy = 0.0;
        for &r in rows {
            if w[r] > 0.0 {
                wsum += w[r];
                wy += w[r] * y[r];
            }
        }
        let mean = wy / wsum.max(1e-300);
        let sse: f64 = rows
            .iter()
            .filter(|&&r| w[r] > 0.0)
            .map(|&r| w[r] * (y[r] - mean) * (y[r] - mean))
            .sum();
        Self {
            n: rows.iter().filter(|&&r| w[r] > 0.0).count(),
            wsum,
            mean,
            sse,
        }
    }
}

/// One design row from column-major storage: `[1, values (0 where missing),
/// NA indicators]`. Allocation-free; the indicator block is included only
/// when `with_indicators`.
pub fn design_row(
    out: &mut [f64],
    cols: &[Vec<Option<f64>>],
    r: usize,
    chosen: &[usize],
    with_indicators: bool,
) {
    design_row_imputed(out, cols, r, chosen, with_indicators, None)
}

/// One design row with optional feature mean imputation for missing values.
pub fn design_row_imputed(
    out: &mut [f64],
    cols: &[Vec<Option<f64>>],
    r: usize,
    chosen: &[usize],
    with_indicators: bool,
    impute_means: Option<&[f64]>,
) {
    let nc = chosen.len();
    for slot in out.iter_mut() {
        *slot = 0.0;
    }
    out[0] = 1.0;
    for (k, &c) in chosen.iter().enumerate() {
        match cols[c][r] {
            Some(v) => out[1 + k] = v,
            None => {
                if let Some(means) = impute_means {
                    out[1 + k] = means.get(k).copied().unwrap_or(0.0);
                }
                if with_indicators {
                    out[1 + nc + k] = 1.0;
                }
            }
        }
    }
}
