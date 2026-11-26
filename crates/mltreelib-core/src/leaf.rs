//! Leaves: what a leaf model is ([`LeafKind`]), which columns it may use
//! ([`LeafCols`]), and fitting / evaluating it on a node's rows.

use crate::family::LeafFamily;
use crate::model_tree::Grower;
use crate::node_stats::{
    design_row, fit_glm_imputed, fit_linear, Gram, HeteroscedasticModel, LeafModel, LeafStats,
    LinearModel, SoftmaxModel, SurvivalStats,
};
use crate::solver::generic::{Bounds, Objective, ObjectiveError, SmoothObjective};
use crate::stats::{huber_location, w_median, w_quantile, w_sd};

/// Which columns a leaf's linear model may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LeafCols {
    /// Every usable predictor column.
    All,
    /// Only the columns used in splits along the root→leaf path.
    PathUsed,
}

/// Missing value policy for leaf model fitting and prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LeafNaPolicy {
    /// Drop rows with NAs in chosen features from leaf linear fit.
    /// At prediction time, rows with NAs in active features fall back to 0D leaf center (Option A).
    #[default]
    DropFallback,
    /// Impute missing feature values with leaf baseline means/modes.
    /// If a feature is 100% missing in this leaf, drop it from the leaf fit (feature subsetting).
    /// At prediction time, impute missing features with stored leaf baselines (Option B).
    ImputeLocal,
    /// Expand design matrix with 1{NA} indicator coefficients (legacy).
    Indicator,
}

/// What the leaf model is.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LeafKind {
    /// Linear model over the chosen columns (the model tree).
    Linear,
    /// Plain mean — the CART/rpart leaf. The split criterion then *is*
    /// rpart's deviance criterion.
    Mean,
    /// Weighted median — L1-optimal constant, outlier-robust.
    Median,
    /// Weighted quantile at tau (0.5 = median; any tau in (0, 1)).
    Quantile(#[serde(with = "crate::persistence::float")] f64),
    /// Weighted expectile at tau (0.5 = mean; any tau in (0, 1)).
    Expectile(#[serde(with = "crate::persistence::float")] f64),
    /// Huber M-location (quadratic near the center, linear in the tails).
    Huber,
    /// Piecewise MARS splines (earth models fit at the leaf).
    Mars,
    /// 0D Class probability: P(Y=1) in [0, 1] for binary classification.
    Prob,
    /// 0D Odds ratio: p / (1 - p).
    Odds,
    /// 0D Log-odds (logit): ln(p / (1 - p)).
    LogOdds,
    /// Non-parametric K-Nearest Neighbors leaf model (Uniform, Distance, Exponential, Dual, Double/DW-KNN).
    Knn,
}

/// Optimization solver used for fitting leaf regression models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LeafSolver {
    /// Iteratively Reweighted Least Squares (IRLS) and exact Cholesky.
    Irls,
    /// Quasi-Newton L-BFGS for non-linear GLM deviance minimization.
    Lbfgs,
    /// Cyclic Coordinate Descent (CCD) for L1/L2 ElasticNet and monotone constraints.
    Ccd,
    /// Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    CmaEs,
    /// Success-History Based Adaptive Differential Evolution (SHADE).
    Shade,
    /// Separable Natural Evolution Strategies / Natural Gradient Learning (sNES).
    NaturalGradient,
    /// Real-Coded Genetic Algorithm with SBX crossover and polynomial mutation.
    GeneticAlgorithm,
}

/// Optimization problem declared by a leaf kind and response family.
///
/// Closed-form and internally fitted leaves still declare their path. This
/// keeps leaf construction independent from solver identity while making the
/// fitted objective visible in persisted models.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LeafProblemType {
    /// Missing diagnostic from snapshots written before problem diagnostics.
    #[default]
    Unknown,
    /// Penalized GLM deviance over an intercept and feature coefficients.
    PenalizedGlm(LeafFamily),
    /// Weighted absolute-deviation location objective.
    WeightedMedian(LeafFamily),
    /// Weighted quantile check-loss objective.
    WeightedQuantile {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        tau: f64,
    },
    /// Weighted Huber M-location objective.
    HuberLocation(LeafFamily),
    /// Multivariate Huber regression with tuning constant delta and robust scale sigma.
    HuberRegression {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        delta: f64,
        #[serde(with = "crate::persistence::float")]
        scale: f64,
    },
    /// Weighted expectile asymmetric squared loss objective.
    ExpectileLocation {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        tau: f64,
    },
    /// Multivariate expectile regression with target level tau.
    ExpectileRegression {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        tau: f64,
    },
    /// Negative binomial overdispersed count regression with dispersion alpha.
    NegativeBinomialGlm {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        alpha: f64,
    },
    /// Beta regression for unit-interval targets with precision phi.
    BetaGlm {
        family: LeafFamily,
        #[serde(with = "crate::persistence::float")]
        phi: f64,
    },
    /// Weighted mean computed without optimization.
    ClosedFormMean(LeafFamily),
    /// Probability, odds, or log-odds transform of the fitted mean.
    ClosedFormMeanTransform(LeafFamily),
    /// MARS basis expansion with internal fitting.
    InternalMars(LeafFamily),
    /// Neighbor index with distance-weighted response.
    Knn(LeafFamily),
    /// Multiclass GLM with internally fitted class models.
    MulticlassGlm(LeafFamily),
    /// Survival model with internal fitting.
    Survival(LeafFamily),
    /// Heteroscedastic location-scale model with internal fitting.
    Heteroscedastic(LeafFamily),
    /// Accelerated failure-time model with internal fitting.
    Aft(LeafFamily),
    /// Mixture cure-rate model with internal fitting.
    Cure(LeafFamily),
    /// Restricted cubic spline model with internal fitting.
    Spline(LeafFamily),
}

impl LeafProblemType {
    /// Resolves the declared problem for one leaf kind and response family.
    pub fn for_leaf(leaf: LeafKind, family: LeafFamily) -> Self {
        match leaf {
            LeafKind::Linear => match family {
                LeafFamily::Softmax => Self::MulticlassGlm(family),
                LeafFamily::Survival => Self::Survival(family),
                LeafFamily::Heteroscedastic => Self::Heteroscedastic(family),
                LeafFamily::Aft => Self::Aft(family),
                LeafFamily::Cure => Self::Cure(family),
                LeafFamily::Spline => Self::Spline(family),
                LeafFamily::Quantile => Self::WeightedQuantile { family, tau: 0.5 },
                LeafFamily::Huber => Self::HuberRegression { family, delta: 1.345, scale: 1.0 },
                LeafFamily::Expectile => Self::ExpectileRegression { family, tau: 0.5 },
                LeafFamily::NegativeBinomial => Self::NegativeBinomialGlm { family, alpha: 0.0 },
                LeafFamily::Beta => Self::BetaGlm { family, phi: 1.0 },
                _ => Self::PenalizedGlm(family),
            },
            LeafKind::Median => Self::WeightedMedian(family),
            LeafKind::Quantile(tau) => Self::WeightedQuantile { family, tau },
            LeafKind::Expectile(tau) => Self::ExpectileLocation { family, tau },
            LeafKind::Huber => Self::HuberLocation(family),
            LeafKind::Mean => Self::ClosedFormMean(family),
            LeafKind::Prob | LeafKind::Odds | LeafKind::LogOdds => {
                Self::ClosedFormMeanTransform(family)
            }
            LeafKind::Mars => Self::InternalMars(family),
            LeafKind::Knn => Self::Knn(family),
        }
    }
}

/// Diagnostic naming what happened to the requested leaf solver.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LeafSolverDiagnostic {
    /// Missing diagnostic from snapshots written before solver diagnostics.
    #[default]
    Unknown,
    /// Requested solver fitted the optimization problem.
    Applied(LeafSolver),
    /// Solver selection does not apply to this closed-form or internal path.
    NotApplicable(LeafSolver),
    /// Compatibility lookup rejected the solver before fitting.
    Rejected { solver: LeafSolver, reason: String },
    /// Compatible solver failed during fitting.
    Failed { solver: LeafSolver, reason: String },
}

/// Smooth Huber location objective exposed through the generic solver seam.
#[derive(Clone, Debug)]
pub struct LeafHuberObjective {
    values: Vec<f64>,
    weights: Vec<f64>,
    delta: f64,
    bounds: Bounds,
}

impl LeafHuberObjective {
    /// Builds a weighted Huber objective.
    pub fn new(values: Vec<f64>, weights: Vec<f64>, delta: f64) -> Result<Self, String> {
        if values.len() != weights.len() {
            return Err("Huber objective values and weights must have equal length".into());
        }
        if !delta.is_finite() || delta <= 0.0 {
            return Err("Huber delta must be finite and positive".into());
        }
        if values.iter().any(|v| !v.is_finite()) || weights.iter().any(|w| !w.is_finite()) {
            return Err("Huber objective values and weights must be finite".into());
        }
        let bounds = location_bounds(&values)?;
        Ok(Self {
            values,
            weights,
            delta,
            bounds,
        })
    }

    /// Huber transition point in response units.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    pub(crate) fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub(crate) fn curvature(&self, x: f64) -> f64 {
        self.values
            .iter()
            .zip(&self.weights)
            .map(|(&value, &weight)| {
                if (value - x).abs() <= self.delta {
                    weight
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl Objective for LeafHuberObjective {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        crate::solver::generic::require_dimension(1, x)?;
        let mut total = 0.0;
        for (&value, &weight) in self.values.iter().zip(&self.weights) {
            let residual = (value - x[0]).abs();
            let loss = if residual <= self.delta {
                0.5 * residual * residual
            } else {
                self.delta * (residual - 0.5 * self.delta)
            };
            total += weight * loss;
        }
        crate::solver::generic::require_finite(total)
    }
}

impl SmoothObjective for LeafHuberObjective {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        crate::solver::generic::require_dimension(1, x)?;
        let mut gradient = 0.0;
        for (&value, &weight) in self.values.iter().zip(&self.weights) {
            let residual = value - x[0];
            gradient += weight * residual.clamp(-self.delta, self.delta);
        }
        Ok(vec![gradient])
    }
}

/// Robust location objective exposed through the generic solver seam.
#[derive(Clone, Debug)]
pub enum LeafObjective {
    /// Weighted absolute deviation.
    WeightedMedian {
        values: Vec<f64>,
        weights: Vec<f64>,
        bounds: Bounds,
    },
    /// Weighted quantile check loss.
    WeightedQuantile {
        values: Vec<f64>,
        weights: Vec<f64>,
        tau: f64,
        bounds: Bounds,
    },
    /// Smooth Huber M-location loss.
    Huber(LeafHuberObjective),
}

impl LeafObjective {
    /// Builds a weighted absolute-deviation objective.
    pub fn median(values: Vec<f64>, weights: Vec<f64>) -> Result<Self, String> {
        validate_location_data(&values, &weights)?;
        let bounds = location_bounds(&values)?;
        Ok(Self::WeightedMedian {
            values,
            weights,
            bounds,
        })
    }

    /// Builds a weighted quantile objective.
    pub fn quantile(values: Vec<f64>, weights: Vec<f64>, tau: f64) -> Result<Self, String> {
        validate_location_data(&values, &weights)?;
        if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            return Err("Quantile tau must be finite and in [0, 1]".into());
        }
        let bounds = location_bounds(&values)?;
        Ok(Self::WeightedQuantile {
            values,
            weights,
            tau,
            bounds,
        })
    }

    /// Builds a Huber objective using the library's 1.345-scaled weighted
    /// standard-deviation transition point.
    pub fn huber(values: Vec<f64>, weights: Vec<f64>) -> Result<Self, String> {
        validate_location_data(&values, &weights)?;
        let delta = (1.345
            * w_sd(
                &values
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .collect::<Vec<_>>(),
            ))
        .max(1.0);
        Ok(Self::Huber(LeafHuberObjective::new(
            values, weights, delta,
        )?))
    }
}

impl Objective for LeafObjective {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        match self {
            Self::WeightedMedian { bounds, .. } | Self::WeightedQuantile { bounds, .. } => bounds,
            Self::Huber(objective) => objective.bounds(),
        }
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        match self {
            Self::WeightedMedian {
                values, weights, ..
            } => {
                crate::solver::generic::require_dimension(1, x)?;
                let total = values
                    .iter()
                    .zip(weights)
                    .map(|(&value, &weight)| weight * (value - x[0]).abs())
                    .sum::<f64>();
                crate::solver::generic::require_finite(total)
            }
            Self::WeightedQuantile {
                values,
                weights,
                tau,
                ..
            } => {
                crate::solver::generic::require_dimension(1, x)?;
                let total = values
                    .iter()
                    .zip(weights)
                    .map(|(&value, &weight)| {
                        let residual = value - x[0];
                        let loss = if residual >= 0.0 {
                            tau * residual
                        } else {
                            (1.0 - tau) * (-residual)
                        };
                        weight * loss
                    })
                    .sum::<f64>();
                crate::solver::generic::require_finite(total)
            }
            Self::Huber(objective) => objective.value(x),
        }
    }
}

fn validate_location_data(values: &[f64], weights: &[f64]) -> Result<(), String> {
    if values.is_empty() {
        return Err("Location objective requires at least one observation".into());
    }
    if values.len() != weights.len() {
        return Err("Location objective values and weights must have equal length".into());
    }
    if values.iter().any(|v| !v.is_finite()) || weights.iter().any(|w| !w.is_finite()) {
        return Err("Location objective values and weights must be finite".into());
    }
    Ok(())
}

fn location_bounds(values: &[f64]) -> Result<Bounds, String> {
    let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Bounds::new(vec![lower], vec![upper])
}

/// Parameter controls for fitting leaf models independently of tree topology.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafControl {
    /// Leaf model kind (Linear, Mean, Median, Quantile(t), Huber, Mars, Prob, Odds, LogOdds).
    pub leaf: LeafKind,
    /// Optimization solver (Irls, Lbfgs, Ccd).
    pub solver: LeafSolver,
    /// Distribution / link function for the leaf model.
    pub family: LeafFamily,
    /// Which predictor columns linear leaves may use (All vs PathUsed).
    pub leaf_cols: LeafCols,
    /// Ridge penalty on leaf coefficients (intercept unpenalized).
    pub l2: f64,
    /// Lasso penalty on leaf coefficients (intercept unpenalized).
    pub l1: f64,
    /// Keep each leaf's t-digest for quantile / interval prediction.
    pub leaf_digest: bool,
    /// Monotone constraints (-1, 0, +1) per feature column for linear leaf models.
    pub leaf_monotone: Vec<i8>,
    /// Legacy combined monotone constraints (-1, 0, +1).
    pub monotone: Vec<i8>,
    /// Specific column subset for leaf models. If Some, overrides leaf_cols policy.
    pub leaf_features: Option<Vec<usize>>,
    /// Family-specific extra parameters.
    pub extra: std::collections::HashMap<String, f64>,
    /// Optional event indicators (1=event, 0=censored) for survival leaves.
    pub events: Option<Vec<f64>>,
    /// Optional treatment indicators (1=treated, 0=control) for causal / uplift leaves.
    pub treatments: Option<Vec<f64>>,
    /// Number of distinct classes for softmax targets.
    pub n_classes: Option<usize>,
    /// Maximum degree of interaction for MARS leaves (default 1).
    pub mars_degree: usize,
    /// Maximum number of terms for MARS leaves (default 10).
    pub mars_terms: usize,
    /// GCV penalty per knot for MARS leaves (default 2.0).
    pub mars_penalty: f64,
    /// Power parameter for Tweedie deviance (default 1.5).
    pub tweedie_p: f64,
    /// Censoring threshold for Tobit response (default 0.0).
    pub tobit_threshold: f64,
    /// Number of nearest neighbors to query at leaf KNN inference (default 5).
    pub knn_k: usize,
    /// Secondary neighborhood size for precomputing neighbor reliability / compactness (default 5).
    pub knn_k_validity: usize,
    /// KNN weighting variant: Uniform, Distance, Exponential, Dual, Double (DW-KNN).
    pub knn_variant: crate::knn::KnnVariant,
    /// Distance metric for leaf KNN (Euclidean, Manhattan, Chebyshev, Gower, etc.).
    pub knn_metric: crate::knn::DistanceMetric,
    /// Decay scale gamma for exponential / double weighting (None => auto-scale by median distance).
    pub knn_gamma: Option<f64>,
    /// Power parameter p for inverse distance / Minkowski metric (default 2.0).
    pub knn_p: f64,
    /// Verbosity level for leaf fitting (0 = silent, >=3 logs GLM and leaf details).
    pub verbose: u8,
    /// Adaptive hierarchical leaf shrinkage penalty lambda (default 0.0 = off).
    pub shrinkage: f64,
    /// Maximum number of distinct integer classes for histogram binning (default 256).
    pub max_classes: usize,
    /// Missing value strategy at leaf level: DropFallback (default), ImputeLocal, or Indicator.
    pub leaf_na_policy: LeafNaPolicy,
    /// AFT parametric distribution (Weibull, LogNormal, LogLogistic).
    pub aft_dist: crate::node_stats::AftDist,
    /// Cure fraction for mixture cure rate leaves.
    pub cure_fraction: Option<f64>,
    /// Number of interior spline knots for Royston-Parmar leaves (default 3).
    pub spline_knots: usize,
    /// Target quantile tau in (0, 1) for quantile leaves (default 0.5).
    pub tau: f64,
    /// Huber tuning constant delta > 0 for Huber regression leaves (default 1.345).
    pub huber_delta: f64,
    /// Target expectile tau in (0, 1) for expectile regression leaves (default 0.5).
    pub expectile_tau: f64,
    /// Enforce non-negative coefficients (slopes >= 0) via Lawson-Hanson NNLS (default false).
    pub non_negative: bool,
    /// Enforce non-negative intercept in addition to slopes in NNLS (default false).
    pub non_negative_intercept: bool,
    /// Fixed negative binomial dispersion parameter alpha >= 0 (None = estimate from data).
    pub nb_alpha: Option<f64>,
    /// Fixed beta regression precision parameter phi > 0 (None = estimate from data).
    pub beta_phi: Option<f64>,
    /// Coefficient group definitions for Group Lasso and Sparse-Group Lasso leaves.
    pub groups: Option<Vec<crate::solver::groups::GroupSpec>>,
    /// Group L2 penalty multiplier for Group Lasso and Sparse-Group Lasso (default 0.0).
    pub group_l2: f64,
}

impl Default for LeafControl {
    fn default() -> Self {
        Self {
            leaf: LeafKind::Linear,
            solver: LeafSolver::Irls,
            family: LeafFamily::Gaussian,
            leaf_cols: LeafCols::All,
            l2: 0.5,
            l1: 0.0,
            leaf_digest: true,
            leaf_monotone: Vec::new(),
            monotone: Vec::new(),
            leaf_features: None,
            extra: std::collections::HashMap::new(),
            events: None,
            treatments: None,
            n_classes: None,
            mars_degree: 1,
            mars_terms: 10,
            mars_penalty: 2.0,
            tweedie_p: 1.5,
            tobit_threshold: 0.0,
            knn_k: 5,
            knn_k_validity: 5,
            knn_variant: crate::knn::KnnVariant::Uniform,
            knn_metric: crate::knn::DistanceMetric::Euclidean,
            knn_gamma: None,
            knn_p: 2.0,
            verbose: 0,
            shrinkage: 0.0,
            max_classes: 256,
            leaf_na_policy: LeafNaPolicy::Indicator,
            aft_dist: crate::node_stats::AftDist::Weibull,
            cure_fraction: None,
            spline_knots: 3,
            tau: 0.5,
            huber_delta: 1.345,
            expectile_tau: 0.5,
            non_negative: false,
            non_negative_intercept: false,
            nb_alpha: None,
            beta_phi: None,
            groups: None,
            group_l2: 0.0,
        }
    }
}

impl From<&crate::model_tree::ModelTreeParams> for LeafControl {
    fn from(p: &crate::model_tree::ModelTreeParams) -> Self {
        let mut extra = std::collections::HashMap::new();
        extra.insert("tweedie_p".to_string(), p.tweedie_p);
        extra.insert("tobit_threshold".to_string(), p.tobit_threshold);
        extra.insert("mars_degree".to_string(), p.mars_degree as f64);
        extra.insert("mars_terms".to_string(), p.mars_terms as f64);
        extra.insert("mars_penalty".to_string(), p.mars_penalty);
        extra.insert("knn_k".to_string(), p.knn_k as f64);
        extra.insert("knn_k_validity".to_string(), p.knn_k_validity as f64);
        extra.insert("knn_p".to_string(), p.knn_p);
        if let Some(g) = p.knn_gamma {
            extra.insert("knn_gamma".to_string(), g);
        }
        let tau = match p.leaf {
            LeafKind::Quantile(t) => t,
            _ => p.tau,
        };
        extra.insert("tau".to_string(), tau);
        extra.insert("huber_delta".to_string(), p.huber_delta);
        let exp_tau = match p.leaf {
            LeafKind::Expectile(t) => t,
            _ => p.expectile_tau,
        };
        extra.insert("expectile_tau".to_string(), exp_tau);
        if p.non_negative {
            extra.insert("non_negative".to_string(), 1.0);
        }
        if p.non_negative_intercept {
            extra.insert("non_negative_intercept".to_string(), 1.0);
        }
        if let Some(alpha) = p.nb_alpha {
            extra.insert("nb_alpha".to_string(), alpha);
        }
        if let Some(phi) = p.beta_phi {
            extra.insert("beta_phi".to_string(), phi);
        }
        let leaf_monotone = if !p.leaf_monotone.is_empty() {
            p.leaf_monotone.clone()
        } else {
            p.monotone.clone()
        };
        Self {
            leaf: p.leaf,
            solver: p.leaf_solver,
            family: p.leaf_family,
            leaf_cols: p.leaf_cols,
            l2: p.l2,
            l1: p.l1,
            leaf_digest: p.leaf_digest,
            leaf_monotone,
            monotone: p.monotone.clone(),
            leaf_features: None,
            extra,
            events: p.events.clone(),
            treatments: p.treatments.clone(),
            n_classes: p.n_classes,
            mars_degree: p.mars_degree,
            mars_terms: p.mars_terms,
            mars_penalty: p.mars_penalty,
            tweedie_p: p.tweedie_p,
            tobit_threshold: p.tobit_threshold,
            knn_k: p.knn_k,
            knn_k_validity: p.knn_k_validity,
            knn_variant: p.knn_variant,
            knn_metric: p.knn_metric,
            knn_gamma: p.knn_gamma,
            knn_p: p.knn_p,
            verbose: p.verbose,
            shrinkage: p.shrinkage,
            max_classes: p.max_classes,
            leaf_na_policy: p.leaf_na_policy,
            aft_dist: p.aft_dist,
            cure_fraction: p.cure_fraction,
            spline_knots: p.spline_knots,
            tau,
            huber_delta: p.huber_delta,
            expectile_tau: exp_tau,
            non_negative: p.non_negative,
            non_negative_intercept: p.non_negative_intercept,
            nb_alpha: p.nb_alpha,
            beta_phi: p.beta_phi,
            groups: p.groups.clone(),
            group_l2: p.group_l2,
        }
    }
}

impl crate::model_tree::ModelTreeParams {
    /// Update leaf fitting options without changing growth configuration.
    pub fn apply_leaf_control(&mut self, control: LeafControl) {
        self.leaf = control.leaf;
        self.leaf_solver = control.solver;
        self.leaf_family = control.family;
        self.leaf_cols = control.leaf_cols;
        self.l1 = control.l1;
        self.l2 = control.l2;
        self.leaf_digest = control.leaf_digest;
        self.leaf_monotone = control.leaf_monotone;
        self.events = control.events;
        self.treatments = control.treatments;
        self.mars_degree = control.mars_degree;
        self.mars_terms = control.mars_terms;
        self.mars_penalty = control.mars_penalty;
        self.tweedie_p = control.tweedie_p;
        self.tobit_threshold = control.tobit_threshold;
        self.knn_k = control.knn_k;
        self.knn_k_validity = control.knn_k_validity;
        self.knn_variant = control.knn_variant;
        self.knn_metric = control.knn_metric;
        self.knn_gamma = control.knn_gamma;
        self.knn_p = control.knn_p;
        self.shrinkage = control.shrinkage;
        self.max_classes = control.max_classes;
        self.leaf_na_policy = control.leaf_na_policy;
        self.aft_dist = control.aft_dist;
        self.cure_fraction = control.cure_fraction;
        self.spline_knots = control.spline_knots;
        self.tau = control.tau;
        self.huber_delta = control.huber_delta;
        self.groups = control.groups;
        self.group_l2 = control.group_l2;
    }
}

impl LeafControl {
    /// Check that an explicitly selected optimizer applies to this leaf.
    pub fn validate_solver(&self) -> Result<(), String> {
        crate::solver::check_leaf_solver_compatibility(
            self.leaf,
            self.family,
            self.solver,
            crate::solver::LeafSolverRequirements {
                l1: self.l1 != 0.0,
                monotone: self.effective_leaf_monotone().iter().any(|&sign| sign != 0),
            },
        )
        .map(|_| ())
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn effective_leaf_monotone(&self) -> &[i8] {
        if !self.leaf_monotone.is_empty() {
            &self.leaf_monotone
        } else {
            &self.monotone
        }
    }

    pub fn with_leaf(mut self, leaf: LeafKind) -> Self {
        self.leaf = leaf;
        self
    }

    pub fn with_family(mut self, family: LeafFamily) -> Self {
        self.family = family;
        self
    }

    pub fn with_leaf_cols(mut self, leaf_cols: LeafCols) -> Self {
        self.leaf_cols = leaf_cols;
        self
    }

    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2;
        self
    }

    pub fn with_l1(mut self, l1: f64) -> Self {
        self.l1 = l1;
        self
    }

    pub fn with_digest(mut self, leaf_digest: bool) -> Self {
        self.leaf_digest = leaf_digest;
        self
    }

    pub fn with_monotone(mut self, monotone: Vec<i8>) -> Self {
        self.monotone = monotone;
        self
    }

    pub fn with_leaf_features(mut self, features: Vec<usize>) -> Self {
        self.leaf_features = Some(features);
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, val: f64) -> Self {
        self.extra.insert(key.into(), val);
        self
    }

    pub fn with_leaf_na_policy(mut self, leaf_na_policy: LeafNaPolicy) -> Self {
        self.leaf_na_policy = leaf_na_policy;
        self
    }
}

/// Helper to select active columns for a leaf under the given column policy.
pub(crate) fn chosen_cols_policy(
    leaf_cols: LeafCols,
    leaf_features: Option<&[usize]>,
    active: &[usize],
    path: &[usize],
) -> Vec<usize> {
    if let Some(feats) = leaf_features {
        let mut p: Vec<usize> = feats
            .iter()
            .copied()
            .filter(|c| active.contains(c))
            .collect();
        p.sort_unstable();
        p.dedup();
        p
    } else {
        match leaf_cols {
            LeafCols::All => active.to_vec(),
            LeafCols::PathUsed => {
                let mut p: Vec<usize> = path
                    .iter()
                    .copied()
                    .filter(|c| active.contains(c))
                    .collect();
                p.sort_unstable();
                p.dedup();
                p
            }
        }
    }
}

/// Helper to determine whether any missing values exist and total design dimension.
pub(crate) fn compute_leaf_design(
    cols: &[Vec<Option<f64>>],
    rows: &[usize],
    chosen: &[usize],
) -> (bool, usize) {
    let any_missing = chosen
        .iter()
        .any(|&c| rows.iter().any(|&r| cols[c][r].is_none()));
    let dim = 1 + chosen.len() + if any_missing { chosen.len() } else { 0 };
    (any_missing, dim)
}

pub(crate) struct LeafDesignContext {
    pub chosen: Vec<usize>,
    pub rows: Vec<usize>,
    pub impute_means: Option<Vec<f64>>,
    pub with_indicators: bool,
    pub valid: bool,
}

pub(crate) fn prepare_leaf_design(
    cols: &[Vec<Option<f64>>],
    w: &[f64],
    rows: &[usize],
    path: &[usize],
    active: &[usize],
    control: &LeafControl,
) -> LeafDesignContext {
    let chosen = chosen_cols_policy(
        control.leaf_cols,
        control.leaf_features.as_deref(),
        active,
        path,
    );
    if chosen.is_empty() {
        return LeafDesignContext {
            chosen,
            rows: rows.to_vec(),
            impute_means: None,
            with_indicators: false,
            valid: false,
        };
    }
    match control.leaf_na_policy {
        LeafNaPolicy::DropFallback => {
            let complete_rows: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&r| chosen.iter().all(|&c| cols[c][r].is_some()))
                .collect();
            if complete_rows.is_empty() || complete_rows.len() < chosen.len() + 1 {
                LeafDesignContext {
                    chosen,
                    rows: rows.to_vec(),
                    impute_means: None,
                    with_indicators: false,
                    valid: false,
                }
            } else {
                LeafDesignContext {
                    chosen,
                    rows: complete_rows,
                    impute_means: None,
                    with_indicators: false,
                    valid: true,
                }
            }
        }
        LeafNaPolicy::ImputeLocal => {
            let mut valid_chosen = Vec::with_capacity(chosen.len());
            let mut means = Vec::with_capacity(chosen.len());
            for &c in &chosen {
                let mut sum = 0.0;
                let mut count = 0.0;
                for &r in rows {
                    if let Some(v) = cols[c][r] {
                        let wt = w[r];
                        if wt > 0.0 {
                            sum += v * wt;
                            count += wt;
                        }
                    }
                }
                if count > 0.0 {
                    valid_chosen.push(c);
                    means.push(sum / count);
                } else if control.verbose >= 1 {
                    eprintln!(
                        "Leaf: column {} is 100% NA in this node; dropped from leaf linear model",
                        c
                    );
                }
            }
            if valid_chosen.is_empty() {
                LeafDesignContext {
                    chosen: valid_chosen,
                    rows: rows.to_vec(),
                    impute_means: None,
                    with_indicators: false,
                    valid: false,
                }
            } else {
                LeafDesignContext {
                    chosen: valid_chosen,
                    rows: rows.to_vec(),
                    impute_means: Some(means),
                    with_indicators: false,
                    valid: true,
                }
            }
        }
        LeafNaPolicy::Indicator => {
            let (any_missing, _) = compute_leaf_design(cols, rows, &chosen);
            LeafDesignContext {
                chosen,
                rows: rows.to_vec(),
                impute_means: None,
                with_indicators: any_missing,
                valid: true,
            }
        }
    }
}

/// Fit a leaf model on the specified rows and columns using `control`.
pub fn fit_leaf_model(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    path: &[usize],
    active: &[usize],
    control: &LeafControl,
) -> LeafModel {
    let stats = if control.leaf_digest {
        let yv: Vec<f64> = rows.iter().map(|&r| y[r]).collect();
        let wv: Vec<f64> = rows.iter().map(|&r| w[r]).collect();
        LeafStats::from_targets(&yv, &wv)
    } else {
        LeafStats::from_rows(y, w, rows)
    };
    let causal = if let Some(treat) = control.treatments.as_ref() {
        let mut n_treated = 0usize;
        let mut n_control = 0usize;
        let mut w_treated = 0.0;
        let mut w_control = 0.0;
        let mut swy_treated = 0.0;
        let mut swy_control = 0.0;
        for &r in rows {
            let wt = w[r];
            if wt <= 0.0 {
                continue;
            }
            let t = treat.get(r).copied().unwrap_or(0.0);
            let y_val = y[r];
            if t > 0.0 {
                n_treated += 1;
                w_treated += wt;
                swy_treated += wt * y_val;
            } else {
                n_control += 1;
                w_control += wt;
                swy_control += wt * y_val;
            }
        }
        let mean_treated = if w_treated > 0.0 {
            swy_treated / w_treated
        } else {
            0.0
        };
        let mean_control = if w_control > 0.0 {
            swy_control / w_control
        } else {
            0.0
        };
        let tau = mean_treated - mean_control;
        Some(crate::node_stats::CausalStats {
            tau,
            mean_treated,
            mean_control,
            n_treated,
            n_control,
            w_treated,
            w_control,
        })
    } else {
        None
    };
    let family = control.family;
    let leaf_problem_type = LeafProblemType::for_leaf(control.leaf, family);
    let mut solver_used = crate::solver::leaf_solver_diagnostic(
        control.leaf,
        family,
        control.solver,
        crate::solver::LeafSolverRequirements {
            l1: control.l1 != 0.0,
            monotone: control
                .effective_leaf_monotone()
                .iter()
                .any(|&sign| sign != 0),
        },
    );
    if family == LeafFamily::Softmax {
        let n_classes = control.n_classes.unwrap_or_else(|| {
            rows.iter()
                .filter(|&&r| w[r] > 0.0)
                .map(|&r| (y[r] as usize) + 1)
                .max()
                .unwrap_or(2)
                .max(2)
        });
        let mut probs = vec![0.0; n_classes];
        let mut total_w = 0.0;
        for &r in rows {
            if w[r] > 0.0 {
                let c = y[r] as usize;
                if c < n_classes {
                    probs[c] += w[r];
                    total_w += w[r];
                }
            }
        }
        if total_w > 0.0 {
            for p in &mut probs {
                *p /= total_w;
            }
        } else {
            probs.fill(1.0 / n_classes as f64);
        }
        let best_c = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c as f64)
            .unwrap_or(0.0);

        let linear_models = if control.leaf == LeafKind::Linear {
            let design_ctx = prepare_leaf_design(cols, w, rows, path, active, control);
            if design_ctx.valid {
                let mut models = Vec::with_capacity(n_classes);
                for c in 0..n_classes {
                    let mut binary_y: Vec<f64> = Vec::with_capacity(y.len());
                    for &val in y.iter() {
                        binary_y.push(if val as usize == c { 1.0 } else { 0.0 });
                    }
                    let mean_c = probs[c].clamp(1e-6, 1.0 - 1e-6);
                    let fitted = fit_glm_imputed(
                        cols,
                        &binary_y,
                        w,
                        &design_ctx.rows,
                        &design_ctx.chosen,
                        design_ctx.with_indicators,
                        design_ctx.impute_means.as_deref(),
                        &control.monotone,
                        control.l1,
                        control.l2,
                        LeafFamily::Binomial,
                        mean_c,
                        Some(&control.extra),
                        control.verbose,
                    );
                    let f_means = design_ctx
                        .impute_means
                        .as_ref()
                        .map(|m| m.iter().map(|&v| Some(v)).collect())
                        .unwrap_or_default();
                    let lm = fitted.unwrap_or_else(|| LinearModel {
                        coef: vec![0.0; design_ctx.chosen.len()],
                        na_coef: if design_ctx.with_indicators {
                            vec![0.0; design_ctx.chosen.len()]
                        } else {
                            Vec::new()
                        },
                        feature_means: f_means,
                        cols: design_ctx.chosen.clone(),
                        intercept: mean_c.ln() - (1.0 - mean_c).ln(),
                        summary: None,
                    });
                    models.push(lm);
                }
                Some(models)
            } else {
                None
            }
        } else {
            None
        };

        return LeafModel {
            stats,
            linear: None,
            center: Some(best_c),
            family,
            softmax: Some(SoftmaxModel {
                n_classes,
                probs,
                linear_models,
            }),
            survival: None,
            heteroscedastic: None,
            causal,
            aft: None,
            cure: None,
            spline: None,
            mars: None,
            mars_cols: None,
            knn: None,
            knn_cols: None,
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }

    if family == LeafFamily::Survival
        || family == LeafFamily::Aft
        || family == LeafFamily::Cure
        || family == LeafFamily::Spline
    {
        let events = control.events.as_ref();
        let mut records: Vec<(f64, f64, f64)> = rows
            .iter()
            .filter(|&&r| w[r] > 0.0 && y[r] >= 0.0 && y[r].is_finite())
            .map(|&r| {
                let e = events.and_then(|evs| evs.get(r)).copied().unwrap_or(1.0);
                (y[r], e, w[r])
            })
            .collect();
        records.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut distinct_event_times: Vec<f64> = Vec::new();
        for &(t, e, _) in &records {
            if e > 0.0
                && (distinct_event_times.is_empty() || *distinct_event_times.last().unwrap() < t)
            {
                distinct_event_times.push(t);
            }
        }

        let mut causes: Vec<usize> = Vec::new();
        for &(_, e, _) in &records {
            let cause_id = e.round() as usize;
            if cause_id > 0 && !causes.contains(&cause_id) {
                causes.push(cause_id);
            }
        }
        causes.sort_unstable();

        let mut cum_hazard = Vec::with_capacity(distinct_event_times.len());
        let mut survival = Vec::with_capacity(distinct_event_times.len());
        let mut cif: Vec<Vec<f64>> =
            vec![Vec::with_capacity(distinct_event_times.len()); causes.len()];
        let mut cur_cif: Vec<f64> = vec![0.0; causes.len()];
        let mut current_h = 0.0;
        let mut current_s = 1.0;
        let mut median_time = records.last().map(|r| r.0).unwrap_or(0.0);
        let mut found_median = false;

        for &tj in &distinct_event_times {
            let mut dj = 0.0;
            let mut nj = 0.0;
            let mut dj_cause = vec![0.0; causes.len()];
            for &(ti, ei, wi) in &records {
                if ti >= tj {
                    nj += wi;
                    if (ti - tj).abs() < 1e-9 && ei > 0.0 {
                        dj += wi;
                        let cid = ei.round() as usize;
                        if let Some(pos) = causes.iter().position(|&c| c == cid) {
                            dj_cause[pos] += wi;
                        }
                    }
                }
            }
            let s_prev = current_s;
            if nj > 0.0 {
                let hj = dj / nj;
                current_h += hj;
                current_s *= (1.0 - hj).clamp(0.0, 1.0);
                for c_idx in 0..causes.len() {
                    let d_c = dj_cause[c_idx];
                    if d_c > 0.0 {
                        cur_cif[c_idx] += s_prev * (d_c / nj);
                    }
                }
            }
            cum_hazard.push(current_h);
            survival.push(current_s);
            for c_idx in 0..causes.len() {
                cif[c_idx].push(cur_cif[c_idx]);
            }
            if !found_median && current_s <= 0.5 {
                median_time = tj;
                found_median = true;
            }
        }

        let mut event_counts = vec![0usize; causes.len()];
        for (c_idx, &c) in causes.iter().enumerate() {
            event_counts[c_idx] = records
                .iter()
                .filter(|&&(_, e, _)| (e.round() as usize) == c)
                .count();
        }

        let surv_stats = SurvivalStats {
            times: distinct_event_times.clone(),
            cum_hazard,
            survival,
            median_time,
            causes,
            cif,
            event_counts,
        };

        // 1. Parametric AFT Model fitting
        let mut aft = None;
        if family == LeafFamily::Aft {
            let dist = control.aft_dist;
            let chosen = chosen_cols_policy(
                control.leaf_cols,
                control.leaf_features.as_deref(),
                active,
                path,
            );
            let p = chosen.len();
            let log_times: Vec<f64> = records.iter().map(|r| r.0.max(1e-6).ln()).collect();
            let mut alpha = 0.0;
            let mut beta = vec![0.0; p];
            let mut sum_w = 0.0;
            for (idx, r) in records.iter().enumerate() {
                alpha += r.2 * log_times[idx];
                sum_w += r.2;
            }
            if sum_w > 0.0 {
                alpha /= sum_w;
            }
            if control.leaf == LeafKind::Linear && p > 0 {
                let (any_missing, dim) = compute_leaf_design(cols, rows, &chosen);
                let mut g = Gram::new(dim);
                let mut row = vec![0.0; dim];
                for (rec_idx, &r) in rows.iter().enumerate() {
                    design_row(&mut row, cols, r, &chosen, any_missing);
                    let target = log_times[rec_idx.min(log_times.len() - 1)];
                    g.add_row(&row, target, w[r]);
                }
                let lin = fit_linear(
                    &g,
                    &chosen,
                    any_missing,
                    control.effective_leaf_monotone(),
                    control.l1,
                    control.l2,
                );
                alpha = lin.intercept;
                beta = lin.coef;
            }
            let mut res_sq = 0.0;
            for (rec_idx, &r) in rows.iter().enumerate() {
                let mut pred = alpha;
                for j in 0..p {
                    let c = chosen[j];
                    let v = cols[c][r].unwrap_or(0.0);
                    pred += beta[j] * v;
                }
                let diff = log_times[rec_idx.min(log_times.len() - 1)] - pred;
                res_sq += w[r] * diff * diff;
            }
            let scale = if sum_w > 0.0 {
                (res_sq / sum_w).sqrt().max(0.05)
            } else {
                1.0
            };
            aft = Some(crate::node_stats::AftModel {
                dist,
                intercept: alpha,
                coef: beta,
                cols: chosen,
                scale,
            });
        }

        // 2. Mixture Cure Rate model fitting
        let mut cure = None;
        if family == LeafFamily::Cure {
            let cure_rate = control
                .cure_fraction
                .unwrap_or_else(|| current_s.clamp(0.0, 0.99));
            cure = Some(crate::node_stats::CureModel {
                cure_rate,
                aft: None,
                survival: Some(surv_stats.clone()),
            });
        }

        // 3. Royston-Parmar Restricted Cubic Spline fitting
        let mut spline = None;
        if family == LeafFamily::Spline {
            let chosen = chosen_cols_policy(
                control.leaf_cols,
                control.leaf_features.as_deref(),
                active,
                path,
            );
            let mut knots = Vec::new();
            if !distinct_event_times.is_empty() {
                let n_knots = control.spline_knots.max(3);
                for k in 0..n_knots {
                    let q = (k as f64 + 0.5) / n_knots as f64;
                    let idx = ((distinct_event_times.len() - 1) as f64 * q).round() as usize;
                    knots.push(distinct_event_times[idx].max(1e-6).ln());
                }
            }
            let mut gamma = vec![0.0; knots.len().max(2)];
            if !distinct_event_times.is_empty() {
                gamma[0] = -1.0;
                gamma[1] = 1.0;
            }
            let coef = vec![0.0; chosen.len()];
            spline = Some(crate::node_stats::SplineSurvivalModel {
                knots,
                gamma,
                coef,
                cols: chosen,
            });
        }

        return LeafModel {
            stats,
            linear: None,
            center: Some(median_time),
            family,
            softmax: None,
            survival: Some(surv_stats),
            heteroscedastic: None,
            causal,
            aft,
            cure,
            spline,
            mars: None,
            mars_cols: None,
            knn: None,
            knn_cols: None,
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }

    if family == LeafFamily::Heteroscedastic {
        let (linear, scale_linear, scale) = if control.leaf == LeafKind::Linear {
            let chosen = chosen_cols_policy(
                control.leaf_cols,
                control.leaf_features.as_deref(),
                active,
                path,
            );
            let (any_missing, dim) = compute_leaf_design(cols, rows, &chosen);
            let mut g = Gram::new(dim);
            let mut row = vec![0.0; dim];
            for &r in rows {
                design_row(&mut row, cols, r, &chosen, any_missing);
                g.add_row(&row, y[r], w[r]);
            }
            let loc_model = fit_linear(
                &g,
                &chosen,
                any_missing,
                control.effective_leaf_monotone(),
                control.l1,
                control.l2,
            );

            let mut g_scale = Gram::new(dim);
            let mut sw = 0.0;
            let mut s_res2 = 0.0;
            for &r in rows {
                if w[r] <= 0.0 {
                    continue;
                }
                design_row(&mut row, cols, r, &chosen, any_missing);
                let pred = loc_model.eval_design(&row);
                let res2 = (y[r] - pred).powi(2);
                sw += w[r];
                s_res2 += w[r] * res2;
                let log_res2 = (res2 + 1e-8).ln();
                g_scale.add_row(&row, log_res2, w[r]);
            }
            let mut s_model = fit_linear(
                &g_scale,
                &chosen,
                any_missing,
                &vec![0; chosen.len()],
                control.l1,
                control.l2,
            );
            // Harvey (1976) bias correction for log chi-squared(1) residuals:
            // E[ln(e_i^2)] = ln(sigma_i^2) + E[ln(chi_1^2)] where E[ln(chi_1^2)] = -1.2703628.
            // Adding 1.2703628 to the intercept makes exp(0.5 * z'gamma) unbiased for sigma.
            s_model.intercept += 1.2703628;
            let raw_scale = if sw > 0.0 {
                (s_res2 / sw).sqrt().max(1e-6)
            } else {
                1.0
            };
            (Some(loc_model), Some(s_model), raw_scale)
        } else {
            let mut sw = 0.0;
            let mut s_res2 = 0.0;
            for &r in rows {
                if w[r] > 0.0 {
                    sw += w[r];
                    s_res2 += w[r] * (y[r] - stats.mean).powi(2);
                }
            }
            let raw_scale = if sw > 0.0 {
                (s_res2 / sw).sqrt().max(1e-6)
            } else {
                stats.std.max(1e-6)
            };
            (None, None, raw_scale)
        };

        let mean_val = stats.mean;
        return LeafModel {
            stats,
            linear,
            center: Some(mean_val),
            family,
            softmax: None,
            survival: None,
            heteroscedastic: Some(HeteroscedasticModel {
                scale,
                scale_linear,
            }),
            causal,
            aft: None,
            cure: None,
            spline: None,
            mars: None,
            mars_cols: None,
            knn: None,
            knn_cols: None,
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }

    if control.leaf == LeafKind::Mars {
        let chosen = chosen_cols_policy(
            control.leaf_cols,
            control.leaf_features.as_deref(),
            active,
            path,
        );
        let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
        let mut y_vec: Vec<f64> = Vec::with_capacity(rows.len());
        let mut w_vec: Vec<f64> = Vec::with_capacity(rows.len());
        for &r in rows {
            let mut row_vals = Vec::with_capacity(chosen.len());
            for &c in &chosen {
                row_vals.push(cols[c][r].unwrap_or(0.0));
            }
            x_mat.push(row_vals);
            y_vec.push(y[r]);
            w_vec.push(w[r]);
        }
        let mars_degree = control
            .extra
            .get("mars_degree")
            .map(|&d| d as usize)
            .unwrap_or(control.mars_degree);
        let mars_terms = control
            .extra
            .get("mars_terms")
            .map(|&t| t as usize)
            .unwrap_or(control.mars_terms);
        let mars_penalty = control
            .extra
            .get("mars_penalty")
            .copied()
            .unwrap_or(control.mars_penalty);
        let mut reg = crate::mars::MarsRegressor::new(mars_degree, mars_terms, mars_penalty);
        if family == LeafFamily::Binomial {
            reg.params.glm = crate::mars::MarsGlmFamily::Binomial;
        } else if family == LeafFamily::Poisson {
            reg.params.glm = crate::mars::MarsGlmFamily::Poisson;
        }
        let _ = reg.fit_weighted(&x_mat, &y_vec, Some(&w_vec));
        return LeafModel {
            stats,
            linear: None,
            center: None,
            family,
            softmax: None,
            survival: None,
            heteroscedastic: None,
            causal,
            aft: None,
            cure: None,
            spline: None,
            mars: reg.model,
            mars_cols: Some(chosen),
            knn: None,
            knn_cols: None,
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }

    if control.leaf == LeafKind::Knn {
        let chosen = chosen_cols_policy(
            control.leaf_cols,
            control.leaf_features.as_deref(),
            active,
            path,
        );
        let n_classes = control.n_classes.unwrap_or(2);
        let knn_model = crate::knn::LeafKnnModel::fit(
            cols,
            y,
            w,
            rows,
            chosen.clone(),
            control.knn_k,
            control.knn_k_validity,
            control.knn_variant,
            control.knn_metric,
            control.knn_gamma,
            control.knn_p,
            family,
            n_classes,
        );
        return LeafModel {
            stats,
            linear: None,
            center: Some(knn_model.fallback_mean),
            family,
            softmax: None,
            survival: None,
            heteroscedastic: None,
            causal,
            aft: None,
            cure: None,
            spline: None,
            mars: None,
            mars_cols: None,
            knn: Some(knn_model),
            knn_cols: Some(chosen),
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }

    if family != LeafFamily::Gaussian
        && family != LeafFamily::Quantile
        && (control.leaf == LeafKind::Mean
            || ((family == LeafFamily::Binomial || family == LeafFamily::Probit)
                && (stats.mean == 0.0 || stats.mean == 1.0))
            || ((family == LeafFamily::Poisson || family == LeafFamily::Tweedie)
                && stats.mean == 0.0)
            || (family == LeafFamily::Tobit && stats.mean <= 0.0))
    {
        return LeafModel {
            center: Some(stats.mean),
            stats,
            linear: None,
            family,
            softmax: None,
            survival: None,
            heteroscedastic: None,
            causal,
            aft: None,
            cure: None,
            spline: None,
            mars: None,
            mars_cols: None,
            knn: None,
            knn_cols: None,
            vector_mean: None,
            solver_used: solver_used.clone(),
            leaf_problem_type: leaf_problem_type.clone(),
        };
    }
    let linear = if control.leaf == LeafKind::Linear {
        let design_ctx = prepare_leaf_design(cols, w, rows, path, active, control);
        if !design_ctx.valid {
            None
        } else {
            let chosen = design_ctx.chosen;
            let actual_rows = design_ctx.rows;
            let any_missing = design_ctx.with_indicators;
            let impute_means = design_ctx.impute_means;
            let f_means: Vec<Option<f64>> = impute_means
                .as_ref()
                .map(|m| m.iter().map(|&v| Some(v)).collect())
                .unwrap_or_default();
            let is_special_linear = family == LeafFamily::Quantile
                || family == LeafFamily::Huber
                || family == LeafFamily::Expectile
                || family == LeafFamily::NegativeBinomial
                || family == LeafFamily::Beta;
            let lm = if is_special_linear && actual_rows.len() < chosen.len() + 1 {
                solver_used = LeafSolverDiagnostic::Rejected {
                    solver: control.solver,
                    reason: format!(
                        "leaf rows {} < features {} + 1 (minimum support rule)",
                        actual_rows.len(),
                        chosen.len()
                    ),
                };
                None
            } else {
                match crate::solver::solve_leaf_problem(
                    &crate::solver::LeafFitProblem::Linear(crate::solver::LeafLinearProblem {
                        cols,
                        y,
                        w,
                        rows: &actual_rows,
                        chosen: &chosen,
                        with_indicators: any_missing,
                        impute_means: impute_means.as_deref(),
                        family,
                        solver: control.solver,
                        l1: control.l1,
                        l2: control.l2,
                        monotone: control.effective_leaf_monotone(),
                        mean: stats.mean,
                        extra: Some(&control.extra),
                        tweedie_p: control.tweedie_p,
                        tobit_threshold: control.tobit_threshold,
                        verbose: control.verbose,
                        groups: control.groups.clone(),
                        group_l2: control.group_l2,
                    }),
                ) {
                    Ok(output) => {
                        solver_used = output.diagnostic;
                        output.linear.or_else(|| {
                            if is_special_linear {
                                None
                            } else {
                                Some(LinearModel {
                                    coef: vec![0.0; chosen.len()],
                                    na_coef: if any_missing {
                                        vec![0.0; chosen.len()]
                                    } else {
                                        Vec::new()
                                    },
                                    feature_means: f_means.clone(),
                                    cols: chosen.clone(),
                                    intercept: f64::NAN,
                                    summary: None,
                                })
                            }
                        })
                    }
                    Err(reason) => {
                        if !matches!(solver_used, LeafSolverDiagnostic::Rejected { .. }) {
                            solver_used = LeafSolverDiagnostic::Failed {
                                solver: control.solver,
                                reason: reason.clone(),
                            };
                        }
                        if is_special_linear {
                            None
                        } else {
                            Some(LinearModel {
                                coef: vec![0.0; chosen.len()],
                                na_coef: if any_missing {
                                        vec![0.0; chosen.len()]
                                    } else {
                                        Vec::new()
                                    },
                                feature_means: f_means.clone(),
                                cols: chosen.clone(),
                                intercept: f64::NAN,
                                summary: None,
                            })
                        }
                    }
                }
            };
            if let Some(ref m) = lm {
                if control.verbose >= 3 {
                    println!(
                        "      [leaf linear fit] rows={} cols={} l1={:.4} l2={:.4} intercept={:.4}",
                        actual_rows.len(),
                        chosen.len(),
                        control.l1,
                        control.l2,
                        m.intercept
                    );
                }
            }
            lm
        }
    } else {
        None
    };
    // Non-mean constant leaves: fit the center on the positive-weight
    // targets only (same population as the stats).
    let yws: Vec<(f64, f64)> = if matches!(
        control.leaf,
        LeafKind::Median | LeafKind::Quantile(_) | LeafKind::Expectile(_) | LeafKind::Huber
    ) || family == LeafFamily::Quantile
        || family == LeafFamily::Expectile
        || family == LeafFamily::Huber
    {
        rows.iter()
            .filter(|&&r| w[r] > 0.0)
            .map(|&r| (y[r], w[r]))
            .collect()
    } else {
        Vec::new()
    };
    let center = match control.leaf {
        LeafKind::Median | LeafKind::Quantile(_) | LeafKind::Huber => {
            let values = yws.iter().map(|&(value, _)| value).collect::<Vec<_>>();
            let weights = yws.iter().map(|&(_, weight)| weight).collect::<Vec<_>>();
            let objective = match control.leaf {
                LeafKind::Median => LeafObjective::median(values, weights),
                LeafKind::Quantile(tau) => LeafObjective::quantile(values, weights, tau),
                LeafKind::Huber => LeafObjective::huber(values, weights),
                _ => unreachable!(),
            };
            let analytic = match control.leaf {
                LeafKind::Median => w_median(&yws),
                LeafKind::Quantile(tau) => w_quantile(&yws, tau),
                LeafKind::Huber => huber_location(&yws),
                _ => unreachable!(),
            };
            match objective {
                Ok(objective) if !matches!(solver_used, LeafSolverDiagnostic::Rejected { .. }) => {
                    match crate::solver::solve_leaf_problem(
                        &crate::solver::LeafFitProblem::Location {
                            solver: control.solver,
                            objective,
                            max_iter: 1000,
                            tol: 1e-9,
                        },
                    ) {
                        Ok(output) => {
                            solver_used = output.diagnostic;
                            output.center
                        }
                        Err(reason) => {
                            solver_used = LeafSolverDiagnostic::Failed {
                                solver: control.solver,
                                reason,
                            };
                            Some(analytic)
                        }
                    }
                }
                Ok(_) => Some(analytic),
                Err(reason) => {
                    solver_used = LeafSolverDiagnostic::Failed {
                        solver: control.solver,
                        reason,
                    };
                    Some(analytic)
                }
            }
        }
        LeafKind::Prob => Some(stats.prob()),
        LeafKind::Odds => Some(stats.odds()),
        LeafKind::LogOdds => Some(stats.log_odds()),
        _ if family == LeafFamily::Quantile => {
            let q_tau = match control.leaf {
                LeafKind::Quantile(t) => t,
                _ => control.tau,
            };
            Some(w_quantile(&yws, q_tau))
        }
        _ if family == LeafFamily::Huber => {
            Some(huber_location(&yws))
        }
        _ if family == LeafFamily::Expectile => {
            let exp_tau = match control.leaf {
                LeafKind::Expectile(t) => t,
                _ => control.expectile_tau,
            };
            Some(crate::solver::problems::expectile::weighted_expectile(&yws, exp_tau))
        }
        LeafKind::Expectile(t) => {
            Some(crate::solver::problems::expectile::weighted_expectile(&yws, t))
        }
        _ if family == LeafFamily::Beta => {
            Some(stats.mean.clamp(1e-4, 1.0 - 1e-4))
        }
        _ => Some(stats.mean),
    };
    LeafModel {
        stats,
        linear,
        center,
        family,
        softmax: None,
        survival: None,
        heteroscedastic: None,
        causal,
        aft: None,
        cure: None,
        spline: None,
        mars: None,
        mars_cols: None,
        knn: None,
        knn_cols: None,
        vector_mean: None,
        solver_used: solver_used.clone(),
        leaf_problem_type: leaf_problem_type.clone(),
    }
}

impl<'a> Grower<'a> {
    /// Columns a leaf model uses for a given root→leaf path (restricted to
    /// the pre-selected active columns).
    pub(crate) fn chosen_cols(&self, path: &[usize]) -> Vec<usize> {
        chosen_cols_policy(self.params.leaf_cols, None, self.active, path)
    }

    /// Leaf design for a node: the leaf columns under the column policy,
    /// whether any of them is missing on these rows (NA-indicator columns),
    /// and the resulting design dimension (intercept included).
    fn leaf_design(&self, rows: &[usize], path: &[usize]) -> (Vec<usize>, bool, usize) {
        let chosen = self.chosen_cols(path);
        let (any_missing, dim) = compute_leaf_design(self.cols, rows, &chosen);
        (chosen, any_missing, dim)
    }

    /// Weighted residual sum of squares of a leaf fit on `rows`, with the
    /// leaf's parameter count (mean-style leaves: 1; linear: the design
    /// dimension). Returns NaN RSS when the fit cannot produce finite
    /// predictions (the gate then skips).
    pub(crate) fn node_rss(&self, rows: &[usize], path: &[usize]) -> (f64, usize) {
        let leaf = self.make_leaf(rows, path);
        let k = match self.params.leaf {
            LeafKind::Linear => self.leaf_design(rows, path).2,
            LeafKind::Mars => leaf.mars.as_ref().map(|m| m.n_terms()).unwrap_or(1),
            _ => 1,
        };
        let mut xrow = vec![None; self.cols.len()];
        let mut rss = 0.0;
        for &r in rows {
            for (j, c) in self.cols.iter().enumerate() {
                xrow[j] = c[r];
            }
            let e = self.y[r] - leaf.predict(&xrow);
            if !e.is_finite() {
                return (f64::NAN, k);
            }
            rss += self.w[r] * e * e;
        }
        (rss, k)
    }

    /// Rows a leaf model fits on: the honesty estimation rows when there
    /// are enough (`min_rows`), else the node's structure rows.
    pub(crate) fn leaf_rows<'b>(&self, rows: &'b [usize], est: &'b [usize]) -> &'b [usize] {
        if self.params.honesty && est.len() >= self.params.min_rows {
            est
        } else {
            rows
        }
    }

    /// Fit the leaf model for a node's rows.
    pub(crate) fn make_leaf(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        let mut leaf = self.fit_leaf(rows, path, self.params.leaf_digest);
        if let Some(target_vecs) = self.target_vectors {
            if let Some(first) = target_vecs.first() {
                let dim = first.len();
                let mut sum_v = vec![0.0; dim];
                let mut sum_w = 0.0;
                for &r in rows {
                    let wt = self.w[r];
                    sum_w += wt;
                    for (d, &val) in target_vecs[r].iter().enumerate() {
                        sum_v[d] += wt * val;
                    }
                }
                if sum_w > 0.0 {
                    for val in &mut sum_v {
                        *val /= sum_w;
                    }
                }
                leaf.vector_mean = Some(sum_v);
            }
        }
        leaf
    }

    /// Candidate fitting shares the payload fitter but does not sort targets
    /// or allocate a target distribution. Never store this leaf in the tree.
    pub(crate) fn make_leaf_for_scan(&self, rows: &[usize], path: &[usize]) -> LeafModel {
        self.fit_leaf(rows, path, false)
    }

    /// Response-scale weighted deviance of an already fitted leaf.
    pub(crate) fn leaf_deviance(&self, rows: &[usize], leaf: &LeafModel) -> f64 {
        let mut total = 0.0;
        for &r in rows {
            if self.w[r] <= 0.0 {
                continue;
            }
            let loss = leaf
                .family
                .deviance(self.y[r], leaf.predict_columns(self.cols, r));
            if !loss.is_finite() {
                return loss;
            }
            total += self.w[r] * loss;
        }
        total
    }

    fn fit_leaf(&self, rows: &[usize], path: &[usize], digest: bool) -> LeafModel {
        let mut ctrl = LeafControl::from(self.params);
        ctrl.leaf_digest = digest;
        fit_leaf_model(self.cols, self.y, self.w, rows, path, self.active, &ctrl)
    }
}
