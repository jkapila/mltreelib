# Spec 11: GAMLSS and Multiparameter Distributional Leaves

## Problem Statement

Standard decision trees predict a single conditional mean $\hat{y} = E[Y \mid X]$. In practice, real-world data generates complex target geometries:
1. **Overdispersed counts**: In genomics, user clicks, and insurance frequency, variance dramatically exceeds the mean ($\text{Var}(Y) \gg \mu$), causing standard Poisson models to severely underestimate confidence bounds and produce false certainty.
2. **Bounded proportions**: Rates, percentages, and recovery fractions lie strictly in $(0, 1)$. Standard regression produces out-of-bounds predictions ($< 0$ or $> 1$) and violates heteroscedastic variance near the boundaries.
3. **Heavy tails and financial outliers**: Asset returns, latency, and sensor telemetry feature fat tails where Gaussian least squares leaves are pulled away by extreme single events.
4. **Ordered categories**: Credit ratings (AAA, AA, A, BBB, ...), customer survey scores, and cancer staging (Stage I-IV) have natural ordering, but cannot be treated as metric continuous numbers or unordered nominal classes.
5. **Structural zero-inflation**: Manufacturing defect counts or insurance claims feature a dual mechanism: an immune zero-state vs a positive intensity-state.

Existing tooling (`gamlss` in R, `disttree`, `statsmodels`) is either strictly linear, slow, not ported to Rust, or cannot fit tree leaves.

## Solution

Build a comprehensive GAMLSS (Generalized Additive Models for Location, Scale, and Shape) multiparameter leaf engine inside `mltreelib-core`:
1. **Negative Binomial Leaves (`leaf_family="negbinom"`)**: Estimate location $\mu = \exp(X\beta)$ and dispersion parameter $\theta$, properly modeling overdispersed count data.
2. **Beta Regression Leaves (`leaf_family="beta"`)**: Estimate location $\text{logit}(\mu) = X\beta$ and precision $\ln(\phi) = X\gamma$, guaranteeing predictions in $(0, 1)$ with exact boundary-variance modeling.
3. **Student's $t$ Robust Leaves (`leaf_family="student_t"`)**: Fit robust location-scale regression $(\mu, \sigma)$ with degrees of freedom $\nu$, completely shielding linear leaves from fat-tailed outliers.
4. **Zero-Inflated Poisson (ZIP) & ZINB Hurdle Leaves**: Fit joint logistic zero-inflation probability $\text{logit}(\pi) = X\alpha$ and count rate $\ln(\lambda) = X\beta$.
5. **Cumulative Logit (Proportional Odds) Leaves (`leaf_family="ordinal"`)**: Estimate ordered cutoffs $\theta_1 < \theta_2 < \dots < \theta_{K-1}$ with shared covariate slopes $\beta$, providing principled ordinal regression.
6. **Generalized Extreme Value (GEV) & Pareto Leaves**: Model location, scale, and shape $(\mu, \sigma, \xi)$ for tail value-at-risk and peak-over-threshold extremes.

## User Stories

1. As an insurance actuary, I want `leaf_family="negbinom"`, so that frequency trees correctly reflect overdispersion without underpricing catastrophic risk tails.
2. As an ad-tech engineer, I want `leaf_family="beta"` for Click-Through Rate (CTR) and conversion predictions, so that output probabilities are mathematically guaranteed to stay in $(0, 1)$ with accurate variance near 0 and 1.
3. As a quantitative portfolio manager, I want `leaf_family="student_t"`, so that market regime linear leaves are not derailed by flash crashes and black swan returns.
4. As a quality engineer, I want `leaf_family="zip"`, so that the tree separates the probability of a machine being in an error-free state from the defect count rate when in an active error state.
5. As a credit risk modeler, I want `leaf_family="ordinal"` with cumulative logit link, so that borrower credit tranches (AAA to C) are predicted with preserved ordering.
6. As a climate scientist, I want `leaf_family="gev"`, so that 100-year rainfall and flood return periods can be modeled conditionally across regional terrain leaves.
7. As an ML researcher, I want `predict_distribution(X)` to return fitted parameter objects (e.g. `NegativeBinomial(mu, theta)` or `Beta(mu, phi)`), so that downstream applications can sample, compute CDFs, or evaluate quantiles at arbitrary levels.
8. As a data scientist, I want `leaf.summary()` to output Fisher information standard errors for all location and scale parameters, so that model credibility can be established per leaf.
9. As an embedded engineer, I want leaf parameter optimization to run Newton-Raphson / IRLS in pure Rust with analytic Hessians, so that leaf model fitting converges in milliseconds.
10. As a maintainer, I want all distribution families to handle sample weights and missing values natively, consistent with the rest of `mltreelib`.

## Implementation Decisions

- **Parameter Multiparameter Representation**:
  Leaves hold parameter vectors $[\beta_\mu, \beta_\sigma, \dots]$ rather than a single scalar or linear vector.
- **Leaf Family Additions**:
  - `LeafFamily::NegativeBinomial { theta: Option<f64> }`
  - `LeafFamily::Beta { phi: Option<f64> }`
  - `LeafFamily::StudentT { nu: f64 }`
  - `LeafFamily::Zip` / `LeafFamily::Zinb`
  - `LeafFamily::CumulativeLogit { n_classes: usize }`
  - `LeafFamily::Gev` / `LeafFamily::Gpd`
- **IRLS and Scoring Engine**:
  Each family implements:
  - Link functions and inverse link functions (`link`, `inv_link`, `d_link`).
  - Unit log-likelihood and deviance.
  - Working weights (diagonal elements of Fisher information matrix).
  - Working responses $z = \eta + (y - \mu) \frac{d\eta}{d\mu}$.
- **High Seam**:
  Extended methods on `ModelTree`:
  - `predict_distribution(X)` -> returns dictionary / object with fitted parameters per sample.
  - `cdf(X, y)` -> cumulative density values.
  - `ppf(X, tau)` / `quantile(X, tau)` -> inverse CDF quantiles.

## Testing Decisions

- **Behavioral Tests**:
  - Overdispersed count test: generate synthetic data with $\mu=10, \text{Var}=50$; verify Negative Binomial leaf recovers dispersion parameter $\theta \approx 2.5$ and achieves lower deviance than Poisson.
  - Bounded $(0, 1)$ beta test: generate beta proportions with heteroscedastic noise; verify predictions remain strictly in $(0, 1)$.
  - Outlier resistance test: add extreme contamination (100 standard deviations away) to 1% of data; verify Student's $t$ linear leaf slopes remain stable while Gaussian OLS slopes collapse.
  - Ordinal monotonicity test: verify Cumulative Logit leaf maintains strict ordering $\theta_1 < \theta_2 < \dots < \theta_{K-1}$ across all segments.

## Out of Scope

- Spline basis expansion within single leaf equations (focus is piecewise-linear and constant parameter leaves within tree partitions).
- Fully non-parametric copula leaf joint estimation.

## Further Notes

- Bridges modern recursive partitioning with Rigby & Stasinopoulos (2005) *Generalized Additive Models for Location, Scale and Shape (GAMLSS)* and Schlosser et al. (2019) *Distributional Trees and Forests*.
