# Spec 19: Frontier Survival Model Trees: Parametric AFT, Mixture Cure, Splines, Pseudo-Value/RMST Splits, and Modern Calibration

## Problem Statement

Spec 10 delivered non-parametric right-censored survival trees (Kaplan-Meier, Nelson-Aalen) and competing risks trees (Gray's test, Aalen-Johansen CIF with simplex invariant). However, several major research frontiers from 2020–2026 remain open:
1. **Flat Step-Function Limitation**: Existing packages (`scikit-survival`, `ranger`, `randomForestSRC`) store constant step functions in terminal leaves. Predictions are flat across covariates inside each leaf.
2. **Crossing Hazards Failure of Log-Rank**: Log-rank test statistics lose statistical power when hazards cross, failing to detect treatments with early adverse effects and long-term cure.
3. **Censoring Masking Continuous Linear Modeling**: Right-censoring prevents standard linear model tree fitting.
4. **Calibration Blindness of Concordance**: C-index evaluates only relative ranking, failing to detect severe probability miscalibration.
5. **Heavy Dataset Bloat**: Bundling multiple tabular benchmark datasets directly in wheels or git repositories bloats package distribution.

## Solution Architecture

1. **Parametric AFT Model Trees (`aft_dist="weibull" | "lognormal" | "loglogistic"`)**:
   Fit log-linear models $\ln(T_i) = \alpha + \mathbf{x}_i^T \boldsymbol{\beta} + \sigma \epsilon_i$ inside leaves via pure-Rust L-BFGS. Outputs smooth, continuous survival curves and hazard functions that vary with covariates within each leaf.
2. **Mixture Cure Rate Leaves (`leaf_family="cure"`)**:
   Model the immune fraction $\pi(\mathbf{x}) = \text{logit}^{-1}(\mathbf{w}^T \mathbf{x})$ that never experiences the event, with an AFT/KM model for the uncured fraction: $S(t \mid \mathbf{x}) = \pi(\mathbf{x}) + (1 - \pi(\mathbf{x})) S_u(t \mid \mathbf{x})$.
3. **Restricted Cubic Spline Leaves (`leaf_family="spline"`)**:
   Royston-Parmar flexible spline modeling of log cumulative hazard $\ln \Lambda(t \mid \mathbf{x}) = s(\ln t; \boldsymbol{\gamma}) + \mathbf{x}^T \boldsymbol{\beta}$.
4. **2020–2026 Split Criteria**:
   - `gain="pseudo_value"`: Jackknife pseudo-observations for RMST (Schenk et al. 2024), converting censored data into continuous targets for $O(N)$ streaming variance reduction.
   - `gain="rmst"`: Restricted Mean Survival Time between-node difference $|\text{RMST}_L - \text{RMST}_R|$, immune to crossing hazards (Liu & Li 2020/2021).
   - `gain="aft_deviance"`: Parametric negative log-likelihood deviance reduction (Chen et al. 2020).
5. **Modern Calibration Metrics**:
   - D-Calibration (`d_calibration`): Haider et al. (JMLR 2020) bucketed $\chi^2$ goodness-of-fit test.
   - Integrated Calibration Index (`ici`): Austin et al. (2020) mean absolute calibration error via LOESS.
   - `one_calibration`: 1-Calibration slope and intercept at specific horizons $t_0$.
   - Multi-cause competing risks calibration divergence (Alberge et al. 2026).
6. **On-Demand Dataset Downloader**:
   Light package policy: no bundled dataset files. Download on demand into `~/.cache/mltreelib/datasets/` from OpenML / GitHub raw releases. Catalog SurvSet benchmark datasets (`support`, `flchain`, `colon`, `rott2`, `pbc`, `veteran`, `gbsg2`).
7. **Experimental Causal Survival**:
   Honest sample-split CATE trees on survival / RMST, documented as research work under consideration.

## User Stories

1. As an actuarial scientist, I want `leaf_family="survival", aft_dist="weibull"`, so that my tree predicts smooth, continuous parametric life expectancy curves rather than discrete step functions.
2. As an oncologist modeling long-term leukemia remission, I want `leaf_family="cure"`, so that the tree simultaneously estimates the cured fraction $\pi(X)$ and the survival trajectory of relapse patients.
3. As a biostatistician analyzing immuno-oncology trials with crossing survival curves, I want `gain="rmst"`, so that tree splits reliably separate cohorts when log-rank power fails.
4. As a machine learning researcher, I want `gain="pseudo_value"`, so that I can train model trees with linear leaves on censored data using $O(N)$ streaming scans.
5. As an ML quality auditor, I want `Metrics().d_calibration(...)`, so that I can statistically test whether predicted individual survival distributions are uniform across risk quantiles.
6. As a package installer, I want `mltreelib` wheels to stay under 10 MB, downloading benchmark datasets on demand into a user cache only when requested.

## Implementation Steps

1. **Step 1**: Implement on-demand dataset downloader and SurvSet catalog in `python/mltreelib/datasets.py`.
2. **Step 2**: Implement parametric AFT distributions (`Weibull`, `LogNormal`, `LogLogistic`), likelihood, gradients, and L-BFGS fitting in `crates/mltreelib-core/src/leaf/` and `leaf.rs`.
3. **Step 3**: Implement Mixture Cure and Restricted Cubic Spline models in `crates/mltreelib-core/src/leaf/` and `node_stats.rs`.
4. **Step 4**: Implement `gain="pseudo_value"`, `gain="rmst"`, and `gain="aft_deviance"` in `crates/mltreelib-core/src/criteria.rs` and `split.rs`.
5. **Step 5**: Implement `d_calibration`, `ici`, `one_calibration`, and competing risks calibration metrics in `crates/mltreelib-core/src/metrics.rs` and Python bindings.
6. **Step 6**: Wire PyO3 bindings in `crates/mltreelib-py/src/lib.rs`.
7. **Step 7**: Build experimental Causal Survival example in `examples/guides/survival_causal_survival_honest.py`.
8. **Step 8**: Comprehensive test suite in `tests/test_frontier_survival.py`.
9. **Step 9**: Update documentation and build verification.
