# Spec 17: Panel, VAR and VECM Longitudinal Trees

## Problem Statement

When modeling longitudinal panels and multivariate time series (e.g. M5 retail sales across 30,490 series, multi-store demand, multi-market pricing, macro-financial indicators), existing methods force an unacceptable compromise:
1. **Mainstream GBDTs (LightGBM/XGBoost)**: Treat rows as independent and identically distributed. They cannot model multi-series endogenous feedback loops, struggle with cointegration and equilibrium error-correction, fail to extrapolate non-stationary trends, and suffer catastrophic memory blowups when unrolling panels (e.g. 59 million rows on M5) with entity dummy encodings.
2. **Econometric Systems (VAR, VECM, Panel Fixed Effects)**: Capture multivariate feedback and unobserved entity heterogeneity ($\alpha_i$), but enforce globally rigid linear relationships, completely failing to capture non-linear regime shifts, categorical interactions, and high-dimensional covariate splits.
3. **Threshold Autoregressive Models (TAR/TVAR)**: Existing R/Stata packages only search a single scalar threshold variable via grid search, unable to handle multidimensional hierarchical feature trees.

## Solution

Implement a native, pure-Rust econometric tree architecture in `mltreelib-core` unifying **Panel Fixed Effects**, **Threshold Vector Autoregression (TVAR)**, and **Threshold Vector Error Correction Models (TVECM)**:

1. **Panel Fixed-Effects (Within-Transformation) Model Trees**:
   - For longitudinal panel data $y_{it} = \alpha_i + \mathbf{x}_{it}^T \boldsymbol{\beta} + \epsilon_{it}$:
   - Eliminate individual entity heterogeneity $\alpha_i$ analytically via within-group demeaning without adding $N$ dummy variables:
     $$\tilde{y}_{it} = y_{it} - \bar{y}_i, \quad \tilde{\mathbf{x}}_{it} = \mathbf{x}_{it} - \bar{\mathbf{x}}_i$$
   - Decision tree splits on cross-sectional static attributes (Store, Category, State, Size) or dynamic macro covariates.
   - Terminal leaves fit Ridge-regularized within-group slopes $\hat{\boldsymbol{\beta}}^{(m)}$ via `faer::llt`.
   - Out-of-sample prediction adds back entity-specific fixed effects $\hat{\alpha}_i = \bar{y}_i - \bar{\mathbf{x}}_i^T \hat{\boldsymbol{\beta}}$.
2. **Dynamic Panel GMM Leaves (Arellano-Bond)**:
   - For autoregressive panel models $y_{it} = \rho y_{i,t-1} + \mathbf{x}_{it}^T \boldsymbol{\beta} + \alpha_i + \epsilon_{it}$, where demeaning induces Nickell bias:
   - First-difference the model to eliminate $\alpha_i$: $\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta \mathbf{x}_{it}^T \boldsymbol{\beta} + \Delta \epsilon_{it}$.
   - Estimate parameters inside leaves using lagged levels $(y_{i,t-2}, y_{i,t-3}, \dots)$ as instrumental variables via 2-stage Generalized Method of Moments (GMM).
3. **Threshold Vector Autoregression (TVAR) Trees**:
   - Model $K$ endogenous interacting series $\mathbf{y}_t \in \mathbb{R}^K$:
     $$\mathbf{y}_t = \mathbf{c}^{(m)} + \sum_{p=1}^P \mathbf{A}_p^{(m)} \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t^{(m)} \quad \text{for } \mathbf{x}_t \in \mathcal{R}_m$$
   - **Split Criterion**: Multivariate trace of residual covariance reduction:
     $$\Delta I = \text{Tr}(\mathbf{E}_{\text{parent}}^T \mathbf{E}_{\text{parent}}) - \left[ \text{Tr}(\mathbf{E}_L^T \mathbf{E}_L) + \text{Tr}(\mathbf{E}_R^T \mathbf{E}_R) \right]$$
   - Leaves estimate coefficient matrices $\hat{\mathbf{B}}^{(m)} = (\mathbf{Z}_m^T \mathbf{Z}_m + \lambda \mathbf{I})^{-1} \mathbf{Z}_m^T \mathbf{Y}_m$ in pure Rust.
4. **Threshold Vector Error Correction Model (TVECM) Trees**:
   - For non-stationary cointegrated series sharing stationary equilibrium spreads $\mathbf{z}_{t-1} = \boldsymbol{\beta}^T \mathbf{y}_{t-1}$:
   - **Global Stage**: Estimate cointegrating vectors $\hat{\boldsymbol{\beta}}$ via Johansen generalized eigenvalue decomposition.
   - **Tree Stage**: Partition on equilibrium error magnitude $|\hat{\boldsymbol{\beta}}^T \mathbf{y}_{t-1}|$ (transaction cost bands) and exogenous regime indicators.
   - **Leaf Models**: Fit regime-specific speeds of adjustment $\hat{\boldsymbol{\alpha}}^{(m)}$ and short-run dynamics $\hat{\boldsymbol{\Gamma}}_p^{(m)}$:
     $$\Delta \mathbf{y}_t = \boldsymbol{\mu}^{(m)} + \boldsymbol{\alpha}^{(m)} (\hat{\boldsymbol{\beta}}^T \mathbf{y}_{t-1}) + \sum_{p=1}^{P-1} \boldsymbol{\Gamma}_p^{(m)} \Delta \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t^{(m)}$$

## User Stories

1. As a demand forecasting scientist, I want `PanelModelTree(entity_col="item_id", time_col="date", leaf="fixed_effects")`, so that I can eliminate millions of item-level dummy variables and fit pure within-item pricing elasticities.
2. As a macroeconomist, I want `VARTree(lags=2, criterion="trace_cov")`, so that multivariate feedback loops between interest rates, inflation, and GDP switch dynamically across monetary policy regimes.
3. As a quantitative commodity trader, I want `VECMTree(cointegrating_rank=1, split_on_spread=True)`, so that the tree detects asymmetric mean-reversion speeds outside commodity arbitrage transaction-cost bands.
4. As an econometrician, I want `leaf="arellano_bond"`, so that dynamic panel autoregression is estimated without Nickell bias using instrumented moment conditions.
5. As an M5 competition modeler, I want wide-format matrix ingestion directly into the panel tree, avoiding multi-gigabyte unrolled long-format memory consumption.
6. As a risk researcher, I want `model.impulse_response(horizon=12, regime="stress")`, computing multi-step vector impulse response functions (IRF) conditional on remaining in a leaf regime.
7. As a data engineer, I want entity-level demeaning statistics $(\bar{y}_i, \bar{\mathbf{x}}_i)$ to be cached in streaming hash tables, running in $O(N \cdot T)$ linear time.
8. As a financial analyst, I want `vecm_tree.johansen_eigenvalues()`, reporting trace and maximum eigenvalue cointegration statistics.
9. As a production engineer, I want `predict_forecast(horizon=h)` to iterate vector autoregressive steps recursively inside Rust without Python boundary overhead.
10. As a retail planner, I want hierarchical pooling (`pool_level="dept"`), so that leaves with fewer than `min_rows` fallback smoothly to parent departmental panel coefficients.
11. As an API user, I want `PanelModelTree` to conform to `fit(X, y)` and `predict(X)` with explicit entity index tracking and zero scikit-learn dependency.
12. As a systems developer, I want all multivariate matrix inversions and Kronecker products to execute using `faer` SIMD primitives with zero heap re-allocations during split evaluation.

## Implementation Decisions

- **Crate Module Structure**:
  - Add module `crates/mltreelib-core/src/longitudinal/`:
    - `panel_demean.rs`: Fast within-group demeaning and entity mean tracking for balanced/unbalanced panels.
    - `panel_tree.rs`: `PanelModelTree` with fixed effects and random effects leaf estimators.
    - `arellano_bond.rs`: First-difference instrument matrix builder and 2-stage GMM estimator.
    - `var.rs`: Multivariate lag matrix constructor and `VARTree` with trace-covariance split search.
    - `vecm.rs`: Johansen eigenvalue solver and `VECMTree` with cointegration spread routing.
    - `irf.rs`: Orthogonalized impulse response function (IRF) simulator via Cholesky decomposition of leaf residual covariance.
- **Multivariate Covariance Gain**:
  $$\text{Gain} = \text{Tr}(\mathbf{E}_P^T \mathbf{E}_P) - \left( \text{Tr}(\mathbf{E}_L^T \mathbf{E}_L) + \text{Tr}(\mathbf{E}_R^T \mathbf{E}_R) \right)$$
  Where $\mathbf{E} = \mathbf{Y} - \mathbf{Z} \hat{\mathbf{B}}$.
- **Python Surface (`crates/mltreelib-py`)**:
  - `mltreelib.longitudinal.PanelModelTree(entity_col="id", leaf="fixed_effects", min_rows=20, l2_reg=1e-3)`
  - `mltreelib.longitudinal.VARTree(lags=2, max_depth=5, min_rows=30)`
  - `mltreelib.longitudinal.VECMTree(lags=2, rank=1, max_depth=5)`

## Testing Decisions

- **Within-Transformation Demeaning Test**:
  - Simulate panel $y_{it} = \alpha_i + 3.0 x_{it} + \epsilon_{it}$ with 1,000 entities and distinct random $\alpha_i \sim \mathcal{N}(0, 100)$.
  - Verify `PanelModelTree` recovers slope $\hat{\beta} \approx 3.00 \pm 0.01$ without fitting dummy variables for $\alpha_i$.
- **TVAR Known Regime Shift Test**:
  - Generate 2-variable VAR(1):
    - Regime 1 ($X_1 \le 0$): $\mathbf{A}^{(1)} = \begin{pmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{pmatrix}$.
    - Regime 2 ($X_1 > 0$): $\mathbf{A}^{(2)} = \begin{pmatrix} 0.0 & 0.8 \\ 0.8 & 0.0 \end{pmatrix}$.
  - Verify tree splits at $X_1 \approx 0$ and leaf coefficient matrices match true regimes within $3\%$ error.
- **TVECM Cointegration & Adjustment Speed Test**:
  - Generate cointegrated pair $y_{1,t} - y_{2,t} = z_t$ where $z_t$ follows stationary AR(1) with threshold adjustment speed:
    - If $|z_{t-1}| \le 1.0$: $\alpha = 0$ (no adjustment).
    - If $|z_{t-1}| > 1.0$: $\alpha = -0.5$ (rapid mean reversion).
  - Verify `VECMTree` splits on spread $|z_{t-1}| = 1.0$ and identifies zero vs non-zero adjustment regimes.
- **Longitudinal Memory Benchmark**:
  - Verify panel within-group fitting on 1,000,000 rows across 5,000 entities consumes $<200$MB peak RAM.

## Out of Scope

- Non-linear continuous-time stochastic differential equations (jump diffusion / Lévy processes).
- R package CRAN export (explicitly deferred per user instruction to later roadmap milestone).

## Further Notes

- By natively embedding panel within-transformations and vector autoregressive systems into tree leaves, `mltreelib` breaks the historical barrier between structural econometrics and machine learning, directly addressing high-scale longitudinal benchmarks like M5 and macroeconomic regime tracking.
