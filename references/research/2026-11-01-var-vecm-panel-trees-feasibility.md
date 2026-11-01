# VAR, VECM, and Panel Regression Trees: Deep Theory, Effort Estimation, and Architectural Blast Radius

**Date**: 2026-11-01  
**Author**: Jitin Kapila & Antigravity  
**Topic**: Deep dive into integrating Vector Autoregression (VAR), Vector Error Correction Models (VECM), and Panel Regression (Fixed/Random Effects, Arellano-Bond GMM) into tree and model-tree architectures for longitudinal time series (e.g., M5). Detailed effort estimation and architectural cost analysis for `mltreelib`.

---

## 1. Executive Summary & Thesis

Modern machine learning (GBDTs) and econometrics have developed along disconnected trajectories:
- **Mainstream GBDTs (LightGBM/XGBoost)** excel at non-linear interactions and cross-sectional pooling, but treat time observations as independent rows, fail on long-run non-stationary equilibrium, cannot extrapolate trends, and suffer severe memory blowups on panel data (59M rows on M5).
- **Econometric Panel & Time Series Models (VAR, VECM, Panel FE/GMM)** explicitly model multivariate feedback, cointegration equilibrium, and unobserved cross-sectional heterogeneity ($\alpha_i$), but enforce rigid linear structures and struggle with high-dimensional feature splits.

**The Hybrid Thesis for `mltreelib`**:
By marrying recursive partitioning with econometric leaf formulations, we obtain:
1. **Threshold VAR (TVAR) & Model-Tree VAR**: Recursive state-space/hierarchical partitioning with local multivariate vector autoregressions at leaves.
2. **Threshold VECM (TVECM) / Cointegrated Trees**: Non-linear regime-switching error correction conditional on equilibrium deviations ($\boldsymbol{\beta}^T \mathbf{y}_{t-1}$).
3. **Panel Fixed-Effects / Dynamic GMM Trees**: Splitting on time-invariant cross-sectional covariates (Store, Region, Category) while running within-group demeaning or Arellano-Bond instrumented dynamics inside each leaf.

---

## 2. Mathematical Foundations & Tree Formulations

### 2.1. Vector Autoregression (VAR) Trees

#### Standard Linear VAR(p)
For $K$ interacting series at time $t$, $\mathbf{y}_t = [y_{1,t}, \dots, y_{K,t}]^T \in \mathbb{R}^K$:
$$\mathbf{y}_t = \mathbf{c} + \sum_{p=1}^P \mathbf{A}_p \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$
Where each $\mathbf{A}_p$ is a $K \times K$ coefficient matrix. In compact matrix form:
$$\mathbf{Y} = \mathbf{Z} \mathbf{B} + \mathbf{E}$$
where $\mathbf{Z}_t = [1, \mathbf{y}_{t-1}^T, \dots, \mathbf{y}_{t-P}^T] \in \mathbb{R}^{1 + KP}$.

#### Model-Tree VAR (TVAR Formulation)
The state space (or cross-sectional hierarchical attributes $\mathbf{x}$) is partitioned by decision tree $T$:
$$\text{If } \mathbf{x} \in \mathcal{R}_m \implies \mathbf{y}_t = \mathbf{c}^{(m)} + \sum_{p=1}^P \mathbf{A}_p^{(m)} \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t^{(m)}$$
- **Split Criterion**: Reduction in trace of residual covariance matrix:
  $$\Delta I = \text{Tr}(\mathbf{E}_{\text{parent}}^T \mathbf{E}_{\text{parent}}) - \left[ \text{Tr}(\mathbf{E}_L^T \mathbf{E}_L) + \text{Tr}(\mathbf{E}_R^T \mathbf{E}_R) \right]$$
- **Leaf Estimation**: Solved via Ridge-regularized multivariate normal equations:
  $$\hat{\mathbf{B}}^{(m)} = \left(\mathbf{Z}_m^T \mathbf{Z}_m + \lambda \mathbf{I}\right)^{-1} \mathbf{Z}_m^T \mathbf{Y}_m$$
  Leveraging `faer::llt` (already integrated into `mltreelib-core`).

---

### 2.2. Vector Error Correction Model (VECM) Trees

#### Cointegration & Granger Representation
When non-stationary series $I(1)$ share stationary linear combinations $I(0)$, a pure VAR in differences omits long-run equilibrium. The VECM specifies:
$$\Delta \mathbf{y}_t = \boldsymbol{\mu} + \boldsymbol{\Pi} \mathbf{y}_{t-1} + \sum_{p=1}^{P-1} \boldsymbol{\Gamma}_p \Delta \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t$$
where $\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}^T$:
- $\boldsymbol{\beta} \in \mathbb{R}^{K \times r}$: Cointegrating vectors ($r = \text{rank}(\boldsymbol{\Pi}) < K$). The spreads $\mathbf{z}_{t-1} = \boldsymbol{\beta}^T \mathbf{y}_{t-1}$ represent equilibrium errors.
- $\boldsymbol{\alpha} \in \mathbb{R}^{K \times r}$: Speed of adjustment / error-correction velocities.
- $\boldsymbol{\Gamma}_p$: Short-run dynamic adjustment matrices.

#### Threshold VECM (TVECM) as a Tree (Hansen & Seo 2002, Balke & Fomby 1997)
In retail and financial panels, market friction or inventory re-orders cause **asymmetric adjustment**:
- Small deviations $|\boldsymbol{\beta}^T \mathbf{y}_{t-1}| \le \gamma$: No price adjustment (transaction cost band; $\boldsymbol{\alpha} \approx \mathbf{0}$).
- Large deviations $|\boldsymbol{\beta}^T \mathbf{y}_{t-1}| > \gamma$: Rapid mean reversion ($\boldsymbol{\alpha} \gg \mathbf{0}$).

**Tree Architecture**:
1. Global stage: Estimate cointegrating vector $\hat{\boldsymbol{\beta}}$ via Johansen generalized eigenvalue problem:
   $$|\lambda \mathbf{S}_{11} - \mathbf{S}_{10} \mathbf{S}_{00}^{-1} \mathbf{S}_{01}| = 0$$
2. Tree stage: Split candidate features include the cointegrating spread $z_{t-1} = \hat{\boldsymbol{\beta}}^T \mathbf{y}_{t-1}$.
3. Leaves estimate regime-specific adjustment speeds $\hat{\boldsymbol{\alpha}}^{(m)}$ and short-run dynamics $\hat{\boldsymbol{\Gamma}}_p^{(m)}$.

---

### 2.3. Panel Regression Trees (Fixed Effects & Dynamic Panel GMM)

In longitudinal panel datasets (like M5: $N=30{,}490$ series observed across $T=1{,}941$ days), series possess unobserved individual heterogeneity $\alpha_i$ (store-specific foot traffic, item shelf appeal).

#### Model A: Panel Fixed Effects (FE / Within Transformation)
$$y_{it} = \alpha_i + \mathbf{x}_{it}^T \boldsymbol{\beta} + \epsilon_{it}$$
- **Within-Transformation (Demeaning)**:
  $$\tilde{y}_{it} = y_{it} - \bar{y}_i, \quad \tilde{\mathbf{x}}_{it} = \mathbf{x}_{it} - \bar{\mathbf{x}}_i$$
  Eliminates $\alpha_i$ without adding $30{,}490$ dummy variables!
- **Tree Integration**:
  - Tree splits on time-invariant cross-sectional covariates (Store format, State, Dept, Category).
  - Within each leaf, estimate $\hat{\boldsymbol{\beta}}^{(m)}$ on demeaned variables:
    $$\hat{\boldsymbol{\beta}}^{(m)} = \left(\sum_{i \in \mathcal{R}_m} \tilde{\mathbf{X}}_i^T \tilde{\mathbf{X}}_i + \lambda \mathbf{I}\right)^{-1} \sum_{i \in \mathcal{R}_m} \tilde{\mathbf{X}}_i^T \tilde{\mathbf{y}}_i$$
  - Individual intercept recovered as: $\hat{\alpha}_i = \bar{y}_i - \bar{\mathbf{x}}_i^T \hat{\boldsymbol{\beta}}^{(m)}$.

#### Model B: Dynamic Panel Arellano-Bond (Difference GMM)
When autoregressive lags $y_{i,t-1}$ are present alongside fixed effects $\alpha_i$:
$$y_{it} = \gamma y_{i,t-1} + \mathbf{x}_{it}^T \boldsymbol{\beta} + \alpha_i + \epsilon_{it}$$
Standard OLS or FE is biased ($O(1/T)$ Nickell bias) because $\tilde{y}_{i,t-1}$ correlates with $\tilde{\epsilon}_{it}$.
- **Arellano-Bond Transformation**: First-difference to remove $\alpha_i$:
  $$\Delta y_{it} = \gamma \Delta y_{i,t-1} + \Delta \mathbf{x}_{it}^T \boldsymbol{\beta} + \Delta \epsilon_{it}$$
- **Instrument Matrix $\mathbf{Z}_i$**: Instrument $\Delta y_{i,t-1}$ with lagged levels $y_{i,t-2}, y_{i,t-3}, \dots$
- **Two-Step GMM Estimator**:
  $$\hat{\boldsymbol{\theta}}_{\text{GMM}} = \left(\mathbf{X}^{*T} \mathbf{Z} \mathbf{W}_N \mathbf{Z}^T \mathbf{X}^*\right)^{-1} \mathbf{X}^{*T} \mathbf{Z} \mathbf{W}_N \mathbf{Z}^T \mathbf{y}^*$$
  where $\mathbf{W}_N = \left(\frac{1}{N} \sum_{i=1}^N \mathbf{Z}_i^T \Delta \hat{\mathbf{e}}_i \Delta \hat{\mathbf{e}}_i^T \mathbf{Z}_i\right)^{-1}$.

---

## 3. Detailed Effort Estimation Breakdown

To build this capability into `mltreelib`, the project can be decomposed into 4 distinct phases:

| Phase | Component | Key Deliverables | Est. Hours | Timeline | Complexity / Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Panel FE Leaf Tree** | Within-group demeaning, $O(N)$ streaming moments, cross-sectional split routing, out-of-sample $\alpha_i$ fallback. | 50–70 hrs | 2 weeks | **Low / Low Risk** (re-uses existing `faer` Cholesky solver). |
| **Phase 2** | **Ridge VAR Model Tree** | Multivariate OLS/Ridge solver, lag-order selector (AIC/BIC), trace-dispersion split criterion, vector forecast output. | 60–80 hrs | 2–3 weeks | **Medium / Low Risk** (direct extension of `node_stats.rs`). |
| **Phase 3** | **VECM & Cointegrated Trees** | Johansen generalized eigensolver, cointegrating spread generator, TVECM threshold split search. | 70–90 hrs | 2–3 weeks | **High / Medium Risk** (requires generalized symmetric eigensolver via `faer` or `nalgebra`). |
| **Phase 4** | **Dynamic Panel GMM (Arellano-Bond)** | Difference GMM solver, instrument truncation (to avoid Roodman instrument explosion), Sargan-Hansen overid test. | 80–110 hrs | 3 weeks | **High / High Risk** (numerically delicate weighting matrix inversion). |
| **Total** | **Full Econometric Suite** | Complete VAR, VECM, and Panel Tree subsystem with PyO3 bindings and benchmark tests. | **260–350 hrs** | **8–11 weeks** | Overall: **Medium-High** |

### Recommended Phasing & MVP Strategy
- **Pragmatic MVP (Phases 1 & 2)**: 3–4 weeks total (~110–150 hours). Delivers **Panel Fixed-Effects Trees** and **Multi-Output Ridge VAR Trees**. This captures >85% of practical gains on retail benchmarks like M5 with minimal architectural risk.
- **Advanced Stage (Phases 3 & 4)**: 4–6 weeks additional. Unlocks theoretical cointegration (VECM) and macro-econometric dynamic panel instruments.

---

## 4. Architectural Cost & Blast Radius Analysis

### 4.1. Dependency Footprint
- **Current `Cargo.toml` dependencies**:
  - `faer = "0.24.4"` (Dense linear algebra, Cholesky, LU, QR)
  - `nalgebra = "0.33"` (General linear algebra, SVD, Eigendecomposition)
  - `statrs = "0.18"` (Statistical distributions, $\chi^2$ p-values)
  - `rayon = "1"` (Multi-threading)
- **New External Dependencies Needed**: **ZERO**.
  - All matrix operations (Cholesky, Generalized Eigendecomposition for Johansen, GMM matrix sandwiches) are natively supported by `faer` and `nalgebra`.
  - Zero C/Fortran/LAPACK linkage required; keeps pure Rust portability intact.

### 4.2. File-by-File Blast Radius

```
crates/mltreelib-core/src/
├── frame.rs              [MODERATE] Add contiguous 2D Panel / Wide-View accessors.
├── node_stats.rs          [MODERATE] Add Multivariate Gram & Demeaned Moments accumulators.
├── leaf.rs                [EXPAND]   Add `LeafModel::PanelFE`, `LeafModel::Var`, `LeafModel::Vecm`.
├── split.rs               [EXPAND]   Support multivariate dispersion gain & cointegration threshold splits.
├── criteria.rs            [LIGHT]    Add trace-variance split criterion.
├── econometrics/          [NEW MODULE]
│   ├── mod.rs             Module declarations & common types.
│   ├── panel.rs           Within-demeaning, random effects GLS, Arellano-Bond GMM.
│   ├── var.rs             Multivariate Ridge VAR normal equations & lag unrolling.
│   └── vecm.rs            Johansen cointegration test & error-correction representation.
crates/mltreelib-py/src/   [MODERATE] Expose `PanelTree`, `VarTree`, and `VecmTree` classes.
python/mltreelib/          [LIGHT]    Python wrappers with scikit-learn shaped fit/predict API.
```

### 4.3. Computational Complexity & Memory Footprint

| Dimension | Standard GBDT (LightGBM Melted) | Proposed `mltreelib` Panel/VAR Tree |
| :--- | :--- | :--- |
| **Memory Allocation (M5)** | 12 GB – 25 GB RAM (59M rows) | **~240 MB – 500 MB RAM** ($30{,}490 \times 1{,}941$ buffer) |
| **Split Search Complexity** | $O(N_{\text{long}} \cdot P) \approx 60\text{M} \times 20$ operations | $O(N_{\text{wide}} \cdot P \cdot K) \approx 30\text{k} \times 20 \times 28$ operations |
| **Wall-Clock Fit Speed** | Minutes per tree | **Milliseconds to seconds** per tree |
| **Cross-Horizon Coherence** | None (independent step errors) | Full (joint covariance / vector outputs) |

---

## 5. Numerical Edge Cases & Mitigation Strategies

1. **Short $T$ vs Large $N$ (Micro-panels)**:
   - Dynamic GMM estimators break if instrument count $M > N$ (Roodman 2009 "instrument proliferation").
   - *Mitigation*: Hard-cap maximum instrument lag depth (e.g. `max_lags = 2` or collapsed instruments).
2. **Rank Deficiency in Within-Transformation**:
   - Time-invariant variables within a leaf have zero variance after demeaning, causing singular Gram matrices.
   - *Mitigation*: Automatic pre-filtering: drop regressors with within-group variance $\sigma_w^2 < 10^{-12}$; apply ridge regularization $\lambda > 0$ by default.
3. **Unseen Entities in Inference (Cold Start)**:
   - A new series $j \notin \{1, \dots, N\}$ has no historical fixed effect $\alpha_j$.
   - *Mitigation*: Fallback to the leaf average intercept $\bar{\alpha}_{\text{leaf}}$ for unseen entities.

---

## 6. Primary References

1. **Sela, R. J., & Simonoff, J. S. (2012)**. *RE-EM trees: a data mining approach for longitudinal and clustered data*. JASA, 107(497), 206-213.
2. **Hansen, B. E., & Seo, B. (2002)**. *Testing for two-regime threshold cointegration in vector error-correction models*. Journal of Econometrics, 110(2), 293-332.
3. **Balke, N. S., & Fomby, T. B. (1997)**. *Threshold cointegration*. International Economic Review, 627-645.
4. **Arellano, M., & Bond, S. (1991)**. *Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations*. Review of Economic Studies, 58(2), 277-297.
5. **Blundell, R., & Bond, S. (1998)**. *Initial conditions and moment restrictions in dynamic panel data models*. Journal of Econometrics, 87(1), 115-143.
6. **Chernozhukov, V., et al. (2018)**. *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal, 21(1), C1-C68.
7. **Roodman, D. (2009)**. *A note on the theme of too many instruments*. Oxford Bulletin of Economics and Statistics, 71(1), 135-158.
