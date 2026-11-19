# Deep Research: SOTA Survival and Competing Risks Trees & Forests

**Date**: 2026-11-19  
**Author**: Jitin Kapila & Antigravity  
**Sources**: arXiv (via treg/crawl4ai), Papers With Code, AISTATS 2025, AAAI 2024, randomForestSRC, scikit-survival  
**Scope**: Algorithmic foundation, splitting rules, parametric/non-parametric leaves, competing risks CIF, and benchmark datasets for Spec 10.

---

## 1. Executive Summary & Recent Breakthroughs (2024–2026)

Survival analysis models right-censored time-to-event outcomes $(T_i, E_i)$. In competing risks, subjects face $J > 1$ mutually exclusive endpoints ($E_i \in \{0, 1, \dots, J\}$), where the occurrence of event $k$ precludes the occurrence of event $j \neq k$.

Recent literature reveals three major breakthroughs beyond legacy heuristics (like basic greedy log-rank or naive right-censoring):
1. **Proper Scoring Rules & Separable Losses (AISTATS 2025, Alberge et al., *SurvivalBoost*)**:
   - Classic survival models couple tree architecture with global partial likelihood, preventing stochastic optimization and mini-batch evaluation.
   - Introducing a strictly proper, censoring-adjusted separable scoring rule via Inverse Probability of Censoring Weights (IPCW) unlocks exact gradient boosting and per-sample additive splitting for competing risks.
2. **Optimal Survival Trees (AAAI 2024, Huisman et al.)**:
   - Formulates survival tree construction via dynamic programming with exact optimality guarantees for shallow depths, bounding the optimality gap of greedy heuristics.
3. **Competing Risks Calibration & Benchmarking (2026, Alberge et al., Sierra et al.)**:
   - Demonstrates that traditional C-index alone fails to detect severe probability miscalibration across multiple competing causes and time horizons.
   - Unified evaluation requires time-dependent IPCW Brier score matrices and cause-specific dynamic concordance.

---

## 2. Core Splitting Criteria Matrix

| Criterion | Mathematical Formulation | Time Complexity | Target Objective | Primary Reference |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Two-Sample Log-Rank** | $Z^2 = \frac{[\sum_{k=1}^K (d_{L,k} - Y_{L,k} \frac{d_k}{Y_k})]^2}{\sum_{k=1}^K \frac{Y_{L,k} Y_{R,k} d_k (Y_k - d_k)}{Y_k^2 (Y_k - 1)}}$ | $O(N \log N + N \cdot M)$ | Non-parametric survival curve separation under proportional hazards. | LeBlanc & Crowley (1993); Ciampi et al. (1986) |
| **Martingale Residual SSE Reduction** | $\text{Gain} = \frac{(\sum_L M_i)^2}{N_L} + \frac{(\sum_R M_i)^2}{N_R} - \frac{(\sum_P M_i)^2}{N_P}$ where $M_i = E_i - \hat{\Lambda}_P(T_i)$ | **$O(N)$ streaming scan** | Fast approximation of log-rank; exact first-order score residual variance reduction. | LeBlanc & Crowley (1992); Therneau et al. (1990) |
| **Weighted Log-Rank (Tarone-Ware / Peto)** | $Z_w^2 = \frac{[\sum_k w_k (d_{L,k} - e_{L,k})]^2}{\sum_k w_k^2 v_{L,k}}$ where $w_k = \sqrt{Y_k}$ or $w_k = \hat{S}(t_k)$ | $O(N \log N + N \cdot M)$ | Prioritizes early divergence when sample sizes are largest; robust to late crossing hazards. | Tarone & Ware (1977); Prentice (1978) |
| **Gray's Test for Competing Risks (`logrankCR`)** | Tests equality of subdistribution hazards $h_j^*(t)$ across partitions without treating competing risks as independent censoring. | $O(N \log N + N \cdot M)$ | Differentiates cause-specific Cumulative Incidence Functions (CIF). | Gray (1988); Ishwaran et al. (2014) |
| **Proper Separable Scoring Rule (SurvivalBoost)** | $\ell(t, \delta, \hat{P}) = \sum_j \frac{I(T \le t, E=j)}{\hat{G}(T)} (1 - \hat{P}_j(t))^2 + \frac{I(T > t)}{\hat{G}(t)} \hat{P}_j(t)^2$ | $O(N)$ per horizon | Minimizes calibrated proper scoring rule for competing risks across time horizons $t$. | Alberge et al. (AISTATS 2025, arXiv:2410.16765) |
| **Harrell's C-Index Concordance Split** | $C = \frac{\sum_{i < j} I(T_i < T_j) I(\hat{R}_i > \hat{R}_j) E_i}{\sum_{i < j} I(T_i < T_j) E_i}$ | $O(N \log N)$ via Fenwick tree | Maximizes order rank concordance; distribution-free, immune to non-proportional hazards. | Harrell et al. (1982); Ishwaran et al. (2008) |

---

## 3. Leaf Model Families for Spec 10

### 3.1 Non-Parametric Curves
1. **Nelson-Aalen & Kaplan-Meier (`leaf="survival"`)**:
   - Computes empirical step survival curves:
     $$\hat{S}(t) = \prod_{t_k \le t} \left(1 - \frac{d_k}{Y_k}\right), \quad \hat{\Lambda}(t) = \sum_{t_k \le t} \frac{d_k}{Y_k}$$
   - Fast $O(K)$ query evaluation via binary search.

2. **Aalen-Johansen Competing Risks CIF (`leaf="competing_risks"`)**:
   - For $J$ competing causes:
     $$\hat{F}_j(t) = \sum_{t_k \le t} \hat{S}(t_{k-1}) \frac{d_{j,k}}{Y_k}$$
   - Guarantees $\sum_{j=1}^J \hat{F}_j(t) + \hat{S}(t) = 1$. Completely prevents the 20% to 50% overestimation of cumulative risk caused by 1-KM naive estimators.

### 3.2 Parametric Accelerated Failure Time (AFT) Leaves (`leaf="aft"`)
- Formulates log-linear survival time inside terminal leaves:
  $$\ln(T_i) = \alpha + \mathbf{x}_i^T \boldsymbol{\beta} + \sigma \epsilon_i$$
- Distributions:
  1. **Weibull** ($\epsilon \sim \text{Gumbel}$ / Extreme Value):
     - Log-likelihood:
       $$\ell(\boldsymbol{\beta}, \sigma) = \sum_{i=1}^n \left[ E_i \left( -\ln \sigma + z_i \right) - \exp(z_i) \right], \quad z_i = \frac{\ln T_i - (\alpha + \mathbf{x}_i^T \boldsymbol{\beta})}{\sigma}$$
  2. **Lognormal** ($\epsilon \sim \mathcal{N}(0, 1)$):
     - Uses standard Gaussian survival functions $\Phi(z)$ and $\phi(z)$.
  3. **Log-logistic** ($\epsilon \sim \text{Logistic}$):
     - Heavy-tailed, non-monotonic hazard curves.
- **Solver**: Pure-Rust L-BFGS (`crates/mltreelib-core/src/solver/lbfgs.rs`) minimizing negative log-likelihood with analytic gradients and line search.

### 3.3 Semi-Parametric Cox Proportional Hazards Leaves (`leaf="cox"`)
- Fits leaf-level covariate adjustments:
  $$h(t \mid \mathbf{x}) = h_0(t) \exp(\mathbf{x}^T \boldsymbol{\beta})$$
- Solves Breslow / Efron partial likelihood using Newton-Raphson or active-set coordinate descent:
  $$\ell(\boldsymbol{\beta}) = \sum_{k=1}^K \left[ \mathbf{s}_k^T \boldsymbol{\beta} - d_k \ln \sum_{i \in R_k} \exp(\mathbf{x}_i^T \boldsymbol{\beta}) \right]$$

---

## 4. Evaluation Metrics & Benchmark Protocol

### 4.1 Evaluation Suite
1. **Harrell's C-Index**: Unweighted concordance on observed failure pairs.
2. **Uno's IPCW C-Index**: Asymptotically unbiased concordance independent of censoring distribution:
   $$C_{\text{Uno}} = \frac{\sum_{i < j} I(T_i < T_j) I(\hat{R}_i > \hat{R}_j) \frac{E_i}{\hat{G}(T_i)^2}}{\sum_{i < j} I(T_i < T_j) \frac{E_i}{\hat{G}(T_i)^2}}$$
3. **Integrated Brier Score (IBS)**: Time-integrated IPCW squared probability error across $[0, t_{\max}]$.
4. **Cause-Specific Cumulative Incidence Brier Score**: Validates multi-cause probabilities against realized binary indicators.

### 4.2 Standard Canonical Datasets
1. **VA Lung Cancer (`veteran`)**: 137 patients, right-censoring, strong performance status predictors.
2. **Primary Biliary Cirrhosis (`pbc`)**: 418 patients, competing risks (transplant vs death vs censored).
3. **Monoclonal Gammopathy (`mgus2`)**: 1,384 patients, competing risks (progression to plasma cell malignancy vs death).
4. **German Breast Cancer Study Group (`gbsg2`)**: 686 patients, recurrence-free survival.
5. **Bank Churn / Mechanical Failure**: Industry-scale time-to-attrition benchmarks with high censoring (>80%).

---

## 5. Architectural Implementation Blueprint in `mltreelib`

1. **`crates/mltreelib-core/src/survival/` (New Submodule)**:
   - `logrank.rs`: Exact two-sample Log-Rank and weighted Tarone-Ware/Peto scan logic.
   - `gray.rs`: Gray's subdistribution CIF test for competing risks.
   - `aft.rs`: Accelerated Failure Time loss, gradient, and Hessian for Weibull/Lognormal leaves.
   - `cif.rs`: Aalen-Johansen multi-state cumulative incidence calculator.
2. **`crates/mltreelib-core/src/split.rs`**:
   - Plug `LogRank` and `Gray` into the parallel `SplitScratch` feature scanner.
   - Wire $O(N)$ streaming martingale scans for instant baseline splitting.
3. **`crates/mltreelib-core/src/leaf.rs` & `node_stats.rs`**:
   - Extend `NodeModel` to store `AftModel` and `AalenJohansenModel`.
4. **`crates/mltreelib-py/src/lib.rs` & `python/mltreelib/`**:
   - `predict_survival(X, times)` -> matrix $[N \times T]$.
   - `predict_cumulative_hazard(X, times)` -> matrix $[N \times T]$.
   - `predict_cif(X, times, cause)` -> matrix $[N \times T]$.
