# Comprehensive Evaluation Metrics Taxonomy: Uplift, Survival, and Time Series Forecasting

**Date**: 2026-11-18  
**Author**: Antigravity (Google DeepMind) & mltreelib Maintainers  
**Status**: Technical Architecture & Research Specification  
**Deliverable**: Architectural blueprint for expanding `mltreelib.Metrics` / `crates/mltreelib-core/src/metrics.rs` into specialized learning domains.

---

## 1. Executive Summary & Architectural Thesis

Tree-based machine learning extends far beyond classical i.i.d. homoscedastic regression and binary classification. In modern tabular applications, practitioners deploy tree models into three high-value specialized regimes:
1. **Causal & Uplift Modeling**: Measuring incremental treatment effects $\tau(x) = E[Y \mid X=x, W=1] - E[Y \mid X=x, W=0]$ under randomized or observational treatment assignments $W \in \{0, 1\}$.
2. **Survival Analysis**: Modeling duration and time-to-event outcomes under right-censoring $(T_i, E_i)$, where unobserved event times invalidate standard MSE and Cross-Entropy loss metrics.
3. **Time Series & Forecasting**: Evaluating sequential forecasts across multi-horizon steps where point errors fail on non-stationary regimes, zero-inflated demand, and distributional interval risks.

Currently, `mltreelib` provides extensive regression and classification suites in `crates/mltreelib-core/src/metrics.rs` (`Metrics.regression`, `Metrics.classification`, ROC, PR, Gain, Lift, Decile tables, KS statistic, PSI, Brier, Kappa). However, survival, uplift, and time series metrics are currently scattered in example scripts (`examples/survival_uplift_evaluation.py`) or absent.

This document establishes the mathematical foundations, formal algorithms, edge cases, and API specifications for elevating `mltreelib.Metrics` to a unified, production-grade evaluation engine for uplift, survival, and time series domains.

---

## 2. Uplift & Causal Evaluation Metrics

### 2.1 Theoretical Setup
Let unit $i \in \{1, \dots, N\}$ have features $X_i \in \mathbb{R}^P$, binary treatment assignment $W_i \in \{0, 1\}$, and observed outcome $Y_i \in \mathbb{R}$ (or $Y_i \in \{0, 1\}$). Under the Rubin Causal Model, potential outcomes are $(Y_i(0), Y_i(1))$, with observed $Y_i = W_i Y_i(1) + (1 - W_i) Y_i(0)$.
The model outputs predicted Individual Treatment Effect (ITE) / conditional uplift $\hat{\tau}_i = \hat{\tau}(X_i)$.

### 2.2 Cumulative Uplift Curve & AUUC
Sort test units descending by predicted uplift $\hat{\tau}_{(1)} \ge \hat{\tau}_{(2)} \ge \dots \ge \hat{\tau}_{(N)}$.
For any cutoff fraction or rank $k \in \{1, \dots, N\}$:
- $N_T(k) = \sum_{i=1}^k W_{(i)}$ (treated units in top $k$)
- $N_C(k) = \sum_{i=1}^k (1 - W_{(i)})$ (control units in top $k$)
- $Y_T(k) = \sum_{i=1}^k W_{(i)} Y_{(i)}$ (total treated response)
- $Y_C(k) = \sum_{i=1}^k (1 - W_{(i)}) Y_{(i)}$ (total control response)

The **Cumulative Uplift** (mean treatment difference) at depth $k$ is:
$$\text{Uplift}(k) = \left( \frac{Y_T(k)}{N_T(k)} - \frac{Y_C(k)}{N_C(k)} \right)$$

The **Cumulative Gain** curve tracks total incremental gains:
$$\text{Gain}(k) = \left( \frac{Y_T(k)}{N_T(k)} - \frac{Y_C(k)}{N_C(k)} \right) \cdot (N_T(k) + N_C(k))$$

The **AUUC (Area Under the Uplift Curve)** is the trapezoidal integral of $\text{Gain}(k)$ plotted against population fraction $k/N \in [0, 1]$.

### 2.3 Qini Curve & Radcliffe Qini Coefficient
Radcliffe (2007) introduced the Qini curve to measure cumulative incremental positive outcomes without dividing by $N_T(k)$:
$$Q(k) = Y_T(k) - Y_C(k) \cdot \frac{N_T(k)}{N_C(k)}$$
- At $k = 0$, $Q(0) = 0$.
- At $k = N$, $Q(N) = Y_T(N) - Y_C(N) \frac{N_T(N)}{N_C(N)}$ (the overall average incremental effect across the entire population).
- **Random Baseline**: The straight diagonal segment connecting $(0, 0)$ to $(N, Q(N))$.
- **Qini Coefficient ($q$)**: The area between the empirical Qini curve and the random diagonal.
- **Normalized Qini ($q_{\text{norm}}$)**:
  $$q_{\text{norm}} = \frac{q_{\text{model}}}{q_{\text{optimal}}}$$
  where the optimal curve places all treated responders and control non-responders first. $q_{\text{norm}} \in [-1, 1]$.

### 2.4 Expected Policy Value & Doubly Robust Value
Given a targeted decision rule $\pi(X) \in \{0, 1\}$ (e.g. treat if $\hat{\tau}(X) > \text{threshold}$):
1. **Horvitz-Thompson Inverse Propensity Weighting (IPW)**:
   $$\hat{V}_{\text{IPW}}(\pi) = \frac{1}{N} \sum_{i=1}^N \left( \frac{W_i Y_i}{e(X_i)} \mathbf{1}\{\pi(X_i) = 1\} + \frac{(1 - W_i) Y_i}{1 - e(X_i)} \mathbf{1}\{\pi(X_i) = 0\} \right)$$
   where $e(X) = P(W=1 \mid X)$ is the propensity score. In randomized trials ($A/B$ tests), $e(X_i) = N_T / N$.
2. **Doubly Robust (DR) Policy Value (Dudík, Langford & Li 2011)**:
   Augments IPW with baseline outcome models $\hat{\mu}_1(X) = E[Y \mid X, W=1]$ and $\hat{\mu}_0(X) = E[Y \mid X, W=0]$:
   $$\hat{V}_{\text{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[ \pi(X_i)\hat{\tau}_{\text{DR}, i} + \hat{\mu}_0(X_i) \right]$$
   $$\hat{\tau}_{\text{DR}, i} = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{W_i (Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1 - W_i)(Y_i - \hat{\mu}_0(X_i))}{1 - e(X_i)}$$
   $\hat{V}_{\text{DR}}$ is asymptotically unbiased if either the propensity score OR the outcome models are consistently estimated.

### 2.5 Net Information Value (NIV) & Net Weight of Evidence
For discrete binned features or deciles $b \in \{1, \dots, B\}$:
$$\text{NWoE}_b = \ln\left( \frac{P(Y=1 \mid W=1, b)}{P(Y=0 \mid W=1, b)} \right) - \ln\left( \frac{P(Y=1 \mid W=0, b)}{P(Y=0 \mid W=0, b)} \right)$$
$$\text{NIV} = \sum_{b=1}^B \left( P(b \mid Y=1, W=1) - P(b \mid Y=1, W=0) \right) \cdot \text{NWoE}_b$$

---

## 3. Survival Analysis Evaluation Metrics

### 3.1 Right-Censoring Framework
Observed triplets $(X_i, T_i, E_i)$ where $T_i = \min(T_i^*, C_i)$ is observed duration, $T_i^*$ is true survival time, $C_i$ is censoring time, and $E_i = \mathbf{1}\{T_i^* \le C_i\}$.
The censoring survival function $G(t) = P(C > t)$ is estimated via Kaplan-Meier on $(T_i, 1 - E_i)$.

### 3.2 Concordance Index (C-Index)
Evaluates rank discrimination: does the model assign higher risk scores $\hat{r}_i$ (or lower predicted survival probabilities) to patients who fail earlier?

1. **Harrell's C-index (Harrell et al. 1982)**:
   Considers all pairs $(i, j)$ with comparable outcomes:
   $$\text{Pair } (i, j) \text{ comparable if } (T_i < T_j \text{ and } E_i = 1)$$
   $$\mathcal{C}_{\text{Harrell}} = \frac{\sum_{i, j} \mathbf{1}\{T_i < T_j, E_i = 1\} \cdot \left[ \mathbf{1}\{\hat{r}_i > \hat{r}_j\} + 0.5 \cdot \mathbf{1}\{\hat{r}_i = \hat{r}_j\} \right]}{\sum_{i, j} \mathbf{1}\{T_i < T_j, E_i = 1\}}$$
   *Limitation*: Biased upwards in the presence of heavy censoring because long survivors are preferentially censored and omitted from pairs.
2. **Uno's IPCW C-index (Uno et al. 2011)**:
   Applies Inverse Probability of Censoring Weighting (IPCW) to reweight comparable pairs by the probability of both individuals being uncensored at time $T_i$:
   $$\mathcal{C}_{\text{Uno}} = \frac{\sum_{i, j} \frac{1}{\hat{G}(T_i)^2} \mathbf{1}\{T_i < T_j, E_i = 1\} \cdot \mathbf{1}\{\hat{r}_i > \hat{r}_j\}}{\sum_{i, j} \frac{1}{\hat{G}(T_i)^2} \mathbf{1}\{T_i < T_j, E_i = 1\}}$$
   $\mathcal{C}_{\text{Uno}}$ is asymptotically independent of the censoring distribution $G(t)$.

### 3.3 Time-Dependent Brier Score & Integrated Brier Score (IBS)
Graf et al. (1999) generalized the Brier score to survival probabilities $\hat{S}(t \mid x_i)$ at a fixed evaluation horizon $t$:
$$\text{BS}(t) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{(0 - \hat{S}(t \mid x_i))^2 \cdot \mathbf{1}\{T_i \le t, E_i = 1\}}{\hat{G}(T_i^-)} + \frac{(1 - \hat{S}(t \mid x_i))^2 \cdot \mathbf{1}\{T_i > t\}}{\hat{G}(t)} \right]$$
Units censored prior to $t$ ($T_i \le t, E_i = 0$) receive weight 0 because their survival status at $t$ is unknown; IPCW weights on the remaining terms rebalance the sample.

**Integrated Brier Score (IBS)** across time horizon $[0, t_{\max}]$:
$$\text{IBS}(t_{\max}) = \frac{1}{t_{\max}} \int_0^{t_{\max}} \text{BS}(t) \, dt \approx \frac{1}{K} \sum_{k=1}^K \text{BS}(t_k)$$

### 3.4 Cumulative / Dynamic Time-Dependent ROC & AUC
Heagerty, Lumley & Pepe (2000):
- **Sensitivity at $t$**: $P(\hat{r} > c \mid T \le t, E = 1) = \frac{\sum_{i: T_i \le t, E_i=1} \mathbf{1}\{\hat{r}_i > c\} / \hat{G}(T_i)}{\sum_{i: T_i \le t, E_i=1} 1 / \hat{G}(T_i)}$
- **Specificity at $t$**: $P(\hat{r} \le c \mid T > t) = \frac{\sum_{i: T_i > t} \mathbf{1}\{\hat{r}_i \le c\}}{\sum_{i: T_i > t} 1}$
Integrating the resulting ROC curve gives $\text{AUC}(t)$, quantifying how well the model discriminates cases failing by time $t$ from controls surviving past $t$.

---

## 4. Time Series & Sequential Forecasting Metrics

### 4.1 Scale-Free & Relative Metrics

1. **Mean Absolute Scaled Error (MASE) (Hyndman & Koehler 2006)**:
   Divides the test MAE by the in-sample naive persistence MAE (or seasonal naive persistence MAE with seasonal lag $s$):
   $$\text{MASE} = \frac{\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|}{\frac{1}{N_{\text{train}} - s} \sum_{t=s+1}^{N_{\text{train}}} |y_t - y_{t-s}|}$$
   - $\text{MASE} < 1.0$: Model outperforms the seasonal naive baseline.
   - $\text{MASE} > 1.0$: Model is worse than naive carry-forward.
   - Scale-invariant, never diverges on zero targets $y_t = 0$ (unlike MAPE), and symmetrical.
2. **Weighted Absolute Percentage Error (WAPE)**:
   $$\text{WAPE} = \frac{\sum_{t=1}^H |y_t - \hat{y}_t|}{\sum_{t=1}^H |y_t|}$$
   Standard in inventory and supply chain retail forecasting (e.g. M5, Walmart). Avoids the small-denominator explosion of MAPE.
3. **Symmetric MAPE (sMAPE)**:
   $$\text{sMAPE} = \frac{100\%}{H} \sum_{t=1}^H \frac{2 \cdot |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$$
   Bounded in $[0\%, 200\%]$.

### 4.2 Probabilistic & Distributional Metrics

1. **Continuous Ranked Probability Score (CRPS) (Gneiting & Raftery 2007)**:
   Strictly proper scoring rule for probabilistic forecasts evaluating CDF $F(z)$ against true observation $y$:
   $$\text{CRPS}(F, y) = \int_{-\infty}^\infty (F(z) - \mathbf{1}\{z \ge y\})^2 \, dz$$
   - **Gaussian Forecast $\mathcal{N}(\mu, \sigma^2)$ Closed Form**:
     $$\text{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma \left[ \frac{y - \mu}{\sigma} \left( 2\Phi\left(\frac{y - \mu}{\sigma}\right) - 1 \right) + 2\phi\left(\frac{y - \mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right]$$
     This allows instant $O(1)$ exact scoring on `ModelTree(leaf_family="heteroscedastic")` output!
   - **Quantile Grid Decomposition**:
     Given quantile predictions $\hat{q}_{\alpha_1}, \dots, \hat{q}_{\alpha_K}$ for grid $\alpha_k \in (0, 1)$:
     $$\text{CRPS} \approx \frac{2}{K} \sum_{k=1}^K \rho_{\alpha_k}(y - \hat{q}_{\alpha_k})$$
2. **Pinball / Quantile Loss ($\rho_\alpha$)**:
   $$\rho_\alpha(u) = u \cdot (\alpha - \mathbf{1}\{u < 0\}) = \max(\alpha u, (\alpha - 1) u)$$
   Used to evaluate asymmetric quantile forecasts (e.g., conditional Value-at-Risk, peak load demand).
3. **Winkler Interval Score**:
   Evaluates a central $(1 - \alpha)$ prediction interval $[L, U]$ (e.g., $90\%$ interval with $\alpha = 0.10$):
   $$\text{Winkler}(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y) \cdot \mathbf{1}\{y < L\} + \frac{2}{\alpha}(y - U) \cdot \mathbf{1}\{y > U\}$$
   Directly penalizes both wide intervals and boundary violations.

### 4.3 Directional & Movement Metrics

1. **Mean Directional Accuracy (MDA)**:
   $$\text{MDA} = \frac{1}{H - 1} \sum_{t=2}^H \mathbf{1}\{\text{sgn}(y_t - y_{t-1}) == \text{sgn}(\hat{y}_t - y_{t-1})\}$$
   Measures accuracy in forecasting the sign of market returns, price momentum, or trend transitions.

---

## 5. Architectural Integration into `mltreelib.Metrics`

### 5.1 Rust Core Additions (`crates/mltreelib-core/src/metrics.rs`)

```rust
// Uplift Report
pub struct UpliftReport {
    pub n_treated: usize,
    pub n_control: usize,
    pub ate: f64,
    pub qini_coef: f64,
    pub qini_norm: f64,
    pub auuc: f64,
    pub top_decile_uplift: f64,
    pub qini_curve: Vec<(f64, f64)>, // (fraction, qini)
}

pub fn uplift_report(
    y: &[f64],
    treatment: &[f64],
    tau_pred: &[f64],
) -> Result<UpliftReport, String>;

// Survival Report
pub struct SurvivalReport {
    pub harrell_c: f64,
    pub uno_c: f64,
    pub ibs: f64,
    pub brier_at_horizons: Vec<(f64, f64)>, // (time, brier)
}

pub fn survival_report(
    durations: &[f64],
    events: &[f64],
    risk_scores: &[f64],
    surv_curves: Option<&[Vec<f64>]>,
    time_grid: Option<&[f64]>,
) -> Result<SurvivalReport, String>;

// Time Series Report
pub struct TimeSeriesReport {
    pub mae: f64,
    pub rmse: f64,
    pub mase: Option<f64>,
    pub smape: f64,
    pub wape: f64,
    pub mda: Option<f64>,
    pub crps_gaussian: Option<f64>,
}

pub fn time_series_report(
    y_true: &[f64],
    y_pred: &[f64],
    y_train: Option<&[f64]>,
    seasonal_period: Option<usize>,
    std_pred: Option<&[f64]>,
) -> Result<TimeSeriesReport, String>;
```

### 5.2 Python Surface (`Metrics`)

```python
from mltreelib import Metrics

m = Metrics()

# 1. Uplift Evaluation
up = m.uplift(y_true, treatment, tau_pred)
# -> {'qini': 0.142, 'qini_norm': 0.681, 'auuc': 0.235, 'top_decile_uplift': 0.084, 'qini_curve': [...]}

# 2. Survival Evaluation
surv = m.survival(durations, events, risk_scores, surv_probs=None, time_grid=None)
# -> {'harrell_c': 0.742, 'uno_c': 0.738, 'ibs': 0.162, 'brier_grid': [...]}

# 3. Time Series Evaluation
ts = m.time_series(y_true, y_pred, y_train=y_train, season=12, y_std=y_std)
# -> {'mase': 0.812, 'smape': 12.4, 'wape': 0.095, 'mda': 0.68, 'crps': 2.14}
```

---

## 6. Citations & Theoretical Provenance

1. **Radcliffe, N. J. (2007)**. *Using control groups to target on predicted lift: Building and assessing uplift models*. Direct Marketing Analytics Journal, 3:14–21.
2. **Dudík, M., Langford, J., & Li, L. (2011)**. *Doubly robust policy evaluation and learning*. ICML 2011.
3. **Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982)**. *Evaluating the yield of medical tests*. JAMA, 247(18):2543–2546.
4. **Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B., & Wei, L. J. (2011)**. *On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data*. Statistics in Medicine, 30(10):1105–1117.
5. **Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999)**. *Assessment and comparison of prognostic classification schemes for survival data*. Statistics in Medicine, 18(17-18):2529–2545.
6. **Hyndman, R. J., & Koehler, A. B. (2006)**. *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4):679–688.
7. **Gneiting, T., & Raftery, A. E. (2007)**. *Strictly proper scoring rules, prediction, and estimation*. JASA, 102(477):359–378.
