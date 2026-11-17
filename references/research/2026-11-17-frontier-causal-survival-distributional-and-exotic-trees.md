# Frontier Tree Architectures: Causal/Uplift, Survival, Multiparameter GAMLSS Leaves, and Exotic Splitting Criteria

**Date**: 2026-11-17  
**Author**: Jitin Kapila & Antigravity  
**Context**: Technical research survey identifying remaining capabilities to implement in `mltreelib` across Causal/Uplift trees, Survival models, GAMLSS distributional leaf families, and exotic statistical splitting criteria.

---

## Executive Summary & Landscape Map

`mltreelib` has established a modern, pure-Rust tree core with zero scikit-learn dependency, first-class missing data handling, Quinlan lineage modernizations (M5 model trees, C5.0 rulesets, Cubist k-NN neighbor corrections, GritBot), and foundational distributional criteria (`tweedie`, `heteroscedastic`, `tobit`, `cv`, `martingale`).

To extend this engine to a comprehensive research-grade system that outperforms both flat tabular foundation models and fragmented legacy packages (`partykit`, `grf`, `causalTree`, `randomForestSRC`, `scikit-survival`, `econml`, `causalml`, `disttree`), we identify four major frontier pillars:

```
                                  mltreelib Frontier Landscape
                                                 │
         ┌───────────────────┬───────────────────┴───────────────────┬───────────────────┐
         ▼                   ▼                                       ▼                   ▼
1. Causal & Uplift    2. Survival & Time-to-Event             3. Distributional GAMLSS 4. Exotic Splitting
   • Honest CATE         • AFT Parametric Leaves (Weibull/LogNorm)• Negative Binomial    • MOB Score-Fluctuation
   • Doubly Robust       • Cox Proportional Hazards Leaves       • Beta Regression       • CTree Rank Permutation
   • R-Learner           • Competing Risks (Fine-Gray / CIF)     • Student's t Robust    • MMD / Energy Distance
   • IV / 2SLS Trees     • C-Index & Brier Score Splits          • Ordinal Cumulative    • Wasserstein / OT
   • Divergence Splits   • Weighted Log-Rank (Tarone-Ware)       • GEV / Pareto Extremes • Constrained Fair Splits
```

---

## 1. Causal & Uplift Modeling Frontier

Traditional decision trees partition feature space to minimize outcome variance ($Y$). Causal and uplift trees partition feature space to maximize **treatment effect heterogeneity** ($\tau(X) = E[Y(1) - Y(0) \mid X]$).

### 1.1 Leaf Models

| Causal Leaf Model | Estimand / Formulation | Advantages & Tradeoffs | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Constant CATE ($\bar{\tau}$)** *(Current baseline)* | $\hat{\tau} = \bar{Y}_{T=1} - \bar{Y}_{T=0}$ | Simple, fast; assumes constant effect within leaf. | Athey & Imbens (2016) |
| **Local Linear CATE ($\tau(X) = \beta^T X$)** | Regress $Y_i = \alpha_0 + \beta_0^T X_i + W_i (\alpha_1 + \tau^T X_i)$ | Captures slope of treatment effect inside leaves; eliminates step-function bias. | Friedberg et al. (2020) *Local Linear Forests* |
| **Doubly Robust (AIPW) Leaf** | Regress pseudo-outcome $\Gamma_i$ on $X$ inside leaf: $\Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{W_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}$ | **Double robustness**: consistent if *either* propensity score $\hat{e}(X)$ or outcome model $\hat{\mu}(X)$ is correctly specified. | Robins & Rotnitzky (1995); Kennedy (2023) |
| **R-Learner (Robinson Residual) Leaf** | Fit $\min_\tau \sum w_i ( (Y_i - \hat{m}(X_i)) - \tau (W_i - \hat{e}(X_i)) )^2$ with sample weight $(W_i - \hat{e}(X_i))^2$ | Quasi-oracle efficiency under Neyman orthogonality; immune to first-stage regularization bias. | Nie & Wager (2021) |
| **Instrumental Variable (IV / 2SLS) Leaf** | Wald estimator: $\hat{\tau}_{\text{IV}} = \frac{\widehat{\text{Cov}}(Y, Z)}{\widehat{\text{Cov}}(W, Z)}$ with instrument $Z$ | Identifies Local Average Treatment Effect (LATE) under unmeasured confounding. | Angrist & Imbens (1995); Athey et al. (2019) *GRF IV* |
| **Continuous Dose-Response Leaf** | Local polynomial derivative $\hat{\tau}(w, x) = \frac{\partial E[Y \mid W=w, X=x]}{\partial w}$ | Continuous policy optimization (pricing elasticity, drug dosing). | Kennedy et al. (2017) |

### 1.2 Splitting Criteria for Causal & Uplift Trees

1. **Honest CATE Variance Maximization (Athey & Imbens 2016)**:
   $$\text{Gain}_{\text{Causal}} = \frac{N_L N_R}{N_P} (\hat{\tau}_L - \hat{\tau}_R)^2 - \lambda \left( \frac{S_{T, L}^2}{N_{T, L}} + \frac{S_{C, L}^2}{N_{C, L}} + \frac{S_{T, R}^2}{N_{T, R}} + \frac{S_{C, R}^2}{N_{C, R}} \right)$$
   Penalizes candidate leaves that have high within-leaf outcome variance or unbalanced treatment/control counts.
2. **Divergence-Based Uplift Splitting (Rzepakowski & Jaroszewicz 2012)**:
   Used extensively in industrial marketing (Wayfair, Uber):
   - **Kullback-Leibler (KL) Divergence**: $D_{\text{KL}}(P(Y \mid T) \parallel P(Y \mid C))$
   - **Euclidean Distance (ED)**: $\sum_y (P(y \mid T) - P(y \mid C))^2$
   - **Chi-Square Distance ($\chi^2$)**: $\sum_y \frac{(P(y \mid T) - P(y \mid C))^2}{P(y \mid C)}$
   $$\text{Gain}_{\text{Uplift}} = D(L) \frac{N_L}{N_P} + D(R) \frac{N_R}{N_P} - D(P)$$
3. **R-Score / Neyman-Orthogonal Split**:
   Using residualized outcome $\tilde{Y}_i = Y_i - \hat{m}(X_i)$ and residualized treatment $\tilde{W}_i = W_i - \hat{e}(X_i)$, score split as weighted SSE reduction on $\tilde{Y}_i / \tilde{W}_i$ with weights $\tilde{W}_i^2$.

---

## 2. Survival & Time-to-Event Frontier

Survival data is fundamentally characterized by right, left, or interval censoring: an observed time $T_i$ with event indicator $E_i \in \{0, 1\}$.

### 2.1 Survival Leaf Models

| Survival Leaf Model | Formulation | Output Function | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Nelson-Aalen / Kaplan-Meier** *(Current baseline)* | $\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right), \quad \hat{\Lambda}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}$ | Non-parametric survival curve $\hat{S}(t)$ and cumulative hazard $\hat{\Lambda}(t)$. | Ishwaran et al. (2008) *RSF* |
| **Accelerated Failure Time (AFT) Linear Leaf** | $\ln(T_i) = \alpha + \beta^T X_i + \sigma \epsilon_i$ where $\epsilon \sim \text{Gumbel}$ (Weibull), $\text{Normal}$, or $\text{Logistic}$ | Direct linear prediction of survival time with parametric distribution. | Stute (1993); Buckley & James (1979) |
| **Piecewise Exponential / Poisson Leaf** | Treat intervals $k$ as Poisson: $E_{ik} \sim \text{Poisson}(\lambda_k \Delta t_{ik})$ with $\ln(\Delta t_{ik})$ offset | Fast GLM link to baseline hazard rates. | Holford (1980); Friedman (1982) |
| **Cox Proportional Hazards Leaf** | $h(t \mid X) = h_0(t) \exp(\beta^T X)$ fitted via Breslow/Efron partial likelihood | Hazard ratio $\exp(\beta^T X)$ relative to leaf baseline. | Ciampi et al. (1986); Therneau et al. (1990) |
| **Competing Risks Leaf (Fine-Gray / CIF)** | Subdistribution hazard: $h_1^*(t \mid X) = h_{10}^*(t) \exp(\beta^T X)$ | Cause-specific Cumulative Incidence Functions $F_1(t), F_2(t)$ where $\sum F_k(t) + S(t) = 1$. | Fine & Gray (1999); Ishwaran et al. (2014) |
| **Mixture Cure Model Leaf** | $S(t \mid X) = \pi(X) + (1 - \pi(X)) S_u(t \mid X)$ | Probability of permanent cure $\pi(X) \in [0, 1]$ vs latency $S_u(t)$ for uncured patients. | Sy & Taylor (2000) |

### 2.2 Survival Splitting Criteria

1. **Two-Sample Log-Rank Test (Leblanc & Crowley 1992)**:
   $$L = \frac{\sum_{j=1}^K (d_{j, L} - e_{j, L})}{\sqrt{\sum_{j=1}^K V_{j, L}}}$$
   where $e_{j, L} = n_{j, L} \frac{d_j}{n_j}$ and $V_{j, L} = \frac{n_{j, L} n_{j, R} d_j (n_j - d_j)}{n_j^2 (n_j - 1)}$.
2. **Weighted Log-Rank Tests**:
   - **Tarone-Ware**: Weight $w_j = \sqrt{n_j}$ (balances early vs late differences).
   - **Peto-Prentice**: Weight $w_j = \hat{S}(t_j)$ (emphasizes early separation where sample size is large).
3. **Harrell's Concordance Index (C-Index) Splitting**:
   Directly maximizes rank correlation between predicted leaf risk and observed survival times:
   $$C = \frac{\sum_{i, j} I(T_i < T_j) \cdot I(\hat{R}_i > \hat{R}_j) \cdot E_i}{\sum_{i, j} I(T_i < T_j) \cdot E_i}$$
   Robust when proportional hazards assumptions fail.
4. **Integrated Brier Score (IBS) Reduction Splitting**:
   Evaluates squared distance between predicted survival probability $\hat{S}(t \mid X)$ and true event status using Inverse Probability of Censoring Weights (IPCW):
   $$\text{BS}(t) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{\hat{S}(t \mid X_i)^2 I(T_i \le t, E_i=1)}{\hat{G}(T_i)} + \frac{(1 - \hat{S}(t \mid X_i))^2 I(T_i > t)}{\hat{G}(t)} \right]$$
5. **Gray's Test for Competing Risks**:
   Evaluates differences between cumulative incidence functions (CIF) for the primary event of interest without censoring competing events.

---

## 3. Distributional Leaves & GAMLSS Multiparameter Families

Standard decision trees predict a scalar mean ($\mu$). Real physical, biological, and economic processes exhibit shifting scale ($\sigma$), skewness ($\nu$), or zero-inflation ($\tau$).

### 3.1 GAMLSS Families Roadmap

| Distribution Family | Parameters | Target Support | Link Functions | Practical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Negative Binomial (NB)** | $\mu$ (mean), $\theta$ (dispersion) | Nonnegative integers $\{0, 1, 2, \dots\}$ | $\ln(\mu) = X\beta, \quad \ln(\theta) = \gamma$ | Overdispersed counts (RNA-seq, website hits, insurance frequency) where $\text{Var}(Y) = \mu + \mu^2/\theta$. |
| **Beta Regression** | $\mu$ (mean), $\phi$ (precision) | Open unit interval $(0, 1)$ | $\text{logit}(\mu) = X\beta, \quad \ln(\phi) = \gamma$ | Proportions, percentages, recovery rates, churn probabilities. |
| **Gamma** *(Already supported)* | $\mu$ (mean), $\nu$ (shape) | Strictly positive $(0, \infty)$ | $\ln(\mu) = X\beta$ | Insurance claim severity, cost modeling, service wait times. |
| **Inverse Gaussian** | $\mu$ (mean), $\lambda$ (shape) | Strictly positive $(0, \infty)$ | $1/\mu^2 = X\beta$ | Highly skewed positive continuous data with heavy tails (first passage times). |
| **Student's $t$** | $\mu$ (location), $\sigma$ (scale), $\nu$ (d.o.f.) | Real line $\mathbb{R}$ | Identity $\mu$, $\ln(\sigma)$, fixed $\nu$ | Robust linear regression impervious to heavy-tailed outliers in financial asset returns. |
| **Zero-Inflated Poisson (ZIP)** | $\pi$ (zero mass), $\lambda$ (Poisson rate) | Nonnegative integers | $\text{logit}(\pi) = X\gamma, \quad \ln(\lambda) = X\beta$ | Excess zeros with distinct causal origin (e.g., manufacturing defects). |
| **Cumulative Logit (Ordinal)** | $\theta_1 < \dots < \theta_{K-1}$, $\beta$ | Ordered classes $\{1 < 2 < \dots < K\}$ | $\text{logit}(P(Y \le k)) = \theta_k - X\beta$ | Survey ratings, medical triage tiers, credit rating tranches (AAA to D). |
| **Generalized Extreme Value (GEV)** | $\mu$ (location), $\sigma$ (scale), $\xi$ (shape) | Bounded or unbounded extremes | Free parameters | 100-year flood levels, portfolio Value-at-Risk (VaR), extreme wind speeds. |
| **Dirichlet** | $\alpha_1, \dots, \alpha_K$ | Simplex $\sum p_k = 1$ | $\ln(\alpha_k) = X\beta_k$ | Portfolio asset allocation, market share breakdown across competing brands. |

---

## 4. Exotic Splitting Criteria Across R & Scientific Literature

Beyond variance and deviance reduction, modern statistical learning has introduced several criteria that solve specific fundamental challenges:

### 4.1 Model-Based Recursive Partitioning (MOB / Zeileis, Hothorn, Hornik 2008)

**The Problem**: In model trees (like M5), refitting a full multivariate model for every candidate threshold on every feature is computationally intensive.

**The MOB Solution**:
1. Fit the parametric model (e.g. OLS, GLM, Survival) **once** at the parent node using maximum likelihood.
2. Compute the individual empirical estimating function scores:
   $$\psi_i(\hat{\theta}) = \left. \frac{\partial \ell_i(\theta)}{\partial \theta} \right|_{\theta = \hat{\theta}}$$
3. Test for **parameter instability** along each candidate partitioning variable $X_j$ using generalized M-fluctuation / cumulative sum (supLM) tests:
   $$W_j(t) = \hat{V}^{-1/2} n^{-1/2} \sum_{i=1}^{\lfloor n t \rfloor} \psi_{(i)}(\hat{\theta})$$
4. Only the variable $X_j$ with the most statistically significant instability ($p < \alpha$) is split, and the split point minimizes the total negative log-likelihood across the two sub-models.

### 4.2 Conditional Inference Framework (CTree / Hothorn, Hornik, Zeileis 2006)

**The Problem**: Standard CART, C4.5, and LightGBM suffer from severe selection bias toward variables with many unique values or categories (the "cardinality trap").

**The CTree Solution**:
Separates **variable selection** from **split-point selection** using permutation hypothesis testing:
1. Linear rank association statistic between transformed target $g(Y)$ and transformed covariate $h(X_j)$:
   $$T_j = \text{vec}\left( \sum_{i=1}^N w_i g(Y_i) h(X_{ij})^T \right)$$
2. Compute multivariate asymptotic normal $p$-value adjusted for multiple testing.
3. Select variable with minimum $p$-value; splitpoint is selected by maximizing two-sample test statistic. Selection is **strictly unbiased**.

### 4.3 Non-Parametric Two-Sample Metric Splits: MMD & Energy Distance

**The Problem**: Target $Y$ is complex (e.g., multi-output, high-dimensional vector, image embedding, or manifold coordinate) where scalar variance is meaningless.

**The Solution**:
- **Energy Distance**:
  $$\mathcal{E}(L, R) = 2 E\|Y_L - Y_R\| - E\|Y_L - Y_L'\| - E\|Y_R - Y_R'\|$$
- **Maximum Mean Discrepancy (MMD)**:
  $$\text{MMD}^2(L, R) = \frac{1}{N_L^2} \sum_{i, j \in L} k(Y_i, Y_j) + \frac{1}{N_R^2} \sum_{i, j \in R} k(Y_i, Y_j) - \frac{2}{N_L N_R} \sum_{i \in L, j \in R} k(Y_i, Y_j)$$
  Maximizing MMD or Energy Distance between child partitions splits complex distributions into distinct clusters without parametric assumptions.

### 4.4 Optimal Transport & Wasserstein Distance Splitting

**The Problem**: Each observation's target $Y_i$ is itself a probability distribution, histogram, or continuous density curve (e.g., daily price curves, spectra, demographic pyramids).

**The Solution**:
Partition node $P$ to maximize Wasserstein-2 dispersion reduction:
$$\text{Gain}_{\mathcal{W}_2} = \mathcal{W}_2^2(\text{Barycenter}(P)) - \left[ \frac{N_L}{N_P} \mathcal{W}_2^2(\text{Barycenter}(L)) + \frac{N_R}{N_P} \mathcal{W}_2^2(\text{Barycenter}(R)) \right]$$
where the barycenter is the Fréchet mean in Wasserstein space.

### 4.5 Fairness-Constrained Splitting

**The Problem**: Predictive trees can inadvertently discriminate against protected attributes $S$ (e.g., race, gender).

**The Solution**:
Penalize split gain by Demographic Parity or Equalized Odds divergence between groups $S$:
$$\text{Gain}_{\text{Fair}} = \text{Gain}(S) - \lambda \cdot D_{\text{fair}}(L, R \mid S)$$
where $D_{\text{fair}}$ penalizes disparate selection rates across groups.

---

## 5. Architectural Implementation Blueprint for `mltreelib`

To translate this research into production code in `mltreelib-core`, we propose a staged implementation matrix:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Implementation Staging Matrix                       │
├─────────────┬───────────────────────────────┬────────────────────────────┤
│ Stage       │ Feature                       │ Core Rust Module           │
├─────────────┼───────────────────────────────┼────────────────────────────┤
│ Phase 3.1   │ Doubly Robust & R-Learner     │ criteria.rs / split.rs     │
│ (Causal)    │ CATE Splits & Leaves          │ leaf.rs (LinearCATE)       │
├─────────────┼───────────────────────────────┼────────────────────────────┤
│ Phase 3.2   │ Accelerated Failure Time (AFT)│ family.rs / leaf.rs        │
│ (Survival)  │ Parametric Leaves (Weibull)   │ node_stats.rs              │
├─────────────┼───────────────────────────────┼────────────────────────────┤
│ Phase 3.3   │ Negative Binomial & Beta      │ family.rs / criteria.rs    │
│ (GAMLSS)    │ Regressions                   │ leaf.rs                    │
├─────────────┼───────────────────────────────┼────────────────────────────┤
│ Phase 3.4   │ MOB Parameter Instability     │ gate.rs / criteria.rs      │
│ (Exotic)    │ Score Fluctuation Tests       │ split.rs                   │
├─────────────┼───────────────────────────────┼────────────────────────────┤
│ Phase 3.5   │ Competing Risks (Fine-Gray)   │ criteria.rs (GrayTest)     │
│ (Survival)  │ Cumulative Incidence Leaves   │ leaf.rs (CIF)              │
└─────────────┴───────────────────────────────┴────────────────────────────┘
```

### 5.1 Concrete Data Structures for Rust Core

```rust
// In criteria.rs:
pub enum GainFn {
    // Existing variants ...
    Tweedie { p: f64 },
    Heteroscedastic,
    Tobit { threshold: f64 },
    Cv,
    Martingale,
    
    // Frontier Additions:
    /// Doubly robust causal split using AIPW pseudo-outcomes
    DoublyRobust { propensity_col: usize },
    /// R-learner Neyman-orthogonal causal split
    RLearner,
    /// Tarone-Ware weighted survival log-rank
    TaroneWare,
    /// Integrated Brier Score reduction
    BrierScore { time_points: Vec<f64> },
    /// Gray's test for competing risks
    GrayTest { cause: usize },
    /// Score-based parameter instability (MOB)
    MobScore,
}

// In family.rs / leaf.rs:
pub enum LeafFamily {
    Gaussian,
    Binomial,
    Poisson,
    Softmax,
    Survival,
    Tobit,
    Tweedie,
    Heteroscedastic,
    
    // Frontier Additions:
    NegativeBinomial { theta: Option<f64> },
    Beta { phi: Option<f64> },
    WeibullAft,
    CumulativeLogit { n_classes: usize },
    StudentT { nu: f64 },
}
```

---

## 6. Conclusion & Recommendation

1. **Immediate Value**:
   - The five newly implemented criteria (`tweedie`, `heteroscedastic`, `tobit`, `cv`, `martingale`) are fully validated on real datasets (`affairs`, `mroz`, `veteran`, `diamonds`).
2. **Next Frontier Step**:
   - Implementing **Doubly Robust (AIPW) and R-Learner CATE leaves** gives `mltreelib` state-of-the-art causal capabilities matching or beating `EconML` and `grf`.
   - Adding **Weibull AFT parametric leaves** and **Fine-Gray competing risks** positions `mltreelib` as the definitive survival tree toolkit in Rust/Python.
   - Adding **Negative Binomial and Beta regression leaves** completes the GAMLSS continuous/discrete spectrum.
