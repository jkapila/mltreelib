# Comprehensive Survey of Decision Tree Splitting Criteria

**Date:** 2026-11-19  
**Author:** Jitin Kapila & Antigravity  
**Status:** Canonical Reference & Research Survey  
**Scope:** Comparative mathematical analysis and taxonomy of splitting criteria across classical CART, Ranger, Generalized Random Forests (GRF), CatBoost, LightGBM, XGBoost, Explainable Boosting Machines (EBM / InterpretML), Partykit (CTree/MOB), Survival/Competing Risks, Causal/Uplift, and Metric/Distributional frameworks.

---

## 1. Executive Summary

The **split search operator** is the mathematical core of recursive partitioning:
$$\arg\max_{j \in \{1,\dots,P\}, \, s \in \mathcal{S}_j} \Delta(j, s; \mathcal{D}_t)$$
where $\Delta$ measures the improvement in sample purity, objective reduction, or parameter divergence when splitting node $t$ into children $L(j, s)$ and $R(j, s)$.

While classical CART (Breiman et al. 1984) and C4.5 (Quinlan 1993) used scalar variance and entropy reduction, modern tree algorithms branch into four distinct frontiers:
1. **Statistical Hypothesis Testing (Unbiased Selection)**: CTree (permutation tests) and MOB (parameter instability tests) separate variable selection from cutpoint selection, eliminating the $O(N \log N)$ cardinality bias trap.
2. **Gradient & Hessian Second-Order Optimization**: GBDT (XGBoost/LightGBM) and Symmetric Oblivious Trees (CatBoost) evaluate Taylor expansions of arbitrary loss functions over discrete histograms.
3. **Econometric & Causal Local Estimation**: GRF (Generalized Random Forests) and Causal Trees split on estimating equations and treatment effect heterogeneity.
4. **Metric & Distributional Partitions**: Energy Distance, Maximum Mean Discrepancy (MMD), and Hellinger distance enable multi-target embedding splitting ($Y \in \mathbb{R}^D$) and extreme imbalance resistance without covariance inversion.

---

## 2. Complete Taxonomy of Splitting Criteria

```
                                  Splitting Criteria Taxonomy
                                               │
   ┌───────────────────┬───────────────────────┼────────────────────────┬─────────────────────┐
   ▼                   ▼                       ▼                        ▼                     ▼
Impurity & Scalar    Statistical & Unbiased  Gradient & Boosted      Econometric & Causal   Metric & Manifold
- SSE / Variance     - CTree (Permutation)   - Newton Step (GBDT)    - GRF Pseudo-Gradients - Energy Distance
- Friedman MSE       - MOB (Score supLM)     - Cosine (CatBoost)     - Causal Honest (A&I)  - MMD (Kernel RKHS)
- Absolute Error     - MaxStat (Rank)        - GOSS (LightGBM)       - R-Learner (Robinson) - Wasserstein-2
- Huber Loss         - Unbiased Gini         - Oblivious Symmetric   - Instrumental (IV)    - Hellinger (Skew)
- Poisson Deviance   - Chow Test             - EBM FAST (Quad-Stump) - Qini Impurity Gain   - Multi-Output SSPV
- Gini / Entropy                                                     - Log-Rank / Gray      - Oblique (SPORF)
- Gain Ratio / Twoing                                                - Martingale Residual  - Fairness Penalized
```

---

## 3. Mathematical Formulations by Framework

### 3.1 Classical Impurity & Quinlan Lineage

| Criterion | Formula | Optimization Target | Complexity |
|---|---|---|---|
| **SSE (Variance Reduction)** | $\Delta = \text{SSE}_P - (\text{SSE}_L + \text{SSE}_R)$ | Gaussian conditional mean | $O(N)$ running sums |
| **Friedman MSE** | $\Delta = \frac{w_L w_R}{w_L + w_R} (\mu_L - \mu_R)^2$ | Weighted difference in child means | $O(N)$ running sums |
| **Absolute Error (L1)** | $\Delta = \sum_{i \in P} \|y_i - m_P\| - \sum_{k \in \{L, R\}} \sum_{i \in k} \|y_i - m_k\|$ | Conditional median (Laplace noise) | $O(N \log N)$ running medians |
| **Huber Loss** | $\Delta = \text{Loss}_P(\delta) - (\text{Loss}_L(\delta) + \text{Loss}_R(\delta))$ | Outlier-resistant regression | $O(N)$ iterative |
| **Poisson Deviance** | $\Delta = 2 \sum y_i \ln\frac{\mu_k}{\mu_P} - 2 \sum (y_i - \mu_k)$ | Count/rate modeling ($y \ge 0$) | $O(N)$ running sums |
| **C4.5 Gain Ratio** | $\text{GainRatio} = \frac{\text{InfoGain}(j, s)}{\text{SplitInfo}(j, s)}$ where $\text{SplitInfo} = -\sum \frac{N_k}{N} \log_2 \frac{N_k}{N}$ | Normalizes multi-branch oversplitting | $O(K)$ |
| **Breiman Twoing** | $2 P_L P_R \left(\sum_k \|p_L(k) - p_R(k)\|\right)^2$ | Contrasting class partitions in multiclass | $O(C)$ |
| **M5 SDR** | $\text{SDR} = \text{sd}_P - \sum_{k \in \{L, R\}} \frac{N_k}{N} \text{sd}_k$ | Standard deviation reduction | $O(N)$ running sums |

---

### 3.2 Ranger & Statistical Forest Mechanisms

**Reference:** Wright & Ziegler (2017) *ranger: A Fast Implementation of Random Forests*, JSS.  
**Reference:** Malley et al. (2012) *Probability machines: consistent probability estimation using non-parametric methods*, Stat. Med.

1. **ExtraTrees (Extremely Randomized Splits - Geurts et al. 2006)**:
   - For each candidate variable in `mtry`, draw a single uniform cutpoint $s \sim \text{Uniform}(\min(X_j), \max(X_j))$.
   - Evaluate impurity gain only at this single point. Reduces candidate evaluation from $O(N)$ to $O(1)$ per feature; acts as strong geometric regularization.
2. **Hellinger Distance Splitting (Cieslak & Chawla 2008)**:
   - Evaluates the affinity divergence for binary classification under severe imbalance (e.g. 1:1,000 fraud prevalence):
     $$H(L, R) = \sqrt{\sum_{k=1}^2 \left( \sqrt{\frac{\|L_k\|}{\|L\|}} - \sqrt{\frac{\|R_k\|}{\|R\|}} \right)^2}$$
   - Unlike Gini or Entropy, Hellinger distance is skew-invariant: it does not drop precipitously when one class is extremely rare.
3. **Maximally Selected Rank Statistics (`maxstat` - Hothorn & Lausen 2003)**:
   - Computes standardized two-sample linear rank statistics $T(s) = \frac{S(s) - \mathbb{E}[S(s)]}{\sqrt{\text{Var}[S(s)]}}$ across all cutpoints $s$.
   - Selects the cutpoint that maximizes $T(s)$ subject to asymptotic $p$-value correction bounding the family-wise error rate over the continuum of cutpoints.
4. **Unbiased Gini Partitioning (Nembrini, Koenig, Wright 2018)**:
   - Standard Gini selection favors variables with many categories purely due to multi-testing noise.
   - Calculates out-of-bag expectation corrections or pseudo-values to subtract expected sample variance from the candidate gain without expensive full permutation tests.

---

### 3.3 Generalized Random Forests (GRF) & Econometric Trees

**Reference:** Athey, Tibshirani, & Wager (2019) *Generalized Random Forests*, Annals of Statistics.  
**Reference:** Friedberg, Tibshirani, Athey, & Wager (2020) *Local Linear Forests*, JCGS.

GRF unifies forest splitting for any local estimating equation $\mathbb{E}[\psi_{\theta(x)}(Y_i, X_i) \mid X=x] = 0$:

1. **Gradient-Based Splitting (The $\rho$-Score)**:
   - At parent node $P$, fit the global parameter $\hat{\theta}_P$.
   - Compute influence pseudo-outcomes via Taylor expansion:
     $$\rho_i = -\xi^T A_P^{-1} \psi(Y_i, \hat{\theta}_P) \in \mathbb{R}$$
     where $A_P = \nabla_\theta \mathbb{E}[\psi(Y_i, \hat{\theta}_P)]$ is the Jacobian of the estimating equation.
   - Run **standard variance reduction directly on $\rho_i$**:
     $$\Delta(j, s) = \frac{N_L N_R}{N_L + N_R} (\bar{\rho}_L - \bar{\rho}_R)^2$$
   - Maximizing variance reduction on $\rho$ directly maximizes heterogeneity in the target parameter $\theta(x)$!

2. **Instrumental Variable (IV) Splitting**:
   - Outcome $Y$, endogenous treatment $W$, valid instrument $Z$. Estimating equation: $\psi(Y_i, W_i, Z_i, \tau) = Z_i (Y_i - \tau W_i)$.
   - $\rho_i$ measures local 2SLS heterogeneity: splits isolate regimes of strong vs weak instrument compliance.

3. **Local Linear Forest (`ll_regression_forest`) Splitting**:
   - Minimizes boundary attenuation by evaluating splits that preserve linear slope contrasts rather than flat step functions.

---

### 3.4 Model-Based & Conditional Inference Trees (Partykit / CTree / MOB)

**Reference:** Zeileis, Hothorn, & Hornik (2008) *Model-Based Recursive Partitioning*, JCGS.  
**Reference:** Hothorn, Hornik, & Zeileis (2006) *Unbiased Recursive Partitioning*, JCGS.

1. **Model-Based Recursive Partitioning (MOB)**:
   - **Step 1 (Parent Fit)**: Fit the model $M_P$ (e.g. OLS, Logit, Poisson) ONCE at parent node $P$, obtaining $\hat{\theta}_P$.
   - **Step 2 (Score Residuals)**: For each observation $i$, compute score residuals:
     $$\psi_i = \nabla_\theta \ell_i(\hat{\theta}_P) \in \mathbb{R}^K$$
     (e.g. for OLS, $\psi_i = (y_i - x_i^T \hat{\beta}) x_i$).
   - **Step 3 (Instability Fluctuation Test)**: For each candidate variable $Z_j$, sort observations by $Z_j$. Compute empirical cumulative fluctuation process:
     $$W_j(t) = \hat{V}^{-1/2} n^{-1/2} \sum_{i=1}^{\lfloor nt \rfloor} \psi_{(i)}$$
     where $\hat{V} = \frac{1}{n} \sum_{i=1}^n \psi_i \psi_i^T$.
   - **Step 4 (supLM Test)**:
     $$\text{supLM}_j = \max_{t \in [\underline{t}, \bar{t}]} \frac{\|W_j(t)\|^2}{t(1-t)}$$
   - **Step 5 (Variable Selection)**: Select $j^* = \arg\min_j p_j$. If $\min p_j < \alpha$, split on $j^*$ at the cutpoint maximizing child deviance reduction. Bypasses candidate-level refits entirely ($O(N)$ vs $O(N \cdot P \cdot K^3)$).

2. **Conditional Inference Trees (CTree)**:
   - Two-stage unbiased splitting:
     - **Stage 1 (Variable Selection)**: Linear rank statistic $T_j = \sum_{i=1}^n w_i g_j(X_{ji}) h(Y_i)$. Standardize $c_j = (T_j - \mu_j) / \sigma_j$ or evaluate quadratic form $Q_j \sim \chi^2$. Test global null of independence with Bonferroni correction. Pick $j^*$ with minimum $p$-value.
     - **Stage 2 (Splitpoint Selection)**: Search cutpoint on $j^*$ maximizing two-sample rank disparity.
   - **Guaranteed Cardinality Unbiasedness**: A 1,000-level factor and a binary factor have identical false-positive selection rates under the null.

---

### 3.5 Gradient Boosted Trees (LightGBM / CatBoost / XGBoost)

1. **Second-Order Taylor Gain (Newton Step Gain)**:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_{i \in P} g_i)^2}{\sum_{i \in P} h_i + \lambda} \right] - \gamma$$
   where $g_i = \partial \ell / \partial \hat{y}_i$, $h_i = \partial^2 \ell / \partial \hat{y}_i^2$, $\lambda$ is $L_2$ leaf penalty, $\gamma$ is tree complexity penalty.

2. **Gradient-Based One-Side Sampling (GOSS - LightGBM)**:
   - Retain instances with top $a \times 100\%$ gradient magnitudes.
   - Subsample remaining instances with rate $b \in (0, 1)$.
   - Multiply small-gradient instances by weight $\frac{1-a}{b}$ to preserve gradient expectations.

3. **Cosine Similarity Splitting & Oblivious Trees (CatBoost)**:
   - Evaluates split gain by maximizing the inner product between target gradients $g$ and binary split step projections.
   - Constrains all nodes at depth $d$ to share the exact same $(j, s)$ feature and threshold, building symmetric oblivious tables evaluateable via parallel bitwise indexing.

4. **Optimal Categorical Splitting (Fisher's Theorem / Ordered Target Stats)**:
   - For convex loss functions, sorting categorical levels by $G_c / H_c$ guarantees that the optimal binary partition subset is a contiguous prefix/suffix slice.
   - Reduces combinatorial $O(2^K)$ subset search to $O(K \log K)$ sorting!

---

### 3.6 Explainable Boosting Machines (EBM / InterpretML)

**Reference:** Lou, Caruana, & Gehrke (2013) *Accurate intelligible models with pairwise interactions*, KDD.

1. **Cyclic Round-Robin 1D Trees**:
   - Single-feature shallow regression trees fit iteratively on residuals: $y - \sum_{k \ne j} f_k(x_k)$.
   - Splitting uses gradient variance reduction with strict `min_samples_leaf` and `min_hessian` smoothing.
2. **FAST Interaction Detector**:
   - Evaluates all $P(P-1)/2$ feature pairs $(X_j, X_k)$ on the residual of the main-effects model.
   - Fits a restricted 2D interaction quad-stump (4 leaves partitioned across $x_j$ and $x_k$).
   - Computes residual variance reduction in $O(N)$ per pair; ranks pairs and selects top-$K$ for bivariate interaction boosting.

---

### 3.7 Survival & Competing Risks Trees

1. **Leblanc & Crowley (1992) Log-Rank Split**:
   $$\text{Gain} = \frac{(O_L - E_L)^2}{V_L} + \frac{(O_R - E_R)^2}{V_R}$$
   where $E_L = \sum_{t_j} d_j \frac{n_{L, j}}{n_j}$, $V_L = \sum_{t_j} d_j \frac{n_{L,j} n_{R,j} (n_j - d_j)}{n_j^2 (n_j - 1)}$.
2. **Tarone-Ware & Peto-Peto Weighted Log-Rank**:
   - Introduces weights $w(t_j) = [\hat{S}(t_j)]^\rho [n_j]^\gamma$ into $(O_L - E_L)$:
     - $\rho = 1, \gamma = 0$ (Peto-Peto Wilcoxon): Emphasizes early survival differences.
     - $\rho = 0, \gamma = 1/2$ (Tarone-Ware): Balances early and late follow-up events.
3. **Gray's Test for Competing Risks (Fine-Gray Formulation)**:
   - In clinical trials where patients can experience event 1 (cancer death) or competing event 2 (stroke death):
   - Computes cumulative incidence function (CIF) differences across partitions, weighting at-risk sets by IPCW probability of remaining event-free.
4. **Martingale Residual Splitting**:
   - Pre-fits baseline hazard $\hat{\Lambda}_0(t)$.
   - Computes continuous martingale residuals $M_i = E_i - \hat{\Lambda}_0(T_i)$.
   - Evaluates fast scalar variance reduction directly on $M_i$.

---

### 3.8 Causal Inference & Uplift Partitioning

1. **Athey & Imbens (2016) Honest Causal Tree**:
   $$\Delta_{\text{causal}} = \frac{N_L N_R}{N} (\hat{\tau}_L - \hat{\tau}_R)^2 - \frac{N}{N_{\text{est}}} \left( \frac{S_{L, 1}^2}{p_L} + \frac{S_{L, 0}^2}{1 - p_L} + \frac{S_{R, 1}^2}{p_R} + \frac{S_{R, 0}^2}{1 - p_R} \right)$$
   - The first term rewards treatment effect divergence (heterogeneity).
   - The second term penalizes high within-leaf outcome variance, preventing noisy, fragile leaves.
2. **R-Learner Splitting (Nie & Wager 2021)**:
   - Uses Robinson's transformation on pre-computed propensity scores $e(X)$ and outcome baselines $m(X)$:
     $$\tilde{Y}_i = Y_i - \hat{m}(X_i), \quad \tilde{W}_i = W_i - \hat{e}(X_i)$$
   - Evaluates split gain minimizing weighted squared loss: $\sum (\tilde{Y}_i - \tau(X_i) \tilde{W}_i)^2$.
3. **Qini Area Gain (Radcliffe & Surry 2011)**:
   - Computes the increase in Qini curve area when routing samples into left and right treatment policies.

---

### 3.9 Metric, Manifold & Multi-Target Splitting

1. **Energy Distance (Székely & Rizzo 2013)**:
   - For multi-output targets $Y \in \mathbb{R}^D$ (e.g. image embeddings, multi-sensor arrays):
     $$\mathcal{E}(L, R) = 2 \mathbb{E}\|Y_L - Y_R\| - \mathbb{E}\|Y_L - Y_L'\| - \mathbb{E}\|Y_R - Y_R'\|$$
   - Maximizing $\mathcal{E}(L, R)$ splits the node into two maximally divergent multi-dimensional point clouds without assuming Gaussianity or inverting ill-conditioned covariance matrices.
2. **Maximum Mean Discrepancy (MMD - Gretton et al. 2012)**:
   - Kernel two-sample test statistic in Reproducing Kernel Hilbert Space (RKHS):
     $$\text{MMD}^2(L, R) = \frac{1}{|L|^2} \sum_{i,j \in L} k(y_i, y_j) + \frac{1}{|R|^2} \sum_{i,j \in R} k(y_i, y_j) - \frac{2}{|L||R|} \sum_{i \in L, j \in R} k(y_i, y_j)$$
   - With RBF kernel $k(y, y') = \exp(-\|y - y'\|^2 / 2\sigma^2)$, detects non-linear multi-modal distributional splits.
3. **Wasserstein-2 Optimal Transport Splitting**:
   - For targets that are probability histograms or empirical distributions $Y_i \sim P_i$:
   - Minimizes Fréchet dispersion around leaf Wasserstein barycenters.

---

### 3.10 Oblique & Linear Combination Splits

1. **SPORF (Sparse Projection Oblique Randomer Forest - Tomita et al. 2020)**:
   - Samples $d \in \{2, 3\}$ features randomly. Draws coefficients $\alpha_k \in \{-1, +1\}$ with probability $1/d$.
   - Projects samples onto 1D line $Z = \sum \alpha_k X_{j_k}$ and runs fast scalar split search on $Z$. Eliminates diagonal axis-aligned staircasing.
2. **Householder CART (Wickramarachchi et al. 2016)**:
   - Computes leading eigenvector $v$ of the node covariance matrix $\Sigma_t$.
   - Constructs Householder reflection matrix $H = I - 2 u u^T$ where $u = \frac{v - e_1}{\|v - e_1\|}$.
   - Reflects feature coordinates so the dominant variance axis aligns with coordinate axis $x_1$, then performs standard CART split.

---

### 3.11 Fairness-Constrained & Robust Splitting

1. **Demographic Parity Penalized Gain**:
   $$\text{Gain}_{\text{fair}} = \text{Gain}_{\text{base}} - \lambda \cdot |P(\text{Left} \mid S = 1) - P(\text{Left} \mid S = 0)|$$
   where $S \in \{0, 1\}$ is a sensitive protected attribute (e.g. gender, race, age). Directly produces fair decision trees with Pareto-optimal accuracy-fairness tradeoffs.
2. **Equalized Odds Penalized Gain**:
   - Penalizes difference in True Positive Rate (TPR) and False Positive Rate (FPR) between protected groups across the candidate split.
3. **Adversarially Robust Tree Splitting (Chen et al. 2019)**:
   - Evaluates worst-case impurity under bounded input perturbations $\|x - x'\|_\infty \le \epsilon$. Protects leaf classifications from adversarial noise near decision boundaries.

---

## 4. Architectural Synthesis for `mltreelib`

To give `mltreelib` an industry-leading splitting suite, we group these criteria into a modular hierarchy:

```
mltreelib Split Suite
├── 1. Standard Scalar & Distributional (Implemented)
│   ├── sse, friedman_mse, absolute_error, huber, poisson
│   ├── entropy, gini, gain_ratio, twoing, sdr
│   ├── tweedie, heteroscedastic, tobit, cv
│   └── logrank, martingale, causal (Athey & Imbens)
│
├── 2. Statistical & Unbiased Frameworks (Spec 12 - Ready)
│   ├── mob: Model-Based Recursive Partitioning (Parent Score Residual Fluctuation)
│   ├── ctree: Conditional Inference Framework (Linear Rank Permutation Statistics)
│   ├── maxstat: Maximally Selected Rank Statistics
│   └── hellinger: Imbalance-Skew Invariant Divergence
│
├── 3. Metric & Multi-Target Manifolds (Spec 12 - Ready)
│   ├── energy_distance: Multi-output target splitting (Székely & Rizzo)
│   ├── mmd: Kernel Maximum Mean Discrepancy (RKHS embedding split)
│   └── wasserstein: Optimal transport histogram splitting
│
├── 4. Fairness & Constraints (Spec 12 - Ready)
│   └── fair: Demographic parity & Equalized Odds penalized splitting
│
└── 5. Advanced Forest & Interaction Engines (Future Specs 13+)
    ├── grf: Gradient-based GMM estimating equations & IV splitting
    ├── fast_interaction: EBM-style quad-stump pairwise residual screening
    ├── sporf: Sparse projection oblique random hyperplanes
    └── oblivious: Symmetric table-driven depth-level splitting (CatBoost)
```

---

## 5. Primary References

1. **Athey, S., & Imbens, G.** (2016). *Recursive partitioning for heterogeneous causal effects*. PNAS, 113(27), 7353–7360.
2. **Athey, S., Tibshirani, J., & Wager, S.** (2019). *Generalized Random Forests*. Annals of Statistics, 47(2), 1148–1178.
3. **Breiman, L., Friedman, J., Olshen, R., & Stone, C.** (1984). *Classification and Regression Trees*. Wadsworth.
4. **Chen, H., Zhang, H., Si, S., Li, Y., Boning, D., & Hsieh, C. J.** (2019). *Robustness verification of tree-based models*. NeurIPS.
5. **Cieslak, D. A., & Chawla, N. V.** (2008). *Learning decision trees for unbalanced data*. ECML PKDD, 241–256.
6. **Friedberg, R., Tibshirani, J., Athey, S., & Wager, S.** (2020). *Local Linear Forests*. JCGS, 30(2), 503–517.
7. **Geurts, P., Ernst, D., & Wehenkel, L.** (2006). *Extremely randomized trees*. Machine Learning, 63(1), 3–42.
8. **Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A.** (2012). *A kernel two-sample test*. JMLR, 13, 723–773.
9. **Hothorn, T., Hornik, K., & Zeileis, A.** (2006). *Unbiased recursive partitioning: A conditional inference framework*. JCGS, 15(3), 651–674.
10. **Hothorn, T., & Lausen, B.** (2003). *On the exact distribution of maximally selected rank statistics*. Computational Statistics & Data Analysis, 43(2), 121–137.
11. **Leblanc, M., & Crowley, J.** (1992). *Relative risk trees for censored survival data*. Biometrics, 48(2), 411–425.
12. **Lou, Y., Caruana, R., & Gehrke, J.** (2013). *Accurate intelligible models with pairwise interactions*. KDD, 623–631.
13. **Malley, J. D., Kruppa, J., Dasgupta, A., Malley, K. G., & Ziegler, A.** (2012). *Probability machines: consistent probability estimation using non-parametric methods*. Statistics in Medicine, 31(8), 776–797.
14. **Nembrini, S., Koenig, I. R., & Wright, M. N.** (2018). *The revival of the Gini importance?* Bioinformatics, 34(21), 3711–3718.
15. **Nie, X., & Wager, S.** (2021). *Quasi-oracle estimation of heterogeneous treatment effects*. Biometrika, 108(2), 299–319.
16. **Radcliffe, N. J., & Surry, P. D.** (2011). *Real-world uplift modelling with significance-based uplift trees*. White Paper TR-2011-1, Stochastic Solutions.
17. **Székely, G. J., & Rizzo, M. L.** (2013). *Energy statistics: A class of statistics based on energy distances*. Journal of Statistical Planning and Inference, 143(8), 1249–1272.
18. **Tomita, T. M., Browne, J., Shen, C., Chung, J., et al.** (2020). *Sparse Projection Oblique Randomer Forests*. JMLR, 21(104), 1–39.
19. **Wickramarachchi, D. C., Robertson, B. L., Reale, M., Price, C. J., & Brown, J.** (2016). *HHCART: An oblique decision tree*. Computational Statistics & Data Analysis, 96, 12–23.
20. **Wright, M. N., & Ziegler, A.** (2017). *ranger: A fast implementation of random forests for high dimensional data in C++ and R*. JSS, 77(1), 1–17.
21. **Zeileis, A., Hothorn, T., & Hornik, K.** (2008). *Model-based recursive partitioning*. JCGS, 17(2), 492–514.
