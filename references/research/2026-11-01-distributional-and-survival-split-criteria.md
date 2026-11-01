# Distributional, Survival, and Multiparameter Tree Splitting Criteria

**Date**: 2026-11-01  
**Author**: Jitin Kapila & Antigravity  
**Topic**: Mathematical formulation of split evaluation criteria across specialized leaf families (Survival, Multiclass Softmax, Tweedie, Tobit, Probit, and NGBoost Heteroscedastic).

---

## 1. Executive Summary & The "Split-Family Mismatch" Problem

In traditional decision trees, splitting criteria are often treated as independent of leaf models (e.g., using MSE for all continuous targets, Gini for all discrete targets). 

However, when moving to **distributional, censored, and survival leaves**, this independence breaks down:
- **Survival data**: A patient right-censored at time $T=5$ is *alive* at year 5. Treating $T$ as a standard continuous target using MSE or Gini severely biases splits toward early-censored cases and distorts prognostic groups.
- **Multiclass targets ($C \ge 3$)**: Class labels lack metric distance ($y \in \{0, 1, 2\}$ does not mean class 1 is halfway between 0 and 2). Scalar variance reduction is mathematically undefined.
- **Zero-inflated / Tweedie data ($1 < p < 2$)**: Variance scales with $\mu^p$. Standard MSE over-penalizes variance in high-value clusters and ignores the zero-spike, failing to separate non-zero probability from positive intensity.
- **Heteroscedastic / NGBoost leaves $(\mu, \sigma)$**: Standard SSE only seeks splits that separate means ($\Delta \mu$). It is completely blind to splits that separate a high-noise regime from a low-noise regime where means are identical.

Below is the formal mathematical specification for split search criteria compatible with each target family.

---

## 2. Survival Trees (Right-Censored Time-to-Event Data)

### Data Representation
Each row carries target pair $(T_i, E_i)$ with weight $w_i$:
- $T_i \ge 0$: Observed duration (event time or censoring time).
- $E_i \in \{0, 1\}$: Event indicator ($1 = \text{event observed}$, $0 = \text{right-censored}$).

### Criterion A: Two-Sample Weighted Log-Rank Test Statistic (Gold Standard)
Let $t_{(1)} < t_{(2)} < \dots < t_{(K)}$ be the distinct ordered event times ($E_i = 1$) in the node.
At time $t_{(j)}$:
- $d_{j}$: Total deaths in the node at $t_{(j)}$ ($d_{L,j}$ in Left child, $d_{R,j}$ in Right child).
- $n_{j}$: Total individuals at risk ($T_i \ge t_{(j)}$) in the node ($n_{L,j}$ in Left, $n_{R,j}$ in Right).

Under the null hypothesis $H_0: S_L(t) = S_R(t)$, the expected deaths in Left child is hypergeometric:
$$e_{L,j} = d_j \frac{n_{L,j}}{n_j}$$
The variance of $(d_{L,j} - e_{L,j})$ is:
$$v_{L,j} = \frac{n_{L,j} n_{R,j} d_j (n_j - d_j)}{n_j^2 (n_j - 1)}$$

**Split Gain (Log-Rank $\chi_1^2$ Statistic)**:
$$U = \sum_{j=1}^K (d_{L,j} - e_{L,j}), \quad V = \sum_{j=1}^K v_{L,j}$$
$$\text{Gain}_{\text{LogRank}} = \frac{U^2}{V}$$
- *Properties*: Non-parametric, scale-invariant, maximizes the statistical divergence between child survival curves.
- *Lineage*: `partykit::ctree`, `randomForestSRC`, Ciampi et al. (1986).

### Criterion B: Martingale Residual Variance Reduction (Fast $O(1)$ Formulation)
Full log-rank sweeps require event-time tracking across candidate partitions. LeBlanc & Crowley (1992) proved that survival split search can be transformed into exact scalar least-squares on **martingale residuals**.
1. At node entry, compute the cumulative hazard $\hat{\Lambda}_0(t)$ (Nelson-Aalen estimator) on the node's pooled data:
   $$\hat{\Lambda}_0(t) = \sum_{t_{(j)} \le t} \frac{d_j}{n_j}$$
2. For each subject $i$, compute their martingale residual:
   $$M_i = E_i - \hat{\Lambda}_0(T_i) \in (-\infty, 1]$$
   *(Negative values: survived longer than expected; near 1: died earlier than expected).*
3. Evaluate candidate splits by standard weighted SSE reduction on $M_i$:
   $$\text{Gain} = \frac{(\sum_{L} w_i M_i)^2}{\sum_{L} w_i} + \frac{(\sum_{R} w_i M_i)^2}{\sum_{R} w_i} - \frac{(\sum_{\text{node}} w_i M_i)^2}{\sum_{\text{node}} w_i}$$
- *Complexity*: **$O(N)$ linear-time streaming scan**, reusing existing `SplitScratch` accumulators!

---

## 3. Multiclass / Softmax Trees ($C \ge 3$ Discrete Classes)

### Data Representation
Target $y_i \in \{0, 1, \dots, C-1\}$.

### Criterion A: Multiclass Gini Impurity Gain
For a node $S$ with weighted class counts $w_{S,c} = \sum_{i \in S, y_i = c} w_i$ and total weight $W_S = \sum_c w_{S,c}$:
$$I_{\text{Gini}}(S) = 1 - \sum_{c=0}^{C-1} \left( \frac{w_{S,c}}{W_S} \right)^2$$
$$\text{Gain}_{\text{Gini}} = I_{\text{Gini}}(P) - \frac{W_L}{W_P} I_{\text{Gini}}(L) - \frac{W_R}{W_P} I_{\text{Gini}}(R)$$
- *Efficiency*: In a candidate sweep, moving row $i$ of class $c$ from Right to Left updates $w_{L,c}$ and $w_{R,c}$ in $O(1)$ scalar additions without looping over all classes.

### Criterion B: Multinomial Deviance / Cross-Entropy
$$\text{Dev}(S) = -2 \sum_{c=0}^{C-1} w_{S,c} \ln \left( \frac{w_{S,c}}{W_S} \right)$$
$$\text{Gain}_{\text{Dev}} = \text{Dev}(P) - [\text{Dev}(L) + \text{Dev}(R)]$$
- Aligns directly with logistic/softmax leaf negative log-likelihood.

---

## 4. Tweedie Trees ($1 < p < 2$, Zero-Heavy Insurance / Volume Data)

### Data Representation
Continuous nonnegative target $y_i \ge 0$, often with a large point-mass at 0 (e.g. 90% zeros) and right-skewed positive values.

### Criterion: Tweedie Unit Deviance
Variance power function: $V(\mu) = \mu^p$. The quasi-deviance $d(y, \mu)$ is:
$$d(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$
For candidate side with mean $\hat{\mu} = \frac{\sum w_i y_i}{\sum w_i}$:
$$\text{Dev}(S) = \sum_{i \in S} w_i d(y_i, \hat{\mu}_S)$$
$$\text{Gain}_{\text{Tweedie}} = \text{Dev}(P) - [\text{Dev}(L) + \text{Dev}(R)]$$
- *Property*: When $p \to 1$, smoothly converges to Poisson deviance; when $p \to 2$, converges to Gamma deviance.

---

## 5. Tobit Trees (Censored Continuous Gaussian Regression)

### Data Representation
Target $y_i^*$ is latent; observed target is left-censored at threshold $a$ (commonly $a=0$):
$$y_i = \max(a, y_i^*)$$

### Criterion: Tobit Censored Log-Likelihood Deviance
For a subgroup with mean $\mu$ and scale $\sigma$:
$$\ln L(\mu, \sigma) = \sum_{y_i > a} w_i \left[ -\ln \sigma + \ln \phi\left( \frac{y_i - \mu}{\sigma} \right) \right] + \sum_{y_i = a} w_i \ln \Phi\left( \frac{a - \mu}{\sigma} \right)$$
$$\text{Gain}_{\text{Tobit}} = 2 \left[ \ln L(\hat{\mu}_L, \hat{\sigma}_L) + \ln L(\hat{\mu}_R, \hat{\sigma}_R) - \ln L(\hat{\mu}_P, \hat{\sigma}_P) \right]$$
- *Approximation for fast split search*: Decompose into hurdle score:
  $$\text{Gain}_{\text{fast}} = \text{Gain}_{\text{binomial}}(I(y > a)) + \text{Gain}_{\text{Gaussian}}(y \mid y > a)$$

---

## 6. NGBoost / Heteroscedastic Gaussian $(\mu, \sigma)$

### Data Representation
Continuous target $y \in \mathbb{R}$ where conditional variance is non-constant: $\sigma^2 = \sigma^2(x)$.

### Criterion: Gaussian Negative Log-Likelihood Deviance
Unlike homoscedastic regression (which minimizes $\sum (y - \mu)^2$), heteroscedastic splitting evaluates:
$$\text{Dev}(S) = \sum_{i \in S} w_i \left[ \ln \hat{\sigma}_S^2 + \frac{(y_i - \hat{\mu}_S)^2}{\hat{\sigma}_S^2} \right] = W_S \left( \ln \hat{\sigma}_S^2 + 1 \right)$$
where $\hat{\sigma}_S^2 = \frac{1}{W_S} \sum_{i \in S} w_i (y_i - \hat{\mu}_S)^2$.

$$\text{Gain}_{\text{NGBoost}} = W_P \ln \hat{\sigma}_P^2 - \left[ W_L \ln \hat{\sigma}_L^2 + W_R \ln \hat{\sigma}_R^2 \right]$$
- *Why this matters*: A split where $\hat{\mu}_L = \hat{\mu}_R$ (zero gain under standard MSE) yields **huge gain** if $\hat{\sigma}_L^2 \ll \hat{\sigma}_P^2 \ll \hat{\sigma}_R^2$. It isolates clean predictive regimes from noisy ones.

---

## 7. Compatibility Matrix: Families vs. Split Criteria

| Leaf Family | Target Type | Primary Recommended Split Gain | Fallback / Alternative | Invalid / Harmful Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian (Mean/Linear)** | Continuous | `sse` / `friedman_mse` | `huber`, `absolute_error` | Gini, Entropy, Log-Rank |
| **Binomial (Logit/Probit)** | Binary $0/1$ | `deviance` (cross-entropy) | `gini`, `entropy`, `twoing` | SSE (suboptimal tails), Log-Rank |
| **Poisson** | Nonnegative counts | `poisson` deviance | `deviance` | SSE, Gini |
| **Softmax / Multinomial** | Classes $0 \dots C-1$ | `gini` (multiclass) | `entropy`, `twoing` | SSE (labels are non-metric) |
| **Tweedie ($1 < p < 2$)** | Zero-heavy positive | `tweedie` deviance ($p$) | `poisson` deviance | SSE (distorted by variance explosion) |
| **Tobit** | Censored continuous | `tobit` log-likelihood | Hurdle (`deviance` + `sse`) | Plain SSE (downward bias from censored floor) |
| **Heteroscedastic $(\mu, \sigma)$**| Continuous | Dual-parameter Gaussian NLL | Continuous CRPS | Plain SSE (blind to variance breaks) |
| **Survival $(T, E)$** | Time + Indicator | **Log-Rank** / Martingale SSE | Cox partial deviance | **SSE, Gini, Entropy (catastrophic bias)** |

---

## 8. Summary of Actionable Implementation Rules

1. **Criterion Enforcement**: When a user selects `family="survival"`, `ModelTree` must reject `gain="gini"` or `gain="sse"` with a helpful error suggesting `gain="log_rank"` or `gain="martingale"`.
2. **Streaming Martingale Formulation**: For survival trees, using the Nelson-Aalen martingale residual $M_i = E_i - \hat{\Lambda}_0(T_i)$ allows survival split search to run on the existing zero-allocation `SplitScratch` streaming loops without tracking full risk tables.
3. **Multiclass Counts**: Multiclass Gini requires a $C$-dimensional vector in `SplitScratch` (`level_counts: Vec<f64>`) rather than scalar sums.
