# Architectural Deep Dive: Ranger and Generalized Random Forests (GRF)

*Date: 2026-11-17*  
*Author: mltreelib Core Research*  
*Status: Architecture Reference & Implementation Blueprint*

---

## 1. Executive Summary & Benchmark Context

In the `mltreelib` Elo Arena across TabArena classification and regression suites:
- **`ranger`** routinely posts the highest single-model Elo amongst unboosted forests and finishes among the top 5 models overall (Elo ~1405 in classification, sub-second execution on medium tabular sets).
- **`grf_rf` / `grf_llf`** (Generalized Random Forests / Local Linear Forests) demonstrates near-optimal performance on continuous physical regimes and causal / distributional tasks, consistently eliminating boundary bias where standard step-function trees collapse.

This document breaks down the exact internal algorithms, data structures, and mathematical mechanisms powering both packages, and derives concrete blueprints for extending `mltreelib-core` in Rust.

---

## 2. Ranger (`ranger`): Engineering & Statistical Mechanics

**Reference**: Wright, M. N. & Ziegler, A. (2017). *ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R*. Journal of Statistical Software, 77(1), 1–17.

### 2.1 The Probability Forest Transformation (Malley et al., 2012)
Standard Breiman Random Forests store a single majority-vote class in each leaf:
$$\hat{C}_b(x) = \arg\max_k n_{k, b}(x)$$
The forest then averages the 0/1 indicator votes across $B$ trees: $P(Y=k \mid x) = \frac{1}{B} \sum_{b=1}^B \mathbf{1}(\hat{C}_b(x) = k)$. This vote aggregation introduces severe step quantization (a forest of 100 trees can only produce probabilities in multiples of $0.01$).

In contrast, Ranger's `probability = TRUE` implements **Probability Estimation Trees (PETs)**:
1. Each leaf $L_{b}$ stores the **exact empirical class distribution vector**:
   $$\mathbf{p}_b(x) = \left[ \frac{n_{1, b}(x)}{N_b(x)}, \frac{n_{2, b}(x)}{N_b(x)}, \dots, \frac{n_{K, b}(x)}{N_b(x)} \right]$$
2. The forest prediction averages the continuous class probability vectors:
   $$\bar{\mathbf{p}}(x) = \frac{1}{B} \sum_{b=1}^B \mathbf{p}_b(x)$$
3. **Splitting criterion**: During tree growth, splits maximize Gini gain calculated on class counts directly, but pruning is completely disabled. Leaves grow to purity or minimal size (`min.node.size = 10`), yielding calibrated, smooth probabilities that dominate log-loss and Brier scores.

### 2.2 Memory Layout & Zero-Copy Transpose in C++
A key reason Ranger runs $5\times$ to $20\times$ faster than standard R `randomForest`:
1. **Contiguous Index Buffers**: The original dataset $X$ ($N \times P$) is stored once in a contiguous flat array.
2. **Sample Indirection via Pointers**: Trees do not copy rows during bootstrap sampling or recursive partitioning. Instead, each node receives a slice of integer indices `std::vector<size_t> sampleIDs`.
3. **In-Place Partitioning**: Splitting a node reorders `sampleIDs` in place around the split threshold via `std::partition` (quicksort partition step), requiring zero heap allocations during the recursive descent.

### 2.3 Categorical Splitting: Order vs Partition
For a categorical variable with $K$ unique levels, searching all $2^{K-1} - 1$ possible binary partition subsets is NP-hard for large $K$. Ranger provides two core modes:
1. **`partition` (Exhaustive)**: For $K \le 8$, evaluates all $2^{K-1}-1$ subsets.
2. **`order` (Target-Ordered Greedy Sort)**:
   - For binary classification or regression: Levels are ordered by their conditional mean $\mathbb{E}[Y \mid X_j = c]$.
   - Fisher (1958) proved that the optimal split on unordered categoricals with convex loss is guaranteed to be a contiguous slice along this sorted order!
   - This reduces the combinatorial $O(2^K)$ subset search to a trivial $O(K \log K)$ sorting step, making high-cardinality categoricals as fast as continuous numeric features.

### 2.4 Splitting Rules: Beyond Gini
Ranger supports multiple splitting rules per node:
- **`gini`**: Standard impurity reduction $\Delta I = I_{\text{parent}} - \frac{N_L}{N} I_L - \frac{N_R}{N} I_R$.
- **`extratrees` (Geurts et al., 2006)**: For each candidate feature selected in `mtry`, randomly draws *one* cut point from $[\min(x_j), \max(x_j)]$ instead of scanning all thresholds. Extremely fast; reduces variance at the expense of slight bias.
- **`hellinger`**: Hellinger distance criterion specifically designed for highly skewed, imbalanced class distributions.

---

## 3. Generalized Random Forests (GRF): Theoretical & Algorithmic Mechanics

**Reference**: Athey, S., Tibshirani, J., & Wager, S. (2019). *Generalized Random Forests*. Annals of Statistics, 47(2), 1148–1178.  
**Reference**: Friedberg, R., Tibshirani, J., Athey, S., & Wager, S. (2020). *Local Linear Forests*. Journal of Computational and Graphical Statistics, 30(2), 503–517.

### 3.1 Paradigm Shift: Forests as Adaptive Kernel Weights
Traditional random forests view each tree as a predictor $\hat{\mu}_b(x)$, returning the ensemble average $\frac{1}{B} \sum_{b=1}^B \hat{\mu}_b(x)$.

**GRF radically re-interprets the forest as an adaptive nearest-neighbor kernel**:
A forest defines a similarity weight $w_i(x)$ between training point $i$ and query point $x$:
$$w_i(x) = \frac{1}{B} \sum_{b=1}^B \frac{\mathbf{1}(X_i \in L_b(x))}{|L_b(x)|}$$
where $L_b(x)$ is the leaf containing $x$ in tree $b$, and $|L_b(x)|$ is the number of training samples in that leaf.
- $\sum_{i=1}^N w_i(x) = 1$.
- Samples that frequently land in the same leaf as $x$ receive high weights; distant samples receive zero weight.

Once the adaptive weights $w_i(x)$ are computed, **any econometric estimating equation, M-estimator, or GMM problem can be solved locally at $x$**:
$$\sum_{i=1}^N w_i(x) \psi_{\theta(x)}(Y_i, X_i) = 0$$

Applications derived under this exact framework:
- **Quantile Regression Forest**: Local pinball loss $\psi_\tau(y, q) = \tau - \mathbf{1}(y \le q)$.
- **Causal Forest**: Instrumental variables and heterogeneous treatment effect estimation $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$.
- **Local Linear Forest (`grf_llf`)**: Local polynomial regression solving boundary attenuation.

### 3.2 Local Linear Forests (`ll_regression_forest`)
Standard CART and Breiman forests suffer from severe boundary bias: near the edges of the support, leaves are one-sided, flattening slopes into step functions.

`grf_llf` solves this by computing the forest weights $w_i(x)$ and then fitting a **locally weighted ridge regression** around $x$:
$$\hat{\theta}(x) = \arg\min_{\beta_0, \beta} \sum_{i=1}^N w_i(x) \left( Y_i - \beta_0 - \beta^T (X_i - x) \right)^2 + \lambda \|\beta\|_2^2$$
Prediction at $x$ is simply the fitted intercept $\hat{\beta}_0(x)$.
- Inside the data support: Behaves as a high-capacity adaptive kernel smoother.
- Near or outside boundaries: Retains the linear slope $\beta$, completely eliminating step-function flattening!

### 3.3 Honest Tree Partitioning
To obtain valid statistical inference, confidence intervals, and eliminate search optimism, GRF enforces **Honesty**:
1. For each tree, draw a subsample $\mathcal{S} \subset \{1, \dots, N\}$ *without replacement* (subsampling fraction $s = 0.5$).
2. Split $\mathcal{S}$ evenly into two disjoint subsets:
   - **Splitting set $\mathcal{J}_1$** ($|\mathcal{J}_1| = sN/2$)
   - **Estimation set $\mathcal{J}_2$** ($|\mathcal{J}_2| = sN/2$)
3. **Partition phase**: The tree splits, thresholds, and topology are grown **strictly using $\mathcal{J}_1$**.
4. **Estimation phase**: The data in $\mathcal{J}_1$ are discarded from the leaves. The tree topology is held fixed, and only observations from $\mathcal{J}_2$ are dropped down the tree to populate the leaf kernel weights.

*Theoretical consequence*: Conditioned on the tree structure, the outcomes in the leaves are independent and identically distributed, enabling asymptotic normality:
$$\frac{\hat{\theta}(x) - \theta(x)}{\sigma(x)} \xrightarrow{d} \mathcal{N}(0, 1)$$

### 3.4 Gradient-Based Splitting Criterion
How can a tree split to maximize the fit of a complex local model (like IV regression or local linear regression) without solving an expensive optimization problem at every single candidate threshold?

GRF introduces **Gradient Tree Splitting**:
1. At node $P$, evaluate the parent model's estimating equation $\theta_P$.
2. Compute the first-order score vector (pseudo-residuals / influence functions) for each observation $i \in P$:
   $$\rho_i = -\xi^T V^{-1} \psi_{\theta_P}(Y_i, X_i)$$
   where $V = \nabla_\theta \mathbb{E}[\psi]$ and $\xi$ is the vector picking the parameter of interest.
3. The optimal split on variable $j$ at threshold $t$ simply maximizes the between-child variance of the scalar pseudo-residuals:
   $$\Delta = \frac{N_L N_R}{N_L + N_R} (\bar{\rho}_L - \bar{\rho}_R)^2$$
4. **Result**: Splitting is as computationally cheap as standard CART ($O(N \log N)$ per feature), yet mathematically guaranteed to find the split that maximizes parameter heterogeneity in the downstream estimating equation!

---

## 4. Architectural Comparison: Ranger vs GRF vs mltreelib

| Dimension | Ranger (`ranger`) | Generalized Random Forest (`grf`) | `mltreelib` (Current Core) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Ensemble (100–500 trees) | Ensemble (100–2,000 trees) | Single Tree / Small Committees (1–10) |
| **Leaf Model** | Empirical class distribution $\mathbf{p} \in \mathbb{R}^K$ | Adaptive kernel weight $w_i(x)$ solving local M-estimator | Multivariate regularized linear model $\mathbf{w}^T \mathbf{x} + b$ |
| **Sampling** | Bootstrap (with replacement) or Subsample | Subsample without replacement (50%) | Full sample or Out-of-Bag (Committees) |
| **Honesty** | No (adaptive optimism present) | **Yes** (disjoint $\mathcal{J}_1$ split vs $\mathcal{J}_2$ estimation) | No (full partition fit + parent Bayesian shrinkage) |
| **Splitting Speed** | C++ in-place partition, $O(N \log N)$ | Gradient score vector approximation, $O(N \log N)$ | Pure Rust SIMD Cholesky scan / Welford variance |
| **Extrapolation** | Step-function flatline | Local linear slope continuation (`ll_regression_forest`) | Piecewise linear projection + range clamping |
| **Primary Strength** | Raw execution speed, high-cardinality categoricals | Theoretical guarantees, causal inference, local linearity | Complete inspectability, zero dependencies, pure Rust SIMD |

---

## 5. Blueprint for `mltreelib-core`: What We Can Adopt

### 5.1 Step 1: Probability Estimation Leaves for `ModelForest` (Ranger Lineage)
When building forest ensembles in Rust (`crates/mltreelib-core`):
- Maintain empirical frequency histograms in each leaf.
- Accumulate softmax logits across trees in parallel using Rayon.
- Avoid 0/1 voting; average continuous leaf probabilities.

### 5.2 Step 2: Gradient-Based Splitting for Fast Model Tree Growth (GRF Lineage)
Currently, evaluating linear leaves during recursive partitioning requires updating matrix moments $(X^T X)^{-1} X^T y$.
- By adopting GRF's **Gradient-Based Splitting**:
  1. Fit linear model at parent node $P \to$ get residuals $r_i = y_i - \hat{y}_i$.
  2. Evaluate candidate splits purely on the variance reduction of the parent linear residuals $r_i$!
  3. Only solve the $(X^T X + \lambda I)^{-1}$ Cholesky factorization *once* at each final child node.
- **Speedup**: Eliminates $O(P)$ incremental matrix operations per split scan, speeding up Model Tree growth by $10\times$ to $50\times$ on wide feature sets!

### 5.3 Step 3: Forest-Induced Adaptive Kernel Weights (`forest.weights(x)`)
Expose an API in `mltreelib-core`:
```rust
pub fn sample_weights(&self, query: &[f64]) -> Vec<(usize, f64)>
```
Returns sparse weights $w_i(x)$ representing leaf co-occurrence across the ensemble. This unlocks:
- Arbitrary downstream plug-in estimators (local Quantile regression, local Tobit censored regression, local causal treatment effects).
- Native Local Linear Forest (`ll_regression_forest`) support in pure Rust.
