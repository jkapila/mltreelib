# Niche Algorithms for Longitudinal, Panel, and Tensor Time Series in Tree Architectures

**Date**: 2026-11-01  
**Author**: Jitin Kapila & Antigravity  
**Topic**: Advanced catalog of niche, cutting-edge architectures combining Tensor-variate statistics, Matrix Autoregression (MAR), Dynamic Factor Models (DFM), Path Signatures, and Functional Data Analysis (FDA) with decision trees and model trees.

---

## 1. Executive Summary & Taxonomy of Niche Methods

Beyond standard univariate tabular GBDT and classical econometric VAR/VECM, there exists an elite family of mathematical architectures specifically designed to process high-dimensional longitudinal and multi-way tensor arrays.

```
                  Advanced Multi-Way / Longitudinal Tree Architectures
                                           │
  ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
  ▼                  ▼                     ▼                     ▼                  ▼
[1. Tensor Trees]  [2. Matrix AR (MAR)]  [3. Dynamic Factor]   [4. Path Signature]  [5. Functional Trees]
 Low-rank CP/Tucker Bilinear A X B^T      Latent state-space    Iterated integrals    Curve integrals &
 array splits &     matrix dynamics at   macro factors +       tensor algebra +      derivatives (FPCA)
 tensor leaves.     tree partitions.     tree regimes.         tree splits.          in tree splits.
```

---

## 2. Detailed Algorithm Deep Dives

### 2.1. Tensor-Input Decision Trees & Forests (TT / Tensor Trees)
* **Pioneering Work**: Luo, Horiguchi, & Ma (arXiv:2305.10985, 2023) — *Efficient Decision Trees for Tensor Regressions*; Kolda & Bader (SIAM Review 2009).
* **Data Structure**: Multi-way tensor $\boldsymbol{\mathcal{X}}_i \in \mathbb{R}^{T \times P \times K}$ per entity $i$ (e.g. Days $\times$ Features $\times$ Lag horizons).
* **Mathematical Mechanics**:
  1. **Tensor Splits**: Instead of splitting on a single scalar feature $x_j \le \theta$, TT evaluates splits along multi-way tensor slices or mode-$k$ fiber projections:
     $$\langle \boldsymbol{\mathcal{X}}_i, \mathbf{u}_1 \circ \mathbf{u}_2 \circ \dots \circ \mathbf{u}_D \rangle \le \theta$$
     where $\circ$ denotes outer product and $\mathbf{u}_d$ are rank-1 projection vectors.
  2. **Low-Rank Leaf Models**: In leaf $m$, the target tensor or trajectory is modeled via **CP (Canonical Polyadic) decomposition** or **Tucker core decomposition**:
     $$\hat{\boldsymbol{\mathcal{Y}}}^{(m)} = \sum_{r=1}^R \lambda_r \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r$$
* **Strength for M5**: Naturally models 3D retail panels (Store $\times$ Department $\times$ Date) without flattening into 2D or losing multi-way geometry.

---

### 2.2. Matrix Autoregressive (MAR) Trees
* **Pioneering Work**: Chen, Xiao, & Yang (JASA 2021) — *Autoregressive models for matrix-valued time series*; Wang, Zheng, & Chen (2023).
* **Data Structure**: Matrix-valued time series $\mathbf{Y}_t \in \mathbb{R}^{R \times C}$ at time $t$ (e.g. $R$ stores $\times C$ product categories).
* **Mathematical Mechanics**:
  - A standard VAR on vectorized $\text{vec}(\mathbf{Y}_t)$ would require estimating $(RC)^2$ parameters (for $R=10, C=10$, that is $100^2 = 10{,}000$ coefficients, causing catastrophic overfitting).
  - **MAR(1) Bilinear Formulation**:
    $$\mathbf{Y}_t = \mathbf{A} \mathbf{Y}_{t-1} \mathbf{B}^T + \mathbf{E}_t$$
    where $\mathbf{A} \in \mathbb{R}^{R \times R}$ captures cross-store spatial spillover, and $\mathbf{B} \in \mathbb{R}^{C \times C}$ captures cross-category basket substitution. Total parameters: only $R^2 + C^2 = 200$ ($98\%$ parameter reduction!).
* **Tree Extension (MAR Model Trees)**:
  - Tree recursively partitions on exogenous macro conditions (holiday, inflation regime, weather).
  - In leaf $m$, solve the bilinear least squares system via alternating least squares (ALS) using `faer::cholesky`:
    $$\min_{\mathbf{A}, \mathbf{B}} \sum_{t \in \mathcal{S}_m} \|\mathbf{Y}_t - \mathbf{A} \mathbf{Y}_{t-1} \mathbf{B}^T\|_F^2$$

---

### 2.3. Dynamic Factor Trees & Forests (DFM-Trees / Factor-Augmented Trees)
* **Pioneering Work**: Goulet Coulombe (Journal of Applied Econometrics 2020) — *Dynamic Factor Trees and Forests*; Stock & Watson (JASA 2002); Bai & Ng (Econometrica 2002).
* **Mathematical Mechanics**:
  1. Large panel of $N$ series ($N \gg 1{,}000$) is decomposed into $K \ll N$ common dynamic factors $\mathbf{f}_t \in \mathbb{R}^K$ and idiosyncratic shocks $\mathbf{e}_t$:
     $$\mathbf{y}_t = \boldsymbol{\Lambda} \mathbf{f}_t + \mathbf{e}_t, \quad \mathbf{f}_t = \sum_{p=1}^P \mathbf{\Phi}_p \mathbf{f}_{t-p} + \boldsymbol{\eta}_t$$
  2. **Tree Augmentation**: Standard linear DFM assumes factor loadings $\boldsymbol{\Lambda}$ and transition dynamics $\mathbf{\Phi}$ are time-invariant.
  3. DFM-Trees allow factor dynamics to switch non-linearly according to recursive tree rules:
     $$\mathbf{f}_t = \mathbf{\Phi}^{(m)} \mathbf{f}_{t-1} + \boldsymbol{\eta}_t \quad \text{if } \mathbf{x}_t \in \mathcal{R}_m$$
* **Strength for M5**: Compresses 30,490 store-item sales series into 5–10 macro supply-chain factors (national demand, holiday wave, pay-day cycle), with trees routing regime changes.

---

### 2.4. Rough Path Theory & Path Signature Ensembles (Signature-GBDT)
* **Pioneering Work**: Terry Lyons (1998, 2014) — *Rough paths, signatures and the modelling of functions on streams*; Chevyrev & Kormilitzin (arXiv:1603.03788); Kidger et al. (NeurIPS 2020).
* **Mathematical Mechanics**:
  - For a multi-channel continuous longitudinal path $X: [0, T] \to \mathbb{R}^d$ (e.g. $[y_{it}, \text{price}_{it}, \text{promo}_{it}]$):
  - The **Path Signature** $S(X)$ is the infinite collection of iterated path integrals:
    $$S(X)_{s,t}^{i_1, \dots, i_k} = \int_{s < u_1 < \dots < u_k < t} dX_{u_1}^{i_1} \dots dX_{u_k}^{i_k}$$
  - Truncated at level $M$ (e.g. $M=3$), $S_M(X)$ yields a fixed-length geometric feature vector.
* **Why Path Signatures are Ideal for Trees**:
  1. **Invariance to Sampling Frequency**: Irregular time grids, missing dates, and varying lengths map into the same fixed signature dimension.
  2. **Nonlinear Interaction Preservation**: Cross-iterated integrals ($\int y_t d\text{price}_t$) explicitly capture price-elasticity momentum without manual feature engineering.
  3. Tree ensembles (LightGBM / `mltreelib`) split directly on signature coordinates.

---

### 2.5. Functional Regression Trees & Forests (FDA Curves)
* **Pioneering Work**: Möller, Tutz, & Gertheiss (Statistical Modelling 2016) — *Functional trees and forests for longitudinal data*; Brockhaus et al. (2017) *Signal regression models for functional data based on boosting*.
* **Mathematical Mechanics**:
  - Smooth each series into continuous trajectory $x_i(t) = \sum c_{ik} B_k(t)$ using cubic B-splines.
  - Evaluate splitting rules based on:
    - **Integral projections**: $\int_{t_1}^{t_2} w(t) x_i(t) dt \le \theta$ (aggregate energy / moving volume).
    - **Functional derivatives**: $\left. \frac{d x_i(t)}{dt} \right|_{t^*} \le \theta$ (instantaneous velocity / trend reversal).
    - **Functional curvature**: $\frac{d^2 x_i(t)}{dt^2}$ (acceleration / panic buying shocks).
* **Strength for M5**: Categorizes items into physical trajectory archetypes (dying, expanding, erratic spike, periodic) directly in the split evaluation loop.

---

### 2.6. Matrix Completion & Synthetic Control Trees
* **Pioneering Work**: Athey, Bayati, Doudchenko, Imbens, & Khosravi (JASA 2021) — *Matrix Completion Methods for Causal Panel Data*.
* **Data Structure**: Incomplete wide panel $\mathbf{Y} \in \mathbb{R}^{N \times T}$ where future values or unobserved counterfactuals are missing.
* **Mathematical Mechanics**:
  $$\min_{\mathbf{L}} \sum_{(i,t) \in \Omega} (Y_{it} - L_{it})^2 + \lambda \|\mathbf{L}\|_*$$
  where $\|\mathbf{L}\|_* = \sum \sigma_i(\mathbf{L})$ is the nuclear norm (sum of singular values).
* **Tree Fusion**: Nuclear norm low-rank completion acts as leaf regularizer or imputing split evaluator.

---

## 3. Comparison Matrix of Niche Architectures

| Algorithm | Primary Input Representation | Core Mechanism | Dimensionality Scaling | Implementation Complexity | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tensor Trees (TT)** | 3D Array ($T \times P \times K$) | CP / Tucker low-rank leaf decomposition | $O(N \cdot T \cdot P \cdot K)$ | **High** | Multi-channel spatio-temporal panels |
| **Matrix AR (MAR)** | Matrix Series ($\mathbf{Y}_t \in \mathbb{R}^{R \times C}$) | Bilinear $\mathbf{A} \mathbf{Y}_{t-1} \mathbf{B}^T$ | $O(R^2 + C^2)$ vs $O(R^2 C^2)$ | **Medium** | Store $\times$ Dept retail grids |
| **Dynamic Factor Trees (DFM)** | Panel Matrix ($N \times T$) | PCA/Kalman common factor projection | $O(N \cdot K + K^2)$ | **Medium** | Macroeconomic & high-cardinality demand |
| **Path Signatures + Trees** | Continuous Stream $X(t)$ | Iterated path integrals ($S_M(X)$) | $O(\sum_{k=1}^M d^k)$ | **Low–Medium** | Irregular sampling & price-sales interaction |
| **Functional Trees (FDA)** | Spline Curves $x_i(t)$ | Spline basis & derivative splits | $O(N \cdot K_{\text{basis}})$ | **Medium** | Trajectory shape clustering |
| **Matrix Completion Trees** | Wide Matrix ($N \times T$) | Nuclear norm low-rank SVD | $O(N \cdot T \cdot \min(N,T))$ | **High** | Missing panels & counterfactual prediction |

---

## 4. Synthesis for `mltreelib` Development

If extending `mltreelib` into high-dimensional longitudinal niches:
1. **Quickest Win with Highest Signal**: **Path Signatures (`timefeat.rs`) + GBDT**.
   - Implementing level-2 and level-3 iterated integrals in pure Rust takes ~250 lines of code.
   - Converts wide history into compact, parameterization-invariant features with zero new dependencies.
2. **Most Statistically Elegant**: **Matrix Autoregressive (MAR) Leaf Models**.
   - Solves the $N \times T$ interaction problem without parameter explosion via bilinear alternating least squares (`faer::llt`).
3. **Most Scalable for Massive Panels (M5)**: **Dynamic Factor Trees (DFM-Trees)**.
   - SVD on the wide matrix extracts 5–10 global factors; trees partition regional/store behaviors on those factors.
