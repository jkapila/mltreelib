# Longitudinal Data & Wide-Format Time Series Architectures for Trees

**Date**: 2026-11-01  
**Author**: Jitin Kapila & Antigravity  
**Topic**: Mathematical and architectural research on training tree-based models and model trees on longitudinal (wide / columnar) panel data for multi-horizon forecasting (e.g., M5, Favorita, retail hierarchies).

---

## 1. Executive Summary & Problem Formulation

In longitudinal and panel time series, data arrives naturally in two formats:
1. **Vertical / Long Format**: Each observation is a separate record `(series_id, timestamp, y_t, x_t)`.
2. **Horizontal / Wide Format**: Each record is an individual entity/series $i \in \{1, \dots, N\}$, with temporal measurements arrayed horizontally across columns $t \in \{1, \dots, T\}$:
   $$\mathbf{Y} \in \mathbb{R}^{N \times T}, \quad \mathbf{y}_i = [y_{i,1}, y_{i,2}, \dots, y_{i,T}]$$
   Accompanied by static attributes $\mathbf{X}_i^{\text{static}} \in \mathbb{R}^{P_{\text{static}}}$ (e.g. store, category, brand) and dynamic covariates $\mathbf{Z}_{i,t} \in \mathbb{R}^{P_{\text{dyn}}}$ (e.g. price, promotions, calendar events).

### The Dilemma in High-Dimensional Panel Competitions (M5 Benchmark)
In the Walmart M5 competition:
- $N = 30{,}490$ series (items $\times$ stores).
- $T \approx 1{,}941$ days historical sales.
- Target horizon $H = 28$ days ahead.

| Format | Row Count | Column Count | Memory Footprint (Float32) | GBDT Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Wide / Columnar** | $30{,}490$ rows | $\approx 1{,}941$ columns | **~237 MB** (trivial) | Out-of-the-box GBDT cannot predict full trajectory |
| **Melted Long** | **~59.18 Million** rows | ~15–30 feature cols | **~4–8 GB** raw, **>20 GB** with lag/rolling features | Standard GBDT target, but severe memory & fit bottlenecks |

Standard industrial practice (e.g. Kaggle M5 winners, LightGBM, CatBoost) brute-forces this by **melting to long format**, truncating history (e.g. training only on the last 365–730 days to stay under 32 GB RAM), and engineering lag/rolling columns.

This research surveys **how to train tree architectures directly or semi-directly on wide longitudinal representations**, analyzing four distinct paradigms, their mathematical mechanics, literature lineage, and implications for `mltreelib`.

---

## 2. Taxonomy of Approaches for Wide Longitudinal Data

```
                         Longitudinal Tree Architectures
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 [1. Wide Multi-Output /]     [2. Functional Trees /]      [3. Model Trees / Mixed]
 [      MIMO GBDT       ]     [    FDA Splitting    ]      [    Effects (RE-EM)   ]
   (Predict vector y           (Treat row as curve          (Split on static X,
    at leaf directly)           split on FPC/shape)          fit leaf trajectory)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       [4. Strided Virtual Blocks]
                       (Zero-copy sliding window
                        views over wide matrices)
```

---

## 3. Paradigm 1: Direct Multi-Output / MIMO Trees (Wide-In, Vector-Out)

Rather than predicting a single scalar $y_{t+1}$ with a univariate tree, the target at each row is the vector of future horizons:
$$\mathbf{y}_i = [y_{i,T+1}, y_{i,T+2}, \dots, y_{i,T+H}]^T \in \mathbb{R}^H$$

### A. Multivariate Split Criterion (Trace of Dispersion)
For a candidate split partition $S \to (S_L, S_R)$ at node $S$:
Instead of scalar variance, the impurity of a node is measured by the **trace of the sample covariance / sum of squared Euclidean deviations**:
$$I(S) = \sum_{i \in S} \|\mathbf{y}_i - \bar{\mathbf{y}}_S\|_2^2 = \sum_{h=1}^H \sum_{i \in S} (y_{i,T+h} - \bar{y}_{S,h})^2$$

The impurity reduction (split gain) is:
$$\Delta I = \sum_{h=1}^H \left[ \frac{\left(\sum_{i \in S_L} y_{i,T+h}\right)^2}{|S_L|} + \frac{\left(\sum_{i \in S_R} y_{i,T+h}\right)^2}{|S_R|} - \frac{\left(\sum_{i \in S} y_{i,T+h}\right)^2}{|S|} \right]$$

### B. Streaming Complexity
- Standard univariate linear scan: Maintains 1 running sum $\sum y$.
- Vector multivariate scan: Maintains an $H$-dimensional accumulator vector $\mathbf{s}_L = \sum_{i \in S_L} \mathbf{y}_i \in \mathbb{R}^H$.
- For $H=28$ (e.g. M5 28-day forecast), updating $\mathbf{s}_L$ and computing dot product $\frac{\mathbf{s}_L^T \mathbf{s}_L}{|S_L|}$ is SIMD-parallelizable (AVX2/AVX-512) and takes $O(H)$ per candidate cut.
- Total split search cost across $N$ rows and $P$ features: $O(P \cdot N \cdot H)$. With $N = 30{,}490$, this evaluates in **tens of milliseconds**, versus minutes on melted long tables!

### C. Leaf Model
The leaf simply outputs the centroid trajectory:
$$\hat{\mathbf{y}}_{\text{leaf}} = \frac{1}{|S_{\text{leaf}}|} \sum_{i \in S_{\text{leaf}}} \mathbf{y}_i \in \mathbb{R}^H$$

### D. Trade-Offs
- **Pros**: Zero memory inflation ($N=30{,}490$ rows throughout). Captures multi-day temporal autocorrelation and cross-horizon dependencies directly in the leaf vector. No error accumulation (unlike recursive single-step forecasting).
- **Cons**: Exogenous dynamic covariates for future periods (e.g. discounts planned on day $T+5$) must be flattened into column blocks $X_{\text{price}, T+1 \dots T+H}$.

---

## 4. Paradigm 2: Functional Data Analysis (FDA) Trees & Curve Partitioning

In Functional Data Analysis (Ramsay & Silverman 2005), wide rows are treated not as discrete sets of points, but as discretely sampled continuous curves:
$$X_i(t) = \sum_{k=1}^K c_{ik} \phi_k(t), \quad t \in [0, T]$$
where $\phi_k(t)$ are orthogonal basis functions (B-splines or Fourier harmonics).

### A. Functional Principal Component Analysis (FPCA) Basis
1. Compute the empirical covariance surface $G(s, t) = \text{Cov}(y_i(s), y_i(t))$ across the wide columns.
2. Extract the leading functional eigencomponents (FPCs) $\psi_1(t), \dots, \psi_m(t)$:
   $$\xi_{im} = \int X_i(t) \psi_m(t) dt \approx \sum_{t=1}^T y_{it} \psi_m(t)$$
3. **Splitting Mechanics**: The tree splits on functional scores $\xi_{im}$ representing:
   - $\xi_{i1}$: Baseline scale / overall sales volume.
   - $\xi_{i2}$: Overall linear trajectory / trend (growing vs decaying items).
   - $\xi_{i3}$: Seasonal amplitude / weekend volatility.

### B. Functional Recursive Partitioning (Möller et al. 2016, Nerini & Ghattas 2007)
Trees split directly on functional integrals:
$$\text{Rule}: \int_{t_a}^{t_b} y_i(t) dt \le \theta \quad \text{or} \quad \frac{d}{dt}y_i(t) \le \theta$$
- **Advantage for M5**: Naturally clusters items into "erratic/intermittent", "smooth high-volume", "strongly seasonal", and "dying product" curves without flattening or expanding rows.

---

## 5. Paradigm 3: Model-Based Trees & Mixed-Effects Trees (RE-EM / MERF / MOB)

Instead of leaves outputting flat scalar or vector means, model trees fit **parametric temporal models** inside each leaf partition.

### A. Literature Foundations
- **RE-EM Trees** (Sela & Simonoff, JASA 2012): Combines regression trees for fixed effects with linear mixed-effects (LME) models for longitudinal clustering:
  $$y_{it} = f(\mathbf{x}_{it}) + \mathbf{z}_{it}^T \mathbf{b}_i + \epsilon_{it}$$
  where $\mathbf{b}_i \sim \mathcal{N}(0, \mathbf{D})$ is a subject-level random effect (random intercept / random slope), and $f(\cdot)$ is partitioned by a decision tree.
- **MERF** (Mixed Effects Random Forests, Hajjem et al. 2014): Extends RE-EM to tree ensembles.
- **MOB / Partykit** (Zeileis, Hothorn, Hornik 2008): Model-Based Recursive Partitioning. Fits local models (e.g. GLM, ARIMA, autoregression) and evaluates parameter instability across candidate split features.

### B. Application to Wide Panel Series in `mltreelib`
In wide format, each row $i$ has static attributes $\mathbf{x}_i^{\text{static}}$ (e.g. Store, Category, Dept, State) and time vector $\mathbf{y}_i \in \mathbb{R}^T$.

```
                    Root Node: All 30,490 Series
                                 │
                   Split on Dept == "FOODS_3"?
                     ├── YES                     └── NO
          Leaf 1: Foods Model Tree              Leaf 2: Non-Foods Model Tree
          Fit local Ridge AR(p):                Fit local Trend + Harmonics:
          y_{t+h} = \sum \phi_k y_{t-k}         y_{t+h} = \alpha + \beta h + \text{Season}
```

1. **Split Phase**: Split exclusively on static hierarchical codes and summary features (e.g. zero-sales fraction, coefficient of variation).
2. **Leaf Fitting**:
   - Leaf collects the wide trajectories of its assigned subset of series $S_{\text{leaf}} \subset \{1, \dots, N\}$.
   - In leaf $L$, solve a shared regularized parameter vector (e.g. Ridge Regression, Spline, or ElasticNet) predicting future days from past days:
     $$\min_{\mathbf{W}_L} \sum_{i \in S_L} \|\mathbf{y}_{i, \text{future}} - \mathbf{y}_{i, \text{history}} \mathbf{W}_L\|_2^2 + \lambda \|\mathbf{W}_L\|_F^2$$
3. **Why this dominates flat trees for longitudinal data**:
   - Captures local dynamics per product cluster without needing 100,000 deep splits.
   - Solves the catastrophic overfitting of leaf averages on non-stationary trends.

---

## 6. Paradigm 4: Strided Virtual Blocks (Zero-Copy Sliding Ingestion)

The biggest barrier to the Melted Long format is **RAM bloat** caused by eager dataframe duplication in Python (e.g. `pandas.melt()` duplicating IDs and static attributes 1,941 times).

### A. The In-Engine Solution: Strided Window Memory Layout
In a native Rust engine (`mltreelib-core`), wide data can be kept in a contiguous flat buffer:
$$\text{Buffer}: \text{Vec}\langle f32 \rangle \text{ of size } N \times T$$

For a history window of length $W$ (e.g. 56 days) and forecast horizon $H$ (e.g. 28 days):
- Rather than allocating $(T - W - H) \times N$ new physical rows, the engine constructs **strided virtual views**:
  $$\text{RowIndex}(i, \tau) \implies \text{base\_ptr} + i \cdot T + \tau$$
- Lags $y_{i, \tau - k}$ and target $y_{i, \tau + h}$ are resolved via pointer offsets directly into the contiguous buffer without allocation!
- **Cache locality**: All historical values for series $i$ reside contiguously on the same cache line.

### B. Memory Comparison on M5 Benchmark

| Strategy | Memory Allocation | Memory Re-indexing | L1/L2 Cache Efficiency |
| :--- | :--- | :--- | :--- |
| **Pandas / Polars Melt** | ~12–25 GB RAM | High GC & allocation overhead | Poor (strided row-pointer jumps) |
| **Rust Contiguous Buffer** | **~240 MB RAM** | **Zero-copy** pointer offset | Optimal (sequential 32-bit floats) |

---

## 7. Direct vs Recursive vs Multi-Horizon: Empirical Findings from M5 Winners

The M5 competition (Makridakis et al., *International Journal of Forecasting* 2022) revealed decisive lessons on multi-horizon strategies:

### 1. Pure Recursive Strategy (1-Step Ahead Iterated)
- Fits a single model: $\hat{y}_{t+1} = f(y_t, y_{t-1}, \dots)$.
- To predict $t+2 \dots t+28$, iteratively feeds predicted $\hat{y}_{t+1}$ into lag features.
- *Outcome*: Severe error compounding. Underpredicted volatility over horizons $>14$ days.

### 2. Pure Direct Strategy (28 Independent Models)
- Fits 28 separate models: $\hat{y}_{t+h} = f_h(y_t, y_{t-7}, \dots)$ for $h \in \{1, \dots, 28\}$.
- *Outcome*: Avoids error accumulation. Highly accurate, but requires $28\times$ compute and fails to enforce smoothness across consecutive days (day 14 and day 15 predictions can oscillate unrealistically).

### 3. Top-Tier Winning Architecture: Direct + Recursive Blends (DRFAM)
- **1st place solution (KaKi)** and top ensembles used a hybrid:
  - Direct models grouped by week (Horizon $1–7$, $8–14$, $15–21$, $22–28$).
  - Lag features restricted to $\text{lag} \ge 28$ days (preventing reliance on recursive estimates).
  - Multi-level hierarchical reconciliation (bottom-up aggregation combined with top-down volume scaling).

---

## 8. Synthesis & Strategic Roadmap for `mltreelib`

To build production-grade, state-of-the-art support for longitudinal panel data in `mltreelib`:

```
                 Phase 1: Zero-Copy Strided Buffer
                 • Ingest wide contiguous matrix (N x T).
                 • Provide virtual row iterator with lag offsets.
                                  │
                                  ▼
                 Phase 2: Multi-Output Vector Splits
                 • SIMD-accelerated trace-of-covariance split gain.
                 • Direct prediction of H-dimensional vectors at leaves.
                                  │
                                  ▼
                 Phase 3: Model Trees on Panel Trajectories
                 • Split on static group hierarchy (Store, Dept, Brand).
                 • Fit local Ridge AR(p) / seasonal dynamics at leaves.
```

1. **Keep ingestion wide**: Never force users to melt 60M rows in Python. Accept wide parquet / arrow buffers directly.
2. **Implement Vector Multi-Output Trees**: Enable $H$-step direct forecasting natively in `mltreelib-core` via SIMD trace dispersion.
3. **Leverage the Model Tree advantage**: Flat trees cannot extrapolate time-series trends without infinite step functions. Fitting local autoregressive models at leaves provides smooth, bounded multi-horizon trajectory forecasts.

---

## 9. References & Primary Literature

1. **Sela, R. J., & Simonoff, J. S. (2012)**. *RE-EM trees: a data mining approach for longitudinal and clustered data*. Journal of the American Statistical Association, 107(497), 206-213.
2. **Hajjem, A., Bellavance, F., & Larocque, D. (2014)**. *Mixed-effects random forest for clustered data*. Journal of Statistical Computation and Simulation, 84(6), 1313-1328.
3. **Zeileis, A., Hothorn, T., & Hornik, K. (2008)**. *Model-based recursive partitioning*. Journal of Computational and Graphical Statistics, 17(2), 492-514.
4. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022)**. *The M5 accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4), 1346-1364.
5. **Ramsay, J. O., & Silverman, B. W. (2005)**. *Functional Data Analysis*. Springer Series in Statistics.
6. **Möller, A., Tutz, G., & Gertheiss, J. (2016)**. *Functional trees and forests for longitudinal data*. Statistical Modelling, 16(6), 469-495.
7. **Taieb, S. B., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012)**. *A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition*. Expert Systems with Applications, 39(8), 7067-7083.
