# KNN Variants and Leaf-Node DW-KNN Specification

**Date**: 2026-11-18  
**Scope**: Non-parametric leaf models, Double-Weighted KNN (DW-KNN), metric spaces in decision trees, and `mltreelib-core` implementation specification.  
**Author**: mltreelib Core & Research Team  

---

## 1. Executive Summary

Standard Model Trees (M5, Cubist, GLMTree) place parametric models (multiple linear regression or generalized linear models) at tree leaves. While effective when local relationships are linear or monotonically smooth, parametric leaves fail when sub-populations exhibit complex, non-linear manifolds, multi-modal clusters, or boundary noise.

Conversely, standard **$K$-Nearest Neighbors (KNN)** is a non-parametric instance-based learner that captures arbitrary local geometries. However, global KNN suffers from two fatal limitations on tabular datasets:
1. **Computational Complexity**: Inference scales as $O(N \cdot P)$, requiring full distance scans over large training sets ($N > 10^5$).
2. **Curse of Dimensionality**: High feature counts ($P \gg 10$) distort Euclidean distances; irrelevant features dominate distance metrics.

This document reviews the taxonomy of KNN variants—highlighting **DW-KNN (Double-Weighted KNN: Pathak, Karthik, Madan, Kapila, arXiv:2512.08956)**—and provides a formal engineering specification for **Tree-KNN / Leaf-DW-KNN** within `mltreelib`. In this architecture, recursive tree partitioning serves as a fast hierarchical router into localized, feature-filtered subspaces ($O(\text{depth})$ routing), where an optimized, noise-robust DW-KNN executes over small, cache-resident leaf subsets ($N_L \ll N$).

---

## 2. Comprehensive Taxonomy of KNN Variants

### 2.1 The Baseline: Standard Uniform KNN (Cover & Hart, 1967)
Given a query instance $x \in \mathbb{R}^P$, uniform KNN identifies the $k$ training samples $\mathcal{N}_k(x) = \{x_1, \dots, x_k\}$ minimizing distance $d(x, x_i)$.
* **Classification**: $\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c)$
* **Regression**: $\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i$
* **Limitation**: Treats the 1st nearest neighbor identically to the $k$-th, making it vulnerable to boundary outliers and noise.

### 2.2 Distance-Weighted KNN (Dudani, 1976)
Dudani introduced linear distance weighting:
$$w_i = \begin{cases} \frac{d_k - d_i}{d_k - d_1} & \text{if } d_k \neq d_1 \\ 1 & \text{if } d_k = d_1 \end{cases}$$
Alternatively, inverse distance weighting uses:
$$w_i = \frac{1}{(d(x, x_i) + \epsilon)^p}, \quad p \in \{1, 2\}$$
* **Advantage**: Gives closer neighbors higher influence.
* **Limitation**: Still assumes all training samples are ground-truth reliable. A mislabeled or noisy sample very close to $x$ receives an excessively high weight.

### 2.3 Kernel / Exponential Distance KNN
Replaces ad-hoc inverse distances with radial basis / exponential attenuation:
$$w_{\text{dist}}(x, x_i) = \exp\left(-\gamma \cdot d(x, x_i)^2\right) \quad \text{or} \quad \exp\left(-\gamma \cdot d(x, x_i)\right)$$
Provides smooth asymptotic falloff with distance scale parameter $\gamma$.

### 2.4 Dual-Weighted KNN (Gou et al., 2012)
Gou et al. introduced dual weighting by multiplying query-to-neighbor distance weight by a neighbor compactness weight derived from the training set:
$$W(x, x_i) = w(x, x_i) \cdot w_{\text{compactness}}(x_i)$$
Where compactness measures average intra-class distance in $x_i$'s own neighborhood.

### 2.5 Double-Weighted KNN: DW-KNN (Pathak, Karthik, Madan, Kapila, 2025; arXiv:2512.08956)
DW-KNN introduces an instance-level transparent formulation integrating **Distance Consistency** with **Neighbor Reliability (Validity)**:

#### 1. Distance Consistency ($w_{\text{dist}}$)
Distance consistency uses an exponential decay function to model proximity confidence:
$$w_{\text{dist}}(x, x_i) = \exp\left(-\gamma \cdot d(x, x_i)\right)$$
where $\gamma > 0$ controls the rate of spatial decay (adaptively scaled by the local neighborhood radius $\bar{d}_k$).

#### 2. Neighbor Reliability / Validity ($v_i$)
To prevent mislabeled points, label flip noise, or ambiguous boundary instances from corrupting predictions, each training point $x_i$ is assigned a precomputed **reliability score** $v_i \in [0, 1]$ based on agreement within its own secondary neighborhood $\mathcal{N}_{K_v}(x_i)$:
* **Classification**:
  $$v_i = \frac{1}{K_v} \sum_{j \in \mathcal{N}_{K_v}(x_i)} \mathbb{I}(y_i = y_j)$$
* **Continuous / Regression**:
  $$v_i = \exp\left(-\frac{|y_i - \bar{y}_{\mathcal{N}_{K_v}(x_i)}|}{\sigma_{y,\mathcal{N}}}\right)$$
If a training instance is surrounded by inconsistent labels or extreme residual deviations, $v_i \to 0$.

#### 3. Integrated Double-Weight ($W$)
The final weight assigned to neighbor $x_i$ when predicting query $x$ is:
$$W(x, x_i) = w_{\text{dist}}(x, x_i) \times v_i = \exp\left(-\gamma \cdot d(x, x_i)\right) \times v_i$$
Predictions are obtained via normalized weighted consensus:
$$\hat{p}(y = c \mid x) = \frac{\sum_{i \in \mathcal{N}_k(x), y_i = c} W(x, x_i)}{\sum_{i \in \mathcal{N}_k(x)} W(x, x_i)}$$
$$\hat{y}_{\text{reg}}(x) = \frac{\sum_{i \in \mathcal{N}_k(x)} W(x, x_i) \, y_i}{\sum_{i \in \mathcal{N}_k(x)} W(x, x_i)}$$

#### 4. Theoretical Advantages of DW-KNN:
- **Outlier / Noise Immunity**: Mislabeled points close to query boundaries have high $w_{\text{dist}}$ but very low $v_i$; the product $W(x, x_i)$ effectively vanishes.
- **Reduced $k$-Sensitivity**: Because unreliable and distant points attenuate naturally, accuracy remains stable across wide choices of $k \in [5, 50]$.
- **Instance-Level Explainability**: Every prediction yields an explicit, inspectable breakdown: (i) which neighbors voted, (ii) their spatial distance, and (iii) their historical reliability.

---

## 3. The Tree-KNN / Leaf-DW-KNN Paradigm

### 3.1 Why KNN Belongs at the Leaf Node
Placing DW-KNN inside tree leaves resolves the mutual weaknesses of both Decision Trees and KNN:

```
                      [ Root: Split X1 <= 4.5 ]
                             /          \
              [ Split X2 <= 10.2 ]     [ Split X3 in {'A', 'B'} ]
                   /          \                 /          \
             Leaf 1         Leaf 2        Leaf 3         Leaf 4
           (DW-KNN:       (DW-KNN:       (DW-KNN:       (DW-KNN:
          N_L = 120)     N_L = 85)      N_L = 340)     N_L = 95)
```

| Dimension | Global KNN | Standard Decision Tree | **Leaf DW-KNN Hybrid** |
| :--- | :--- | :--- | :--- |
| **Search Space** | $N$ (entire dataset, up to $10^6$) | None (static lookup) | **$N_L$ (leaf subset, 50–500 samples)** |
| **Query Latency** | $O(N \cdot P)$ or slow graph ANN | $O(\text{depth})$ | **$O(\text{depth}) + O(N_L \cdot P_{\text{active}})$ (< 50 $\mu$s)** |
| **Distance Distortion** | Distorted by all $P$ features | N/A (axis splits only) | **Calculated only on localized subspace** |
| **Decision Boundary** | Smooth but unguided | Rigid axis-aligned steps | **Flexible local manifolds within partitioned blocks** |
| **Boundary Noise** | Vulnerable to label noise | Prone to overfitting splits | **Suppressed via neighbor reliability $v_i$** |
| **Memory Footprint** | Global distance index | Minimal | **Cache-local contiguous row-index slices** |

---

## 4. Engineering Specification for `mltreelib-core`

### 4.1 Architecture & Data Structures

In `crates/mltreelib-core/src/model_tree.rs`, extend the leaf model representations:

```rust
/// Supported leaf model architectures in mltreelib.
#[derive(Debug, Clone, PartialEq)]
pub enum LeafFamily {
    /// Constant leaf mean/mode (standard CART/C4.5).
    Constant,
    /// Linear/GLM model fitted to leaf samples (M5/Cubist/GLMTree).
    Glm,
    /// Double-Weighted K-Nearest Neighbors within leaf partition.
    DwKnn {
        /// Number of nearest neighbors to query at inference (default: 7).
        k: usize,
        /// Neighborhood size for precomputing neighbor reliability (default: 5).
        k_validity: usize,
        /// Distance decay parameter gamma (None => auto-scaled by median leaf distance).
        gamma: Option<f64>,
        /// Metric for distance calculation (Euclidean, Manhattan, or Gower for mixed types).
        metric: DistanceMetric,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Gower,
}
```

### 4.2 Leaf Storage Struct

```rust
/// Contiguous leaf-resident dataset slice and precomputed reliability weights.
#[derive(Debug, Clone)]
pub struct LeafKnnStore {
    /// Row indices from the original training frame belonging to this leaf.
    pub row_indices: Vec<usize>,
    /// Precomputed neighbor reliability scores v_i for each leaf row.
    pub validity_scores: Vec<f64>,
    /// Number of active features evaluated in this leaf.
    pub n_features: usize,
    /// Cached empirical leaf target mean/class distribution (fallback if N_L < k).
    pub fallback_mean: f64,
    pub class_counts: Vec<u32>,
}
```

### 4.3 Fit Phase Algorithm

When building a tree with `LeafFamily::DwKnn`:
1. **Tree Growth**: Tree is grown using standard split search (Entropy, Gini, or Variance gain). The splitting logic does not need modification.
2. **Leaf Finalization**:
   For each terminal node with sample subset $S_L \subseteq \{1, \dots, N\}$:
   - If $|S_L| \le k_{\text{validity}}$, set $v_i = 1.0$ for all $i \in S_L$.
   - Else, for each point $i \in S_L$, find its $k_{\text{validity}}$ nearest neighbors within $S_L$ (excluding itself).
   - Compute:
     $$v_i = \frac{1}{k_{\text{validity}}} \sum_{j \in \mathcal{N}_{k_v}(i, S_L)} \mathbb{I}(y_i = y_j)$$
   - Store $S_L$ row indices and $v_i$ array inside `LeafKnnStore`.
   - **Cost**: Performed once during `fit()`. Because each leaf is small ($N_L \le 500$), computing all-pairs distances within the leaf is $O(N_L^2 \cdot P)$, which takes $< 1\text{ ms}$ per leaf in SIMD Rust.

### 4.4 Predict / Inference Phase Algorithm

Given a test query row $x \in \mathbb{R}^P$:
1. **Routing**: Traverse the tree from root to leaf following split tests:
   $$\text{node} = \text{route}(x) \implies \text{Leaf } L$$
2. **Leaf Candidate Evaluation**:
   Retrieve $S_L = \{i_1, \dots, i_{N_L}\}$ and their validity scores $\{v_1, \dots, v_{N_L}\}$ from `LeafKnnStore`.
   - If $N_L \le 1$: return `fallback_mean` / empirical class distribution.
   - Compute distance from $x$ to each training point in $S_L$:
     $$d_j = \text{dist}(x, x_{i_j})$$
   - Select top $k_{\text{eff}} = \min(k, N_L)$ neighbors with smallest distances using a min-heap or partial quickselect (`select_nth_unstable_by`).
   - If `gamma` is unset, scale $\gamma = \frac{1}{\text{median}(d_1, \dots, d_{k_{\text{eff}}}) + 1\text{e-}6}$.
   - Compute final combined weights:
     $$W_j = \exp(-\gamma \cdot d_j) \cdot v_{i_j} + 1\text{e-}8$$
3. **Consensus Aggregation**:
   - For classification:
     $$\hat{P}(Y = c \mid x) = \frac{\sum_{j \in \text{TopK}, y_{i_j} = c} W_j}{\sum_{j \in \text{TopK}} W_j}$$
   - For regression:
     $$\hat{y}(x) = \frac{\sum_{j \in \text{TopK}} W_j \, y_{i_j}}{\sum_{j \in \text{TopK}} W_j}$$

### 4.5 Handling Categoricals & Missing Data in Leaves (Gower Metric)
Tabular data contains categorical variables and missing entries. The distance between query $x$ and neighbor $z$ inside the leaf uses the normalized Gower component:
$$d_{\text{Gower}}(x, z) = \frac{\sum_{p=1}^P \delta_p(x, z) \cdot d_p(x, z)}{\sum_{p=1}^P \delta_p(x, z)}$$
where:
* $\delta_p(x, z) = 0$ if either $x_p$ or $z_p$ is missing ($NaN$), else $1$.
* For continuous features: $d_p(x, z) = \frac{|x_p - z_p|}{\text{range}_p}$.
* For categorical features: $d_p(x, z) = 0$ if $x_p == z_p$, else $1$.

---

## 5. Summary of Expected Benefits

1. **Accuracy**: Bridges the gap between axis-aligned decision trees and non-linear metric manifolds without requiring heavy neural networks.
2. **Speed**: Evaluates KNN in small, partitioned memory pools ($N_L \approx 100$) rather than querying millions of points globally.
3. **Robustness**: DW-KNN's dual-weighting factor $v_i$ neutralizes label noise and split-boundary leakage.
4. **Transparency**: Predictions provide clear evidence chains: `Leaf #12 -> Top 5 Neighbors (Distances: [0.12, 0.18, ...], Validity: [1.0, 0.8, ...])`.
