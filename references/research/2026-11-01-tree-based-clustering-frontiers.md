# Tree-Based Clustering Frontiers: Unsupervised Forests, Predictive Clustering Trees, Evidence Trees, and Manifold Partitioning

**Date**: 2026-11-01  
**Author**: Jitin Kapila & Antigravity  
**Topic**: Comprehensive research survey into clustering using decision trees and recursive partitioning: Unsupervised Random Forest Proximities, Predictive Clustering Trees (PCT/CLUS), Evidence Tree Clustering (Polychoric Odds), Random Projection Trees (RP-Trees), Density Level-Set Trees, and BIRCH CF-Trees.

---

## 1. Executive Summary & The Core Thesis

Traditional clustering algorithms (K-Means, GMM, DBSCAN, Hierarchical Agglomerative) suffer from three fatal weaknesses:
1. **Uninterpretable Cluster Boundaries**: K-Means creates Voronoi cells across linear combinations of all features. Explaining *why* a customer is in Cluster 4 requires inspecting high-dimensional distance formulas.
2. **Failure on Mixed Types & Missing Data**: Euclidean distances break down on mixed continuous/categorical features and cannot handle missing values without imputation artifacts.
3. **Curse of Dimensionality**: Distance concentration makes pairwise Euclidean distances meaningless in high dimensions ($D > 50$).

**The Tree Clustering Solution**:
Decision trees partition space using axis-aligned or manifold-adapted cuts. Every cluster is an **interpretable logical rule** (`IF Income > $80k AND State == CA THEN Cluster 1`), with native missing value handling and invariance to monotonic scaling.

```
                         Tree-Based Clustering Paradigms
                                        │
  ┌───────────────────────┬─────────────┴─────────────┬───────────────────────┐
  ▼                       ▼                           ▼                       ▼
[1. Unsupervised RF]    [2. Predictive]             [3. Evidence Tree]      [4. Space Partitioning]
 Synthetic Null Trick;  [   Clustering Trees (PCT)] Polychoric odds ratio;  RP-Trees, BIRCH CF-Trees,
 Leaf Co-occurrence     Multi-target variance drop; discrete evidence accumulation; Density Level-Set
 Proximity Matrix.      interpretable rule leaves.  positive/negative criteria. Trees.
```

---

## 2. Paradigm 1: Unsupervised Random Forests & Proximity Clustering

* **Foundations**: Leo Breiman & Adele Cutler (2003); Shi & Horvath (*J. Comp. & Graph. Stat.* 2006).
* **The Mathematical Trick (The Synthetic Null)**:
  How do you train a supervised Random Forest when there are no labels $Y$?
  1. Let the original dataset be Class 1: $\mathcal{D}_{\text{real}} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$.
  2. Synthesize a reference null distribution $\mathcal{D}_{\text{synthetic}}$ of size $N$ by **independently permuting each column** of $\mathcal{D}_{\text{real}}$.
     - This preserves the univariate empirical marginal distributions $P(X_j)$ but completely destroys all multivariate dependencies and correlations.
  3. Train a standard binary classification Random Forest to distinguish $\mathcal{D}_{\text{real}}$ ($Y=1$) from $\mathcal{D}_{\text{synthetic}}$ ($Y=0$).
* **The Random Forest Proximity Matrix $\mathbf{P}$**:
  Run all real samples down the trained forest ($M$ trees).
  For every pair of samples $(i, j)$, count the fraction of trees where both samples land in the **exact same terminal leaf**:
  $$P_{ij} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left(\text{leaf}_m(\mathbf{x}_i) == \text{leaf}_m(\mathbf{x}_j)\right)$$
* **Clustering via Dissimilarity**:
  - Distance matrix $\mathbf{D} = \sqrt{1 - \mathbf{P}}$.
  - Feed $\mathbf{D}$ into Partitioning Around Medoids (PAM), Spectral Clustering, or Hierarchical Agglomerative Clustering.
* **Why this dominates standard clustering**:
  - Automatically weights informative features while ignoring noise variables.
  - Invariant to monotonic transformations and scale differences.
  - Native handling of mixed numeric and high-cardinality categorical variables.

---

## 3. Paradigm 2: Predictive Clustering Trees (PCT / CLUS)

* **Foundations**: Hendrik Blockeel, Luc De Raedt, & Jan Ramon (*Machine Learning* 1998, 2000); Džeroski et al. (2007).
* **Core Concept**:
  Instead of viewing a tree as a supervised regressor or classifier, view it as a **hierarchical clusterer where leaves are clusters**.
* **Mathematical Split Criterion**:
  At node $S$, candidate split $S \to (S_L, S_R)$ maximizes variance reduction over a multi-metric distance $d(\cdot)$:
  $$\text{Gain} = \text{Var}(S) - \frac{|S_L|}{|S|} \text{Var}(S_L) - \frac{|S_R|}{|S|} \text{Var}(S_R)$$
  where $\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} d(\mathbf{y}_i, \bar{\mathbf{y}}_S)^2$.
* **Key Innovations**:
  - **Multi-Target Clustering**: Can cluster on target attributes $\mathbf{Y}$ (e.g. multi-step time series or multi-label preferences) while using explanatory features $\mathbf{X}$ as split candidates.
  - **Monothetic Explanations**: Every cluster is defined by a conjunction of simple attribute thresholds:
    $$\text{Cluster 3} \iff \text{Age} \le 35 \land \text{Usage} > 100 \land \text{Tier} == \text{"Gold"}$$

---

## 4. Paradigm 3: Evidence Tree Clustering (Jitin's Architecture)

* **Lineage**: Jitin Kapila (in `references/repos/paper-codebase/misc - work/Evidence Tree Clustering.ipynb`); Fred & Jain (*IEEE TPAMI* 2005, Evidence Accumulation Clustering).
* **The Methodological Workflow**:
  1. **Discretization & Binning**: Discretize continuous attributes into robust quantile/Sturges evidence bins.
  2. **Evidence Factor**: Compute log-odds evidence ratio:
     $$\text{EF} = \ln\left(\frac{P(\text{evidence})}{P(\text{non-evidence})}\right)$$
  3. **Polychoric Dependency Matrix**:
     - Compute contingency tables across pairs of binned variables.
     - Approximate polychoric correlation using cross-product odds ratios:
       $$r_{\text{poly}} \approx \cos\left(\frac{180^\circ}{\sqrt{\frac{A \cdot D}{B \cdot C}} + 1}\right)$$
  4. **Weighted Polychoric Scoring**: Score each candidate split variable via one-vs-rest association across the remaining feature set.
  5. **Signed Split Criteria**: Split data into positive vs negative evidence regimes.
  6. **Rejection & Filtering Rules**: Attach negative criteria at each node to reject out-of-distribution patterns.
* **Superpowers**:
  - Traversal depth directly measures **cumulative evidence confidence**.
  - Naturally produces cluster associations, rule importance, and anomaly rejection thresholds in a unified graph pass.

---

## 5. Paradigm 4: Manifold & Space-Partitioning Clustering Trees

### 5.1. Random Projection Trees (RP-Trees)
* **Foundations**: Sanjoy Dasgupta & Yoav Freund (*STOC* 2008) — *Random projection trees and low-dimensional manifolds*.
* **The Problem**: Standard k-d trees split along coordinate axes, failing when data lies on a low-dimensional curved manifold embedded in $\mathbb{R}^D$.
* **The RP-Tree Mechanics**:
  - Chooses a random unit direction $\mathbf{v} \sim \mathcal{S}^{D-1}$.
  - Computes projections $z_i = \mathbf{v} \cdot \mathbf{x}_i$.
  - Splits at the median projection or at a dispersion threshold.
  - **Theorem**: RP-Tree depth scales with the **intrinsic manifold dimension $d$**, not ambient dimension $D$ ($O(2^d \log N)$ vs $O(2^D \log N)$).

### 5.2. Density Level-Set Trees
* **Foundations**: Kamalika Chaudhuri, Sanjoy Dasgupta, Samory Kpotufe, & Ulrike von Luxburg (*JMLR* 2014).
* **Mechanics**:
  - Recursively partitions space such that leaves correspond to empirical density estimates.
  - Clusters are defined as the connected components of the **density level-set** $\{x : f(x) \ge \lambda\}$.
  - Provides mathematical guarantees of cluster tree consistency: as $N \to \infty$, the tree converges to the true topological contour tree of the probability density.

### 5.3. BIRCH Clustering Feature Trees (CF-Trees)
* **Foundations**: Tian Zhang, Raghu Ramakrishnan, & Miron Livny (*SIGMOD* 1996).
* **Mechanics**:
  - Balanced tree where each node holds a Clustering Feature triple:
    $$CF = (N, \mathbf{LS}, SS), \quad \mathbf{LS} = \sum \mathbf{x}_i, \quad SS = \sum \|\mathbf{x}_i\|^2$$
  - Ingests streaming points in a single pass: inserts point into closest leaf, updates $CF$ vectors in $O(\log N)$, and splits nodes if cluster radius exceeds threshold $T$.
  - Linear time $O(N)$ clustering of millions of rows.

---

## 6. Synthesis Matrix: Tree-Based Clustering Paradigms

| Paradigm | Input Types | Splitting Mechanism | Interpretability | Outlier Robustness | Best Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unsupervised RF (URF)** | Mixed (Cat + Num) | Real vs Permuted Synthetic noise classifier | Via proximity MDS / TreeSHAP | **Extreme** (tree isolation) | Complex tabular datasets with unknown geometry |
| **Predictive Clustering (PCT)** | Multi-target continuous/discrete | Multi-metric distance variance drop | **Direct logical rules** | Moderate | Multi-objective / multi-horizon time-series clustering |
| **Evidence Tree Clustering** | Binned / Categorical | Polychoric odds ratio evidence score | **Direct log-odds rules** | **High** (rejection criteria) | Forensic data auditing & credit risk segmentation |
| **Random Projection (RP)** | High-dimensional numeric | Random hyperplane projection cuts | Low (linear combinations) | Moderate | High-dimensional manifold clustering |
| **Density Level-Set Tree** | Continuous numeric | Empirical density threshold cuts | Topological tree graph | High | Discovering arbitrary multi-modal shapes |
| **BIRCH CF-Tree** | Continuous numeric | Threshold radius on $(N, \mathbf{LS}, SS)$ | Cluster centroids | Sensitive to radius $T$ | Streaming / out-of-core big data clustering |

---

## 7. Strategic Implications for `mltreelib`

`mltreelib` can implement tree-based clustering with immediate practical advantage:
1. **Unsupervised Random Forest Proximity (`mltreelib::cluster::urf`)**:
   - Trivial to build on top of our existing tree engine: create synthetic permuted column copies, fit binary forest, and output proximity matrix $\mathbf{P}$.
2. **Predictive Clustering Trees (`mltreelib::cluster::pct`)**:
   - Re-uses our existing `node_stats.rs` multivariate trace variance accumulator to partition data into rule-based clusters.
3. **Rust Port of Evidence Tree**:
   - High synergy with our `index.rs` and `schema.rs` categorical encoding infrastructure.
