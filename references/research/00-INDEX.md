# Master Research Index & Architecture Taxonomy

**Updated**: 2026-11-01  
**Curators**: Jitin Kapila & Antigravity  
**Repository**: `mltreelib` (`crates/mltreelib-core`, `crates/mltreelib-py`, `python/mltreelib`)  
**Mission**: Advance tree and ensemble methods beyond flat tabular heuristics by unlocking **categorical, grouped, hierarchical, longitudinal, and structural signal**.

---

## 1. Thematic Research Map (24 Foundation Documents)

```
                                  MLTREELIB RESEARCH REPOSITORY
                                                │
   ┌────────────────────┬───────────────────────┼───────────────────────┬────────────────────┐
   ▼                    ▼                       ▼                       ▼                    ▼
[Pillar 1: Core      [Pillar 2: Split       [Pillar 3: Longitudinal, [Pillar 4: Sequence, [Pillar 5: Cross-
 Engine & Mechanics]  Families & Distr.]     Panel & Time Series]     Tries & Retrieval]   Paradigm & Unsupervised]
 • Tabular vs FM      • Split Survey         • Wide Panel Trees       • Context Trees      • Cross-Paradigm AI Map
 • Linear Scan Opt    • Distributional/Surv. • VAR / VECM Feasibility • Suffix Tries/TDM   • TDA, Copulas & Anomalies
 • Earth/MARS Port    • NGBoost Deep Dive    • Tensor / Bilinear MAR  • Dynamic Decay Rec  • Tree-Based Clustering
 • C5.0/Cubist Parity • Preselect Catalog    • Jitin Methods Map      • TrieKNN Structure  • GritBot Source Analysis
```

### Pillar 1: Core Tree Mechanics, Replications & Performance
1. [tree-vs-foundation-tabular.md](file:///mnt/d/Programs/mltreelib/references/research/tree-vs-foundation-tabular.md): Core thesis: Tabular Foundation Models (TFMs) vs modern tree ensembles on hierarchical/grouped data.
2. [jkapila-methods-map.md](file:///mnt/d/Programs/mltreelib/references/research/jkapila-methods-map.md): Architectural roadmap connecting DAR, TrieKNN, CrossTab Sparsity, pygemodels, and tree builders.
3. [dataset-component-design.md](file:///mnt/d/Programs/mltreelib/references/research/dataset-component-design.md): Synthetic benchmark generator with adjustable categorical, grouped, and hierarchical signal.
4. [ours-vs-c5-cubist.md](file:///mnt/d/Programs/mltreelib/references/research/ours-vs-c5-cubist.md): Structural comparison against RuleQuest C5.0 and Cubist classics.
5. [mars-port-plan.md](file:///mnt/d/Programs/mltreelib/references/research/mars-port-plan.md): Porting Multivariate Adaptive Regression Splines (Earth) into Rust.
6. [2026-10-31-incremental-linear-scan-speedup.md](file:///mnt/d/Programs/mltreelib/references/research/2026-10-31-incremental-linear-scan-speedup.md): Numerical streaming formulation achieving $O(N)$ split scans.
7. [2026-10-31-simplification-audit.md](file:///mnt/d/Programs/mltreelib/references/research/2026-10-31-simplification-audit.md): Codebase health and structural pruning analysis.
8. [2026-11-01-top5-and-earth-architectural-analysis.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-top5-and-earth-architectural-analysis.md): Architectural benchmark of Earth, EBM, C5.0, Cubist, and GritBot.
9. [knob-gap-review.md](file:///mnt/d/Programs/mltreelib/references/research/knob-gap-review.md): Parameter mapping and feature parity across C5.0, Cubist, and XGBoost/LightGBM.
10. [2026-11-01-gold-standard-hierarchical-macro-micro-routing.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-gold-standard-hierarchical-macro-micro-routing.md): **Gold Standard Architecture**: Two-tier macro-tree segmentation with decoupled micro-leaf retrain and primary-key entity routing.

### Pillar 2: Distributional, Survival & Specialized Split Families
10. [split-criteria-survey.md](file:///mnt/d/Programs/mltreelib/references/research/split-criteria-survey.md): Comprehensive survey of tree splitting criteria (Gini, Entropy, MSE, Poisson, Gamma).
11. [2026-11-01-distributional-and-survival-split-criteria.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-distributional-and-survival-split-criteria.md): Mathematical derivations for Survival (Log-Rank & Martingale), Multiclass Softmax, Tweedie, and NGBoost Heteroscedastic leaves.
12. [ngboost-deep-dive.md](file:///mnt/d/Programs/mltreelib/references/research/ngboost-deep-dive.md): Probabilistic prediction via natural gradient tree boosting over multi-parameter distributions.
13. [categorical-encoding-research.md](file:///mnt/d/Programs/mltreelib/references/research/categorical-encoding-research.md): Cardinality thresholds, target encoding, and partition algorithms for categorical features.
14. [preselection-and-expert-thesis.md](file:///mnt/d/Programs/mltreelib/references/research/preselection-and-expert-thesis.md): Pre-selection and mixture-of-experts routing in trees.
15. [preselect-preprocessing-catalog.md](file:///mnt/d/Programs/mltreelib/references/research/preselect-preprocessing-catalog.md): Catalog of feature filter heuristics and ranking metrics.

### Pillar 3: Longitudinal, Panel, and Multi-Horizon Time Series
16. [2026-11-01-longitudinal-wide-format-trees-and-time-series.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-longitudinal-wide-format-trees-and-time-series.md): Wide vs Long format trade-offs in retail forecasting (M5), multi-output vector splits, and zero-copy Rust buffers.
17. [2026-11-01-var-vecm-panel-trees-feasibility.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-var-vecm-panel-trees-feasibility.md): Vector Autoregressive (VAR), Vector Error Correction (VECM), and Panel Fixed Effects / Arellano-Bond GMM trees with effort estimation.
18. [2026-11-01-niche-tensor-matrix-functional-panel-trees.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-niche-tensor-matrix-functional-panel-trees.md): Multi-way Tensor Decision Trees (TT), Bilinear Matrix Autoregression (MAR), Dynamic Factor Trees, and Rough Path Signatures.

### Pillar 4: Sequence Suffix Trees, Tries & Deep Retrieval
19. [2026-11-01-context-trees-and-suffix-trie-siblings.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-context-trees-and-suffix-trie-siblings.md): Comprehensive survey of Context Tree Weighting (CTW), BCT, PCT, Suffix Tries, and connection to TrieKNN.
20. [tree-decay-sequential-rec-survey.md](file:///mnt/d/Programs/mltreelib/references/research/tree-decay-sequential-rec-survey.md): Dynamic temporal decay, Alibaba TDM / TDR deep hierarchical retrieval, and cognitive forgetting curves.
21. [model-tree-survey.md](file:///mnt/d/Programs/mltreelib/references/research/model-tree-survey.md): Survey of M5/M5P, Cubist, and linear model trees.

### Pillar 5: Cross-Paradigm Frontiers (RL, CI, Logic, Copulas, TDA & Clustering)
22. [2026-11-01-advanced-tree-frontiers-tda-copulas-anomalies.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-advanced-tree-frontiers-tda-copulas-anomalies.md): Contextual anomaly detection (**GritBot**), Density Estimation Trees (**DET**), Vine Copulas (R-Vines), TDA Persistent Homology, and Tree-Wasserstein optimal transport.
23. [2026-11-01-cross-paradigm-ai-trees-survey.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-cross-paradigm-ai-trees-survey.md): Trees across Reinforcement Learning (FQI, RLT, VIPER), Causal Inference (Honest IV-Forests), Symbolic Logic (TILDE), Operations Research (MIP Learn-to-Branch), and Probabilistic Circuits (Generative Forests GeFs).
24. [2026-11-01-tree-based-clustering-frontiers.md](file:///mnt/d/Programs/mltreelib/references/research/2026-11-01-tree-based-clustering-frontiers.md): Unsupervised Random Forest Proximities, Predictive Clustering Trees (PCT), Jitin's Evidence Tree Clustering (Polychoric Odds), and Random Projection Trees.

---

## 2. Top 10 High-Impact Implementations for `mltreelib`

These 10 concrete implementations operationalize >85% of the theoretical advances surveyed:

```
                               TOP 10 IMPLEMENTATION ROADMAP
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                 CORE SUPERVISED ENGINES                                │
 │  [1. Model Tree (Cubist/AR)]  [2. Direct Multi-Output MIMO]  [3. Distributional/Tweedie]│
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │                               LONGITUDINAL & PANEL SUITE                               │
 │  [4. Panel Fixed Effects Tree]      [5. Bilinear Matrix AR & Path Signatures]          │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │                           UNSUPERVISED & DATA AUDITING SUITE                           │
 │  [6. GritBot Contextual Auditor]    [7. Unsupervised RF & Predictive Clustering (PCT)] │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │                         PROBABILISTIC, CAUSAL & STRUCTURED TRIES                       │
 │  [8. Generative Circuits (GeFs)]   [9. Honest Causal Forests]   [10. TrieKNN Index]    │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Implementation 1: Parametric Model Tree Engine (Cubist + Local AR/Trend Leaves)
* **What**: Extends `crates/mltreelib-core/src/model_tree.rs` to fit Ridge-regularized parametric models inside leaves.
* **Why**: Solves the fundamental failure of flat trees on $I(1)$ non-stationary time series; extrapolates trends smoothly.
* **Math**: Leaf solves $\hat{\boldsymbol{\beta}}_L = (\mathbf{X}_L^T \mathbf{X}_L + \lambda \mathbf{I})^{-1} \mathbf{X}_L^T \mathbf{y}_L$ using existing `faer::llt` Cholesky.

---

### Implementation 2: Direct Multi-Output / MIMO Tree Engine (Vector Leaves)
* **What**: Multi-target tree predicting an $H$-dimensional trajectory vector $\hat{\mathbf{y}} \in \mathbb{R}^H$ at each leaf.
* **Why**: Native multi-horizon forecasting (e.g. 28-day M5) without compounding recursive error or 28 separate models.
* **Math**: Split criterion minimizes the trace of covariance matrix: $\Delta I = \sum_{h=1}^H \text{VarReduction}(y_{\cdot, h})$ via SIMD dot products.

---

### Implementation 3: Panel Fixed-Effects Tree Engine (`PanelTree`)
* **What**: Ingests wide contiguous buffers ($N \times T$) and fits tree models with within-group demeaning.
* **Why**: Eliminates the 60-million-row memory explosion of melted long formats; controls for unobserved entity heterogeneity $\alpha_i$ without incidental parameter overfitting.
* **Math**: $\tilde{y}_{it} = y_{it} - \bar{y}_i$, $\tilde{\mathbf{x}}_{it} = \mathbf{x}_{it} - \bar{\mathbf{x}}_i$. Recovers entity intercept as $\hat{\alpha}_i = \bar{y}_i - \bar{\mathbf{x}}_i^T \hat{\boldsymbol{\beta}}$.

---

### Implementation 4: GritBot Contextual Anomaly & Data Quality Auditor
* **What**: Pure Rust replication of Quinlan's GritBot engine (`references/sources/extracted/GritBot`).
* **Why**: Standard Isolation Forests only find global outliers; GritBot finds contextual errors (`attribute anomalous given condition subgroup`), outputting human-readable forensic rules.
* **Math**: Builds rule subsets; standardizes continuous values: $\frac{|x_{ij} - \mu_{j,\text{sub}}|}{\sigma_{j,\text{sub}}} > \kappa$.

---

### Implementation 5: Unsupervised Random Forest Proximity & Predictive Clustering (PCT)
* **What**: Unsupervised clustering via the Synthetic Null trick (Breiman) and Predictive Clustering Trees (Blockeel).
* **Why**: Solves K-Means' lack of interpretability and failure on mixed types. Every cluster is defined by an exact logical rule with a scale-invariant proximity matrix.
* **Math**: Permute features independently to generate synthetic Class 0; cluster on dissimilarity $\mathbf{D} = \sqrt{1 - \mathbf{P}}$.

---

### Implementation 6: Generative Random Forests as Probabilistic Circuits (GeFs)
* **What**: Maps tree leaves to univariate density estimators, converting decision trees into tractable Sum-Product Networks.
* **Why**: Grants decision trees exact analytical marginalization $P(X_1 \mid X_5)$ in linear time and native missing-value inference without imputation.

---

### Implementation 7: Distributional & Survival Split Engine (Tweedie + Martingale)
* **What**: Expands `criteria.rs` beyond MSE/Gini to Tweedie ($1 < p < 2$) for zero-inflated demand, and Martingale residuals for right-censored survival curves.
* **Why**: Captures intermittent zero-spike retail sales and time-to-event attrition in linear $O(N)$ streaming time.

---

### Implementation 8: Rough Path Signatures Engine (`timefeat::signatures`)
* **What**: Computes truncated iterated path integrals ($S_M(X)$) of continuous multi-channel streams in pure Rust (~250 LOC).
* **Why**: Exposes non-linear cross-channel interaction momentum (e.g. price elasticity $\int y_t d\text{price}_t$) directly to tree split scans with complete invariance to sampling frequency.

---

### Implementation 9: Honest Causal Forests (GRF / CATE Estimation)
* **What**: Generalized Random Forests with sample-splitting honesty (Athey & Wager).
* **Why**: Unbiased non-parametric estimation of Heterogeneous Treatment Effects ($\hat{\tau}(x)$) with valid asymptotic Gaussian confidence intervals.

---

### Implementation 10: Structured Categorical Suffix & Prefix Tries (TrieKNN / CTW)
* **What**: Ingests categorical hierarchies and temporal symbol histories as prefix/suffix tries with leaf-local regression.
* **Why**: Realizes Jitin's TrieKNN and Willems' Context Tree Weighting, unlocking deep hierarchical signal without one-hot dimensionality blowups.

---

## 3. Engineering Sizing & Feasibility Overview

| # | Implementation Name | Target Module | Est. Pure Rust LOC | Timeline | Crate Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Parametric Model Tree (Cubist/AR)** | `model_tree.rs`, `leaf.rs` | ~450 LOC | 2 weeks | `faer` (already present) |
| **2** | **Direct Multi-Output MIMO Tree** | `split.rs`, `node_stats.rs` | ~350 LOC | 1–2 weeks | Pure Rust / SIMD |
| **3** | **Panel Fixed-Effects Tree** | `econometrics/panel.rs` | ~500 LOC | 2 weeks | `faer` (already present) |
| **4** | **GritBot Contextual Auditor** | `audit/gritbot.rs` | ~400 LOC | 2 weeks | Pure Rust |
| **5** | **Unsupervised RF & PCT Clustering** | `cluster/mod.rs` | ~450 LOC | 1–2 weeks | Pure Rust |
| **6** | **Generative Forests (Circuits)** | `prob/circuit.rs` | ~500 LOC | 2 weeks | Pure Rust |
| **7** | **Distributional Tweedie/Survival** | `criteria.rs`, `family.rs` | ~300 LOC | 1 week | `statrs` (already present) |
| **8** | **Path Signatures Engine** | `timefeat.rs` | ~250 LOC | 1 week | Pure Rust |
| **9** | **Honest Causal Forests** | `causal/forest.rs` | ~600 LOC | 2–3 weeks | Pure Rust |
| **10** | **TrieKNN / Suffix Context Trie** | `trie/mod.rs` | ~350 LOC | 1–2 weeks | Pure Rust |
