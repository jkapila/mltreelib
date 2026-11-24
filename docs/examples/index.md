---
title: "Examples & Case Studies Library"
description: "Comprehensive worked tutorials, case studies, and specialized topic guides for mltreelib: regression trees, cubist rules, distributional splits, causal uplift, and bounded solvers."
keywords:
  - decision tree examples
  - model tree tutorials
  - mltreelib examples
  - regression showcase
  - cubist tree rust
---

# Examples & Specialized Case Studies

Worked tutorials, case studies, and focused topic guides demonstrating `mltreelib` on real-world datasets across data ingestion, metrics, regression, causal inference, unsupervised density trees, and numerical optimization.

!!! tip "Looking for the 11 Core Benchmark Suites?"
    The formal mathematical formulations, empirical benchmark tables, and solver evaluations live in the dedicated [**Benchmark Suites**](../suites/index.md) section.

---

## Featured Flagship Case Studies

<div class="grid cards" markdown>

-   :lucide-bar-chart-3: __Comprehensive Regression Showcase (4-Study Empirical Benchmark)__

    ---

    Systematic evaluation of 0D location leaves, 1D linear trees, Huber M-regression, quantile envelopes, MARS, B-spline GAMs, and Group Lasso across 4 datasets.

    [:octicons-arrow-right-24: Open Showcase](regression-methods-showcase.md)

-   :lucide-gem: __Piecewise Linear Model Trees & Cubist Rules__

    ---

    Worked comparison of constant mean leaves vs piecewise linear regression tree leaves and Quinlan M5 rule extractions on Diamond and Concrete benchmarks.

    [:octicons-arrow-right-24: Open Case Study](piecewise-linear-and-cubist.md)

-   :lucide-scale: __Distributional & Censored Trees (Tweedie & Tobit)__

    ---

    Correcting for attenuation bias on censored boundaries (Tobit) and modeling positive zero-inflated claim skewness (Tweedie quasi-deviance).

    [:octicons-arrow-right-24: Open Case Study](distributional-and-censored-trees.md)

-   :lucide-sliders: __Generic Bounded Solvers & Hyperparameter Tuning__

    ---

    14 pure-Rust numerical optimization algorithms (L-BFGS-B, Nelder-Mead, Powell, SHADE, CMA-ES) for standalone scalar minimization and automated tree HPO.

    [:octicons-arrow-right-24: Open Guide](optimization-and-tuning.md)

</div>

---

## Topic Catalog by Domain

### 1. Data Layer & Metrics

| Guide | Focus Area | Primary Benchmark / Target |
| :--- | :--- | :--- |
| **[Data Ingestion, Missing Routing & Encoders](data-matrix-and-routing.md)** | Data Layer & Ingestion | Typed columnar frames, surrogate split routing, and native categorical indexing. |
| **[Metrics, Diagnostics & Probability Calibration](metrics-plots.md)** | Model Diagnostics | ROC, Precision-Recall, Decile Gain, Double Lift, PSI drift, and reliability calibration. |
| **[PCA Rotations & Simple Structure](pca-rotations.md)** | Factor Analysis | Varimax, Quartimax, and Promax orthogonal/oblique simple structure. |

---

### 2. Regression & Continuous Models

| Guide | Focus Area | Primary Benchmark / Target |
| :--- | :--- | :--- |
| **[Comprehensive Regression Showcase](regression-methods-showcase.md)** | Full Regression Taxonomy | 4-study empirical benchmark across OLS, Huber, Quantile, NNLS, Group Lasso, GAM, and GLMs. |
| **[Piecewise Linear Trees & Cubist Rules](piecewise-linear-and-cubist.md)** | Local Linear Leaves | Diamond price prediction ($N=6,000$) and Concrete compressive strength ($N=1,030$). |
| **[Distributional & Censored Trees](distributional-and-censored-trees.md)** | Non-Gaussian Responses | Tweedie quasi-deviance on Miami home sales and Tobit latent models on labor hours. |

---

### 3. Causal & Unsupervised

| Guide | Focus Area | Primary Benchmark / Target |
| :--- | :--- | :--- |
| **[Causal Uplift & Treatment Effects](causal-uplift.md)** | Causal Inference | Treatment-effect recovery, honest subsampling, and Qini curves on simulated trials. |
| **[Unsupervised Density Trees: Clustering & Anomalies](unsupervised-clustering-and-anomalies.md)** | Density Partitioning | CLTree clustering, Extended Isolation Forest (EIF), and GritBot condition mining. |

---

### 4. Architecture, Inspection & Tuning

| Guide | Focus Area | Primary Benchmark / Target |
| :--- | :--- | :--- |
| **[Tree Architecture, Growth Knobs & Constraints](tree-architecture-and-knobs.md)** | Tree Controls | Two-control decoupling, monotone constraints, interaction groups, and honest splits. |
| **[Model Inspection, Quinlan Rules & Stumps](model-inspection-and-rules.md)** | White-Box Auditing | Graphviz DOT exports, flat IF-THEN rulesets, explainable stumps, and leaf k-NN smoothing. |
| **[Generic Solvers & Hyperparameter Tuning](optimization-and-tuning.md)** | Numerical Optimization | 14 bounded solvers, Rosenbrock minimization, and validation loss sweeps. |
