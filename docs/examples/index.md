---
title: "Examples Library Overview"
description: "Worked examples, tutorials, and specialized studies for mltreelib: concrete strength, diamond pricing, miami housing, categorical encoders, and tree inspection."
keywords:
  - decision tree examples
  - model tree tutorials
  - mltreelib examples
---

# Examples & Specialized Studies

Worked tutorials, case studies, and focused topic guides demonstrating `mltreelib` on real-world datasets.

!!! tip "Looking for the 11 Core Benchmark Suites?"
    The formal mathematical formulations, empirical benchmark tables, and solver evaluations live in the dedicated [**Benchmark Suites**](../suites/index.md) section.

---

## Featured Worked Tutorials

<div class="grid cards" markdown>

-   :lucide-gem: __Diamond Price Prediction (Linear Leaves vs Step Functions)__

    ---

    Worked comparison of constant mean leaves vs piecewise linear regression tree leaves on 6,000 diamond sales.

    [:octicons-arrow-right-24: Open Tutorial](piecewise-linear-model-trees-vs-gbdt.md)

-   :lucide-hammer: __Concrete Compressive Strength (Cubist & M5-Style Trees)__

    ---

    Inspect local linear equations, split conditions, and Quinlan rule handoffs on 1,030 industrial concrete mixtures.

    [:octicons-arrow-right-24: Open Tutorial](quinlan-m5-cubist-rust.md)

-   :lucide-home: __Miami Housing Valuation (Tweedie vs Squared Error)__

    ---

    Choosing the right split loss for positive, skewed, and zero-inflated targets on 8,000 Miami home sales.

    [:octicons-arrow-right-24: Open Tutorial](distributional-trees-tweedie-tobit.md)

-   :lucide-sliders: __Generic Bounded Optimizer & HPO Tuning__

    ---

    Optimizing ModelTree hyperparameters directly on tabular validation loss using Nelder-Mead, L-BFGS-B, and Powell.

    [:octicons-arrow-right-24: Open Guide](generic-optimizer.md)

</div>

---

## Specialized Topic Catalog

| Guide | Primary Focus | Target Problem |
| :--- | :--- | :--- |
| **[Generic Optimizer Guide](generic-optimizer.md)** | Bounded Scalar Minimization | 14 algorithms (L-BFGS-B, Nelder-Mead, Powell, SHADE, CMA-ES) on black-box functions. |
| **[PCA Rotations & Simple Structure](pca-rotations.md)** | Factor Rotations | Varimax, Quartimax, and Promax orthogonal/oblique simple structure. |
| **[Data Processing](data-processing.md)** | Frame & Ingest Engine | Typed column frames, NA policies, and zero-copy Rust transforms. |
| **[Optional Categorical Encoders](categorical-encoders.md)** | Categorical Ingestion | Native integer indexing, target encoding, and frequency scaling. |
| **[Missing Data & Routing](missing-routing.md)** | Native NA Handling | Surrogate split mechanics and learned missing-left branching. |
| **[Model Building & Growth Knobs](model-building.md)** | Tree Hyperparameters | `max_depth`, `min_rows`, `min_gain`, and cost-complexity pruning. |
| **[Hyperparameter Search (HPO)](hpo.md)** | Automated Tuning | Grid search, random search, and Latin hypercube validation sweeps. |
| **[Tree Inspection & Export](inspection.md)** | White-Box Auditing | DOT Graphviz exports, Quinlan rulesets, and leaf equations. |
| **[Rules, Summaries & Neighbors](rules-summary-knn.md)** | Instance Explanations | Leaf k-nearest neighbor adjustments and local rule filtering. |
