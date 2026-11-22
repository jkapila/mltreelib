---
title: "Technical Articles & Case Studies"
description: "In-depth case studies on decision trees: credit risk auditing, predictive maintenance, causal uplift modeling, modernizing Quinlan classics, distributional trees, and two-stage survival analysis."
keywords:
  - decision tree case studies
  - glass box credit scoring
  - zero imputation predictive maintenance
  - causal trees uplift modeling
  - model trees vs GBDT
  - distributional decision trees
  - two stage survival trees
---

# Technical Articles & Case Studies

In-depth investigations into tree architecture, regulatory compliance, industrial IoT, econometrics, and survival analysis. Each study is grounded in real benchmark data with reproducible code, empirical holdout metrics, and publication figures.

---

## Featured Case Studies

<div class="grid cards" markdown>

-   :lucide-shield-alert: __Decision Trees for Credit Scoring__

    ---

    Regulatory compliance, path-level auditing, and probability calibration under fair lending constraints.

    **Dataset**: Give Me Some Credit (150,000 borrowers)

    [:octicons-arrow-right-24: Read Case Study](why-credit-scoring-needs-glass-box-trees.md)

-   :lucide-truck: __Zero-Imputation Predictive Maintenance__

    ---

    Industrial sensor feeds: native missing-data routing and surrogate splits under severe asymmetric misclassification costs.

    **Dataset**: Scania Truck APS Failure (76,000 vehicles)

    [:octicons-arrow-right-24: Read Case Study](zero-imputation-predictive-maintenance.md)

-   :lucide-heart-handshake: __Uplift Modeling with Causal Trees__

    ---

    Commercial decision optimization: honest sample-splitting ($S^{\text{tr}}$ vs $S^{\text{est}}$) to identify persuadable customers without sleeping-dog harm.

    **Dataset**: Hillstrom Randomized Trial (64,000 customers)

    [:octicons-arrow-right-24: Read Case Study](uplift-modeling-with-causal-trees.md)

-   :lucide-cpu: __Modernizing Quinlan Classics in Rust__

    ---

    Re-architecting M5, Cubist, and C5.0 into high-throughput systems ML: linear leaves, dynamic rule extraction, and exact parity.

    **Dataset**: Concrete Compressive Strength (1,030 mixtures)

    [:octicons-arrow-right-24: Read Case Study](modernizing-quinlan-classics-in-rust.md)

-   :lucide-chart-scatter: __Beyond Mean Squared Error: Distributional Trees__

    ---

    Actuarial & risk modeling: Tweedie compound Poisson-Gamma claims, Tobit censored MLE, and heteroscedastic regimes.

    **Dataset**: Miami Housing & German Credit (TabArena)

    [:octicons-arrow-right-24: Read Case Study](beyond-mse-distributional-trees.md)

-   :lucide-trending-up: __Breaking Step Functions: Model Trees vs. GBDT__

    ---

    Systems ML & continuous tabular regression: why piecewise linear leaves eliminate staircase artifacts with far shallower trees than GBDT.

    **Dataset**: Diamonds (53,940 stones)

    [:octicons-arrow-right-24: Read Case Study](breaking-step-functions-model-trees-vs-gbdt.md)

-   :lucide-users: __Segregate First, Model Later: Two-Stage Survival Trees__

    ---

    Actuarial & retention analytics: decoupling ID-level tree cohort discovery from leaf-level Kaplan-Meier curves to eliminate right-censoring bias and non-proportional hazard violations.

    **Dataset**: Bank Customer Churn (10,000 customers)

    [:octicons-arrow-right-24: Read Case Study](segregate-first-two-stage-survival-trees.md)

-   :lucide-package-check: __Prescriptive Analytics: Predict-then-Optimize Inventory Replenishment__

    ---

    Supply chain optimization: predict-then-optimize replenishment with ModelTree per-leaf quantile curves and dual search under warehouse capacity and working capital caps.

    **Application**: 50-SKU Retail Supply Chain under Joint Capacity Constraints

    [:octicons-arrow-right-24: Read Case Study](prescriptive-inventory-replenishment.md)

</div>

---

## Quick Problem Navigator

| Business & Modeling Question | Case Study | Benchmark Dataset |
| :--- | :--- | :--- |
| How do I audit a classifier and choose an operating threshold? | [Decision trees for credit scoring](why-credit-scoring-needs-glass-box-trees.md) | Give Me Some Credit |
| Can a tree score industrial telemetry with missing sensor values without imputation? | [Decision trees and missing sensor data](zero-imputation-predictive-maintenance.md) | Scania APS Failure |
| Does an uplift model improve targeting without waking sleeping dogs? | [Causal trees for uplift modeling](uplift-modeling-with-causal-trees.md) | Hillstrom randomized trial |
| How do Quinlan's M5 and Cubist rule sets translate to high-speed Rust? | [Cubist-style model trees](modernizing-quinlan-classics-in-rust.md) | UCI concrete strength |
| What changes when I use Tweedie and Tobit distributional split criteria? | [Distributional tree splits](beyond-mse-distributional-trees.md) | Miami home sales |
| When do piecewise linear leaves outperform gradient boosted trees? | [Regression trees with linear leaves](breaking-step-functions-model-trees-vs-gbdt.md) | Diamonds |
| How do I model longitudinal customer churn without right-censoring bias? | [Two-stage hierarchical survival trees](segregate-first-two-stage-survival-trees.md) | Bank Customer Churn |
| How do I prescribe inventory replenishment under warehouse and budget limits? | [Prescriptive inventory replenishment](prescriptive-inventory-replenishment.md) | 50-SKU Retail Replenishment |

---

## Next Steps

- For procedural recipes and execution workflows, see the **[Practical Modeling Pathways](../guides/index.md)**.
- For focused operation scripts, browse the **[Example Library](../examples/index.md)**.
- For complete parameter definitions and methods, inspect the **[API Reference](../reference/model.md)**.
