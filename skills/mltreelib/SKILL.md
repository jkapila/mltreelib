---
name: mltreelib
description: Use when fitting, explaining, evaluating, or saving mltreelib models for regression, classification, survival or competing risks, causal uplift, clustering, anomaly detection, or forecasting.
---

# mltreelib

1. Identify the target, prediction horizon, feature availability, and evaluation unit. Pick the matching [recipe](references/recipes.md).
2. Fit preprocessing on training data. Keep input column order and types consistent. Supply categorical values as labels and represent missing values explicitly where the chosen engine supports them.
3. For a `ModelTree`, configure splits, stopping, and pruning with `GrowthControl`. Configure the leaf form, family, and optimizer with `LeafControl`. Use `solver` inside a leaf control. The leaf form and family determine compatible solvers.
4. Start with the default solver. Consider L-BFGS for smooth likelihoods or coordinate descent for compatible sparse fits. Compare derivative-free solvers with a seed and the same evaluation budget. Measure convergence and held-out error before changing the default.
5. Compare against a baseline on held-out data. Inspect splits and leaf support before interpreting rules or reading coefficients. Choose a plot for the decision under review. [Evaluation guide](https://jkapila.github.io/mltreelib/guides/evaluation/) · [Visualization guide](https://jkapila.github.io/mltreelib/guides/visualization/)
6. When handing off a fitted `ModelTree`, save its JSON and record feature schema, prediction kind, and package version. Reload the artifact and compare predictions on representative rows. [Serving guide](https://jkapila.github.io/mltreelib/guides/serving-contract/)

| Task | Target and first check |
|---|---|
| Regression | Numeric outcome. Compare MAE or RMSE with a constant baseline. |
| Classification | Class labels. Check held-out probabilities and tune decision thresholds on validation data. |
| Survival | Follow-up time and event indicator or cause. Check calibration and censoring-adjusted error by horizon. |
| Uplift | Outcome and randomized treatment. Compare policy value with treat-all and treat-none. |
| Clustering | No target. Check cluster support and assignment stability across independent refits. |
| Anomaly detection | Rank review candidates by score. Report precision at the available review budget when labels exist. |
| Forecasting | Entity, timestamp, and future outcome. Split by time and compare against a seasonal naive baseline. |

Use the [parameter reference](https://jkapila.github.io/mltreelib/reference/parameters/) for supported names and the [solver benchmark](https://jkapila.github.io/mltreelib/examples/solver-benchmarks/) for measured tradeoffs.
