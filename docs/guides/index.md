# Practical modeling pathways

Start with the decision you need to make. Each path uses a small reproducible example and covers data, fitting, evaluation and interpretation.

| Decision | Path | Output |
|---|---|---|
| Predict a quantity | [Regression](regression.md) | A numeric estimate |
| Predict an event | [Classification](classification.md) | A class score and decision |
| Describe groups | [Clustering](clustering.md) | Segment assignments and rules |
| Prioritize unusual records | [Anomaly detection](anomaly.md) | A review ranking |
| Estimate time to an event | [Survival](survival.md) | Survival probabilities over time |
| Choose an intervention | [Uplift](uplift.md) | Estimated treatment effects |

## Before and after modeling

Start with [data preparation](data.md). Follow the [real-data walkthrough](real-data.md) for a TabArena dataset. After fitting, use [evaluation and monitoring](evaluation.md), then [visualize the results](visualization.md). For a model handoff, see [prediction and schema](serving-contract.md). For Tobit leaves, constraints, retraining and tuning, see [extend a model tree](advanced-modeling.md).

## A repeatable sequence

**Decision → Data → Baseline → Fit → Evaluate → Interpret → Monitor**

Keep test data independent of model and threshold selection. Fit transformations on training data only. Record feature names, category handling and missingness. A prediction or segment does not establish what happens after an intervention.

These introductory paths use synthetic data. Use the linked examples and API references for more detail. They do not establish production performance or full drop-in compatibility.
