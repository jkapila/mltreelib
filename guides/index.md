# Practical modeling pathways

Start with the decision you need to make. Each path uses a small reproducible example and covers data, fitting, evaluation and interpretation.

| Decision | Path | Output |
| --- | --- | --- |
| Predict a quantity | [Regression](<https://jkapila.github.io/mltreelib/guides/regression/index.md>) | A numeric estimate |
| Predict an event | [Classification](<https://jkapila.github.io/mltreelib/guides/classification/index.md>) | A class score and decision |
| Describe groups | [Clustering](<https://jkapila.github.io/mltreelib/guides/clustering/index.md>) | Segment assignments and rules |
| Prioritize unusual records | [Anomaly detection](<https://jkapila.github.io/mltreelib/guides/anomaly/index.md>) | A review ranking |
| Estimate time to an event | [Survival](<https://jkapila.github.io/mltreelib/guides/survival/index.md>) | Survival probabilities over time |
| Choose an intervention | [Uplift](<https://jkapila.github.io/mltreelib/guides/uplift/index.md>) | Estimated treatment effects |

## Before and after modeling

Start with [data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>). Follow the [real-data walkthrough](<https://jkapila.github.io/mltreelib/guides/real-data/index.md>) for a TabArena dataset. After fitting, use [evaluation and monitoring](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), then [visualize the results](<https://jkapila.github.io/mltreelib/guides/visualization/index.md>). For a model handoff, see [prediction and schema](<https://jkapila.github.io/mltreelib/guides/serving-contract/index.md>). For Tobit leaves, constraints, retraining and tuning, see [extend a model tree](<https://jkapila.github.io/mltreelib/guides/advanced-modeling/index.md>).

## A repeatable sequence

**Decision → Data → Baseline → Fit → Evaluate → Interpret → Monitor**

Keep test data independent of model and threshold selection. Fit transformations on training data only. Record feature names, category handling and missingness. A prediction or segment does not establish what happens after an intervention.

These introductory paths use synthetic data. Use the linked examples and API references for more detail. They do not establish production performance or full drop-in compatibility.
