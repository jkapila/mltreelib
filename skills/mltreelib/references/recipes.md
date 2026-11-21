# Modeling recipes

The supervised snippets use row-major `x_train` and `x_test`. Align the target and optional row weights with the training rows. Supply `col_names` and `dtypes` to `fit` for named or mixed-type features. Preserve the same feature order when predicting.

## Regression

```python
from mltreelib import GrowthControl, LeafControl, Metrics, ModelTree

tree = ModelTree(
    growth_control=GrowthControl(max_depth=3, min_rows=20),
    leaf_control=LeafControl(leaf="linear", l2=0.05),
)
tree.fit(x_train, y_train)
predictions = tree.predict(x_test)
print(Metrics().regression(y_test, predictions)["mae"])
print(tree.splits())
```

Compare with the training median when using MAE. Check residuals across the prediction range and inspection groups. Use `tree.leaves()` to check local support.

## Binary classification

```python
from mltreelib import GrowthControl, LeafControl, ModelTree

tree = ModelTree(
    growth_control=GrowthControl(gain="deviance", max_depth=3, min_rows=30),
    leaf_control=LeafControl(leaf="mean", family="binomial"),
)
tree.fit(x_train, y_train)
probabilities = tree.predict(x_test, kind="proba")
```

Use 0/1 training labels. If specifying the class count, set `n_classes` in `GrowthControl`. Compare probabilities against training prevalence. Check calibration and rank quality. Use validation data to choose a decision threshold. Read the [classification pathway](https://jkapila.github.io/mltreelib/guides/classification/).

## Survival and competing risks

```python
from mltreelib import GrowthControl, LeafControl, ModelTree

tree = ModelTree(
    growth_control=GrowthControl(max_depth=2, min_rows=30, prune="none"),
    leaf_control=LeafControl(family="survival"),
)
tree.fit(x_train, (time_train, event_train))
curves = tree.predict(x_test, kind="survival", times=[30.0, 90.0, 180.0])
```

Encode right censoring as 0 and an observed event as 1. Express prediction horizons in the same units as follow-up. For multiple causes, follow the [competing risks example](https://jkapila.github.io/mltreelib/examples/survival-trees/) and inspect cumulative incidence by cause.

## Causal uplift

```python
from mltreelib import GrowthControl, LeafControl, ModelTree

tree = ModelTree(
    growth_control=GrowthControl(
        gain="causal", honesty=True, prune="none", min_rows=50, seed=42
    ),
    leaf_control=LeafControl(leaf="mean"),
)
tree.fit(x_train, (outcome_train, treatment_train))
uplift = tree.predict(x_test, kind="uplift")
```

Use only features available before treatment. Check treated and control support in each group. Measure selected-policy value in a randomized holdout. [Uplift pathway](https://jkapila.github.io/mltreelib/guides/uplift/)

## Clustering

```python
from mltreelib.cluster import ClusterTree

tree = ClusterTree(kind="pct", max_depth=2, min_rows=20)
tree.fit(x_train)
labels = tree.predict(x_test, kind="cluster")
print(tree.n_clusters_, tree.rules())
```

Give numeric features comparable scales for variance or distance objectives. Compare assignments on a common review set after independent refits. [Clustering pathway](https://jkapila.github.io/mltreelib/guides/clustering/)

## Anomaly detection

```python
from mltreelib.anomaly import AnomalyTree

tree = AnomalyTree(kind="eif", contamination=0.05, seed=42)
tree.fit(x_train)
scores = tree.predict(x_test, kind="score")
ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
```

Prefix the ranking at the review budget. Report precision at that budget when labels exist. Use `kind="binary"` for labels from the fitted threshold. [Anomaly pathway](https://jkapila.github.io/mltreelib/guides/anomaly/)

## Forecasting

```python
from mltreelib import GrowthControl, LeafControl, ModelTree, lag_by_days

lag7 = lag_by_days(entity_ids, dates, actual, 7)
x = [[value] for value in lag7]
tree = ModelTree(
    growth_control=GrowthControl(max_depth=2, min_rows=20),
    leaf_control=LeafControl(leaf="linear"),
)
tree.fit(x[:cut], actual[:cut])
forecasts = tree.predict(x[cut:])
```

Pass string entity IDs and ISO date strings. Sort by time and use a cut at a date boundary. The seven-day lag in this example assumes each lagged observation exists at scoring time. For a fixed-origin forecast, build every feature from data available at that origin. Use a rolling-origin or blocked holdout. [Forecasting example](https://jkapila.github.io/mltreelib/examples/timeseries-forecasting/)

## Save and inspect

```python
from mltreelib import ModelTree

tree.save("model.json")
restored = ModelTree.load("model.json")
assert restored.predict(x_test) == tree.predict(x_test)
for rule in restored.extract_rules().rules:
    print(rule.conditions, rule.prediction)
```

Use this pattern with a fitted `ModelTree`. Compare the actual prediction kind used in deployment. The JSON artifact holds model state. The extracted rules provide a separate readable view of the tree.
