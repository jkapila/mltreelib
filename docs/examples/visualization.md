# Visualization — tree diagrams and diagnostic plots

The end-to-end picture: a real fit on
[`airfoil_self_noise`](../benchmarks/datasets.md) (1,503 rows, 5 numeric +
1 categorical), the tree drawn left-to-right, and the diagnostic +
explanation charts. Script:
[`examples/visualization.py`](https://github.com/jkapila/mltreelib/blob/main/examples/visualization.py)
(`uv run --with pandas --with pygraphviz --with matplotlib python examples/visualization.py`).
Every image here is click-to-zoom.

## The tree, depth 3 for readability

`max_depth=3, min_rows=30` — constrained for clarity, not score. Rectangles
test a numeric threshold, diamonds are leaves (fitted value + support):

![airfoil regression tree, 8 leaves, depth 3](../assets/airfoil_tree.svg)

*left-to-right (`rankdir=LR`): the split path reads as a sentence.*

## Diagnostics

![airfoil predicted vs actual and residuals](../assets/airfoil_diagnostics.png)

## Explanation

Feature importance (share of total split gain) and the DALE-style breakdown
of one test row — baseline → per-column steps → prediction:

![feature importance](../assets/airfoil_importance.png)
![breakdown of one test row](../assets/airfoil_breakdown.png)

```text
tree diagram: docs/assets/airfoil_tree.svg (8 leaves, depth 3, rendered by graphviz)
diagnostics: docs/assets/airfoil_diagnostics.png
feature importance: docs/assets/airfoil_importance.png
breakdown of one test row: docs/assets/airfoil_breakdown.png
test rmse: 2.3
```

## The chart layer

All fourteen chart functions (`mltreelib.plots`) are catalogued under
[API reference — Visualization](../reference/plots.md); the full metrics +
charts showcase is [Metrics & plots](metrics-plots.md).

**Examples — next: [Metrics & plots](metrics-plots.md)**
