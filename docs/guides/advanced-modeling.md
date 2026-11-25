# Extend a model tree

Start with a [regression or classification pathway](index.md). Then choose the change that matches the modeling problem. These options are different decisions, not a checklist to enable together.

| Need | Technique | Read next |
|---|---|---|
| Outcome is censored at a known bound | Tobit leaves | [Tobit example](distributional-leaves.md) |
| Domain limits on direction or combinations | Monotone or interaction constraints | [Constraint example](tree-architecture-and-knobs.md) |
| Preserve segments while updating leaf models | Leaf retraining | [Retraining examples](hierarchical-retrain.md) |
| Control fit complexity and flexibility | Growth and leaf settings | [Growth knob tour](tree-architecture-and-knobs.md) |
| Different outcome distributions | Distributional families | [Distributional leaves](distributional-leaves.md) |
| Predict only from approved input columns | Feature and interaction controls | [Parameters](../reference/parameters.md) |

## Censored outcomes: Tobit

Tobit is a model for a latent response observed only above a censoring boundary. Structural zeros, truncation and missing outcomes are different problems. Its interpretation depends on the assumed latent model and censoring mechanism.

```python
from random import Random
from mltreelib import ModelTree

rng = Random(12)
x = [[rng.uniform(-2, 2)] for _ in range(160)]
y = [max(0.0, 0.5 + 2 * row[0] + rng.gauss(0, 1)) for row in x]
model = ModelTree(max_depth=0, min_rows=10, leaf="linear",
                  leaf_family="tobit", tobit_threshold=0.0)
model.fit(x[:120], y[:120], col_names=["signal"])
print("Latent predictions:", model.predict(x[120:])[:3])
```

The response above is the fitted latent expectation, so negative values are possible even though observations are censored at zero. Evaluate the desired observed outcome separately. Correct censoring assumptions do not guarantee an unbiased result on every dataset. See [Tobin (1958)](https://doi.org/10.2307/1907382).

## Constrain shape and interactions

```python
from mltreelib import ModelTree

x = [[float(i)] for i in range(40)]
y = [float(i // 10) for i in range(40)]
model = ModelTree(max_depth=2, min_rows=5, leaf="mean", monotone=[1])
model.fit(x, y, col_names=["volume"])
print("Predictions:", model.predict([[0.0], [10.0], [20.0], [30.0]]))
```

A positive constraint encodes a domain requirement. Check predictions on meaningful grids and held-out data. A check on the healthcare example found two decreases with linear leaves when age increased. The same check found no decreases with mean leaves. This is a tracked behavior gap; a finite grid does not prove monotonicity elsewhere. Do not assume local coefficient signs guarantee global monotonicity. Interaction constraints restrict paths and should also be checked with `split_paths()`.

## Retrain leaves with fixed partitions

```python
from mltreelib import LeafControl, ModelTree

x = [[float(i)] for i in range(40)]
y = [float(i // 10) for i in range(40)]
model = ModelTree(max_depth=1, min_rows=5, leaf="mean")
model.fit(x, y, col_names=["volume"])
before = model.splits()
model.retrain(x, [v + 1 for v in y], control=LeafControl(leaf="mean"))
print("Splits preserved:", before == model.splits())
print("Updated prediction:", model.predict([[5.0]]))
```

This example mutates the model. Use appropriate independent data for evaluation and check that each leaf has sufficient new observations. Fixed partitions can become stale when the population changes. See the [hierarchical retraining guide](hierarchical-retrain.md) for alternative feature sets and routing.

## Tune deliberately

Start with `max_depth` and `min_rows` to control partition size. Compare mean and linear leaves only when their assumptions and interpretation fit the outcome. Tune on a validation set and keep a final test set untouched. The [knob tour](tree-architecture-and-knobs.md) is a parameter reference, not a recommended sequence or a substitute for evaluation.
