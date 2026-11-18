# Classification: a practical pathway

Estimate the chance of a binary event, then choose an action threshold. A score and a decision are separate outputs.

**Path:** Define the decision → Prepare data → Fit a baseline → Evaluate → Inspect → Act and monitor.

## 1\. Define the data and split

The example uses a numeric risk signal and a 0/1 target. It fits 2,000 synthetic observations and evaluates 1,000 held-out observations. The binomial model returns class-1 scores.

## 2\. Fit and inspect a small example

This complete example uses the package and Python standard library. It uses synthetic data and a fixed seed. Run it from a Python environment with `mltreelib` installed.

```
from random import Random
from math import exp
from mltreelib import Metrics, ModelTree

rng = Random(42)
x = [[rng.uniform(-3, 3)] for _ in range(3000)]
y = [float(rng.random() < 1 / (1 + exp(-r[0]))) for r in x]
model = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=25)
model.fit(x[:2000], y[:2000], col_names=["risk_signal"])
scores = model.predict_proba(x[2000:])
labels = [int(p >= .5) for p in scores]
report = Metrics().classification([int(v) for v in y[2000:]], labels)
print("classification held-out confusion:", report["confusion"])
```

## 3\. See the result

The figure comes from [`examples/modeling_pathways.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/modeling_pathways.py>). Rendering the figures requires matplotlib. The example is a teaching illustration, not a benchmark.

## 4\. Evaluate before deciding

The confusion matrix uses 0.5 as an illustration. Select the threshold on validation data using false-positive costs, missed events, and capacity. Check calibration, class balance, precision and recall on a separate test set.

## 5\. Interpret and monitor

Review errors by subgroup and inspect the named tree. A prediction is not proof of the cause of an outcome. Check whether the estimated probabilities are calibrated before using them as risks.

## Real-data example

The [bank churn walkthrough](<https://jkapila.github.io/mltreelib/guides/real-data/index.md>) runs this workflow on a 10,000-row TabArena dataset with independent test rows.

## Next steps and references

[Distributional leaves](<https://jkapila.github.io/mltreelib/examples/distributional-leaves/index.md>) · [Metrics and plots](<https://jkapila.github.io/mltreelib/examples/metrics-plots/index.md>) · [Calibration](<https://jkapila.github.io/mltreelib/reference/plots/index.md>)

Continue with [data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>), [evaluation and monitoring](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), or [model extensions](<https://jkapila.github.io/mltreelib/guides/advanced-modeling/index.md>). The [practical modeling pathways](<https://jkapila.github.io/mltreelib/guides/index.md>) show the other tasks. For broader coverage, see the [documentation audit](<https://github.com/jkapila/mltreelib/tree/main/references/research>).
