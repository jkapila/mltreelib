# Honest trees &amp; sample weights

`ModelTree` supports sample-splitting honesty to avoid in-sample optimism bias, as well as observation weights for cost-sensitive learning and survey sampling.

Script: [`examples/honesty_weights.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/honesty_weights.py>) (`uv run python examples/honesty_weights.py`).

## Output

```
== 1. Honest tree estimation (honesty=True) ==
  standard tree: 3 leaves, total leaf rows: 300
  honest tree  : 3 leaves, estimation leaf rows: 120 (structure used 40%)
    leaf 0: standard mean=0.004 (n=60) | honest mean=0.386 (n=50)
    leaf 1: standard mean=0.602 (n=60) | honest mean=1.474 (n=49)
    leaf 2: standard mean=1.797 (n=180) | honest mean=2.402 (n=21)

== 2. Weighted tree fitting (fit(x, y, w=...)) ==
  x0=0 target: eight rows at y=2.0, two rows at y=20.0
  unweighted prediction (equal weight): 5.600
  weighted prediction (10x rare weight) : 14.857
```

## How to use

### Honest estimation (`honesty=True`)

In standard greedy decision trees, the same training observations select the splits and estimate leaf parameters (constants, linear slopes, and empirical quantiles). On noisy data, this double use introduces optimism bias and over-confident leaf distributions.

Honest estimation (adapted from Athey, Tibshirani, and Wager's `grf`) splits the dataset: - **Structure sample** (`honesty_fraction`, e.g. `0.4`): evaluates split candidates, impurity reductions, and tree topology. - **Estimation sample** (`1.0 - honesty_fraction`, e.g. `0.6`): flows down the fixed topology to fit leaf parameters and empirical t-digests.

```
from mltreelib import ModelTree

model = ModelTree(
    max_depth=3,
    honesty=True,
    honesty_fraction=0.5,  # 50% structure, 50% estimation
)
model.fit(X, y)
```

The resulting leaf predictions and conditional quantiles are unbiased out-of-sample estimates for the leaf's sub-population.

### Sample weights (`w=weights`)

Pass a 1D sequence of non-negative weights to `ModelTree.fit()`:

```
from mltreelib import ModelTree

# Assign higher weight to rare or critical observations:
weights = [10.0 if is_critical(i) else 1.0 for i in range(len(X))]

model = ModelTree(max_depth=3, leaf="mean")
model.fit(X, y, w=weights)
```

`w` weights observations across: - Impurity score and gain calculations during split search. - Leaf means and weighted least-squares linear fits. - Weighted IRLS for binomial (`leaf_family="binomial"`) and Poisson (`leaf_family="poisson"`) models.
