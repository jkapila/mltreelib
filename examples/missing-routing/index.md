# Missing data routing &amp; surrogate splits

In `mltreelib`, missing data is a first-class citizen. Missing entries (`None` in Python, `NaN` normalized at ingest) never raise an error and do not require prior imputation.

`ModelTree` resolves missing data through three mechanisms: 1. **Learned missing directions (`missing_left`)**: during split search, the tree evaluates both sending missing values left vs. right, selecting whichever branch minimizes error. 2. **Surrogate splits (`maxsurrogate=N`)**: when a split feature is missing at prediction time, the highest-agreement surrogate split directs the sample. 3. **NA indicator coefficients in linear leaves**: linear leaves estimate dedicated indicator coefficients for missing predictors so within-leaf regressions remain valid.

Script: [`examples/missing_routing.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/missing_routing.py>) (`uv run python examples/missing_routing.py`).

## Output

```
== 1. Learned missing direction (missing_left) ==
  split details: [{'col': 'x0', 'threshold': 4.0, 'missing_left': False, 'gain': 5000.0}]
  pred(x=None): 10.000 (matches pred(x=8.0): 10.000, pred(x=1.0): 0.000)

== 2. Surrogate splits (maxsurrogate=1) ==
  primary split : x0 (threshold=0.497)
  maxsurrogate=0 (fallback only): pred(None, 0.1)=0.223, pred(None, 0.9)=0.223
  maxsurrogate=1 (routed by x1) : pred(None, 0.1)=0.223, pred(None, 0.9)=5.000

== 3. Linear leaves with NA indicator coefficients ==
  leaf 0: cols=['x0', 'x1'] intercept=0.002 na_coef=[0.0, 4.997]
  predict([5.0, 2.0]) : 16.001
  predict([5.0, None]): 14.998
```

## How it works

### 1\. Learned split directions (`missing_left`)

For every candidate numeric or categorical split, the partition search evaluates: - Send observed values matching the predicate to the left branch. - Send observed values failing the predicate to the right branch. - Score placing `None` values on the left branch vs. placing `None` values on the right branch.

The configuration minimizing SSE or deviance is stored as `missing_left: bool`. When an unseen record with `None` arrives, it routes along this learned direction.

### 2\. Surrogate splits (`maxsurrogate=N`)

When `maxsurrogate >= 1`, the tree identifies correlated secondary features whose splits best mimic the primary split partition:

```
from mltreelib import ModelTree

# Compute up to 2 surrogate splits per internal node:
model = ModelTree(max_depth=3, maxsurrogate=2)
model.fit(X, y)
```

At prediction time, if the primary split feature is missing, the highest-ranking surrogate is checked. Only if all surrogate features are also missing does the tree fall back to `missing_left`.

### 3\. NA-indicator leaf regression

When `leaf="linear"`, the leaf fits a coefficient for each observed feature plus an indicator term $\\beta\_{\\text{NA}, j}$ for whether feature $j$ was missing:

$$\\hat{y} = \\beta\_0 + \\sum\_{j \\in \\text{obs}} \\beta\_j x\_j + \\sum\_{j \\in \\text{missing}} \\beta\_{\\text{NA}, j}$$

This allows leaves to make continuous linear adjustments when predictors are partially missing without collapsing to a simple global mean.
