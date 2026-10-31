# Distributional leaves — binomial & Poisson

Binary targets and count data require non-Gaussian distributions:
`leaf_family="binomial"` (logit link) for binary classification probabilities and
`leaf_family="poisson"` (log link) for rates and counts. Pair with `gain="deviance"`
so split searches score deviance reduction under the family.

Script: [`examples/leaf_family.py`](https://github.com/jkapila/mltreelib/blob/main/examples/leaf_family.py) (`uv run python examples/leaf_family.py`).

```text
binomial P(1): [0.25, 0.75]
binomial P(Y <= 0): [0.75, 0.25]
poisson means: [1.0, 2.0, 4.0]
poisson P(Y <= 0): [0.367879, 0.018316]
categorical + NA P(1): [0.0, 1.0, 1.0]
```

## Binomial Probability Estimation

```python
from mltreelib import ModelTree

# Binomial tree with logit leaves and deviance split scoring
mb = ModelTree(max_depth=1, min_rows=2, leaf_family="binomial", gain="deviance")
mb.fit([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]],
       [0, 0, 0, 1, 0, 1, 1, 1])

# predict_proba returns P(Y = 1)
probs = mb.predict_proba([[0.0], [1.0]])
# cdf returns fitted binomial CDF P(Y <= value)
cdf_0 = mb.cdf([[0.0], [1.0]], 0)
```

- Predictions stay on the response scale ($P(Y=1) \in [0, 1]$).
- Linear coefficients inside leaves are fitted via weighted IRLS on the link scale (logit).
- `cdf(x, value)` returns the fitted parametric distribution's cumulative probability, distinct from the empirical target quantiles (`quantile()`).

## Poisson Rate & Count Modeling

```python
# Poisson tree for nonnegative count data with log link
mp = ModelTree(max_depth=2, min_rows=2, leaf_family="poisson", gain="deviance")
mp.fit([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]],
       [1, 1, 2, 2, 4, 4])

# predict returns response-scale expected counts
expected_counts = mp.predict([[0.0], [1.0], [2.0]])
# cdf computes Poisson P(Y <= value) from fitted rate lambda
prob_zero = mp.cdf([[0.0], [2.0]], 0)
```

- Target values must be finite, nonnegative counts or rates.
- Missing values and categorical features route through the same native handlers as Gaussian trees.
