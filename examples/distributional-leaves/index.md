# Distributional leaves — GLM &amp; Censored Families

Real-world targets often violate Gaussian error assumptions. `ModelTree` provides native Rust implementations of six specialized distributional families: `leaf_family="binomial"`, `"probit"`, `"poisson"`, `"gamma"`, `"tweedie"`, and `"tobit"`. Pair with `gain="deviance"` so split searches score exact deviance reduction under the respective family.

Script: [`examples/leaf_family.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/leaf_family.py>) (`uv run python examples/leaf_family.py`).

```
binomial P(1): [0.25, 0.75]
binomial P(Y <= 0): [0.75, 0.25]
poisson means: [1.0, 2.0, 4.0]
poisson P(Y <= 0): [0.367879, 0.018316]
probit P(1): [0.25, 0.75]
gamma means: [1.0, 2.0, 4.0]
tweedie means: [0.333827, 4.331555]
tobit latent means: [-2.765143, 1.19807]
```

---

## 1\. Binomial &amp; Probit Probability Estimation

For binary classification targets $y \\in {0, 1}$:

- **`leaf_family="binomial"`**: Standard logistic link $\\mu = \\frac{1}{1 + e^{-\\eta}}$.
- **`leaf_family="probit"`**: Gaussian CDF link $\\mu = \\Phi(\\eta)$ (ideal for threshold crossing and latent utility models).

```
from mltreelib import ModelTree

# Probit tree with standard normal CDF leaves
m_probit = ModelTree(max_depth=1, min_rows=2, leaf_family="probit", gain="deviance")
m_probit.fit([[0.0]] * 4 + [[1.0]] * 4, [0, 0, 0, 1, 0, 1, 1, 1])

# predict_proba returns P(Y = 1) on response scale
probs = m_probit.predict_proba([[0.0], [1.0]])
labels = m_probit.predict_label([[0.0], [1.0]], threshold=0.5)
```

---

## 2\. Poisson Rate &amp; Count Modeling

For nonnegative counts or rates ($y \\ge 0$) with log link $\\mu = e^\\eta$:

```
mp = ModelTree(max_depth=2, min_rows=2, leaf_family="poisson", gain="deviance")
mp.fit([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], [1, 1, 2, 2, 4, 4])

# Returns expected event counts
expected_counts = mp.predict([[0.0], [1.0], [2.0]])
prob_zero = mp.cdf([[0.0], [2.0]], 0)
```

---

## 3\. Gamma Severity &amp; Duration Modeling

For strictly positive continuous outcomes ($y \> 0$) with constant coefficient of variation (e.g. claim severity, service duration, customer lifetime value):

```
mg = ModelTree(max_depth=2, min_rows=5, leaf_family="gamma", gain="deviance")
mg.fit(x_positive, y_severity)

preds = mg.predict(x_test)  # Strictly positive predictions
```

---

## 4\. Tweedie Compound Poisson-Gamma Modeling

For zero-inflated continuous positive targets ($y \\ge 0$) where a mass of exact zeros exists alongside continuous positive values (e.g. insurance pure premiums, warranty loss costs):

```
mt = ModelTree(max_depth=2, min_rows=10, leaf_family="tweedie", gain="deviance", l2=1e-3)
mt.fit(x_claims, y_loss_cost)

preds = mt.predict(x_test)
```

---

## 5\. Tobit Left-Censored Regression

For continuous targets censored at a lower threshold (default $0.0$, e.g. warranty claims, consumer expenditures):

```
tobit_tree = ModelTree(max_depth=2, min_rows=10, leaf_family="tobit")
tobit_tree.fit(x_censored, y_censored)

latent_means = tobit_tree.predict(x_test)
```

- Fits latent Gaussian expectations via Expectation-Maximization in iteratively reweighted least squares (IRLS).
- Eliminates downward attenuation bias caused by standard linear regression on censored zeros.
