# Distributional Leaves — GLM, Censored, Survival & Volatility Families

Real-world targets often violate Gaussian error assumptions. `ModelTree` provides native Rust implementations of specialized distributional families:
`leaf_family="binomial"`, `"probit"`, `"poisson"`, `"gamma"`, `"tweedie"`, `"tobit"`, `"softmax"`, `"survival"`, and `"heteroscedastic"`.

Scripts:
- [`examples/leaf_family.py`](https://github.com/jkapila/mltreelib/blob/main/examples/leaf_family.py)
- [`examples/stump_distributional_leaves.py`](https://github.com/jkapila/mltreelib/blob/main/examples/stump_distributional_leaves.py) (`uv run python examples/stump_distributional_leaves.py`)

```text
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

## 1. Binomial & Probit Probability Estimation

For binary classification targets $y \in \{0, 1\}$:

- **`leaf_family="binomial"`**: Standard logistic link $\mu = \frac{1}{1 + e^{-\eta}}$.
- **`leaf_family="probit"`**: Gaussian CDF link $\mu = \Phi(\eta)$ (ideal for threshold crossing and latent utility models).

```python
from mltreelib import ModelTree

# Probit tree with standard normal CDF leaves
m_probit = ModelTree(max_depth=1, min_rows=2, leaf_family="probit", gain="deviance")
m_probit.fit([[0.0]] * 4 + [[1.0]] * 4, [0, 0, 0, 1, 0, 1, 1, 1])

# predict_proba returns P(Y = 1) on response scale
probs = m_probit.predict_proba([[0.0], [1.0]])
labels = m_probit.predict_label([[0.0], [1.0]], threshold=0.5)
```

---

## 2. Poisson Rate & Count Modeling

For nonnegative counts or rates ($y \ge 0$) with log link $\mu = e^\eta$:

```python
mp = ModelTree(max_depth=2, min_rows=2, leaf_family="poisson", gain="deviance")
mp.fit([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], [1, 1, 2, 2, 4, 4])

# Returns expected event counts
expected_counts = mp.predict([[0.0], [1.0], [2.0]])
prob_zero = mp.cdf([[0.0], [2.0]], 0)
```

---

## 3. Gamma Severity & Duration Modeling

For strictly positive continuous outcomes ($y > 0$) with constant coefficient of variation (e.g. claim severity, service duration, customer lifetime value):

```python
mg = ModelTree(max_depth=2, min_rows=5, leaf_family="gamma", gain="deviance")
mg.fit(x_positive, y_severity)

preds = mg.predict(x_test)  # Strictly positive predictions
```

---

## 4. Tweedie Compound Poisson-Gamma Modeling

For zero-inflated continuous positive targets ($y \ge 0$) where a mass of exact zeros exists alongside continuous positive values (e.g. insurance pure premiums, warranty loss costs):

```python
mt = ModelTree(max_depth=2, min_rows=10, leaf_family="tweedie", gain="deviance", l2=1e-3)
mt.fit(x_claims, y_loss_cost)

preds = mt.predict(x_test)
```

---

## 5. Tobit Left-Censored Regression

For continuous targets censored at a lower threshold (default $0.0$, e.g. warranty claims, consumer expenditures):

```python
tobit_tree = ModelTree(max_depth=2, min_rows=10, leaf_family="tobit")
tobit_tree.fit(x_censored, y_censored)

latent_means = tobit_tree.predict(x_test)
```

- Fits latent Gaussian expectations via Expectation-Maximization in iteratively reweighted least squares (IRLS).
- Eliminates downward attenuation bias caused by standard linear regression on censored zeros.

---

## 6. Softmax Multi-Class Probability Vectors

For discrete targets with $C \ge 3$ mutually exclusive classes:

```python
m_softmax = ModelTree(max_depth=0, leaf_family="softmax", min_rows=10)
m_softmax.fit(x_train, y_classes)

# Query class probabilities or predicted class labels
probs = m_softmax.predict(x_test, kind="proba")      # [[P(C_0), P(C_1), P(C_2)], ...]
labels = m_softmax.predict(x_test, kind="response")   # [0, 1, 2, ...]
```

---

## 7. Survival Analysis (Nelson-Aalen & Kaplan-Meier)

For right-censored time-to-event outcomes $(T, E)$ where $T$ is elapsed duration and $E \in \{0, 1\}$ is event occurrence (1 = event, 0 = censored):

```python
m_surv = ModelTree(max_depth=0, leaf_family="survival", min_rows=10)
m_surv.fit(x_patients, times, event=events)

# Inspect empirical survival curves and median survival
leaf = m_surv.leaves()[0]
print(f"Cohort Median Survival: {leaf['median_time']} days")

# Predict survival probability S(t) and cumulative hazard H(t) at milestone horizons
surv_curve = m_surv.predict(x_test, kind="survival", times=[30.0, 90.0, 180.0, 365.0])
cum_hazard = m_surv.predict(x_test, kind="hazard", times=[30.0, 90.0, 180.0, 365.0])
```

---

## 8. Heteroscedastic Location & Scale Modeling

For continuous targets exhibiting conditional volatility or non-constant variance $\sigma_i^2 = \exp(z_i^\top \gamma)$:

```python
m_het = ModelTree(max_depth=0, leaf_family="heteroscedastic", min_rows=10)
m_het.fit(x_train, y_train)

# Jointly predict location mu and scale sigma
dists = m_het.predict(x_test, kind="dist")  # [(mu, sigma), ...]
for mu, sigma in dists:
    ci_low = mu - 1.96 * sigma
    ci_high = mu + 1.96 * sigma
```

- Uses **Harvey (1976) bias correction** (+1.2703628 to log-variance intercept) to guarantee unbiased standard deviation estimation.
- Enables conditional volatility bands and predictive intervals for risk reserves and financial volatility.

---

## 9. 0D Leaf Formulations & Stats Dictionaries

For non-parametric or summary leaves, `ModelTree` computes a comprehensive 0D distributional summary at every terminal node:

- **0D Leaf Formulations**: `leaf="prob"`, `leaf="odds"`, `leaf="log_odds"`, `leaf="median"`, `leaf="huber"`, `leaf="quantile"`, `leaf="mean"`.
- **Leaf Stats Dictionary**:
  Every leaf dictionary contains a nested `stats` dict and direct fields:
  `prob`, `odds`, `log_odds`, `majority_class`, `entropy`, `gini`, `var`, `skew`, `kurtosis`, `median`, `mad`, `iqr`, `zero_fraction`, `dispersion`.

```python
from mltreelib import ModelTree, LeafControl

# Binary classification with 0D probability leaves:
m = ModelTree(leaf="prob", max_depth=3, min_rows=10)
m.fit(X, y)

# 1. Inspect all 0D stats across leaves:
for st in m.leaf_stats():
    print(f"Prob: {st['prob']:.3f}, Odds: {st['odds']:.3f}, Gini: {st['gini']:.3f}")

# 2. Predict specific 0D metrics directly on test data:
p_prob = m.predict(X_test, kind="prob")
p_odds = m.predict(X_test, kind="odds")
p_lo   = m.predict(X_test, kind="log_odds")
p_med  = m.predict(X_test, kind="median")

# 3. Predict full stats dictionary per row:
per_row_stats = m.predict(X_test, kind="stats")
```
