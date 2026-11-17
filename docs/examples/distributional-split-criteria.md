# Distributional, Hurdle & Survival Split Criteria

This guide demonstrates `mltreelib`'s specialized tree-splitting criteria on real-world benchmark datasets.

While standard decision trees rely on homoscedastic SSE or Gini impurity, real data often exhibits **heavy zero-inflation**, **pure variance breaks**, **censoring floors**, **heterogeneous relative noise**, or **survival censoring**.

---

## 1. Tweedie Quasi-Deviance Splitting (`gain="tweedie"`)

**Best for**: Zero-inflated continuous quantities ($y \ge 0$, e.g., insurance claim amounts, financial defaults, or event intensity).

For power parameter $1 < p < 2$, variance scales with $\mu^p$. The quasi-deviance is:

$$d(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$

$$\text{Gain}_{\text{Tweedie}} = \text{Dev}(P) - [\text{Dev}(L) + \text{Dev}(R)]$$

### Benchmark: Fair's Affairs Dataset

```python
import csv
from mltreelib import ModelTree

with open("datasets/external/affairs.csv", "r") as f:
    rows = list(csv.DictReader(f))

cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
X = [[float(r[c]) for c in cols] for r in rows]
y = [float(r["affairs"]) for r in rows]

# 75% of rows are exactly zero
tree = ModelTree(
    max_depth=3,
    min_rows=25,
    gain="tweedie",
    tweedie_p=1.5,
    prune="none",
)
tree.fit(X, y, col_names=cols)

for s in tree.splits():
    print(f"Split on {s['col']} at {s['threshold']:.2f} (quasi-deviance gain = {s['gain']:.4f})")
```

---

## 2. Heteroscedastic Gaussian NLL Splitting (`gain="heteroscedastic"`)

**Best for**: Detecting regime shifts in volatility, risk, or noise where the conditional mean remains unchanged ($\Delta \mu = 0$).

Standard MSE trees look only for shifts in the mean. The heteroscedastic Gaussian NLL deviance evaluates:

$$\text{Dev}(S) = W_S \ln(\hat{\sigma}_S^2 + \epsilon)$$

$$\text{Gain}_{\text{Hetero}} = W_P \ln \hat{\sigma}_P^2 - \left[ W_L \ln \hat{\sigma}_L^2 + W_R \ln \hat{\sigma}_R^2 \right]$$

```python
from mltreelib import ModelTree

# Left regime has sigma=0.2, right regime has sigma=4.0 (both with E[Y|X] = 0)
tree = ModelTree(
    max_depth=2,
    min_rows=20,
    gain="heteroscedastic",
    prune="none",
)
tree.fit(X, y)
```

---

## 3. Tobit Hurdle Splitting (`gain="tobit"`)

**Best for**: Left-censored continuous responses (e.g., hours worked censored at 0, purchase amount with non-buyers at 0).

Combines binary classification deviance ($y_i > a$ vs $y_i \le a$) with Gaussian deviance on the positive slice ($y_i > a$):

$$\text{Gain}_{\text{Tobit}} = \text{Gain}_{\text{binomial}}(I(y > a)) + \text{Gain}_{\text{Gaussian}}(y \mid y > a)$$

### Benchmark: Mroz Female Labor Supply

```python
tree = ModelTree(
    max_depth=3,
    min_rows=30,
    gain="tobit",
    tobit_threshold=0.0,
    prune="none",
)
tree.fit(X_mroz, y_hours, col_names=cols)
```

---

## 4. Coefficient of Variation Reduction (`gain="cv"`)

**Best for**: Scale-heterogeneous continuous data where relative variation $\text{CV} = \sigma / \mu$ drives homogeneity:

$$\text{CV}(S) = \frac{\hat{\sigma}_S}{|\hat{\mu}_S| + \epsilon}$$

$$\text{Gain}_{\text{CV}} = \text{CV}(P) - \left[ \frac{W_L}{W_P} \text{CV}(L) + \frac{W_R}{W_P} \text{CV}(R) \right]$$

---

## 5. Martingale Survival Residual Splitting (`gain="martingale"`)

**Best for**: Right-censored time-to-event outcomes with precomputed cumulative baseline hazard $\hat{\Lambda}_0(T_i)$.

Evaluates variance reduction on martingale residuals $M_i = E_i - \hat{\Lambda}_0(T_i)$:

$$\text{Gain}_{\text{Martingale}} = \text{SSE}_M(P) - [\text{SSE}_M(L) + \text{SSE}_M(R)]$$

### Benchmark: Veteran Lung Cancer Trial

```python
tree = ModelTree(
    max_depth=2,
    min_rows=20,
    gain="martingale",
    prune="none",
)
tree.fit(X_covariates, martingale_residuals, col_names=cols)
```
