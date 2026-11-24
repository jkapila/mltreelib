!!! info "Consolidated Flagship Guide"
    This topic is now part of the comprehensive flagship guide: [**Distributional & Censored Trees**](distributional-and-censored-trees.md).

# Tobit & Censored Regression Trees

Tobit models a latent response censored at a known boundary. A natural lower bound or a real zero alone does not establish censoring. Latent-normal and censoring assumptions must fit the application.

`ModelTree` provides native **Tobit regression leaves** (`leaf_family="tobit"`) fitted via iterative reweighted least squares (IRLS / EM) inside tree partitions.

Script:
- [`examples/tobit_benchmark.py`](https://github.com/jkapila/mltreelib/blob/main/examples/tobit_benchmark.py) (`python3 examples/tobit_benchmark.py`)

```text
======================================================================
1. TOBIT VS OLS: OVERCOMING ATTENUATION BIAS ON CENSORED DATA
======================================================================
Sample size: N=1500 | Censored at zero: 38.5%
True Latent Model: y* = 0.5 + 2.0 * x + N(0, 1)

Fitted Parameters:
  OLS (Gaussian) : Intercept = +0.8850 | Slope = +1.3257 (Attenuated by 33.7%)
  Tobit (Censored): Intercept = +0.4458 | Slope = +2.0920 (Recovery error: 0.0920)
  [Pass] Tobit recovers the latent slope in this synthetic example.

======================================================================
2. MROZ (1987) FEMALE LABOR SUPPLY: ECONOMETRIC BENCHMARK
======================================================================
Total Women in Sample: 753 | Non-working (hours=0): 325 (43.2%)
Fitted Piecewise Tobit Tree: 2 Leaves

  --- Subgroup / Leaf 0 (n=434) ---
  Intercept:  Tobit=+1507.61  |  OLS= +994.41
    kidslt6    -> Tobit:  -436.19  |  OLS:  -403.61
    age        -> Tobit:   -14.63  |  OLS:   -25.11
    educ       -> Tobit:    -3.37  |  OLS:   +30.25
    exper      -> Tobit:   +42.73  |  OLS:   +47.45

  --- Subgroup / Leaf 1 (n=319) ---
  Intercept:  Tobit=+2061.55  |  OLS=+1854.89
    kidslt6    -> Tobit:  -179.89  |  OLS:  -493.32
    age        -> Tobit:   -16.57  |  OLS:   -35.24
    educ       -> Tobit:   -25.34  |  OLS:   +24.33
    exper      -> Tobit:   +26.50  |  OLS:   +34.02
```

---

## 1. The Tobit Model Formulation

The standard formulation assumes a latent continuous variable $y^*_i$:
$$y^*_i = x_i' \beta + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Observed outcomes are left-censored at threshold $c$ (default $c = 0$):
$$y_i = \begin{cases} y^*_i & \text{if } y^*_i > c \\ c & \text{if } y^*_i \le c \end{cases}$$

The log-likelihood consists of discrete probability mass at $c$ and continuous Gaussian density for uncensored observations:
$$\ln L = \sum_{y_i = c} \ln \left[ 1 - \Phi\left(\frac{x_i' \beta - c}{\sigma}\right) \right] + \sum_{y_i > c} \left[ -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_i - x_i' \beta)^2}{2\sigma^2} \right]$$

---

## 2. ModelTree Tobit Usage

```python
from mltreelib import ModelTree

# Fit a piece-wise linear Tobit tree
tree = ModelTree(
    max_depth=2,
    min_rows=30,
    leaf="linear",
    leaf_family="tobit",
    l2=1e-3,
    tobit_threshold=0.0,
)
tree.fit(X, y)

# Predict latent expected response
predictions = tree.predict(X_test)
```

---

## 3. Econometric Benchmark: Mroz (1987)

On the classic **Mroz female labor supply dataset** (`datasets/external/mroz.csv`):
- 43.2% of women in the survey reported 0 hours worked.
- OLS attenuates the disincentive effect of infant children (`kidslt6`), giving $-403.61$ hours.
- The fitted Tobit tree leaf coefficient is of $-436.19$ hours.
