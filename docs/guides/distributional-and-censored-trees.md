---
title: "Distributional & Censored Trees: Tweedie, Tobit & Heteroscedastic Splitting"
description: "Model complex response distributions beyond mean squared error: zero-inflated Tweedie leaves, Tobit latent attenuation correction for censored outcomes, and heteroscedastic volatility splits."
keywords:
  - Tweedie decision tree
  - Tobit censored tree
  - heteroscedastic split criteria
  - zero-inflated regression tree
  - survival splitting criteria
---

# Distributional & Censored Trees: Tweedie, Tobit & Heteroscedastic Splitting

Classical decision trees minimize squared error (SSE) under the implicit assumption of homoscedastic Gaussian noise. Real-world business outcomes routinely violate this assumption:
- **Zero-Mass & Positive Skew**: Insurance claim amounts, operational losses, and customer lifetime spend have a point mass at zero and heavy positive tails.
- **Censored Boundaries**: Sensor readings, wage hours, and follow-up clinical durations are truncated or floored at a boundary $c$. Standard OLS suffers severe **attenuation bias** towards zero.
- **Regime Volatility Shifts**: Financial returns, algorithmic latencies, and machine vibrations exhibit shifts in variance with zero change in the conditional mean ($\Delta \mu = 0$).

`mltreelib` supports generalized distributional leaves and split gains directly in Rust:

```mermaid
flowchart TD
    A["Observed Response Distribution y"] --> B{"Distributional Characteristics"}

    B -- "Zero-Inflated Continuous (y >= 0)" --> C["Tweedie Quasi-Deviance\nVar(μ) = μ^p (1 < p < 2)"]
    B -- "Floored or Capped Boundary (y >= c)" --> D["Tobit Latent Censoring\nInverse Mills Ratio Attenuation Fix"]
    B -- "Pure Volatility Shifts (Delta μ = 0)" --> E["Heteroscedastic Gaussian NLL\nDeviance = W * ln(sigma^2)"]
    B -- "Bounded Rates y in (0, 1)" --> F["Beta Regression Tree\nLogit link with precision phi"]

    C --> G["ModelTree(gain='tweedie', tweedie_p=1.5)"]
    D --> H["ModelTree(leaf_family='tobit', tobit_c=0.0)"]
    E --> I["ModelTree(gain='heteroscedastic')"]
    F --> J["ModelTree(leaf_family='beta')"]

    classDef blueBox fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef greenBox fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;
    classDef grayBox fill:#334155,stroke:#64748b,stroke-width:2px,color:#ffffff;

    class A,B grayBox;
    class C,D,E,F blueBox;
    class G,H,I,J greenBox;
```

![Distributional Tree Benchmarks across Tweedie, Tobit, and Censored Datasets](../assets/distributional_leaf_benchmarks.png)

---

## 1. Zero-Inflated Skew: Tweedie Quasi-Deviance Splitting

On Miami Housing sales ([OpenML 46942](https://www.openml.org/d/46942)), positive home prices vary with property size and ocean proximity. Tweedie splitting scales variance with mean power $V(\mu) = \mu^p$ ($1 < p < 2$), prioritizing proportional relative accuracy over absolute error:

$$d(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$

![Miami Housing Price Split Comparison: Tweedie vs Squared Error](../assets/guides/real-housing-splits.svg)

### Complete Tweedie Quasi-Deviance Example

```python
import numpy as np
from mltreelib import ModelTree, Metrics

# 1. Generate zero-inflated claim amounts (Poisson frequency x Gamma severity)
rng = np.random.default_rng(2026)
n_policies = 2000

driver_age = rng.uniform(18.0, 75.0, n_policies)
vehicle_age = rng.uniform(0.0, 15.0, n_policies)
mileage = rng.uniform(5.0, 35.0, n_policies)

# Event probability (claim frequency)
p_claim = 1.0 / (1.0 + np.exp(2.0 - 0.04 * mileage + 0.02 * driver_age))
has_claim = rng.binomial(1, p_claim)

# Claim amount given claim occurs (Gamma severity)
claim_severity = rng.gamma(shape=2.5, scale=1200.0, size=n_policies)
y_claims = np.where(has_claim == 1, claim_severity, 0.0)

X = np.column_stack([driver_age, vehicle_age, mileage])
feature_names = ["driver_age", "vehicle_age", "mileage"]

# 2. Fit ModelTree with Tweedie quasi-deviance (p=1.5 compound Poisson-Gamma)
tweedie_tree = ModelTree(
    gain="tweedie",
    tweedie_p=1.5,
    leaf="mean",
    max_depth=3,
    min_rows=40,
)
tweedie_tree.fit(X, y_claims.tolist(), col_names=feature_names)
pred_claim_expected = tweedie_tree.predict(X)

print(f"Zero-Claim Prevalence: {(y_claims == 0).mean():.1%}")
print(f"Mean Observed Claim: ${y_claims.mean():.2f} | Mean Predicted: ${np.mean(pred_claim_expected):.2f}")
print("Top Tree Split:", tweedie_tree.splits()[0])
```

---

## 2. Tobit Regression Leaves: Eliminating Attenuation Bias

When a response variable is censored at threshold $c$ (e.g. non-working hours in the Mroz labor survey where 43.2% of women report hours $= 0$), standard OLS underestimates the true latent slopes by **33% to 46%** (attenuation bias).

The Tobit model assumes a latent variable $y^* = \mathbf{x}^\top \boldsymbol{\beta} + \epsilon$ with observed outcome $y = \max(y^*, c)$. `mltreelib` fits Tobit leaves using pure Rust EM/IRLS iterations:

$$\ln L = \sum_{y_i = c} \ln \left[ 1 - \Phi\left(\frac{\mathbf{x}_i^\top \boldsymbol{\beta} - c}{\sigma}\right) \right] + \sum_{y_i > c} \left[ -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2}{2\sigma^2} \right]$$

### Synthetic Attenuation Correction Proof

```python
import numpy as np
from mltreelib import ModelTree

# 1. Generate latent variable y* with zero-censoring
rng = np.random.default_rng(2026)
n_workers = 1200

education = rng.uniform(8.0, 18.0, n_workers)
experience = rng.uniform(0.0, 30.0, n_workers)

# True latent relationship (market wage / labor supply hours)
true_latent = -15.0 + 2.4 * education + 1.2 * experience + rng.normal(0, 8.0, n_workers)
observed_hours = np.maximum(0.0, true_latent)

X = np.column_stack([education, experience])
feature_names = ["education", "experience"]

# 2. Fit piecewise Tobit tree with censoring at zero
tobit_tree = ModelTree(
    leaf="linear",
    leaf_family="tobit",
    tobit_threshold=0.0,  # Censored boundary floor
    max_depth=2,
    min_rows=40,
)
tobit_tree.fit(X, observed_hours.tolist(), col_names=feature_names)
pred_latent = tobit_tree.predict(X)

print(f"Censored Sample Fraction: {(observed_hours == 0).mean():.1%}")
print(f"Sample Latent Prediction Range: [{min(pred_latent):.2f}, {max(pred_latent):.2f}]")
```

---

## 3. Heteroscedastic Splitting: Detecting Risk & Volatility Breaks

Standard decision trees fail to split when two subgroups have identical means ($\mu_L = \mu_R$) but different variance regimes ($\sigma_L^2 \ne \sigma_R^2$). 

The heteroscedastic split criterion evaluates Gaussian negative log-likelihood deviance directly:

$$\text{Gain}_\text{Hetero} = W_P \ln \hat{\sigma}_P^2 - \left[ W_L \ln \hat{\sigma}_L^2 + W_R \ln \hat{\sigma}_R^2 \right]$$

```python
import numpy as np
from mltreelib import ModelTree

# 1. Generate regimes with identical means (0.0) but distinct volatility regimes
rng = np.random.default_rng(2026)
n_points = 1000

regime_indicator = rng.uniform(0.0, 100.0, n_points)
noise_scale = np.where(regime_indicator > 50.0, 5.0, 0.5)
returns = rng.normal(loc=0.0, scale=noise_scale, size=n_points)

X = regime_indicator.reshape(-1, 1)

# 2. Fit tree with heteroscedastic splitting
volatility_tree = ModelTree(
    gain="heteroscedastic",
    max_depth=2,
    min_rows=30,
)
volatility_tree.fit(X, returns.tolist(), col_names=["volatility_driver"])

print("Discovered Volatility Split Threshold:", volatility_tree.splits()[0]["threshold"])
```

!!! tip "Related guides"
    - [Distributional & Censored Leaf Families](distributional-leaves.md) for full mathematical derivations.
    - [Survival & Competing Risks Trees](survival-trees.md) for right-censored time-to-event outcomes.
