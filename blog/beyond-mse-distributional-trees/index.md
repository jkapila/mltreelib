# Beyond Mean Squared Error: Distributional Decision Trees for Heavy Tails, Insurance, and Censored Data

**Author:** Jitin Kapila &amp; mltreelib Team  
**Category:** Actuarial Science, Econometrics &amp; Risk Modeling

---

## The Flaw at the Heart of Standard Tree Splitting

Almost every machine learning practitioner defaults to the same split criterion: **Mean Squared Error (MSE)** for regression or **Gini Impurity / Log-Loss** for classification. In algorithms like CART, XGBoost, and LightGBM, the regression splitting rule seeks partitions that maximize variance reduction:

\\\[\\Delta \\text{Var} = \\text{Var}(Y\_{\\text{parent}}) - \\frac{N\_L}{N}\\text{Var}(Y\_L) - \\frac{N\_R}{N}\\text{Var}(Y\_R)\\\]

MSE targets the conditional mean and can be used without Gaussian errors. Its squared-error objective weights large residuals heavily. A Gaussian likelihood interpretation adds assumptions about the error distribution and variance. Censoring requires separate treatment.

Different target structures can require different modeling choices:

- **Zero-Inflated Insurance Claims**: In auto, property, and health insurance, Many claim datasets contain a mass at zero. Positive claims are strictly positive, highly skewed, and heavy-tailed. Standard MSE splits are dominated by the sheer volume of zeros, diluting the signal of severe catastrophe losses.
- **Top-Coded / Censored Metrics (Tobit)**: Customer lifetime value surveys, credit limits, warranty caps, or sensor cutoffs can introduce censoring, which differs from truncation. Running standard OLS splits treats censored values as literal outcomes, which can bias estimates under a censored latent-outcome model.
- **Heteroscedastic Market Volatility**: In financial risk management, the expected return \\(\\mu(X)\\) may remain near zero while the conditional variance \\(\\sigma^2(X)\\) explodes during market regime shifts. Standard MSE splits are blind to pure variance shifts when the local mean does not move.
- **Time-to-Event / Survival Analysis**: Customer churn, loan default timing, and mechanical equipment life are subject to right-censoring. Splitting on raw duration without adjusting for risk exposure distorts survival hazard rates.

---

## Distributional Tree Splitting in `mltreelib`

Starting in `mltreelib` v0.4.2, we introduced native Rust splitting criteria designed specifically for non-Gaussian, censored, and heteroscedastic data distributions with zero external dependencies:

```
from mltreelib import ModelTree

# Auto Insurance Claims Modeling with Tweedie Deviance (p = 1.5)
# Features: [DriverAge, VehiclePower, BonusMalus, AnnualMileage]
col_names = ["DriverAge", "VehiclePower", "BonusMalus", "AnnualMileage"]

model = ModelTree(
    gain="tweedie",
    tweedie_p=1.5,      # 1 < p < 2: Compound Poisson-Gamma
    max_depth=3,
    min_rows=40,
    prune="none",
    seed=42,
)
model.fit(X_train, y_train, col_names=col_names)

# Predict a mean claim cost; assess calibration separately
predicted_cost = model.predict(X_test)
```

---

## The Mathematical Mechanics of Specialized Criteria

### 1\. Tweedie Compound Poisson-Gamma Deviance (`gain="tweedie"`)

For insurance claim frequencies and aggregate losses, targets follow a Tweedie distribution with power parameter \\(p \\in (1, 2)\\), where \\(\\text{Var}(Y) = \\phi \\mu^p\\). The split search maximizes Tweedie deviance reduction:

\\\[d(y, \\hat{\\mu}) = 2 \\left( \\frac{y^{2-p}}{(1-p)(2-p)} - \\frac{y \\hat{\\mu}^{1-p}}{1-p} + \\frac{\\hat{\\mu}^{2-p}}{2-p} \\right)\\\]

This penalizes deviation from the Poisson-frequency and Gamma-severity likelihood, capturing both whether a claim occurs and how severe it becomes.

### 2\. Tobit Censored Log-Likelihood (`gain="tobit"`)

For data bounded below at \\(c\\) (e.g., \\(y = \\max(y^\*, 0)\\)), observations with \\(y\_i = c\\) represent latent outcomes \\(y\_i^\* \\le c\\). The Tobit split criterion evaluates the log-likelihood of the censored normal model:

\\\[\\ell(\\mu, \\sigma) = \\sum\_{y\_i \> c} \\left( -\\frac{1}{2}\\log(2\\pi\\sigma^2) - \\frac{(y\_i - \\mu)^2}{2\\sigma^2} \\right) + \\sum\_{y\_i = c} \\log \\Phi\\left(\\frac{c - \\mu}{\\sigma}\\right)\\\]

This models censoring under its assumed latent-normal specification; it does not guarantee bias removal when assumptions fail.

### 3\. Heteroscedastic Gaussian Deviance (`gain="heteroscedastic"`)

When features modulate both risk level and uncertainty, partitions evaluate reductions in the full Gaussian deviance:

\\\[D(Y\_m) = N\_m \\log(\\hat{\\sigma}\_m^2) + N\_m\\\]

Partitions are rewarded for grouping points of uniform volatility together, even when leaf means \\(\\hat{\\mu}\_L \\approx \\hat{\\mu}\_R\\).

---

## Empirical Benchmark &amp; Distributional Diagnostics

The following synthetic illustration compares distributional modeling choices. It is not a real insurance benchmark:

1. **Panel 1 (Tweedie Poisson-Gamma)**: Captures the 84% zero-spike at $0 without zero-truncation artifacts, while accurately modeling the continuous heavy tail.
2. **Panel 2 (Tobit Censoring)**: OLS-based splits suffer severe slope flattening due to boundary accumulation at \\(y=0\\). The Tobit criterion recovers the true latent slope.
3. **Panel 3 (Heteroscedastic Volatility)**: Isolates high-volatility regimes (\\(\\sigma = 1.84\\)) from low-volatility regimes (\\(\\sigma = 0.12\\)), detecting structural variance shifts.

---

## Trained Distributional Scorecard Tree Structure

Below is the trained Tweedie tree diagram generated via `model.to_dot()` and Graphviz, showing explicit domain feature names and split thresholds:

Every node displays exact actuarial sample sizes \\(n\\) and deviance statistics: - High-risk young drivers with powerful engines (`DriverAge < 26.5` AND `VehiclePower >= 112.0`) route to elevated claim cost leaves. - Experienced drivers with favorable bonus-malus ratings route directly to base minimal cost leaves.

---

## Summary &amp; When to Use Distributional Trees

| Business Problem | Recommended Criterion | Why MSE Fails |
| :--- | :--- | :--- |
| **P&amp;C Insurance Loss Cost** | `gain="tweedie", tweedie_p=1.5` | Zero-inflation flattens severe claim signals. |
| **Survey Spend / Floor Censoring** | `gain="tobit", tobit_threshold=0.0` | Floor boundary produces attenuated, biased leaf estimates. |
| **Financial Volatility Regimes** | `gain="heteroscedastic"` | MSE cannot detect variance shifts when means are identical. |
| **Relative Scale Invariance** | `gain="cv"` | High-scale groups dominate absolute variance splits. |
| **Churn / Equipment Survival** | `gain="martingale"` | Right-censoring distorts unadjusted duration targets. |

By stepping outside the homoscedastic Gaussian box, `mltreelib` brings modern parametric likelihoods directly into tree splitting, with a Rust core. Performance depends on the dataset and configuration.

## Which tree should I use for censored or skewed data?

Choose the target and censoring mechanism first. Compare alternatives on held-out data using a metric suited to the decision. The [distributional leaves](<https://jkapila.github.io/mltreelib/examples/distributional-leaves/index.md>), [Tobit](<https://jkapila.github.io/mltreelib/examples/tobit-regression/index.md>), and [survival pathway](<https://jkapila.github.io/mltreelib/guides/survival/index.md>) provide runnable examples and assumptions.
