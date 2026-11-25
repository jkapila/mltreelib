# Time Series Feature Engineering & Additive Forecasting (Suite 7)

In enterprise demand planning and energy load forecasting, models face secular growth trends, intra-week cycles, and calendar shocks.

Standard flat decision trees fail out-of-sample on trending series because constant-leaf partitions cannot extrapolate beyond the maximum target value seen during training.

`mltreelib` implements a zero-allocation, pure Rust feature engineering engine—providing ISO calendar extraction, entity-grouped calendar day lags, windowed rolling statistics, and additive trend decomposition—coupled with piecewise-linear model trees that extrapolate trends out-of-time.

Script: [`benchmarks/suites/timeseries_additive_forecasting.py`](https://github.com/jkapila/mltreelib/blob/main/benchmarks/suites/timeseries_additive_forecasting.py) (`uv run python benchmarks/suites/timeseries_additive_forecasting.py`).

---

## 1. Domain Context & The Extrapolation Barrier

```mermaid
flowchart LR
    subgraph DataEngine["mltreelib-core Pure Rust Transforms"]
        direction LR
        Raw["Raw Multi-Store Panel\n(store_id, date, sales)"] --> Cal["calendar_features\n(weekday, weekend, month, day_of_year)"]
        Raw --> Lag["lag_by_days\n(1-day & 7-day exact calendar shifts)"]
        Raw --> Win["windowed_transform\n(7-day trailing mean & std)"]
        Raw --> Dec["decomposed_transform\n(additive Trend, Season, Residual)"]
    end

    subgraph Modeling["Out-of-Time Forecasting Engine"]
        direction LR
        Features["Engineered Design Matrix\n(Zero Allocation, NaNs Preserved)"] --> SplitChoice{"Decision Tree Architecture"}
        SplitChoice -- "Standard CART" --> StepModel["Step-Function Mean Leaves\nFlattens Out at Historical Max"]
        SplitChoice -- "ModelTree" --> LinearModel["Piecewise Linear Ridge Leaves\nExtrapolates Continuous Secular Trend"]
    end

    DataEngine ==> Modeling

    classDef rawBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef rustBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef failBox fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;
    classDef passBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;

    class Raw rawBox;
    class Cal,Lag,Win,Dec rustBox;
    class SplitChoice gateBox;
    class StepModel failBox;
    class LinearModel passBox;
```

### Why Flat Trees Fail on Time Series:
1. **The Step-Function Ceiling**: Standard regression trees predict leaf sample means $\bar{y}_{\text{leaf}}$. In a growing business where future sales exceed historical peaks, a constant-leaf tree predicts the historical maximum for all future dates.
2. **Calendar Offset Misalignment**: Naive array row shifts (`df['sales'].shift(7)`) fail when store closures or holiday gaps omit records. Real forecasting requires shifting by elapsed **wall-clock calendar days** ($t - 7 \text{ days}$) grouped by entity.
3. **Imputation Distortion**: Shifting introduces unobserved initial histories (`None`). Forcing median imputation erases autoregressive relationships.

---

## 2. Mathematical Formulation of Pure Rust Transforms

### A. Entity-Grouped Calendar Day Lags (`lag_by_days`)
Given entity keys $\mathbf{k}$, date timestamps $\mathbf{t}$, and values $\mathbf{y}$, the Rust engine computes:

$$L_k(t, \Delta d) = y(t') \quad \text{where } t' = t - \Delta d \text{ days, with } \text{entity}(t') = \text{entity}(t)$$

If no transaction occurred on exact calendar date $t'$, the cell is assigned `None` and routed via `ModelTree`'s native missing handler without distorting observed lags.

### B. Additive Classical Time Series Decomposition (`decomposed_transform`)
For periodic series with period $m$ (e.g. $m = 7$ for weekly retail cycles):

$$y_t = T_t + S_t + I_t$$

1. **Trend ($T_t$)**: Computed via two-sided moving average filter:
   $$T_t = \frac{1}{m} \left( \frac{1}{2} y_{t - m/2} + \sum_{k = -m/2 + 1}^{m/2 - 1} y_{t+k} + \frac{1}{2} y_{t + m/2} \right)$$
2. **Seasonality ($S_t$)**: Detrended averages normalized to sum to zero over the cycle:
   $$\tilde{S}_k = \text{mean}(y_t - T_t \mid t \equiv k \pmod m), \quad S_k = \tilde{S}_k - \frac{1}{m} \sum_{j=1}^m \tilde{S}_j$$
3. **Residual Irregularity ($I_t$)**: $I_t = y_t - T_t - S_t$.

### C. Continuous Trend Extrapolation via Linear Leaves
In terminal leaf $\mathcal{R}_m$, `ModelTree` fits local ridge regression:

$$\hat{y}(t, \mathbf{x}) = \beta_0 + \beta_{\text{trend}} \cdot t + \sum_{k=1}^P \beta_k \cdot L_k(t)$$

Because time $t$ enters as an explicit continuous predictor with non-zero slope $\beta_{\text{trend}}$, predictions continue along the secular trajectory beyond the training horizon.

---

## 3. Benchmark Study: Multi-Store Retail Demand Panel

We evaluate on a simulated multi-store retail panel (240 records across 2 stores over 120 days from 2025-01-01 to 2025-04-30) exhibiting weekly seasonality and strong upward secular growth.
- **Training Period**: First 90 days (2025-01-01 to 2025-03-31, 180 records).
- **Out-of-Time Test Period**: Final 30 days (2025-04-01 to 2025-04-30, 60 records).

```python
from mltreelib import Metrics, ModelTree
from mltreelib.timefeat import calendar_features, decomposed_transform, lag_by_days, windowed_transform

# 1. Transform features in pure Rust
cal = calendar_features(dates)
lag_1 = lag_by_days(keys=stores, dates=dates, values=sales, days=1)
lag_7 = lag_by_days(keys=stores, dates=dates, values=sales, days=7)
win_7 = windowed_transform(keys=stores, dates=dates, values=sales, days=7, stat="mean")
decomp = decomposed_transform(values=sales, period=7, model="additive")

# 2. Build feature matrix and split out-of-time
X = list(zip(cal["weekday"], cal["month"], lag_1, lag_7, win_7, decomp["trend"]))
X_train, y_train = X[:180], sales[:180]
X_test, y_test = X[180:], sales[180:]

# 3. Fit standard CART step-function baseline
tree_cart = ModelTree(max_depth=3, min_rows=15, leaf="mean")
tree_cart.fit(X_train, y_train)
pred_cart = tree_cart.predict(X_test)

# 4. Fit ModelTree piecewise linear model tree
tree_linear = ModelTree(max_depth=3, min_rows=15, leaf="linear", l2=1e-3)
tree_linear.fit(X_train, y_train)
pred_linear = tree_linear.predict(X_test)
```

---

## 4. Empirical Benchmark Output & Quantitative Findings

```text
Generated retail sales panel: 240 rows (120 days x 2 stores).
Date range: 2025-01-01 to 2025-04-30

=== 1. Pure Rust Time Feature Transforms ===
Engineered features in pure Rust (mltreelib-core):
  - calendar_features : weekday, weekend, month, day_of_year
  - lag_by_days       : 1-day lag, 7-day lag (initial 14 rows None)
  - windowed_transform: 7-day trailing mean
  - decomposed        : detrended, deseasonalized, residual

Out-of-time evaluation split:
  Train: 180 rows (2025-01-01 to 2025-03-31)
  Test : 60 rows (2025-04-01 to 2025-04-30)

=== 2. ModelTree Out-of-Time Forecasting vs CART Step Function ===
Standard CART (Mean) : Test RMSE = 12.87, R2 = 0.787, MAE = 10.06
ModelTree (Linear)   : Test RMSE = 4.26, R2 = 0.977, MAE = 3.56
--> Out-of-time forecast error reduction: 66.9%
```

| Model Configuration | Leaf Type | Out-of-Time Test RMSE | Out-of-Time $R^2$ | Test MAE | Forecast Error Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Standard CART Baseline** | Piecewise Mean | 12.87 | 0.787 | 10.06 | Baseline |
| **`mltreelib.ModelTree`** | Piecewise Linear Ridge | **4.26** | **0.977** | **3.56** | **66.9% Lower Error** |

![Time Series Additive Decomposition and Out-of-Time Forecasting](../assets/timeseries_decomposition.png)

### Key Quantitative Findings:

1. **Massive Out-of-Time Error Drop (66.9%)**: `ModelTree` reduces out-of-sample RMSE from **12.87 to 4.26** and lifts $R^2$ from **0.787 to 0.977**, accurately capturing post-March trend expansion.
2. **Elimination of the Flat Ceiling (Panel 2)**: Standard CART flattens into horizontal plateaus as test dates exceed training values. `ModelTree` linear leaves maintain positive trend derivatives across all store cohorts.
3. **Rust Transformation Latency**: Feature engineering on the multi-store panel executes in compiled Rust, completely eliminating Python date-parsing and pandas groupby bottlenecks.

---

## 5. Domain Limitations & Operational Boundaries

1. **Multi-Step Recursive Error Compounding**:
   In multi-step autoregressive rollouts, errors in $\hat{y}_{t+1}$ feed into lagged inputs for $\hat{y}_{t+2}$. For horizons exceeding 30 days, direct forecasting or periodic re-anchoring is required.
2. **Trend Over-Extrapolation Risk**:
   Linear leaves fit unconstrained local slopes. If historical growth was parabolic or transient, linear extrapolation will over-predict over long horizons. Setting `l2 >= 1.0` or clipping leaf slopes bounds long-term drift.
3. **Irregular Calendar Gaps**:
   When telemetry ceases for extended periods (e.g., store shutdown for 3 months), all day-lags up to $\Delta d = 90$ become missing, routing records through the missing path.
