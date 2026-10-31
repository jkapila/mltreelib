# Time series feature engineering & forecasting

`mltreelib` provides time-feature transforms executed in pure Rust (`mltreelib-core`), including calendar cycle extractions, entity-grouped calendar day lags, rolling window statistics, and additive decomposition.

Script: [`examples/timeseries_forecasting.py`](https://github.com/jkapila/mltreelib/blob/main/examples/timeseries_forecasting.py) (`uv run python examples/timeseries_forecasting.py`).

## Output

```text
== 1. Synthetic Store Sales Panel (90 days x 2 stores = 180 observations) ==
  dates: 2025-01-01 to 2025-03-31

== 2. Engineered Time Features (first 4 rows) ==
  store      : ['store_nyc', 'store_lon', 'store_nyc', 'store_lon']
  weekday    : [2.0, 2.0, 3.0, 3.0]
  weekend    : [0.0, 0.0, 0.0, 0.0]
  lag_1d     : [None, None, 150.0, 100.0] (first day per store is None)
  win7_mean  : [150.0, 100.0, 150.25, 100.25]

== 3. ModelTree Fit (Linear leaves with regularized trend) ==
  test RMSE : 0.512
  test MAE  : 0.435
  test R2   : 1.000

== 4. Out-of-Time Predictions vs Actuals (first 6 rows) ==
  date=2025-03-12 store=store_nyc  actual=185.0 predicted=184.8
  date=2025-03-12 store=store_lon  actual=135.0 predicted=134.8
  date=2025-03-13 store=store_nyc  actual=185.5 predicted=185.6
  date=2025-03-13 store=store_lon  actual=135.5 predicted=135.6
  date=2025-03-14 store=store_nyc  actual=186.0 predicted=186.3
  date=2025-03-14 store=store_lon  actual=136.0 predicted=136.3
```

## How to use

### Calendar feature extraction

Extract calendar properties from ISO 8601 date strings (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`):

```python
from mltreelib import calendar_features

dates = ["2026-10-28", "2026-10-29", "2026-10-30"]
cal = calendar_features(dates, hemisphere="north")
# Returns dict: 'year', 'month', 'quarter', 'weekday', 'season', 'weekend', 'day_of_year'
```

### Entity-grouped lags

`lag_by_days` shifts observations by actual calendar elapsed days per group entity, ensuring irregular intervals and missing dates are handled correctly:

```python
from mltreelib import lag_by_days

# Shift values by 7 calendar days within each store key:
lag7 = lag_by_days(keys, dates, sales, lag_days=7)
```

Unobserved history produces `None`, which `ModelTree` natively routes without requiring imputation.

### Trailing window aggregates

Compute rolling trailing metrics over calendar windows:

```python
from mltreelib import windowed_transform

# 7-day trailing mean per entity:
rolling_mean = windowed_transform(keys, dates, sales, window_days=7, kind="mean")

# 14-day trailing sum:
rolling_sum = windowed_transform(keys, dates, sales, window_days=14, kind="sum")
```

### Training ModelTree on time panels

Assemble features into a row-major matrix and train with categorical declarations:

```python
from mltreelib import ModelTree

X = [
    [store, cal["weekday"][i], cal["weekend"][i], lag1[i], lag7[i], rolling_mean[i]]
    for i, store in enumerate(keys)
]
dtypes = ["categorical", "numeric", "numeric", "numeric", "numeric", "numeric"]

tree = ModelTree(max_depth=3, leaf="linear", l2=0.1)
tree.fit(X_train, y_train, dtypes=dtypes)

forecasts = tree.predict(X_test)
```
