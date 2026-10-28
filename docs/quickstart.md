# Quickstart

## Data layer

Ingest any data with the semantics that matter: category inference, explicit NA
policies, digit rounding, optional intercept.

```python
from mltreelib import DatasetBuilder

b = DatasetBuilder(n_category=0.05, na_policy="impute", digits=2, add_intercept=True)
b.add_numeric("price", [1.234, None, 2.356])
b.add_categorical("region", ["north", "south", None])
ds = b.build()
ds["n_rows"], ds["dtypes"], ds["na_counts"]
```

Missing values are `None` **or NaN** — both are missing, counted in
`na_counts`, and filling is never implicit (`allow` / `drop` / `impute`).

## Summaries and encodings

```python
from mltreelib import summarize, IndexMapper

summarize([1.0, None, 3.0])   # count, n_missing, mean, std, min, q1, median, q3, max

m = IndexMapper()
m.fit("region", ["north", "south", "north"])
m.transform("region", ["north", "unseen"])   # [0, None] — unseen levels are missing
```

## Feature transforms

```python
from mltreelib import GroupedScaler, VariableScaler, grouped_transform

grouped_transform(keys, values, kind="mean")        # per-group summary per row
g = GroupedScaler(); g.fit(keys, values, "zscore"); g.transform(keys, values)
v = VariableScaler(); v.fit({"price": prices}, "min-max")   # per-column, exact inverse
```

18 distributional strategies (`zscore`, `min-max`, `median-iqr`, `min-qd`, … and
the ratio family), all with algebraic `inverse_transform`.

## Time features

```python
from mltreelib import calendar_features, lag_by_days, trended_transform, windowed_transform, decomposed_transform

calendar_features(["2026-10-28"])                  # weekday, season, ISO week, ...
lag_by_days(keys, dates, values, 7)                # day lags within each key
trended_transform(keys, dates, values, granularity="year_month")
windowed_transform(keys, dates, values, window_days=7, kind="sum")
decomposed_transform(keys, dates, values, season_granularity="weekday", trend_window=7)
```

Decomposition follows the statsmodels `seasonal_decompose` (additive)
convention: `detrended`, `deseasonalized`, `residual`. Time-series transforms
drop incomplete rows and raise a `UserWarning` with the count.

## Generator with ground truth

```python
from mltreelib import generate

g = generate(n_rows=10_000, cardinalities=[8, 64], hierarchy_levels=2,
             group_effect_strength=0.6, seed=42)
g["truth"]["variance"]   # realized variance share per component
```

Every learner can be scored on **structure recovery**, not just accuracy.
