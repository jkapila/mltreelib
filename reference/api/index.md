# API reference

## DatasetBuilder

```
class DatasetBuilder(n_category=None, na_policy='allow', digits=None, add_intercept=False)
```

Ingest any data into the typed dataset: `n_category` category inference, NA policy (allow/drop/impute), digit rounding, optional intercept column.

### DatasetBuilder.add\_categorical

```
add_categorical(self, /, name, values)
```

*(no docstring)*

### DatasetBuilder.add\_numeric

```
add_numeric(self, /, name, values)
```

*(no docstring)*

### DatasetBuilder.build

```
build(self, /)
```

Build the typed dataset; returns `columns`, `dtypes`, `n_rows`, `n_cols`.

## GroupedScaler

```
class GroupedScaler()
```

Fit/transform group distributional scaling with exact inverse (sklearn-shaped: fit once, then transform / inverse\_transform).

### GroupedScaler.fit

```
fit(self, /, keys, values, strategy)
```

Fit group statistics of `values` under the given strategy.

### GroupedScaler.inverse\_transform

```
inverse_transform(self, /, keys, values)
```

*(no docstring)*

### GroupedScaler.transform

```
transform(self, /, keys, values)
```

*(no docstring)*

## IndexMapper

```
class IndexMapper()
```

Categorical string \<-\> stable code mapping (sklearn-style fit/transform).

### IndexMapper.fit

```
fit(self, /, name, values)
```

Learn the vocabulary of `name` (first-seen order). Re-fitting resets it.

### IndexMapper.get\_all\_names

```
get_all_names(self, /, name)
```

*(no docstring)*

### IndexMapper.get\_index

```
get_index(self, /, name, value)
```

*(no docstring)*

### IndexMapper.get\_name

```
get_name(self, /, name, index)
```

*(no docstring)*

### IndexMapper.inverse\_transform

```
inverse_transform(self, /, name, codes)
```

*(no docstring)*

### IndexMapper.transform

```
transform(self, /, name, values)
```

Codes for `values`; unseen levels map to None (missing).

## VariableScaler

```
class VariableScaler()
```

Per-column distributional scaling (each variable independently) — sklearn-shaped: fit once, then transform / inverse\_transform. None/NaN are missing and pass through as missing.

### VariableScaler.fit

```
fit(self, /, columns, strategy)
```

Fit all columns at once: {name: values}.

### VariableScaler.inverse\_transform

```
inverse_transform(self, /, name, values)
```

*(no docstring)*

### VariableScaler.transform

```
transform(self, /, name, values)
```

*(no docstring)*

## calendar\_features

```
def calendar_features(dates, hemisphere='north')
```

Calendar features for ISO dates (YYYY-MM-DD or timestamp): year, halfyear, quarter, month, week (ISO), iso\_year, weekday (0=Mon), day, day\_of\_year, season, weekend. Missing dates give missing features.

## decomposed\_transform

```
def decomposed_transform(keys, dates, values, season_granularity='month', trend_window=3, hemisphere='north')
```

Additive decomposition (statsmodels `seasonal_decompose` convention): returns `detrended` (series − trend), `deseasonalized` (series − seasonal) and `residual` (series − trend − seasonal). Incomplete rows are dropped from estimation and reported via a warning.

## generate

```
def generate(n_rows=10000, n_numeric=5, cardinalities=None, hierarchy_levels=3, branching=None, group_effect_strength=0.5, hierarchy_effect_strength=0.3, interaction_strength=0.2, noise=0.1, task='regression', n_classes=2, tail_behavior='gaussian', missingness=0.0, seed=42)
```

Generate an adjustable benchmark dataset with known ground truth.

Knobs mirror `mltreelib_core::generator::GeneratorParams`; see its docs for semantics. Deterministic for a given `seed`.

## grouped\_scale

```
def grouped_scale(keys, values, strategy='zscore')
```

One-shot group scaling: fit on (keys, values) and transform the same rows.

## grouped\_transform

```
def grouped_transform(keys, values, kind='mean')
```

Per-row group summary of `values` within groups defined by `keys`.

## lag\_by\_days

```
def lag_by_days(keys, dates, values, lag_days)
```

Lag values by whole days within each key (day granularity). Incomplete rows are dropped from estimation and reported via a warning.

## list\_datasets

```
def list_datasets() -> 'list[str]'
```

Names of the available benchmark datasets (parquet stems).

## load\_dataset

```
def load_dataset(name: 'str') -> 'dict'
```

Load one dataset as `name`/`columns`/`dtypes`/`n_rows`/`n_cols`.

`columns` maps column name to a list with `None` for missing values; `dtypes` is "numeric" or "categorical" (strings/dictionaries).

## summarize

```
def summarize(values)
```

Summary statistics of observed values (NAs skipped).

## trended\_transform

```
def trended_transform(keys, dates, values, granularity='year_month', kind='mean', hemisphere='north')
```

Grouped summary trended at a time granularity (e.g. "year\_month", "year\_week", "month\_weekday"): each row gets the summary of its (key x time-bucket) cell. Incomplete rows are dropped from estimation and reported via a warning.

## windowed\_transform

```
def windowed_transform(keys, dates, values, window_days=7, kind='mean', include_current=True)
```

Trailing-window aggregation per key over day windows (`window_days` days ending at the row's date). Incomplete rows are dropped from estimation and reported via a warning.
