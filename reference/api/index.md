# API reference

## DatasetBuilder

```
class DatasetBuilder(n_category=None, na_policy='allow', digits=None, add_intercept=False)
```

Ingest any data into the typed dataset.

Owns the ingestion semantics: `n_category` category inference, explicit NA policy, digit rounding and an optional intercept column. Missing is `None` **or NaN** — both are counted in `na_counts`, and filling is opt-in only.

Args (constructor): n\_category: integral numeric columns with at most this many distinct levels become categorical; values \< 1 mean a fraction of rows (None = no inference). na\_policy: "allow" (keep missing), "drop" (drop rows containing any missing) or "impute" (numeric mean, categorical mode). digits: round floats to this many decimal places (None = no rounding). add\_intercept: append a constant 1.0 column named `intercept`.

### DatasetBuilder.add\_categorical

```
add_categorical(self, /, name, values)
```

Add a categorical column from strings; the vocabulary is first-seen order. None = missing.

Args: name: column name. values: one level per row (None = missing).

### DatasetBuilder.add\_numeric

```
add_numeric(self, /, name, values)
```

Add a numeric column. NaN is treated as missing.

Args: name: column name. values: one value per row (None/NaN = missing).

### DatasetBuilder.build

```
build(self, /)
```

Build the typed dataset with the configured policies applied.

Returns: dict with `columns` (name -\> values), `dtypes` ("numeric" or "categorical"), `na_counts` (missing per column), `n_rows`, `n_cols`.

Raises: ValueError: inconsistent column lengths, duplicate names, invalid policy usage (e.g. "drop" removing every row).

## GroupedScaler

```
class GroupedScaler()
```

Fit/transform group distributional scaling with exact inverse (sklearn-shaped: fit once, then transform / inverse\_transform).

Strategies: centered family — "zscore", "min-max", "median-iqr", "mean-iqr", "median-qd", "median-std", "median-range", "mean-range", "min-qd", "min-iqr"; ratio family — "min", "max", "mean", "median", "iqr", "qd", "std", "range". Missing in -\> missing out; degenerate (zero) scales give missing, never inf.

### GroupedScaler.fit

```
fit(self, /, keys, values, strategy)
```

Fit group statistics of `values` under the given strategy.

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing; counted in each group's `n_missing`). strategy: strategy name (see the class docstring).

Raises: ValueError: unknown strategy or mismatched lengths.

### GroupedScaler.inverse\_transform

```
inverse_transform(self, /, keys, values)
```

Exact inverse of `transform` under the fitted statistics.

Args: keys: group label per row. values: previously transformed values.

Returns: original-scale values (up to floating-point error).

Raises: ValueError: not fitted or mismatched lengths.

### GroupedScaler.transform

```
transform(self, /, keys, values)
```

Scale values against the fitted group statistics.

Args: keys: group label per row. values: numeric value per row.

Returns: scaled values; None for missing key/value, unseen groups and degenerate scales.

Raises: ValueError: not fitted or mismatched lengths.

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

All fitted levels in code order.

Raises: ValueError: column not fitted.

### IndexMapper.get\_index

```
get_index(self, /, name, value)
```

Code for one level (None when the level is unseen).

Raises: ValueError: column not fitted.

### IndexMapper.get\_name

```
get_name(self, /, name, index)
```

Level for one code.

Raises: ValueError: column not fitted or code out of range.

### IndexMapper.inverse\_transform

```
inverse_transform(self, /, name, codes)
```

Map codes back to their string levels.

Args: name: fitted column name. codes: codes per row (None = missing).

Returns: string levels (None where the code is missing).

Raises: ValueError: column not fitted or code out of range.

### IndexMapper.transform

```
transform(self, /, name, values)
```

Codes for `values`; unseen levels map to None (missing).

## VariableScaler

```
class VariableScaler()
```

Per-column distributional scaling — each variable transformed independently (legacy `VariableTransformation` semantics).

Same strategy names as `GroupedScaler`, applied per column, with exact `inverse_transform`. None/NaN are missing and pass through as missing; degenerate scales give missing, never inf.

### VariableScaler.fit

```
fit(self, /, columns, strategy)
```

Fit all columns at once under the given strategy.

Args: columns: {name: values} for every column to scale (None/NaN = missing; counted in each column's `n_missing`). strategy: strategy name (see `GroupedScaler`).

Raises: ValueError: unknown strategy.

### VariableScaler.inverse\_transform

```
inverse_transform(self, /, name, values)
```

Exact inverse of `transform` under the fitted statistics.

Args: name: fitted column name. values: previously transformed values.

Returns: original-scale values (up to floating-point error).

Raises: ValueError: not fitted or unknown column.

### VariableScaler.transform

```
transform(self, /, name, values)
```

Scale one fitted column's values.

Args: name: fitted column name. values: numeric values (None/NaN = missing).

Returns: scaled values; None for missing inputs and degenerate scales.

Raises: ValueError: not fitted or unknown column.

## calendar\_features

```
def calendar_features(dates, hemisphere='north')
```

Calendar features for ISO dates (YYYY-MM-DD or timestamp prefix).

Produces year, halfyear, quarter, month, week (ISO), iso\_year, weekday (0 = Monday), day, day\_of\_year, season and weekend as numeric parts, ready for tree learners.

Args: dates: ISO date strings or timestamps (None = missing). hemisphere: "north" or "south" (southern shifts the season by six months).

Returns: dict of 11 feature lists aligned with `dates`; rows with a missing date give None features.

Raises: ValueError: unparseable date or unknown hemisphere.

## decomposed\_transform

```
def decomposed_transform(keys, dates, values, season_granularity='month', trend_window=3, hemisphere='north')
```

Additive decomposition, following the statsmodels `seasonal_decompose` (additive) convention.

The trend is a centered moving average over each key's rows ordered by date; the seasonal component is the mean of (series - trend) per (key, season bucket), estimated from rows with a complete trend window. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). season\_granularity: bucket spec for the seasonal component (e.g. "month", "weekday" — same vocabulary as `trended_transform`). trend\_window: centered moving-average window in rows (\>= 1). hemisphere: "north" or "south" (affects the season component).

Returns: dict with `detrended` (series - trend), `deseasonalized` (series - seasonal) and `residual` (series - trend - seasonal); incomplete rows give None.

Raises: ValueError: trend\_window == 0, unknown granularity/hemisphere, unparseable date or mismatched lengths.

## generate

```
def generate(n_rows=10000, n_numeric=5, cardinalities=None, hierarchy_levels=3, branching=None, group_effect_strength=0.5, hierarchy_effect_strength=0.3, interaction_strength=0.2, noise=0.1, task='regression', n_classes=2, tail_behavior='gaussian', missingness=0.0, seed=42)
```

Generate an adjustable benchmark dataset with known ground truth.

The knobs control how much signal lives in numeric effects, categorical levels, hierarchical groups and their interaction — so learners can be scored on **structure recovery**, not just accuracy. Deterministic for a given `seed`.

Args: n\_rows: number of rows to generate. n\_numeric: numeric feature columns; the first two carry linear signal. cardinalities: level count per categorical column (each \>= 2). hierarchy\_levels: depth of the categorical hierarchy (0 = none). branching: child count per hierarchy level (last value repeats). group\_effect\_strength: scale of categorical level effects. hierarchy\_effect\_strength: scale of hierarchy node effects. interaction\_strength: scale of the numeric x categorical interaction. noise: scale of the residual noise. task: "regression" or "classification". n\_classes: class count for classification (equal-frequency cuts). tail\_behavior: "gaussian", "heavy" (Student-t residuals) or "bathtub" (noise grows with the signal — the DAR motivation). missingness: fraction of feature cells set to missing, in \[0, 1). seed: RNG seed.

Returns: dict with `columns` (name -\> values, None = missing), `target`, `labels` (classification only) and `truth` — the true effects plus `truth["variance"]`, the realized variance share of each component.

Raises: ValueError: invalid parameter combination.

## grouped\_scale

```
def grouped_scale(keys, values, strategy='zscore')
```

One-shot group scaling: fit on (keys, values) and transform the same rows.

Each row's value is scaled by its group's summary under `strategy` (centered family like "zscore" or ratio family like "mean"; see `GroupedScaler` for the full list and the exact inverse).

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing). strategy: one of "zscore", "min-max", "median-iqr", "mean-iqr", "median-qd", "median-std", "median-range", "mean-range", "min-qd", "min-iqr", "min", "max", "mean", "median", "iqr", "qd", "std", "range".

Returns: scaled values; rows with a missing key/value, an unseen group or a degenerate (zero) scale give None — missing in, missing out.

Raises: ValueError: unknown strategy or mismatched lengths.

## grouped\_transform

```
def grouped_transform(keys, values, kind='mean')
```

Per-row group summary of `values` within groups defined by `keys`.

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing; excluded from the group statistics but counted in each group's `n_missing`). kind: "mean", "sum", "std", "count", "min", "max", "median", "q1" or "q3".

Returns: the summary of each row's group; rows with a missing key/value or an unseen group give None.

Raises: ValueError: unknown kind or mismatched lengths.

## lag\_by\_days

```
def lag_by_days(keys, dates, values, lag_days)
```

Lag values by whole days within each key (day granularity).

Each row receives the value of its key at `date - lag_days`; duplicate (key, date) rows keep the last occurrence. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). lag\_days: lag in whole days (0 returns the row's own value).

Returns: lagged values; None where no source row exists.

Raises: ValueError: unparseable date or mismatched lengths.

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

Summary of observed values, with missing counts shown by default.

NAs and NaNs are excluded from the statistics but always reported in `n_missing` — nothing is silently hidden.

Args: values: numeric values (None/NaN = missing).

Returns: dict with `count` (observed), `n_missing`, `sum`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max` (NaN for the statistics of an all-missing input).

## trended\_transform

```
def trended_transform(keys, dates, values, granularity='year_month', kind='mean', hemisphere='north')
```

Grouped summary trended at a time granularity.

Each row gets the summary of its (key x time-bucket) cell — e.g. with granularity "year\_month" and kind "mean", the row's key-and-month mean. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). granularity: bucket spec joined by "\_" from {year, halfyear, season, quarter, month, week, weekday, day} — e.g. "year\_month", "year\_week", "month\_weekday". kind: summary kind ("mean", "sum", "std", "count", "min", "max", "median", "q1", "q3"). hemisphere: "north" or "south" (affects the season component).

Returns: per-row cell summaries; incomplete rows give None.

Raises: ValueError: unknown granularity/kind/hemisphere, unparseable date or mismatched lengths.

## windowed\_transform

```
def windowed_transform(keys, dates, values, window_days=7, kind='mean', include_current=True)
```

Trailing-window aggregation per key over day windows.

Each row summarizes the observed values of its key in a window of `window_days` days; incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). window\_days: window length in days (\>= 1); covering the `window_days` days ending at the row's date, or strictly earlier days when `include_current` is false. kind: summary kind ("mean", "sum", "std", "count", "min", "max", "median", "q1", "q3"). include\_current: include the row's own date in the window.

Returns: per-row window summaries; None for incomplete rows and for empty/all-missing windows.

Raises: ValueError: window\_days == 0, unknown kind, unparseable date or mismatched lengths.
