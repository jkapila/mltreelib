# API reference — Feature functions

Grouped/trended/windowed transforms, lags and calendar parts. Guide: [Data processing](<https://jkapila.github.io/mltreelib/examples/data-processing/index.md>).

## grouped\_scale

`pythongrouped_scale((keys, values, strategy='zscore'))` One-shot group scaling: fit on (keys, values) and transform the same rows.

Each row's value is scaled by its group's summary under `strategy` (centered family like "zscore" or ratio family like "mean"; see `GroupedScaler` for the full list and the exact inverse).

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing). strategy: one of "zscore", "min-max", "median-iqr", "mean-iqr", "median-qd", "median-std", "median-range", "mean-range", "min-qd", "min-iqr", "min", "max", "mean", "median", "iqr", "qd", "std", "range".

Returns: scaled values; rows with a missing key/value, an unseen group or a degenerate (zero) scale give None — missing in, missing out.

Raises: ValueError: unknown strategy or mismatched lengths.

## grouped\_transform

`pythongrouped_transform((keys, values, kind='mean'))` Per-row group summary of `values` within groups defined by `keys`.

Args: keys: group label per row (None = missing). values: numeric value per row (None/NaN = missing; excluded from the group statistics but counted in each group's `n_missing`). kind: "mean", "sum", "std", "count", "min", "max", "median", "q1" or "q3".

Returns: the summary of each row's group; rows with a missing key/value or an unseen group give None.

Raises: ValueError: unknown kind or mismatched lengths.

## trended\_transform

`pythontrended_transform((keys, dates, values, granularity='year_month', kind='mean', hemisphere='north'))` Grouped summary trended at a time granularity.

Each row gets the summary of its (key x time-bucket) cell — e.g. with granularity "year\_month" and kind "mean", the row's key-and-month mean. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). granularity: bucket spec joined by "\_" from {year, halfyear, season, quarter, month, week, weekday, day} — e.g. "year\_month", "year\_week", "month\_weekday". kind: summary kind ("mean", "sum", "std", "count", "min", "max", "median", "q1", "q3"). hemisphere: "north" or "south" (affects the season component).

Returns: per-row cell summaries; incomplete rows give None.

Raises: ValueError: unknown granularity/kind/hemisphere, unparseable date or mismatched lengths.

## windowed\_transform

`pythonwindowed_transform((keys, dates, values, window_days=7, kind='mean', include_current=True))` Trailing-window aggregation per key over day windows.

Each row summarizes the observed values of its key in a window of `window_days` days; incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). window\_days: window length in days (\>= 1); covering the `window_days` days ending at the row's date, or strictly earlier days when `include_current` is false. kind: summary kind ("mean", "sum", "std", "count", "min", "max", "median", "q1", "q3"). include\_current: include the row's own date in the window.

Returns: per-row window summaries; None for incomplete rows and for empty/all-missing windows.

Raises: ValueError: window\_days == 0, unknown kind, unparseable date or mismatched lengths.

## lag\_by\_days

`pythonlag_by_days((keys, dates, values, lag_days))` Lag values by whole days within each key (day granularity).

Each row receives the value of its key at `date - lag_days`; duplicate (key, date) rows keep the last occurrence. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). lag\_days: lag in whole days (0 returns the row's own value).

Returns: lagged values; None where no source row exists.

Raises: ValueError: unparseable date or mismatched lengths.

## decomposed\_transform

`pythondecomposed_transform((keys, dates, values, season_granularity='month', trend_window=3, hemisphere='north'))` Additive decomposition, following the statsmodels `seasonal_decompose` (additive) convention.

The trend is a centered moving average over each key's rows ordered by date; the seasonal component is the mean of (series - trend) per (key, season bucket), estimated from rows with a complete trend window. Incomplete rows (missing key/date/value) are dropped from estimation and reported via a warning.

Args: keys: series label per row (None = missing). dates: ISO date strings per row (None = missing). values: numeric values per row (None/NaN = missing). season\_granularity: bucket spec for the seasonal component (e.g. "month", "weekday" — same vocabulary as `trended_transform`). trend\_window: centered moving-average window in rows (\>= 1). hemisphere: "north" or "south" (affects the season component).

Returns: dict with `detrended` (series - trend), `deseasonalized` (series - seasonal) and `residual` (series - trend - seasonal); incomplete rows give None.

Raises: ValueError: trend\_window == 0, unknown granularity/hemisphere, unparseable date or mismatched lengths.

## calendar\_features

`pythoncalendar_features((dates, hemisphere='north'))` Calendar features for ISO dates (YYYY-MM-DD or timestamp prefix).

Produces year, halfyear, quarter, month, week (ISO), iso\_year, weekday (0 = Monday), day, day\_of\_year, season and weekend as numeric parts, ready for tree learners.

Args: dates: ISO date strings or timestamps (None = missing). hemisphere: "north" or "south" (southern shifts the season by six months).

Returns: dict of 11 feature lists aligned with `dates`; rows with a missing date give None features.

Raises: ValueError: unparseable date or unknown hemisphere.
