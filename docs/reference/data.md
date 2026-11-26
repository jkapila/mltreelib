# API reference — Data

The data & feature layer: everything that turns raw columns into numeric
matrices. All names come from `import mltreelib`.

## Dataset helpers

- `list_datasets()` — the bundled benchmark sets (see
  [DataSets](../benchmarks/datasets.md)).
- `load_dataset(name)` — load one as a frame.
- `summarize(df)` — per-column summary including `n_missing` by default.

## Data helpers (`mltreelib.data`)

Standard-library helpers for profiling, screening, splitting, binning, scaling and reading model inputs. Import them from `mltreelib.data`, not from the top-level package. Every helper takes plain lists, dicts of columns or an `xy` result, returns plain Python types, and never mutates its input. None of them imputes, drops, recodes or reorders a value: `None` and NaN are missing, counted where it matters and passed through otherwise. Infinite values are reported separately and ignored by statistics. The guide [Prepare data](../guides/data.md) walks through them on one dataset.

### Profiling and quality (`profile`)

| Function | Signature | Purpose | Returns | Missing and infinite values |
|---|---|---|---|---|
| `describe` | `describe(columns, n_category=20, kinds=None, trim=0.1, percentiles=(0.05, 0.25, 0.5, 0.75, 0.95), inf_as_missing=False)` | One summary row per column: kind hint, missingness, statistics, outlier counts. | list of dicts: `name`, `kind`, `kind_source`, `n`, `n_missing`, `pct_missing`, `n_inf`, `n_distinct`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, `skew`, `kurtosis`, `pNN` per percentile, `iqr`, `mad`, `cv`, `trimmed_mean`, `trimmed_std`, `n_outliers_iqr`, `pct_outliers_iqr`, `n_zero`, `n_negative`, `n_positive`, `mode`, `top` | Counted in `n_missing`; statistics use finite observed values. Infinities go to `n_inf` and count as missing only with `inf_as_missing=True`. `kinds` overrides the kind hint (`numeric`, `categorical`, `ordinal`) and sets `kind_source` to `user`. |
| `infer_types` | `infer_types(columns, n_category=20, kinds=None)` | Kind and number of distinct levels per column, using the same hint as `describe`. | `{name: {kind, n_levels, ordinal_candidate}}` | Excluded from `n_levels`. |
| `missing_profile` | `missing_profile(columns, inf_as_missing=False)` | Missingness per column, per row and by pattern, with a co-missing matrix. | `n_rows`, `n_cols`, `per_column`, `per_row`, `n_complete_rows`, `patterns` (`columns`, `count`), `co_missing` (`names`, `matrix`) | Counted only; nothing removed. |
| `missing_vs_target` | `missing_vs_target(columns, y, inf_as_missing=False)` | Mean target when each column is missing versus observed. | `{name: n_missing, n_observed, y_mean_when_missing, y_mean_when_observed, n_y_missing_excluded}` | Rows with a missing or infinite target are excluded from every count and mean. Means are `None` for an empty group. |
| `class_balance` | `class_balance(y)` | Class counts, shares and imbalance ratio for a target. | `counts`, `proportions`, `imbalance_ratio` (`None` with fewer than two levels), `n_missing`, `n` | Missing targets are left out of counts and shares and reported in `n_missing`. |
| `constant_columns` | `constant_columns(columns)` | Names of columns with at most one distinct observed value. | list of names, input order | All-missing columns are included. |
| `near_zero_variance` | `near_zero_variance(columns, freq_cut=19.0, unique_cut=10.0)` | Caret near-zero-variance rule on observed values. | flagged columns only: `name`, `freq_ratio`, `pct_unique`, `zero_var` | Computed on observed values only. |
| `duplicate_rows` | `duplicate_rows(columns)` | Groups of identical rows. | list of ascending index lists (groups of two or more) | Missing equals missing (`None` and NaN match). `1` equals `1.0`. |
| `duplicate_columns` | `duplicate_columns(columns)` | Groups of columns with identical values row by row. | list of name lists (groups of two or more) | Same rule as `duplicate_rows`. |
| `outlier_flags` | `outlier_flags(values, method="iqr", k=None)` | Flags outliers in one numeric column: IQR fences (default `k=1.5`) or modified z-score on the MAD (default `k=3.5`). Flags only. | one entry per value: `None` missing, `True` infinite, otherwise `True` or `False` | Missing gets `None`; infinite gets `True`; quartiles and MAD use finite observed values. |
| `schema_capture` | `schema_capture(columns)` | Records kind, missing count, range (numeric) or sorted levels (other kinds, at most 200) for later checks. | `{"columns": {name: {kind, kind_source, n_missing, min, max, levels}}, "n_rows"}` | Missing counted in `n_missing`; not part of range or levels. |
| `schema_validate` | `schema_validate(columns, schema, allow_missing=True)` | Checks columns against a captured schema. | list of `{column, problem, detail}`; problems are `missing_column`, `extra_column`, `kind_mismatch`, `out_of_range`, `unknown_level`, `unexpected_missing`. Empty list means valid. | Missing allowed unless `allow_missing=False`. Infinities are not range-checked. |

### Association and screening (`assoc`)

| Function | Signature | Purpose | Returns | Missing and infinite values |
|---|---|---|---|---|
| `correlation_matrix` | `correlation_matrix(columns, method="pearson")` | Pairwise correlation between numeric columns; `method` is `pearson` or `spearman`. | `names`, `matrix` (`None` under 3 rows or zero variance), `n_pairs`, `skipped`, `method` | Pairwise complete rows: both values must be finite. |
| `correlation_filter` | `correlation_filter(result, cutoff=0.9)` | Greedy caret rule on a `correlation_matrix` result: drops the column with the larger mean absolute correlation until no pair exceeds `cutoff`. | `drop` (in drop order), `kept`, `pairs` (`a`, `b`, `corr`) | `None` entries are ignored. |
| `cramers_v_matrix` | `cramers_v_matrix(columns, bias_corrected=True, n_category=20)` | Cramer's V and chi-square p-values between categorical columns (strings, booleans, or integer codes with at most `n_category` levels). | `names`, `matrix`, `p_values`, `n_pairs`, `skipped` | Each pair uses rows where both are observed. |
| `vif` | `vif(columns)` | Variance inflation factor of each numeric column, from a regression on all the others. See the note below. | `vif` (float, `inf` or `None`), `n_used`, `n_excluded`, `names`, `skipped`, `warning` | Complete cases only: a row with a missing or non-finite numeric value is excluded and counted in `n_excluded`. |
| `leakage_hints` | `leakage_hints(columns, y, threshold=0.95)` | Single-feature predictive score per column against the target; flags only. | list of `name`, `score`, `metric`, `flagged`, `n_used` | Rows with a missing or non-finite feature or target are excluded from that column's score (`n_used`). |

Scores by metric: AUC (folded, binary target), `r2` (numeric target), `auc_level_rate` (categorical feature, binary target, in-sample rates) and `eta2` (categorical feature, numeric target).

### Splitting (`split`)

Every splitter returns row indices. A split depends only on its arguments and `seed`, never on global random state. Missing labels in `stratify` and `groups` form their own level or group. Missing `time_order` keys sort last. No row is dropped.

| Function | Signature | Purpose | Returns | Missing values |
|---|---|---|---|---|
| `train_test_split` | `train_test_split(n, test_size=0.2, seed=0, stratify=None, groups=None, time_order=None)` | Train and test row indices: shuffled, stratified (largest remainder), grouped (whole groups to test) or time-ordered (last rows to test). | `(train_idx, test_idx)`, both sorted | Missing labels are their own level or group; missing time keys sort last. |
| `train_val_test_split` | `train_val_test_split(n, val_size=0.2, test_size=0.2, seed=0, stratify=None, groups=None, time_order=None)` | Train, validation and test indices; the test split is made first, then validation from the rest. `val_size` is a fraction of `n`. | `(train_idx, val_idx, test_idx)` | As for `train_test_split`. |
| `kfold` | `kfold(n, k=5, seed=0, stratify=None, groups=None, time_order=None)` | K folds: shuffled blocks, stratified round-robin, whole groups per fold, or an expanding time window. | list of `k` `(train_idx, test_idx)` tuples | As for `train_test_split`. `stratify` and `groups` cannot be combined. |
| `repeated_kfold` | `repeated_kfold(n, k=5, repeats=3, seed=0, stratify=None, groups=None)` | `kfold` repeated with seeds `seed + r`. | list (length `repeats`) of `kfold` results | As for `kfold`. No `time_order`. |
| `take` | `take(columns, indices)` | Subsets rows of a dict of columns, or of a list or tuple, keeping the order of `indices`. | same container type with the selected rows | Values copied by reference, never transformed; missing stays missing. Negative or out-of-range indices raise `IndexError`. |

### Model inputs (`access`)

| Function | Signature | Purpose | Returns | Missing values |
|---|---|---|---|---|
| `xy` | `xy(columns, target, features=None, exclude=())` | Feature rows, target and metadata from columns; a `load_dataset` result uses its own dtypes. | `x` (one new list per row), `y`, `names`, `dtypes` (`numeric` or `categorical`), `n_y_missing` | Missing feature values stay in `x`; rows with a missing target are kept and counted. |
| `survival_xy` | `survival_xy(columns, time, event, features=None, exclude=())` | As `xy`, with `time` and `event` split out of the features. The event column is passed through unchanged. | `x`, `time`, `event`, `names`, `dtypes`, `n_time_missing`, `n_event_missing` | Kept and counted. |
| `uplift_xy` | `uplift_xy(columns, treatment, outcome, features=None, exclude=())` | As `xy`, with treatment and outcome split out. The treatment is not recoded. | `x`, `treatment`, `outcome`, `names`, `dtypes`, `n_treatment_missing`, `n_outcome_missing` | Kept and counted. |
| `select_rows` | `select_rows(result, indices)` | Subsets the row-aligned fields of an `xy`, `survival_xy` or `uplift_xy` result. Repeats are allowed. | a new dict with the same keys | Missing counts are recomputed for the selected rows. Out-of-range indices raise `IndexError`. |

### Binning and information value (`binning`)

| Function | Signature | Purpose | Returns | Missing and infinite values |
|---|---|---|---|---|
| `bin_edges` | `bin_edges(values, n_bins=10, method="quantile", min_obs=None)` | Interior cut points for numeric binning: R type-7 quantiles or equal width. | strictly increasing list of cut points | Missing and infinite values are ignored. |
| `binner` | `binner(values, n_bins=10, method="quantile", min_obs=None, edges=None, right=True, digits=4)` | Assigns numeric values to bins and counts them. | `edges`, `labels`, `counts`, `n_missing`, `n_inf`, `index` (bin per row, `None` if missing) | Missing gets no bin and is counted in `n_missing`. Infinities go to the end bins and are counted in `counts` and `n_inf`. |
| `iv_bins` | `iv_bins(values, y, n_bins=10, method="quantile", monotone=None, max_bins=None, min_bin_frac=0.05, smoothing=0.5, edges=None, optimize="greedy", n_prebins=50, trend=None, special_values=None)` | Information value of a numeric feature against a binary target, with ordered bins, optional trend constraint and special-value bins. | `iv`, `bins` (`lo`, `hi`, `special`, `n`, `events`, `non_events`, `event_rate`, `woe`, `iv`), `missing_bin`, `n_rows`, `n_used`, `n_y_missing`, `trend`, `optimize`, `objective`, `constraint_relaxed` | Missing feature values form `missing_bin`, never merged or dropped. Rows with a missing target are excluded (`n_y_missing`). Infinities fall into the end bins. |
| `iv_levels` | `iv_levels(values, y, min_level_frac=0.0, smoothing=0.5)` | Information value of a categorical feature: one bin per level, rare levels pooled as `__other__`. | `iv`, `bins` (`level`, `n`, `events`, `non_events`, `event_rate`, `woe`, `iv`), `missing_bin`, `n_rows`, `n_used`, `n_y_missing`, `trend` (always `None`) | Missing values form `missing_bin`. |
| `information_value_table` | `information_value_table(columns, y, n_bins=10, method="quantile", monotone=None, max_bins=None, min_bin_frac=0.05, smoothing=0.5, min_level_frac=0.0)` | IV for every column: `iv_bins` for numeric, `iv_levels` for categorical. | list sorted by IV descending: `name`, `iv`, `kind`, `n_bins`, `strength` | As for the two functions above. |

### Scaling and transforms (`scaling`)

| Function | Signature | Purpose | Returns | Missing and infinite values |
|---|---|---|---|---|
| `FeatureTransformer` | `FeatureTransformer(kind, **params)`; methods `fit(values)`, `transform(values)`, `fit_transform(values)`, `inverse_transform(values)` | One fitted transform per column. `kind` is `log1p`, `power` (`p`), `mean_diff`, `median_diff`, `boxcox` or `yeojohnson`; `lmbda` fixes lambda for the power transforms. | after `fit`: `params_` (`lambda`, `mean`, `median` or `p`), `n_fit_`, `n_missing_`, `n_inf_` | Missing and infinite rows pass through unchanged and are not used in the fit. A value outside the transform's domain raises `ValueError` naming the first bad index. |
| `GroupedFeatureTransformer` | `GroupedFeatureTransformer(kind, *, fallback=None, **params)`; methods `fit(values, groups)`, `transform(values, groups)`, `inverse_transform(values, groups)` | One `FeatureTransformer` per group label. | after `fit`: `transformers_` (by group), `global_` | A group unseen at fit time raises `KeyError` unless `fallback="global"`. Missing rules as for `FeatureTransformer`. |
| `log1p` | `log1p(values)` | `log(1 + x)` per row; no fit needed. | list | Missing passes through. |
| `boxcox_lambda` | `boxcox_lambda(values)` | Box-Cox lambda that maximises the profile likelihood on observed values (`x > 0`). | float; NaN with no observed values, `1.0` with fewer than two distinct values | Missing ignored. |
| `yeojohnson_lambda` | `yeojohnson_lambda(values)` | Yeo-Johnson lambda that maximises the profile likelihood on observed values. | float; NaN with no observed values, `1.0` with fewer than two distinct values | Missing ignored. |

`FeatureTransformer` fits lambda by golden-section search on `[-5, 5]`. A maximum at the boundary means the data wants a lambda outside that range.

### Notes

**VIF cost.** `vif` runs one regression per numeric column, each on all the other numeric columns, over the complete rows. The work therefore grows faster than linearly with the number of columns, and every column added makes every other column's regression larger. Screen the columns first (`correlation_filter`, `information_value_table`, domain knowledge), then run `vif` on the shortlist. On a wide table, do not run it on everything.

**WOE sign convention.** `iv_bins` and `iv_levels` report a bin's weight of evidence as `woe = ln(event share / non-event share)`. The event share is the bin's events over all events, and the non-event share is the bin's non-events over all non-events, both smoothed. This is the opposite sign of `ln(good / bad)` in many scorecard tools, where good means non-event. A bin with a high event rate therefore has positive WoE here and negative WoE in those tools. Information value is the same under both conventions, because each term of the sum changes sign with `woe`. Check the convention of any scorecard tool before copying a WoE value between them.

## Winnowing (column screening)

- `winnow(x, y, min_relevance=0.05)` → `{"kept": [column indices],
  "relevance": [score per column]}` — screen columns by univariate relevance
  against the target before training (the cheap pre-split screening LightGBM
  calls feature pre-filtering). The same screen runs inside
  `ModelTree.fit` when `min_relevance` is set.

## Categorical encoders

Encoder fit and transform signatures differ. Target-based encoders also need training targets. Missing and unseen-level behavior depends on the encoder. See the individual API and test the behavior required by your workflow.

| encoder | shape |
|---|---|
| `OneHotEncoder` | `levels()`, `transform` → `{level: [0/1 column]}` (missing → all-zero row) |
| `HashingEncoder(n_buckets)` | fixed-width hashing |
| `TargetEncoder(smoothing)` | smoothed target mean; `prior()` |
| `CatBoostEncoder(prior, permutations, seed)` | ordered target statistics |
| `WoEEncoder(prior)` | weight of evidence |
| `JamesSteinEncoder` | shrinkage estimator |
| `CountEncoder` | frequency encoding |
| `LeaveOneOutEncoder` | leave-one-out target mean |
| `MEstimatorEncoder(smoothing)` | m-estimate smoothing |
| `BaseNEncoder` | base-N numeric code |
| `SumEncoder` | sum (effect) coding |
| `QuantileEncoder` | quantile target statistics |
| `TfidfEncoder` | tf-idf over categories |
| `EmbeddingEncoder(dim, lr, epochs, seed, task)` | learned embeddings (Rust core) |

## Scalers & builders

- `VariableScaler(normalize)` — per-column numeric scaling with invertible
  `inverse_transform`.
- `GroupedScaler` — grouped/trended/windowed transforms; module functions
  `grouped_transform`, `trended_transform`, `windowed_transform`,
  `lag_by_days`, `decomposed_transform`, `calendar_features`.
- `DatasetBuilder` — `add_numeric(name, values)` / `add_categorical(name,
  values)` / `build()` → schema summary.
- `IndexMapper` — string ↔ code mapping: `fit`, `transform`,
  `inverse_transform`, `get_index`, `get_name`, `get_all_names`.

## Missing-value semantics

`None` **and NaN** = missing (NaN normalized at ingestion). Missing in →
missing out; imputation is opt-in; unseen categorical levels map to missing;
degenerate scalings produce missing, not inf. Time-series transforms drop
incomplete rows and warn with the count.

## Generator

`generate(...)` — the adjustable benchmark generator with known ground truth
(group/hierarchy signal knobs). See `docs/field-notes.md` for the research
context.

Next: [Model](model.md) for the tree itself, or [Metrics](metrics.md) for
evaluation.

## Class reference

Full docstrings and every method: the [full generated API](api/).

**`DatasetBuilder(n_category=None, na_policy='allow', digits=None,
add_intercept=False)`** — ingest any raw columns into the typed dataset.
`add_categorical(name, values)`, `add_numeric(name, values)`, `build()`
→ `{"columns", "dtypes", "na_counts", "n_rows", "n_cols"}`.

**Encoders** — signatures, output shapes and handling of missing or unseen levels vary. Consult the individual API before constructing a pipeline.

**`VariableScaler(normalize)` / `GroupedScaler`** — `fit` /
`transform` / `inverse_transform` (invertible by construction).

**`IndexMapper`** — `fit`, `transform`, `inverse_transform`, `get_index`,
`get_name`, `get_all_names`.

## A real example (churn)

```python
from mltreelib import DatasetBuilder, OneHotEncoder, load_dataset

d = load_dataset("churn")            # the typed dataset dict
cols = d["columns"]
print(f"churn: {d['n_rows']} rows x {d['n_cols']} cols; "
      f"na_counts (state) = {d['na_counts']['state']}")

b = DatasetBuilder(n_category=5, na_policy="impute")
b.add_categorical("state", [str(v) for v in cols["state"]])
b.add_numeric("account_length", [float(v) for v in cols["account_length"]])
b.add_numeric("total_day_charge", [float(v) for v in cols["total_day_charge"]])
d2 = b.build()
print(f"build(): {d2['n_rows']} rows x {d2['n_cols']} cols; "
      f"dtypes={d2['dtypes']}; na_counts={d2['na_counts']}")

enc = OneHotEncoder()
enc.fit([str(v) for v in cols["state"]])
X = enc.transform([str(v) for v in cols["state"]])
print(f"OneHotEncoder on state: fit -> {len(enc.levels())} levels; "
      f"transform -> dict of {len(X)} columns x {len(next(iter(X.values())))} rows")
```

```text
churn: 5000 rows x 20 cols; na_counts (state) = 0
build(): 5000 rows x 3 cols; dtypes={'state': 'categorical',
  'account_length': 'numeric', 'total_day_charge': 'numeric'};
  na_counts={'state': 0, 'account_length': 0, 'total_day_charge': 0}
OneHotEncoder on state: fit -> 51 levels; transform -> dict of 51 columns
  x 5000 rows
```

`load_dataset` returns the **built** typed dataset (`name`, `columns`,
`dtypes`, `na_counts`, `n_rows`, `n_cols`) — the same schema
`DatasetBuilder.build()` produces, which is why the two interoperate.
