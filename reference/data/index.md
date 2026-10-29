# API reference — Data

The data &amp; feature layer: everything that turns raw columns into numeric matrices. All names come from `import mltreelib`.

## Dataset helpers

- `list_datasets()` — the bundled benchmark sets (see [DataSets](<https://jkapila.github.io/mltreelib/benchmarks/datasets/index.md>)).
- `load_dataset(name)` — load one as a frame.
- `summarize(df)` — per-column summary including `n_missing` by default.

## Winnowing (column screening)

- `winnow(x, y, min_relevance=0.05)` → `{"kept": [column indices],   "relevance": [score per column]}` — screen columns by univariate relevance against the target before training (the cheap pre-split screening LightGBM calls feature pre-filtering). The same screen runs inside `ModelTree.fit` when `min_relevance` is set.

## Categorical encoders

All follow `fit(values)` → `transform(values)` with `None` passed through as missing; unseen levels map to missing.

| encoder | shape |
| --- | --- |
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

## Scalers &amp; builders

- `VariableScaler(normalize)` — per-column numeric scaling with invertible `inverse_transform`.
- `GroupedScaler` — grouped/trended/windowed transforms; module functions `grouped_transform`, `trended_transform`, `windowed_transform`, `lag_by_days`, `decomposed_transform`, `calendar_features`.
- `DatasetBuilder` — `add_numeric(name, values)` / `add_categorical(name,   values)` / `build()` → schema summary.
- `IndexMapper` — string ↔ code mapping: `fit`, `transform`, `inverse_transform`, `get_index`, `get_name`, `get_all_names`.

## Missing-value semantics

`None` **and NaN** = missing (NaN normalized at ingestion). Missing in → missing out; imputation is opt-in; unseen categorical levels map to missing; degenerate scalings produce missing, not inf. Time-series transforms drop incomplete rows and warn with the count.

## Generator

`generate(...)` — the adjustable benchmark generator with known ground truth (group/hierarchy signal knobs). See `docs/field-notes.md` for the research context.

Next: [Model](<https://jkapila.github.io/mltreelib/reference/model/index.md>) for the tree itself, or [Metrics](<https://jkapila.github.io/mltreelib/reference/metrics/index.md>) for evaluation.

## Class reference

Full docstrings and every method: the [generated full listing](<https://jkapila.github.io/mltreelib/reference/api/index.md>).

**`DatasetBuilder(n_category=None, na_policy='allow', digits=None, add_intercept=False)`** — ingest any raw columns into the typed dataset. `add_categorical(name, values)`, `add_numeric(name, values)`, `build()` → `{"columns", "dtypes", "na_counts", "n_rows", "n_cols"}`.

**Encoders** — all: `fit(values)` → `transform(values)` (dict of columns), `levels()`; `None` = missing, unseen → missing. Constructors vary per encoder (see the table above); every arg is optional.

**`VariableScaler(normalize)` / `GroupedScaler`** — `fit` / `transform` / `inverse_transform` (invertible by construction).

**`IndexMapper`** — `fit`, `transform`, `inverse_transform`, `get_index`, `get_name`, `get_all_names`.

## A real example (churn)

```
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

```
churn: 5000 rows x 20 cols; na_counts (state) = 0
build(): 5000 rows x 3 cols; dtypes={'state': 'categorical',
  'account_length': 'numeric', 'total_day_charge': 'numeric'};
  na_counts={'state': 0, 'account_length': 0, 'total_day_charge': 0}
OneHotEncoder on state: fit -> 51 levels; transform -> dict of 51 columns
  x 5000 rows
```

`load_dataset` returns the **built** typed dataset (`name`, `columns`, `dtypes`, `na_counts`, `n_rows`, `n_cols`) — the same schema `DatasetBuilder.build()` produces, which is why the two interoperate.
