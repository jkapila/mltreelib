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
