# API reference — Data & Model

Two layers: the **data** side (encoders, scalers, builders, metrics) and the
**model** (`ModelTree`). Everything below is the surface exported by
`import mltreelib`. The exhaustive generated listing (every class, every
method signature) is at [`reference/api.md`](api.md).

## ModelTree

```python
ModelTree(
    max_depth=4, min_rows=10, min_gain=1e-9,
    leaf_cols="all",        # "all" | "path" — columns available to leaf models
    leaf="linear",          # "linear" | "mean"
    l2=0.1, l1=0.1,         # ridge / lasso penalties on leaf coefficients
    monotone=[],            # per column: 0 | +1 | -1
    interaction_groups=[],  # [[col, ...], ...] — paths may not mix groups
    min_relevance=None,     # relevance threshold for column screening
)
```

Defaults are the arena bake-off winner (see [Benchmarks](../benchmarks/elo.md)) —
plain unregularized leaves were beaten by shallow regularized ones.

### Methods

| method | returns | notes |
|---|---|---|
| `fit(x, y, w=None)` | `self` | rows of `float`/`None` (`None` = missing); `w` = row weights (0 = row ignored) |
| `predict(x)` | `list[float]` | missing values use the learned NA direction; never panics on short rows |
| `__repr__()` | `str` | rpart-style text (`print(model)`); refuses to print an unfitted tree |
| `to_svg()` | `str` | self-contained SVG diagram (boxes + stats + threshold-labeled edges) |
| `splits()` | `list[dict]` | `{col, threshold, missing_left}` per split |
| `leaves()` | `list[tuple]` | `(leaf_model, n)` per terminal node; leaf model has `stats` (mean/std/min/max/digest) and `linear` (coef/na_coef/intercept) |
| `split_paths()` | `list[list[str]]` | split column names per root→leaf path (constraint verification) |
| `n_leaves()` | `int` | terminal node count |
| `profile()` | `list[dict]` | per-node timing/shape counters (fit diagnostics) |

### Constraints

- `monotone[j] = +1` → predictions never decrease in column `j` (and never
  increase for `-1`). Enforced at split choice **and** in leaf coefficients.
- `interaction_groups` → every root-to-leaf path splits only columns of one
  group (LightGBM-style interaction constraints).

Both are verified against reference implementations in
`examples/constraints_check.py` (R `gbm`'s `var.monotone`, LightGBM's
`interaction_constraints`).

## Data layer

### Encoders (categorical → numeric)

All follow `fit(values)` → `transform(values)` with `None` passed through as
missing: `OneHotEncoder`, `HashingEncoder(n_buckets)`, `TargetEncoder(smoothing)`,
`CatBoostEncoder(prior, permutations, seed)`, `WoEEncoder(prior)`,
`JamesSteinEncoder`, `CountEncoder`, `LeaveOneOutEncoder`,
`MEstimatorEncoder(smoothing)`, `BaseNEncoder`, `SumEncoder`,
`QuantileEncoder`, `TfidfEncoder`, `EmbeddingEncoder(dim, lr, epochs, seed, task)`.

### Scalers & builders

- `VariableScaler(normalize)` — per-column numeric scaling with invertible
  `inverse_transform`.
- `GroupedScaler` — grouped/trended/windowed numeric transforms
  (`grouped_transform`, `trended_transform`, `windowed_transform`,
  `lag_by_days`, `decomposed_transform`, `calendar_features`).
- `DatasetBuilder` — `add_numeric` / `add_categorical` / `build()` schema
  summary.
- `IndexMapper` — string ↔ code mapping with `get_index` / `get_name`.

### Metrics

- `Metrics().regression(y, yhat)` → n, bias, mae, mad, mape, rmse, r2,
  max_error, explained_variance, decile table (missing-aware).
- `Metrics().classification(y_true, y_pred)` — **integer class labels** —
  accuracy, macro precision/recall/F1, per-class table, confusion matrix.
- `Metrics().regression_by_group(...)` — regression report per group.

### Datasets

`list_datasets()`, `load_dataset(name)`, `summarize(df)` — the bundled
regression/classification sets (see [DataSets](../benchmarks/datasets.md)).
