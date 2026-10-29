# API reference — Data &amp; Model

Two layers: the **data** side (encoders, scalers, builders, metrics) and the **model** (`ModelTree`). Everything below is the surface exported by `import mltreelib`. The exhaustive generated listing (every class, every method signature) is at [`reference/api.md`](<https://jkapila.github.io/mltreelib/reference/api/index.md>).

## ModelTree

```
ModelTree(
    max_depth=4, min_rows=10, min_gain=1e-9,
    leaf_cols="all",        # "all" | "path" — columns available to leaf models
    leaf="linear",          # "linear" | "mean"
    l2=0.1, l1=0.0,         # ridge on leaf coefficients (l1 off by default)
    monotone=[],            # per column: 0 | +1 | -1
    interaction_groups=[],  # [[col, ...], ...] — paths may not mix groups
    min_relevance=None,     # relevance threshold for column screening
)
```

Defaults are the arena bake-off winner (see [Benchmarks](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>)) — linear leaves at shallow depth with a small **ridge** (`l2`) penalty. `l1` (lasso/elastic) is available but off by default: coordinate descent is slow inside the split scan and the accuracy gain over ridge is marginal.

### Methods

| method | returns | notes |
| --- | --- | --- |
| `fit(x, y, w=None)` | `self` | rows of `float`/`None` (`None` = missing); `w` = row weights (0 = row ignored) |
| `predict(x)` | `list[float]` | missing values use the learned NA direction; never panics on short rows |
| `__repr__()` | `str` | rpart-style text (`print(model)`); refuses to print an unfitted tree |
| `to_dot()` | `str` | Graphviz DOT — render with pygraphviz/graphviz (layout is graphviz's job) |
| `splits()` | `list[dict]` | `{col, threshold, missing_left}` per split |
| `leaves()` | `list[dict]` | one flat dict per leaf: `cols, coef, na_coef, intercept, n, mean, std, min, max, quantiles` |
| `split_paths()` | `list[list[str]]` | split column names per root→leaf path (constraint verification) |
| `n_leaves()` | `int` | terminal node count |
| `profile()` | `list[dict]` | per-node timing/shape counters (fit diagnostics) |

### Constraints

- `monotone[j] = +1` → predictions never decrease in column `j` (and never increase for `-1`). Enforced at split choice **and** in leaf coefficients.
- `interaction_groups` → every root-to-leaf path splits only columns of one group (LightGBM-style interaction constraints).

Both are verified against reference implementations in `examples/constraints_check.py` (R `gbm`'s `var.monotone`, LightGBM's `interaction_constraints`).

## Data layer

### Encoders (categorical → numeric)

All follow `fit(values)` → `transform(values)` with `None` passed through as missing: `OneHotEncoder`, `HashingEncoder(n_buckets)`, `TargetEncoder(smoothing)`, `CatBoostEncoder(prior, permutations, seed)`, `WoEEncoder(prior)`, `JamesSteinEncoder`, `CountEncoder`, `LeaveOneOutEncoder`, `MEstimatorEncoder(smoothing)`, `BaseNEncoder`, `SumEncoder`, `QuantileEncoder`, `TfidfEncoder`, `EmbeddingEncoder(dim, lr, epochs, seed, task)`.

### Scalers &amp; builders

- `VariableScaler(normalize)` — per-column numeric scaling with invertible `inverse_transform`.
- `GroupedScaler` — grouped/trended/windowed numeric transforms (`grouped_transform`, `trended_transform`, `windowed_transform`, `lag_by_days`, `decomposed_transform`, `calendar_features`).
- `DatasetBuilder` — `add_numeric` / `add_categorical` / `build()` schema summary.
- `IndexMapper` — string ↔ code mapping with `get_index` / `get_name`.

### Winnowing (column screening)

- `winnow(x, y, min_relevance=0.05)` → `{"kept": [column indices],   "relevance": [score per column]}` — screen columns by univariate relevance before training (LightGBM's "feature pre-filtering" idea).
- The same screen runs inside `fit` when `min_relevance` is set.

### Metrics

- `Metrics().regression(y, yhat)` → n, bias, mae, mad, mape, rmse, r2, max\_error, explained\_variance, decile table (missing-aware).
- `Metrics().classification(y_true, y_pred)` — **integer class labels** — accuracy, macro precision/recall/F1, per-class table, confusion matrix.
- `Metrics().regression_by_group(...)` — regression report per group.

### Datasets

`list_datasets()`, `load_dataset(name)`, `summarize(df)` — the bundled regression/classification sets (see [DataSets](<https://jkapila.github.io/mltreelib/benchmarks/datasets/index.md>)).
