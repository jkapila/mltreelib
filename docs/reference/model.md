# API reference — Model

`ModelTree` — the tree: greedy splits by fitted-leaf SSE reduction, linear or
mean leaves, NA-aware (learned split directions + NA-indicator leaf terms).
Rust core is the source of truth; this page mirrors it.

## Constructor

```python
ModelTree(
    max_depth=4, min_rows=10, min_gain=1e-9,
    leaf_cols="all",        # "all" | "path" — columns available to leaf models
    leaf="linear",          # "linear"|"mean"|"median"|"quantile"|"huber"
    tau=0.5,                # the quantile for leaf="quantile"
    l2=0.1, l1=0.0,         # ridge on leaf coefficients (l1 off by default)
    monotone=[],            # per column: 0 | +1 | -1
    interaction_groups=[],  # [[col, ...], ...] — paths may not mix groups
    min_relevance=None,     # relevance threshold for column screening
    gain="sse",             # regression: "sse"|"friedman_mse"|"absolute_error"|"huber"|"poisson"|"sdr";
                            # class: "entropy"|"gini"|"gain_ratio"|"twoing"; or callable
    split_precision=0.001, # numeric candidate grid (thousandths); None = exact unique scan
    cat_order="frequency_desc",  # "frequency_desc" | "target_mean" | [level codes]
    cat_search="cumulative",     # "cumulative" | "single_vs_rest" | "both"
    leaf_digest=True,       # keep each leaf's t-digest (quantile() needs it)
    verbose=0,              # 0 silent | 1 per depth | 2 per node (training_log)
    alpha=0.0,              # min child weight share (grf's alpha); 0 = off
    imbalance_penalty=0.0,  # subtract penalty*(1/n_l + 1/n_r) from the gain
    col_sample=None,        # per-node random column fraction (forest spirit)
    seed=42,                # column-subset seed (deterministic)
    prune="none",           # "none" | "bic" | "mdl" — bottom-up cost pruning
)
```

The gain family is **sample-weight aware end to end**: weighted class
masses for entropy/gini/gain_ratio/twoing, weighted moments for sdr/SSE,
and a custom `gain_fn(left, right)` receives lazy `(y, w)` pair slices of
the two sides (higher = better). `gain_ratio` is C4.5 (information gain ÷
split information), `sdr` is the model-tree standard-deviation reduction,
`twoing` is Breiman's balanced-contrast rule.

Defaults are the arena bake-off winner (see [Benchmarks](../benchmarks/elo.md))
— linear leaves at shallow depth with a small **ridge** (`l2`) penalty. `l1`
(lasso/elastic) is available but off by default: coordinate descent is slow
inside the split scan and the accuracy gain over ridge is marginal. Turning
regularization fully off (`l1=l2=0`) on raw scales is a known blowup mode —
keep the ridge unless you standardize your inputs.

## Split search — per dtype, no encoding

Categorical columns are **levels as-is** (`dtypes` declares them; a distinct
value is a level), NAs route as-is. Nothing is ever encoded on the growth
path — the encoders in [Data](data.md) are pipeline tools for feeding
sklearn and friends, never invoked by the tree. Binary level partitions
(`cat_search`) are built along `cat_order`; multiway partitions are reserved
for a later build. `gain` picks the split-quality function: the built-ins
scan on summary stats, a custom `gain_fn(left_y, right_y) -> float` receives
lazy y-slices per candidate (higher = better; sees y values only). Float
columns scan candidates on the `split_precision` grid (thresholds are grid
values; prediction rounds to the same grid), `None` restores the exact scan.

```python
m = ModelTree(max_depth=1, min_rows=5, gain="entropy", cat_search="cumulative")
m.fit(x, y, dtypes=["categorical"])
m.predict([[0.0], [1.0], [2.0]])
```

```text
categorical fit: pred(0,1,2) = [0.0, 5.0, 5.0]     # {0} | {1,2} partition
grid threshold: 1.3 (on the grid: True)
```

## Quantiles and the training log

`leaf_digest=True` keeps a t-digest in every leaf (weighted targets) —
`quantile(x, tau)` then answers **any** τ after the fit (median now, p90
later, no refit). `leaf_digest=False` drops the digests (leaner; quantile
returns NaN). `training_log()` returns the scan log when `verbose > 0`:
scan config, per-depth/per-node candidate counts, timings.

```python
mq = ModelTree(max_depth=1, min_rows=5)          # leaf_digest=True default
mq.fit(xg, yg)
mq.quantile([[1.25]], 0.5), mq.quantile([[2.25]], 0.9)
```

```text
quantile 0.5: 0.0 | p90: 1.0
training_log: 3 lines; first: depth 0: nodes=1 rows=60 candidates=3 scan_ms=0
```

## Fitting & predicting

| method | returns | notes |
|---|---|---|
| `fit(x, y, w=None, dtypes=None)` | `self` | rows of `float`/`None` (`None` = missing); `w` = row weights (0 = row ignored); `dtypes` = per column `"numeric"`/`"categorical"` (levels as-is); rows with `None` target drop |
| `predict(x)` | `list[float]` | missing cells route by the learned NA direction; never panics on short rows |
| `quantile(x, tau)` | `list[float]` | conditional τ-quantile per row from the leaf t-digests |

## Reading the model

| method | returns | notes |
|---|---|---|
| `__repr__()` | `str` | rpart-style text (`print(model)`); refuses to print an unfitted tree |
| `to_dot()` | `str` | Graphviz DOT — render with pygraphviz/graphviz (layout is graphviz's job) |
| `splits()` | `list[dict]` | `{col, threshold, missing_left, gain}` per split (`col` is the column **name**) |
| `leaves()` | `list[dict]` | one flat dict per leaf: `cols, coef, na_coef, intercept, n, mean, std, min, max, quantiles` |
| `split_paths()` | `list[list[str]]` | split column names per root→leaf path (constraint verification) |
| `n_leaves()` | `int` | terminal node count |
| `profile()` | `list[dict]` | per-**depth** fit diagnostics: `depth, nodes, rows, cols, candidates, group_ns, scan_ns, leaf_ns` |

## Explaining predictions

| method | returns | notes |
|---|---|---|
| `break_down(row)` | `dict` | DALEX-style decomposition of one prediction: `{"baseline", "steps": [[column, contribution], ...], "residual", "prediction"}` |
| `feature_importance()` | `list[(name, share)]` | each column's share of total split gain (rpart's variable importance); sorted descending, shares sum to 1 |

**Break-down contract**: `baseline` is the training mean; each split on the
row's route contributes (child node mean − parent node mean) to its column in
decision-path order; `residual` is the within-leaf model adjustment (0 for
mean leaves). They telescope exactly:

```
baseline + sum(contributions) + residual == prediction
```

Routing in `break_down` is identical to `predict` (inclusive threshold
comparison; missing cells follow the split's learned missing direction), so
the explanation always matches the prediction.

## Constraints

- `monotone[j] = +1` → predictions never decrease in column `j` (`-1`: never
  increase). Enforced at split choice **and** in leaf coefficients (linear
  leaves carry effects in their coefficients; split checks alone don't fire
  when there are no splits).
- `interaction_groups` → every root-to-leaf path splits only columns of one
  group (LightGBM-style interaction constraints).

Both are validated against reference implementations in
`examples/constraints_check.py` (R `gbm`'s `var.monotone`, LightGBM's
`interaction_constraints`).

## Inspection via text output

`print(model)` reproduces rpart's print byte-for-byte: binary-heap node
numbering, `2·depth` indent, `x< t` / `x>=t` labels, `format(signif(x,7))`
columns, `*` terminal markers. Node ids in `to_dot()` match the text print.

Next: [Metrics](metrics.md) to evaluate, or the
[examples](../examples/model-building.md) to see it all run.
