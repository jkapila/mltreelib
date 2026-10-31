# API reference — Model

`ModelTree` — the tree: greedy splits by fitted-leaf SSE reduction, linear or mean leaves, NA-aware (learned split directions + NA-indicator leaf terms). Rust core is the source of truth; this page mirrors it.

## Constructor

```
ModelTree(
    # Growth & stopping
    max_depth=5, min_rows=50, minsplit=None, minbucket=None,
    minprob=0.02, max_leaf_nodes=None, min_gain=1e-9, purity_gate=None,
    # Search
    gain=None, split_precision=0.001, split_cols=None,
    col_sample=None, seed=42,
    # Leaf & family
    leaf="linear", leaf_family="gaussian", tau=0.5, l1=0.0, l2=0.5,
    leaf_cols="all", leaf_digest=True,
    # Categorical
    cat_order=None, cat_search="cumulative",
    # Constraints
    monotone=[], interaction_groups=[],
    # Balance (grf)
    alpha=0.0, imbalance_penalty=0.0,
    # Pruning
    prune="cp", cp_alpha=5.0,
    # Honesty & surrogates
    honesty=False, honesty_fraction=0.5, maxsurrogate=0,
    # Significance gates (MOB)
    split_test="none", split_alpha=0.05, bonferroni=False,
    # Screening
    preselect="none", min_relevance=None,
    # Diagnostics
    verbose=0,
)
```

The gain family is **sample-weight aware end to end**: weighted class masses for entropy/gini/gain\_ratio/twoing, weighted moments for sdr/SSE, fitted deviance for binomial/Poisson, and a custom `gain_fn(left, right)` receives lazy `(y, w)` pair slices of the two sides (higher = better). `gain="deviance"` scores the fitted leaf-family likelihood and is recommended for binomial and Poisson leaves.

Defaults are the arena bake-off winner (see [Benchmarks](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>)) — linear leaves at shallow depth with a small **ridge** (`l2`) penalty. `l1` (lasso/elastic) is available but off by default: coordinate descent is slow inside the split scan and the accuracy gain over ridge is marginal. Turning regularization fully off (`l1=l2=0`) on raw scales is a known blowup mode — keep the ridge unless you standardize your inputs.

## Split search — per dtype, raw data in

Categorical columns are **levels as-is** (`dtypes` declares `"categorical"`; string values and numeric level codes enter directly — identity, never order), NAs route as-is. Raw strings are ingested directly: the tree builds an internal identity dictionary on the fly without requiring an external encoder. Binary level partitions (`cat_search`) are built along `cat_order` (`"frequency_desc"`, `"target_mean"`, or custom); multiway partitions are reserved for a later build. `gain` picks the split-quality function: built-ins scan on summary stats or deviance; a custom `gain_fn(left_y, right_y) -> float` receives lazy y-slices per candidate. Numeric columns scan candidates on the `split_precision` grid (thresholds are grid values; prediction compares raw values), `None` restores the exact unique-value scan.

```
# Raw strings ingested directly in categorical columns:
m = ModelTree(max_depth=2, min_rows=2, cat_order="target_mean")
m.fit([["paris", 1.0], ["tokyo", 2.0], ["paris", 3.0], ["london", 4.0]],
      [10.0, 20.0, 15.0, 30.0], dtypes=["categorical", "numeric"])
m.predict([["paris", 1.5], ["london", 3.5]])
```

## Significance gates &amp; Honesty

- **`split_test="chow"`**: A Chow structural break F-test gates node splitting. The candidate split must demonstrate statistically significant parameter heterogeneity over the pooled node model at `split_alpha` (e.g. 0.05), optionally adjusted with `bonferroni=True` across evaluated split candidates.
- **`honesty=True`**: GRF-style sample splitting partition. `honesty_fraction` (default 0.5) of rows select tree structure and splits; the held-out estimation rows fit leaf parameters, yielding unbiased honest predictions.

## Distributional leaves — binomial &amp; Poisson

`leaf_family` switches the leaf distribution and link function: - `"gaussian"` (default): Identity link, least squares regression. - `"binomial"`: Logit link for 0/1 binary targets. Coefficients are on the log-odds scale; `predict` and `predict_proba` return response probabilities `P(Y=1)`. - `"poisson"`: Log link for nonnegative counts or rates. Coefficients are on the log-rate scale; `predict` returns expected counts.

Pair with `gain="deviance"` so split search maximizes deviance reduction under the family.

## Fitting &amp; predicting

| method | returns | notes |
| --- | --- | --- |
| `fit(x, y, w=None, dtypes=None)` | `self` | rows of `float`/`str`/`None` (`None` = missing); `w` = row weights; `dtypes` = per column `"numeric"`/`"categorical"`; strings in categorical columns ingested natively |
| `predict(x)` | `list[float]` | response-scale predictions (regression mean, binomial P(1), Poisson expected count); NAs route by learned NA direction |
| `predict_proba(x)` | `list[float]` | response probability P(Y=1) for `leaf_family="binomial"` |
| `cdf(x, value)` | `list[float]` | conditional fitted-distribution CDF P(Y \<= value) for binomial/Poisson |
| `quantile(x, tau)` | `list[float]` | empirical conditional τ-quantile from leaf t-digests (`leaf_digest=True`) |

## Reading the model

| method | returns | notes |
| --- | --- | --- |
| `__repr__()` | `str` | rpart-style text (`print(model)`); refuses to print an unfitted tree |
| `to_dot()` | `str` | Graphviz DOT — render with pygraphviz/graphviz (layout is graphviz's job) |
| `splits()` | `list[dict]` | `{col, threshold, missing_left, gain}` per split (`col` is the column **name**) |
| `leaves()` | `list[dict]` | one flat dict per leaf: `cols, coef, na_coef, intercept, n, mean, std, min, max, quantiles` |
| `split_paths()` | `list[list[str]]` | split column names per root→leaf path (constraint verification) |
| `n_leaves()` | `int` | terminal node count |
| `profile()` | `list[dict]` | per-**depth** fit diagnostics: `depth, nodes, rows, cols, candidates, group_ns, scan_ns, leaf_ns` |

## Explaining predictions

| method | returns | notes |
| --- | --- | --- |
| `break_down(row)` | `dict` | DALEX-style decomposition of one prediction: `{"baseline", "steps": [[column, contribution], ...], "residual", "prediction"}` |
| `feature_importance()` | `list[(name, share)]` | each column's share of total split gain (rpart's variable importance); sorted descending, shares sum to 1 |

**Break-down contract**: `baseline` is the training mean; each split on the row's route contributes (child node mean − parent node mean) to its column in decision-path order; `residual` is the within-leaf model adjustment (0 for mean leaves). They telescope exactly:

```
baseline + sum(contributions) + residual == prediction
```

Routing in `break_down` is identical to `predict` (inclusive threshold comparison; missing cells follow the split's learned missing direction), so the explanation always matches the prediction.

## Constraints

- `monotone[j] = +1` → predictions never decrease in column `j` (`-1`: never increase). Enforced at split choice **and** in leaf coefficients (linear leaves carry effects in their coefficients; split checks alone don't fire when there are no splits).
- `interaction_groups` → every root-to-leaf path splits only columns of one group (LightGBM-style interaction constraints).

Both are validated against reference implementations in `examples/constraints_check.py` (R `gbm`'s `var.monotone`, LightGBM's `interaction_constraints`).

## Inspection via text output

`print(model)` reproduces rpart's print byte-for-byte: binary-heap node numbering, `2·depth` indent, `x< t` / `x>=t` labels, `format(signif(x,7))` columns, `*` terminal markers. Node ids in `to_dot()` match the text print.

Next: [Metrics](<https://jkapila.github.io/mltreelib/reference/metrics/index.md>) to evaluate, or the [examples](<https://jkapila.github.io/mltreelib/examples/model-building/index.md>) to see it all run.
