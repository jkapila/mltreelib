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
    # Leaf & family (or pass control=LeafControl(...))
    leaf="linear", leaf_family="gaussian", tau=0.5, l1=0.0, l2=0.5,
    leaf_cols="all", leaf_digest=True,
    control=None, leaf_control=None,
    # Categorical
    cat_order=None, cat_search="cumulative",
    # Constraints & weights
    tree_monotone=None, leaf_monotone=None, col_weights=None,
    interaction_groups=None,
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

### Two-Control Architecture: `LeafControl`

Leaf models can be independently configured or retrained via `LeafControl`:

```
from mltreelib import ModelTree, LeafControl

# Define leaf model controls separately:
leaf_ctrl = LeafControl(
    leaf="linear",           # "linear", "mean", "median", "quantile", "huber", "mars", "prob", "odds", "log_odds"
    leaf_family="gaussian",   # "gaussian", "binomial", "poisson", "probit", "gamma", "tweedie", "tobit", "softmax"
    l1=0.0, l2=1e-3,
    leaf_cols="all",         # "all" or "path_used"
    leaf_monotone=[1, -1],   # local slope sign constraints per feature
    mars_degree=1, mars_terms=10,
)

# Pass control directly to constructor without parameter bloat:
model = ModelTree(max_depth=4, control=leaf_ctrl)
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
| `fit(x, y, w=None, dtypes=None, ...)` | `self` | rows of `float`/`str`/`None` (`None` = missing); `w` / `sample_weight` / `row_weights` = row weights; `col_weights` = column split priority; `dtypes` = per column `"numeric"`/`"categorical"` |
| `predict(x, kind="response")` | `list[float]` or `list[dict]` | `kind="response"` (default), `kind="prob"`, `kind="odds"`, `kind="log_odds"`, `kind="median"`, or `kind="stats"` (full 0D stats dict per row) |
| `predict_proba(x)` | `list[float]` or `matrix` | response probability P(Y=1) for binomial/probit, or (N, C) probability matrix for softmax |
| `cdf(x, value)` | `list[float]` | conditional fitted-distribution CDF P(Y \<= value) for binomial/Poisson |
| `quantile(x, tau)` | `list[float]` | empirical conditional τ-quantile from leaf t-digests (`leaf_digest=True`) |

## Reading the model &amp; 0D Leaf Formulations

| method | returns | notes |
| --- | --- | --- |
| `__repr__()` | `str` | rpart-style text (`print(model)`); refuses to print an unfitted tree |
| `to_dot()` | `str` | Graphviz DOT — render with pygraphviz/graphviz (layout is graphviz's job) |
| `splits()` | `list[dict]` | `{col, threshold, missing_left, gain}` per split (`col` is the column **name**) |
| `leaves()` | `list[dict]` | one flat dict per leaf: `cols, coef, na_coef, intercept, n, mean, std, var, median, mad, iqr, min, max, skew, kurtosis, prob, odds, log_odds, majority_class, entropy, gini, zero_fraction, dispersion, stats, quantiles` |
| `leaf_stats()` | `list[dict]` | list of 0D stats dictionaries across all leaves: `{mean, std, var, median, mad, iqr, min, max, skew, kurtosis, prob, odds, log_odds, majority_class, entropy, gini, zero_fraction, dispersion}` |
| `split_paths()` | `list[list[str]]` | split column names per root→leaf path (constraint verification) |
| `n_leaves()` | `int` | terminal node count |
| `profile()` | `list[dict]` | per-**depth** fit diagnostics: `depth, nodes, rows, cols, candidates, group_ns, scan_ns, leaf_ns` |

## Constraints &amp; Weights

- **Dual Monotonicity**:
- `tree_monotone`: Enforces directional ordering on split child target means across tree splits: $(\\bar{y}\_L - \\bar{y}\_R) \\cdot m \\le 0$.
- `leaf_monotone` (or in `LeafControl`): Enforces sign bounds on local linear leaf slopes ($\\beta\_j \\ge 0$ for $+1$, $\\le 0$ for $-1$).
- `monotone`: Sets both tree and leaf monotonicity simultaneously.
- **Interaction Constraints (`interaction_groups`)**:
- Restricts root-to-leaf paths to features within the same group (LightGBM-style).
- Supports column names (e.g. `[["age", "income"], ["city", "zip"]]`) as well as numeric column indices.
- **Weights (Row &amp; Column)**:
- **Row weights**: Pass via `w`, `sample_weight`, or `row_weights` to `fit()` or `retrain()`.
- **Column weights**: `col_weights` scales candidate split gains ($gain \\times W\_j$) and weights random feature subset sampling during `col_sample`.

Both are validated against reference implementations in `examples/constraints_check.py` (R `gbm`'s `var.monotone`, LightGBM's `interaction_constraints`).

## Inspection via text output

`print(model)` reproduces rpart's print byte-for-byte: binary-heap node numbering, `2·depth` indent, `x< t` / `x>=t` labels, `format(signif(x,7))` columns, `*` terminal markers. Node ids in `to_dot()` match the text print.

Next: [Metrics](<https://jkapila.github.io/mltreelib/reference/metrics/index.md>) to evaluate, or the [examples](<https://jkapila.github.io/mltreelib/examples/model-building/index.md>) to see it all run.
