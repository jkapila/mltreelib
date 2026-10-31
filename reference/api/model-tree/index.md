# API reference — ModelTree

The model tree: typed fit, split-search knobs, quantiles, training log. Guide: [Model](<https://jkapila.github.io/mltreelib/reference/model/index.md>).

## ModelTree

`pythonclass ModelTree((max_depth=5, min_rows=50, minsplit=None, minbucket=None, minprob=0.02, max_leaf_nodes=None, min_gain=1e-09, purity_gate=None, gain=None, split_precision=Ellipsis, split_cols=None, col_sample=None, seed=42, leaf='linear', tau=0.5, l1=0.0, l2=0.5, leaf_cols='all', leaf_digest=True, cat_order=None, cat_search='cumulative', monotone=Ellipsis, interaction_groups=Ellipsis, alpha=0.0, imbalance_penalty=0.0, prune='cp', cp_alpha=5.0, honesty=False, honesty_fraction=0.5, maxsurrogate=0, split_test='none', split_alpha=0.05, bonferroni=False, preselect='none', min_relevance=None, verbose=0, leaf_family='gaussian'))` Model tree: greedy splits with Gaussian, binomial-logit or Poisson-log leaves, NA-aware (learned split directions + NA-indicator leaf terms). Task: regression, binary probability estimation, or count modeling.

### ModelTree.break\_down

`pythonbreak_down((self, /, row))` DALEX-style break-down of one response-scale prediction.

Args: row: one row of predictors (`None` = missing).

Returns: `{"baseline", "steps": [[column_name, contribution], ...], "residual", "prediction"}`; baseline + steps + residual == prediction.

### ModelTree.cdf

`pythoncdf((self, /, x, value))` Fitted binomial/Poisson CDF P(Y \<= value), conditional on each row.

Args: x: raw rows of predictors, with the same missing/level handling as predict. value: response threshold; discrete CDFs use its integer floor.

Returns: one fitted-distribution probability per row. Unlike quantile, this uses the fitted family, not the empirical target t-digest.

Raises: ValueError for Gaussian leaves or an invalid threshold.

### ModelTree.feature\_importance

`pythonfeature_importance((self, /))` Feature importance - each column's share of the total split gain (rpart's variable importance).

Args: none.

Returns: list of `(column_name, share)` sorted descending; shares sum to 1.

### ModelTree.fit

`pythonfit((self, /, x, y, w=None, dtypes=None))` Fit the tree on row-major data using the selected leaf family.

Args: x: rows of predictors; `None` entries are missing (NA rows train). y: targets; `None`/NaN drops that row for binomial/Poisson. Gaussian NaN handling follows the historical Gaussian path. Binomial targets are 0 or 1; Poisson targets are finite nonnegative counts or rates (including fractional values). Gaussian targets retain ordinary regression. w: optional per-row sample weights (row weightage); zero-weight rows are ignored. dtypes: optional per-column types — "numeric" (default) or "categorical" (values are level codes; the split search partitions levels, no encoding).

Returns: nothing; the fitted tree replaces the inner model.

### ModelTree.leaves

`pythonleaves((self, /))` Fitted leaf models with their empirical target distribution stats.

Args: none.

Returns: one dict per leaf with `cols`, `coef`, `na_coef`, `intercept`, `response_constant`, `n`, `mean`, `std`, `min`, `max`, `quantiles`. Linear `coef`, `na_coef`, and `intercept` use the family's link scale (identity, logit, or log); `mean` and other stats describe raw targets. Constant leaves have no link-scale coefficient vector: `intercept` is `None`, while `response_constant` is the response-scale constant.

### ModelTree.n\_leaves

`pythonn_leaves((self, /))` Number of leaves.

Args: none.

Returns: leaf count.

### ModelTree.predict

`pythonpredict((self, /, x))` Predict response-scale means; NAs never block a prediction.

Args: x: rows of predictors; `None` entries are missing.

Returns: one mean per row (binomial P(Y=1), Poisson expected count, or Gaussian regression value). No thresholding into hard labels.

### ModelTree.predict\_proba

`pythonpredict_proba((self, /, x))` Predict binomial P(Y=1), one probability per input row.

Args: x: raw rows of predictors, with the same missing/level handling as predict.

Returns: one probability per row, not a two-column class matrix.

Raises: ValueError unless the fitted leaf family is binomial.

### ModelTree.profile

`pythonprofile((self, /))` Per-level timing profile (group/sort, scan, leaf fit).

Args: none.

Returns: one dict per depth level with `depth`, `nodes`, `rows`, `cols`, `candidates`, `group_ns`, `scan_ns`, `leaf_ns`.

### ModelTree.quantile

`pythonquantile((self, /, x, tau))` Empirical conditional target quantiles from the leaves' t-digests: any tau in \[0, 1\] after fitting. These are not quantiles of the fitted binomial/Poisson distribution; use cdf for that distribution's CDF.

Args: x: rows of predictors (NA never blocks). tau: quantile level.

Returns: one quantile per row (NaN when a leaf's digest was dropped).

### ModelTree.split\_paths

`pythonsplit_paths((self, /))` Split columns along each root-to-leaf path.

Args: none.

Returns: one list of column names per root-to-leaf path (the split columns on that path) — the introspection behind interaction-group checking.

### ModelTree.splits

`pythonsplits((self, /))` Learned splits.

Args: none.

Returns: one dict per split with `col`, `threshold`, `missing_left`, `gain` (goodness of split).

### ModelTree.to\_dot

`pythonto_dot((self, /))` Graphviz DOT for the tree (render with pygraphviz/graphviz — layout is graphviz's job): `pgv.AGraph(string=model.to_dot()).draw("tree.svg", prog="dot")`.

### ModelTree.training\_log

`pythontraining_log((self, /))` The training log (populated when `verbose > 0`): scan config, per depth/per node candidate counts, precision grid, timings.

Args: none.

Returns: log lines.
