# Model building &amp; predictions

Every parameter and nuance the tree handles, on one big, tedious dataset — the bundled diamonds parquet (53,940 rows, mixed numeric/categorical, missing cells) — with deeper trees than the quickstart.

Script: [`examples/model_building.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/model_building.py>). Fits use the **shipped defaults** (linear leaves + small ridge — closed-form fast). Regularization gets its own dedicated example later.

```
diamonds: 53940 rows x 26 cols (deeper trees below)
baseline depth=8                       rmse=     577.0  r2=      0.979  leaves= 140
deeper depth=12                        rmse=     598.6  r2=      0.977  leaves= 370
coarser leaves min_rows=500            rmse=     555.6  r2=      0.980  leaves=  57
leaf="mean"                            rmse=     884.5  r2=      0.950  leaves= 150
leaf="linear", leaf_cols="path"        rmse=    1137.5  r2=      0.918  leaves= 127
linear, l1=0 l2=0 (no ridge!)          rmse=  798670.6  r2= -40499.654  leaves=  21
row weights (2x carat<1)               rmse=     884.5  r2=      0.950  leaves= 150
winnow kept 15/26 columns (dropped: [1, 7, 9, 10, 11, 12, 21, 22, 23, 24, 25])
min_relevance=0.05 (winnowed fit)      rmse=     898.1  r2=      0.949  leaves= 134
monotone +1 on carat                   rmse=     601.8  r2=      0.977  leaves= 139
interaction_groups [[carat],[rest]]    rmse=     583.7  r2=      0.978  leaves= 138
predictions with missing carat: [1175.4, -1371.3, 1042.0]
```

## Growth knobs — `max_depth`, `min_rows`, `min_gain`

Deeper is not better: depth 12 chases noise (598.6 vs 577.0), and **coarser leaves win outright** (555.6 with `min_rows=500`) — on 54k rows the tree prefers fewer, steadier leaves. `min_gain` is the numerical noise floor.

## Leaf models — and why the default keeps ridge

`leaf="mean"` fits piecewise constants (884.5); `leaf="linear"` fits piecewise slopes and wins here (577.0). `leaf_cols="path"` restricts leaf regressions to the columns used on the path — fewer coefficients, less variance, more bias (1137.5).

Then the important one: **`l1=0, l2=0` explodes** (r² = −40499). Pure-OLS leaves on raw, collinear scales produce wild coefficients and nine-million-dollar predictions. The small default ridge is not cosmetic — it is load-bearing, and it is closed-form (fast). This is exactly why regularization earns its own upcoming example.

## Row weights

`fit(x, y, w)` — upweighting small stones (`2x carat<1`) shifts the fitted leaf values toward the weighted rows without touching the structure.

## Winnowing — `min_relevance` / `winnow()`

```
res = winnow(xtr, ytr, min_relevance=0.05)   # {"kept": [...], "relevance": [...]}
```

Univariate relevance screening drops 11 of 26 columns (mostly one-hot levels) before the split search; the winnowed fit keeps its accuracy on half the columns (898.1). Same screen runs inside `fit` when `min_relevance` is set.

## Monotone constraints

A domain fact: **price never falls as carat rises**. `monotone=[…, +1, …]` enforces it at split choice *and* in leaf coefficients — costing almost nothing here (601.8 vs 577.0) because the fact is true.

## Interaction groups

`interaction_groups=[[carat…], [rest…]]` — every root-to-leaf path may split only one group (LightGBM-style interaction constraints). Small cost (583.7), bounded interaction structure.

## Missing values are first-class

Predicting with a missing `carat` cell just works — the learned NA direction and indicator terms answer:

```
predictions with missing carat: [1175.4, -1371.3, 1042.0]
```

(The negative one is honest linear-leaf extrapolation in an unobserved direction, not a crash — see [Inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) for the leaf coefficients behind it.)

## Distributional leaves — `leaf_family`

Binary targets and counts get a family instead of a Gaussian mean: `leaf_family="binomial"` (logit link) for 0/1 targets and `"poisson"` (log link) for nonnegative counts or rates. Pair them with `gain="deviance"` so the split search scores the fitted likelihood, not squared error. Predictions stay on the response scale — `predict_proba` returns `P(class 1)`, and `cdf(x, value)` returns the fitted distribution's `P(Y <= value)` (distinct from the empirical `quantile`).

Script: [`examples/leaf_family.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/leaf_family.py>).

```
binomial P(1): [0.25, 0.75]
binomial P(Y <= 0): [0.75, 0.25]
poisson means: [1.0, 2.0, 4.0]
poisson P(Y <= 0): [0.367879, 0.018316]
categorical + NA P(1): [0.0, 1.0, 1.0]
```

The exact grouped MLEs recover the planted log-odds and log-rates; categorical levels and missing values route through the same native handlers as Gaussian trees. Linear coefficients are on the **link scale** (logit / log); `predict`/`predict_proba` convert back to response means.

Next: [Inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) for the metrics and structure reports, or [Visualization](<https://jkapila.github.io/mltreelib/examples/visualization/index.md>) to draw the tree.
