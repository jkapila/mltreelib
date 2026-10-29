# Inspection

Metrics reports and every structural view of a fitted tree — the numbers behind the model. Same big diamonds dataset as the model-building example.

Script: [`examples/inspection.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/inspection.py>).

## Metrics reports

```
reg = Metrics().regression(y_test, pred)
```

```
regression report (test set):
  n                    10788.000
  rmse                 577.033
  mae                  284.464
  mad                  284.508
  r2                   0.979
  bias                 0.999
  max_error            9181.634
  explained_variance   0.979
  decile calibration (first 3):
    decile 1: n=1078  mean_error=-13.6  mae=39.1
    decile 2: n=1079  mean_error=-15.7  mae=46.1
    decile 3: n=1079  mean_error=7.5  mae=62.0
```

The decile table is calibration, not just accuracy: it buckets predictions into tentiles and reports the error inside each — a model that is right on average but biased at the top end shows up here immediately.

(`Metrics().classification(y_true, y_pred)` gives the classification counterpart: accuracy, macro precision/recall/F1, per-class table, confusion matrix.)

## Structure

```
model.n_leaves()       # terminal node count
model.splits()         # [{col, threshold, missing_left}, ...]
model.split_paths()    # split column names per root->leaf path
model.leaves()         # per leaf: cols, coef, na_coef, intercept, n,
                       #           mean, std, min, max, quantiles
model.profile()        # per-depth fit diagnostics (nodes, rows, candidates,
                       #                          group/scan/leaf timings)
```

```
n_leaves: 140
splits: 139 total; first 5:
  x6  < 0.500  missing_left=True
  x3  < 4.025  missing_left=True
  x3  < 4.005  missing_left=True
  x3  < 4.215  missing_left=True
  x2  < 56.500  missing_left=True

split_paths() - split columns along each root-to-leaf path:
   x4 -> x0 -> x3 -> x5 -> x0 -> x4 -> x3 -> x6
   x4 -> x0 -> x3 -> x5 -> x0 -> x4 -> x3 -> x3
   ...

first leaf: n=304  mean=474.8  std=67.4
  range: [326.0, 710.0]  quantiles: [381, 419, 484, 529, 552]
  linear coefs on: ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']  intercept=-860.2

fit profile: 9 depth levels; slowest depth's split scan: 10238.2 ms
```

`splits()` and `split_paths()` report **column names** — the tree knows its columns even when you don't pass names (they default to `x0…xN`). Note the split columns' `missing_left=True`: missing values learned their own routing direction per split.

## Explaining one prediction (break-down)

`break_down(row)` decomposes a prediction along its decision path — baseline (training mean) + per-column contributions + residual, telescoping exactly to the prediction. Columns can repeat along a path (a feature may gate several splits):

```
break_down of one row: baseline 3936.3 -> prediction 1779.7 (8 steps)
  x4     -2299.7
  x0     +851.1
  x0     -309.9
  x0     -465.3
  x14    +189.9
  x0     -85.0
  ... 2 more steps
  residual -166.6  (sum check: 1779.7)
```

## Feature importance

`feature_importance()` — each column's share of total split gain (rpart's variable importance):

```
feature importance (top 5 of 26):
  x4     0.707
  x0     0.124
  x20    0.040
  x23    0.032
  x14    0.028
```

## The tree text itself

`print(model)` is the full rpart-format listing (see the [Visualization](<https://jkapila.github.io/mltreelib/examples/visualization/index.md>) page for the diagram of the same tree):

```
n= 43152

node), split, n, deviance, yval
      * denotes terminal node

  1) root 43152 6.885623e+11 3936.2915
    2) x4< 6.275 27855 3.628925e+10 1636.6104
      4) x0< 0.465 13953 7.944035e+08 788.6131
```

Next: [Visualization](<https://jkapila.github.io/mltreelib/examples/visualization/index.md>) to draw it, or [Benchmarks — ELO](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) to see it against the field.
