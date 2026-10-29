# Model build & Tree Visualization

How to build a model, read its output, draw it, and steer it with constraints
and regularization. Scripts:
[`examples/model_viz_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/model_viz_demo.py),
[`examples/constraints_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/constraints_demo.py),
[`examples/weights_regularization.py`](https://github.com/jkapila/mltreelib/blob/main/examples/weights_regularization.py).

## Read the tree (rpart-style text)

`print(model)` reproduces rpart's layout: binary-heap node numbering,
two-space depth indent, `x< t` / `x>=t` split labels, deviance and fitted
value columns, `*` on leaves.

```text
n= 300

node), split, n, deviance, yval
      * denotes terminal node

 1) root 300 10020.240000000 6.84000000
   2) x0< 0.525 165 1403.247272727 2.32727273
     4) x1< 0.5769231 102 0.085525490 0.03549020
       8) x0< 0.075 19 0.001894737 0.01052632 *
       9) x0>=0.075 83 0.069079518 0.04120482 *
     5) x1>=0.5769231 63 0.056088889 6.03777778
      10) x0< 0.125 17 0.004800000 6.02000000 *
      11) x0>=0.125 46 0.043930435 6.04434783 *
   3) x0>=0.525 135 1149.933333333 12.35555556
      ...
```

## Inspect it programmatically

```python
model.n_leaves()       # 8
model.splits()         # [('x0', 0.075), ('x0', 0.125), ('x1', 0.577), ...]
model.split_paths()    # split column names per root->leaf path
model.leaves()         # per leaf: cols, coef, na_coef, intercept, n, mean,
                       #           std, min, max, quantiles
```

```text
split paths (column names per root-to-leaf path):
    ['x0', 'x1', 'x0']
    ...
leaf n=19  yval=0.011
leaf n=83  yval=0.041
leaf n=17  yval=6.020
...
```

## Draw it (`to_svg`)

`model.to_svg()` returns a self-contained SVG — one box per node (split
condition, `n`, deviance or fitted value), edges labeled with `< t` / `>= t`.
Write it to a file or inline it in a notebook.

## Monotone constraints

`monotone=[+1, 0, -1, …]` per column pins the direction of the response —
enforced at split choice **and** in leaf coefficients (linear leaves carry
effects in their coefficients).

```python
model = ModelTree(monotone=[1, 0])
```

```text
unconstrained        predictions non-decreasing along x0: False
monotone +1 on x0    predictions non-decreasing along x0: True
```

## Interaction groups

`interaction_groups=[[0, 1], [2]]` — every root-to-leaf path may split only
columns of one group (LightGBM-style interaction constraints). Verify with
`split_paths()`:

```text
unconstrained        paths mixing groups: 2
groups [[0,1],[2]]   paths mixing groups: 0
```

Both constraint types are validated against reference implementations (R
`gbm`'s `var.monotone`, LightGBM's `interaction_constraints`) in
`examples/constraints_check.py`.

## Row weights & regularization

`fit(x, y, w)` accepts row weights (0 = row ignored). Upweighting the top of
`x0` moves the fitted leaf values where the weight is:

```text
unweighted leaf means: [1.25, 2.26, 3.27, 4.26]
weighted leaf means:   [1.25, 2.26, 3.47, 4.43]
```

`l1`/`l2` penalize leaf coefficients. With two collinear predictors the three
personalities are textbook:

```text
ridge  (l2)  leaf coefficients: [1.472, 1.469]   # shares the effect
lasso  (l1)  leaf coefficients: [2.8, 0.0]       # picks one
elastic      leaf coefficients: [0.933, 0.933]   # in between
```

## Relevance screening

`min_relevance` screens columns before the split search — cheap insurance on
wide data (drop pure noise):

```text
min_relevance=0.05 -> splits use columns ['x0', 'x1'] (noise column 2 gone: True)
```

Next: [Benchmarks — ELO](../benchmarks/elo.md) to see what all of this is
worth on real datasets, or the [API reference](../reference/data-model.md)
for the full parameter list.
