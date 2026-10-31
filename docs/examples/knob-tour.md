# Growth knob tour

Every growth knob on `ModelTree` with real, deterministic output.

Script: [`examples/knob_tour.py`](https://github.com/jkapila/mltreelib/blob/main/examples/knob_tour.py) (`uv run --with pandas --with pyarrow python examples/knob_tour.py`).

```text
== leaf kinds (skewed target with one outlier) ==
  mean      leaves=2   pred(x=0)=  2.0000
  median    leaves=2   pred(x=0)=  1.9999
  huber     leaves=2   pred(x=0)=  2.0000
  quantile  leaves=2   pred(x=0)=  2.0037
  linear    leaves=2   pred(x=0)=  1.9999
  digest    q10=1.996 q50=2.000 q90=2.004 (any tau post-fit)
== criterion family (planted regime at x=0.5) ==
  sse            leaves=2   pred(0.1)=-0.000 pred(0.9)=5.000
  friedman_mse   leaves=2   pred(0.1)=-0.000 pred(0.9)=5.000
  absolute_error leaves=2   pred(0.1)=-0.000 pred(0.9)=5.000
  huber          leaves=2   pred(0.1)=-0.000 pred(0.9)=5.000
== classification criteria (0/1 labels, same break) ==
  entropy    leaves=2   train_acc=100.000
  gini       leaves=2   train_acc=100.000
  gain_ratio leaves=2   train_acc=100.000
  twoing     leaves=2   train_acc=100.000
  sdr        leaves=2   train_acc=100.000
== categorical levels-as-is (level codes 0..4) ==
  frequency_desc/cumulative: leaves=5   splits=[('x0', 0.0), ('x0', 0.0), ('x0', 0.0)]
  target_mean/single_vs_rest: leaves=4   splits=[('x0', 0.0), ('x0', 0.0), ('x0', 0.0)]
== split_precision grid (exact scan vs 0.1 grid) ==
  precision=None  thresholds=[0.497]
  precision=0.1   thresholds=[0.4, 0.5]
== per-node column subsets (col_sample + seed) ==
  col_sample=None  seed=42 leaves=4   top_importance=[('x3', 1.0)]
  col_sample=0.3   seed=42 leaves=1   top_importance=[('x0', 0.0)]
  col_sample=0.3   seed=42 leaves=1   top_importance=[('x0', 0.0)]
  col_sample=0.3   seed=7 leaves=1   top_importance=[('x0', 0.0)]
== cost pruning (prune=none|bic|mdl) ==
  prune=none leaves=10
  prune=bic  leaves=2
  prune=mdl  leaves=10
== Chow significance gate (split_test='chow') ==
  pure noise     -> leaves=1 (gate must stop)
    node depth=0 rows=400 cols=1 candidates=19 scan_ms=0 leaf_ms=0 chow_f=2.4940 chow_p=0.1151 chow_alpha=0.0000 chow=stop
  planted regime -> leaves=2 (gate must pass)
    depth 0: nodes=1 rows=200 candidates=199 scan_ms=0
    depth 1: nodes=2 rows=200 candidates=198 scan_ms=0
    scan: gain=Sse split_precision=Some(0.001) cat_search=Cumulative (FrequencyDesc) leaf=Mean leaf_digest=true
    node depth=0 rows=200 cols=1 candidates=199 scan_ms=0 leaf_ms=0 chow_f=40019076.5361 chow_p=0.0000 chow_alpha=0.0003 chow=pass
    node depth=1 rows=100 cols=1 candidates=99 scan_ms=0 leaf_ms=0 chow_f=2.1184 chow_p=0.1487 chow_alpha=0.0005 chow=stop
    node depth=1 rows=100 cols=1 candidates=99 scan_ms=0 leaf_ms=0 chow_f=5.7155 chow_p=0.0187 chow_alpha=0.0005 chow=stop
== monotone constraint (y increases with x) ==
  monotone=[]   preds=[-0.0003184, 5.000025000000002, 5.000025000000002]
  monotone=[1]  preds=[-0.0003184, 5.000025000000002, 5.000025000000002]
== split_cols (split-search column subset; leaves stay full) ==
  split_cols=None   split_columns=['x0', 'x1'] leaves=6
  split_cols=[1]    split_columns=['x1'] leaves=6
== honesty (structure/estimation split; grf-style fairness) ==
  honesty=False frac=0.5: leaves=2 preds=[-0.0, 4.999]
  honesty=True  frac=0.5: leaves=2 preds=[-0.001, 5.078]
  honesty=True  frac=0.2: leaves=2 preds=[-0.0, 5.061]
== NA surrogate splits (maxsurrogate; rpart lineage) ==
  maxsurrogate=0: pred(x0=None, x1=0.1/0.9) = [0.22, 0.22]
  maxsurrogate=1: pred(x0=None, x1=0.1/0.9) = [0.22, 5.0]
== cost-complexity pruning (prune='cp'; cp_alpha sweep) ==
  cp_alpha=0.0   : leaves=15
  cp_alpha=0.5   : leaves=5
  cp_alpha=5.0   : leaves=5
  cp_alpha=1000000000.0: leaves=1
== knob batch: leaf-cap / min trio / purity gate / preselect family ==
  max_leaf_nodes=4: leaves=4 (<= 4)
  minsplit=200:     leaves=3
  purity_gate=0.5:  leaves=5
  preselect=none        split_columns=['x0']
  preselect=winnow      split_columns=['x0']
  preselect=boruta      split_columns=['x0']
  preselect=shadowmax   split_columns=['x0']
  preselect=variance    split_columns=['x0']
  preselect=mutual_info split_columns=['x0']
  preselect=unitree     split_columns=['x0']
```

## Significance Testing Gate — `split_test="chow"`

The Chow structural break test evaluates whether splitting a node into two children yields a statistically significant parameter shift over the pooled node model:

```python
model = ModelTree(split_test="chow", split_alpha=0.05, bonferroni=True, verbose=2)
model.fit(x, y)
```

- When fitting pure noise, `chow=stop` halts growth immediately (1 leaf), avoiding false splits without needing aggressive manual tuning.
- `bonferroni=True` divides `split_alpha` by the number of candidate cuts tested at the node, correcting for multiple testing.

## Honesty — `honesty=True`

In the spirit of generalized random forests (`grf`), honesty splits the training rows into two independent samples:
- **Structure rows** (`honesty_fraction=0.5` by default) determine split candidate scoring, significance gating, and tree structure.
- **Estimation rows** determine leaf parameters and fitted responses.

This prevents structure search from overfitting leaf estimates.

## NA Surrogate Splits — `maxsurrogate`

`maxsurrogate=k` (following `rpart` lineage) stores up to $k$ surrogate splits at each internal node that mimic the primary partition. When predicting a test row missing the primary split feature, the primary surrogate routes the row instead of defaulting to a static branch.

## Feature Screening — `preselect`

Before tree growth begins, candidate features can be filtered via screening:
- `"winnow"`: Univariate linear correlation screening.
- `"boruta"`: Comparison against shadow randomized permutations.
- `"unitree"`: Univariate shallow surrogate trees scored by explained variation $R^2$.
