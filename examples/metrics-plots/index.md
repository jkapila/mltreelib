# Metrics &amp; plots

Every evaluation surface and every chart, demonstrated on real data (bank churn, with the tree's predictions as scores). Script: [`examples/metrics_showcase.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/metrics_showcase.py>).

```
pip install mltreelib[plotting]        # matplotlib + pygraphviz + seaborn
uv run python examples/metrics_showcase.py
```

## The full surface, in one run

```
churn: 2000 test rows, event rate 0.205

== reports ==
regression: rmse 0.328  r2 0.342  deciles 10
classification: accuracy 0.851  macro f1 0.734  confusion [[1515.0, 75.0], [223.0, 187.0]]
regression_by_group: ['high', 'low']
classification_by_group: {'high': 0.75, 'low': 0.952}

== curves ==
roc: auc 0.849 over 1799 points
pr:  ap 0.684
ks:  0.554 at score 0.174 (tpr 0.79 / fpr 0.23)
gain/decile 1: capture 0.390  lift 3.90
calibration: 10 non-empty bins

== probability & margin scores ==
log_loss 0.441  brier 0.107  hinge 0.417

== classification scorecard ==
  accuracy=0.851  balanced_accuracy=0.704  precision_macro=0.793  recall_macro=0.704  f1_macro=0.734  precision_micro=0.851  recall_micro=0.851  f1_micro=0.851  precision_weighted=0.839  recall_weighted=0.851  f1_weighted=0.838  mcc=0.489  jaccard_macro=0.611  jaccard_micro=0.741  jaccard_weighted=0.743  hamming_loss=0.149  zero_one_loss=0.149

== multilabel ==
  subset_accuracy 1.000  hamming 0.000  lrap 1.000  coverage 1.83

== agreement ==
  weighted_kappa(none): 0.472
  weighted_kappa(linear): 0.472
  weighted_kappa(quadratic): 0.472

== stability ==
  PSI (score halves): 0.0104
  CSI (age train vs test): 0.0055

== double lift (tree vs mean baseline) ==
  bucket 1: n=400 actual=0.593 tree=0.632 base=0.205
  bucket 2: n=400 actual=0.240 tree=0.224 base=0.205
  bucket 3: n=400 actual=0.115 tree=0.103 base=0.205
  bucket 4: n=400 actual=0.048 tree=0.049 base=0.205
  bucket 5: n=400 actual=0.030 tree=0.008 base=0.205

== pairwise distances ==
  euclidean    [[5.0, 0.0], [3.606, 1.414]]
  manhattan    [[7.0, 0.0], [5.0, 2.0]]
  chebyshev    [[4.0, 0.0], [3.0, 1.0]]
  minkowski    [[5.0, 0.0], [3.606, 1.414]]
  cosine       [[1.0, 1.0], [0.01, 1.0]]
  correlation  [[1.0, 1.0], [1.0, 1.0]]
  hamming      [[1.0, 0.0], [1.0, 1.0]]
  jaccard      [[1.0, 0.0], [0.0, 1.0]]
```

## The chart surface

All sixteen charts from `mltreelib.plots` render as click-to-zoom figures in [Visualization Reference](<https://jkapila.github.io/mltreelib/reference/plots/index.md>):

| chart | what it answers |
| --- | --- |
| `plot_roc` / `plot_pr` | discrimination; PR is the honest view on class imbalance |
| `plot_cumulative_gain` / `plot_lift` | targeting efficiency: "top decile captures X% at Y× lift" |
| `plot_ks` | maximum separation distance between classes + Score@KS |
| `plot_decile` | scorecard decile profile: event rate bars + cumulative capture |
| `plot_calibration` / `plot_response` | probability reliability vs. diagonal and decile response |
| `plot_double_lift` | head-to-head model comparison along one ranking (CAS) |
| `plot_psi` | score/feature stability and population shift between two periods |
| `plot_confusion_matrix` | classification error distribution across class boundaries |
| `plot_residuals` | regression fit diagnostics: actual vs. predicted and equality line |
| `plot_importance` | variable importance shares of total split gain |
| `plot_breakdown` | DALE-style waterfall attribution of one observation |
| `plot_variance_shares` | realized variance shares against ground truth |
| `plot_decomposition` | 4-panel additive time-series decomposition (trend/season/residual) |

Each chart takes `ax=` and returns Axes, so any panel composes. The library itself never imports matplotlib — this layer is opt-in via the `[plotting]` extra. Every number on this page comes from the script; the API shapes are in [Metrics](<https://jkapila.github.io/mltreelib/reference/metrics/index.md>).
