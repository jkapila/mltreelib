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
regression: rmse 0.420  r2 -0.084  deciles 10
classification: accuracy 0.796  macro f1 0.648  confusion [[1444.0, 146.0], [262.0, 148.0]]
regression_by_group: ['high', 'low']
classification_by_group: {'high': 0.709, 'low': 0.883}

== curves ==
roc: auc 0.676 over 1554 points
pr:  ap 0.386
ks:  0.361 at score 0.183 (tpr 0.60 / fpr 0.24)
gain/decile 1: capture 0.241  lift 2.41
calibration: 10 non-empty bins

== probability & margin scores ==
log_loss 3.204  brier 0.177  hinge 0.506

== classification scorecard ==
  accuracy=0.796  balanced_accuracy=0.635  precision_macro=0.675  recall_macro=0.635  f1_macro=0.648  precision_micro=0.796  recall_micro=0.796  f1_micro=0.796  precision_weighted=0.776  recall_weighted=0.796  f1_weighted=0.783  mcc=0.307  jaccard_macro=0.523  jaccard_micro=0.661  jaccard_weighted=0.674  hamming_loss=0.204  zero_one_loss=0.204

== multilabel ==
  subset_accuracy 1.000  hamming 0.000  lrap 1.000  coverage 1.83

== agreement ==
  weighted_kappa(none): 0.301
  weighted_kappa(linear): 0.301
  weighted_kappa(quadratic): 0.301

== stability ==
  PSI (score halves): 0.0143
  CSI (age train vs test): 0.0055

== double lift (tree vs mean baseline) ==
  bucket 1: n=400 actual=0.443 tree=0.748 base=0.205
  bucket 2: n=400 actual=0.223 tree=0.186 base=0.205
  bucket 3: n=400 actual=0.120 tree=0.103 base=0.205
  bucket 4: n=400 actual=0.060 tree=0.041 base=0.205
  bucket 5: n=400 actual=0.180 tree=0.003 base=0.205

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

All ten charts from `mltreelib.plots`, one figure (click to zoom):

| chart | what it answers |
| --- | --- |
| `plot_roc` / `plot_pr` | discrimination; PR is the honest view on imbalance |
| `plot_cumulative_gain` / `plot_lift` | targeting: "top decile captures X% at Y× lift" |
| `plot_ks` | separation + Score@KS |
| `plot_decile` | scorecard profile: event rate bars + cumulative capture |
| `plot_calibration` / `plot_response` | can the probabilities be trusted? |
| `plot_double_lift` | head-to-head model comparison along one ranking (CAS) |
| `plot_psi` | score/feature stability between two periods |

Each chart takes `ax=` and returns Axes, so panels compose like above. The library itself never imports matplotlib — this layer is opt-in via the `[plotting]` extra. Every number on this page comes from the script; the API shapes are in [Metrics](<https://jkapila.github.io/mltreelib/reference/metrics/index.md>).
