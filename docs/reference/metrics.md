# API reference — Metrics

`Metrics` — evaluation reports, curves, scorecards, stability and distances.
Everything returns plain structures (numeric series + summary scalars);
**the library never plots** — rendering lives in `mltreelib.plots` (matplotlib
as an optional extra) and the [Visualization](../examples/visualization.md)
example. Charts on these docs are click-to-zoom.

Unless noted, classification inputs take **scores/probabilities** with
positive class `y > 0.5`; ties sweep together in stable order.

## Reports

- `Metrics().regression(y_true, y_pred)` → `n, n_missing, mean_error, bias,
  mae, mad, mape, rmse, r2, max_error, explained_variance`, plus a `deciles`
  table (`decile, n, mean_error, mae, mape`) — calibration by tentile, not
  just accuracy. Missing-aware.
- `Metrics().classification(y_true, y_pred)` — **integer class labels** —
  `n, n_missing, n_classes, accuracy, macro_precision/recall/f1, per_class`,
  `confusion` matrix.
- `Metrics().regression_by_group(keys, y_true, y_pred)` — regression report
  per group key.

## Ranking & calibration curves

| call | returns |
|---|---|
| `roc_curve(y, scores)` | `{"points": [[fpr, tpr], ...], "auc"}` — trapezoid over the empirical curve |
| `pr_curve(y, scores)` | `{"points": [[recall, precision], ...], "average_precision"}` — step-wise AP (sklearn convention) |
| `gain_curve(y, scores)` | `{"points": [[population_fraction, capture_rate], ...]}` |
| `lift_curve(y, scores)` | `{"points": [[population_fraction, lift], ...]}` — lift = capture / fraction |
| `ks_statistic(y, scores)` | `{"points": [[fpr, tpr, score], ...], "ks", "score_at_ks", "tpr_at_ks", "fpr_at_ks"}` |
| `decile_table(y, scores, n_buckets=10)` | `[{"decile", "n", "avg_score", "events", "event_rate", "cum_capture", "cum_event_rate", "lift"}]` — decile 1 = top scores (kds/PerfScoreCard profile; response + cumulative response at decile granularity) |
| `calibration(y, scores, n_bins=10)` | `{"bins": [{"left", "right", "mean_score", "observed", "n"}]}` — equal-width over [0, 1], edge-clipped, empty bins omitted |

## Probability & margin scores

- `log_loss(y, probs)` — mean log loss (probabilities clip to (1e-15, 1−1e-15)).
- `brier_score(y, probs)` — mean squared probability error.
- `hinge_loss(y, decision)` — mean hinge over signed margins (labels map to ±1).

## Classification scorecard

`Metrics().classification_scores(y_true, y_pred)` — **single-label binary or
multiclass**, labels = sorted distinct union of truth and predictions:

`n, n_classes, accuracy, balanced_accuracy, precision/recall/f1 ×
(macro|micro|weighted), mcc` (Gorodkin multiclass Matthews),
`jaccard × (macro|micro|weighted), hamming_loss, zero_one_loss`.

## Multilabel scorecard

`Metrics().multilabel_scores(y_true, y_scores)` — rows are instances, columns
labels, entries `> 0.5` positive; the raw entries double as ranking scores:

`subset_accuracy, hamming_loss, precision/recall/f1 × (micro|macro|samples),
label_ranking_ap, coverage_error`. Rows with no true labels are skipped in
the samples averages and ranking metrics.

## Agreement

`Metrics().weighted_kappa(y_true, y_pred, weights="quadratic"|"linear"|"none")`
— Cohen's kappa over the sorted distinct labels; `1` = perfect agreement
(PerfScoreCard's weighted kappas).

## Stability (scorecard monitoring)

`Metrics().stability_index(reference, current, n_bins=10)` →
`{"psi": float, "bins": [{"left", "right", "reference_share",
"current_share", "contribution"}]}` — the Population Stability Index
(equal-width bins over the reference range, zero shares floored at 1e-6).
CSI is the same call on a feature column instead of scores.

## Model comparison

`Metrics().double_lift(y_true, scores_a, scores_b, n_buckets=10)` →
`[{"bucket", "n", "actual", "pred_a", "pred_b"}]` — equal-count buckets by
model A's ranking (CAS monograph double-lift chart; the additive vs ratio
display is a plotting concern — `plots.plot_double_lift(model_type=...)`).

## Distances

`Metrics().pairwise_distances(a, b, metric="euclidean", p=2.0)` →
`len(a) × len(b)` matrix. Metrics: `euclidean`, `manhattan`, `chebyshev`,
`minkowski` (power `p`), `cosine`, `correlation`, `hamming`, `jaccard`
(binary presence). Zero-variance rows → correlation distance 1.0; zero
vectors → cosine distance 1.0; both-empty → jaccard 0.0.

## Plotting (`mltreelib.plots`)

Optional layer, installed with `pip install mltreelib[plotting]` (matplotlib
+ pygraphviz + seaborn; each function takes `ax=None`, returns Axes):
`plot_roc, plot_pr, plot_cumulative_gain, plot_lift, plot_ks, plot_decile,
plot_calibration, plot_response, plot_double_lift, plot_psi`.

```python
from mltreelib import Metrics, plots
r = Metrics().roc_curve(y_test, scores)
plots.plot_ks(y_test, scores)          # click the figure in these docs to zoom
```

Time-series metrics (MAPE/SMAPE variants, MASE, pinball for quantiles,
Tweedie/Poisson deviance with the GLM leaves) are parked with their feature
work.

Next: [Data](data.md) or [Model](model.md).
