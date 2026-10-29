# API reference — Evaluation reports

What `Metrics` returns, field by field — the contract every learner is
scored under. Reports are plain dicts; missing pairs are dropped from the
math and counted in `n_missing`, never hidden.

## `Metrics().regression(y_true, y_pred)` → dict

| field | meaning |
|---|---|
| `n`, `n_missing` | observed pairs; dropped missing pairs |
| `mean_error` | mean(actual − fitted) — signed |
| `bias` | mean_pred / mean_true (1.0 = unbiased) |
| `mae`, `mad` | mean / median absolute deviation |
| `mape` | absolute % error, non-zero targets only |
| `rmse`, `r2`, `max_error` | the usual |
| `explained_variance` | 1 − Var(error)/Var(y) |
| `deciles` | per-target-decile error profile — the DAR "bathtub" view |

## `Metrics().classification(y_true, y_pred, probabilities=None)` → dict

| field | meaning |
|---|---|
| `n`, `n_missing`, `n_classes` | observed pairs; classes seen |
| `accuracy` | overall accuracy |
| `macro_precision/recall/f1` | unweighted mean over classes |
| `per_class` | precision/recall/f1/support per class |
| `confusion` | counts matrix |
| `auc`, `logloss`, `brier` | only with `probabilities` (P(class 1)) |

## Grouped reports

`classification_by_group(keys, ...)` / `regression_by_group(keys, y_true,
y_pred)` return **one report per group**, as a dict keyed by the group
label — the format [ModelTree's group API](model.md) consumes.

Real values for every field on real data: the
[Metrics & plots showcase](../examples/metrics-plots.md) and the
[Metrics page](metrics.md) example.

[Full generated listing](api.md) · next: [Visualization](plots.md)
