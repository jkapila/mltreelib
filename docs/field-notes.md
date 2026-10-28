# Field notes: what it actually costs to just run a tree

Building the Elo arena meant wiring up every major tree/model-tree
implementation. The friction is the point of this page: `mltreelib` is one
import with zero transitive quirks, and *this* is what the alternatives cost.

## The hassle log (all real, all this week)

| what | what happened |
|---|---|
| `perpetual` (Rust GBM) | imports `sklearn` internally — install fails at runtime without scikit-learn, error surfaces only on first fit |
| `ctboost` | sklearn-style estimators refuse to import without scikit-learn ("ctboost scikit-learn estimators require scikit-learn") |
| `catboost` | 92MB wheel; on a slow link the whole benchmark blocks on the download |
| `forust` (Rust GBM) | regression objective is `objective_type="SquaredLoss"` — `"SquaredError"` raises ValueError; the name is only in the API docs |
| R `Cubist` | the function is `cubist()` (lowercase) though the class is `Cubist` |
| R `grf` | `local_linear_forest` was renamed `ll_regression_forest` in 2.x; `lm_forest` is a *causal* forest that demands `W`; `predict` is S3, not `grf::predict` |
| R packages on this box | site-library not writable → `R_LIBS_USER` + user library dance |
| sklearn trees | cannot consume NAs at all — mean-imputation required before `fit` |
| PyPI/README hunting | install names ≠ import names (`py-forust` vs `forust`), example fences not greppable, bot-blocked pages |

## What our side of the page looks like

```python
from mltreelib import ModelTree            # that is the install story
m = ModelTree(leaf="linear")               # or leaf="mean", leaf_cols="path"
m.fit(x, y)                                # x may contain None — NAs first-class
m.predict(x)                               # never fails on a missing value
m.leaves()[0]                              # coef, na_coef, mean, std, quantiles
m.profile()                                # per-level timings built in
```

- one dependency family (Rust core + thin Python shell);
- NAs are first-class (learned split directions, NA-indicator leaf terms);
- every leaf keeps its statistics and a t-digest;
- the tree reproduces `rpart` exactly when configured as CART (mean leaves),
  so results are checkable against the reference implementations.

## Why this page exists

Benchmarks should be about models, not plumbing. The Elo arena
(`examples/elo_prepare.py` → `elo_score_r.R` / `elo_score_python.py` /
`elo_score_rust.py` → `elo_report.py`, one scores file) exists so the field
can be re-run incrementally — but assembling it highlighted how much incidental
complexity the alternatives impose. That complexity is our documentation.
