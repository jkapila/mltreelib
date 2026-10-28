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

## The Elo board (arena run, stock configs)

Five real regression sets (concrete, abalone, ozone, boston, fc2008), one
80/20 split each, identical encodings; round-robin Elo on per-dataset RMSE
(K=32, start 1000). Reproduce: `elo_prepare.py` → `elo_score_r.R` →
`elo_score_python.py` → `elo_score_rust.py` → `elo_report.py`; the numbers
live in `benchmarks/elo_scores.csv`.

| Elo | model | note |
|---|---|---|
| 1418.7 | CatBoost (200) | best on fc2008 and boston |
| 1327.0 | forust (Rust) | |
| 1216.0 | grf local-linear forest | fc2008 NAs defeat it (4/5) |
| 1212.8 | LightGBM (200) | |
| 1202.3 | perpetual (Rust) | |
| 1129.3 | CTBoost | ctree-based GBDT |
| 1092.0 | Cubist (R) | the C5-family regressor |
| 1081.0 | grf RF | |
| 972.8 | earth (MARS) | cannot take NAs (4/5) |
| 963.1 | glmtree (partykit) | |
| 948.4 | modeltrees lmtree | matches partykit to 3 decimals - port certified |
| 945.6 | lmtree (partykit) | |
| 884.9 | ctree (partykit) | |
| 867.9 | sklearn tree | needs mean-imputation for NAs |
| 858.2 | **ModelTree(mean)** — ours | unoptimized defaults |
| 714.9 | rpart | |
| 597.0 | ModelTree(path) — ours | |
| 568.2 | ModelTree(all) — ours | unregularized leaves blow up on fc2008 |

Reference scale (different protocol): TabArena best overall Kumo-Tabular
1955; best tree-based LightGBM tuned+ens 1411.

The board is the yardstick: every default change, leaf variant, and forest
feature moves these numbers, and `examples/default_search.py` bakes the
winning defaults against the lmtree/cubist/rpart bars.

## Why this page exists

Benchmarks should be about models, not plumbing. The Elo arena
(`examples/elo_prepare.py` → `elo_score_r.R` / `elo_score_python.py` /
`elo_score_rust.py` → `elo_report.py`, one scores file) exists so the field
can be re-run incrementally — but assembling it highlighted how much incidental
complexity the alternatives impose. That complexity is our documentation.
