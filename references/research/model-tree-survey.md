# Model-tree & linear-tree implementation survey

**Date:** 2026-10-28
**Decision this informs:** the Cubist/model-tree replication design (leaf-local
models) and the metrics-first build order.

## The four studied repos

| Repo | What it is | Takeaway for us |
|---|---|---|
| Shamiri-Institute/modeltrees | MOB (model-based recursive partitioning, Zeileis/Hothorn/Hornik 2008): fit a parametric model, test coefficient stability across partitioning vars, split on the most unstable one, repeat — leaves = subgroups with their own model. Port of R **partykit 1.3-0** (`mob`, `lmtree`, `glmtree`) + neg-binom/safe-logit/multinomial models | The partykit lineage we already hold (`sources/partykit_1.3-23.tar.gz`); stability-test splitting is a genuine alternative to gain-based splitting. Uses **narwhals** for dataframe-agnostic input (polars/pandas/pyarrow) — validates our any-data layer |
| cerlymarco/linear-tree | Linear Trees: leaves are linear models, split gain = improvement from fitting linear models in nodes. Plus **LinearForest** (linear fit + RF on residuals) and **LinearBoost** (worst-leaf binary feature boosting) | Node-wise linear gain = the classic model-tree split criterion; LinearForest = the mixing recipe (linear + residual trees) for later |
| schufa-innovationlab/model-trees | Model trees as sklearn estimators with a **gradient-based split criterion** | Split-on-gradient-improvement is a third criterion family (gain / stability / gradient) |
| adc-trust-ai/trust-free | TRUST: Transparent, Robust, Ultra-Sparse Linear Model Trees (PRICAI 2025) + explain()/compare() reports. 60-dataset benchmark: R2 0.67 vs RF 0.62 vs Lasso 0.57 | Sparse linear leaves can beat forests on R2 *and* stay interpretable — the exact claim our program tests. Their auto-explanation reports are a UX pattern worth copying |

## Wider landscape (searched)

- **TAO** (NeurIPS 2025): Tree Alternating Optimization for regression trees
  with linear leaves — monotone convergence, faster than greedy recursive
  partitioning. The strongest algorithmic reference for our model trees.
- **Cubist** (CRAN vignette): model trees + committees + instance-based
  (kNN) corrections + tuning + rules + variable importance — still the spec to
  replicate; the four repos are modern partial reinventions of it.
- **Rust**: `sklears-tree` (CHAID + **oblique splits**, incremental trees),
  `treeboost` (linear models + GBDT + RF in one, zero-copy), `affinitree`
  (distill trees from piecewise-linear nets) — reference implementations for
  the Rust build; also `rpart` (already cloned) and the RuleQuest sources.
- Model trees survey (UPC thesis) for the classical framing: leaves carry
  linear models instead of constants (Quinlan's M5, Kramer's M5P lineage).

## Synthesis

1. Leaf-local models (Cubist/M5/MOB/TRUST) is a live, converging line — the
   same bet as our program, and the "unstudied" gap is real: every modern
   package is a partial reinvention, none is Rust, none measures structure
   recovery against foundation models.
2. Three split-criterion families to implement/compare: impurity gain (CART),
   coefficient-stability tests (MOB), gradient improvement (schufa/TAO).
3. Mixing recipes already exist to steal later: LinearForest (linear + residual
   RF), LinearBoost, Cubist committees + kNN correction.
4. Trust-free's explain()/compare() reports set the bar for our reporting UX.

## Build-order decision (metrics)

**Metrics first, as a separate commit** — "metrics is king"; the tree build
needs the yardstick before it starts. Single `Metrics` class computing complete
reports at once, grounded in the legacy featuregen `SummarizeRegression` suite
(`me, bias, mae, mad, mape, rmse, r2, max_error, explained_variance`, plus its
per-key/per-date summaries) extended with: the DAR decile error profile
(the bathtub view), and a classification suite (accuracy, per-class
precision/recall/f1, macro averages, confusion, logloss, brier, ROC-AUC).

**Legacy quirk found:** the legacy `mae` divides mean absolute error by `n` a
second time; we implement the standard MAE and note the fix (never re-pin
incidental behavior).
