# 02 — Model trees (leaves hold models)

## M5 / M5P (Quinlan, Kramer)

**Source:** literature (M5P = Wang & Witten's Weka version).
**Idea:** regression tree with **linear models at leaves**; split by
maximizing the standard deviation **reduction** of the target, then smooth
predictions along the path and prune with expected error.

```text
grow(rows):
    sd = std(y in rows)
    for feature f, threshold t:
        sdr = sd - |L|/|rows| * sd(L) - |R|/|rows| * sd(R)
    best = argmax sdr; if sdr < threshold: stop
    split; recurse
leaf model: multivariate linear regression of y on features in the leaf
           (features with too few values / low correlation dropped)
smoothing: p = (n_leaf*p_leaf + k*p_parent) / (n_leaf + k)    # k ≈ 15
prune: replace subtree where expected error at leaves decreases
```

**Knobs:** pruning constant, smoothing constant, unpruned/pruned, min leaf
instances.
**Note:** the original model tree; Cubist is its commercial descendant.

## Cubist (Quinlan / RuleQuest)

**Source:** `extracted/cubist-public/` (C sources), `references/docs/cubist-unix.md`.
**Idea:** model trees + **committees** (boosted committees of trees) +
**instance-based correction** (nearest-neighbor adjustment of leaf
predictions) + rule-based optimization and variable usage.

```text
train:
    committee of M trees, each reweighted toward cases the earlier trees
        mispredicted (like boosting; cases weighted by error history)
    each leaf stores: linear model over case-weighted regressors,
        plus the centroid of training cases in the leaf
predict(x):
    route x to a leaf in each committee tree
    v_m = leaf model prediction (linear combination)
    v = mean over committee of v_m
    correction: find k nearest training cases (distance in scaled feature
        space); adjust v by mean(target_i - v_i) of neighbors
        (optionally only neighbors within the leaf's rule)
output also: rules per tree path, variable usage statistics
```

**Knobs:** committees (on/off, number), neighbors (0 = off, else k), rules,
extrapolation limits, sample fraction, max rules.
**Note:** THE spec our model tree aims toward: committees + neighbor
correction are what make small-data regression strong — exactly the
foundation-model niche we contest.

## Linear Tree (cerlymarco/linear-tree)

**Source:** `repos/linear-tree/lineartree/_classes.py` (`Node`,
`_LinearTree._split`).
**Idea:** split gain = improvement in error from fitting **linear models on
both sides** of each candidate; leaves hold arbitrary sklearn linear models.

```text
grow(rows):
    base = fit(linear_model, rows); loss0 = loss(rows, base)
    for feature f, threshold t (candidates from quantiles):
        L, R = partition(rows, f, t)
        m_L = fit(linear_model, L); m_R = fit(linear_model, R)
        gain = loss0 - (loss(L, m_L) + loss(R, m_R))
    best = argmax gain; if gain < min_gain: return Leaf(base)
    split at best; recurse
predict(x): leaf's linear model
```

**Knobs:** max_depth, min_samples_split/leaf, max_leaf_nodes, criterion
(variance/mae/mse...), fit_intercept, base linear model choice.
**Note:** our v1 recipe — do exactly this first, in Rust, with faer least
squares. Also hosts LinearForest / LinearBoost (see 03/04).

## TRUST trees (trust-free, PRICAI 2025)

**Source:** `repos/trust-free/` (README + MANUAL).
**Idea:** ultra-**sparse** linear model trees: leaves hold linear models with
very few active terms, splitting aims at RF-level accuracy with fully readable
segments; ships `explain()`/`compare()` reports.

```text
grow(rows):
    like Linear Tree but leaf fit is a SPARSE linear fit
        (few non-zero coefficients; regularized selection)
    split criterion balances segment error against leaf complexity
predict(x): sparse linear leaf model
report: per-leaf coefficients, segment descriptions, comparison tables
```

**Knobs:** sparsity of leaves, tree depth, segment minimum size.
**Note:** their 60-dataset result (R² 0.67 vs RF 0.62) is the claim to test
against; the report UX sets our bar.

## LightGBM `linear_tree` — model trees inside boosting

**Source:** LightGBM docs (Parameters: `linear_tree`, `linear_lambda`,
`linear_tree_output`).
**Idea:** leaf values become **regularized linear models** fit on the
gradient/hessian pseudo-targets of the current boosting round.

```text
for m in 1..M:                       # boosting rounds
    r_i = negative gradient at current prediction
    grow histogram tree where each leaf refines to a linear model
        w_leaf = argmin Σ_{i in leaf} (r_i - w·x_i)^2 + lambda * ||w||^2
        (hessian weighting; linear_lambda regularizes)
    prediction update: sum of leaf models' outputs (share/separate modes)
```

**Knobs:** `linear_tree`, `linear_lambda`, `linear_tree_output` = share |
separate, min_data_in_leaf (needs more data than constant leaves).
**Note:** the caveat you remembered is real — it is a flag in GBDT serving
residual fitting; no leaf inference/reporting, categoricals stay histogram
splits. Our model tree stands alone instead.

## MOB-leaf models (the general recipe)

**Source:** `repos/modeltrees/` (`NodeModel` protocol).
**Idea:** any differentiable-in-scores parametric model as the leaf object:
OLS, GLM, negative binomial, safe logit, multinomial. The tree tests the
model's score stability to choose splits.

```text
NodeModel protocol:
    fit(y, x, weights, ...) -> FitResult{coefficients, objfun, estfun, ...}
    predict(result, x) -> predictions
    loglik(result) -> (value, df)           # optional
    split_objectives(...) -> left/right objective vectors   # optional batch
grow: fluctuation tests on estfun -> split variable -> recurse
```

**Note:** for our Rust build: define exactly this trait so leaves can be
constant (CART), linear (model tree), quantile, or distributional (NGBoost
head) without touching the growth code.
