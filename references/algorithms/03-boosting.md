# 03 — Boosting

## AdaBoost (Freund & Schapire) — SAMME / SAMME.R

**Source:** literature; SAMME.R flavor used by modern libraries.
**Idea:** reweight training cases each round toward the misclassified;
weighted vote of weak learners. SAMME uses class labels; SAMME.R uses
probability estimates (smoother, usually better).

```text
w_i = 1/n
for m in 1..M:
    h_m = weak_learner.fit(X, y, sample_weight = w)      # e.g. decision stump
    err_m = Σ w_i * [h_m(x_i) != y_i] / Σ w_i
    if err_m >= 1 - 1/K: stop
    alpha_m = log((1 - err_m)/err_m) + log(K - 1)        # SAMME
    w_i *= exp(alpha_m * [h_m(x_i) != y_i])
    normalize w
predict(x) = argmax_k Σ_m alpha_m * [h_m(x) == k]         # SAMME
SAMME.R variant: weight update from predicted class probabilities
                 (expected error under w), vote weighted by log-probabilities
```

**Knobs:** n_estimators, learning_rate (shrink alpha), base estimator depth.
**Note:** the reference for reweighting schemes — LinearBoost below reuses
SAMME/SAMME.R over linear base learners.

## Gradient Boosting / GBRT (Friedman 2001)

**Source:** literature; sklearn `GradientBoosting*`.
**Idea:** stage-wise additive model minimizing a differentiable loss; each
round fits a tree to the **negative gradient** (pseudo-residuals), line-search
step size.

```text
F_0(x) = argmin_c Σ L(y_i, c)                 # constant init
for m in 1..M:
    r_i = -dL(y_i, F(x_i))/dF                  # pseudo-residuals
    h_m = tree.fit(X, r)                       # regression tree to residuals
    gamma_j = argmin_g Σ_{x_i in leaf j} L(y_i, F(x_i) + g)   # line search
    F += nu * Σ_j gamma_j * 1[x in leaf j]     # nu = learning rate
predict(x) = F_M(x) (+ link inverse for classification)
```

**Knobs:** loss (squared/log/poisson/quantile), n_estimators, learning_rate,
max_depth (stumps typical), subsample (stochastic GB).
**Note:** quantile loss variant = #23; the "boosted residuals" pattern is
what LinearForest copies with RFs.

## XGBoost (Chen & Guestrin)

**Source:** literature.
**Idea:** GBDT with **second-order** objective (gradient + hessian), explicit
regularization on trees, histogram/weighted quantile split finding, shrinkage
and column subsampling. Exact gain formula:

```text
for m in 1..M:
    g_i = dL/dF,  h_i = d2L/dF2
    grow tree by gain(f, t) = 1/2 [ (G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ)) ] - γ
        G/H = sums of g/h on each side; λ = L2 leaf reg, γ = min split loss
    leaf weight w_j = -G_j / (H_j + λ)
    F += nu * tree
missing values: learn default direction per split (try both, keep gain)
```

**Knobs:** max_depth, eta, min_child_weight, subsample, colsample, lambda/alpha,
gamma, max_bin.
**Note:** the reference "giant"; our comparisons use tuned versions with equal
budgets.

## LightGBM (Ke et al.)

**Source:** literature.
**Idea:** faster GBM: gradient-based one-sided sampling (GOSS, keep all
high-gradient rows, sample low), exclusive feature bundling (EFB), leaf-wise
growth, native categorical splits, linear leaves option (see 02).

```text
GOSS: keep rows with large |grad|; random subset of small-grad rows,
      scale their gradients up by (1-a)/b
EFB: bundle mutually-exclusive sparse features into single features
grow: leaf-wise — expand leaf with max delta loss (histogram splits)
categorical split: sort categories by gradient statistics, best 2-partition
```

**Knobs:** num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction,
lambda_l1/l2, cat_smooth/max_cat_to_onehot, linear_tree.
**Note:** the production default everyone reaches for — our "make trees fast
in Rust" target.

## CatBoost (Prokhorenkova et al.)

**Source:** CatBoost docs + NeurIPS 2018 paper (see
`references/research/categorical-encoding-research.md` for the encoding).
**Idea:** ordered target statistics + ordered boosting — leakage-free
categorical handling and unbiased gradient boosting; symmetric (oblivious)
trees by default.

```text
ordered TS: shuffle; per row, ctr = (countInClass_prefix + prior)/(count_prefix + 1)
combinations: greedy cat×cat features built at split time
for m in 1..M:
    permutation_m = new shuffle
    for i in permutation order:
        residual_i computed using models built WITHOUT row i
        (ordered boosting: only earlier-in-permutation rows update the model)
    grow oblivious tree (same split test at all nodes at a depth)
```

**Knobs:** iterations, depth, l2_leaf_reg, bagging_temperature (Bayesian
bootstrap), one_hot_max_size, random_strength, grow_policy (symmetric/
depthwise/lossguide), boosting_type (Ordered/Plain).
**Note:** the categorical king; its ordered principle is the anti-leakage
gold standard our encoders follow.

## mboost — model-based boosting (Bühlmann, Hothorn)

**Source:** `extracted/mboost_2.9-14/` (R package).
**Idea:** component-wise gradient boosting in **function space** over a large
library of base learners (linear terms, splines, trees); each round selects
whichever single base learner reduces the empirical risk most.

```text
F_0 = offset
for m in 1..M:
    r_i = -dL(y_i, F(x_i))/dF
    for each base learner b_j (e.g. one linear term, one spline, one tree):
        fit b_j to r -> risk improvement R_j
    j* = argmax R_j; F += nu * b_j*      # only ONE component per step
select M by cvrisk / early stopping
variants: glmboost (linear terms only), gamboost (smooth terms),
          blackboost (trees + arbitrary loss), gamboostLSS (distributional)
```

**Knobs:** family (loss), baselearner, nu (shrinkage), mstop, risk ("inbag"/
"oob"), df (spline degrees of freedom).
**Note:** the "feature selection by boosting" school — sparse, interpretable;
the natural bridge between our encoders and model trees.

## NGBoost — natural gradient boosting

**Source:** `repos/ngboost/` (Stanford ML Group).
**Idea:** GBM that predicts **parameters of a distribution**; each round takes
a **natural gradient** step on the distribution's score (Fisher-scaled), with
a base learner per parameter (e.g. Normal: mean + std).

```text
init: theta_0 = base parameter values (e.g. mean(y), std(y))
for m in 1..M:
    for each parameter k of the distribution:
        g_k = dS(y, theta)/dtheta_k                # score gradient
        F_k = Fisher information of the distribution
        ng = F_k^-1 g_k                            # natural gradient
        fit base learner h_{m,k} to ng
    theta += nu * h_m
predict: the distribution parameters -> mean/median/quantiles/samples
```

**Knobs:** Dist (Normal/LogNormal/Exponential...), Base (tree/linear),
n_estimators, learning_rate, natural_gradient (on/off), minibatch_fraction,
verbose_eval.
**Note:** probabilistic predictions with trees — the principled alternative to
post-hoc quantile estimation; pairs with our `Metrics` via proper scores.

## Quantile GBT (pinball loss)

**Source:** literature (Friedman 2001 §losses); crates `gbt-quantile`,
LightGBM/XGBoost objective.
**Idea:** GBM with pinball (quantile) loss; each tree fits the quantile
gradient (asymmetric step function).

```text
L_q(y, f) = max(q*(y - f), (q-1)*(y - f))
gradient step: r_i = q - [y_i < F(x_i)]        # asymmetric pseudo-residuals
fit regression tree to r; line-search gamma minimizing Σ L_q
F += nu * tree
multi-quantile: fit one chain per quantile or share trees with quantile heads
```

**Knobs:** alpha (quantile), learning_rate, depth, min_samples_leaf.
**Note:** direct competitor to QRF (05) and NGBoost quantiles.

## LinearBoost (LinearBoost/linearboost-classifier)

**Source:** `repos/linearboost-classifier/` (README + `src/`).
**Idea:** AdaBoost-style boosting where the weak learner is the **SEFR linear
classifier** (a fast sparse-hyperplane linear model), optionally on kernel
features. SAMME / SAMME.R reweighting.

```text
w_i = 1/n
for m in 1..M:
    h_m = SEFR.fit(X (or kernel features), y, weights = w)
    SAMME/SAMME.R update of w from h_m's errors/probabilities
    vote weighted by alpha_m
predict = weighted vote of SEFR classifiers
```

**Knobs:** n_estimators, algorithm (SAMME | SAMME.R), kernel features on/off,
SEFR's own sparsity settings.
**Note:** proof that boosting over *linear* base learners is competitive and
extremely fast — relevant to our "fast trees in Rust" story.

## SEFRBoost (same repo)

**Source:** `repos/linearboost-classifier/` (`linearboost.sefr_boost`).
**Idea:** GBDT-style boosting of **shallow trees whose internal splits are
SEFR hyperplanes on pseudo-residuals** — an oblique-split gradient boosting
machine (classification + regression).

```text
F_0 = base value
for m in 1..M:
    r_i = pseudo-residuals (gradient of loss)
    grow shallow tree where each split is a SEFR hyperplane on (X, r):
        best oblique split maximizing residual separation
    leaf outputs = leaf means of r (or gamma line-search)
    F += nu * tree
```

**Knobs:** depth, n_estimators, learning_rate, SEFR split settings.
**Note:** oblique splits inside boosting — sits between XGBoost and the
oblique-tree literature (see arXiv finds in the consolidation note).
