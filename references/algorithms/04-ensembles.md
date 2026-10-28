# 04 — Ensembles and meta-ensembles

## Bagging (Breiman 1996)

**Source:** literature; mlxtend.ensemble `BaggingClassifier/Regressor`.
**Idea:** bootstrap the training set, fit one base model per sample, average
(majority vote). Variance reduction; helps unstable learners (trees).

```text
for b in 1..B:
    sample n rows with replacement (optionally max_samples fraction)
    h_b = base_model.fit(sample, y[sample])
predict = mean(h_b(x)) or majority vote
out-of-bag error: score each row only with models that did not see it
```

**Knobs:** n_estimators, max_samples, max_features, bootstrap features too.
**Note:** OOB error is the free validation tree ensembles rarely exploit.

## Random Forest (Breiman 2001)

**Source:** `extracted/randomForest_4.7-1.2/` (R reference).
**Idea:** bagged trees + **random feature subsets** at each split; average.
De-correlates trees → big variance drop.

```text
for b in 1..B:
    bootstrap sample S_b
    grow tree on S_b: at each node, try only mtry random features
                      (split = best among those; no pruning)
predict = mean of tree predictions (regression) / majority vote (classification)
importance: permutation of OOB features -> accuracy drop
```

**Knobs:** ntree, mtry (=sqrt(p) class / p/3 reg), nodesize, sampsize,
replace, maxnodes.
**Note:** the anchor of every comparison; sklearn's `RandomForest*` mirrors it.

## Extra Trees (Geurts et al. 2006)

**Source:** literature.
**Idea:** like RF but thresholds are drawn **randomly** (K per feature) instead
of optimized, and the whole sample is used. Faster, lower variance.

```text
for b in 1..B:
    tree on (optional subsample of) full data
    at each node: for each of K random features, draw a random threshold
                  between observed min/max; pick the best of those K*p draws
predict = average
```

**Knobs:** n_estimators, max_features, n_jobs, random_state.
**Note:** the "randomize harder" end of the forest spectrum.

## wsrf — weighted subspace random forests

**Source:** `extracted/wsrf_1.7.32/` (R package).
**Idea:** weight features by accumulated importance and sample the split
subspace **proportionally to weights**; parallel split-point search ("weights
computed in a parallel fashion") — accuracy/speed trade-off vs RF.

```text
w_f = 1 for all features
for tree in 1..B:
    at each node: draw subspace S with probability ∝ w_f
    find best split within S (parallel over features)
    after tree: w_f *= (importance_f of this tree)   # or decayed update
predict = vote/average
```

**Knobs:** ntree, mtry, nodesize, weights update scheme.
**Note:** feature-subspace *adaptation* — the fix for "RF wastes splits on
junk features", close to our structure-recovery question.

## RLT — reinforcement learning trees (Zhu, Zeng, Kosorok)

**Source:** `extracted/RLT_3.2.6/` (R package).
**Idea:** use **reinforcement learning at the top of the tree**: learn (by
Q-learning over split-variable choices) which variables matter for splitting
near the root; classic splitting below; embedded Monte Carlo imputation for
missingness.

```text
RL phase: simulate tree growing episodes; Q(s = depth/variable state,
          a = split variable) updated by resulting subtree error
grow tree with policy from Q at shallow depths (large subtrees);
    standard impurity splits deeper
aggregate many such trees (forest); importance from Q / splits
```

**Knobs:** ntrees, mtry, splitsize, nodesize, reinforcement (on/off), qlearning
iterations.
**Note:** the most radical "trees learn to split" idea; a natural future
experiment for our structure-recovery benchmark (does RL pick the
hierarchical features?).

## grf — generalized random forests / Local Linear Forests (Athey et al.)

**Source:** grf-labs (R/C++), the R classic that never gets highlighted.
**Idea:** forests as **adaptive neighborhood machines**: split criterion =
honest maximization of a target quantity (CATE for causal forests; gradient
for regression); Local Linear Forests fit local linear models in leaves.

```text
grow tree: split to maximize heterogeneity of the estimating quantity
           (honest splitting: separate samples for split choice and fitting)
leaf estimate: average of y in leaf (regression forest)
Local Linear Forest: at prediction, take the forest's neighbor weights W(x),
    fit weighted local linear model: beta = argmin Σ_i W_i(x)(y_i - x_i·beta)^2
```

**Knobs:** num.trees, mtry, honesty, alpha/min.node.size, ci.group.size,
local.linearcorrections.
**Note:** LLF = random forest + linear leaves (weighting version of our model
trees); causal forests are the applied workhorse of econ/industry.

## Linear Forest (cerlymarco/linear-tree)

**Source:** `repos/linear-tree/` (`_LinearForest`).
**Idea:** fit a linear model on all data, then a random forest on the
**residuals**; prediction = linear + RF residual.

```text
lin = linear_model.fit(X, y)
r = y - lin.predict(X)
forest = RandomForest.fit(X, r)
predict(x) = lin.predict(x) + forest.predict(x)
```

**Knobs:** the linear model + all RF knobs.
**Note:** the simplest strong hybrid; candidate "mixing" recipe (#8 in the
build TODO).

## Voting (mlxtend `EnsembleVoteClassifier`)

**Source:** mlxtend.classifier.
**Idea:** combine classifiers by majority rule (hard) or averaged
probabilities (soft), optionally weighted.

```text
hard: predict(x) = argmax_k Σ_m w_m [h_m(x) == k]
soft: predict(x) = argmax_k Σ_m w_m * p_m(k | x)
```

**Knobs:** voting hard|soft, weights, use_clones, fit_base_estimators.

## Blending (mlxtend `BlendingClassifier/Regressor`)

**Source:** mlxtend.ensemble.
**Idea:** two-level stacker with a **holdout split** instead of CV: base
models fit on the training fold, meta-model fit on the holdout fold's base
predictions.

```text
split train into (fit_fold, holdout_fold)
base_m.fit(fit_fold)
meta_X = [base_m.predict(holdout_fold) for m]
meta.fit(meta_X, y[holdout])
predict: meta over base predictions on new data
```

**Knobs:** holdout size, use_features_in_secondary, verbose.

## Stacking (mlxtend `StackingClassifier/Regressor`)

**Source:** mlxtend.
**Idea:** meta-learner over base-model predictions, meta-features computed
**in-sample** (fast; slight leak risk → use CV variant).

```text
base_m.fit(X, y) for all m
meta_X = [base_m.predict(X) for m]  (+ optionally X itself)
meta.fit(meta_X, y)
predict = meta over base predictions
```

**Knobs:** use_features_in_secondary, use_clones, store_train_meta_features.

## StackingCV (mlxtend `StackingCVClassifier/StackingCVRegressor`)

**Source:** mlxtend.
**Idea:** leak-free stacking: out-of-fold base predictions train the
meta-model; base models refit on all data for deployment.

```text
for fold (X_tr, X_te) in CV(X):
    base_m.fit(X_tr, y_tr); meta_X[X_te] = base_m.predict(X_te)
meta.fit(meta_X, y)
for m: base_m.fit(X, y)          # final refit
predict = meta over full-data base predictions
```

**Knobs:** cv, shuffle, refit, use_features_in_secondary, multi_output.
**Note:** the honest stacking variant — the one to copy in our mixing phase.
