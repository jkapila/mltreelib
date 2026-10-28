# 06 — Interpretable models

## EBM — Explainable Boosting Machine / GA2M (InterpretML)

**Source:** https://interpret.ml/docs/ebm.html (docs), `extracted/` lineage
(GA2M paper, Lou et al. 2013).
**Idea:** boosted **generalized additive model**: one shape function per
feature (each learned by a bagged, cyclic gradient-boosting pass over small
trees on that feature alone), optionally top pairwise interactions. Global +
local interpretability without losing much accuracy.

```text
init: intercept = base_rate
repeat for R rounds (outer bagging b = 1..B smooths the curves):
    for feature j in cycle order (least improved first):
        compute residuals r_i = y_i - F(x_i)          # all other terms fixed
        fit a small regression tree on (x_j, r); take its leaf values
        f_j += learning_rate * tree_j                 # the shape function
    optionally: same cyclic fit restricted to top interaction pairs (j, k)
F(x) = intercept + Σ_j f_j(x_j) + Σ_{j<k} f_jk(x_j, x_k)
```

**Knobs:** interactions (0..N), max_bins, outer_bags, inner_bags,
learning_rate, max_rounds, early_stopping, smoothing/monotone constraints.
**Note:** the modern "GA2M done at scale"; its cyclic per-feature boosting is
a feature-engineering goldmine — shape functions can *become* features for
our trees (roadmap's "EBM-style consolidation").

## GritBot (RuleQuest)

**Source:** `extracted/GritBot/` (`check.c`, `cluster.c`, ...),
`references/docs/gritbot-unix.md`.
**Idea:** find **surprising subgroups**: segments whose target rate deviates
strongly from expectation, with correction for multiple testing; outputs
human-readable conditions.

```text
search over conjunctions of conditions (attribute = value / ranges):
    for each candidate segment S:
        surprise(S) = standardized deviation of target rate in S
                      vs expectation (binomial / normal approx)
        penalize by segment count (multiple testing) and small |S|
    keep top non-dominated segments (maximal surprises, minimal overlap)
report: conditions, size, observed vs expected, significance
```

**Knobs:** significance level, minimum segment size, penalty for numbers of
conditions, overlap control, sampling.
**Note:** the direct ancestor of "surprising-pattern mining" — pairs with our
CrossTab Sparsity idea and the structure-recovery scoring.

## Rule extractions from trees (C5.0 rules, Cubist rules)

**Source:** `see5-unix.md`, `cubist-unix.md`.
**Idea:** convert tree paths to IF-THEN rules, then simplify with pessimistic
error pruning; Cubist rules carry linear models on the right-hand side.

```text
for each root->leaf path: rule = conjunction of node conditions
                conclusion = leaf model (Cubist) or class (C5.0)
prune each rule's conditions while pessimistic error does not increase
global optimization: drop/reorder rules by coverage-weighted error
```

**Note:** rule form is what regulators accept; our model-tree output should be
exportable as such rules.
