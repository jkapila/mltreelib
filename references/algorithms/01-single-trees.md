# 01 — Single trees and tree-shaped models

## CART (Breiman et al.) — classification and regression trees

**Source:** oracle `legacy/mltreelib/tree/models.py`; literature.
**Idea:** greedy recursive binary splits on one feature; leaves carry a
constant (mean / majority class). Regression uses variance reduction,
classification uses Gini or entropy.

```text
grow(rows, depth):
    if stopping(rows, depth):            # min_samples, max_depth, pure, no gain
        return Leaf(mean(y) or majority(y))
    best = None
    for feature in features:
        for threshold in candidate_splits(feature, rows):   # midpoints
            L, R = partition(rows, feature, threshold)
            gain = impurity(rows) - wL*impurity(L) - wR*impurity(R)
            best = max(best, (gain, feature, threshold))
    if best.gain <= min_gain: return Leaf(...)
    L, R = partition(rows, best.feature, best.threshold)
    return Node(best, grow(L, depth+1), grow(R, depth+1))
predict(x): walk the tree to a leaf
```

**Knobs:** max_depth, min_samples_split/leaf, min_impurity_decrease, ccp_alpha
(cost-complexity pruning: minimize R(T) + alpha|T| over subtrees).
**Note:** our v1 baseline; pruning and surrogate splits (missing values) come
from this family.

## C4.5 (Quinlan)

**Source:** literature; See5 lineage in `extracted/see5-public/`.
**Idea:** CART-like with information gain **ratio** (penalizes many-way
splits), gain-ratio pruning (pessimistic error), rules extracted from trees.

```text
grow(rows):
    if stopping: return Leaf(majority)
    gain_ratio(f, t) = (info_gain(f, t)) / split_info(f, t)      # split_info = -Σ p log p of sides
    best = argmax over (f, t) of gain_ratio, subject to gain >= 0.1 * best_gain
    split on best; recurse
prune(tree): replace subtree with leaf while estimated error drops
                (pessimistic binomial error estimate, with leaves' errors + 0.5)
```

**Knobs:** confidence factor (pruning), min cases per leaf, windowing (train on
a subset, order trees by accuracy).

## C5.0 (Quinlan) — See5

**Source:** `extracted/see5-public/` (C sources: `global.c`, `defns.h`),
`references/docs/see5-unix.md`.
**Idea:** faster, more accurate C4.5: boosting committees of trees (trial
weighting like AdaBoost), **winnowing** (discard unhelpful attributes),
probabilistic boosting with normalized weights, rule sets from trees.

```text
f = uniform weights over cases
for trial in 1..T:                       # boosting committee
    tree = grow_c45(D, weights = f)      # gain-ratio splits, min cases
    err = weighted error of tree
    if err > 0.5: stop
    beta = err / (1 - err)
    f[i] *= beta^(1 - mistake[i])        # upweight misclassified
    normalize f
    optionally winnow: drop attributes consistently unused
predict(x) = weighted vote across trials (log beta weighting)
```

**Knobs:** trials (boosting rounds), winnowing on/off, rules vs trees, case
weights, confidence factor.
**Note:** committees + winnowing = the "Quinlan additions" our replication must
show; see `see5-unix.md` for option semantics.

## OneR (Holte)

**Source:** mlxtend.classifier.
**Idea:** one-level decision tree: pick the single attribute whose
per-value majority class predicts best.

```text
for each feature f:
    rule(f) = map value -> majority class over training rows
    err(f) = misclassifications of rule(f)
best = argmin err
predict(x) = best[x[best_feature]] (unseen value -> overall majority)
```

**Knobs:** min_buckets (merge rare values), min_accuracy.
**Note:** the honest baseline — if a fancy model cannot beat OneR, stop.

## CHAID (Kass 1980)

**Source:** literature.
**Idea:** chi-square interaction detection: merge categories of each predictor
until no pair differs significantly (Bonferroni-adjusted), split on the most
significant predictor, recurse. Target can be categorical or binned numeric.

```text
grow(rows):
    for each predictor x:
        merge levels of x while some pair's target distributions
            are not significantly different (chi-square test, adjusted p)
        p(x) = chi-square significance of x vs target after merging
    x* = argmin p(x); if p(x*) > alpha: stop -> Leaf(mode)
    split on x*'s merged level groups; recurse
```

**Knobs:** alpha (split), min_split/min_bucket sizes, max_depth, Bonferroni.
**Note:** statistical-test splitting (like ctree/MOB) rather than impurity.

## ctree — conditional inference trees (Hothorn et al.)

**Source:** `extracted/party_1.3-23/` (R `party`), `extracted/partykit_1.3-1/`.
**Idea:** stop splitting when **no** predictor shows a significant association
with the response (permutation test); among significant ones, split where the
linear statistic is maximized. Removes CART's bias toward high-cardinality
features.

```text
grow(rows):
    for each z: statistic = max over splits of linear statistic
                (e.g. standardized mean differences for numeric y)
                p(z) = permutation test p-value, multiplicity-adjusted
    if min p(z) > alpha: return Leaf(conditional mean / distribution)
    z* = argmin p; break at split maximizing z*'s linear statistic
    recurse on both sides
```

**Knobs:** teststat ("max"/"quadratic"), splitstat, splittest, alpha, minsplit,
minbucket, stump.
**Note:** p-values gate growth — the reference for "honest" splitting; our
"native categorical" requirement benefits from its unbiased treatment.

## MOB — model-based recursive partitioning (Zeileis, Hothorn, Hornik)

**Source:** `extracted/partykit_1.3-1/` (`mob()`), port in `repos/modeltrees/`
(see its `instability.py`: supLM/chisq statistics, decorrelated processes).
**Idea:** fit a parametric model in the node; test whether its **coefficients
are stable** across partitioning variables (M-fluctuation tests on estimating
functions); split where instability is strongest; repeat. Leaves = models.

```text
grow(rows):
    fit = fit_node_model(y ~ x, rows)               # lm/glm/negbin/... gives estfun
    for z in partitioning_vars:
        stat(z) = supLM or chisq of decorrelated empirical process of fit's
                  scores over z (cluster-robust if clusters given)
        p(z) = p-value with multiplicity adjustment
    if all p(z) > alpha: return Leaf(fit)
    z* = argmin p; break at z* value maximizing the process
    recurse on both sides
predict(x): route to leaf; leaf model's predict
```

**Knobs:** alpha, bonferroni, minsplit/minbucket, parm (which coefficients are
tested), objfun (node model objective), cluster handling.
**Note:** the second split-criterion family for our TODO; modeltrees'
`NodeModel` protocol (fit → coefficients/objfun, `split_objectives` batch) is
the cleanest blueprint.

## MARS — multivariate adaptive regression splines (`earth`)

**Source:** `extracted/earth_5.3-6/` (R package `earth`).
**Idea:** basis expansion of hinge functions with greedy forward selection of
pairs, then backward pruning by GCV. Effectively a tree of basis functions.

```text
B = {1}
repeat until max_terms:
    for each basis h in B, feature x, knot t:
        candidates h*max(0, x-t) and h*max(0, t-x)
    add the pair of candidates minimizing RSS (must beat min RSS improvement)
prune: remove term pairs minimizing GCV = RSS / (n - eff_params)^2
```

**Knobs:** degree (interaction order), nterms, penalty (pruning), minspan,
thresh (forward stopping), fast.k.
**Note:** `earth` also reports variable importance — a non-tree method that
still answers "which features matter".

## semtree — structural equation model trees

**Source:** `extracted/semtree_0.9.23/`.
**Idea:** leaves hold SEMs (covariance structure models); split on any variable
that significantly improves the SEM fit (score test / LRT / Wald), often with
sequential testing.

```text
grow(rows):
    fit SEM to rows -> loglik L0
    for feature z:
        for split s: fit SEM left/right -> L(z,s); stat = 2*(L(z,s) - L0)
        best_z = max over s of stat (chi-square p-value)
    if best p > alpha: return Leaf(SEM)
    split at best (z, s); recurse
```

**Knobs:** alpha, method (score/lrt/wald), min.bucket, num.threads.
**Note:** the extreme "leaves are full statistical models" case — same skeleton
as MOB with richer leaf objects.

## glmertree — GLMM trees

**Source:** `extracted/glmertree_0.2-7/`.
**Idea:** MOB where the node model is a **generalized linear mixed model**
(random effects per group), so leaves keep group-level structure — relevant to
our group/hierarchy thesis.

```text
grow(rows):
    fit GLMM (fixed + random effects) in node
    M-fluctuation test of fixed-effect scores across partitioning vars
    split at most unstable variable; recurse
predict(x, group): leaf's fixed + random effects for that group
```

**Knobs:** alpha, minbucket, glmer control; `cluster` for robust scores.
**Note:** direct precedent for "leaves keep hierarchical structure".
