# Spec 29: `ModelBoost` — The Boosting Family

Status: proposed

Family: boosting. Stages are fitted sequentially to loss gradients and combined as a shrunk sum.

Scope boundary: this specification owns the boosting family only. The bagging family lives in Spec 28 (`ModelForest`). The meta-estimator family lives in Spec 30 (`ModelMeta`).

## Problem Statement

`mltreelib` has no boosting. The local 56-dataset Elo arena shows why that matters and also shows how much work it is.

**The gap is real.** On the local board, the boosting family occupies the top of the ranking: CatBoost(200) 1457.8 overall, `perpetual (Rust)` 1400.1, ChimeraBoost 1340.0, sklearn GBM 1274.0, HistGradientBoosting 1258.6, LightGBM(200) 1240.9. `ModelTree(all)` sits at 1290.1, above three of them and below three.

**But the gap is not what it looks like.** `ModelTree(all)` is 1204.7 on classification, ahead of sklearn GBM (1188.6), HistGradientBoosting (1173.2), and LightGBM(200) (1155.5). Its regression Elo of 715.9 is the problem, and bagging fixes that (Spec 28). Boosting is a different lever, aimed at a different weakness.

**The bar is set by an existing Rust library.** `perpetual (Rust)` scores 1400.1 overall, 1505.9 regression, 1314.7 classification, with a 7.19 s mean fit time. It beats CatBoost(200) on regression Elo and beats LightGBM(200), sklearn GBM, and sklearn RF on overall Elo. It does this with no leaf families, no monotone constraints, no categorical partition search, and no native missing-value routing.

**So a plain GBDT is not the deliverable.** A `ModelBoost` that is merely a slower `perpetual` has no reason to exist. It has to earn its place with something the plain Rust booster does not have.

### What the library already has that a booster needs

This is the good news, and it is substantial.

| Need | Existing | Source |
|---|---|---|
| Tree growth with a chosen split criterion | `GainFn` with 30 variants including `FriedmanMse`, `Sse`, `Sdr`, `AbsoluteError`, `Huber` | `crates/mltreelib-core/src/criteria.rs` |
| Leaf model fitting with a family and a solver | `LeafKind`, `LeafFamily`, `LeafSolver` (`Irls`, `Lbfgs`, `Ccd`, `CmaEs`, `Shade`, `NaturalGradient`, `GeneticAlgorithm`) | `crates/mltreelib-core/src/leaf.rs` |
| Monotone constraints on splits and leaf coefficients, separately | `tree_monotone`, `leaf_monotone` | `crates/mltreelib-core/src/model_tree.rs` |
| Missing-value routing with per-node stored direction | `split.rs` | `crates/mltreelib-core/src/split.rs` |
| Categorical partition search | `CatOrder`, `CatSearch` | `crates/mltreelib-core/src/split.rs` |
| Binned split evaluation | `SplitEval::Histogram`, 256 bins | `crates/mltreelib-core/src/split.rs` |
| Weighted fits with fractional weights | `fit_weighted_named` | `crates/mltreelib-core/src/model_tree.rs` |
| Monotone-constrained linear regression in leaves | `leaf_monotone` plus `l1`/`l2` | `crates/mltreelib-core/src/leaf.rs` |

The library also has the solver seam from Spec 26, so a stage's leaf fit can route through the shared objective, gradient, and budget contracts rather than a private path.

### What is missing

1. **No gradient or Hessian machinery.** Nothing computes $\partial \ell / \partial \hat y$ or its second derivative for a fitted ensemble.
2. **No stage loop.** No sequential accumulation, no learning rate, no stage-wise early stopping.
3. **No monotone constraint over leaf values.** The library constrains split direction and leaf coefficients, but a boosted leaf is a *constant* in the standard formulation, so the constraint has no coefficient to bind.
4. **No binned split support for the regression gains.** `SplitEval::Histogram` is gated to gini, entropy, gain ratio, and twoing. **This lands in Spec 27, Decision 15 and Phase 1**, not here. See Decision 5.
5. **No out-of-bag or validation-based diagnostics for stages.**

### The three corrections this specification must not inherit

Verified during research, and each one is a trap:

**DART is not a Müller, Mentrup, and Guestrin paper.** The 2015 paper is Rashmi and Gilad-Bachrach, *Dropout Adaptive boosting for Machine Learning*, arXiv:1505.01866, PMLR v38 (2015). A search of the NeurIPS 2015 accepted-paper list, the conference book author lists, arXiv, Crossref, OpenAlex, Semantic Scholar, and CORE finds no "Fast Boosting with Outlier Detection" paper by Müller, Mentrup, or Guestrin, in NeurIPS 2015 or in any workshop. Do not cite it.

**ChimeraBoost is not a paper.** It is a PyPI package and GitHub repository, `bbstats/chimeraboost`, Apache-2.0, version 0.34.0, described as "CatBoost-inspired gradient boosting in pure Python with a numba backend". arXiv, Crossref, and OpenAlex have zero records under that title. It is a strong competitor on the TabArena board (1302 Elo default, 1338 tuned, 1365 tuned and ensembled) and a useful reference implementation, but citing it as a paper is wrong.

**The GA²M paper is Lou, Caruana, Gehrke, and Hooker, KDD 2013**, DOI `10.1145/2487575.2487579`, "Accurate intelligible models with pairwise interactions". arXiv:1303.3217 is a different paper entirely (a note on diastatic entropy and balanced metrics).

## Solution

One facade, `ModelBoost`, with a `kind` switch, one mode axis, trait-gated capabilities, and one shared report.

### 1. The kinds

| Kind | What it is | Verdict |
|---|---|---|
| `gbdt` | Friedman's gradient boosting with second-order stage updates | First wave |
| `dart` | Rashmi and Gilad-Bachrach dropout boosting, with the drop and rescale mechanics | Second wave |
| `linear` | Boosting with a linear model as the stage, the `gblinear` shape | Deferred |
| `forest_stage` | Each stage is a random forest fitted to the negative gradient, the LightGBM `rf` shape | Deferred |
| `ordered` | CatBoost-style ordered boosting with ordered target statistics | Deferred, large project |

### 2. The core mechanism, and why it fits this library

Friedman's gradient boosting computes, for empirical risk $\widehat C(F) = \frac{1}{n}\sum_i \rho(y_i, F(x_i))$:

$$u_i = -\left.\frac{\partial \rho(y,f)}{\partial f}\right|_{f = F_{m-1}(x_i)},$$

fits a base learner to the pseudo-responses $u_i$, and adds it with a line search: $F_m = F_{m-1} + \nu \hat g_m$.

**The library does not need a special constant-leaf path.** XGBoost and LightGBM compute a stage's leaf values as a Newton step, $w = -\sum g / (\sum h + \lambda)$, which is a *constant* per leaf. That requires them to special-case leaf fitting entirely, because a constant cannot express a model.

`mltreelib` can instead fit the stage tree with its existing engine on pseudo-residuals. For squared error, $u_i = y_i - F_{m-1}(x_i)$ and the least-squares leaf fit gives $\sum u_i / n$, which equals $-\sum g / (\sum h + \lambda)$ when $h_i = 1$. The two formulations coincide for squared loss, and the library's formulation additionally supports:

- linear leaves in a stage, with the library's ridge and lasso penalties,
- leaf-family leaves (Poisson, Gamma, Tweedie, Tobit) fitted by IRLS,
- monotone constraints on leaf coefficients, which the Newton-step formulation cannot express at all,
- the leaf solver seam from Spec 26.

**The trade-off is honest.** The pseudo-residual formulation is exact for squared error and for any loss whose leaf fit is the matching M-estimator. For logistic loss it is an approximation to the Newton step unless the leaf family is set to the matching GLM. The library's `LeafFamily` and `LeafSolver` make this selectable rather than hidden.

### 3. Monotone constraints over stages

The library already has `tree_monotone` for split direction and `leaf_monotone` for leaf coefficients. A boosted stage needs a third thing.

**The problem.** A constraint that says "feature $f$ must not decrease the prediction" has to hold for the *sum of all stages*, not for each stage independently. If stage 1 raises the prediction on $f$ and stage 2 lowers it, the ensemble can violate the constraint even if each stage respects it.

**The mechanism XGBoost uses.** XGBoost enforces constraints during tree construction using leaf-weight ordering and propagated bounds. For a candidate split on feature $f$, with unconstrained leaf weights $w_L = -G_L/(H_L + \lambda)$ and $w_R = -G_R/(H_R + \lambda)$:

- an increasing constraint accepts only $w_L \le w_R$,
- a decreasing constraint accepts only $w_L \ge w_R$,
- a violating candidate gets $-\infty$ gain, or the constrained child weights are replaced by a common pooled weight.

Bound propagation then sets $m = (w_L + w_R)/2$; for an increasing feature the left child inherits upper bound $m$ and the right child inherits lower bound $m$; for a decreasing feature the bounds swap. Descendant leaf weights are clipped to the inherited $[L, U]$ interval.

XGBoost's official documentation warns that `hist` and `approx` can yield shallow trees under constraints, because constrained candidates may wipe out all splits, and recommends increasing `max_bin`.

**The library's design.** The existing `tree_monotone` already orders children by target mean for a constrained column. The booster adds the ensemble-level guarantee: a per-stage bound vector is carried forward, and each stage's leaf values are clipped to the bounds implied by the accumulated stages. The report states whether the constraint holds exactly and, if a stage was clipped, how often.

### 4. The facade

```text
ModelBoost
  kind: BoostKind      # gbdt | dart | linear | forest_stage | ordered
  mode: ModelMode      # regression | classification
```

| Trait | `gbdt` | `dart` | `linear` | `forest_stage` | `ordered` |
|---|---|---|---|---|---|
| `supports_predict_proba` | classification | classification | classification | classification | classification |
| `supports_oob` | with row subsample | with row subsample | no | no | no |
| `supports_early_stopping` | yes | yes | yes | yes | yes |
| `supports_staged_predict` | yes | yes | yes | yes | yes |
| `supports_apply` | yes | yes | yes | yes | yes |
| `supports_feature_importance` | yes | yes | yes | yes | yes |
| `supports_monotone_constraints` | yes | yes | yes | yes | yes |
| `supports_distributional_output` | no | no | no | no | no |

`supports_distributional_output` is false for every kind in the first wave. A distributional boosting stage needs a multi-parameter leaf and a joint Newton step, which is a different research problem from a scalar one. This is the same argument that puts distributional output in the forest family in Spec 31, and it is the honest answer.

### 5. Public surface

```python
from mltreelib.boost import ModelBoost

boost = ModelBoost(
    kind="gbdt",
    mode="regression",
    n_estimators=300,
    learning_rate=0.1,
    subsample=1.0,
    max_depth=3,
    min_samples_leaf=20,
    l2=1.0,                     # the lambda in the leaf weight
    max_features=1.0,
    split_eval="histogram",
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=None,
    tree_monotone=None,         # per-feature -1 / 0 / +1
)

boost.fit(X, y, weights=None)
boost.predict(X)
boost.predict_proba(X)          # trait-gated
boost.staged_predict(X)         # generator over stage predictions
boost.apply(X)                  # per-stage leaf index
boost.feature_importances(kind="impurity" | "permutation")
boost.report()
boost.config()
boost.save(path); ModelBoost.load(path)
```

`get_params` and `set_params` are absent, per Spec 25 Decision 11. There is no sklearn import or dependency.

### 6. DART, second wave

Rashmi and Gilad-Bachrach's Algorithm 1, verbatim from the paper:

```text
Let N be the total number of trees to be added to the ensemble
S1 ← {x, −L0(x)}
T1 be a tree trained on the dataset S1
M ← {T1}
for t = 2, ..., N do
    D ← the subset of M such that T ∈ M is in D with probability p_drop
    if D = ∅ then D ← a random element from M
    M̂ ← M \ D
    St ← {x, −L0(x) − M̂(x)}
    Tt be a tree trained on the dataset St
    M ← M ∪ {Tt}
    for T ∈ D do
        Multiply T in M by a factor of |D| / (|D| + 1)
    end for
end for
Output M
```

The dropped trees are scaled by $|D|/(|D|+1)$, and the new tree absorbs the correction. This is the "dropout" that regularises the ensemble by forcing later stages not to duplicate earlier ones.

**XGBoost's implementation differs from the paper in two ways**, both verified from `src/gbm/gbtree.cc`:

1. **Weighted dropping.** The paper drops uniformly with probability $p_{drop}$. XGBoost drops with probability proportional to each tree's weight: the drop condition is `runif(rnd) < rate_drop * weight_drop.size() * weight_drop[i] / sum_weight`. A `normalize_type` option controls the post-drop rescaling, with a variant computing `factor = num_drop / (num_drop + lr)`.
2. **Skip and count parameters.** `skip_drop` is the probability of skipping the dropout procedure entirely in an iteration, and `num_drop` caps the number of dropped trees.

**LightGBM's parameters**, verified from `docs/Parameters.rst`:

| Parameter | Default | Meaning |
|---|---:|---|
| `drop_rate` | 0.1 | Fraction of previous trees to drop, used only in `dart` |
| `max_drop` | 50 | Maximum trees dropped in one iteration; `<= 0` means no limit |
| `skip_drop` | 0.5 | Probability of skipping dropout in an iteration |
| `xgboost_dart_mode` | false | Switch to XGBoost's DART semantics |
| `uniform_drop` | false | Uniform dropping instead of weighted |
| `drop_seed` | 4 | Seed for the drop selection |

The `xgboost_dart_mode` flag exists precisely because LightGBM's default DART differs from XGBoost's, and a library implementing DART must pick one and document it. `mltreelib` implements the **paper's** uniform dropping as the default, with an explicit `drop_mode` control for weighted dropping.

**DART's reported result.** From the paper's Table 4, L2 error on the CT slices regression task, with the best parameter combination for each ensemble size:

| Ensemble size | 25 | 50 | 100 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| MART | 35.13 | 31.79 | 30.92 | 30.07 | 29.76 | 29.28 |
| DART | 32.50 | 30.50 | 29.66 | 28.14 | 28.11 | 27.98 |
| Random Forest | 32.76 | 33.21 | 32.88 | 32.36 | 32.66 | 32.33 |

DART beat MART at every ensemble size and beat random forest at every size. The paper's ranking task result was approximately 0.4 NDCG points over MART.

### 7. The other kinds

**`linear`.** The `gblinear` shape: each stage is a regularised linear model fitted by coordinate descent. XGBoost's `gblinear` defaults, verified from source: `eta` 0.5, `lambda` 0, `alpha` 0, `updater` `shotgun`, `feature_selector` `cyclic`, `top_k` 0. The update is a soft-thresholded proximal step with a zero-crossing clamp, `eta` shrinkage, an early-out when the Hessian sum falls below $10^{-5}$, and L1 applied as a threshold, with penalties denormalised by the instance-weight sum. `eta` 0.5 with `lambda` 0 makes the unregularised case a plain coordinate-descent least-squares solve.

**`forest_stage`.** Each stage is a random forest fitted to the negative gradient, and the stage's contribution is the forest's prediction. This is LightGBM's `boosting_type="rf"` shape. It is bagging inside boosting, and it needs the forest machinery from Spec 28. It is deferred until that lands.

**`ordered`.** CatBoost's ordered boosting with ordered target statistics. The mechanism, from the paper: ordered target statistics compute a categorical feature's encoding from a random permutation so a case's own target never enters its own statistic, which removes the prediction shift that plain GBDT suffers on categoricals; ordered boosting extends this with permutation-driven supporting models, so each stage sees statistics computed from a strictly earlier permutation rather than from the full data.

Verified CatBoost parameters: `boosting_type` is `Plain` or `Ordered`; `bootstrap_type` is `Bayesian`, `Bernoulli`, `MVS`, or `No`. The current CPU default is `Plain`. On GPU, the documented rule is `Plain` for MultiClass and MultiClassOneVsAll and for more than 50,000 objects, and `Ordered` otherwise; the exact release that introduced the 50,000-object threshold is unverified. The tree structure is oblivious, meaning every node at a given depth splits on the same feature and threshold.

This is a large project. It needs the library's categorical partition search to be replaced by, or complemented with, ordered target statistics, which changes the split engine rather than the leaf fit. It is deferred for that reason.

### 8. The bar this family must clear

`perpetual (Rust)` is the reference. A `ModelBoost` must beat it on at least one of these, on the local Elo board:

1. **Native missing-value routing.** The plain Rust booster has none. The library does, per member.
2. **Monotone constraints that hold exactly across the ensemble.** The plain Rust booster has none, and XGBoost's own documentation warns that constrained `hist` trees can end up shallow.
3. **Categorical handling with high cardinality.** The plain Rust booster has none.
4. **Distributional or survival leaves in a stage.** The plain Rust booster has none.

A `ModelBoost` that beats `perpetual` on none of these is a slower `perpetual` and should not be built.

## User Stories

### Core Boosting

1. As a user, I want `kind="gbdt"`, so that I get Friedman's gradient boosting with second-order stage updates.
2. As a user, I want the learning rate and stage count as separate controls, so that I can trade bias against variance.
3. As a user, I want early stopping with a validation fraction and a patience, so that I do not overfit the stage count.
4. As a user, I want `staged_predict`, so that I can see the ensemble improve stage by stage.
5. As a user, I want the training loss per stage reported, so that I can see convergence.
6. As a user, I want the best iteration recorded, so that I can restore the model at its best point.
7. As a user, I want row subsampling, so that I can trade variance for speed.
8. As a user, I want the stage count and the resulting tree size reported, so that model size is visible.

### Leaf Integration

9. As a user, I want stages fitted with the library's leaf models, so that a stage can carry a ridge-regularised linear fit.
10. As a user, I want the leaf family selectable per stage, so that a count target uses a Poisson leaf.
11. As a user, I want monotone constraints on leaf coefficients to hold across the whole ensemble, so that the constraint is a property of the model and not of one stage.
12. As a user, I want to know when a stage had to clip a leaf value to satisfy a constraint, so that I know the constraint bound.
13. As a user, I want missing values routed natively in every stage, so that the booster handles them like the rest of the library.
14. As a user, I want categorical splits honoured in every stage, so that high-cardinality categoricals work.

### DART

15. As a user, I want `kind="dart"`, so that I get dropout boosting with the paper's drop and rescale mechanics.
16. As a user, I want the drop rate and the drop-mode control explicit, so that I know whether dropping is uniform or weighted.
17. As a user, I want the skip-drop probability, so that I can control how often dropout is applied at all.
18. As a user, I want a cap on the number of dropped trees, so that a stage cannot wipe out the ensemble.

### Deferred Kinds

19. As a user, I want `kind="linear"` for a linear-model stage, so that I can boost a global linear model.
20. As a user, I want `kind="forest_stage"`, so that I can bag inside a boosting loop.
21. As a user, I want `kind="ordered"` with ordered target statistics, so that I can boost on categorical-heavy data without prediction shift.

### Diagnostics and Reporting

22. As a user, I want the report to name the aggregation rule, so that a booster is never mistaken for a forest.
23. As a user, I want the report to state the leaf estimator used in the stages, so that I know whether it is a Newton constant or a fitted model.
24. As a user, I want the constraint-satisfaction status reported, so that I know the monotone claim holds.
25. As a user, I want the fit time and the stage count reported, so that cost is visible.

## Implementation Decisions

### Decision 1: The Family Is Built, With a Named Bar

`ModelBoost` is built, and Decision 8 states the bar it must clear.

The reasoning is the local board. Boosting occupies the top of the ranking, and a Rust booster (`perpetual`) already sits third overall. But the library's own single tree is competitive on classification and under-ensembled on regression. Spec 28 addresses the regression gap with bagging, which is the cheaper and larger win. Building the boosting family as well is correct, because it covers a different weakness: bias on irregular targets where variance reduction is not enough.

The order across the whole ensemble layer is forest first, then boost, then meta, and that order comes from the board rather than from the literature.

### Decision 2: Stages Use the Existing Tree Engine, Not a Special Constant-Leaf Path

A stage is fitted with the library's tree builder on pseudo-residuals, using the selected leaf family and solver.

This is the decision that makes the family worth building for this library. XGBoost and LightGBM must special-case leaf fitting to a Newton constant, and therefore cannot offer linear leaves, monotone leaf coefficients, or distributional leaves in a stage. The library gets all three for free from `LeafKind`, `LeafFamily`, and `leaf_monotone`.

**The limitation is stated.** For squared error, the pseudo-residual least-squares leaf fit and the Newton-step constant coincide. For logistic loss, the pseudo-residual fit is an approximation to the Newton step unless the leaf family is the matching GLM. The report states which regime applies.

### Decision 3: The Gradient Contract Is Explicit, Not Implicit

Stage fitting computes gradients and, where the loss supplies them, second derivatives, through a `StageObjective` interface over the shared solver seam:

```text
StageObjective
  loss: LossKind                # squared | absolute | huber | logistic | poisson | gamma
  gradient(y, prediction) -> Vec<f64>
  hessian(y, prediction) -> Option<Vec<f64>>
  initial_estimate(y) -> f64
```

The interface is small on purpose. A stage needs the gradient at the current prediction, and optionally the curvature for the leaf-weight regularisation. Everything else is the tree engine's job.

### Decision 4: Monotone Constraints Are Checked at the Ensemble Level

The booster carries a per-feature bound vector across stages. Each stage's leaf values are clipped to the bounds implied by the accumulated stages, and the report states whether the constraint holds exactly.

Per-stage monotonicity alone does not give ensemble monotonicity, and a model that claims a monotone constraint must deliver it on the sum. XGBoost's own approach, leaf-weight ordering during construction plus bound propagation to descendants, is the reference. The library's `tree_monotone` already orders children by target mean for a constrained column, so the stage construction side exists; the ensemble-level guarantee is what the booster adds.

### Decision 5: Binned Evaluation Is a Prerequisite, Not an Optimisation

Stage fitting uses binned split evaluation by default. The binned path itself lands in **Spec 27, Decision 15 and Phase 1**, which lifts the `uses_class_histogram` gate to the regression gains and adds the criteria that specification introduces. This specification consumes it and adds the `split_eval` control.

**Why this is a prerequisite rather than a later optimisation.** A 300-stage booster over raw-value scans costs 300 times a single tree's fit time. The local board's mean fit times make the stakes plain: LightGBM(200) fits in 0.34 s and `perpetual (Rust)` in 7.19 s, while ranger takes 17.78 s. A booster that cannot use binned scans will be the slowest entrant on the board.

**Why it is not this specification's work.** The criteria that need the binned path are the regression criteria, and the regression families are Spec 27's subject. Spec 27 therefore owns the scan path, and this specification owns only the stage loop that uses it.

### Decision 6: `kind="gbdt"` Is the Only First-Wave Kind

DART lands second. `linear` and `forest_stage` land third. `ordered` is deferred.

Each has a specific blocker:

- **`dart`** needs the drop and rescale bookkeeping on top of the stage loop. Small, self-contained, and it has a clear paper. Second wave.
- **`linear`** needs a global linear stage fitted by coordinate descent, which is a different learner from a tree stage. It also needs the stage loop's prediction update to accept a non-tree contribution.
- **`forest_stage`** needs Spec 28's forest machinery inside a stage.
- **`ordered`** needs ordered target statistics, which changes the split engine rather than the leaf fit. Largest of the four.

### Decision 7: DART Implements the Paper, With an Explicit Alternative

The default is the paper's uniform dropping with the $|D|/(|D|+1)$ rescale. A `drop_mode` control exposes weighted dropping.

XGBoost's implementation drops proportionally to tree weight, and LightGBM defaults to its own variant with an `xgboost_dart_mode` flag to switch. Neither is the paper. A library that implements DART must pick one and say so, because the two produce different models. The paper is the default because it is the citable specification, and the weighted variant is available because it is what the strongest implementations ship.

### Decision 8: The Bar Is Measured Against `perpetual (Rust)`

`ModelBoost` must beat `perpetual (Rust)` on at least one of four properties that the plain Rust booster lacks:

1. Native missing-value routing.
2. Monotone constraints that hold exactly across the ensemble.
3. Categorical handling with high cardinality.
4. Distributional, survival, or quantile leaves in a stage.

`perpetual (Rust)` scores 1400.1 overall, 1505.9 regression, 1314.7 classification. A `ModelBoost` that matches those numbers without adding any of the four properties is a worse `perpetual`.

### Decision 9: `supports_distributional_output` Is False in the First Wave

A distributional boosting stage needs a multi-parameter leaf and a joint Newton step across the parameters.

This is the same boundary that puts distributional output in the forest family in Spec 31. The library's forest members inherit survival and quantile leaves for free; a booster cannot, because its stage update is a scalar Newton step. The report states this rather than implying the capability exists.

### Decision 10: No New Runtime Dependencies

The booster uses `rand`, `rand_chacha`, `rayon`, `faer`, `nalgebra`, and `statrs`, all already present, plus the Spec 26 solver seam for leaf fitting. XGBoost, LightGBM, CatBoost, and `perpetual` are benchmark-only oracles.

### Decision 11: One Multi-Core Job at a Time

Stage fitting is sequential by nature, but the stage's split scan parallelises across columns, and the fitted stages parallelise across prediction. The booster must not run concurrently with another multi-core fit.

## Exact Build Sequence

### Phase 0 — Stage Objective and Losses

1. Add `ensemble/boost_objective.rs` with the `StageObjective` interface and the `LossKind` set: squared, absolute, huber, logistic, poisson, gamma.
2. Add the gradient and, where defined, the Hessian for each loss.
3. Add the initial-estimate rule per loss, matching the library's existing GLM family conventions.
4. Add Rust tests asserting each gradient against a finite-difference approximation, and each Hessian against a finite-difference of its gradient.
5. Add Python tests for the loss selection and the reported objective value.

Gate: `cargo test` and Python tests pass. Every gradient matches its finite difference within tolerance.

### Phase 1 — Stage Loop

1. Add `ensemble/boost.rs` with the stage loop: compute the current prediction, derive pseudo-residuals, fit a stage tree with the existing builder, scale by the learning rate, add to the running prediction.
2. Add the learning rate, stage count, row subsample, and column subsample controls.
3. Add the per-stage training loss and the stage-count report.
4. Add `staged_predict`.
5. Add early stopping with a validation fraction and a patience, plus the best-iteration record.
6. Add Rust tests for loss decrease across stages on synthetic data, the learning-rate effect, and early stopping triggering.
7. Add Python tests for the stage loop, `staged_predict`, and the early-stopping report.

Gate: `cargo test` and Python tests pass. Training loss decreases monotonically on a well-specified synthetic problem.

### Phase 2 — Leaf and Constraint Integration

1. Route the stage's leaf fit through the shared solver seam with the selected leaf family and solver.
2. Add the ensemble-level monotone bound vector and the leaf-value clipping.
3. Report the constraint-satisfaction status and the clipping frequency.
4. Add the missing-value and categorical policies to the stage fit.
5. Add Rust tests asserting monotone constraints hold on the *summed* prediction across stages, and that the clipping frequency is reported.
6. Add Python tests for the monotone control and the constraint report.

Gate: `cargo test` and Python tests pass. A monotone-boosted model is monotone in the summed prediction, not merely per stage.

### Phase 3 — Report and Persistence

1. Add the boost-specific report sections: the aggregation rule, the leaf estimator, the constraint status, the stage losses, the best iteration, and the fit time.
2. Add `save`, `load`, and pickle round trips preserving the stages, the best iteration, and the constraint state.
3. Add Rust tests for report section presence and persistence round-trip fidelity.
4. Add Python tests for the report structure and the persistence round trip.

Gate: `cargo test` and Python tests pass. A reloaded booster reproduces identical predictions.

### Phase 4 — DART

1. Add the drop selection, the $|D|/(|D|+1)$ rescale, the skip probability, and the drop cap.
2. Add the `drop_mode` control with uniform as the default and weighted as the alternative.
3. Add the report fields for the drop statistics.
4. Add Rust tests asserting the rescale factor, the drop-set membership, and the skip behaviour.
5. Add Python tests for the DART controls and the drop report.

Gate: `cargo test` and Python tests pass. The dropped trees carry the paper's rescale factor exactly.

### Phase 5 — Elo Benchmark

1. Run the `gbdt` kind over the local 56-dataset Elo arena with the same protocol as the current board.
2. Report the regression Elo, the classification Elo, the overall Elo, and the mean fit time.
3. Compare against `perpetual (Rust)` at 1400.1 overall, 1505.9 regression, 1314.7 classification, and 7.19 s mean fit time.
4. Compare the mean fit time against LightGBM(200)'s 0.34 s.
5. State which of the four properties in Decision 8 the implementation delivers, and the measured cost of each.

Gate: the benchmark exists, its numbers are on the board, and the report states plainly whether the bar was cleared.

### Phase 6 — Deferred Kinds

1. `linear`: a coordinate-descent regularised linear stage.
2. `forest_stage`: a forest stage fitted to the negative gradient, using Spec 28's machinery.
3. `ordered`: CatBoost-style ordered boosting with ordered target statistics.

Each lands as its own phase with its own gate.

## Testing Decisions

- **Good tests assert external behaviour.** Tests cross the highest available seam. They verify recovered parameters, stage losses, constraint satisfaction, trait errors, report contents, and reproducibility. They do not assert private update formulas.
- **Gradient seam.** Assert every loss's gradient against a finite-difference approximation of its derivative, and every Hessian against a finite-difference of its gradient.
- **Stage-loop seam.** Assert the training loss decreases across stages on a synthetic problem where the answer is known, and that the final prediction matches a hand-computed two-stage fit.
- **Learning-rate seam.** Assert that halving the learning rate with the stage count fixed produces a smaller total change, and that the stage count and learning rate trade off as documented.
- **Early-stopping seam.** Assert early stopping triggers at the recorded best iteration and that the restored model reproduces the best-iteration predictions.
- **Monotone seam.** Assert the summed prediction is monotone on synthetic monotone data across multiple stages, and that a deliberately violated stage constraint is detected and reported.
- **Leaf seam.** Assert a stage fitted with a linear leaf produces coefficients, and that a stage fitted with a Poisson leaf on count data behaves correctly.
- **DART seam.** Assert the rescale factor is exactly $|D|/(|D|+1)$, that the drop set is drawn from the paper's uniform distribution, that the skip probability suppresses dropping at the documented rate, and that the drop cap holds.
- **Trait seam.** Assert every unsupported method raises a typed error naming the trait and kind. Assert `predict_proba` on a regression booster raises rather than returning a default.
- **Report seam.** Assert all four required sections are present, the Objective section names the aggregation rule as a shrunk sum of stages, and the constraint status is present.
- **Determinism seam.** Fit the same booster with `n_jobs=1` and `n_jobs=4`. Assert identical predictions and identical stage losses.
- **Persistence seam.** Save and reload a fitted booster. Assert identical predictions and an identical best iteration.
- **Benchmark smoke test.** Fit every kind on one deterministic dataset with a small budget. Verify output schema, counts, statuses, and artifact creation.
- **Benchmark oracles.** Compare the stage update against a fixed reference implementation of Friedman's update on a small dataset. Compare the loss values against a direct computation. Compare DART's drop mechanics against a direct implementation of the paper's Algorithm 1.
- **Prior art.** Reuse the existing deterministic-RNG tests, the weighted-fit tests, and the GLM family tests in `crates/mltreelib-core/tests/`.

## Out of Scope

- The bagging family. Spec 28.
- The meta-estimator family. Spec 30.
- Distributional, survival, and quantile outputs from a boosted ensemble. The forest family covers them, and a multi-parameter stage is a different research problem.
- Natural gradient boosting and multiparameter probabilistic outputs such as NGBoost.
- Bayesian posterior sampling. That is a separate family with a different inferential contract.
- Explainable Boosting Machine and GA²M. These are additive models with pairwise interactions, not tree ensembles. They belong with the GAM and interpretable-additive work, and they are the interpretability comparator, not a forest member.
- Tabular foundation models.
- Quantile and Huber boosting losses beyond the first wave's loss set, and the R-loss used for causal tuning.
- Any sklearn import, dependency, or `BaseEstimator` compatibility.

## Further Notes

- Research basis for the boosting mechanics, the DART algorithm, the CatBoost and LightGBM parameter semantics, the monotone-constraint implementations, and the API surfaces: `references/research/2026-11-22-boosting-methods-landscape.md`.
- The selection framework that decides when to reach for `ModelBoost` rather than `ModelForest` or `ModelMeta`: Spec 31.
- Spec 28 owns the shared Rust `ensemble/` module, the member lifecycle, and the cross-validation generator this specification builds on. Spec 27 owns the binned split-evaluation path for the regression criteria, which this specification consumes as a prerequisite.
- Spec 25 owns the common report contract and the sklearn compatibility policy.
- Spec 26 owns the generic bounded optimisation seam used for the leaf fit.
- Verified corrections to carry forward: DART is Rashmi and Gilad-Bachrach, arXiv:1505.01866, PMLR v38, not a Müller, Mentrup, and Guestrin NeurIPS 2015 paper; ChimeraBoost is the `bbstats/chimeraboost` package, not a paper; GA²M is Lou, Caruana, Gehrke, and Hooker, KDD 2013, DOI 10.1145/2487575.2487579.
- The local Elo board is the reference for the family's bar: `perpetual (Rust)` at 1400.1 overall, 1505.9 regression, 1314.7 classification, 7.19 s mean fit time.
- One multi-core job at a time.
