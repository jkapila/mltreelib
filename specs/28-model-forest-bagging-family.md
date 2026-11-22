# Spec 28: `ModelForest` — The Bagging Family

Status: proposed

Family: bagging. Members are grown on resampled training data and combined by averaging or voting.

Scope boundary: this specification owns the bagging family only. The boosting family lives in Spec 29 (`ModelBoost`). The meta-estimator family lives in Spec 30 (`ModelMeta`). The Generalized Random Forest lives in its own later specification, because its splitting rule cannot run on today's tree engine.

## Problem Statement

`mltreelib` has a strong single-tree core and no ensemble layer.

`crates/mltreelib-core/src/model_tree.rs` already provides CART growth, linear leaves with `l1` and `l2` penalties, monotone constraints on both splits and leaf coefficients, `interaction_groups`, per-node column subsampling (`col_sample`), a seeded RNG, honest sample splitting, `alpha` child balance, `imbalance_penalty`, categorical partition search, missing-value routing, `predict_adjusted` for k-NN residual correction, and thirty split criteria through `GainFn`, including `Sdr`, `Causal`, `LogRank`, `Mob`, `Ctree`, `MaxStat`, and `Hellinger`.

What it lacks:

- No `ensemble` module.
- No bootstrap or subsample sampler, and no per-tree seed derivation, so there is no reproducibility contract under parallel tree building.
- No in-bag count tracking, no out-of-bag predictions, no OOB error.
- No aggregation layer.
- No committee loop, no Cubist smoothing constant, no SD-based split rule, no extrapolation clipping.
- No forest facade. `python/mltreelib/anomaly.py` has `AnomalyTree`, `python/mltreelib/cluster.py` has `ClusterTree`, and there is no supervised ensemble surface at all.

The design question this specification answers is not "which forest algorithm is best". It is:

> **What is the smallest honest surface that covers the whole bagging family, keeps the richness of `ModelTree` intact inside every member, and leaves room for the boosting and meta families beside it?**

### The naming question

Earlier drafts proposed one umbrella class covering forests, committees, and boosting. That was wrong, for two reasons.

**First, the families are statistically different objects.** A forest averages independent members. A committee trains members sequentially on relabelled targets. A booster sums shrunk stages fitted to loss gradients. A meta-estimator learns a combination rule from out-of-fold predictions. One class with a `kind` switch over all four hides the difference that matters most to a user deciding what to use.

**Second, the parameters explode.** `ModelTree` already exposes leaf kind, leaf family, leaf solver, two penalty terms, two monotone vectors, interaction groups, split criterion, split test, split alpha, bonferroni, prune cost, categorical order and search, split precision, tail quantile, honesty and its fraction, alpha, imbalance penalty, col sample, seed, and NA policy. A forest multiplies every one of those by a per-kind configuration surface, and a flat union across families would accept invalid combinations that silently do nothing.

So there are three separate public classes, one per family, and one shared Rust module underneath:

```text
ModelForest   bagging family      Spec 28
ModelBoost    boosting family     Spec 29
ModelMeta     meta-estimator      Spec 30
```

```text
crates/mltreelib-core/src/ensemble/
  mod.rs        shared member lifecycle, seed derivation, in-bag counts, report types
  forest.rs     ModelForest
  boost.rs      ModelBoost          (Spec 29)
  meta.rs       ModelMeta           (Spec 30)
  sampler.rs    bootstrap, subsample, bootstrap weights
  aggregation.rs averaging, voting, weighted combination
  oob.rs        out-of-bag predictions and diagnostics
  cv.rs         cross-validation split generator
```

The Rust layer shares the mechanics that are genuinely common: member lifecycle, per-member seed derivation, in-bag bookkeeping, the cross-validation generator, and the report data model. The Python layer shares nothing but the report renderer.

## Solution

One facade, `ModelForest`, with a `kind` switch over the bagging family, one mode axis, trait-gated capabilities, and one shared report.

### 1. The bagging family and its members

| Kind | What it is | Member tree | Combination |
|---|---|---|---|
| `bagging` | Breiman bagging. Bootstrap replicates, uniform average | `ModelTree`, default leaf | Uniform mean, or plurality vote |
| `rf` | Breiman random forest. Bagging plus random feature subsetting per node | `ModelTree`, default leaf | Uniform mean, or plurality vote |
| `extra_trees` | Geurts et al. Extremely Randomized Trees. Random thresholds, default full-sample without bootstrap | `ModelTree`, default leaf | Uniform mean, or plurality vote |
| `committee` | Quinlan's Cubist committee. Sequential members on relabelled targets | `ModelTree` with linear leaves | **Unweighted** member average |
| `linear_forest` | Random forest whose members carry linear model leaves | `ModelTree` with linear leaves | Uniform mean |

### 2. The three-way distinction that must not blur

This is the conceptual core of the specification.

#### 2.1 Bagging family (this spec)

**Mechanism.** For each member $b$, draw a bootstrap replicate of the training data, fit a member on it, and combine members by averaging or voting.

**Purpose.** Variance reduction through averaging over an unstable learning procedure. The classic condition, from Breiman (1996), is that the base procedure is *unstable*: it changes materially when the learning set changes. Trees are unstable. Nearest neighbours are stable, and bagging them does not help.

The exact identity for squared error:

$$E[(Y - \phi_A(x))^2] = E[(Y - \phi(x,L))^2] - E_L[\operatorname{Var}_L(\phi(x,L) \mid x)],$$

where $\phi_A(x) = E_L[\phi(x,L)]$. Averaging removes part of the learning-set variance. Instability is the *mechanism*, not a finite-sample guarantee of improved test error.

**Implementation requirement that is easy to get wrong.** Every sampled index must be preserved *with its multiplicity*, and the multiplicity must be exposed. Drawing unique indices silently turns bagging into subsampling, which changes the variance structure and the out-of-bag membership.

**Guarantees.** None beyond prediction. Out-of-bag error is a useful diagnostic but is **not unbiased**: each OOB prediction averages only about 36.8% of the members, so finite-forest OOB error exceeds full-forest error, and the overestimation is parameter-dependent, worst with balanced classes, small $n$, many predictors, weak signal, and low predictor correlation. Selecting `max_features` by OOB error inherits that bias.

#### 2.2 Boosting family (Spec 29)

**Mechanism.** Sequential. Each stage is fitted to the negative gradient of the loss given the current ensemble, then added with shrinkage. The ensemble is a **sum** of shrunk stages.

**Distinction from bagging.** Boosting is bias-reducing; bagging is variance-reducing. Boosting stages are highly dependent by construction; bagging members are independent by construction. Boosting needs a learning rate and usually early stopping; bagging needs neither.

**Distinction from the Cubist committee.** The Cubist committee is also sequential, but it trains member $m \ge 2$ on the relabelled target $y' = 2y - \hat y_{m-1}$ and aggregates by an **unweighted average**. There is no learning rate, no stage weights, and no row resampling. The effective correction rate is $1/2$ with no decay across members, which is why Cubist recommends about 5 members and why the gain saturates. The committee is a bagging-family member by *combination rule* (an unweighted average) even though its *training* is sequential. This asymmetry is a real property of the method, not a modelling convenience.

#### 2.3 Meta-estimator family (Spec 30)

**Mechanism.** A second-level learner is fitted on out-of-fold predictions from base learners.

**Distinction from both.** Bagging and boosting change how the *base* model is trained. Stacking changes how *predictions are combined*, and it learns that combination rule. A stack can combine a forest with a booster with a linear model; a bag cannot, because bagging members all share one training procedure.

**Failure modes.** Meta-features must be out-of-fold. In-sample base predictions are optimistically biased, and the meta-learner learns misleading reliability. Base models refit on the full data create a train/test shift relative to the out-of-fold meta-features.

#### 2.4 The summary table

| Dimension | Bagging family | Boosting family | Meta family |
|---|---|---|---|
| Member training | Independent, parallel | Sequential | Independent base learners plus a fitted combiner |
| Row resampling | Bootstrap with replacement | Optional per stage | Cross-validation folds for meta-features |
| Base learner | One `ModelTree` type | One `ModelTree` type | Any mix of fitted models |
| Combination | Uniform mean or vote | Sum of shrunk stages | Learned rule on out-of-fold predictions |
| Learning rate | None | Yes | None |
| Needs held-out data | No | No, but early stopping wants it | Yes, by construction |
| Error direction reduced | Variance | Bias | Neither directly; combines |
| Natural scale | 50 to 5000 members | 100 to 5000 stages | As many base learners as fit |
| Out-of-bag | Yes, with known optimism | Only with row subsample | Not applicable; out-of-fold instead |
| Inference | None | None | None |

### 3. `ModelTree` is the member, and that is the whole point

The user's framing was right: if the member is not `ModelTree`, the forest loses the library's actual advantage.

`ModelTree` is not a CART implementation with extra options. It is a tree whose leaves carry fitted models and whose split vocabulary spans thirty criteria:

- Leaf kinds: `Linear`, `Mean`, `Median`, `Quantile(tau)`, `Huber`, `Mars`, `Prob`, `Odds`, `LogOdds`, `Knn`, plus family-parameterised variants including `Survival`, `Heteroscedastic`, `Aft`, `Cure`, and `Spline`.
- Leaf families: Gaussian, Binomial, Probit, Poisson, Gamma, Tweedie, Tobit.
- Leaf solvers: IRLS, L-BFGS, CCD, CMA-ES, SHADE, sNES, GA.
- Split criteria: `Sse`, `Deviance`, `FriedmanMse`, `AbsoluteError`, `Huber`, `Poisson`, `Entropy`, `Gini`, `GainRatio`, `Sdr`, `Twoing`, `Causal`, `LogRank`, `LogRankWeighted`, `GrayTest`, `SurvivalBrier`, `PseudoValue`, `Rmst`, `AftDeviance`, `Tweedie`, `Heteroscedastic`, `Tobit`, `Cv`, `Martingale`, `Mob`, `Ctree`, `EnergyDistance`, `Mmd`, `Wasserstein2`, `FairConstrained`, `Hellinger`, `MaxStat`, and `Custom`.
- Monotone constraints on splits and on leaf coefficients separately.
- Missing-value routing with per-node stored direction.
- Categorical partition search with target-ordered or exhaustive policy.
- `extract_rules` for human-readable rule sets.
- `shapley`, `break_down`, `permutation_importance`, `feature_importances`.
- `predict_adjusted` for k-NN residual correction.

**Decision.** `ModelForest` holds `Vec<ModelTree>`. Members are fitted `ModelTree` objects, not opaque prediction closures and not capability flags. A forest of `ModelTree`s therefore gets survival leaves, causal leaves, monotone constraints, categorical splits, missing-value routing, rule extraction, and per-member SHAP for free.

**Consequence.** The `committee` and `linear_forest` kinds differ from `rf` and `extra_trees` only in the leaf configuration and in the combination rule, not in the member type. This is why `ModelTree` is the right base and a generic `Tree` interface is not.

**What stays out of the member.** The `grf` kind needs a per-node relabeling step that today's `GainFn::Custom` cannot express: GRF solves a local moment at the parent node, derives a pseudo-outcome per row, and then splits on the relabelled values, and `Custom` only sees the left and right `(y, w)` slices. GRF therefore gets its own later specification, not a half-working member here.

### 4. The facade

#### 4.1 Two axes

```text
ModelForest
  kind: ForestKind    # bagging | rf | extra_trees | committee | linear_forest
  mode: ModelMode     # regression | classification
```

`kind` selects the algorithm family within bagging. `mode` selects the task. `kind` never means a task and `mode` never means an algorithm. This follows the tidymodels `parsnip` model-type and mode split.

#### 4.2 Trait-gated capabilities

Every optional operation is guarded by a trait query, so callers branch on capability and never on the kind string. A trait violation raises a typed error naming the trait and the kind.

| Trait | `bagging` | `rf` | `extra_trees` | `committee` | `linear_forest` |
|---|---|---|---|---|---|
| `supports_predict_proba` | classification | classification | classification | no | classification |
| `supports_oob` | yes | yes | yes | no | yes |
| `supports_apply` | yes | yes | yes | yes | yes |
| `supports_decision_path` | yes | yes | yes | yes | yes |
| `supports_feature_importance` | yes | yes | yes | yes | yes |
| `supports_rules` | yes | yes | yes | yes | yes |
| `supports_smoothing` | no | no | no | yes | no |
| `supports_extrapolation_limit` | no | no | no | yes | no |
| `supports_composite_correction` | no | no | no | yes | no |
| `supports_predict_interval` | no | no | no | no | no |
| `supports_inference` | no | no | no | no | no |

Two gaps worth stating explicitly. `committee` has no out-of-bag error because its members are trained sequentially on relabelled targets, so the bagging notion of "this row was not in my training sample" does not transfer. And `predict_interval` is deliberately absent for every kind in this specification: split-conformal intervals need out-of-bag nonconformity scores, which only the bagging family with bootstrap sampling produces, so conformal intervals land with the forest's OOB machinery in a later phase rather than being stubbed here.

#### 4.3 Public surface

```python
from mltreelib.forest import ModelForest

forest = ModelForest(
    kind="rf",                 # bagging | rf | extra_trees | committee | linear_forest
    mode="regression",         # regression | classification
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",       # "sqrt" | "log2" | int | float
    bootstrap=True,
    max_samples=None,
    random_state=None,
    # member tree configuration, forwarded to ModelTree
    tree=ModelTree(...),       # or leaf / leaf_family / leaf_solver fields
    # kind-specific, validated per kind
    committee_size=5,
    smoothing=None,            # {"kind": "quinlan_backup", "k": 15.0} | {"kind": "cubist_shrink"}
    extrapolation_limit=None,
    composite_neighbors=0,
)

forest.fit(X, y, weights=None)
forest.predict(X)
forest.predict_proba(X)            # trait-gated
forest.apply(X)                   # per-member leaf index
forest.decision_path(X)            # per-member node paths
forest.oob_score()                 # trait-gated
forest.oob_predictions()           # trait-gated
forest.feature_importances(kind="impurity" | "permutation")
forest.extract_rules()             # per-member rule sets
forest.report()                    # Objective / Learner / Controller / Diagnostics
forest.config()
forest.save(path); ModelForest.load(path)
```

Naming follows `parsnip`'s cross-engine harmonisation: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `max_samples`, `random_state`. R's `num.trees`, `ntree`, and `n.trees` normalise at the adapter boundary only.

`get_params` and `set_params` are absent, per Spec 25 Decision 11. There is no sklearn import or dependency.

#### 4.4 Rust layout

```text
crates/mltreelib-core/src/ensemble/
  mod.rs          shared member lifecycle, per-member seeds, report data model
  sampler.rs      bootstrap with multiplicity, subsample, bootstrap weights
  aggregation.rs  uniform mean, plurality vote, probability averaging, weighted combination
  oob.rs          OOB membership, OOB predictions, OOB error and its caveats
  cv.rs           K-fold, leave-one-out, Monte Carlo splits with stable fold identity
  forest.rs       ModelForest: bagging, rf, extra_trees, committee, linear_forest
  report.rs       forest-specific report sections
```

The Rust core owns numerical behaviour and persisted state. Python stays a thin binding.

## User Stories

### Facade and Lifecycle

1. As a Python user, I want one `ModelForest` class with a `kind` switch, so that I select a bagging variant without changing workflow code.
2. As a Python user, I want `kind` and `mode` as separate axes, so that classification and regression never share one overloaded switch.
3. As a Python user, I want unsupported kind and mode combinations rejected at validation, so that an invalid request never silently changes semantics.
4. As a Python user, I want kind-specific parameters validated against the selected kind, so that irrelevant knobs cannot be set by mistake.
5. As a user, I want `fit` to return the fitted facade, so that one lifecycle works across kinds.
6. As a user, I want `fit` to accept optional row weights, so that survey weights work in one place.
7. As a user, I want `predict` to use only fitted state, so that prediction never refits preprocessing.
8. As a user, I want `config()` to return the effective configuration, so that I can inspect settings without sklearn's `get_params`.
9. As a user, I want `report()` to return one structured report, so that I can inspect every forest consistently.
10. As a user, I want `save`, `load`, and pickle round trips to preserve fitted state and diagnostics, so that deployed forests are reproducible.
11. As a library maintainer, I want no sklearn import or dependency, so that the package keeps its own lifecycle.
12. As a library maintainer, I want `ModelForest`, `ModelBoost`, and `ModelMeta` to be separate public classes, so that no single class accumulates every family's parameters.

### Member Richness

13. As a user, I want the forest to expose its fitted member trees, so that I can inspect each member's splits, leaves, and rules.
14. As a user, I want leaf model coefficients per member, so that a linear-leaf forest is inspectable rather than opaque.
15. As a user, I want monotone constraints honoured inside every member, so that a monotone forest is monotone.
16. As a user, I want missing-value routing honoured inside every member, so that a forest handles missing values the same way a single tree does.
17. As a user, I want categorical split policy honoured inside every member, so that high-cardinality categoricals work.
18. As a user, I want per-member rule extraction, so that I can read the forest as a rule collection.
19. As a user, I want per-member SHAP and break-down attribution, so that explanation is available for the forest, not only for a single tree.
20. As a library maintainer, I want the member type to stay `ModelTree`, so that new leaf families reach the forest without forest changes.

### Bagging Mechanics

21. As a data scientist, I want `kind="bagging"`, so that I get Breiman's original bagging with a uniform average.
22. As a data scientist, I want bootstrap multiplicity preserved per member, so that the variance structure and OOB membership are correct.
23. As a data scientist, I want the aggregate to be a uniform mean for regression, so that the method matches the published definition.
24. As a data scientist, I want plurality voting for classification, so that I match the published definition.
25. As a data scientist, I want probability averaging when members expose class distributions, so that I get calibrated probabilities rather than vote proportions.
26. As a data scientist, I want `max_samples` to control the bootstrap size, so that I can trade bias for variance.
27. As a data scientist, I want `bootstrap=False` to mean sampling without replacement, so that pasting is reachable and distinguishable from bagging.

### Random Forest and ExtraTrees

28. As a data scientist, I want `kind="rf"`, so that I get Breiman's random forest with random feature subsetting per node.
29. As a data scientist, I want `max_features` as `sqrt`, `log2`, an integer, or a fraction, so that I control tree correlation directly.
30. As a practitioner, I want `kind="extra_trees"` with full-sample default sampling, so that I reproduce Geurts et al. rather than ranger's approximation.
31. As a practitioner, I want the ExtraTrees split rule available on the `rf` kind too, so that both behaviours are reachable.
32. As a practitioner, I want a named unordered-factor policy, so that categorical handling is explicit rather than implied.
33. As a practitioner, I want a named missing-value policy with per-node stored direction, so that missing handling is inspectable.

### Cubist Committee

34. As a data scientist, I want `kind="committee"`, so that I get Quinlan's sequential model-tree committee.
35. As a data scientist, I want the member-2-onward target relabelling $2y - \hat y_{m-1}$ implemented exactly, so that my results match the published mechanism.
36. As a data scientist, I want the final prediction to be the unweighted member average, so that the committee is not silently turned into boosting.
37. As a data scientist, I want the Quinlan backup smoothing with $k = 15$ available, so that boundary discontinuities are reduced as in M5.
38. As a data scientist, I want the Cubist variance-shrink smoothing available as a separate named mode, so that the two algorithms are not conflated.
39. As a data scientist, I want no parameter named `m0`, so that the smoothing semantics are unambiguous.
40. As a data scientist, I want an extrapolation range limit, so that linear leaves cannot produce absurd values outside the training range.
41. As a data scientist, I want the non-negative sign rule for leaves whose targets are all one sign, so that rates and prices stay non-negative.
42. As a data scientist, I want the composite k-NN correction with `composite_neighbors=0` as the default, so that the correction is opt-in as in the CRAN package.
43. As a data scientist, I want the committee size default to be 5, so that it matches the published recommendation.
44. As a data scientist, I want the SD-based split rule on residuals of the node's own linear model, so that the growth behaviour matches Cubist rather than textbook Quinlan.

### Linear-Leaf Forest

45. As a data scientist, I want `kind="linear_forest"`, so that I get a forest whose members carry linear model leaves.
46. As a data scientist, I want the split criterion evaluated under the linear leaf model rather than under CART impurity, so that the partition matches the leaf that will be fitted.
47. As a data scientist, I want ridge regularisation on leaf coefficients, so that small leaves do not overfit.
48. As a data scientist, I want extrapolation clipping on linear leaves, so that out-of-range predictions stay bounded.

### Out-of-Bag and Diagnostics

49. As a user, I want exact out-of-bag membership from the preserved bootstrap multiplicities, so that OOB predictions are computed from the true member set.
50. As a user, I want the OOB error report to state its optimism caveat, so that I do not treat it as unbiased.
51. As a user, I want a warning when rows were never out-of-bag, so that missing OOB predictions are explained rather than silently zero.
52. As a user, I want both MDI and permutation importance reported separately, so that neither measure's bias is hidden.
53. As a user, I want a null predictor to show near-zero permutation importance while retaining non-zero MDI, so that the documented MDI bias is observable.
54. As a user, I want the fit time and prediction time reported, so that cost is visible.
55. As a user, I want the tree count, depth distribution, and leaf size distribution reported, so that capacity is visible.
56. As a user, I want the aggregation rule named in the report objective, so that a bagging forest is never mistaken for a committee or a booster.

### Reproducibility and Parallelism

57. As a user, I want identical results for one worker and many workers, so that parallel training is safe.
58. As a user, I want per-member seeds stored in the fitted model, so that a reloaded forest reproduces exactly.
59. As a user, I want bootstrap indices stored in the fitted model, so that the aggregate can be audited.

### Later Coverage

60. As a data scientist, I want a survival forest, so that I can ensemble survival leaves with Nelson-Aalen estimation.
61. As a data scientist, I want a causal forest, so that I can ensemble uplift leaves for heterogeneous treatment effects.
62. As a data scientist, I want a forecasting forest, so that I can ensemble multi-horizon leaves.
63. As a user, I want these later kinds to reuse the same member and aggregation machinery, so that no second forest implementation appears.

## Implementation Decisions

### Decision 1: Three Families, Three Public Classes

`ModelForest` covers bagging. `ModelBoost` covers boosting. `ModelMeta` covers meta-estimators. No umbrella class holds all three.

The reason is not taste. The families differ in what the user is choosing between: a variance-reduction method, a bias-reduction method, and a combination-rule learner. A single class with a kind switch over all of them would have to accept, for one call, either an OOB error flag or a learning rate or a CV splitter, and would have to answer "which family am I in" before it could validate the rest.

### Decision 2: Shared Rust Module, Separate Python Facades

The Rust layer shares what is genuinely common: member lifecycle, per-member seed derivation, in-bag bookkeeping, the cross-validation split generator, and the report data model. The Python layer shares only the report renderer.

The shared Rust pieces are few because most of what looks common is not. The sampler differs between bagging (bootstrap with multiplicity) and boosting (per-stage row subsample). The aggregation differs (mean versus shrunk sum versus learned rule). The diagnostics differ (OOB versus stage loss versus out-of-fold meta-features). Only the plumbing is common, and the plumbing is what causes bugs when duplicated.

### Decision 3: `ModelForest` Holds `Vec<ModelTree>`

Members are fitted `ModelTree` objects. Not opaque prediction closures, not a generic `Tree` trait, not capability flags.

Every richness the library already has survives into the forest: leaf models, leaf families, monotone constraints, missing-value routing, categorical partitions, interaction groups, split criteria, rule extraction, SHAP, break-down, permutation importance, and the k-NN residual correction path.

A generic `Tree` interface was considered and rejected. It would have to be wide enough to cover linear leaves, survival leaves, and rule extraction, at which point it is `ModelTree` with extra indirection.

### Decision 4: `kind` and `mode` Are Separate Axes

`kind` selects the bagging algorithm. `mode` selects the task. This follows `parsnip`'s model-type and mode split. The rule is that `kind` never means a task.

### Decision 5: Traits, Not `kind` Checks

Every capability is a trait query on the fitted forest. User code branches on the trait. The trait table in §4.2 is normative.

This follows the Julia MLJ trait pattern. It is what keeps a `kind` switch from producing fragile user code.

### Decision 6: No Flat Parameter Union

Each kind has its own validated parameter set. The common set holds only parameters that mean the same thing across every kind: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `max_samples`, `random_state`.

`committee` adds `committee_size`, `smoothing`, `extrapolation_limit`, `composite_neighbors`. Setting a parameter that does not belong to the selected kind is a validation error, not a stored no-op.

### Decision 7: The `committee` Kind Is in the Bagging Family

The Cubist committee trains members sequentially but aggregates by an unweighted average. The family assignment follows the combination rule, because that is what determines the error characteristics and the aggregation machinery.

This is worth naming explicitly because it will look like a mistake. The spec states the reason in the table: committee training is sequential, committee combination is an average. Both facts are true and they are in different columns.

### Decision 8: Two Named Smoothing Modes, No `m0`

```text
SmoothingKind::QuinlanBackup { k: f64 }   # default k = 15.0
SmoothingKind::CubistShrink                # variance-minimising parent shrink
```

Quinlan backup blends predictions as $p' = (n p + k q)/(n + k)$, or blends coefficients as in Frank et al. (1998), giving $\big(\tfrac{na}{n+k}, \tfrac{nb+kc}{n+k}, \tfrac{kd}{n+k}\big)$ for a leaf $p = ax + by$ under a parent $q = cy + dz$.

Cubist shrink computes the variance-minimising convex combination against the parent's already-smoothed model, using the current and parent residual variances and their covariance, then blends coefficients and zeroes coefficients below $0.01 \cdot \text{GlobalSD}/\text{AttSD}_a$.

There is no parameter named `m0`. The constant in Quinlan's algorithm is $k$. A grep across the Cubist GPL source returns zero hits for `m0` or `M0`.

### Decision 9: Bootstrap Multiplicity Is Preserved and Exposed

Each member stores the bootstrap index vector with duplicates. The out-of-bag machinery derives membership from it, and the aggregate divides by the member count, not by the unique-row count.

Drawing unique indices silently turns bagging into subsampling and changes both the variance structure and the out-of-bag membership. This is the single most common implementation error in hand-rolled bagging.

### Decision 10: The Report Names the Aggregation Rule

The Objective section states the aggregation rule in words: "uniform mean of $B$ independent bagged model trees", or "unweighted average of $M$ sequential committee members trained on relabelled targets $2y - \hat y_{m-1}$".

This is the fastest way for a reader to tell a committee from a booster, and it prevents the family boundary from being invisible in the output.

### Decision 11: OOB Error Is a Diagnostic, Not a Performance Claim

The OOB report states the optimism caveat in substance: each OOB prediction averages only about 36.8% of the members, so finite-forest OOB error exceeds full-forest error; the overestimation is parameter-dependent; `max_features` selected by OOB error inherits that bias. Repeated cross-validation or a holdout is the recommended route for final performance claims.

### Decision 12: Per-Member Seed Derivation

Member $b$ uses a seed derived from the forest seed and the member index. The derived seeds and the bootstrap index vectors are stored in the fitted model. Thread scheduling never affects randomness.

### Decision 13: The Cross-Validation Generator Is Shared, Not Duplicated

The K-fold, leave-one-out, and Monte Carlo split generators live in `ensemble/cv.rs` because Spec 30's stacking needs stable fold identity to keep meta-feature columns aligned, and because a fold generator duplicated across two specs is a fold generator that will drift.

`mltreelib` already has a cross-validation surface for the survival and calibration work. The ensemble split generator must interoperate with it rather than fork it.

### Decision 14: GRF Is Deferred, Not Half-Built

The GRF kind is not in this specification. Its splitting rule requires a per-node relabeling step that `GainFn::Custom` cannot express, and a half-implemented GRF that silently falls back to CART splitting would be worse than none, because the resulting model would carry GRF's inference claims without GRF's mechanism.

GRF gets its own specification with its own splitting design.

### Decision 15: Later Forest Kinds Reuse This Machinery

Survival forest, causal forest, and forecasting forest are named in the user stories as later coverage. They reuse the member type, the sampler, the aggregation layer, the OOB machinery, and the report. No second forest implementation appears.

### Decision 16: No New Runtime Dependencies

The forest layer uses `rand`, `rand_chacha`, `rayon`, `faer`, `nalgebra`, and `statrs`, all already present. Reference implementations — `ranger`, the Cubist C source, `caret`, `baguette`, `scikit-learn` — are benchmark-only oracles, never runtime dependencies.

### Decision 17: Binned Split Evaluation Is a Dependency, Not This Spec's Work

The forest needs binned split evaluation. It does not build it.

**The requirement.** The forest exposes a `split_eval` control defaulting to `Histogram`, and every member is fitted on the binned path.

**Where it lands.** **Spec 27, Decision 15 and Phase 1.** The criteria that need the binned path are the regression criteria — `Sse`, `FriedmanMse`, `Sdr`, `AbsoluteError`, `Huber`, `Deviance`, plus the pinball, asymmetric-squared, group-aware, and negative binomial and beta criteria that Spec 27 introduces. Spec 27 owns the scan path because the regression criteria are its subject; splitting the change across specifications would make the same edit twice.

**Why the forest cannot ship without it.** On the local regression board, `LightGBM(200)` fits in 0.34 s mean while `ranger` takes 17.78 s and `sklearn RF` takes 18.56 s, on comparable regression Elo of 1542.4, 1700.6, and 1686.2. The difference is not the algorithm. LightGBM bins each numeric feature once and scans the binned histogram at every node; the other two rescan raw values at every node. A 50× fit-time gap comes almost entirely from binning. A 200-member forest over the library's current raw-only scan costs roughly 200 times a single tree's fit time, which at the current 4 s single-tree fit time is over 13 minutes per fit on the regression board's scale. That is not shippable.

**Consequence for this specification's phase order.** The binned path is a **prerequisite**, not a parallel workstream. Spec 27 Phase 1 lands before this specification's Phase 0, or this specification's Phase 0 cannot meet its fit-time expectations. The forest's Phase 0 consumes the existing scan path and adds the `split_eval` control; it does not change the scan path itself.

### Decision 18: The Forest Is the Measured Fix for the Library's Largest Gap

`ModelTree(all)` scores 1290.1 overall Elo on the local 56-dataset arena, ninth of 47 entrants. Its split is stark:

- **Classification Elo 1204.7.** Ahead of sklearn GBM (1188.6), HistGradientBoosting (1173.2), LightGBM(200) (1155.5), ExtraTrees (1084.1), and `forust (Rust)`.
- **Regression Elo 715.9.** Far behind ChimeraBoost (1763.7), ranger (1700.6), sklearn RF (1686.2), ExtraTrees (1640.4), and LightGBM(200) (1542.4).

A single tree has high variance on regression and comparatively low variance on classification. Squared-error loss punishes variance directly; accuracy on a separable problem tolerates a wobbly boundary. The tree is therefore **under-ensembled on regression**, not badly modelled.

Bagging is the direct remedy. The target is a `ModelForest(kind="rf")` over today's `ModelTree` landing near ranger's 1700.6 regression Elo rather than ModelTree's 715.9. This is the single largest measured improvement available to the library, and it is the reason the bagging family comes first.

**The testable prediction.** Phase 1 must report the forest's regression Elo against the single tree's 715.9. If a forest over the same tree does not close a large part of that gap, the cause is not variance and the tree needs different work, not ensembling. The benchmark distinguishes those two cases, so the claim stays honest.

### Decision 19: One Multi-Core Job at a Time

Forest training parallelises across members. It must not run concurrently with another multi-core fit. The driver scripts serialise fits.

## Exact Build Sequence

Each phase lands as its own commit with focused tests. A phase does not start before the previous phase passes its gate.

### Phase 0 — Shared Ensemble Skeleton

**Prerequisite:** Spec 27 Phase 1 has landed, so the binned scan path covers the regression gains. This phase consumes it; it does not change the scan path.

1. Add `ensemble/mod.rs` with the shared member lifecycle, per-member seed derivation, in-bag bookkeeping, and the report data model.
2. Add `ensemble/sampler.rs` with bootstrap preserving multiplicity, subsample without replacement, and bootstrap weights.
3. Add `ensemble/cv.rs` with K-fold, leave-one-out, and Monte Carlo splits with stable fold identity.
4. Add `ensemble/aggregation.rs` with uniform mean, plurality vote, and probability averaging.
5. Add `ensemble/oob.rs` with OOB membership derivation, OOB predictions, and OOB error.
6. Add a `split_eval` control to the forest, defaulting to `Histogram`, and report the scan path and bin count each member used.
7. Add Python `ModelForest` with `kind`, `mode`, `fit`, `predict`, `apply`, `decision_path`, `oob_score`, `oob_predictions`, `feature_importances`, `config`, `report`.
8. Add Rust tests for determinism, single-worker versus multi-worker equality, exact OOB membership from multiplicities, aggregation correctness, and that a member fitted with `split_eval=Histogram` selects the same split as one fitted with `split_eval=Fast` on data falling on bin boundaries.
9. Add Python tests for the facade lifecycle, trait gating, and validation errors.

Gate: `cargo test` and Python tests pass. `n_jobs=1` and `n_jobs=4` produce identical predictions.

### Phase 1 — Bagging and Random Forest

1. Add `ensemble/forest.rs` with the `bagging` and `rf` kinds over the existing `ModelTree` builder.
2. Add the `max_features` handling for `sqrt`, `log2`, integer, and fraction.
3. Add the ExtraTrees split rule as a `GainFn` variant or a split-scan flag.
4. Wire `kind="extra_trees"` with full-sample default sampling.
5. Add the named unordered-factor and missing-value policies, reusing the existing categorical and missing handling.
6. Add Rust tests for bootstrap multiplicity, OOB membership, feature subsetting effect on tree correlation, and ExtraTrees threshold randomness.
7. Add Python tests for the `rf` and `extra_trees` parameters and the OOB report caveat.

Gate: `cargo test` and Python tests pass. A bootstrap member's stored indices contain duplicates and OOB membership matches the multiplicities exactly.

### Phase 2 — Cubist Committee

1. Add the committee kind to `ensemble/forest.rs` with sequential member training and relabelled targets.
2. Add `ensemble/smoothing.rs` with `QuinlanBackup` and `CubistShrink`.
3. Add `ensemble/clamping.rs` with the extrapolation range limit and the sign rule.
4. Add `ensemble/composite.rs` with the composite k-NN correction and the `composite_neighbors=0` default.
5. Add the SD-based split rule as a `GainFn` variant, computed on residuals of the node's own linear model.
6. Wire the committee kind with its own validated parameters.
7. Add Rust tests for member relabelling, unweighted averaging, both smoothing modes, clamping, the sign rule, and composite correction.
8. Add Python tests for the committee parameters and the report objective wording.

Gate: `cargo test` and Python tests pass. Member predictions match the relabelling formula exactly and the aggregate is the unweighted mean.

### Phase 3 — Linear-Leaf Forest

1. Add the `linear_forest` kind with linear leaf members.
2. Add the split criterion evaluated under the linear leaf model rather than CART impurity.
3. Add ridge regularisation on leaf coefficients and the small-leaf guard.
4. Add extrapolation clipping for linear leaves.
5. Add Rust tests for the split criterion under a linear leaf, coefficient recovery on synthetic data, and clipping bounds.
6. Add Python tests for the linear-leaf forest parameters.

Gate: `cargo test` and Python tests pass. The linear-leaf split criterion selects the split that minimises the linear-leaf fit error.

### Phase 4 — Report Polish and Feature Selection

1. Add `ensemble/report.rs` with the forest Objective, Learner, Controller, and Diagnostics sections and the trait table.
2. Add the OOB optimism caveat to the report.
3. Add warnings for never-OOB rows, empty leaves, and extrapolation clipping frequency.
4. Add recursive forest feature selection with ranking-stability reporting.
5. Add Rust tests for report section presence and for feature-selection stability on synthetic data.
6. Add Python tests for the report structure.

Gate: `cargo test` and Python tests pass. All four report sections are present for every kind and the aggregation rule is named in the objective.

### Phase 5 — Later Coverage

1. Survival forest: Nelson-Aalen leaf estimation, aggregate over members.
2. Causal forest: uplift leaves, aggregate treatment effects.
3. Forecasting forest: multi-horizon leaves.
4. Quantile forest through the GRF-style forest weights, in its own later specification alongside GRF.

Each lands as its own phase with its own gate, reusing the Phase 0 to Phase 4 machinery.

### Phase 6 — Elo Benchmark

1. Run the `rf` kind over the local 56-dataset Elo arena with the same protocol as the current board.
2. Report the regression Elo, the classification Elo, the overall Elo, and the mean fit time.
3. Compare against the single tree's 715.9 regression Elo, 1204.7 classification Elo, and 1290.1 overall Elo.
4. Compare the mean fit time against ranger's 17.78 s and LightGBM(200)'s 0.34 s on the regression board.
5. Record the inter-member correlation and the member count needed to reach ranger's regression Elo.

Gate: the benchmark exists, its numbers are in the board, and the report states whether the forest closed the regression gap.

## Testing Decisions

- **Good tests assert external behaviour.** Tests cross the highest available seam. They verify recovered parameters, predictions, weights, intervals, trait errors, report contents, and reproducibility. They do not assert private update formulas or internal vector layout.
- **Primary Rust seam.** Fit each kind through the forest entry point on deterministic synthetic data. Assert predictions, OOB membership, and status.
- **Primary Python seam.** Fit each kind through `ModelForest`. Assert trait gating, report sections, config output, and persistence.
- **Determinism seam.** Fit the same forest with `n_jobs=1` and `n_jobs=4`. Assert identical predictions, identical OOB membership, identical importance values, and identical stored seeds.
- **Persistence seam.** Save and reload a fitted forest. Assert identical predictions, identical per-member seeds, and identical bootstrap index vectors.
- **Sampling seam.** Assert each member's stored indices contain duplicates when the bootstrap draws them, and that OOB membership matches the multiplicities exactly. Assert that `bootstrap=False` produces distinct rows per member.
- **Member richness seam.** Assert that a member's leaf coefficients, split list, rule extraction, and per-member SHAP are all reachable from the fitted forest, and that they match the same tree fitted standalone.
- **Monotone seam.** Fit a monotone forest on synthetic monotone data. Assert every member respects its monotone constraints and the aggregate is monotone.
- **Missing seam.** Fit a forest with missing values in the split column. Assert each member routes missing rows by its own stored direction and the aggregate is finite.
- **Categorical seam.** Fit a forest with a high-cardinality categorical column. Assert each member applies the configured categorical policy.
- **Bagging seam.** Assert the aggregate equals the mean of member predictions, and the classification aggregate equals plurality vote over member labels.
- **RF seam.** Assert the `max_features` subsetting reduces inter-member correlation relative to `bagging` with the same seed, and that `max_features="sqrt"` selects the documented count.
- **ExtraTrees seam.** Assert thresholds are drawn randomly rather than optimised, and that the `extra_trees` kind defaults to full-sample sampling with no bootstrap.
- **Committee seam.** Assert member 2 onwards is trained on $2y - \hat y_{m-1}$ and that the final prediction is the unweighted member average. Assert the smoothing modes differ in output and neither is named `m0`. Assert clamping bounds and the sign rule. Assert the composite correction changes predictions only when enabled.
- **Linear-leaf forest seam.** Assert the split criterion under the linear leaf differs from CART impurity on data where the two disagree, and that leaf coefficients recover the truth on synthetic data.
- **OOB seam.** Assert OOB membership matches in-bag multiplicities exactly. Assert rows never OOB produce a reported warning and a documented missing value, not a silent zero.
- **Trait seam.** Assert every unsupported method raises a typed error naming the trait and kind. Assert no unsupported method silently returns a default.
- **Importance seam.** Assert MDI and permutation are reported separately and that a null predictor receives near-zero permutation importance while retaining non-zero MDI.
- **Report seam.** Assert all four required sections are present for every kind. Assert the Objective section names the aggregation rule. Assert the trait table is present.
- **Benchmark smoke test.** Fit every kind on one deterministic dataset with a small budget. Verify output schema, counts, statuses, and artifact creation.
- **Regression-gain seam.** Assert that a `ModelForest(kind="rf")` over the existing `ModelTree` scores a materially higher regression Elo than the single tree's 715.9, and record the number. This is the acceptance test for Decision 18.
- **Histogram parity seam.** Assert that a member fitted with `split_eval=Histogram` selects the same split as one fitted with `split_eval=Fast` on data whose values already fall on bin boundaries, and report a fit-time difference on a large dataset. The binned path itself is Spec 27's work; this seam only checks that the forest wires it correctly.
- **Fit-time seam.** Assert that the forest's mean fit time on the regression board's scale is within a stated multiple of the single tree's, not the member count times the single tree's.
- **Benchmark oracles.** Compare the `bagging` and `rf` kinds against `ranger` and `randomForest` on a fixed dataset and seed protocol. Compare the committee kind against the Cubist C source and the CRAN `Cubist` package on the RuleQuest fc2008-style protocol. Compare the aggregation layer against `caret::bag` semantics.
- **Prior art.** Reuse the existing deterministic-RNG tests, the weighted-fit tests, the survival cross-validation tests, and the split-scan test conventions in `crates/mltreelib-core/tests/`.

## Out of Scope

- The boosting family. Spec 29.
- The meta-estimator family. Spec 30.
- The Generalized Random Forest and its splitting rule. Its own later specification.
- Bayesian additive regression trees and their GPU variants. Not a bagging method and not a boosting method; they are posterior samplers with a different inferential contract. They belong with the probabilistic tree work, not in the bagging family.
- Random subspace and rotation forests. Separate methods with their own member transforms.
- Double bagging and iterated bagging. Distinct staged methods with their own OOB reuse semantics.
- Bayesian bootstrap. A weighted-fit alternative to count-based sampling.
- Stability selection as a forest kind. It is a wrapper above model fitting, and it belongs with the selection work.
- Ordered bagging with member pruning. Needs member storage plus a separate selection set.
- Voting, averaging, and stacking. Spec 30.
- Survival, causal, and forecasting forests in the first wave. Later coverage phases.
- Conformal intervals. They need the OOB machinery, and they land in a later phase rather than as a stub.
- Any sklearn import, dependency, or `BaseEstimator` compatibility.

## Further Notes

- Research basis for bagging mechanics, the bagging-to-boosting boundary, meta-estimator semantics, cross-validation machinery, and API precedent: `references/research/2026-11-22-ensemble-families-and-api-architecture.md`.
- Forest-specific mechanics for random forest, ExtraTrees, GRF, Cubist smoothing, and the committee mechanism: `references/research/2026-11-22-forest-methods-landscape.md`.
- The selection framework that decides when to reach for `ModelForest` rather than `ModelBoost` or `ModelMeta`: Spec 31.
- Spec 25 owns the common report contract and the sklearn compatibility policy. This specification contributes the forest-specific report sections, the trait table, and the member-richness rules.
- Spec 27 owns the regression model expansion, including group-aware penalties that a forest's linear leaves inherit. It also owns the binned split-evaluation path for the regression criteria, which this specification consumes as a prerequisite rather than re-specifying.
- `ModelTree` is the member type. The `Vec<ModelTree>` decision is what makes a forest of this library worth more than a forest of plain CART trees.
- The Cubist GPL source at `https://github.com/topepo/Cubist` is the authoritative reference for the committee kind, and `ranger` is the authoritative reference for the `rf` and `extra_trees` kinds. Both are oracle-only, never runtime dependencies.
- Current source wins where older notes disagree. `references/research/2026-11-17-ranger-and-grf-forest-mechanisms.md` is superseded by the 2026-11-22 forest note on GRF mechanics, on the Cubist smoothing algorithm, and on the attribution of the target-ordered categorical split.
- One multi-core job at a time. Forest training parallelises across members and must not run concurrently with another multi-core fit.
