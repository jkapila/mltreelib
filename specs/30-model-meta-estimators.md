# Spec 30: `ModelMeta` — Voting, Averaging, and Stacking

Status: proposed

Family: meta-estimators. Base learners are trained independently, then a second-level rule combines their predictions.

Scope boundary: this specification owns the meta-estimator family only. The bagging family lives in Spec 28 (`ModelForest`). The boosting family lives in Spec 29 (`ModelBoost`).

## Problem Statement

`mltreelib` can train one model tree. It cannot yet train several and combine them.

The meta-estimator family is the third of the three ensemble families, and it is the one that is genuinely different in kind from the other two.

**Bagging** changes the *training data* of each member and combines members uniformly. **Boosting** changes the *target* each stage is fitted to and combines stages with shrinkage. Both use one base learner type, one member lifecycle, and one aggregation rule.

**Meta-estimators** change neither. They take several already-trained models of possibly different families, produce predictions from each, and combine those predictions by a rule that is either fixed (voting, averaging) or learned (stacking).

That difference has consequences the other two families do not have:

1. **Members can be heterogeneous.** A stack can combine a `ModelForest` with a `ModelBoost` with a linear model. A bag cannot, because all its members share one training procedure.
2. **A second-level learner must be fitted.** Voting and averaging have no fitted state beyond the weights. Stacking fits a model on top.
3. **Out-of-fold predictions are mandatory for stacking.** In-sample base predictions are optimistically biased, and the meta-learner learns misleading reliability from them.
4. **The combination can be conditional.** A learned combiner can weight members differently in different regions of feature space. Uniform averaging cannot.
5. **Cross-validation becomes a first-class dependency.** The fold generator must produce stable fold identities so meta-feature columns stay aligned across base learners.

`mltreelib` already has cross-validation machinery for the survival, calibration, and isotonic work. `ModelMeta` must reuse it rather than fork it, because a fold generator duplicated across specs is a fold generator that will drift.

The bagging specification (Spec 28) already builds the shared Rust `ensemble/` module, the member lifecycle, the per-member seed derivation, and the cross-validation generator. This specification builds on that foundation.

## Solution

One facade, `ModelMeta`, with a `kind` switch over the three combinators, one mode axis, trait-gated capabilities, and one shared report.

### 1. The three combinators

| Kind | Combination rule | Fitted state | Needs held-out data |
|---|---|---|---|
| `voting` | Plurality over predicted labels, or argmax over averaged class probabilities | Optional fixed weights | No |
| `averaging` | Weighted mean of member predictions | Optional fixed weights | No |
| `stacking` | A second-level model fitted on out-of-fold predictions | The fitted meta-model | Yes, by construction |

### 2. The distinction that must not blur

#### 2.1 Voting

**Mechanism.** Count predicted labels, or sum predicted class probabilities and take the argmax.

$$\hat y(x) = \arg\max_c \sum_m w_m \mathbf{1}\{\hat y_m(x) = c\}$$

for hard voting, and

$$\hat y(x) = \arg\max_c \sum_m w_m \hat p_m(x)_c$$

for soft voting.

**Purpose.** Combine models whose errors are partly independent. No training stage, no data requirement.

**Failure modes.** Correlated members produce the same errors, and majority voting cannot repair them. Minority-class members may be drowned out. Hard voting discards confidence entirely. Soft voting is harmed by poor calibration, because the probability scales of different members must be comparable for a weighted sum to mean anything. Fixed weights are not learned from held-out data, so they are either arbitrary or hand-tuned.

#### 2.2 Averaging

**Mechanism.** Weighted mean of member predictions.

$$\hat y(x) = \sum_m w_m \hat f_m(x), \qquad w_m \ge 0, \quad \sum_m w_m = 1.$$

**Purpose.** The regression counterpart of voting, and the prerequisite for soft voting. It is a variance-reduction combinator when member errors are sufficiently diverse.

**Difference from voting.** Voting combines *labels* and is not a linear rule. Averaging combines *measurements* and is linear. For classification, averaging over class probabilities is the soft-voting rule.

#### 2.3 Stacking

**Mechanism.** Wolpert's two-level formulation. The learning set is partitioned; base generalizers are trained on one part and predict the held-out part; the held-out predictions form level-1 observations; a level-1 generalizer is trained on those.

The modern $K$-fold algorithm:

1. Partition rows into $K$ folds.
2. For each base learner $m$, for each fold $k$: fit a clone on the other $K-1$ folds, predict fold $k$.
3. Concatenate the out-of-fold predictions into the meta-feature matrix $Z$.
4. Fit the final estimator on $(Z, y)$.
5. For inference: fit the base learners on the full training data, generate $Z_{\text{test}}$, apply the final estimator.

**Purpose.** Learn how much to trust each member, and let that trust vary with the input. A stack can learn that member $m$ is reliable in one region of feature space and not in another. No uniform rule can do that.

**Breiman's stacked regression** fits a linear combination, usually with non-negative least-squares coefficients. Non-negativity regularises strongly correlated predictors, which is why it is the right default for a numeric stack.

**Failure modes.**

- Meta-features must be out-of-fold. In-sample base predictions are optimistically biased, especially for flexible models, so the meta-learner learns misleading reliability.
- The meta-learner can overfit when base predictions are highly correlated or the sample is small.
- Base models refit on the full data create a train/test distribution shift relative to the out-of-fold meta-features.
- Tuning the meta-learner on the same cross-validation predictions requires nested cross-validation for an honest performance estimate.
- Multi-level stacking magnifies all of the above.

### 2.4 The difference from the other two families

| Dimension | Bagging (Spec 28) | Boosting (Spec 29) | Meta (this spec) |
|---|---|---|---|
| Changes training data | Yes, bootstrap per member | Sometimes, per stage | No |
| Changes training target | No | Yes, negative gradient | No |
| Learns a combination rule | No, uniform | No, fixed shrinkage | Yes, when stacking |
| Members can differ in family | No | No | Yes |
| Needs held-out data | No | No, but early stopping wants it | Yes, for stacking |
| Fitted state beyond members | None | Stage weights | The meta-model |

### 3. The facade

#### 3.1 Two axes

```text
ModelMeta
  kind: MetaKind       # voting | averaging | stacking
  mode: ModelMode      # regression | classification
```

`kind` selects the combinator. `mode` selects the task. The rule is that `kind` never means a task.

#### 3.2 Members are named, typed, and inspectable

Members are supplied as a list of `(name, model)` pairs. Names must be unique, must not collide with facade constructor parameters, and must not contain `__`.

```python
from mltreelib.forest import ModelForest
from mltreelib.boost import ModelBoost
from mltreelib.meta import ModelMeta

stack = ModelMeta(
    kind="stacking",
    mode="regression",
    estimators=[
        ("forest", ModelForest(kind="rf", mode="regression", n_estimators=200)),
        ("boost", ModelBoost(kind="gbdt", mode="regression", n_estimators=300)),
    ],
    final_estimator=None,     # default: NNLS for regression, logistic for classification
    cv=5,
    passthrough=False,
)
```

Members are supplied **already configured but unfitted**. `ModelMeta` clones each member, fits the clones, and owns them.

#### 3.3 Trait-gated capabilities

| Trait | `voting` | `averaging` | `stacking` |
|---|---|---|---|
| `supports_predict_proba` | classification, soft voting | no | depends on the final estimator |
| `supports_weights` | yes | yes | no |
| `supports_passthrough` | no | no | yes |
| `supports_oob` | no | no | no |
| `supports_feature_importance` | member-level only | member-level only | member-level plus meta-model |
| `supports_member_inspection` | yes | yes | yes |
| `supports_inference` | no | no | no |

Two traits deserve comment. `supports_weights` is false for stacking, because the combination rule is learned rather than supplied; supplying weights would contradict the method. And `supports_oob` is false for all three, because the meta-estimator family's notion of held-out data is the cross-validation fold, not the bootstrap.

#### 3.4 Public surface

```python
from mltreelib.meta import ModelMeta

meta = ModelMeta(
    kind="stacking",
    mode="regression",
    estimators=[("rf", forest), ("gbdt", boost)],
    final_estimator=None,
    cv=5,
    passthrough=False,
    weights=None,             # voting and averaging only
    voting="hard",            # voting only: "hard" | "soft"
    random_state=None,
)

meta.fit(X, y, weights=None)
meta.predict(X)
meta.predict_proba(X)        # trait-gated
meta.transform(X)            # the member prediction matrix
meta.member_report(name)     # per-member report section
meta.report()                # Objective / Learner / Controller / Diagnostics
meta.config()
meta.save(path); ModelMeta.load(path)
```

`get_params` and `set_params` are absent, per Spec 25 Decision 11. There is no sklearn import or dependency.

#### 3.5 Rust layout

```text
crates/mltreelib-core/src/ensemble/
  mod.rs          shared member lifecycle, per-member seeds, report data model
  cv.rs           K-fold, leave-one-out, Monte Carlo splits with stable fold identity
  meta.rs         ModelMeta: voting, averaging, stacking
  meta_model.rs   the fitted second-level model and its contract
  report.rs       family-specific report sections
```

## User Stories

### Voting and Averaging

1. As a user, I want `kind="voting"`, so that I combine models whose errors are partly independent with no training stage.
2. As a user, I want hard voting over predicted labels, so that I match the published definition.
3. As a user, I want soft voting over averaged class probabilities, so that I keep confidence information.
4. As a user, I want per-member weights, so that I can express that one member deserves more trust.
5. As a user, I want weights normalised to sum to one, so that the combination stays on the response scale.
6. As a user, I want `kind="averaging"`, so that I combine regression predictions by a weighted mean.
7. As a user, I want soft voting to require every member to expose class probabilities, so that the failure is explicit rather than a silent fallback.
8. As a user, I want calibration diagnostics for soft voting, so that I know whether averaging probabilities is meaningful.

### Stacking

9. As a user, I want `kind="stacking"`, so that I learn how much to trust each member.
10. As a user, I want the meta-features computed out-of-fold, so that the meta-learner does not learn from optimistically biased base predictions.
11. As a user, I want `cv` to accept an integer, a custom splitter, or an iterable of `(train, test)` pairs, so that I control the fold structure.
12. As a user, I want stratified folds for classification and plain folds for regression, so that class balance is respected where it matters.
13. As a user, I want the default final estimator to be non-negative least squares for regression, so that correlated members are regularised rather than given cancelling weights.
14. As a user, I want the default final estimator to be logistic regression for classification, so that the stack produces probabilities.
15. As a user, I want `passthrough` to optionally expose the original features to the meta-model, so that I can model residual structure the members miss.
16. As a user, I want to supply my own final estimator, so that I control the second-level model.
17. As a user, I want the meta-model's coefficients and their meaning reported, so that I can see which members carry the weight.
18. As a user, I want the number of meta-features reported, so that I know the meta-model's capacity relative to the sample size.
19. As a user, I want a warning when the number of meta-features approaches the sample size, so that I know the stack is overfitting.

### Member Management

20. As a user, I want members supplied as `(name, model)` pairs, so that I can refer to them in reports and in the final estimator.
21. As a user, I want member names validated for uniqueness and for collision with constructor parameters, so that a mistyped name fails at fit time rather than at prediction time.
22. As a user, I want a member marked as dropped to be excluded from fitting, so that I can disable a member without rebuilding the list.
23. As a user, I want the fitted members exposed with their names, so that I can inspect each one.
24. As a user, I want members of different families in one stack, so that I can combine a forest with a booster.
25. As a user, I want `transform` to return the member prediction matrix, so that I can inspect what the meta-model is learning from.
26. As a user, I want per-member reports, so that I can see which member is weak.

### Reproducibility and Reporting

27. As a user, I want identical results for one worker and many workers, so that parallel fitting is safe.
28. As a user, I want the fold assignment stored in the fitted model, so that a reloaded stack reproduces exactly.
29. As a user, I want the base learners refitted on the full training data after the fold phase, so that inference uses the strongest base models.
30. As a user, I want the report to state that meta-features were out-of-fold, so that the provenance of the second-level training data is visible.
31. As a user, I want the report to state whether `passthrough` was used, so that the meta-model's input width is explicable.
32. As a user, I want a warning when members are highly correlated, so that I know the stack may be overfitting.
33. As a library maintainer, I want the fold generator shared with the rest of the package, so that fold semantics do not drift.

## Implementation Decisions

### Decision 1: `ModelMeta` Is a Separate Public Class

Meta-estimators get their own class, not a `kind` inside `ModelForest` or `ModelBoost`.

The reason is that the parameters have nothing in common. A forest's parameters describe how to grow and aggregate one member type. A stack's parameters describe a member list, a fold structure, a second-level model, and a passthrough policy. Sharing a constructor between them would mean every forest knob is also a stack knob.

### Decision 2: Members Are Supplied Unfitted and Cloned

`ModelMeta` receives configured but unfitted models, clones each one, and fits the clones. The originals are never mutated.

Cloning is required because the fold phase fits each member $K$ times and the final phase fits it once more. Without cloning, the caller's model object would end up holding a fitted state it never asked for.

### Decision 3: Named Members, With Explicit Validation

Members are `(name, model)` pairs. Names must be unique, must not collide with facade constructor parameters, and must not contain `__`. A member whose name equals `"drop"` is excluded from fitting.

The validation is deliberately stricter than the loosest convention in the wild, where a dropped member can remain visible as a placeholder in the named-members mapping. The `mltreelib` contract keeps two explicit lists: active members and dropped names. That removes the ambiguity of a fitted member list containing a string.

### Decision 4: Out-of-Fold Meta-Features Are the Only Accepted Form

The stacking implementation computes out-of-fold predictions for the meta-training matrix. There is no in-sample option.

A "prefit" mode, where the caller supplies already-fitted members and the meta-features are in-sample predictions, is **deferred**. It is a genuine method with a genuine use case, but it carries a very high overfitting risk that the package must be able to state clearly, and the fold generator must exist first. When it lands it must carry an explicit leakage warning.

### Decision 5: Fold Identity Is Stable and Shared

The fold generator assigns each row a stable fold identifier. Every base learner uses the *same* fold assignment, so the meta-feature columns align row-wise across members.

This is not a detail. If member $m$ uses fold assignment $A$ and member $m'$ uses assignment $B$, then the meta-feature matrix pairs row $i$'s prediction from $m$ with row $j \ne i$'s prediction from $m'$, and the meta-model learns from noise.

The generator lives in `ensemble/cv.rs` and interoperates with the package's existing cross-validation surface rather than forking it.

### Decision 6: Default Final Estimators

- **Regression:** non-negative least squares over the member predictions, matching Breiman's stacked regression. Non-negativity regularises strongly correlated members and prevents the degenerate solution where two correlated members receive large cancelling weights.
- **Classification:** logistic regression over the member probabilities.

Both are overridable. The defaults are chosen for regularisation, not for raw accuracy.

### Decision 7: Binary Classification Drops One Probability Column

For binary classification, the member probability matrix has one redundant column, because the two probabilities sum to one. The meta-feature matrix therefore drops one column per member.

Without this, the meta-model's design matrix is exactly singular on that column pair, and solvers behave differently from one another on near-singular input.

### Decision 8: `passthrough` Is Off by Default

With `passthrough=True`, the meta-model receives the member predictions concatenated with the original features.

This raises the meta-model's capacity from "a function of $M$ numbers" to "a function of $M$ numbers plus $p$ features", and it is the fastest way to overfit a stack. It is available because it is sometimes exactly right, but it is off by default and the report states whether it was used.

### Decision 9: Voting Requires Uniform Probability Scales for Soft Voting

Soft voting averages class probabilities across members. This is only meaningful when the members' probability scales are comparable.

The implementation does not attempt to calibrate members before averaging. It reports a calibration diagnostic and warns when the members' probability scales differ materially, because silently calibrating members would change their behaviour in ways the user did not request.

### Decision 10: Weights Are Fixed, Not Learned, in Voting and Averaging

Voting and averaging accept optional per-member weights, and those weights are supplied, not fitted. There is no weight-learning stage in this specification.

Learning weights requires held-out data and is what stacking is for. A weight-learning variant belongs with the stacking machinery.

### Decision 11: The Report States Meta-Feature Provenance

The Objective section states: "combination rule learned on out-of-fold member predictions", and the Controller section records the fold count, the fold assignment identity, whether `passthrough` was used, the meta-feature width, and the final estimator's identity.

This makes the provenance of the second-level training data visible, which is the single most common source of overconfidence in a stack.

### Decision 12: Correlation and Capacity Warnings

Two warnings fire when the conditions are met:

- **Capacity.** When the number of meta-features approaches the sample size, the stack is at risk of overfitting regardless of the fold structure.
- **Correlation.** When the members' out-of-fold predictions are highly correlated, the stack adds little over the best single member and the meta-model has little to learn.

Both are computed from the out-of-fold matrix, which the implementation already has.

### Decision 13: No sklearn Estimator Protocol

No `get_params`, no `set_params`, no `clone`, no `deep`, no generic `score`, and no sklearn import or dependency. `config()` is the inspection surface. Member configuration is a typed control object per family.

### Decision 14: One Multi-Core Job at a Time

Base learners are fitted in parallel within the fold phase. The phase must not run concurrently with another multi-core fit.

## Exact Build Sequence

Each phase lands as its own commit with focused tests. A phase does not start before the previous phase passes its gate.

### Phase 0 — Voting and Averaging

1. Add `ensemble/meta.rs` with `voting` and `averaging`, on top of the shared member lifecycle from Spec 28.
2. Add the member registry with name validation and the active and dropped lists.
3. Add hard voting, soft voting, weighted averaging, and weight normalisation.
4. Add `transform` returning the member prediction matrix.
5. Add per-member report sections.
6. Add Python `ModelMeta` with `kind`, `mode`, `fit`, `predict`, `transform`, `member_report`, `config`, `report`.
7. Add Rust tests for hard voting, soft voting, weighted averaging, weight normalisation, and dropped members.
8. Add Python tests for the facade lifecycle, name validation, and trait gating.

Gate: `cargo test` and Python tests pass. The voting aggregate matches a hand-computed count and the weighted average matches a hand-computed value.

### Phase 1 — Cross-Validation Generator

1. Add `ensemble/cv.rs` with K-fold, leave-one-out, and Monte Carlo splits.
2. Give each row a stable fold identifier.
3. Support stratification for classification and an optional shuffle with a seed.
4. Interoperate with the package's existing cross-validation surface rather than duplicating it.
5. Add Rust tests for fold coverage, disjointness, stratification balance, and stable identity across runs.
6. Add Python tests for the generator's public behaviour.

Gate: `cargo test` and Python tests pass. Every row appears in exactly one test fold per repetition, and fold identity is stable for a given seed.

### Phase 2 — Stacking

1. Add the fold-phase fitting: for each member and each fold, fit a clone on the other folds and predict the held-out fold.
2. Assemble the out-of-fold meta-feature matrix with the stable fold identity.
3. Add the final estimator contract: non-negative least squares for regression, logistic regression for classification.
4. Add the binary-classification single-column drop.
5. Add the full-data refit of base learners after the fold phase.
6. Add `passthrough`.
7. Add the correlation and capacity warnings.
8. Add Rust tests for out-of-fold assembly, column alignment across members, the final estimator fit, and the full-data refit.
9. Add Python tests for the stack lifecycle, the meta-model report, and the warnings.

Gate: `cargo test` and Python tests pass. Meta-feature columns align row-wise across members and the final estimator fits on the out-of-fold matrix.

### Phase 3 — Report and Persistence Polish

1. Add the meta-specific report sections with the meta-feature provenance, the fold identity, and the passthrough flag.
2. Add `save`, `load`, and pickle round trips preserving the fold assignment and the fitted members.
3. Add Rust tests for report section presence and for persistence round-trip fidelity.
4. Add Python tests for the report structure and the persistence round trip.

Gate: `cargo test` and Python tests pass. A reloaded stack reproduces identical predictions.

### Phase 4 — Deferred Depth

1. A prefit mode with already-fitted members and an explicit leakage warning.
2. Nested cross-validation for an honest estimate of the stack's own performance.
3. Multi-level stacking.
4. Learned weights as an alternative to a fixed combination rule.
5. A stacking-aware calibration diagnostic for soft voting.

Each lands as its own phase with its own gate.

## Testing Decisions

- **Good tests assert external behaviour.** Tests cross the highest available seam. They verify predictions, member alignment, report contents, trait errors, and reproducibility. They do not assert private update formulas or internal vector layout.
- **Primary Rust seam.** Fit each kind through the meta entry point on deterministic synthetic data. Assert predictions and status.
- **Primary Python seam.** Fit each kind through `ModelMeta`. Assert trait gating, report sections, config output, and persistence.
- **Voting seam.** Assert the hard-vote aggregate matches a hand-computed count over member labels. Assert the soft-vote aggregate matches a hand-computed weighted probability sum.
- **Averaging seam.** Assert the aggregate equals the weighted mean of member predictions and that weights normalise to one.
- **Member seam.** Assert the caller-supplied member objects are not mutated by fitting, that clones are fitted instead, that a dropped member is excluded, and that duplicate or colliding names raise a validation error.
- **Fold seam.** Assert every row appears in exactly one test fold per repetition, that train and test folds are disjoint, that the fold assignment is stable for a given seed, and that stratified folds preserve class proportions within tolerance.
- **Alignment seam.** On data where two members are deliberately permuted relative to each other, assert the out-of-fold matrix detects the misalignment. This is the failure mode that silently produces a worse model rather than an error, so it deserves a dedicated test.
- **Out-of-fold seam.** Assert that the meta-training matrix is built from held-out predictions, by checking that a member fitted on the full data would give a different matrix, and that the meta-model's training loss is worse on the out-of-fold matrix than an in-sample matrix would suggest.
- **Final estimator seam.** Assert the default regression final estimator produces non-negative weights over member predictions, and that the default classification final estimator produces probabilities.
- **Singularity seam.** Assert the binary-classification meta-feature matrix drops one probability column per member and that the design matrix is well conditioned.
- **Passthrough seam.** Assert `passthrough=True` widens the meta-feature matrix by the feature count and that the report states it.
- **Refit seam.** Assert the members used for inference are fitted on the full training data, by checking that they differ from the fold-phase clones.
- **Trait seam.** Assert every unsupported method raises a typed error naming the trait and kind. Assert soft voting without member probabilities raises a typed error rather than falling back to hard voting.
- **Report seam.** Assert all four required sections are present for every kind. Assert the Objective section names the combination rule. Assert the meta-feature provenance appears in the report.
- **Warning seam.** Assert the capacity warning fires when the meta-feature count approaches the sample size, and that the correlation warning fires for deliberately near-duplicate members.
- **Determinism seam.** Fit the same stack with `n_jobs=1` and `n_jobs=4`. Assert identical predictions and identical fold assignment.
- **Persistence seam.** Save and reload a fitted stack. Assert identical predictions, identical fold assignment, and identical member states.
- **Benchmark smoke test.** Fit every kind on one deterministic dataset with a small budget. Verify output schema, counts, statuses, and artifact creation.
- **Benchmark oracles.** Compare voting and averaging semantics against scikit-learn's `VotingClassifier` and `VotingRegressor` on a fixed dataset. Compare the stacking construction against scikit-learn's `StackingRegressor` on a fixed dataset with the same folds.
- **Prior art.** Reuse the existing cross-validation tests, the deterministic-RNG tests, and the weighted-fit test conventions in `crates/mltreelib-core/tests/`.

## Out of Scope

- The bagging family. Spec 28.
- The boosting family. Spec 29.
- The Generalized Random Forest. Its own later specification.
- Bayesian additive regression trees. Not a meta-estimator family member; a posterior sampler with a different inferential contract.
- Prefit stacking with already-fitted members. Deferred to Phase 4 with an explicit leakage warning.
- Nested cross-validation for honest stack performance. Phase 4.
- Multi-level stacking. Phase 4.
- Learned weights as an alternative to a fixed rule. Phase 4.
- Bagging over members, such as bagging the stacks. A meta-estimator over a meta-estimator needs its own error analysis.
- Heterogeneous members with incompatible output shapes, such as a classifier and a regressor in one stack. The mode axis forbids it.
- Any sklearn import, dependency, or `BaseEstimator` compatibility.

## Further Notes

- Research basis for voting, averaging, stacking, cross-validation machinery, Bayesian bootstrap, ordered bagging, stability selection, and API precedent: `references/research/2026-11-22-ensemble-families-and-api-architecture.md`.
- The selection framework that decides when to reach for `ModelMeta` rather than `ModelForest` or `ModelBoost`: Spec 31.
- Spec 28 owns the shared Rust `ensemble/` module, the member lifecycle, the per-member seed derivation, the bootstrap sampler, and the cross-validation generator that this specification builds on.
- Spec 25 owns the common report contract and the sklearn compatibility policy. This specification contributes the meta-specific report sections and the trait table.
- The `grf` forest weights are the closest analogue to meta-features that the package will have, since both are per-training-row vectors derived from an ensemble. The two should not be conflated: forest weights come from leaf co-occurrence within one forest, meta-features come from predictions of several separate models.
- One multi-core job at a time. The fold phase fits members in parallel and must not run concurrently with another multi-core fit.
