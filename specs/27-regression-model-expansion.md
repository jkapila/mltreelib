# Spec 27: Regression Model Expansion, Group Behavior, and Panel Foundation

Status: proposed

## Problem Statement

`mltreelib` has a solid penalized-linear regression core. Exact Gaussian OLS and Ridge solve through normal equations and Cholesky. Lasso and Elastic Net solve through cyclic coordinate descent. L-BFGS fits smooth generalized linear deviance. The GLM family set covers Gaussian, Binomial, Probit, Poisson, Gamma, Tweedie, and Tobit.

Three gaps limit that core.

### 1. The asymmetric and robust regression families are missing or mis-scoped

`LeafKind::Median`, `LeafKind::Quantile`, and `LeafKind::Huber` are **0D location estimators**. In `crates/mltreelib-core/src/stats.rs`, `w_median`, `w_quantile`, and `huber_location` each consume value-weight pairs and return one scalar. They ignore every predictor column.

A user asking for "quantile regression" wants conditional quantiles $\hat Q_\tau(y \mid x)$, not a constant. A user asking for "Huber regression" wants a robust multivariate fit, not a robust scalar. Both requests currently produce a model that ignores the design matrix.

Missing entirely: expectile regression, non-negative least squares, negative binomial regression, beta regression, and group-structured penalties.

### 2. Group behavior is absent from the regression core

The package thesis is grouped and hierarchical structure. The tree layer already carries `interaction_groups` and `train_groups`. The regression core carries no group metadata at all.

`crates/mltreelib-core/src/solver/design.rs` defines `ColumnSliceView` and `DenseView` with no group structure. `crates/mltreelib-core/src/solver/ccd.rs` applies scalar L1 and L2 penalties with per-coordinate soft thresholding. There is no way to express "these six columns are one group" or "these coefficients must be selected or rejected together."

Consequences:

- Group lasso and sparse-group lasso cannot be expressed.
- Monotone constraints are per coefficient, not per group.
- Panel entity structure cannot reach a leaf fit. `crates/mltreelib-core/src/model_tree.rs` has no `entity_col` or `group_col` parameter, so Spec 17's within-transformation has no way to receive entity identity.

### 3. Panel regression has no foundation

Spec 17 defines panel fixed-effects trees, dynamic panel GMM, TVAR, and TVECM. None of it can land because the leaf fit path has no entity grouping and no within-transformation.

`crates/mltreelib-core/src/encode.rs` computes pooled within-level variance for a categorical encoder summary. That is a variance estimator, not a within-transformation for regression.

The package needs the group substrate first. Panel regression is the first consumer of it, not a parallel workstream.

### 4. The solver seam exists but the regression layer does not use it

`crates/mltreelib-core/src/solver/generic/` now holds `Objective`, `SmoothObjective`, `Bounds`, `Constraint`, `Budget`, `OptimizationRequest`, `OptimizerKind`, `OptimizationResult`, and `SolveStatus`. `OptimizerKind` already lists `RandomSearch`, `LatinHypercube`, `LbfgsB`, `Bfgs`, `NonlinearCg`, `NelderMead`, `Powell`, `SimulatedAnnealing`, `Cobyla`, and `BasinHopping`.

No regression estimator routes through that seam. Every GLM fit goes through `solver/dense.rs` and `solver/mod.rs` with hard-wired family and penalty semantics.

## Solution

Build one **group-aware regression layer** and add six regression families on top of it.

The layer has three parts.

**Group metadata.** One group declaration on the design view, covering three group kinds:

| Group kind | Meaning | Used by |
|---|---|---|
| Coefficient groups | Named sets of predictor columns | Group lasso, sparse-group lasso, grouped monotone constraints |
| Panel entities | The unit a row belongs to, for repeated measurements | Within-transformation, entity fixed effects, hierarchical pooling |
| Hierarchical groups | Nested entity levels, for example department then store | Partial pooling across levels |

**Regression problems.** Each regression family becomes a leaf problem with an explicit loss, gradient, and penalty structure. Problems plug into the generic solver seam rather than branching on solver identity.

**Panel foundation.** Entity identity and within-transformation become first-class regression inputs, so panel fixed-effects leaves can land on top of the same group substrate that group lasso uses.

Build order:

| Phase | Content |
|---:|---|
| 0 | Group metadata on the design view |
| 1 | Quantile regression |
| 2 | Huber regression with robust scale |
| 3 | Expectile regression |
| 4 | Non-negative least squares |
| 5 | Group lasso and sparse-group lasso |
| 6 | Negative binomial and beta GLM |
| 7 | Panel entity grouping and within-transformation |
| 8 | Panel fixed-effects leaf |
| 9 | Hierarchical partial pooling foundation |
| 10 | Conditional GAM work |

Phase 0 lands first. Every later phase depends on it.

## User Stories

### Group Metadata

1. As a library maintainer, I want the design view to carry coefficient group declarations, so that grouped penalties become expressible without a separate code path.
2. As a library maintainer, I want group declarations validated on construction, so that an out-of-range index or an overlapping group fails before fitting.
3. As a library maintainer, I want group membership to survive the missing-indicator expansion, so that indicator columns inherit the group of their source feature.
4. As a library maintainer, I want the intercept excluded from every group, so that no penalty shrinks the intercept.
5. As a model user, I want to declare groups by column name, so that group definitions survive column reordering.
6. As a model user, I want to declare groups by column index, so that programmatic pipelines stay simple.
7. As a model user, I want an empty group list to mean "no grouping", so that existing behavior is the default.
8. As a model user, I want overlapping groups rejected with one precise message, so that the estimator never silently changes.
9. As a model user, I want single-column groups allowed but documented as equivalent to scalar penalties, so that no special case is hidden.
10. As a library maintainer, I want group definitions to serialize with the model snapshot, so that a reloaded model applies the same penalties.

### Panel Entity Grouping

11. As a data scientist, I want to pass an entity identifier with a panel design, so that the estimator can remove entity-level heterogeneity.
12. As a data scientist, I want the within-transformation computed analytically, so that millions of entities need no dummy columns.
13. As a data scientist, I want unbalanced panels supported, so that entities with different observation counts fit correctly.
14. As a data scientist, I want the within-transformation cached per fit, so that repeated leaf fits do not recompute entity means.
15. As a data scientist, I want entity means available for out-of-sample prediction, so that the entity fixed effect can be added back.
16. As a data scientist, I want prediction for an unseen entity documented, so that I know the fallback is the group mean rather than a silent failure.
17. As a data scientist, I want hierarchical entity levels supported, so that a store nested in a department can pool correctly.
18. As a data scientist, I want pooling level configurable, so that I can choose department, store, or neither.

### Quantile Regression

19. As a model user, I want multivariate quantile regression, so that I can model conditional quantiles rather than a constant.
20. As a model user, I want the pinball loss reported, so that I can judge fit quality in the loss the estimator minimized.
21. As a model user, I want several taus fitted jointly, so that I can see the whole conditional distribution.
22. As a model user, I want quantile crossing detected and reported, so that I know when fitted curves are incoherent.
23. As a model user, I want monotone quantile curves enforced or reported, so that tau ordering is respected.
24. As a library maintainer, I want the pinball loss as an explicit regression problem, so that any compatible solver can fit it.
25. As a library maintainer, I want the asymmetric weight function shared between quantile and expectile problems, so that one implementation serves both.
26. As a benchmark author, I want quantile fits compared against a reference implementation, so that a hand-written solver can be checked.
27. As a model user, I want a minimum-support rule for quantile leaves, so that tiny leaves fail visibly instead of returning unstable fits.
28. As a model user, I want quantile fits to carry no OLS standard errors, so that invalid inference is never presented.

### Huber Regression

29. As a model user, I want multivariate Huber regression, so that outliers in a leaf do not distort the fit.
30. As a model user, I want the robust scale reported, so that I can see how heavy the tail was.
31. As a model user, I want the tuning constant explicit, so that I know how close the estimator is to least squares or least absolute deviation.
32. As a model user, I want the scale and location iterations to converge or fail visibly, so that a divergent robust fit never passes silently.
33. As a model user, I want the 0D Huber location leaf to keep working unchanged, so that existing models stay stable.
34. As a benchmark author, I want RANSAC and Theil-Sen available as oracles, so that robust-fit improvements are verifiable.
35. As a library maintainer, I want the Huber loss as a regression problem with a gradient, so that L-BFGS fits it directly.

### Expectile Regression

36. As a model user, I want expectile regression, so that I have a smooth asymmetric loss alternative to quantiles.
37. As a model user, I want mean recovery at the central expectile, so that I can verify the estimator.
38. As a model user, I want the expectile level configurable, so that one leaf fits many asymmetric targets.
39. As a library maintainer, I want expectiles and quantiles to share one asymmetric-weight implementation, so that the two losses stay consistent.

### Non-Negative Least Squares

40. As a model user, I want non-negative least squares, so that count rates, exposures, and prices fit with nonnegative coefficients.
41. As a library maintainer, I want non-negativity to be a solver capability, so that it composes with other constraints.
42. As a library maintainer, I want the intercept treatment documented, so that a nonnegative intercept is not a surprise.
43. As a library maintainer, I want NNLS to reuse the existing Cholesky path, so that no second linear-algebra stack lands.
44. As a library maintainer, I want NNLS as the simplest bounded-solver test case, so that bounds work gets exercised early.
45. As a model user, I want an active-set solution reported with its selected variables, so that the sparsity pattern is visible.

### Group and Sparse-Group Lasso

46. As a model user, I want group lasso, so that whole feature groups are selected or rejected together.
47. As a model user, I want sparse-group lasso, so that I can select groups and individual members within them.
48. As a model user, I want grouped monotone constraints, so that a whole group shares one sign restriction.
49. As a model user, I want group selection reported per group, so that I know which groups survived.
50. As a library maintainer, I want group soft thresholding inside the existing coordinate descent, so that scalar lasso behavior stays unchanged.
51. As a library maintainer, I want deterministic group ordering, so that repeated runs give identical results.
52. As a model user, I want the penalty to scale with group size in a documented way, so that a large group is not unfairly shrunk.
53. As a benchmark author, I want group lasso compared against an independent implementation, so that group thresholding is verified.
54. As a model user, I want group lasso usable with panel entities, so that entity-level selection is possible.

### Negative Binomial and Beta Regression

55. As a model user, I want negative binomial regression, so that overdispersed counts are modeled correctly.
56. As a model user, I want the dispersion parameter reported, so that I can see how far the data sit from Poisson.
57. As a model user, I want a dispersion test against Poisson, so that I can justify the extra parameter.
58. As a model user, I want the NB1 and NB2 parameterizations documented and not mixed, so that results are comparable.
59. As a model user, I want the zero-dispersion boundary handled explicitly, so that the estimator does not divide by zero.
60. As a model user, I want beta regression for outcomes bounded in zero and one, so that I do not force a Gaussian or binomial model.
61. As a library maintainer, I want both families to reuse the existing family machinery, so that no second likelihood framework appears.
62. As a model user, I want small-leaf dispersion and precision estimates to fail visibly, so that unstable parameters never ship.

### Panel Foundation

63. As a data scientist, I want a panel fixed-effects leaf, so that entity heterogeneity is removed without dummy columns.
64. As a data scientist, I want within-estimator slopes comparable to pooled slopes, so that I can test whether fixed effects matter.
65. As a data scientist, I want the entity fixed effect recoverable from the fit, so that I can inspect heterogeneity.
66. As a data scientist, I want entity means cached across leaf fits, so that panel fitting stays linear in rows.
67. As a data scientist, I want hierarchical pooling across nested entity levels, so that sparse subgroups borrow strength.
68. As a data scientist, I want pooling level configurable per fit, so that I can compare pooled, partially pooled, and fixed-effects results.
69. As a data scientist, I want the pooled estimator to remain available, so that I can compare against the within estimator.
70. As a library maintainer, I want the panel foundation to reuse the group metadata substrate, so that group lasso and panel fitting share one mechanism.
71. As a library maintainer, I want Spec 17's TVAR and TVECM leaves to remain out of this specification, so that scope stays bounded.
72. As a data scientist, I want panel rows to keep the same missing-data policy as other leaves, so that no separate imputation path appears.

### Conditional GAM Work

73. As a model user, I want MARS knot placement and regularization improved first, so that the existing spline leaf gets stronger before a new one arrives.
74. As a model user, I want a penalized-spline GAM only if it beats MARS, so that I do not pay for a duplicate family.
75. As a benchmark author, I want a MARS-versus-GAM comparison in the benchmark suite, so that the decision is evidence-driven.

## Implementation Decisions

### Decision 1: One Group Declaration Type

Add a group declaration type to the solver module. It carries a group kind, a group name, and the member columns.

```text
GroupSpec
  kind: CoefficientGroup | PanelEntity | HierarchicalGroup
  name: String
  members: Vec<usize>
```

The design views gain a group declaration field. `ColumnSliceView` and `DenseView` both accept it at construction. Existing constructors keep working with an empty declaration.

Validation rules:

- Every member index is within the design range.
- The intercept is never a member.
- Coefficient groups do not overlap.
- Missing-indicator columns inherit the group of their source feature.
- A group with one member is allowed and documented as equivalent to a scalar penalty.

Panel entity identity and hierarchical levels are represented by separate declarations, not by overloading coefficient groups. A panel entity is a row attribute, not a column selection.

### Decision 2: Group-Aware Coordinate Descent

Group lasso applies an L2 norm penalty over each group:

$$\min_\beta \frac{1}{2n}\|X\beta - y\|_2^2 + \alpha \sum_g \sqrt{\beta_g^\top\beta_g}$$

Sparse-group lasso adds a scalar L1 penalty:

$$\min_\beta \frac{1}{2n}\|X\beta - y\|_2^2 + \alpha\lambda_1\|\beta\|_1 + \alpha\lambda_2\sum_g \sqrt{\beta_g^\top\beta_g}$$

Group soft thresholding replaces coordinate soft thresholding inside the existing cyclic update. For a group with coefficients $\beta_g$, the update scales the group vector by

$$\left(1 - \frac{\alpha\lambda_2}{\|\beta_g\|_2}\right)_+$$

when the group norm exceeds the threshold, and leaves it unchanged otherwise. Sparse-group lasso applies coordinate soft thresholding inside the group first, then group soft thresholding.

The penalty scaling convention is fixed and documented. Group penalty terms are normalized by group size so that a large group is not penalized more heavily than a small group purely for having more members.

Deterministic group ordering is required. Groups are processed in sorted name order, and members within a group in ascending column index.

### Decision 3: Regression Problems as Explicit Objectives

Each regression family becomes a problem type that owns its loss, gradient, and penalty structure.

| Problem | Loss | Gradient | Penalty support |
|---|---|---|---|
| GaussianLeastSquares | Squared error | Exact | L2, monotone, group |
| QuantileRegression(tau) | Pinball | Exact, asymmetric weight | L1, L2 |
| ExpectileRegression(tau) | Asymmetric squared | Exact | L1, L2 |
| HuberRegression(delta) | Huber | Exact, piecewise | L2, monotone, group |
| NegativeBinomial(alpha) | Negative binomial deviance | Exact | L2 |
| BetaRegression(phi) | Beta deviance | Exact | L2 |

Each problem implements `Objective` and, where the gradient exists, `SmoothObjective` over the generic seam. Quantile and expectile problems share one asymmetric-weight implementation with different weight functions. Huber and Gaussian share the squared-error term with different robustness weighting.

Problems solve through the generic seam, not through per-solver branches. The Spec 26 Phase 1 leaf-to-solver entry point becomes the single dispatch point.

### Decision 4: Quantile Regression Details

Quantile regression minimizes the pinball loss:

$$\rho_\tau(r) = r\left(\tau - \mathbb{1}(r < 0)\right)$$

with gradient contribution $\left(\tau - \mathbb{1}(r_i < 0)\right)x_i$.

Implementation choices:

- The convex problem is solved through the generic seam. The smooth part of the loss plus an asymmetric proximal term suits the proximal-gradient path that Spec 26 introduces.
- A smoothed iteratively reweighted least squares path is the fallback when the proximal path is unavailable. The smoothing parameter is a numerical artifact and must be documented as such.
- Linear programming is not used. The package rejected a heavyweight LP dependency in the COBYLA decision.
- A minimum-support rule rejects leaves with fewer observations than parameters plus one.
- Several taus may be fitted jointly. Crossing between adjacent taus is detected and reported. An optional monotone enforcement projects the fitted curves into tau order.
- Coefficient-level inference is not reported. OLS standard errors do not apply to asymmetric losses.

The 0D weighted-quantile leaf keeps its current behavior when no predictors are present. The multivariate path activates when predictor columns exist.

### Decision 5: Huber Regression Details

Huber regression minimizes

$$\sum_i \rho_\delta\left(\frac{y_i - x_i^\top\beta}{\hat\sigma}\right)$$

with the Huber loss quadratic below the threshold and linear above it.

The scale $\hat\sigma$ comes from a weighted median absolute deviation of the initial residuals, scaled for consistency under normality. Scale and location alternate between outer iterations.

Rules:

- The tuning constant $\delta$ is explicit in the result and never hidden.
- Divergence between scale and location iterations produces an explicit failure status, not a silent fallback.
- The 0D Huber location leaf is unchanged. It remains the default for a leaf with no predictors.
- The Huber problem provides an exact gradient, so L-BFGS fits it directly through the generic seam.

### Decision 6: Expectile Regression Details

Expectile regression minimizes the asymmetric squared loss:

$$\rho_\tau^{\exp}(r) = \begin{cases}\tau r^2, & r \ge 0\\ (1-\tau) r^2, & r < 0\end{cases}$$

The loss is smooth. It fits the existing L-BFGS path with no proximal machinery.

The asymmetric weight function is shared with quantile regression. One implementation serves both, with a different weight rule. Mean recovery at the central expectile is a required test.

### Decision 7: Non-Negative Least Squares Details

NNLS solves

$$\min_{\beta \ge 0} \frac{1}{2}\|X\beta - y\|_2^2$$

with the Lawson-Hanson active-set algorithm over the existing Cholesky path.

Non-negativity is a solver capability, not a leaf family. It registers in the Spec 26 solver capability table. The intercept treatment is documented: the intercept is unpenalized like every other method, and whether it is constrained is an explicit option.

NNLS is the simplest bounded-solver test case for the Spec 26 bounds work.

### Decision 8: Negative Binomial and Beta Details

Negative binomial adds a dispersion parameter $\alpha$ to the Poisson-style family, with variance $V(\mu) = \mu + \alpha\mu^2$. Fitting alternates between estimating coefficients given the dispersion and estimating the dispersion given the coefficients.

The NB1 and NB2 parameterizations are documented separately. The implementation uses one and states which.

Beta regression targets outcomes strictly inside zero and one, with a logit link and a beta distribution. Its precision parameter is reported.

Both families extend the existing family machinery in `solver/family.rs`: variance function, deviance, gradient, and initial intercept. No second likelihood framework appears.

Small-leaf estimates of the dispersion or precision parameter are unstable. A minimum-support rule applies.

### Decision 9: Panel Entity Grouping and Within-Transformation

Entity identity enters the regression layer as a panel declaration on the design view: one entity key per row, plus optional nested levels.

The within-transformation removes entity-level heterogeneity analytically:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i, \qquad \tilde{\mathbf{x}}_{it} = \mathbf{x}_{it} - \bar{\mathbf{x}}_i$$

No dummy columns are created. The entity means $\bar{y}_i$ and $\bar{\mathbf{x}}_i$ are computed once per fit and cached, so leaf fits stay linear in rows.

Out-of-sample prediction adds the entity effect back:

$$\hat{\alpha}_i = \bar{y}_i - \bar{\mathbf{x}}_i^\top\hat{\beta}$$

An unseen entity at prediction time falls back to the fit-level group mean. That fallback is documented and reported in diagnostics, never silent.

Unbalanced panels are supported. Entities with a single observation contribute nothing to the within fit and are reported.

The panel declaration reuses the group substrate. One mechanism serves grouped penalties and panel structure.

### Decision 10: Panel Fixed-Effects Leaf

The panel fixed-effects leaf fits within-transformed slopes per leaf and reports the entity effects.

Design:

- Leaf rows are partitioned by the tree as usual.
- Within-transformation uses entity means over the full fit, not per leaf. Demeaning per leaf would bias the transformation.
- The leaf estimate is the within-estimator slope, plus an optional ridge penalty on slopes.
- Entity effects are recovered after the tree is fitted.
- The pooled estimator remains available for comparison.

This spec delivers the panel fixed-effects foundation. Spec 17 keeps dynamic panel GMM, TVAR, and TVECM.

### Decision 11: Hierarchical Partial Pooling Foundation

Nested entity levels support partial pooling. A subgroup with few observations borrows strength from its parent level.

Foundation scope:

- A pooling level selector chooses which level receives the pooled estimate.
- A shrinkage weight combines a subgroup estimate with its parent estimate.
- The weight follows a documented rule based on observation counts.
- The mechanism is exposed to the group-lasso machinery through the same group substrate.

Full mixed-effects estimation with a random-effects design and sparse Cholesky stays out of scope. This is a deterministic partial-pooling foundation, not a mixed-model solver.

### Decision 12: Module Layout

New regression modules live under the solver module, not inside the leaf module:

```text
solver/
  groups.rs              group declarations, validation, panel entity keys
  problems/
    mod.rs               regression problem registry
    gaussian.rs          squared error with L2, monotone, and group penalties
    quantile.rs          pinball loss and asymmetric weight
    expectile.rs         asymmetric squared loss
    huber.rs             Huber loss and robust scale iteration
    nnls.rs              non-negative least squares
    glm_extra.rs         negative binomial and beta family extensions
  panel/
    demean.rs            within-transformation and entity mean caching
    fixed_effects.rs     within-estimator and entity effect recovery
    pooling.rs           hierarchical partial pooling
```

The leaf module consumes problems and panel declarations. It does not implement loss functions.

### Decision 13: Python Surface

The Python surface exposes:

- `fit_regression` or an extended `mltreelib.regression` module with one entry point per problem type.
- Group declaration through a `groups` argument accepting names or indices.
- Panel fitting through `entity_col` and optional `levels` arguments.
- Typed results exposing coefficients, intercept, loss value, convergence status, group selection, robust scale, dispersion, and entity effects where applicable.

Existing `mltreelib.solvers.fit` keeps working unchanged.

### Decision 14: Build Order Rationale

Group metadata lands first because every later phase touches it. Quantile regression lands before Huber and expectile because it is the family users request by name. Group lasso lands after the asymmetric losses because it needs the group substrate and the shared problem framework. Panel work lands after the regression problems because panel leaves are regression problems with a transformed design.

## Exact Build Sequence

Each phase lands as its own commit with focused tests. A phase does not start before the previous phase passes its gate.

### Phase 0 — Group Metadata

1. Add `solver/groups.rs` with `GroupSpec`, `GroupKind`, and validation for range, overlap, and intercept exclusion.
2. Add the group declaration to `ColumnSliceView` and `DenseView` in `solver/design.rs`, keeping existing constructors working.
3. Make missing-indicator columns inherit their source feature group.
4. Add `solver/problems/mod.rs` with the problem registry and the shared group-aware penalty helpers.
5. Add group name resolution from column names and indices.
6. Add Rust tests at `crates/mltreelib-core/tests/regression.rs` for validation, indicator inheritance, serialization, and empty-declaration equivalence.

Gate: `cargo test` passes. Existing fits produce identical results with an empty declaration.

### Phase 1 — Quantile Regression

1. Add `solver/problems/quantile.rs` with the pinball loss, gradient, and asymmetric weight.
2. Add the quantile problem over the generic seam, with a proximal path and a smoothed-IRLS fallback.
3. Promote `LeafKind::Quantile(tau)` to a multivariate model when predictors exist, keeping the 0D behavior for the no-predictor case.
4. Add joint-tau fitting, crossing detection, and optional monotone enforcement.
5. Add minimum-support enforcement and explicit failure reporting.
6. Add the pinball loss to the result contract and remove OLS inference from quantile fits.
7. Add Rust tests on synthetic data with known conditional quantiles, plus crossing detection and support-rule tests.
8. Add Python tests for result fields, tau handling, and reproducibility.

Gate: `cargo test` and Python tests pass. Known conditional quantiles are recovered within tolerance.

### Phase 2 — Huber Regression with Scale

1. Add `solver/problems/huber.rs` with the Huber loss, gradient, and robust scale iteration.
2. Add a weighted median absolute deviation scale estimator with the consistency correction.
3. Add the Huber problem over the generic seam.
4. Add the multivariate Huber leaf, keeping the 0D location leaf unchanged.
5. Report the tuning constant and final scale in the result.
6. Add explicit failure when scale and location iterations diverge.
7. Add Rust tests for outlier resistance, scale recovery, and divergence handling.
8. Add Python tests for result fields and solver-name round trips.

Gate: `cargo test` and Python tests pass. A contaminated fit stays close to the uncontaminated estimate.

### Phase 3 — Expectile Regression

1. Add `solver/problems/expectile.rs` with the asymmetric squared loss and gradient.
2. Share the asymmetric-weight implementation with the quantile problem.
3. Add the expectile leaf with mean-recovery at the central expectile.
4. Add Rust tests for mean recovery, asymmetry, and level ordering.
5. Add Python tests for result fields and reproducibility.

Gate: `cargo test` and Python tests pass. Mean recovery holds within tolerance.

### Phase 4 — Non-Negative Least Squares

1. Add `solver/problems/nnls.rs` with the Lawson-Hanson active-set algorithm over the existing Cholesky path.
2. Register non-negativity as a solver capability in the generic capability table.
3. Add the active-set selection to the result.
4. Add Rust tests for feasibility, active-set correctness, degenerate designs, and intercept treatment.
5. Use NNLS as the bounded-solver test case for the generic seam.

Gate: `cargo test` and Python tests pass. Every returned coefficient is nonnegative.

### Phase 5 — Group Lasso and Sparse-Group Lasso

1. Add group soft thresholding to the coordinate descent update in `solver/ccd.rs`, preserving scalar behavior when no groups exist.
2. Add the group-size normalization convention and document it.
3. Add sparse-group lasso as coordinate thresholding inside group thresholding.
4. Add deterministic group ordering by name and member index.
5. Add grouped monotone constraints so a group shares one sign restriction.
6. Add group selection reporting to the result.
7. Add Rust tests comparing against a fixed reference, plus scalar-equivalence and overlap-rejection tests.
8. Add Python tests for group declaration by name and index.

Gate: `cargo test` and Python tests pass. Scalar lasso results are unchanged when no groups exist.

### Phase 6 — Negative Binomial and Beta Regression

1. Add `NegativeBinomial` and `Beta` variants to the family set in `solver/family.rs`, with variance, deviance, gradient, and initial intercept.
2. Add the dispersion and precision estimation loops in `solver/problems/glm_extra.rs`.
3. Handle the zero-dispersion boundary explicitly.
4. Add a Poisson-versus-negative-binomial dispersion test to the result.
5. Add minimum-support rules for dispersion and precision.
6. Add Rust tests for overdispersion recovery, dispersion reporting, and boundary handling.
7. Add Python tests for family selection and result fields.

Gate: `cargo test` and Python tests pass. Overdispersed data recovers a positive dispersion estimate.

### Phase 7 — Panel Entity Grouping and Within-Transformation

1. Add panel entity keys and optional nested levels to the design declaration in `solver/groups.rs`.
2. Add `solver/panel/demean.rs` with within-transformation, entity mean caching, and unbalanced-panel handling.
3. Report entities with a single observation as contributing nothing to the within fit.
4. Add the unseen-entity fallback to the fit-level group mean, with a diagnostic.
5. Add Rust tests for slope recovery on simulated fixed-effects data, unbalanced panels, and caching reuse.

Gate: `cargo test` passes. Within-estimator slopes match the simulated truth within tolerance.

### Phase 8 — Panel Fixed-Effects Leaf

1. Add `solver/panel/fixed_effects.rs` with the within-estimator and entity effect recovery.
2. Add the panel fixed-effects leaf to the leaf module, consuming the panel declaration.
3. Keep the pooled estimator available for comparison.
4. Report entity effects in the result and the model snapshot.
5. Add Rust tests comparing pooled and within slopes, and checking entity effect recovery.
6. Add Python tests for panel fitting and entity-key handling.

Gate: `cargo test` and Python tests pass. Entity heterogeneity is removed.

### Phase 9 — Hierarchical Partial Pooling Foundation

1. Add `solver/panel/pooling.rs` with the pooling level selector and shrinkage weight.
2. Add a documented observation-count-based weight rule.
3. Expose the pooling level through the leaf control.
4. Add Rust tests showing a sparse subgroup borrows strength from its parent.
5. Add Python tests for pooling level selection.

Gate: `cargo test` and Python tests pass. Sparse subgroups improve over unpooled fits.

### Phase 10 — Conditional GAM Work

1. Improve MARS knot placement and regularization strength.
2. Build a penalized-spline GAM prototype over the generic seam.
3. Add a MARS-versus-GAM comparison to the benchmark suite.
4. Add a GAM leaf only when the prototype wins on accuracy or stability at equal interpretability.

Gate: benchmark evidence exists. A GAM leaf ships only with a documented win.

## Testing Decisions

- **Good tests assert external behavior.** Tests cross the highest available seam. They verify recovered parameters, loss values, feasibility, status, counts, and reproducibility. They do not assert private update formulas or internal vector layout.
- **Primary Rust seam.** Call each regression problem through the generic optimization entry point with deterministic synthetic objectives. Assert recovered parameters, loss, feasibility, and determinism.
- **Primary Python seam.** Call each public regression entry point. Assert result types, result fields, exception behavior, and deterministic repeatability.
- **Model-fitting seam.** Compare standalone regression fitting and single-leaf fitting on the same weighted design, family, penalty, and method. Assert coefficients, predictions, and objective agree within solver tolerance.
- **Group seam.** Assert validation for out-of-range members, overlaps, and intercept inclusion. Assert indicator inheritance. Assert scalar lasso equivalence with no groups. Assert group selection reporting.
- **Quantile seam.** Assert recovery of known conditional quantiles. Assert crossing detection. Assert monotone enforcement. Assert the minimum-support rule. Assert that no OLS inference appears in quantile results.
- **Huber seam.** Assert outlier resistance against a clean fit. Assert scale recovery on simulated contaminated data. Assert explicit failure on divergence. Assert the 0D leaf is unchanged.
- **Expectile seam.** Assert mean recovery at the central expectile, asymmetry of the fitted curve, and monotonicity across levels.
- **NNLS seam.** Assert coefficient nonnegativity, active-set correctness against a fixed reference, degenerate-design handling, and documented intercept treatment.
- **Negative binomial seam.** Assert dispersion recovery on simulated overdispersed counts, dispersion reporting, the Poisson comparison test, and the zero-dispersion boundary.
- **Panel seam.** Assert within-slope recovery on simulated fixed-effects data, entity effect recovery, unbalanced-panel correctness, single-observation entity handling, and the documented unseen-entity fallback.
- **Pooling seam.** Assert a sparse subgroup borrows strength from its parent, the weight rule holds, and pooling level selection changes results as documented.
- **Compatibility seam.** Table-driven tests cover problem, family, L1, L2, group, monotone, and panel combinations. Unsupported combinations fail before objective evaluation.
- **Failure seam.** Invalid groups, wrong dimensions, invalid objective values, exhausted budgets, unstable dispersion, and diverging robust fits return distinct documented statuses or errors.
- **Benchmark smoke test.** Run a tiny budget through every regression problem on one deterministic objective and one small dataset. Verify output schema, counts, statuses, and artifact creation.
- **Benchmark oracles.** Compare against independent implementations for quantile regression, group lasso, and NNLS. Include RANSAC and Theil-Sen as robust-regression references.
- **Prior art.** Reuse patterns from `crates/mltreelib-core/tests/solver.rs`, the existing weighted GLM reference comparisons, and the deterministic RNG tests.

## Out of Scope

- Kernel ridge regression, support vector regression, and Gaussian process regression.
- BART, Bayesian tree ensembles, and other Monte Carlo posterior methods.
- Deep neural regressors and foundation-model regressors.
- Explainable Boosting Machine as a leaf family. It is an ensemble strategy.
- Zero-inflated and hurdle models.
- Mixed-effects models with a random-effects design and sparse Cholesky. Phase 9 is a deterministic partial-pooling foundation.
- Dynamic panel GMM, TVAR, and TVECM. Spec 17 owns them.
- Partial least squares and supervised projection. The component-analysis note owns them.
- Overlapping coefficient groups.
- Joint quantile regression with cross-quantile constraints beyond monotone ordering.
- External regression crates as runtime dependencies.

## Further Notes

- Research basis for method selection, priority, and out-of-scope decisions: `references/research/2026-11-21-regression-algorithm-landscape.md`.
- This specification depends on Spec 26 Phase 0 for the generic seam and Phase 1 for the leaf-to-solver entry point. It also depends on Spec 17 for the panel vision, but implements only the fixed-effects foundation.
- Spec 23 covers the expanded classical leaf catalog. Where it lists ordinal regression, GAM, or mixture leaves, this spec owns the regression problems those leaves need. Spec 23 owns leaf promotion and capability gating.
- Spec 25 owns the common report contract. This spec adds loss values, group selection, robust scale, dispersion, and entity effects to the report through that contract.
- The generic solver seam in `crates/mltreelib-core/src/solver/generic/` already carries the objective, bounds, constraint, budget, request, and result types. This specification consumes them rather than redefining them.
- Current source wins where older notes disagree. `solver/lbfgs.rs` implements L-BFGS, `solver/ccd.rs` implements coordinate descent, `solver/family.rs` holds the GLM families, and `solver/evolutionary.rs` holds four population methods.
- One multi-core job at a time. Rayon-backed solvers and benchmark suites must not run concurrently with other multi-core fits.
