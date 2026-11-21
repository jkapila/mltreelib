# Spec 23: Expanded Classical Leaf-Family Catalog for `mltreelib`

Status: proposed

## Problem Statement

`mltreelib` has grown around a strong divide-and-conquer architecture: the tree separates observations in data space, while each terminal leaf fits a local response model. The current leaf portfolio covers linear and generalized linear models, constant robust locations, MARS, binary probability summaries, multinomial softmax, KNN, heterogeneous Gaussian NLL, Tobit, AFT, mixture-cure survival, spline survival, and planned causal or copula families.

The portfolio does not yet expose many classical machine-learning models as direct, Rust-backed consumers. Users cannot directly fit or reuse LDA, QDA, regularized discriminant analysis, categorical or Gaussian Naive Bayes, mixed-type Naive Bayes, ordinal regression, generalized additive models, latent-class mixtures, prototypes, regularized quantile regression, kernel models, Gaussian processes, or several structured distributional families as local leaf models.

Implementing each family as an unrelated estimator would create inconsistent fitting, prediction, missing-data, fallback, serialization, and benchmark behavior. Adding a large flat `LeafKind` enum would also force every leaf family to share unrelated controls and fitted payloads.

The package needs a comprehensive classical-model leaf architecture, not a one-off discriminant or Naive Bayes patch. The architecture must permit direct consumption of individual Rust estimators while keeping the tree as an optional higher-level coordinator.

## Solution

Build one extensible leaf-model subsystem with:

1. **A common leaf-model contract** covering fit, prediction, probability, distribution, explanation, diagnostics, and serialization.
2. **A typed model catalog** containing the classical and modern families most useful for local tabular learning.
3. **A registry and capability matrix** that declares supported response types, feature types, penalties, constraints, missing-data policies, minimum sample requirements, and fallback behavior.
4. **Shared sufficient statistics and numerical utilities** for count, moment, covariance, spline, and EM calculations.
5. **Direct estimator access** in Rust and Python so users can use a leaf model without growing a tree.
6. **Tree integration** through the same model contract, with leaf-specific fallback diagnostics.
7. **An evaluation harness** that compares every leaf against constant, linear, softmax, KNN, and boosted-tree baselines under realistic budgets.

The first implementation wave prioritizes stable classical models with high value and manageable complexity. Later waves add flexible, structured, and research-grade families. No model enters the default public catalog without an external reference, deterministic behavior, documented fallbacks, and evidence against existing leaves.

## User Stories

### Common Leaf Contract

1. As a Rust user, I want to fit a leaf model directly on a typed frame, so that I can consume classical estimators without constructing a tree.
2. As a Python user, I want to fit the same leaf model directly, so that Rust and Python expose equivalent behavior.
3. As a tree user, I want every supported leaf family to work through `LeafControl`, so that tree growth does not require special-case constructor paths.
4. As a tree user, I want a leaf model to report whether it used its requested estimator or a documented fallback, so that a weak local model is never hidden.
5. As a tree user, I want leaf fitting to use row weights consistently, so that honesty samples and weighted objectives remain coherent.
6. As a tree user, I want unsupported family and feature combinations rejected before fitting, so that categorical columns are never silently coerced.
7. As a tree user, I want missing-data behavior explicit per model, so that complete-case, skip, local imputation, and missing-indicator semantics remain auditable.
8. As a tree user, I want leaf predictions to expose class probabilities and response distributions where supported, so that the same fitted leaf can serve prediction, calibration, and anomaly analysis.
9. As a tree user, I want leaf explanations to identify selected features, constraints, components, or neighborhoods, so that local models remain inspectable.
10. As a tree user, I want leaf fit diagnostics to report rows used, rows dropped, effective parameters, fallback reason, and fit time, so that tree behavior can be audited.
11. As a tree user, I want fitted leaf state to round-trip through the existing snapshot format, so that predictions remain identical after persistence.
12. As a library maintainer, I want each leaf family to own its fitted payload and fit logic, so that the common contract remains shallow and stable.
13. As a library maintainer, I want family-specific options grouped under typed configuration, so that the root model interface does not become a flat parameter bag.
14. As a library maintainer, I want shared sufficient statistics reused across LDA, QDA, NB, GAM, and mixture models, so that implementations remain efficient.
15. As a benchmark author, I want one evaluation harness for all direct and tree-used leaf models, so that evidence remains comparable.

### Ordinal Regression Leaves

16. As a data scientist, I want cumulative logit leaves, so that ordered targets use their ordering information.
17. As a data scientist, I want cumulative probit leaves, so that I can choose a latent-normal ordinal model.
18. As a user, I want ordinal leaves to return class probabilities and latent thresholds, so that predictions and explanations remain inspectable.
19. As a user, I want proportional-odds behavior documented, so that violations are visible.
20. As a researcher, I want an optional non-proportional or continuation-ratio ordinal leaf, so that richer ordinal structure can be evaluated separately.

### Discriminant Analysis Leaves

21. As a data scientist, I want pooled LDA leaves, so that I can use a classical generative classifier locally.
22. As a data scientist, I want diagonal LDA leaves, so that tiny leaves retain a stable fallback.
23. As a data scientist, I want Ledoit-Wolf shrinkage LDA leaves, so that covariance estimation adapts to leaf size.
24. As a data scientist, I want regularized discriminant analysis leaves, so that LDA and QDA are controlled by explicit interpolation parameters.
25. As a data scientist, I want QDA leaves with mandatory covariance regularization, so that class-specific covariance cannot silently overfit.
26. As a data scientist, I want sparse LDA leaves, so that irrelevant predictors receive zero coefficients.
27. As a researcher, I want sparse QDA and Community Bayes leaves, so that selected interactions and feature communities remain visible.
28. As a researcher, I want robust discriminant leaves with optional robust scatter estimation, so that contaminated leaves can be handled explicitly.
29. As a researcher, I want flexible discriminant analysis over spline or MARS bases, so that nonlinear boundaries remain interpretable.
30. As a researcher, I want an experimental exponential-family discriminant leaf, so that mixed count, positive, and binary features can use correct marginals.

### Naive Bayes and Bayesian Classifier Leaves

31. As a data scientist, I want categorical Naive Bayes leaves, so that category frequencies become local class evidence.
32. As a data scientist, I want Gaussian Naive Bayes leaves, so that continuous features use class-conditional normals.
33. As a data scientist, I want mixed-type Naive Bayes leaves, so that each feature uses an appropriate marginal likelihood.
34. As a data scientist, I want m-estimate and Laplace smoothing options, so that sparse cells remain finite.
35. As a data scientist, I want AODE leaves, so that one-dependence structure is captured without full network search.
36. As a researcher, I want TAN leaves with support guards, so that one additional predictor parent can be learned safely.
37. As a researcher, I want KDB leaves with an explicit dependence order, so that higher-order local dependencies can be evaluated.
38. As a researcher, I want generalized Naive Bayes or copula-based structure leaves, so that selected feature dependence can be tested.
39. As a user, I want each Bayesian leaf to expose class priors, conditional support, selected parents, and smoothing, so that predictions remain explainable.

### Additive, Nonlinear, and Robust Leaves

40. As a data scientist, I want a GAM leaf, so that smooth per-feature effects remain visible.
41. As a data scientist, I want GAM-tooth or cyclic-spline shapes, so that nonlinear response patterns can be represented.
42. As a data scientist, I want explicit GAM interaction controls, so that pairwise terms do not appear silently.
43. As a data scientist, I want robust linear or LAD leaves, so that local outliers do not dominate least squares.
44. As a data scientist, I want regularized quantile regression leaves, so that conditional quantiles can be fitted directly.
45. As a data scientist, I want pinball-loss and distribution reporting for quantile leaves, so that loss and predicted quantiles remain consistent.
46. As a data scientist, I want Huber linear leaves with optional IRLS or robust weighting, so that residual assumptions are explicit.
47. As a data scientist, I want prototype or medoid leaves, so that local classification can use representative observations.
48. As a data scientist, I want Gower prototype leaves, so that mixed numeric and categorical rows can use local representatives.

### Mixture and Latent-Class Leaves

49. As a data scientist, I want Gaussian mixture leaves, so that a leaf can contain multiple local regimes.
50. As a data scientist, I want class-conditional mixture leaves, so that class distributions can be multimodal.
51. As a data scientist, I want latent-class leaves for categorical or mixed data, so that hidden segments can be modeled.
52. As a user, I want mixture weights and component parameters exposed, so that fits remain auditable.
53. As a user, I want deterministic component ordering, so that snapshots and comparisons remain stable.
54. As a user, I want mixture leaves to fall back to their parent model when components are unsupported by leaf size, so that sparse leaves remain stable.

### Distributional Leaves

55. As a user, I want beta regression leaves, so that bounded continuous targets receive correct likelihoods.
56. As a user, I want zero-inflated count leaves, so that excess zeros can be modeled explicitly.
57. As a user, I want hurdle or zero-inflated regression leaves, so that point mass and continuous components remain separate.
58. As a user, I want negative-binomial count leaves, so that overdispersion can be represented.
59. As a user, I want GAMLSS-style location, scale, and shape leaves, so that a leaf can model more than its mean.
60. As a user, I want parameter-domain validation for every distributional leaf, so that invalid likelihoods fail clearly.
61. As a user, I want distributional leaves to expose predictive CDF and density where supported, so that interval and anomaly queries remain available.

### Kernel and Gaussian Process Leaves

62. As a researcher, I want kernel SVM leaves, so that local maximum-margin classification can be evaluated.
63. As a researcher, I want kernel ridge leaves, so that local nonlinear regression can be evaluated.
64. As a researcher, I want Gaussian-process leaves with posterior uncertainty, so that local predictive distributions can be studied.
65. As a user, I want GP leaves to require an explicit approximation or leaf-size limit, so that exact cubic scaling cannot surprise me.
66. As a user, I want kernel and GP methods separated from native tree fitting, so that their cost and semantics remain clear.

### Structured Leaves

67. As a researcher, I want matrix-variate leaves, so that matrix-shaped observations preserve their geometry.
68. As a researcher, I want tensor or functional leaves, so that structured responses do not require flattening.
69. As a researcher, I want panel or grouped-data leaves, so that repeated entities can use hierarchical structure.
70. As a user, I want structured leaves to declare their input type, so that they cannot be selected for ordinary flat frames accidentally.

### Direct Consumption and Tree Use

71. As a Rust user, I want direct leaf-model constructors and fit methods, so that I can compare and reuse the estimators independently.
72. As a Python user, I want a `fit_leaf` or equivalent public function, so that I can use the model catalog without tree boilerplate.
73. As a tree user, I want direct and tree-fitted versions to share numerical code, so that results do not diverge.
74. As a tree user, I want tree split criteria to remain separate from leaf model choice, so that the tree keeps its divide-and-conquer responsibility.
75. As a tree user, I want leaf objective improvements to influence split selection only through the shared model-based scan when explicitly enabled, so that default growth semantics remain stable.
76. As a benchmark author, I want tree-level and direct-fit results compared, so that local flexibility shows measurable value.

### Evaluation and Governance

77. As a benchmark author, I want every leaf compared against a constant baseline, so that complexity must earn its gain.
78. As a benchmark author, I want every leaf compared against the best existing linear or softmax leaf, so that local specialization has a strong comparator.
79. As a benchmark author, I want every leaf compared with a global version of the same model, so that local fitting has a direct benefit test.
80. As a benchmark author, I want accuracy, log loss, Brier score, calibration, interval coverage, runtime, memory, and fallback rate reported, so that no single metric dominates.
81. As a benchmark author, I want benchmark results grouped by data dimension, leaf size, missingness, class imbalance, and feature type, so that failure conditions remain visible.
82. As a library maintainer, I want an experimental label for preprints and unstable estimators, so that stable defaults are clear.
83. As a library maintainer, I want model deprecation and compatibility rules, so that snapshots remain readable across versions.
84. As a library maintainer, I want external reference tests for every public leaf family, so that numerical correctness is not defined by internal consistency alone.

## Implementation Decisions

### Decision 1: A Common Leaf-Model Contract

The Rust core owns a common leaf-model interface. The interface supports:

- Fit on selected rows, weights, typed predictors, and typed response.
- Predict response mean, class probability, and response distribution when supported.
- Produce a per-row explanation or diagnostic contribution when supported.
- Report fit diagnostics, effective parameter count, rows used, and fallback reason.
- Serialize and restore fitted state through versioned snapshots.
- Validate requested capabilities before fitting.

The interface is implemented through an internal leaf-model enum or type-specific fitted payload. Public dispatch stays small and does not expose a giant flat options struct.

### Decision 2: Public Catalog Is Capability-Gated

Each family declares:

- Supported response kind: continuous, binary, multinomial, ordinal, count, bounded continuous, censored, or survival.
- Supported predictor types: numeric, binary, categorical, grouped, ordered, matrix, tensor, or panel.
- Required sample size and dimensional conditions.
- Supported penalties and constraints.
- Missing-data policies.
- Probability, density, CDF, interval, and explanation capabilities.
- Whether it is stable or experimental.

The catalog rejects unsupported combinations before fitting.

### Decision 3: Direct Use and Tree Use Share One Core

Every public leaf estimator can be fitted directly in Rust and Python. Tree fitting calls the same estimator. No duplicate implementation may exist between standalone and tree paths.

The tree remains responsible for:

- Selecting rows reaching each leaf.
- Choosing predictors and split columns.
- Measuring candidate split gain.
- Choosing when to stop splitting.
- Pruning and honest sample assignment.

The leaf remains responsible for the local response model.

### Decision 4: Typed Configuration

Leaf-family configuration uses typed per-family settings rather than a root-level flat bag. Shared controls cover solver, column policy, L1/L2, monotone constraints, missing-data policy, seed, and iteration budget. Family-specific controls cover ordinal thresholds, discriminant shrinkage, Bayesian smoothing and dependence order, mixture count, GAM basis and smoothing, quantile level, GP approximation, and structured input shape.

Existing `LeafControl` remains backward compatible. New configuration may be represented by an enum or nested settings while preserving existing flat arguments.

### Decision 5: Stable Shared Numerical Utilities

The core provides shared utilities for:

- Weighted counts, sums, means, and variances.
- Class priors and class-conditional moments.
- Pooled and class-specific covariance matrices.
- Numerically stable Cholesky and pseudoinverse operations.
- Categorical count tables with smoothing.
- Spline bases and penalty evaluation.
- EM responsibility updates and deterministic component ordering.
- Quantile and pinball-loss calculations.

These utilities live behind narrow internal seams and are reused by direct estimators and trees.

### Decision 6: First Implementation Wave

The first wave ships the highest-value classical families:

1. Cumulative logit and probit ordinal leaves.
2. Shrinkage LDA, diagonal LDA fallback, and RDA leaves.
3. Categorical, Gaussian, and mixed-type Naive Bayes leaves.
4. GAM leaf implemented through shared spline and backfitting utilities, with explicit comparison against MARS.
5. Robust linear and regularized quantile leaves.
6. Low-dimensional Gaussian mixture leaf.
7. Gower prototype leaf.

The first wave also establishes the direct estimator catalog, capability matrix, diagnostics, and evaluation harness.

### Decision 7: Second Implementation Wave

The second wave adds:

- Sparse LDA.
- AODE.
- TAN and KDB with support and leaf-size guards.
- Latent-class and class-conditional mixture leaves.
- Beta, negative-binomial, zero-inflated, and hurdle leaves.
- Flexible discriminant analysis over spline bases.
- Modular GAM interaction options.

### Decision 8: Research and Heavy Leaves

Experimental models include:

- Sparse QDA and Community Bayes.
- Robust discriminant analysis.
- Exponential-family discriminant analysis.
- Generalized or copula Naive Bayes.
- Neural KDB.
- Kernel SVM and kernel ridge leaves.
- Gaussian-process leaves with sparse approximations.
- Matrix, tensor, functional, and panel leaves.

These require independent reproduction and explicit computational budgets before stable status.

### Decision 9: Fallback Policy

Fallbacks are typed and visible.

- One-class leaf: constant class probability.
- Insufficient rows for requested structure: documented simpler model within the same family, or explicit failure when the caller forbids fallback.
- Singular covariance: diagonal or ridge covariance according to documented policy.
- Missing required features: explicit missing component or documented failure.
- Unsupported family: validation error before objective evaluation.

A fallback must never masquerade as the requested estimator.

### Decision 10: Missing Data

The leaf catalog defines supported policies per family:

- `DropFallback`: drop rows lacking required fields and use a documented local fallback at prediction.
- `ImputeLocal`: impute with training-leaf statistics where the family permits.
- `Indicator`: add explicit missingness components where supported.
- `SkipLikelihood`: omit unavailable feature terms, used by probabilistic categorical models.
- `ModelMissing`: estimate a missingness component.

No leaf converts missing numeric values to zero implicitly.

### Decision 11: Split and Leaf Separation

Default tree growth keeps scalar variance-reduction or configured split criteria. Leaf-family changes do not silently alter split semantics.

A future model-based scan may evaluate candidate splits using fitted leaf objectives. That scan must be explicit, budgeted, and benchmarked. It is not part of the first-wave requirement.

### Decision 12: Persistence

The existing snapshot format gains a versioned leaf-model tag. Each fitted leaf stores enough state for exact prediction:

- Class probabilities and priors.
- Means, variances, covariances, and selected features.
- Count tables and smoothing.
- Selected parents or dependency structures.
- Spline bases and coefficients.
- Mixture weights, components, and ordering.
- Thresholds and ordinal link parameters.
- Feature scales and missingness state.

Restored models must produce identical predictions and diagnostics.

### Decision 13: Benchmark Contract

Every leaf family receives one realistic benchmark program containing:

- Synthetic data with known generating model.
- Public tabular datasets.
- Multiple seeds.
- Small, medium, and large leaf sizes.
- Numeric, categorical, mixed, ordinal, count, bounded, censored, and missing-heavy responses where supported.
- Direct-fit and tree-fit comparisons.
- Constant, linear, softmax, KNN, and boosted-tree baselines.

Metrics include task-appropriate accuracy, log loss, Brier score, calibration error, interval coverage, objective value, fit and prediction time, memory, effective parameters, fallback rate, and explanation size.

### Decision 14: Promotion Gates

A family becomes stable when:

- External reference values match within documented tolerance.
- Deterministic behavior is covered.
- Snapshot round-trip is exact.
- Missing and fallback behavior is covered.
- It improves or specializes meaningfully against existing leaves on at least one documented use case.
- It does not regress default runtime or snapshot size materially.
- Documentation and examples exist.

Preprints and unverified numerical claims remain experimental.

### Decision 15: Extension Points

Future families can be added without modifying the core contract. The catalog uses an internal registry keyed by model kind. Python bindings are generated or manually maintained from the same canonical names and capability matrix.

Plugin discovery is not part of this specification.

### Decision 16: Relationship to Existing Specifications

- Spec 03 remains the baseline distributional-leaf architecture.
- Spec 08 remains the multinomial and survival-leaf baseline.
- Spec 11 remains the detailed GAMLSS specification.
- Spec 13 remains the Context Tree Weighting specification.
- Spec 14 remains the Cubist and model-forest direction.
- Spec 16 remains the copula and generative-forest direction.
- Spec 21 defines the shared optimizer infrastructure used by leaf estimators.
- Spec 24 defines the separate association-rules and Context Tree integration.

### Decision 17: Common Model Report Contract

Every fitted leaf model exposes one report with four required sections:

- **Objective:** mathematical objective, optimization direction, final value, objective components, and units where defined.
- **Learner:** estimator family, mathematical representation, fitted structure, nominal and effective complexity, and supported capabilities.
- **Controller:** selected configuration, tuned regularization or structural knobs, seed, budgets, missing-data policy, and fallback reason.
- **Diagnostics:** convergence status, fit and evaluation time, evaluation count, objective improvement, task-specific quality, stability, calibration or coverage where supported, and warnings.

The report has one structured representation, one text renderer, and one machine-readable serialization. Renderers consume structured values and never reparse text.

Classical standard errors, confidence intervals, and p-values appear only for estimators with valid inference. Models without coefficient-level inference expose descriptive diagnostics, bootstrap intervals when justified, or an explicit statement that inferential statistics are unavailable. The common report must not fabricate statsmodels-style tests for KNN, clustering-like, mixture, discriminant, or penalized estimators without defensible inference.

## Testing Decisions

- **Interface seam:** Test every family through direct fit and predict. Assert returned capability fields match the catalog.
- **Reference seam:** Compare each family with fixed expected values and external reference implementations in benchmark-only tests.
- **Tree equivalence seam:** A direct leaf fit and the same model fitted through a depth-zero tree must agree in coefficients, probabilities, and predictions.
- **Snapshot seam:** Round-trip each fitted family and assert identical predictions, diagnostics, and selected structure.
- **Capability seam:** Every unsupported family, feature type, response type, penalty, and constraint combination fails before fitting.
- **Fallback seam:** Every fallback path reports its exact reason and returns documented behavior.
- **Missing-data seam:** Every supported missing policy preserves missing semantics and never treats missing as zero.
- **Weight seam:** Weighted fits match equivalent replicated-row references.
- **Ordinal seam:** Verify thresholds, monotonic probabilities, logit and probit link limits, and reference likelihoods.
- **Discriminant seam:** Recover known class means and covariance structure. Verify LDA, RDA, and QDA boundaries against reference implementations. Small leaves must trigger documented covariance fallback.
- **Bayesian seam:** Verify smoothed probabilities, missing-feature skipping, AODE aggregation, TAN parent selection, KDB support coverage, and NB-tree benchmark behavior.
- **GAM seam:** Verify shape-function recovery, monotonicity when constrained, interaction controls, and comparison against MARS.
- **Mixture seam:** Verify component recovery, deterministic ordering, EM responsibility semantics, and fallback when components exceed supported size.
- **Distribution seam:** Verify support constraints, log likelihood, CDF, density, and predictive intervals for every distributional family.
- **Benchmark smoke seam:** Fit every family under a tiny budget on one deterministic dataset and verify result schema and capability reporting.
- **Prior art:** Reuse patterns from existing solver, leaf-family, MARS, softmax, survival, categorical encoding, and snapshot tests.

## Out of Scope

- Neural network leaves, deep tabular foundation models, and autodiff integration.
- Unlimited exact Gaussian-process leaves.
- Unbounded kernel matrices without explicit memory limits.
- Full external plugin discovery.
- Replacing the tree's divide-and-conquer growth mechanism with a monolithic learner.
- Treating association-rule sets as terminal leaf predictions.
- Implementing every research paper in one release.
- Reimplementing SciPy, statsmodels, or an AutoML framework.
- GPU-specific estimator kernels.

## Further Notes

- This specification deliberately captures a broad classical-model roadmap while preserving staged implementation gates.
- Association mining is specified separately because its output is a rule set rather than one local response value.
- Kernel and Gaussian-process models may be better as split-side methods because their repeated leaf cost can exceed the value of local fitting.
- Ordinal regression, discriminant analysis, and Naive Bayes provide the best immediate complement to current leaves.
- The exact catalog should evolve through evidence. New model families should first demonstrate why existing leaves cannot cover their use case.
- Research evidence is summarized in `references/research/2026-11-21-core-ml-leaf-family-gap-analysis.md`, `references/research/2026-11-21-discriminant-analysis-leaf-landscape.md`, and `references/research/2026-11-21-naive-bayes-bayesian-leaf-landscape.md`.
