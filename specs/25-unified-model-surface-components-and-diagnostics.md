# Spec 25: Unified Model Surface, Component Analysis, and Statistical Diagnostics

Status: proposed

## Problem Statement

`mltreelib` has grown across four distinct unsupervised or semi-supervised capability families:

- Anomaly detection.
- Clustering.
- Association-rule mining.
- Component analysis and matrix factorization.

Each family currently exposes a different fit lifecycle, result shape, configuration-inspection method, and diagnostic depth. The tree package also exposes `get_params`, a scikit-learn estimator compatibility convention, despite `mltreelib` not importing or depending on scikit-learn. That convention creates an accidental public contract around cloning, nested parameter dictionaries, and `deep` inspection that does not fit the package's typed-control architecture.

The package also lacks a consolidated component-analysis capability. PCA, truncated SVD, whitening, FastICA, NMF, related factorizations, and their anomaly or clustering uses exist across research notes and the Rust ecosystem, but no native `mltreelib` facade owns them.

Finally, model reporting is uneven. Linear regression and GLM paths expose R/statsmodels-style coefficient tables. Other estimators expose fragments such as rules, leaf statistics, GLOSH scores, or solver convergence, but no common contract explains:

- What objective the model optimizes.
- Which mathematical learner produced the fit.
- Which controller settings and budgets shaped learning.
- Which diagnostics show whether the result is trustworthy.
- Which inference statements are mathematically valid.

Forcing one universal estimator interface across every task would be too rigid. Leaving each task with an unrelated lifecycle would make agents and humans reason from ad hoc APIs. The solution is one stable surface pattern per task, backed by one shared report model and one shared component-analysis capability.

## Solution

Introduce one coherent model-surface architecture:

1. **Task facades with a common lifecycle.**
   - `mltreelib.anomaly.AnomalyModel`
   - `mltreelib.cluster.ClusterModel`
   - `mltreelib.rules.RuleModel`
   - `mltreelib.components.ComponentModel`
2. **Existing names remain compatibility facades.**
   - `AnomalyTree` and `ClusterTree` continue working while delegating to stable task facades.
3. **A common `fit`, `predict` or `score`, `report`, and persistence lifecycle.**
4. **Keep familiar scikit-learn-shaped verbs without `BaseEstimator` compatibility.**
   - Preserve `fit`, `predict`, `transform`, and task-specific `score_samples` vocabulary.
   - Remove public `get_params` and `set_params` conventions.
   - Do not add cloning, `deep`, or generic `score` compatibility.
   - Keep the package free of any scikit-learn import or dependency.
   - Preserve typed controls, raw typed columns, explicit weights, dtypes, and package-native diagnostics.
5. **A shared report contract with Objective, Learner, Controller, and Diagnostics sections.**
6. **A consolidated component-analysis capability.**
   - PCA.
   - Truncated SVD.
   - Whitening.
   - FastICA.
   - NMF.
7. **Stable and experimental namespaces.**
   - Stable methods live in the primary task facade.
   - Research methods live under `mltreelib.experimental`.
8. **Shared solver infrastructure.**
   - Existing IRLS, L-BFGS, CCD, CMA-ES, SHADE, sNES, and GA.
   - Add BFGS, nonlinear conjugate gradient, Nelder-Mead, and Powell.
   - L-BFGS-B remains part of the bounded-solver roadmap.
9. **Rust-first implementation.**
   - The Rust core owns numerical behavior and persisted state.
   - Python remains a thin, stable binding.

This is a separate specification theme. It does not expand the model-tree leaf catalog in Spec 23, association mining in Spec 24, or solver mechanics in Spec 21. It defines how all those capabilities meet users through consistent facades, reports, and diagnostics.

## User Stories

### Common Task Facade Lifecycle

1. As a Python user, I want one `AnomalyModel` facade for anomaly methods, so that I select a method with `kind` rather than importing a different class.
2. As a Python user, I want one `ClusterModel` facade for clustering methods, so that I select a method with `kind` rather than managing separate APIs.
3. As a Python user, I want one `RuleModel` facade for rule mining and associative classification, so that changing mining strategy does not change workflow code.
4. As a Python user, I want one `ComponentModel` facade for PCA, SVD, ICA, NMF, and related decomposition, so that component analysis remains discoverable.
5. As a user, I want `fit` to accept raw typed rows, optional row weights, column dtypes, names, and task-specific targets, so that preprocessing stays explicit and package-native.
6. As a user, I want `fit` to return the fitted facade or a task-specific fitted result, so that one familiar lifecycle works across tasks.
7. As a user, I want unsupported method, task, dtype, and target combinations rejected before learning, so that invalid configuration never silently changes semantics.
8. As a user, I want `predict` or `score` to use only fitted state, so that prediction never refits preprocessing.
9. As a user, I want `report` to return one structured report after fitting, so that I can inspect every model consistently.
10. As a user, I want `save`, `load`, JSON, bytes, and pickle round trips to preserve fitted state and diagnostics, so that deployed models remain reproducible.
11. As a user, I want `config()` to return the effective typed configuration, so that I can inspect settings without adopting scikit-learn's `get_params` contract.
12. As a user, I want direct attribute access for common configuration values, so that simple inspection stays concise.
13. As an agent, I want one task-level entry point per capability, so that method selection guidance remains stable.
14. As an agent, I want method-specific settings nested under their method or task control, so that I can change one knob without reconstructing an estimator.
15. As a library maintainer, I want existing `AnomalyTree` and `ClusterTree` names retained, so that current code does not break.

### Stable and Experimental Separation

16. As a user, I want stable methods in primary task namespaces, so that ordinary imports remain small and dependable.
17. As a researcher, I want experimental methods under `mltreelib.experimental`, so that promising methods are available without weakening stable defaults.
18. As a user, I want experimental results marked in metadata, so that I know which guarantees apply.
19. As a user, I want an experimental method to fail clearly when its required optional feature is absent, so that missing dependencies remain visible.
20. As a benchmark author, I want stable and experimental methods reported in separate groups, so that experimental wins cannot silently redefine stable defaults.

### Objective Report Section

21. As a user, I want every report to state the objective, so that I know what the model optimizes.
22. As a user, I want the objective's mathematical form, so that I understand model semantics without reading source.
23. As a user, I want the optimization direction, so that I know whether lower or higher is better.
24. As a user, I want the final objective value, so that I can compare fits.
25. As a user, I want objective components and units where defined, so that composite losses remain interpretable.
26. As a user, I want reconstruction objectives labeled as reconstruction rather than minimization, so that PCA and NMF reports remain accurate.
27. As a user, I want classification objectives described by log loss or deviance, so that model comparisons use the same basis.

### Learner Report Section

28. As a user, I want every report to identify the learner family, so that I know which mathematical method produced the result.
29. As a user, I want the fitted structure described, so that I understand model capacity.
30. As a tree user, I want tree depth, leaf count, leaf sizes, and split diagnostics in the learner section, so that partition complexity remains visible.
31. As a clustering user, I want cluster count, membership counts, centers or prototypes, and noise rate, so that cluster structure remains visible.
32. As an anomaly user, I want detector kind, reference size, score direction, threshold, and search mode, so that scoring semantics remain visible.
33. As a rules user, I want mining algorithm, item representation, hierarchy policy, discretization policy, and rule source, so that rule semantics remain visible.
34. As a component user, I want component count, factor dimensions, loadings, explained variance, and reconstruction mode, so that decomposition remains visible.
35. As a user, I want nominal and effective complexity reported, so that model capacity can be compared.
36. As a user, I want supported capabilities listed, so that I know which prediction, transform, interval, explanation, and update operations are valid.

### Controller Report Section

37. As a user, I want every report to record the selected method and controller, so that I can reproduce the fit.
38. As a user, I want regularization, shrinkage, smoothing, support, component count, or other tuned knobs reported, so that I know what learning controlled.
39. As a user, I want the random seed, iteration budget, and evaluation budget reported, so that stochastic fits remain reproducible.
40. As a user, I want the missing-value policy reported, so that data treatment remains visible.
41. As a user, I want fallback reasons reported, so that a simpler estimator cannot masquerade as the requested one.
42. As a user, I want preprocessing and target transformations reported, so that fitted state remains auditable.

### Diagnostics Report Section

43. As a user, I want every report to include convergence status and reason, so that fit completion is explicit.
44. As a user, I want fit time, prediction time, and evaluation count, so that computational cost remains visible.
45. As a user, I want objective improvement or convergence history, so that progress can be audited.
46. As a user, I want task-specific quality metrics, so that model quality matches the task.
47. As a user, I want calibration diagnostics when probabilities exist, so that ranking quality is not confused with probability quality.
48. As a user, I want stability diagnostics when models can be refit, so that assignment or score stability remains visible.
49. As a user, I want warnings for missingness, unseen levels, non-finite fits, local optima, degenerate covariance, low effective rank, and unsupported inference, so that failure modes remain visible.
50. As a user, I want a single stable method for rendering reports as text, so that console and documentation output remain consistent.

### Inference and statsmodels-Style Output

51. As a user, I want statsmodels-style coefficient tables for models with valid inference, so that linear and GLM output remains familiar.
52. As a user, I want standard errors, test statistics, p-values, and confidence intervals only when mathematically valid, so that reports never fabricate inference.
53. As a user, I want discriminative or regularized models to state why classical inference is unavailable, so that absence is explicit.
54. As a user, I want bootstrap intervals labeled as bootstrap when used, so that uncertainty provenance remains visible.
55. As a user, I want one structured report source with text and Markdown renderers, so that values cannot drift through reparsing.
56. As a user, I want Cubist-style diagnostic summaries for local model families, so that leaf behavior remains inspectable beyond coefficient tables.

### Component Analysis

57. As a user, I want one component facade with a `kind` switch, so that decomposition methods remain easy to select.
58. As a user, I want classical PCA, so that I can reduce dimensionality and inspect variance structure.
59. As a user, I want standardized PCA, so that scale-sensitive features are handled explicitly.
60. As a user, I want truncated SVD, so that wide or sparse data can be decomposed without centering.
61. As a user, I want whitening, so that I can prepare uncorrelated component coordinates for ICA or other models.
62. As a user, I want FastICA, so that I can separate independent non-Gaussian sources.
63. As a user, I want NMF, so that I can obtain additive nonnegative factors.
64. As a user, I want explained variance and reconstruction error, so that component quality remains measurable.
65. As a user, I want component transforms to fit only training data, so that prediction cannot leak information.
66. As a user, I want PCA reconstruction anomaly detection, so that low-rank normal structure can define novelty.
67. As a user, I want NMF reconstruction anomaly detection, so that additive normal structure can define novelty.
68. As a user, I want component scores reusable as cluster features, so that one decomposition supports several downstream tasks.
69. As a user, I want deterministic component ordering where mathematically identifiable, so that reports and snapshots remain stable.
70. As a user, I want ICA ambiguities documented, so that component order and sign are not mistaken for learned semantics.
71. As a user, I want NMF convergence and local-optimum diagnostics, so that reconstruction quality is not overstated.
72. As a user, I want robust, sparse, kernel, probabilistic, tensor, and contrastive component methods under the experimental namespace, so that advanced research remains accessible without changing stable guarantees.

### Extended Solver Catalog

73. As a user, I want IRLS available, so that supported GLM families retain efficient analytic-gradient fitting.
74. As a user, I want L-BFGS available, so that smooth large problems remain memory-efficient.
75. As a user, I want BFGS available, so that smaller smooth problems can use full-memory curvature history.
76. As a user, I want nonlinear conjugate gradient available, so that low-memory first-order search remains available.
77. As a user, I want Nelder-Mead available, so that derivative-free local search can handle nonsmooth objectives.
78. As a user, I want Powell available, so that derivative-free coordinate search can solve separable objectives.
79. As a user, I want every solver to share objective, gradient, penalty, monotone-projection, budget, and convergence semantics, so that method changes remain comparable.
80. As a user, I want solver reports to identify method, iterations, evaluations, final objective, and stop reason, so that solver behavior remains auditable.
81. As a benchmark author, I want new solvers compared under equal evaluation budgets, so that speed and quality claims remain fair.
82. As a library maintainer, I want solver additions to avoid new runtime dependencies, so that the core remains lean.
83. As a library maintainer, I want advanced constrained or trust-region methods deferred until a concrete workload requires them, so that numerical complexity remains controlled.

### Removal of Scikit-Learn Estimator Compatibility

84. As a user, I want no `get_params` method, so that the package does not imply scikit-learn estimator compatibility.
85. As a user, I want no `set_params` method, so that mutation and cloning conventions remain package-native.
86. As a user, I want no generic sklearn-style `score` method, so that task-specific evaluation remains explicit.
87. As a user, I want `config()` instead of `get_params`, so that useful configuration inspection remains available.
88. As a user, I want direct attribute access to remain, so that common settings stay easy to inspect.
89. As a library maintainer, I want the package to keep `fit`, `transform`, `predict`, and `score_samples` vocabulary where semantically meaningful, so that removing estimator compatibility does not require renaming every valid operation.
90. As a library maintainer, I want domain-specific anomaly scoring preserved, so that `score_samples` remains an anomaly operation rather than a generic estimator score.

## Implementation Decisions

### Decision 1: Four Task Facades, No Universal Dispatcher

The primary Python surface contains four task facades:

- `mltreelib.anomaly.AnomalyModel`
- `mltreelib.cluster.ClusterModel`
- `mltreelib.rules.RuleModel`
- `mltreelib.components.ComponentModel`

Each facade owns one task's method registry, result type, controls, and validation. A universal dispatcher is rejected because it would hide meaningful differences between scoring rows, assigning labels, mining rules, and transforming components.

### Decision 2: Existing Tree Names Are Compatibility Facades

`AnomalyTree` and `ClusterTree` remain available. They delegate to stable task facades and retain current engine names during migration. They do not become a second method registry.

### Decision 3: Package-Native Fit Lifecycle

The stable lifecycle uses `fit` for learning, `predict` or `score` for task output, `report` for diagnostics, and persistence methods for storage. `transform` remains valid for stateful transformations. `score_samples` remains valid for anomaly scoring.

No `BaseEstimator`, cloning, `get_params`, `set_params`, `deep`, or generic sklearn `score` contract is introduced.

### Decision 4: Configuration Inspection

`config()` returns the effective configuration as a plain dictionary. Direct attribute access remains available for common settings. Typed controls remain the source of truth.

This is inspection, not a scikit-learn compatibility promise. It has no deep recursion and no clone semantics.

### Decision 5: One Shared Report Model

Rust core owns a versioned report model with four required sections:

- Objective.
- Learner.
- Controller.
- Diagnostics.

Optional inference is a fifth section and appears only when valid.

Each section has one structured representation. Text, R/statsmodels-style, and Markdown renderers consume structured values.

### Decision 6: Capability-Gated Inference

Standard errors, confidence intervals, test statistics, and p-values require a defensible estimator-specific inference path. Otherwise the report provides descriptive diagnostics, bootstrap intervals when justified, or an explicit inference-unavailable warning.

No generic fake p-values are permitted.

### Decision 7: Stable and Experimental Namespaces

Stable methods live in primary task namespaces. Experimental methods live under `mltreelib.experimental.anomaly`, `mltreelib.experimental.cluster`, `mltreelib.experimental.rules`, and `mltreelib.experimental.components`.

Experimental methods still expose task-specific fit and report contracts. Their numerical guarantees and API stability remain lower.

### Decision 8: Consolidated Component Capability

Rust core owns PCA, truncated SVD, whitening, FastICA, NMF, and their adapters using existing `faer` and `nalgebra` dependencies.

Rust crates such as `linfa-ica`, `linfa-reduction`, and `efficient_pca` serve as benchmark references. They do not become runtime dependencies.

### Decision 9: Component Output Semantics

Component models expose components, loadings, scores, explained variance where defined, reconstruction, fitted scales, and report.

PCA reconstruction error and NMF reconstruction error can produce anomaly scores. Component scores can become cluster features. Every transform uses training-fitted state.

### Decision 10: Extended Solver Catalog

The shared solver layer adds:

- IRLS, already present.
- L-BFGS, already present.
- Full-memory BFGS.
- Nonlinear conjugate gradient.
- Nelder-Mead.
- Powell.
- CCD, already present.
- CMA-ES, SHADE, sNES, and GA, already present.

All methods consume one GLM objective convention and one convergence/result contract. Advanced SLSQP, trust-region, and generic constrained solvers remain deferred.

### Decision 11: Remove Scikit-Learn Parameter Compatibility

Remove public `get_params` from the Python model facade and replace it with `config()`. Do not add `set_params`. Remove sklearn-compatible wording from public docs and package guidance.

The private control normalization helper may retain an internal dictionary conversion under a non-public name.

Domain-specific `score_samples` and Rust anomaly scoring remain.

### Decision 12: Rust-Owned Persistence

Every facade persists through one versioned Rust document format. The document contains effective controls, fitted state, and report metadata required for exact restoration.

### Decision 13: Component and Report Relationship to Other Specifications

- Spec 21 owns general bounded optimization and solver mechanics.
- Spec 22 owns clustering algorithms, distances, and clustering evaluation.
- Spec 23 owns the classical leaf-family catalog.
- Spec 24 owns association mining and Context Tree integration.
- This specification owns cross-capability facades, common reports, component analysis, and estimator-compatibility policy.

### Decision 14: Dependency Policy

No new runtime dependency is required for the first component and solver wave. Reference implementations and benchmark-only dependencies may be installed explicitly in benchmark commands.

## Testing Decisions

- **Facade seam:** Every stable method is fitted through its task facade. Verify method selection, validation, prediction or scoring, report, and persistence.
- **Compatibility seam:** Verify `get_params` and `set_params` are absent. Verify `config()` and direct attribute access work.
- **Report seam:** Every stable facade returns all four required report sections. Verify inference appears only for valid families.
- **Renderer seam:** Text and Markdown renderers consume the same structured values and preserve numeric provenance.
- **Component seam:** Compare PCA, SVD, whitening, FastICA, and NMF against fixed expected values and external reference implementations in benchmark-only tests.
- **Component anomaly seam:** Verify PCA and NMF reconstruction scores against deterministic low-rank and additive synthetic data.
- **Component transform seam:** Verify fit-on-train and transform-on-test semantics, including unseen and missing values.
- **Solver parity seam:** Compare BFGS, conjugate gradient, Nelder-Mead, and Powell on deterministic quadratic, Rosenbrock, and nonsmooth objectives under matched budgets.
- **Solver report seam:** Verify iteration count, evaluation count, final objective, and distinct stop reasons.
- **Existing-model seam:** Existing standalone GLM, leaf, anomaly, clustering, and rule tests remain valid after facade migration.
- **Snapshot seam:** Every facade round-trips through JSON, bytes, file, and pickle with identical predictions, scores, and report values.
- **Agent-contract seam:** Documentation examples show one task facade and one `kind` switch rather than separate algorithm imports.
- **Prior art:** Reuse patterns from existing Python API tests, Rust solver tests, snapshot round-trip tests, report rendering tests, and clustering benchmark harnesses.

## Out of Scope

- A universal `fit(task=..., method=...)` dispatcher.
- Scikit-learn `BaseEstimator`, cloning, `get_params`, `set_params`, or generic `score` compatibility.
- GPU implementations for component analysis or solvers.
- Neural component analysis and autoencoder leaves.
- Full trust-region, SLSQP, COBYLA, and generic constrained optimization in the first wave.
- Kernel and one-class SVM leaves before a linear SVM foundation exists.
- Replacing task-specific validation with generic schema validation.
- Removing `fit`, `transform`, `predict`, `score_samples`, or domain-specific `score` where those verbs remain semantically correct.
- Migrating every internal Rust module in one change.

## Further Notes

- This specification separates common model UX from algorithm-specific implementation. Specs 21–24 remain authoritative for their algorithms.
- The existing `GlmSummary` and `RegressionSummary` become model-specific renderers under the common report model. Their coefficient tables remain useful but do not define the report architecture.
- The existing `ModelTree.summary()` and `leaf_summary()` remain tree-specific views. They can later consume common report sections without changing user-visible behavior.
- The four facades keep agents from needing algorithm-specific import knowledge for ordinary work.
- `config()` preserves inspection without importing scikit-learn's estimator protocol.
- Component analysis and extended solvers were researched in `references/research/2026-11-21-component-analysis-and-unified-model-reports.md`.
