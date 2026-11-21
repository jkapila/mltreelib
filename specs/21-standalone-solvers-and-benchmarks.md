# Spec 21: General bounded optimizer, production CMA-ES, standalone model fitting, and realistic benchmarking

Status: ready-for-agent

## Problem Statement

`mltreelib` already contains valuable numerical machinery for standalone weighted linear and generalized linear models: exact normal-equation fitting, IRLS, coordinate descent, L-BFGS, SHADE, CMA-ES, sNES, and a real-coded genetic algorithm. These solvers also power model-tree leaves.

The current solver surface remains narrower than the package thesis requires:

1. **Solvers are tied to GLM coefficients.** Evolutionary code accepts a `DesignMatrix`, but its fitness function is hard-wired to GLM deviance, L1/L2 penalties, and coefficient-sign projection. Users cannot optimize a tree split threshold, model hyperparameter, anomaly cutoff, clustering objective, or other bounded research objective through a public general optimizer.
2. **Standalone fitting and leaf fitting do not share one dispatch.** They reuse lower numerical kernels, but duplicate option construction, defaults, validation, and failure behavior. Standalone fitting raises errors when a solver fails. Leaf fitting can fall back to a mean leaf and return a missing intercept. This makes solver behavior inconsistent across two public pathways.
3. **Existing CMA-ES is partial and contains correctness defects.** It uses standard population sizing, recombination weights, rank-mu covariance learning, and path updates, but its covariance-scale path is not whitened correctly. Monotone projection mutates evaluated candidates but not their corresponding step vectors. Initial step size is fixed, stopping rules are weak, factorization failure silently falls back to identity covariance, and there are no bounds, ask/tell controls, restart behavior, or benchmark oracle.
4. **The solver portfolio is fragmented.** Users who want random search, Latin hypercube sampling, PSO, DE, L-BFGS-B, NSGA-II, islands, or other general methods cannot access them through a consistent interface. Twenty proposed algorithms are partly conceptual, partly implemented under different names, and partly absent. This creates duplicated control flow and misleading expectations.
5. **Benchmarks cover too little of the realistic use case.** Existing suites compare selected leaf solvers on a few regressions and one real dataset. They do not consistently compare convergence quality, evaluation efficiency, held-out predictive quality, constraint satisfaction, reproducibility, or multiobjective behavior across analytic, synthetic, statistical, and tree-research objectives.
6. **Optimization vocabulary and result semantics are inconsistent.** Core and Python defaults differ. Solver aliases vary. Termination reasons collapse distinct conditions. Evaluation counts are absent for population methods. Bounds, invalid objective values, and budget exhaustion lack one shared contract.

`mltreelib` needs optimization as a core research capability. It does not need a full-scale optimizer platform. The target is a small, composable interface with good-enough methods for tree research, model fitting, experiments, and reproducible teaching.

## Solution

Extend Spec 21 into one coherent optimizer product with three layers:

1. **A general bounded optimization module.** One Rust entry point accepts a scalar objective, bounds, starting point, method, shared budget, seed, and method-specific options. One Python entry point accepts a callable scalar objective and returns one typed result. A separate multiobjective entry point accepts an objective vector and returns a Pareto result.
2. **A model-specific adapter.** Weighted Gaussian and GLM fitting becomes an objective adapter over the same optimizer dispatch. Existing exact and specialized paths remain available where they are mathematically superior. Leaf fitting uses shared validation and failure semantics without losing established NA policies.
3. **A realistic benchmark program.** One reproducible benchmark suite compares optimizers under equal evaluation budgets across analytic regression, statistical model families, constrained and multimodal objectives, and tree-research objectives. Results include quality, evaluation efficiency, held-out error, runtime, status, and reproducibility.

The initial public method portfolio is deliberately bounded:

- Existing methods exposed through the general interface: direct normal equations, IRLS, CCD/proximal CCD, L-BFGS, SHADE, CMA-ES, sNES, and real-coded GA.
- New first-wave methods: random search, Latin hypercube sampling, PSO, DE/rand/1/bin, and L-BFGS-B.
- Production CMA-ES upgrades: ask/tell-compatible state machine, correct covariance-scale path, bounds, explicit initial step size, meaningful termination statuses, evaluation budget, and IPOP restart.
- Second-wave research methods: NSGA-II and a serial island orchestrator.
- Evidence-driven only: jDE as a separate public method, MOEA/D, nonlinear conjugate gradient, trust-region methods, SGD momentum, and AdamW.

The general optimizer must remain a deep module. Callers should learn one small interface, while algorithm-specific control flow, bounds handling, population state, and convergence logic remain behind it.

## User Stories

### General Optimization Interface

1. As a quantitative researcher, I want to minimize an arbitrary bounded scalar objective through one public Rust function, so that I can optimize research quantities that are not GLM coefficients.
2. As a Python user, I want to pass a callable objective to one public optimization function, so that I can optimize tree thresholds, hyperparameters, and custom losses without writing Rust.
3. As a Python user, I want the Python objective to receive a one-dimensional numeric vector and return one finite scalar, so that the calling contract stays simple.
4. As a Python user, I want a separate multiobjective function for objectives returning vectors, so that scalar and vector results never have ambiguous types.
5. As an API user, I want one canonical method name for every optimizer, so that aliases do not fragment the public interface.
6. As an API user, I want method-specific options grouped under their method, so that common controls stay distinct from algorithm controls.
7. As an API user, I want to provide lower and upper bounds for every parameter, so that optimization respects natural parameter domains.
8. As an API user, I want unbounded parameters to work when the selected method supports them, so that I am not forced to invent artificial finite bounds.
9. As an API user, I want invalid bounds rejected before optimization starts, so that failures identify configuration errors instead of numerical instability.
10. As an API user, I want the returned solution guaranteed to be finite and within supported bounds, so that downstream model code can rely on it.
11. As an API user, I want a starting point accepted by every method, so that I can warm-start a search from a known solution.
12. As an API user, I want initial population samples accepted by population methods, so that I can reproduce or seed research experiments.
13. As an API user, I want a maximum evaluation budget, so that expensive objectives cannot run without a user-defined limit.
14. As an API user, I want a maximum iteration budget, so that algorithms with meaningful generations can stop even when evaluations remain.
15. As an API user, I want a tolerance, so that successful convergence has a reproducible stopping rule.
16. As an API user, I want an explicit seed, so that stochastic runs reproduce exactly.
17. As an API user, I want one documented default seed policy for every stochastic method, so that omitted seeds remain reproducible.
18. As an API user, I want the result to report the best point, best objective, iterations, evaluations, and termination status, so that I can judge search quality and cost.
19. As an API user, I want convergence, budget exhaustion, invalid objective value, numerical failure, infeasible bounds, and unsupported method to remain distinct statuses, so that results never imply success after failure.
20. As an API user, I want optional convergence history, so that I can diagnose stagnation and compare methods.
21. As an API user, I want invalid objective values rejected according to one documented rule, so that NaN or infinite losses cannot silently distort population updates.
22. As an API user, I want a deterministic error when a method cannot support the requested bounds or objective type, so that the library never silently substitutes another algorithm.
23. As a library maintainer, I want shared objective and bounds abstractions, so that new algorithms reuse one validation and result contract.
24. As a library maintainer, I want shared population, evaluation-budget, and convergence machinery, so that each algorithm contains only algorithm-specific behavior.
25. As a library maintainer, I want existing GLM solvers adapted to the general objective seam, so that public standalone fitting and leaf fitting do not maintain separate dispatch logic.
26. As an existing user, I want current standalone model fitting and leaf solver names to keep working, so that adopting the general optimizer does not break existing code.

### Random Search and Latin Hypercube Sampling

27. As a benchmark author, I want random search as a baseline, so that every advanced method can be compared with an honest uninformed baseline.
28. As a benchmark author, I want Latin hypercube sampling, so that I can test whether stratified initialization improves population methods.
29. As a population-method user, I want Latin hypercube samples available as initial populations, so that CMA-ES, DE, SHADE, PSO, and GA start from controlled coverage.
30. As a user, I want Latin hypercube samples reproducible for a seed, so that initialization comparisons are fair.
31. As a user, I want each Latin hypercube dimension stratified independently, so that the sampler covers every marginal range.
32. As a user, I want Latin hypercube samples shuffled within strata, so that samples do not share coordinate ordering artifacts.
33. As a user, I want random search to respect bounds, so that every returned candidate is valid.

### Particle Swarm Optimization

34. As a researcher, I want PSO for continuous black-box objectives, so that I have a simple, strong population optimizer.
35. As a researcher, I want inertia, cognitive, and social coefficients configurable, so that I can study exploration and exploitation.
36. As a researcher, I want velocity clamping or bound repair configurable, so that particles remain feasible.
37. As a researcher, I want global-best and neighborhood topologies supported, so that I can compare global and local information flow.
38. As a researcher, I want the global incumbent preserved across generations, so that a later distribution shift does not erase the best solution.
39. As a researcher, I want PSO to report evaluation count and generation count, so that its cost can be compared fairly with other population methods.
40. As a benchmark author, I want PSO compared on rotated, multimodal, and noisy objectives, so that its strengths and weaknesses remain visible.

### Differential Evolution

41. As a researcher, I want plain DE/rand/1/bin available separately from SHADE, so that I can use a transparent baseline and test whether adaptation helps.
42. As a researcher, I want the DE mutation factor and crossover probability configurable, so that I can reproduce standard DE experiments.
43. As a researcher, I want DE to support an external archive, so that I can compare it with SHADE-like search behavior.
44. As a researcher, I want DE to respect bounds through a documented repair rule, so that feasible solutions are produced consistently.
45. As a benchmark author, I want DE tested on multimodal and noisy objectives, so that population diversity is measured rather than assumed.
46. As a Python user, I want `de` to mean plain DE, so that method names describe actual behavior.

### Production CMA-ES

47. As a researcher, I want CMA-ES to implement the standard weighted recombination and rank-mu covariance learning correctly, so that it can solve correlated and ill-conditioned landscapes.
48. As a researcher, I want the covariance-scale path whitened correctly, so that step-size control remains valid after covariance adaptation.
49. As a researcher, I want an explicit initial standard deviation, so that search scale matches the objective and bounds.
50. As a researcher, I want an explicit initial mean or starting point, so that CMA-ES can be initialized reproducibly.
51. As a researcher, I want CMA-ES population size configurable, so that evaluation budget and search behavior can be controlled.
52. As a researcher, I want CMA-ES to expose meaningful termination statuses for step size, coordinate change, objective stagnation, function spread, and budget exhaustion, so that convergence claims are meaningful.
53. As a researcher, I want CMA-ES to repair or reject infeasible samples consistently before evaluation, so that sampled points and adaptation vectors describe the same candidate.
54. As a researcher, I want covariance factorization failure reported explicitly, so that the method never silently becomes isotropic.
55. As a researcher, I want optional IPOP restart after stagnation, so that multimodal problems can use larger populations without replacing the method.
56. As a researcher, I want the global best incumbent preserved across restarts, so that restarting cannot worsen the returned result.
57. As a researcher, I want IPOP population and restart counts reported, so that restart overhead is measurable.
58. As a Python user, I want an ask/tell-compatible CMA-ES interface, so that I can build custom evaluation loops like the hardmaru estool pattern.
59. As a Python user, I want ask to return a population and tell to accept objective values for that same population, so that external batch evaluation remains possible.
60. As a Python user, I want CMA-ES result metadata to expose best point, best objective, current mean, coordinate standard deviations, evaluations, iterations, and stop conditions, so that I can inspect search behavior.
61. As a Python user, I want the convenience minimize function and the ask/tell interface to share one implementation, so that behavior cannot diverge.
62. As a benchmark author, I want CMA-ES compared against pycma on identical objectives, seeds, bounds, initial mean, step size, and population size, so that the Rust implementation has an independent oracle.
63. As a benchmark author, I want CMA-ES tested on sphere, rotated ellipsoid, Rosenbrock, Rastrigin, and noisy sphere, so that correctness and scaling behavior are covered.
64. As a benchmark author, I want invariance under rotation and coordinate scaling tested, so that the covariance-scale correction is protected by regression tests.
65. As a tree user, I want full CMA-ES available as a leaf solver, so that difficult non-convex leaf losses can use the corrected implementation.
66. As a tree user, I want existing CMA-ES leaf behavior to remain compatible unless a documented numerical fix changes results, so that existing models do not fail unexpectedly.

### L-BFGS-B and Smooth Optimization

67. As a researcher, I want L-BFGS-B for smooth bounded objectives, so that box constraints do not force penalty-function approximations.
68. As a researcher, I want L-BFGS-B to retain limited-memory curvature updates, so that large parameter vectors remain practical.
69. As a researcher, I want the projected-gradient step and bounds enforcement documented, so that constrained solutions remain feasible.
70. As a researcher, I want L-BFGS and L-BFGS-B to share objective and convergence machinery, so that their differences remain explicit.
71. As a model user, I want bounded leaf coefficients where the leaf family supports them, so that regularization and domain constraints can coexist.

### SHADE, sNES, and Real-Coded GA

72. As a researcher, I want SHADE exposed through the general optimizer, so that its adaptive DE behavior can be used outside GLM leaves.
73. As a researcher, I want sNES exposed through the general optimizer, so that lightweight natural-gradient evolution remains available.
74. As a researcher, I want the real-coded GA exposed through the general optimizer, so that mixed and discrete encodings can be optimized.
75. As a researcher, I want GA selection, crossover, mutation, and elitism configurable, so that experiments can reproduce published setups.
76. As a researcher, I want the GA to consume decoded candidate vectors through encoding adapters, so that continuous, integer, binary, categorical, and tree-structure encodings share one optimizer.
77. As a researcher, I want historical elite behavior configurable, so that experiments can compare elitist and forgetful variants.
78. As a researcher, I want weight decay or regularization applied through one documented objective convention, so that CMA-ES, DE, SHADE, PSO, and GA do not silently penalize different quantities.

### Multiobjective and Islands

79. As a model-selection researcher, I want NSGA-II for two or three conflicting objectives, so that I can inspect accuracy, complexity, calibration, and latency trade-offs.
80. As a model-selection researcher, I want nondominated sorting and crowding-based survival, so that the Pareto set remains diverse.
81. As a model-selection researcher, I want the Pareto set, crowding metrics, and evaluation count returned, so that front quality and cost can be measured.
82. As a researcher, I want NSGA-II to support constrained objectives, so that feasible solutions can be preferred without hiding constraint semantics inside penalties.
83. As a long-search researcher, I want a serial island orchestrator, so that several populations can evolve independently and migrate elites.
84. As a long-search researcher, I want ring and complete migration topologies, so that migration behavior can be studied.
85. As a long-search researcher, I want migration interval, elite count, per-island method, and per-island seed configurable, so that experiments remain reproducible.
86. As a long-search researcher, I want one global incumbent archive across islands, so that migration cannot lose the best result.
87. As a package user, I want asynchronous islands deferred until scheduler behavior is safe, so that package jobs do not oversubscribe multicore resources.

### Standalone Model Fitting and Shared Semantics

88. As a model user, I want weighted Gaussian and GLM fitting through the public standalone fitting interface, so that I can fit models without constructing a tree.
89. As a model user, I want the standalone fitting result to expose coefficients, intercept, predictions, objective, convergence status, and inference statistics where available, so that I can inspect model quality.
90. As a model user, I want Gaussian direct fitting to remain the analytical default, so that the general optimizer does not replace an exact solution with an iterative approximation.
91. As a model user, I want CCD to remain the default for Gaussian L1 and sign-constrained fitting, so that exact sparsity remains available.
92. As a model user, I want L-BFGS to remain the default for supported smooth non-Gaussian families, so that analytic gradients remain efficient.
93. As a model user, I want one explicit family-by-solver compatibility table, so that unsupported combinations fail before fitting.
94. As a model user, I want L1 semantics and coefficient scaling documented per solver, so that penalties compare correctly.
95. As a model user, I want monotone coefficient constraints supported wherever the current solver already supports them, so that behavior does not regress.
96. As a model user, I want standalone and single-leaf fits with the same design and penalty to agree, so that tree leaves remain trustworthy numerical estimators.
97. As a model user, I want explicit solver failures to propagate consistently, so that standalone fitting and leaf fitting do not hide numerical failure differently.
98. As a model user, I want existing leaf missing-data policies preserved, so that general solver work does not alter imputation and NA-indicator behavior.
99. As a library maintainer, I want one source of truth for solver defaults, so that core, standalone Python, and leaf pathways no longer drift.
100. As a library maintainer, I want inference statistics returned only from analytically supported paths, so that iterative or evolutionary fits never fabricate standard errors.

### Benchmarks and Reproducibility

101. As a benchmark author, I want a single solver benchmark entry point, so that users do not encounter duplicate benchmark scripts.
102. As a benchmark author, I want all methods compared under equal evaluation budgets, so that faster convergence is measured fairly.
103. As a benchmark author, I want every stochastic method run across multiple fixed seeds, so that one lucky run cannot define quality.
104. As a benchmark author, I want median and worst-seed results reported, so that reliability remains visible.
105. As a benchmark author, I want evaluations-to-target reported, so that methods can be compared before final runtime.
106. As a benchmark author, I want wall time, peak memory, and evaluations reported separately, so that no single cost metric hides another.
107. As a benchmark author, I want objective-gap-to-oracle reported for analytic and synthetic problems, so that near-optimal and failed runs are distinguishable.
108. As a benchmark author, I want held-out RMSE, deviance, log loss, or task metric reported for statistical objectives, so that low training loss cannot hide overfitting.
109. As a benchmark author, I want benchmark input data and split seeds fixed, so that method comparisons remain reproducible.
110. As a benchmark author, I want external references used as benchmark-only dependencies, so that core installation remains lean.
111. As a benchmark author, I want optional pycma comparisons for CMA-ES, so that Rust results have an independent reference.
112. As a benchmark author, I want optional SciPy and scikit-learn comparisons for smooth and statistical objectives, so that numerical quality has external anchors.
113. As a benchmark author, I want statsmodels comparisons for GLM likelihoods, so that deviance and parameter estimates have external anchors.
114. As a benchmark author, I want the benchmark to cover weighted Gaussian regression, so that exact, iterative, and population methods share one realistic baseline.
115. As a benchmark author, I want the benchmark to cover ridge and elastic-net regression, so that L2 and L1 behavior remain visible.
116. As a benchmark author, I want the benchmark to cover logistic regression, so that bounded smooth and black-box methods operate on a real classification objective.
117. As a benchmark author, I want the benchmark to cover Poisson regression, so that count-family likelihoods remain verified.
118. As a benchmark author, I want the benchmark to cover Gamma and Tweedie regression, so that positive continuous targets remain supported.
119. As a benchmark author, I want the benchmark to cover Tobit regression, so that censored Gaussian behavior remains supported.
120. As a benchmark author, I want the benchmark to cover the real Concrete Compressive Strength dataset, so that solver quality is measured on real tabular structure.
121. As a benchmark author, I want the benchmark to cover diabetes and breast-cancer regression or classification data, so that results are comparable with common external baselines.
122. As a benchmark author, I want a model-tree leaf benchmark, so that solver improvements affect tree fitting rather than only standalone fits.
123. As a benchmark author, I want a tree-hyperparameter search benchmark, so that the general optimizer demonstrates a native package use case.
124. As a benchmark author, I want a split-threshold or split-rule search benchmark, so that optimization proves useful inside tree construction.
125. As a benchmark author, I want multiobjective tree selection on error, node count, and inference latency, so that NSGA-II has a package-native validation.
126. As a benchmark author, I want benchmark output written as machine-readable JSON plus a concise Markdown or plot summary, so that results can be compared and reviewed.
127. As a benchmark author, I want benchmark runs to record package version, platform, seed set, budgets, and options, so that historical results remain interpretable.
128. As a benchmark author, I want no multi-core optimizer runs to execute concurrently, so that timing comparisons remain meaningful.
129. As a benchmark author, I want benchmark thresholds based on external references and repeated seeds, so that regressions fail reliably without brittle single-run assertions.
130. As a benchmark author, I want optional heavy comparison dependencies isolated from normal test runs, so that the core suite remains fast and dependency-light.

## Implementation Decisions

### Decision 1: One General Optimization Seam

The Rust solver module exposes one scalar minimization entry point and one multiobjective entry point. Both accept shared problem metadata, bounds, budget, seed, and method-specific options. The interface returns a typed result containing best candidates, objective values, iterations, evaluations, status, and optional history.

The scalar objective seam reports dimension, bounds, and one finite objective value. A separate smooth-objective seam reports value and gradient. Smooth methods fail clearly when gradients are absent. Multiobjective methods use a separate objective-vector seam and return a Pareto result.

The seam is placed above current numerical kernels. Current GLM-specific fitness evaluation becomes an adapter rather than the foundation of the general optimizer.

### Decision 2: Shared Types and Validation

The module defines one bounds representation, one optimizer-method enumeration, one shared request structure, one result structure, and one termination-status enumeration.

Shared validation covers dimension agreement, finite bounds where required, lower-upper ordering, finite starting points, nonnegative budgets, nonnegative population sizes, and supported objective types.

Method-specific options live under their method. Common controls cover maximum evaluations, maximum iterations, tolerance, seed, initial population, and verbosity.

### Decision 3: Method Portfolio and Waves

Wave 1 exposes current methods and adds random search, LHS, PSO, DE/rand/1/bin, and L-BFGS-B. Current methods retain canonical names: `direct`, `irls`, `ccd`, `lbfgs`, `lbfgs_b`, `shade`, `cmaes`, `snes`, and `ga`.

Wave 2 adds NSGA-II and a serial island orchestrator. Wave 3 remains evidence-driven.

jDE does not become a separate public method unless benchmark evidence shows SHADE and DE cannot cover required per-individual adaptation. MOEA/D, nonlinear conjugate gradient, trust-region methods, SGD momentum, and AdamW remain out of scope.

### Decision 4: Common Population and Budget Machinery

Population methods share initialization, evaluation accounting, incumbent preservation, candidate repair, and termination machinery. Algorithm modules own only mutation, recombination, selection, and adaptation logic.

The evaluation budget counts every user objective evaluation, including warm starts, archive evaluation, and island migration. Iteration budget is separate and never substitutes for evaluation accounting.

### Decision 5: Public Python Interface

The Python solver package exposes:

- `optimize` for scalar callable objectives.
- `minimize_many` for vector callable objectives.
- `fit` for weighted Gaussian and GLM model fitting.
- Typed result objects exposing best point, best objective, evaluations, iterations, status, optional history, and Pareto data when applicable.

The scalar objective callback receives a one-dimensional numeric vector. The multiobjective callback receives the same vector shape and returns a numeric vector. Unsupported callback behavior raises an actionable error.

Existing `fit` and `fit_glm` names remain compatible. Existing leaf solver aliases remain accepted, but documentation lists one canonical name per method.

### Decision 6: Shared Model-Fitting Dispatch

Standalone fitting and leaf fitting use one compatibility and default table. The table defines supported families, penalties, monotone constraints, gradient availability, and canonical method names.

Gaussian direct fitting remains exact and preferred when compatible. CCD remains preferred for Gaussian L1 and sign constraints. L-BFGS remains preferred for supported smooth non-Gaussian families. Evolutionary methods remain explicit choices for non-convex or derivative-free objectives.

The shared dispatch removes duplicated option construction. It does not force all leaf solvers through the generic black-box path when an exact or specialized analytic path exists.

### Decision 7: Defaults and Failure Semantics

One documented defaults table governs general optimization, standalone fitting, and leaf fitting. Method-specific overrides remain available.

A solver failure must not silently become a different solver. Standalone fitting and leaf fitting must report the same numerical failure cause. If a leaf requires a documented fallback for an unsupported modeling condition, that fallback must be visible in diagnostics and must not occur for ordinary numerical failure.

Inference statistics remain available only for analytically supported direct Gaussian fitting. Iterative and evolutionary fits do not fabricate standard errors.

### Decision 8: Penalty and Constraint Semantics

One penalty convention applies to general optimizer objectives. Model adapters convert L1, L2, monotone, and family-specific constraints into that convention explicitly.

L1 scaling differences must be removed or documented per method. Monotone projection must happen before evaluation and must update the same candidate used by adaptation. Objective functions never receive silently imputed data.

### Decision 9: Production CMA-ES Behavior

CMA-ES becomes a native Rust ask/tell-capable implementation rather than a Python wrapper around pycma. The convenience minimization entry point and ask/tell interface share one state machine.

The implementation includes weighted recombination, rank-mu covariance learning, rank-one covariance learning, correct covariance-scale whitening, explicit initial mean, explicit initial standard deviation, configurable population size, box-bound repair, objective evaluation budget, meaningful termination statuses, and optional IPOP restart.

The covariance-scale path uses a mathematically consistent whitened step vector. Candidate repair occurs before evaluation and adaptation. Failed covariance factorization returns an explicit numerical error instead of silently substituting identity covariance.

IPOP restart increases population size after a documented stagnation condition and preserves the global incumbent across restarts. Restart count, population sizes, and stop conditions appear in result metadata.

The implementation follows the interface pattern demonstrated by hardmaru estool: ask returns candidates, tell accepts their objective values, and result exposes best candidate, best objective, current mean, coordinate standard deviations, evaluations, iterations, and termination state. estool itself wraps pycma, so it is an interface reference rather than the numerical oracle. pycma is the independent numerical comparison oracle.

### Decision 10: Reproducibility

Every stochastic method accepts an explicit seed and produces identical candidates and results for identical inputs, bounds, options, and single-thread execution. Determinism guarantees are documented per method.

Floating-point comparisons in benchmarks use tolerances appropriate to numerical method and platform. Benchmark outputs never rely on exact cross-platform bit equality except for deterministic Rust-only paths explicitly designed for it.

### Decision 11: One Benchmark Suite

The repository keeps one canonical solver benchmark program. Existing duplicate top-level and suite benchmark copies converge into one maintained entry point with a stable documented command.

The benchmark includes:

1. Analytic regression: weighted Gaussian OLS and ridge, with exact normal-equation oracle.
2. Sparse regression: Lasso and elastic net, with SciPy or scikit-learn reference when available.
3. Statistical likelihoods: logistic, Poisson, Gamma, Tweedie, and Tobit, with statsmodels or equivalent references when available.
4. Real tabular data: Concrete Compressive Strength, diabetes, and breast cancer.
5. Synthetic optimization functions: sphere, rotated ellipsoid, Rosenbrock, Rastrigin, and noisy sphere.
6. Native package use cases: model-tree leaves, tree-hyperparameter search, split-threshold search, and multiobjective tree selection.
7. Independent CMA-ES comparison against pycma on matching problems.

Every benchmark records method, objective, dataset, seed, budget, evaluations, iterations, wall time, peak memory when available, best objective, objective gap, held-out metric, constraint violations, and termination status.

### Decision 12: Realistic Benchmark Thresholds

Thresholds are based on repeated-seed medians and external oracles. They must detect material regressions without depending on one random run.

Required standards:

- Exact Gaussian path matches the analytical solution within tight numerical tolerance.
- Direct, L-BFGS, and L-BFGS-B reach equivalent objective quality on well-conditioned ridge regression within the shared budget.
- CCD matches sparse-reference objective and support on fixed sparse synthetic data within documented tolerance.
- Supported GLM families remain within documented likelihood or deviance tolerance against reference implementations.
- CMA-ES reaches target quality on sphere, rotated ellipsoid, and Rosenbrock across the seed set and does not use materially more evaluations than pycma under matched settings.
- CMA-ES rotated-ellipsoid performance remains comparable to the unrotated equivalent, protecting the covariance-scale correction.
- Population methods improve or match random search on seeded multimodal tasks within a documented evaluation budget.
- Determinism tests reproduce candidate populations, best values, and result metadata for fixed seeds.

The benchmark produces machine-readable JSON and a concise Markdown or plot summary. It records package version, platform, seed set, budgets, and method options.

### Decision 13: Documentation and Examples

Public documentation explains the general optimizer, standalone fitting, solver compatibility table, defaults table, result statuses, deterministic behavior, and benchmark command.

Examples include:

- Bounded sphere minimization with random search and LHS.
- PSO and DE on a rotated ellipsoid.
- CMA-ES ask/tell custom loop.
- CMA-ES with IPOP restart on a multimodal objective.
- L-BFGS-B on a bounded smooth objective.
- NSGA-II on error-versus-node-count-versus-latency.
- Standalone weighted Gaussian, logistic, and Poisson fitting.
- One tree-hyperparameter search using the public optimizer.

### Decision 14: Dependencies

Core optimization remains dependency-light and uses existing Rust numerical dependencies where possible. Heavy comparison dependencies belong only to benchmark extras. pycma, SciPy, scikit-learn, and statsmodels never become runtime requirements of `mltreelib`.

## Testing Decisions

- **Good tests assert external behavior.** Tests cross the highest available seam. They verify returned objective quality, feasibility, status, counts, reproducibility, and public model predictions. They do not assert private update formulas, internal vector layout, or mutation order unless a numerical regression specifically requires a mathematical invariant.
- **Primary Rust seam:** call the general scalar optimization entry point with deterministic toy objectives. Assert best values, feasibility, evaluation budgets, status, and determinism.
- **Primary Python seam:** call `optimize` with a callable objective. Assert result types, result fields, exception behavior, and deterministic repeatability.
- **Model-fitting seam:** compare standalone fitting and single-leaf fitting on the same weighted design, family, penalty, and method. Assert coefficients, predictions, and objective agree within solver tolerance.
- **Compatibility seam:** table-driven tests cover family, method, L1, L2, and monotone combinations. Unsupported combinations fail before objective evaluation.
- **Defaults seam:** assert one documented defaults table drives general, standalone, and leaf paths.
- **Failure seam:** invalid bounds, wrong dimensions, invalid objective values, exhausted budgets, and numerical failures return distinct documented statuses or errors. No test accepts silent fallback to another solver.
- **Bounds seam:** every supported method returns feasible points for finite and partially unbounded domains.
- **Budget seam:** objective call counts never exceed the configured evaluation budget.
- **CMA-ES correctness seam:** use sphere, rotated ellipsoid, Rosenbrock, Rastrigin, and noisy sphere over multiple seeds. Assert target quality, rotation/scaling behavior, incumbent preservation, and parity class against pycma when the benchmark extra is installed.
- **CMA-ES regression seam:** retain a test that would fail under the current non-whitened covariance-scale path. Test monotone repair consistency and explicit covariance-factorization failure.
- **CMA-ES ask/tell seam:** assert ask population size, tell objective ordering, rejection of mismatched populations, and state continuity across repeated ask/tell cycles.
- **IPOP seam:** force stagnation and assert population growth, restart count, incumbent preservation, and explicit restart metadata.
- **Random/LHS seam:** assert bounds, marginal stratification, deterministic samples, and distinct shuffled strata.
- **PSO seam:** assert bounds, global incumbent preservation, topology behavior, and deterministic populations.
- **DE seam:** assert DE and SHADE remain distinct methods, archive and crossover settings take effect, and deterministic runs reproduce.
- **CCD/proximal seam:** compare sparse coefficients and objective against a fixed reference. Assert exact zeros under the contract already promised by CCD.
- **L-BFGS-B seam:** assert feasibility, smooth-objective convergence, and consistency with unbounded L-BFGS when bounds do not bind.
- **NSGA-II seam:** assert returned points are nondominated on known toy objectives, crowding behavior is diverse, and evaluation counts remain accurate.
- **Island seam:** assert migration topology, elite counts, per-island determinism, and global incumbent preservation in serial mode.
- **Standalone statistical seams:** weighted Gaussian, logistic, Poisson, Gamma, Tweedie, and Tobit reference comparisons follow the existing solver-test pattern while using documented external or fixed-reference expectations.
- **Benchmark smoke test:** run a tiny budget through every method on one deterministic objective and one small dataset. Verify output schema, counts, statuses, and artifact creation without running the full benchmark in routine tests.
- **Full benchmark:** run separately under a benchmark command. Store JSON and Markdown results outside routine unit tests. Benchmark thresholds use repeated seeds and external references rather than one brittle sample.
- **Prior art:** reuse patterns from existing standalone solver tests, solver benchmark scripts, model-tree leaf tests, deterministic RNG tests, and weighted GLM reference comparisons already present in the repository.

## Out of Scope

- Building a SciPy, never-ending optimizer, or commercial solver replacement.
- Guaranteeing global optimality for heuristic population methods.
- Distributed, MPI, GPU, or asynchronous island execution.
- Running multiple multi-core optimizer fits concurrently.
- MIP, SAT, conic, trust-region, nonlinear conjugate-gradient, SGD momentum, and AdamW implementations.
- Deep neural-network optimization and autodiff integration.
- Plugin discovery or third-party solver installation.
- Automatic hyperparameter-optimization products beyond providing optimizer primitives.
- Replacing exact Gaussian fitting with iterative optimization.
- Changing tree split formulas, growth controls, or leaf families beyond solver support needed by this specification.
- Supporting inference statistics for iterative or evolutionary fits.
- Claiming bit-identical results across different CPUs or mathematical libraries for floating-point optimizer kernels.

## Further Notes

- This specification expands and adapts existing Spec 21. It retains standalone weighted Gaussian and GLM fitting, shared solver dispatch, and benchmark criteria while adding the general bounded optimizer and production CMA-ES scope.
- Highest-seam implementation plan:
  1. Establish shared objective, bounds, budget, status, and result contracts.
  2. Correct and productionize CMA-ES.
  3. Adapt existing GLM and leaf dispatch to shared semantics.
  4. Add random search, LHS, PSO, DE, and L-BFGS-B.
  5. Consolidate and expand realistic benchmarks.
  6. Add NSGA-II and serial islands.
- The current CMA-ES implementation must be treated as partial. Its covariance-scale whitening defect, projection inconsistency, silent identity fallback, fixed initial step size, and weak stopping rules require correction before claiming full CMA-ES behavior.
- hardmaru estool is a useful interface reference because its CMA-ES wrapper exposes ask/tell, current mean, best candidate, coordinate standard deviations, and result metadata. Its CMA-ES mathematics comes from pycma. pycma remains the independent numerical oracle.
- Current existing research notes provide broader algorithm context. Current source wins where those notes conflict.
