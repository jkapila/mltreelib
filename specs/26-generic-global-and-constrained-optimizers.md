# Spec 26: Generic Global and Constrained Optimizers

Status: proposed

## Problem Statement

`mltreelib` has strong numerical kernels for statistical model fitting, but no public way to optimize an arbitrary bounded scalar objective.

Current solvers in `crates/mltreelib-core/src/solver/` are bound to GLM coefficients. `normal.rs` solves Gaussian OLS and Ridge. `ccd.rs` solves Elastic Net and monotone constraints. `lbfgs.rs` minimizes GLM deviance. `dense.rs` fits weighted dense Gaussian and GLM designs through the same kernels. `evolutionary.rs` supplies CMA-ES, SHADE, sNES, and a real-coded GA, but its fitness function is hard-wired to GLM deviance, L1 and L2 penalties, and coefficient-sign projection.

Four consequences follow.

1. **No generic objective seam exists.** There is no `Objective` trait, no `Bounds` type, no `OptimizerKind` enumeration, and no shared `OptimizationRequest` or `OptimizationResult`. Every new public optimizer would need its own contract.
2. **No shared budget or status semantics exist.** `StopReason` in `solver/mod.rs` has four values and no evaluation count. Population methods cannot report cost honestly. Nothing distinguishes convergence from budget exhaustion.
3. **No constrained solver exists.** `SolverOptions.monotone` supports coefficient sign projection inside CCD and the evolutionary kernels. No method accepts general nonlinear constraints. Bounds are absent from every solver option.
4. **No global trajectory method exists.** CMA-ES, SHADE, sNES, and the GA are population methods. No single-point global method with a cooling schedule or a perturbation-and-local-solve loop exists.

Three requested capabilities map onto this gap.

- **Bounded simulated annealing** provides cheap derivative-free global search for non-smooth objectives.
- **Basin hopping** provides global search for rugged landscapes with separated local minima, by composing random perturbation, a local minimizer, and Metropolis acceptance.
- **One nonlinear constraint solver** completes the portfolio. SciPy exposes COBYLA, COBYQA, SLSQP, and `trust-constr`. `mltreelib` must choose one, not all four.

The implementation order matters. Basin hopping requires a trustworthy bounded local minimizer. Any constrained solver requires constraint semantics before it can be trusted. A global method without a random-search baseline cannot be shown to add value.

Research basis: `references/research/2026-11-21-generic-global-and-constrained-solvers.md` and `references/research/2026-11-21-general-solver-api-roadmap.md`.

## Solution

Build one generic bounded optimization module and add seven methods in dependency order.

| Order | Method | Status | Role |
|---:|---|---|---|
| 0 | Shared optimization seam | New foundation | Contracts for every method below |
| 1 | **Leaf solver decoupling** | New architecture | Leaf problem and solver become independent axes |
| 2 | Random Search and Latin Hypercube | New baseline | Uninformed reference for every global method |
| 3 | L-BFGS-B | New bounded local solver | Smooth bounded local method |
| 4 | **BFGS** and **nonlinear CG** | New leaf-level competitors | Direct competitors against existing L-BFGS leaf path |
| 5 | **Nelder-Mead** and **Powell** | New derivative-free local solvers | Black-box local methods and basin-hopping inner solvers |
| 6 | Bounded simulated annealing | New global trajectory method | Derivative-free global search |
| 7 | COBYLA | New nonlinear constraint solver | One nonlinear constraint method |
| 8 | Basin hopping | New global composition method | Perturbation plus local solve plus Metropolis acceptance |

BFGS and nonlinear conjugate gradient come from the SciPy and statsmodels competitor audit. They compete directly against the existing L-BFGS leaf solver on the same unconstrained L2 GLM objective, so they land at leaf level first. Nelder-Mead and Powell are the derivative-free local methods. They fill the black-box local-solver gap, and both serve as candidate inner minimizers for basin hopping.

Phase 1 is a hard architectural requirement, not a later cleanup. Leaf building must not depend on the solver attached to it. Every leaf declares the optimization problem it poses. Any compatible solver can solve that problem. Adding a solver must not require touching leaf-building code, and changing a leaf must not require touching solver code.

Phases 1 and 4 to 5 are additions to this specification. The solver additions came from `references/research/2026-11-21-optimization-solvers-implementation-landscape.md` section 3.6.1. The leaf decoupling requirement came from the same investigation and was missing from the earlier draft.

The seam sits above current numerical kernels. GLM-specific fitness evaluation becomes an adapter over the generic seam rather than the foundation of it. Existing specialized paths stay available where they are mathematically superior. Exact Gaussian fitting never routes through an iterative method.

One Rust entry point accepts a scalar objective, bounds, starting point, method, shared budget, seed, and method-specific options. One Python entry point accepts a callable scalar objective and returns one typed result. Constraint evaluation is part of the problem definition, not a method option.

## User Stories

### Shared Optimization Seam

1. As a quantitative researcher, I want to minimize an arbitrary bounded scalar objective through one public Rust function, so that I can optimize research quantities that are not GLM coefficients.
2. As a Python user, I want to pass a callable objective to one public optimization function, so that I can optimize tree thresholds, hyperparameters, and custom losses without writing Rust.
3. As a Python user, I want the objective to receive a one-dimensional numeric vector and return one finite scalar, so that the calling contract stays simple.
4. As a Python user, I want a separate multiobjective function for objectives returning vectors, so that scalar and vector results never share one ambiguous return type.
5. As an API user, I want one canonical method name per optimizer, so that aliases do not fragment the public interface.
6. As an API user, I want method-specific options grouped under their method, so that common controls stay separate from algorithm controls.
7. As an API user, I want lower and upper bounds for every parameter, so that optimization respects natural parameter domains.
8. As an API user, I want unbounded parameters to work when the selected method supports them, so that I am not forced to invent finite bounds.
9. As an API user, I want invalid bounds rejected before optimization starts, so that failures identify configuration errors instead of numerical instability.
10. As an API user, I want the returned solution guaranteed to be finite and within supported bounds, so that downstream code can rely on it.
11. As an API user, I want a starting point accepted by every method, so that I can warm-start a search.
12. As an API user, I want initial population samples accepted by population methods, so that experiments are reproducible.
13. As an API user, I want a maximum evaluation budget, so that expensive objectives cannot run without a limit.
14. As an API user, I want a maximum iteration budget, so that algorithms with meaningful generations stop even when evaluations remain.
15. As an API user, I want a tolerance, so that convergence has a reproducible stopping rule.
16. As an API user, I want an explicit seed, so that stochastic runs reproduce exactly.
17. As an API user, I want one documented default seed policy for every stochastic method, so that omitted seeds remain reproducible.
18. As an API user, I want the result to report best point, best objective, iterations, evaluations, and termination status, so that I can judge quality and cost.
19. As an API user, I want convergence, budget exhaustion, invalid objective value, numerical failure, infeasible bounds, and unsupported method to remain distinct statuses, so that results never imply success after failure.
20. As an API user, I want optional convergence history, so that I can diagnose stagnation.
21. As an API user, I want invalid objective values rejected by one documented rule, so that NaN or infinite losses cannot silently distort updates.
22. As an API user, I want a deterministic error when a method cannot support requested bounds, objective type, or constraints, so that the library never silently substitutes another algorithm.
23. As a library maintainer, I want shared objective, bounds, budget, and result abstractions, so that new algorithms reuse one validation and result contract.
24. As a library maintainer, I want existing GLM solvers adapted to the shared seam, so that standalone and leaf fitting do not maintain separate dispatch logic.
25. As an existing user, I want current solver names and standalone fitting behavior to keep working, so that adopting the generic optimizer breaks nothing.

### Constraint Semantics

26. As a constrained-optimization user, I want nonlinear inequality constraints, so that I can express feasibility directly.
27. As a constrained-optimization user, I want the feasibility sign convention fixed and documented, so that a constraint value means the same thing in every method.
28. As a constrained-optimization user, I want every constrained result to report maximum constraint violation, so that a slightly infeasible point is never presented as feasible.
29. As a constrained-optimization user, I want equality constraints rejected explicitly by methods that cannot support them, so that I never receive an approximate answer with an unstated guarantee.
30. As a constrained-optimization user, I want the policy for evaluating the objective at infeasible points documented per method, so that behavior is predictable.
31. As a library maintainer, I want one constraint model shared by every constrained method, so that adding SLSQP later does not break the public API.
32. As a library maintainer, I want bounds held separately from general constraints, so that each method uses the strongest native bound support it has.

### Random Search and Latin Hypercube

33. As a benchmark author, I want random search as a baseline, so that every advanced method can be compared against an honest uninformed baseline.
34. As a benchmark author, I want Latin hypercube sampling, so that I can test whether stratified initialization improves population methods.
35. As a population-method user, I want Latin hypercube samples available as initial populations, so that CMA-ES, DE, SHADE, PSO, and GA start from controlled coverage.
36. As a user, I want Latin hypercube samples reproducible for a seed, so that initialization comparisons are fair.
37. As a user, I want each Latin hypercube dimension stratified independently, so that the sampler covers every marginal range.
38. As a user, I want Latin hypercube samples shuffled within strata, so that samples do not share coordinate-ordering artifacts.
39. As a user, I want random search to respect bounds, so that every returned candidate is valid.

### L-BFGS-B

40. As a researcher, I want L-BFGS-B for smooth bounded objectives, so that box constraints do not force penalty-function approximations.
41. As a researcher, I want L-BFGS-B to retain limited-memory curvature updates, so that large parameter vectors stay practical.
42. As a researcher, I want the generalized Cauchy point and subspace minimization documented, so that the implementation matches the published algorithm.
43. As a researcher, I want L-BFGS and L-BFGS-B to share objective and convergence machinery, so that their difference stays explicit.
44. As a researcher, I want unbounded L-BFGS behavior unchanged when bounds do not bind, so that existing leaf fits stay stable.
45. As a researcher, I want the projected stopping test defined over free variables only, so that active bounds do not block convergence.

### BFGS and Nonlinear Conjugate Gradient

46. As a researcher, I want full-memory BFGS as a leaf solver, so that I can compare it against the existing L-BFGS leaf path on the same GLM objective.
47. As a researcher, I want nonlinear conjugate gradient with Polak-Ribiere plus restart as a leaf solver, so that I have a first-order competitor with linear memory instead of curvature history.
48. As a researcher, I want BFGS and nonlinear CG to solve the same unconstrained L2 objective as L-BFGS, so that the comparison stays meaningful.
49. As a researcher, I want BFGS memory cost documented as quadratic in parameter count, so that I only choose it for small leaves.
50. As a researcher, I want the Polak-Ribiere restart rule documented, so that the search direction resets predictably and stagnation cannot persist.
51. As a researcher, I want BFGS and nonlinear CG to fail with the same numerical statuses as L-BFGS, so that leaf fitting reports failure consistently.
52. As a researcher, I want BFGS and nonlinear CG to reject L1 and monotone-constrained leaf fits explicitly, so that unsupported combinations never silently change the objective.
53. As a benchmark author, I want BFGS, nonlinear CG, and L-BFGS compared on identical GLM objectives, weights, penalties, warm starts, and budgets, so that convergence speed, final deviance, gradient norm, memory, and evaluation count can be compared.
54. As a benchmark author, I want those leaf-level competitors compared across Binomial, Poisson, Gamma, Tweedie, and Tobit leaves, so that family-specific behavior stays visible.

### Nelder-Mead and Powell

55. As a researcher, I want Nelder-Mead available as a derivative-free local solver, so that I can minimize a black-box objective without gradients.
56. As a researcher, I want Powell available as a derivative-free local solver, so that I have a second derivative-free local method with different search behavior.
57. As a researcher, I want Nelder-Mead and Powell to report objective value and evaluations only, so that no derivative information is implied.
58. As a researcher, I want Nelder-Mead and Powell to respect bounds, so that returned points stay inside the parameter domain.
59. As a researcher, I want Nelder-Mead simplex parameters configurable, so that initial simplex shape and tolerance match published setups.
60. As a researcher, I want Powell direction-set and line-search parameters configurable, so that I can reproduce published setups.
61. As a researcher, I want both methods selectable as basin-hopping inner minimizers, so that the composition can be benchmarked across smooth and derivative-free local solvers.
62. As a benchmark author, I want Nelder-Mead and Powell compared against L-BFGS-B and against each other on the same bounded objective, so that derivative-free cost is measured rather than assumed.
63. As a library maintainer, I want both methods to reuse the shared budget, bounds, and result contracts, so that no method carries a private status vocabulary.

### Bounded Simulated Annealing

64. As a researcher, I want bounded simulated annealing, so that I have a cheap derivative-free global search for non-smooth objectives.
65. As a researcher, I want the proposal distribution configurable, so that I can match objective geometry.
66. As a researcher, I want the cooling schedule configurable with geometric cooling as default, so that exploration and exploitation balance is tunable.
67. As a researcher, I want the boundary policy documented and configurable between resampling and reflection, so that bound-active bias is a known choice rather than an accident.
68. As a researcher, I want the initial temperature expressed in objective units, so that one setting transfers across objectives.
69. As a researcher, I want the best incumbent tracked separately from the current point, so that the returned result never degrades.
70. As a researcher, I want simulated annealing to report evaluations and iterations separately, so that cost is comparable with population methods.
71. As a researcher, I want a stagnation counter available, so that a cooling schedule can stop early.
72. As a benchmark author, I want simulated annealing compared against random search under equal evaluation budgets, so that its added value is proven rather than assumed.

### COBYLA

73. As a constrained researcher, I want COBYLA for derivative-free nonlinear inequality constraints, so that I can optimize without supplying gradients.
74. As a constrained researcher, I want COBYLA bounds treated as unrelaxable, so that every evaluated point stays inside the box.
75. As a constrained researcher, I want a configurable feasibility tolerance, so that near-feasible points are accepted deliberately.
76. As a constrained researcher, I want COBYLA to report maximum constraint violation on its result, so that feasibility is measurable.
77. As a library maintainer, I want COBYLA cross-validated against a reference implementation, so that a hand-written port cannot silently disagree with the published algorithm.
78. As a library maintainer, I want the COBYLA linear subproblem solved with the existing pure-Rust linear-algebra dependencies, so that no new runtime dependency lands in the core crate.

### Basin Hopping

79. As a researcher, I want basin hopping, so that I have a global search for rugged landscapes with separated local minima.
80. As a researcher, I want the inner local minimizer selectable through the shared seam, so that the same method works with L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, or Powell.
81. As a researcher, I want the perturbation scaled per dimension by bound width, so that one step size fits heterogeneous parameter scales.
82. As a researcher, I want constant-temperature Metropolis acceptance plus a monotonic mode, so that I can compare exploration policies.
83. As a researcher, I want adaptive step size toward a target acceptance rate, so that step size is not a manual tuning burden.
84. As a researcher, I want the best incumbent preserved across iterations including failed local solves, so that one failed solve cannot erase the best result.
85. As a researcher, I want inner local-solve evaluations counted in the outer budget, so that reported cost is honest.
86. As a benchmark author, I want basin hopping compared against random search and annealing under equal evaluation budgets, so that the extra local-solve cost is visible.

### Benchmarks and Reproducibility

87. As a benchmark author, I want one canonical optimizer benchmark entry point, so that duplicate benchmark scripts do not accumulate.
88. As a benchmark author, I want all methods compared under equal evaluation budgets, so that faster convergence is measured fairly.
89. As a benchmark author, I want every stochastic method run across multiple fixed seeds, so that one lucky run cannot define quality.
90. As a benchmark author, I want median and worst-seed results reported, so that reliability stays visible.
91. As a benchmark author, I want evaluations-to-target reported, so that methods can be compared before final runtime.
92. As a benchmark author, I want wall time, peak memory, and evaluations reported separately, so that no single cost metric hides another.
93. As a benchmark author, I want objective gap to a known oracle reported for analytic and synthetic problems, so that near-optimal and failed runs are distinguishable.
94. As a benchmark author, I want constraint violation reported for constrained problems, so that feasibility does not hide behind objective value.
95. As a benchmark author, I want benchmark input data and split seeds fixed, so that comparisons remain reproducible.
96. As a benchmark author, I want external references used as benchmark-only dependencies, so that core installation stays lean.
97. As a benchmark author, I want no multi-core optimizer runs to execute concurrently, so that timing comparisons stay meaningful.
98. As a benchmark author, I want benchmark output written as machine-readable JSON plus a concise Markdown summary, so that results can be compared and reviewed.
99. As a benchmark author, I want benchmark runs to record package version, platform, seed set, budgets, and options, so that historical results remain interpretable.

### Leaf Solver Independence

100. As a library maintainer, I want leaf building to depend on the optimization problem rather than on the solver name, so that a leaf never hard-codes which solver may fit it.
101. As a library maintainer, I want every leaf that poses an optimization problem to declare its parameters, objective, optional gradient, bounds, and required solver capabilities, so that any compatible solver can fit it.
102. As a library maintainer, I want solver capabilities declared once in a capability table, so that compatibility is a lookup rather than a chain of conditionals.
103. As a library maintainer, I want a compatibility matrix over leaf kind, response family, and solver, so that an unsupported pairing fails with one precise reason instead of a family-specific error.
104. As a library maintainer, I want adding a new solver to require no leaf-building changes, so that solver work stays inside the solver module.
105. As a library maintainer, I want adding a new leaf to require no solver changes, so that leaf work stays inside the leaf module.
106. As a model user, I want a median leaf fitted by an optimizer rather than a hard-coded routine, so that the same solver family applies across robust leaf types.
107. As a model user, I want a Huber leaf to expose its M-location objective to the solver seam, so that the robust objective and the solver's convergence criteria stay consistent.
108. As a model user, I want a quantile leaf to expose its check loss, so that the fitted quantile matches the loss the solver minimizes.
109. As a model user, I want closed-form leaf kinds to ignore solver selection with a documented diagnostic status, so that setting a solver on a mean leaf does not fail the whole fit.
110. As a model user, I want a diagnostic field naming the solver actually used, so that I can tell whether a requested solver ran, was substituted, or did not apply.
111. As a model user, I want a diagnostic field naming the leaf problem type, so that I can tell which objective the fit minimized.
112. As a model user, I want solver rejection to name the exact missing capability, so that I know whether the solver lacks gradients, L1 support, monotone support, or unbounded support.
113. As a library maintainer, I want leaf dispatch to route through one solver entry point, so that the current per-solver branches in leaf fitting disappear.
114. As a benchmark author, I want the same solver to fit different leaf kinds under identical settings, so that solver comparisons hold across leaf types.
115. As a benchmark author, I want leaf fits compared across solvers on the same leaf problem, so that solver differences do not confound leaf differences.

## Implementation Decisions

### Decision 1: New Module Namespace

Create `crates/mltreelib-core/src/solver/generic/`. It contains:

- `objective.rs` — `Objective` and `SmoothObjective` traits.
- `bounds.rs` — `Bounds` type and validation.
- `constraint.rs` — `Constraint`, `ConstraintKind`, and feasibility reporting.
- `request.rs` — `OptimizerKind`, `OptimizationRequest`, and shared options.
- `result.rs` — `OptimizationResult`, `SolveStatus`, and history.
- `budget.rs` — evaluation and iteration budget accounting.
- `random.rs` — Random Search and Latin Hypercube.
- `lbfgs_b.rs` — bounded smooth local solver.
- `bfgs.rs` — full-memory quasi-Newton local solver, shared with the leaf path.
- `conjugate_gradient.rs` — nonlinear Polak-Ribiere plus restart local solver, shared with the leaf path.
- `nelder_mead.rs` — derivative-free simplex local solver.
- `powell.rs` — derivative-free conjugate-direction local solver.
- `annealing.rs` — bounded simulated annealing.
- `cobyla.rs` — nonlinear constraint solver.
- `basin_hopping.rs` — global composition method.

Existing `solver/mod.rs` files keep their current roles. `solver/mod.rs` gains re-exports for the generic surface and keeps `SolverOptions`, `StopReason`, and `Convergence` for the GLM path. The generic result type reuses `StopReason` values where they apply and adds the missing statuses.

The GLM path becomes an adapter over the generic seam in a later step. It does not move first.

### Decision 2: Objective and Bounds Contracts

`Objective` reports dimension, bounds, and one finite objective value. `SmoothObjective` adds gradient. `Bounds` holds per-dimension lower and upper values with rules:

- `lower[i] <= upper[i]`.
- Infinite bounds allowed only where the method supports them.
- Every returned point is finite and within bounds.
- Unbounded methods report an unsupported-bounds error instead of silently clipping.
- Initial points outside bounds are projected or rejected according to documented method behavior.

Constraint sign convention: an inequality constraint is feasible when its value is `>= 0`. An equality constraint is feasible when its value is zero within tolerance. This matches the SciPy and PRIMA convention and reduces translation mistakes.

### Decision 3: Request, Options, and Result Contracts

`OptimizationRequest` carries the objective, optional smooth objective, starting point, method, shared options, and method-specific options. Shared options cover maximum evaluations, maximum iterations, tolerance, seed, initial population, and verbosity.

`OptimizationResult` carries best point, best objective, iterations, evaluations, status, optional history, optional maximum constraint violation, and optional local-solve metadata for basin hopping.

`SolveStatus` values:

- `Converged`
- `IterationLimit`
- `EvaluationLimit`
- `Tolerance`
- `InvalidObjective`
- `NumericalFailure`
- `Infeasible`
- `Unsupported`
- `HardError`

Budget exhaustion is never reported as success.

### Decision 4: Budget Accounting

One budget counter counts every user objective evaluation. It counts evaluations inside local solves, inside perturbation steps, inside population initialization, and inside archive or migration updates. Basin hopping forwards the outer budget into the inner local solve so no evaluation escapes counting.

Iteration budget is separate. Reaching either limit produces its own status.

### Decision 5: Determinism

Every stochastic method accepts an explicit seed and produces identical candidates and results for identical inputs, bounds, options, and single-thread execution. Use the seeded `rand` generators already in the dependency set. No wall-clock decision appears inside a solver loop.

### Decision 6: Random Search and Latin Hypercube Land First

Random Search samples uniformly inside bounds. Latin Hypercube stratifies each dimension independently, draws one sample per stratum, and shuffles strata with the seeded generator.

These land before any global method. Without an uninformed baseline, no later method can claim improvement. Population methods in `evolutionary.rs` may accept LHS samples as initial populations later.

### Decision 7: L-BFGS-B Algorithm

Port the L-BFGS-B iteration as defined by Byrd, Lu, Nocedal, and Zhu (1995). Each iteration:

1. Computes the generalized Cauchy point along the piecewise-linear projected-gradient path to identify the active set.
2. Approximately minimizes the quadratic model over free variables using the compact limited-memory BFGS representation.
3. Applies a line search.
4. Projects onto bounds.
5. Tests the projected gradient norm over free variables for convergence.

Do not substitute a projected-gradient method with L-BFGS preconditioning. That method is weaker, differs in behavior, and cannot reproduce reference L-BFGS-B results.

Reuse the two-loop recursion in `solver/lbfgs.rs` as the curvature memory base. Keep unbounded L-BFGS behavior unchanged so leaf fits stay stable.

Cross-validate against the original Fortran implementation and against an independent pure-Rust implementation.

### Decision 8: BFGS and Nonlinear Conjugate Gradient at Leaf Level

Two methods from the SciPy and statsmodels competitor audit compete directly against the existing L-BFGS leaf solver. They land at leaf level because they solve the same unconstrained L2 GLM objective through the same objective and gradient machinery already present in `solver/lbfgs.rs`.

**BFGS** maintains a full dense inverse-Hessian approximation. Memory is quadratic in parameter count, so it is a benchmark candidate for small leaves rather than a default for large leaves. The implementation follows the Broyden, Fletcher, Goldfarb, and Shanno update with a standard line search.

**Nonlinear conjugate gradient** uses the Polak-Ribiere direction with a restart. It needs only the objective and gradient, keeps memory linear in parameter count, and carries no curvature history. The restart rule must be documented so search-direction resets stay predictable under stagnation.

Both methods:

- Reuse the existing GLM objective and gradient evaluation in `solver/lbfgs.rs` so the compared objective is identical across methods.
- Report the same convergence statuses and evaluation counts as L-BFGS.
- Reject L1 and monotone-constrained leaf fits explicitly. Those objectives differ, so a silent reroute would invalidate the comparison.
- Stay experimental in the Python surface until benchmark evidence supports promotion.

They also register in the generic method enumeration. BFGS and nonlinear CG support smooth objectives without box bounds. The generic entry point returns an explicit unsupported error when bounds are finite and binding for either method.

### Decision 9: Nelder-Mead and Powell as Derivative-Free Local Solvers

These two fill the black-box local-solver gap and serve as candidate inner minimizers for basin hopping.

**Nelder-Mead** maintains a simplex in parameter space and replaces its worst vertex each iteration. It needs objective values only. Initial simplex shape, expansion and contraction coefficients, and the two convergence tolerances are configurable so published setups can be reproduced.

**Powell** performs sequential one-dimensional minimizations along a maintained set of conjugate directions. It also needs objective values only. Direction-set size and line-search tolerance are configurable.

Both methods:

- Reuse the shared `Objective` trait, `Bounds`, budget, and result contracts. Neither carries a private status vocabulary.
- Respect bounds. Nelder-Mead clips or reflects simplex vertices into the box according to a documented policy. Powell keeps every line-search evaluation inside the box, matching SciPy behavior for an in-bounds start.
- Report objective value and evaluation count only. No gradient or Hessian information is implied or fabricated.
- Serve as selectable inner minimizers for basin hopping alongside L-BFGS-B, BFGS, and nonlinear CG.

### Decision 10: Bounded Simulated Annealing

Classical annealing applied to a continuous bounded domain. Required explicit choices:

- **Proposal.** Per-dimension bounded random displacement with configurable scale.
- **Cooling.** Geometric schedule `T(k+1) = alpha * T(k)` by default, with configurable `alpha`, minimum temperature, and stagnation count.
- **Boundary.** Resampling or reflection, configurable and documented. Clipping is not offered because it piles proposal mass on bound-active points and biases the search.
- **Acceptance.** Metropolis rule with temperature in objective units.

Initial temperature is derived from observed objective spread when not supplied. Best incumbent tracked separately from current point.

This method ships as heuristic. Logarithmic cooling supports global convergence results but is far too slow for normal use, so no optimality claim appears in documentation.

### Decision 11: COBYLA as the One Nonlinear Constraint Solver

COBYLA models objective and constraints by linear interpolation over an interpolation set, then solves a small linear subproblem inside a shrinking trust region.

Reasons for this choice:

- It needs function values only, matching the generic `Objective` seam.
- It supports arbitrary nonlinear inequality constraints.
- Bounds are expressible as constraints, so one constraint model covers both.
- A pure-Rust reference implementation exists for cross-validation.
- COBYLA lacks favourable convergence theory, so it ships as a robust practical method without an optimality claim.

Reasons against the alternatives:

- **SLSQP** handles equality constraints natively but needs a working-set quadratic subproblem. It is deferred until equality constraints become a real requirement.
- **COBYQA** is a stronger successor with quadratic models and native equality support. It is deferred because it is a much larger build.
- **`trust-constr`** is the most capable and the most expensive. It stays out.

Equality constraints are rejected explicitly by COBYLA. The constraint model already supports `ConstraintKind::Equality` so adding SLSQP later does not break the public API.

The linear subproblem uses the existing `nalgebra` and `faer` dependencies. No new runtime dependency enters `mltreelib-core`.

### Decision 12: Basin Hopping Composition

Basin hopping is a composition, not a standalone numerical kernel.

- Perturbation: bounded random displacement scaled per dimension by bound width.
- Local minimizer: supplied through the shared seam, defaulting to L-BFGS-B.
- Acceptance: constant-temperature Metropolis test, plus a monotonic mode at zero temperature.
- Step size: fixed or adaptive toward a target acceptance rate.
- Incumbent: best point tracked across iterations and never replaced by a failed local solve.

Local solve failures produce an explicit status or are skipped with the incumbent preserved. They never abort the whole run silently.

### Decision 13: Reuse Policy for External Crates

No external optimization crate becomes a runtime dependency of `mltreelib-core`.

Pure-Rust options considered:

| Crate | Coverage | Role |
|---|---|---|
| `basin` | L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, COBYLA, SLSQP, simulated annealing, basin hopping, constraint adapters | Cross-validation oracle only. Broad framework would duplicate the package's own solver semantics and add a second state and error model. |
| `cobyla` | COBYLA | Cross-validation oracle only. |
| `slsqp` | SLSQP | Deferred oracle for a future equality-constraint method. |
| `good_lp` | Linear and mixed-integer programming | Considered for the COBYLA subproblem. Rejected for the core crate because it pulls a solver stack. `nalgebra` and `faer` cover the subproblem. |
| `nlopt` | Broad catalog via C binding | Rejected. Requires a C toolchain. |
| `gomez`, `argmin` | Derivative-free methods and frameworks | Design references only. |

Reason for the policy: `mltreelib-core` already ships L-BFGS, CCD, and evolutionary solvers with package-specific semantics for `GlmFamily`, monotone projection, and missing-value handling. A second framework would duplicate that surface and require an adapter over every call site. Determinism, evaluation budgets, and stop reasons must live in the package's own seam.

These crates are used as dev-dependencies and test oracles. This mirrors the practice in the `basin` project itself, which validated its Powell-family solvers against PRIMA and its L-BFGS-B against the original Fortran code.

### Decision 14: Python Surface

`python/mltreelib/solvers.py` grows from standalone fitting into the package optimizer surface:

- `fit` — weighted Gaussian and GLM fitting. Existing name and behavior stay unchanged.
- `optimize` — scalar callable objective.
- `minimize_many` — vector callable objective, added with the multiobjective methods.

The Python objective callback receives a one-dimensional numeric vector and returns one finite scalar. Unsupported callback behavior raises an actionable error.

The PyO3 binding lives in `crates/mltreelib-py/src/solvers.rs`, which already holds the standalone fitting binding. The typed result class exposes best point, best objective, iterations, evaluations, status, optional history, and constraint violation.

### Decision 15: GLM Adapter Migration

After the generic seam works, adapt the GLM path:

- `solver/dense.rs` keeps its specialized dispatch for exact Gaussian, CCD for Gaussian L1 and sign constraints, and L-BFGS for smooth non-Gaussian families.
- Evolutionary solvers gain a generic objective adapter so their population machinery becomes reachable from the public generic entry point.
- The shared compatibility table defines supported families, penalties, monotone constraints, gradient availability, and canonical names.
- Unsupported combinations fail before objective evaluation.
- Inference statistics stay available only from analytically supported direct Gaussian fitting.

### Decision 16: Benchmark Program

One canonical benchmark entry point, `benchmarks/speed/benchmark_optimizers.py`, with a documented command. It covers:

1. Analytic regression with an exact normal-equation oracle.
2. Synthetic optimization functions: sphere, rotated ellipsoid, Rosenbrock, Rastrigin, and a multimodal function with a known global minimum.
3. Constrained problems with known feasible optima and known infeasible starts.
4. Statistical likelihoods compared against external references when available.
5. Native package use cases: tree hyperparameter search and split-threshold search.

Every record holds method, objective, seed, budget, evaluations, iterations, wall time, best objective, objective gap, constraint violation, and termination status. Output goes to JSON plus a concise Markdown summary.

### Decision 17: Leaf Problems Are Solver-Independent

Leaf building and solver selection are two independent axes. A leaf states the problem it poses. A solver states what it can solve. Compatibility is a lookup between them.

Current state is the opposite. `LeafControl::validate_solver` rejects any non-`Linear` leaf when a solver is set, and the leaf fitting path in `leaf.rs` branches directly on solver identity: `use_lbfgs` is computed from `control.solver == LeafSolver::Lbfgs`, a separate `else if control.solver == LeafSolver::Ccd` branch exists, and evolutionary solvers get a third branch. Each branch re-implements option construction and result unwrapping.

The decoupling has three parts.

**Leaf problem.** Each leaf kind declares the optimization problem it poses, if any.

| Leaf kind | Problem | Solver-independent? |
|---|---|---|
| Linear | Penalized GLM deviance over intercept and coefficients | Yes. Smooth for most families. Gradient available. |
| Median | Weighted absolute deviation for a location estimate | Yes. Convex, nonsmooth. Derivative-free solvers fit it naturally. |
| Quantile | Weighted check loss at the requested tau | Yes. Convex, nonsmooth. |
| Huber | Huber M-location loss | Yes. Convex, smooth. Gradient available. |
| Mean | Closed-form weighted mean | No optimization problem. Solver selection does not apply. |
| Prob, Odds, LogOdits | Closed-form transforms of the fitted mean | No optimization problem. Solver selection does not apply. |
| MARS | Successive basis expansion with its own internal fitting | Not yet. Keep internal fitting. |
| KNN | Neighbor index plus distance-weighted response | No optimization problem. Solver selection does not apply. |

**Solver capability.** Each solver declares its capabilities once: supports gradients, supports derivative-free operation, supports L1 penalties, supports monotone coefficient constraints, supports finite box bounds, supports equality constraints, supports nonlinear inequality constraints, and reports a memory profile.

**Compatibility lookup.** A leaf problem plus a solver plus a response family resolves through one table to either a supported pairing or a precise rejection reason. The reason names the missing capability, not the leaf.

Consequences:

- Median, quantile, and Huber leaves become optimizer-driven rather than hard-coded routines. Their current analytic implementations remain the default path for exactness, but a user can now select a solver for them.
- Adding a solver touches only the capability table and the solver module. No leaf-building code changes.
- Adding a leaf touches only the leaf module. No solver code changes.
- Closed-form leaf kinds ignore solver selection. They report a documented diagnostic status naming the solver as not applicable instead of failing the fit.

### Decision 18: One Leaf-to-Solver Entry Point

Leaf fitting routes through one entry point that accepts a leaf problem and a solver. The current per-solver branches in `leaf.rs` are replaced by one call.

The entry point performs these steps in order:

1. Resolve the leaf problem from leaf kind, response family, and leaf control.
2. Resolve solver capabilities from the solver registry.
3. Check compatibility. On failure, return one precise reason.
4. Build the objective, gradient, bounds, and constraints for the problem.
5. Call the solver through the shared seam.
6. Map the solver result back into the leaf model.

The leaf model gains two diagnostic fields: the solver actually used, and the leaf problem type. When a requested solver does not apply, the diagnostic names it. When a solver is substituted for a family-specific reason, the diagnostic names the substitution.

The existing `LeafSolver` enum remains the public selector. Its variants grow with the new methods. The internal dispatch stops branching on solver identity and starts branching on leaf problem type.

### Decision 19: Documentation

Public docs cover the generic optimizer, the solver capability table, the leaf problem and solver compatibility matrix, the defaults table, result statuses, deterministic behavior, constraint semantics, and the benchmark command.

Examples cover bounded sphere minimization with Random Search and Latin Hypercube, L-BFGS-B on a bounded smooth objective, BFGS and nonlinear CG on a GLM leaf, Nelder-Mead and Powell on a derivative-free objective, simulated annealing on a multimodal objective, COBYLA on a nonlinear constraint problem, basin hopping on a rugged objective, one leaf fitted with two different solvers, and one tree-hyperparameter search.

## Exact Build Sequence

Each phase lands as its own commit with focused tests. A phase does not start before the previous phase passes its gate.

### Phase 0 — Shared Seam

1. Add `solver/generic/objective.rs` with `Objective` and `SmoothObjective` traits.
2. Add `solver/generic/bounds.rs` with `Bounds` and validation for ordering, finiteness, and dimension agreement.
3. Add `solver/generic/constraint.rs` with `Constraint`, `ConstraintKind`, feasibility tolerance, and maximum-violation computation.
4. Add `solver/generic/result.rs` with `OptimizationResult` and the full `SolveStatus` set.
5. Add `solver/generic/budget.rs` with one evaluation counter and one iteration counter.
6. Add `solver/generic/request.rs` with `OptimizerKind`, `OptimizationRequest`, shared options, and method option structs.
7. Re-export the generic surface from `solver/mod.rs` and `crates/mltreelib-core/src/lib.rs`.
8. Add Rust tests at `crates/mltreelib-core/tests/optimize.rs` for bounds validation, budget accounting, status distinctness, and feasibility sign convention.

Gate: `cargo test` passes. No method implemented yet.

### Phase 1 — Leaf Solver Decoupling

1. Add a leaf problem abstraction in `crates/mltreelib-core/src/leaf.rs` that maps each leaf kind and response family to the optimization problem it poses, or to a closed-form path that poses no problem.
2. Add a solver capability table in `crates/mltreelib-core/src/solver/mod.rs` declaring, per solver, support for gradients, derivative-free operation, L1 penalties, monotone coefficient constraints, finite box bounds, equality constraints, nonlinear inequality constraints, and memory profile.
3. Add one compatibility lookup over leaf kind, response family, and solver that returns either a supported pairing or one precise rejection reason naming the missing capability.
4. Replace the per-solver branches in the leaf fitting path with one entry point that resolves problem, checks compatibility, builds objective and gradient, calls the solver, and maps the result back into the leaf model.
5. Expose the median, quantile, and Huber leaf objectives through the seam so a solver can fit them. Keep their current analytic implementations as the default exact path.
6. Relax `LeafControl::validate_solver` so closed-form leaf kinds accept any solver selection and report a diagnostic that the solver does not apply.
7. Add diagnostic fields for the solver actually used and the leaf problem type.
8. Add Rust tests proving the same solver fits different leaf kinds under identical settings, and that an unsupported pairing names the missing capability.

Gate: `cargo test` and full `uv run pytest tests/ -q` pass. Leaf fitting contains no branch on solver identity. Adding a solver requires no leaf-module change.

### Phase 2 — Random Search and Latin Hypercube

1. Implement `solver/generic/random.rs` with uniform random search inside bounds.
2. Implement Latin Hypercube with per-dimension stratification and seeded shuffle.
3. Wire both into the generic entry point.
4. Add Python `optimize` binding and `OptimizationResult` class in `crates/mltreelib-py/src/solvers.rs`, exposed through `python/mltreelib/solvers.py`.
5. Add Python tests in `tests/test_optimize.py` for result fields, bounds, seed reproducibility, and evaluation counts.

Gate: `cargo test` and `uv run pytest tests/test_optimize.py -q` pass. Random Search finds a known optimum on a deterministic toy objective within a fixed budget.

### Phase 3 — L-BFGS-B

1. Implement `solver/generic/lbfgs_b.rs` with the generalized Cauchy point, subspace minimization, line search, projection, and projected stopping test.
2. Reuse the two-loop recursion from `solver/lbfgs.rs` as curvature memory.
3. Confirm unbounded L-BFGS-B output matches current `solve_lbfgs` within numerical tolerance.
4. Add cross-validation tests against an independent implementation for a bounded smooth objective and a problem where bounds bind.
5. Expose `lbfgs_b` in the generic method enumeration and Python surface.

Gate: `cargo test` and Python tests pass. Feasibility holds. Unbounded equivalence holds.

### Phase 4 — BFGS and Nonlinear Conjugate Gradient

1. Add `solver/generic/bfgs.rs` with full-memory inverse-Hessian update and line search, reusing the GLM objective and gradient evaluation from `solver/lbfgs.rs`.
2. Add `solver/generic/conjugate_gradient.rs` with Polak-Ribiere direction and documented restart rule.
3. Add `LeafSolver::Bfgs` and `LeafSolver::NonlinearConjugateGradient` variants in `crates/mltreelib-core/src/leaf.rs`, with explicit rejection of L1 and monotone-constrained leaf fits.
4. Wire both into `solve_iterative` in `solver/mod.rs` and register their capabilities in the solver capability table. Phase 1 already removed solver-identity branching from leaf fitting, so no leaf dispatch change is needed.
5. Add Python parsing and name mapping in `crates/mltreelib-py/src/lib.rs`, and expose both as experimental solver names.
6. Add Rust tests proving BFGS, nonlinear CG, and L-BFGS reach equivalent objective quality on the same weighted GLM design within a shared budget.
7. Add Python tests in `tests/test_solvers.py` for solver-name round trips and family coverage.

Gate: `cargo test` and `uv run pytest tests/test_solvers.py -q` pass. All three methods solve the same objective to equivalent quality. No existing leaf fit changes numerically.

### Phase 5 — Nelder-Mead and Powell

1. Add `solver/generic/nelder_mead.rs` with simplex initialization, reflection, expansion, contraction, shrink, and configurable tolerances.
2. Add `solver/generic/powell.rs` with conjugate-direction line searches and a configurable direction set.
3. Wire both into the generic method enumeration with the shared bounds, budget, and result contracts.
4. Expose both in the Python surface as `nelder_mead` and `powell`.
5. Add Rust tests for bound feasibility, convergence on known deterministic objectives, evaluation counting, and absence of gradient requirements.
6. Add Python tests for result fields and reproducibility.

Gate: `cargo test` and Python tests pass. Both methods return feasible points and terminate with a documented status.

### Phase 6 — Bounded Simulated Annealing

1. Implement `solver/generic/annealing.rs` with Metropolis acceptance, geometric cooling, configurable proposal scale, and resampling or reflection boundary policy.
2. Derive initial temperature from observed objective spread when not supplied.
3. Track best incumbent separately from current point.
4. Add stagnation counter and explicit budget statuses.
5. Add Rust tests for determinism, bound feasibility, incumbent monotonicity, and known multimodal optimum reachability.
6. Add a benchmark record comparing annealing against Random Search under equal evaluation budgets.

Gate: `cargo test` and Python tests pass. Annealing matches random search or beats it within the documented budget on the benchmark set, otherwise the method stays internal.

### Phase 7 — COBYLA

1. Implement `solver/generic/cobyla.rs` with linear models over the interpolation set, the trust-region radius schedule, and the linear subproblem solved with `nalgebra` and `faer`.
2. Enforce equality-constraint rejection with an explicit unsupported status.
3. Treat bounds as unrelaxable constraints and report maximum constraint violation.
4. Add Rust tests for known constrained optima, infeasible starts, equality rejection, and feasibility tolerance.
5. Add cross-validation against an independent pure-Rust COBYLA implementation on fixed problems.
6. Expose `cobyla` in the generic method enumeration and Python surface with a constraint argument.

Gate: `cargo test` and Python tests pass. Every returned point is inside bounds and feasible within tolerance.

### Phase 8 — Basin Hopping

1. Implement `solver/generic/basin_hopping.rs` composing perturbation, a seam-supplied local minimizer, and Metropolis acceptance.
2. Support L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, and Powell as selectable inner minimizers.
3. Forward the outer evaluation budget into the inner local solve.
4. Add monotonic mode and adaptive step size toward a target acceptance rate.
5. Preserve the best incumbent across failed local solves.
6. Add Rust tests for determinism, incumbent preservation, budget accounting including inner solves, bound feasibility, and local-minimum property under the supplied local solver.
7. Add benchmark records comparing basin hopping against Random Search and annealing under equal evaluation budgets, and against each candidate inner minimizer.

Gate: `cargo test` and Python tests pass. Reported evaluations equal actual objective calls.

### Phase 9 — GLM Adapter Migration

1. Add a generic objective adapter over the existing evolutionary kernels so CMA-ES, SHADE, sNES, and the GA become reachable from the generic entry point.
2. Move the compatibility table into one shared location used by generic optimization, standalone fitting, and leaf fitting.
3. Confirm standalone fitting and single-leaf fitting agree on the same weighted design, family, penalty, and method.
4. Confirm explicit solver failures propagate identically through standalone and leaf paths.
5. Preserve existing leaf missing-data policies.

Gate: `cargo test` and full `uv run pytest tests/ -q` pass. No existing leaf fit changes numerically.

### Phase 10 — Benchmark and Documentation

1. Write `benchmarks/speed/benchmark_optimizers.py` with the canonical command, JSON output, and Markdown summary.
2. Add analytic, synthetic, constrained, statistical, and tree-native objective sets.
3. Include BFGS, nonlinear CG, Nelder-Mead, and Powell in the local-solver comparison block alongside L-BFGS-B.
4. Add the examples listed in Decision 19.
5. Regenerate API docs and update the solver section of the package README.
6. Update the roadmap checkboxes in `README.md`.

Gate: benchmark runs end to end and writes both artifacts. Documentation matches actual method names, statuses, and defaults.

## Testing Decisions

- **Good tests assert external behavior.** Tests cross the highest available seam. They verify returned objective quality, feasibility, status, counts, reproducibility, and public model predictions. They do not assert private update formulas, internal vector layout, or mutation order, except where a numerical regression protects a mathematical invariant.
- **Primary Rust seam.** Call the generic scalar optimization entry point with deterministic toy objectives. Assert best values, feasibility, evaluation budgets, status, and determinism.
- **Primary Python seam.** Call `optimize` with a callable objective. Assert result types, result fields, exception behavior, and deterministic repeatability.
- **Model-fitting seam.** Compare standalone fitting and single-leaf fitting on the same weighted design, family, penalty, and method. Assert coefficients, predictions, and objective agree within solver tolerance.
- **Compatibility seam.** Table-driven tests cover family, method, L1, L2, and monotone combinations. Unsupported combinations fail before objective evaluation.
- **Leaf independence seam.** Fit the same leaf kind with different solvers under identical settings and assert the fit only changes where the objective differs. Fit different leaf kinds with the same solver and assert the leaf problem, not the solver, drives the objective. Assert an unsupported pairing names the exact missing capability. Assert a closed-form leaf with a selected solver succeeds and reports the solver as not applicable.
- **Defaults seam.** Assert one documented defaults table drives generic, standalone, and leaf paths.
- **Failure seam.** Invalid bounds, wrong dimensions, invalid objective values, exhausted budgets, and numerical failures return distinct documented statuses or errors. No test accepts silent fallback to another solver.
- **Bounds seam.** Every supported method returns feasible points for finite and partially unbounded domains.
- **Budget seam.** Objective call counts never exceed the configured evaluation budget, and match the reported count exactly.
- **Feasibility seam.** Assert the sign convention, maximum-violation reporting, equality rejection, and infeasible-start handling.
- **Random and LHS seam.** Assert bounds, marginal stratification, deterministic samples, and distinct shuffled strata.
- **L-BFGS-B seam.** Assert feasibility, smooth-objective convergence, consistency with unbounded L-BFGS when bounds do not bind, and cross-validation against an independent implementation.
- **BFGS and nonlinear CG seam.** Fit the same weighted GLM design with BFGS, nonlinear CG, and L-BFGS. Assert equivalent objective quality within solver tolerance, matching convergence statuses, evaluation counts, and explicit rejection of L1 and monotone leaf fits.
- **Nelder-Mead and Powell seam.** Assert bound feasibility, termination on known deterministic objectives, evaluation counting, absence of gradient requirements, and reproducibility for every seeded configuration.
- **Annealing seam.** Assert determinism, bounds, incumbent monotonicity, boundary policy behavior, and known multimodal optimum reachability within a documented budget.
- **COBYLA seam.** Assert feasibility, equality rejection, tolerance behavior, and cross-validation against an independent implementation.
- **Basin hopping seam.** Assert determinism, incumbent preservation across failed local solves, complete budget accounting including inner solves, and the local-minimum property.
- **Benchmark smoke test.** Run a tiny budget through every method on one deterministic objective and one small dataset. Verify output schema, counts, statuses, and artifact creation without running the full benchmark in routine tests.
- **Full benchmark.** Run separately under the benchmark command. Store JSON and Markdown results outside routine unit tests. Thresholds use repeated seeds and external references rather than one brittle sample.
- **Prior art.** Reuse patterns from `crates/mltreelib-core/tests/solver.rs`, `tests/test_solvers.py`, the existing deterministic RNG tests, and the weighted GLM reference comparisons already in the repository.

## Out of Scope

- Building a SciPy, never-ending optimizer, or commercial solver replacement.
- Guaranteeing global optimality for heuristic methods.
- Multiobjective methods, NSGA-II, and island orchestration. They follow in a later wave.
- SLSQP, COBYQA, and `trust-constr` implementations.
- MIP, SAT, conic, trust-region, SGD momentum, and AdamW.
- Newton, Newton-CG, and TNC variants. Full-memory BFGS and nonlinear conjugate gradient are in scope; second-order and truncated-Newton methods are not.
- Deep neural-network optimization and autodiff integration.
- Plugin discovery or third-party solver installation.
- Automatic hyperparameter-optimization products beyond providing optimizer primitives.
- Replacing exact Gaussian fitting with iterative optimization.
- Changing tree split formulas, growth controls, or leaf families beyond solver support needed here.
- Supporting inference statistics for iterative or evolutionary fits.
- Claiming bit-identical results across CPUs or math libraries for floating-point kernels.
- Adding external optimization crates as runtime dependencies of `mltreelib-core`.

## Further Notes

- Research basis for the method choices, ordering, constraint semantics, and reuse policy: `references/research/2026-11-21-generic-global-and-constrained-solvers.md`.
- Research basis for BFGS, nonlinear conjugate gradient, Nelder-Mead, and Powell: `references/research/2026-11-21-optimization-solvers-implementation-landscape.md` section 3.6.1, the SciPy and statsmodels competitor audit.
- This specification extends `specs/21-standalone-solvers-and-benchmarks.md`. Spec 21 owns the general bounded optimizer seam, production CMA-ES, shared model-fit dispatch, and the benchmark program. This spec owns the constraint model, the leaf solver decoupling, the seven local and global methods, and their exact build order. Where the two overlap on the seam, Spec 21's contract governs.
- Phase 1 leaf decoupling changes `LeafControl::validate_solver`. Current behavior rejects any non-`Linear` leaf when a solver is set. Decoupled behavior accepts the pairing, resolves it through the capability lookup, and reports the outcome in diagnostics. This is a deliberate behavior change and needs its own release note.
- `specs/21-standalone-solvers-and-benchmarks.md` proposes PSO, DE, and NSGA-II. Those remain in Spec 21's wave structure. This spec's global methods are complementary, not replacements, and must each beat Random Search under equal evaluation budgets before public exposure.
- Current source wins where older notes conflict. `solver/lbfgs.rs` already implements L-BFGS, `solver/ccd.rs` implements coordinate descent, and `solver/evolutionary.rs` implements four population methods.
- Existing uncommitted work already adds `solver/dense.rs`, the standalone PyO3 binding in `crates/mltreelib-py/src/solvers.rs`, the Python `fit` surface in `python/mltreelib/solvers.py`, and `benchmarks/speed/benchmark_standalone_solvers.py`. Phase 0 extends that surface. It does not rewrite it.
- One multi-core job at a time. Rayon-backed solvers and benchmark suites must not run concurrently with other multi-core fits.
