# Generic Global and Constrained Solvers for `mltreelib`

**Date:** 2026-11-21
**Scope:** Bounded simulated annealing, basin hopping, and one nonlinear constraint solver, plus their dependency order on the generic optimizer seam described in `2026-11-21-general-solver-api-roadmap.md`.
**Decision informed:** Which generic optimizers to build in `mltreelib-core`, in what order, and what each one must guarantee.
**Confidence:** High for algorithm mechanics and dependency ordering. Medium for comparative performance claims, which come from papers measured on other problem families.

**Implementation spec:** `specs/26-generic-global-and-constrained-optimizers.md` holds the exact build sequence, module layout, contracts, and phase gates for this note. Spec 26 extends this phase list with a leaf solver decoupling phase and with the BFGS, nonlinear conjugate gradient, Nelder-Mead, and Powell phases. Spec 26 phase numbering governs.

---

## 1. Decision

Build three generic methods, in this order:

| Order | Method | Why this position |
|---:|---|---|
| 0 | Shared optimizer seam | Prerequisite for everything. `Objective`, `SmoothObjective`, `Bounds`, `Constraint`, evaluation budget, `SolveStatus`, feasibility reporting. |
| 1 | Random Search + LHS | Cheap baseline. Every later global method needs something to beat. |
| 2 | **L-BFGS-B** | Smooth bounded local solver. Also the inner minimizer that basin hopping needs, so it unblocks two later items. |
| 3 | **Bounded simulated annealing** | Trajectory method with no local-minimizer dependency. Works with a black-box objective only. Small implementation. |
| 4 | **COBYLA** | The one nonlinear constraint solver. Derivative-free, supports nonlinear inequality constraints, matches the black-box objective seam, and has a published reference implementation to test against. |
| 5 | **Basin hopping** | Composes perturbation, a local minimizer, and Metropolis acceptance. Needs item 2 to be useful. Comes last. |

Constraint-solver decision: **COBYLA first for the derivative-free path. SLSQP later when gradients exist and equality constraints become a real requirement. COBYQA later when equalities, noisy objectives, or hard bound domains appear.** COBYLA handles nonlinear inequality constraints only. SLSQP handles equality and inequality constraints but needs a sequential quadratic programming implementation with a working-set QP subproblem, which is a much larger and more failure-prone build. `trust-constr` stays out: it is the most capable and the most expensive to get right.

Simulated annealing is a heuristic in any practical bounded form. The logarithmic cooling schedule that supports global convergence results is far too slow for normal use. Build it because it is cheap and composable, and benchmark it against Random Search before exposing it publicly.

Do not add `basin` or `cobyla` as runtime dependencies of `mltreelib-core`. Use them as cross-validation oracles in tests instead. Reasoning in section 8.

---

## 2. Current Baseline

`crates/mltreelib-core/src/solver/` currently exposes:

| Module | Contents | Usable as a generic optimizer today? |
|---|---|---|
| `normal.rs` | Exact Gaussian OLS and Ridge via Cholesky | No. Requires a `DesignMatrix` and a response. |
| `ccd.rs` | Elastic net and monotone constraints via coordinate descent | No. Same reason. |
| `lbfgs.rs` | L-BFGS for smooth GLM deviance | No. Takes `DesignMatrix`, `GlmFamily`, and `LbfgsOptions`. |
| `dense.rs` | Dense GLM fitting with solver dispatch | No. Fixed families and fixed response semantics. |
| `evolutionary.rs` | CMA-ES, SHADE, sNES, real-coded GA | Partially. Already objective-value driven, but no bounds contract and no shared budget accounting. |
| `summary.rs` | R-style and statsmodels-style coefficient statistics | Not an optimizer. |
| `family.rs` | GLM family and link functions | Not an optimizer. |

`LeafSolver` in `crates/mltreelib-core/src/leaf.rs` exposes `Irls`, `Lbfgs`, `Ccd`, `CmaEs`, `Shade`, `NaturalGradient`, `GeneticAlgorithm`. These are leaf-level choices. They are not a general optimization interface.

`fit_model` in `solver/mod.rs` routes on family, penalty, and monotone flags. It rejects L1 for non-Gaussian families with:

```text
L1 regularization currently supported for Gaussian family. Use L2 with GLM families.
```

No current core solver accepts an arbitrary bounded scalar objective. That gap is the reason a separate generic module is required.

---

## 3. Simulated Annealing

### 3.1 Mechanics

Simulated annealing is a trajectory method over a single current point. Each step proposes a neighbour, evaluates the objective, and accepts the proposal with the Metropolis probability

\[
P(\text{accept}) =
\begin{cases}
1, & f(x_{\text{new}}) \le f(x_{\text{current}}),\\
\exp\left(-\frac{f(x_{\text{new}}) - f(x_{\text{current}})}{T}\right), & \text{otherwise},
\end{cases}
\]

where the temperature $T$ decreases over the run. A geometric cooling schedule $T_{k+1} = \alpha T_k$ with $\alpha \in (0,1)$ is the standard choice. Kirkpatrick, Gelatt, and Vecchi introduced the method for combinatorial optimization; Geman and Geman established the convergence conditions for the Markov chain. Bertsimas and Tsitsiklis give the modern probabilistic treatment.

Kirkpatrick, Gelatt, and Vecchi formulated simulated annealing for combinatorial state spaces. A continuous bounded version is not that algorithm by default. It needs four defined choices:

1. **Proposal distribution.** How a neighbour is drawn inside the bounds.
2. **Cooling schedule.** How temperature decreases.
3. **Boundary policy.** How a proposal that leaves the box is handled.
4. **Acceptance rule.** The Metropolis test above.

Two properties matter for the package:

- The global best point must be tracked separately from the current point. A cooling schedule that ends too hot leaves the final iterate worse than the best found.
- The temperature scale must be expressed in units of the objective. A fixed temperature is meaningless across objectives with different scales. Derive an initial temperature from observed objective spread, or accept a user-supplied one.

Boundary policy deserves care. Clipping a proposal onto a bound piles proposal probability mass on bound-active points, which biases the search toward bounds. Resampling or reflection preserves the intended proposal better. Whichever policy ships must be documented, because it changes the search behaviour.

### 3.2 Fit

Good fit for a generic black-box bounded objective:

- No derivatives needed.
- Handles non-smooth and discontinuous objectives.
- Simple to implement and to reason about.
- One incumbent point, so memory is trivial.

Weaknesses:

- Slow convergence compared to population methods on many problems.
- Strongly objective-scale sensitive through the temperature.
- Provides no global optimality guarantee in practical schedules. Hajek showed global convergence for logarithmic cooling, which is far too slow for normal use. Any shipped geometric schedule is a heuristic without a guarantee.

### 3.3 Package decision

Implement a bounded classical simulated annealing:

- Geometric cooling with an explicit `initial_temperature`, `cooling_factor`, and `min_temperature`.
- Default initial temperature derived from the spread of objective values seen during the first accepted evaluations, or from a user-supplied value.
- Uniform random neighbour inside the bounds, or a per-dimension step scale.
- Track best point, best value, iteration count, and evaluation count.
- Deterministic given `seed`.
- Stop on budget exhaustion, minimum temperature, or a stagnation counter.

Do not implement dual annealing. It adds a generalized visiting distribution, a reannealing schedule, and a local-search inner loop for a marginal gain over classical annealing in this setting.

### 3.4 Use cases in `mltreelib`

- Threshold and split-level calibration when the objective is non-smooth.
- Anomaly threshold selection.
- Cluster-count and initialization search with a non-smooth objective.
- Feature-subset screening with a discontinuous objective.

---

## 4. Basin Hopping

### 4.1 Mechanics

Basin hopping is a two-phase method. Each iteration performs three steps:

1. Perturb the current point $x$ to a proposal $y$, usually by a bounded random displacement.
2. Run a local minimizer from $y$ to reach a local minimum $z$.
3. Accept $z$ through a Metropolis test at constant temperature, so a slightly worse minimum can be entered:

\[
P(\text{accept}) = \min\left(1,\ \exp\left(-\frac{f(z) - f(x)}{T}\right)\right).
\]

Wales and Doye describe the current form. Li and Scheraga give the earlier Monte Carlo minimization form. The best minimum found is retained separately from the current iterate.

Basin hopping is therefore a composition of three separable parts:

| Part | Package component |
|---|---|
| Local minimization | `L-BFGS-B` (item 1 in the build order) |
| Perturbation | Bounded random displacement, scaled per dimension |
| Acceptance | Constant-temperature Metropolis test |

Monotonic basin hopping, which accepts only strict improvements, is a documented special case at $T = 0$. It is simpler and often adequate. Support it as an option rather than a separate method.

### 4.2 Fit

Strengths:

- Simple structure, about 100 lines once the local minimizer exists.
- Effective on rugged landscapes with many separated local minima.
- The returned point is always a local minimum of the objective, which is a property population methods do not give.

Weaknesses:

- Quality depends almost entirely on the local minimizer, the perturbation size, and the temperature. A bad choice in any of the three ruins the run.
- Requires many more objective evaluations than a local method, because every perturbation is followed by a full local solve.
- No global optimality guarantee. SciPy documentation is explicit that repeated runs from different starts are the only consistency check available.

A 2024 comparison of basin hopping against differential evolution, PSO, and CMA-ES on the BBOB suite and two cluster-energy problems found basin hopping roughly comparable to CMA-ES on synthetic functions and better on the two hard real-world problems. That result supports using basin hopping as a cheap, robust default global search, not as a guaranteed winner.

### 4.3 Package decision

Implement basin hopping only after a bounded local minimizer exists. Required semantics:

- Bounded perturbation, scaled by the per-dimension bound width.
- Local minimizer supplied by the caller through the shared optimizer interface, so the same method works with L-BFGS-B, CCD, or a future method.
- Constant-temperature Metropolis acceptance, plus a monotonic mode.
- Adaptive step size targeting a configurable acceptance rate, or a fixed step size with a documented default.
- Best-incumbent tracking across iterations, including across local solves that fail.
- Evaluation budget counts objective evaluations in both the local solves and the acceptance step.
- Explicit status when the budget ends before the iteration limit.

### 4.4 Use cases in `mltreelib`

- Hyperparameter and model-structure search over rugged objectives.
- Ensemble weight search with a non-convex objective.
- Tree-structure search where each local solve is a greedy tree fit.

---

## 5. Nonlinear Constraint Solver

### 5.1 Options compared

| Method | Constraints | Derivatives | Implementation cost in pure Rust | Fit for a generic black-box objective |
|---|---|---|---|---|
| **COBYLA** | Nonlinear inequality only | None | Moderate. Linear models of objective and constraints, a small LP per iteration, trust-region radius that only shrinks. | Good |
| **COBYQA** | Linear and nonlinear inequality, bounds always respected | None | High. Derivative-free trust-region SQP with underdetermined quadratic interpolation. | Good but expensive to build |
| **SLSQP** | Equality and inequality, plus bounds | Objective gradient optional, constraint Jacobian optional | Very high. Sequential quadratic programming with a working-set QP subproblem. | Weak for a black-box objective |
| **trust-constr** | Equality, inequality, bounds, second-order | Gradients and Hessian or Hessian-vector products | Highest. Interior-point or Byrd-Omojokun SQP switch. | Weak for a black-box objective |

SciPy exposes all four. `COBYLA` uses the PRIMA implementation of Powell's method. `COBYQA` is the derivative-free trust-region SQP method designed to supersede `COBYLA`, and it keeps bounds respected throughout the optimization.

### 5.2 Why COBYLA first

- It matches the generic objective seam. Only function values are needed, so the same `Objective` trait serves unconstrained and constrained problems.
- It supports arbitrary nonlinear inequality constraints. Bounds are expressible as constraints, so a single constraint model covers both.
- It has a compact reference implementation and a long history, so a Rust port can be cross-validated against PRIMA.
- A pure-Rust `cobyla` crate already exists and can serve as a test oracle.

The limitation is real and must be stated in the public API: COBYLA does not support equality constraints. Equality constraints must be handled either as two opposite inequality constraints with a tolerance, which weakens the guarantee, or rejected with a clear error.

### 5.3 Why SLSQP is later, not rejected

SLSQP is the method to reach for when a user needs genuine equality constraints, because it handles them natively. When that need is real, the cost is a sequential quadratic programming implementation with a working-set QP subproblem. That is a substantial build, and it must not delay the first three items.

Recommendation: keep the constraint model general from the start, with `ConstraintKind::Equality` and `ConstraintKind::Inequality`, so adding SLSQP later does not force an API break. Implement only the inequality path in COBYLA. Return an explicit unsupported error for equality constraints in that method.

### 5.4 Constraint model

Define once, in the shared seam:

```text
Constraint
  kind: Equality | Inequality
  evaluate: x -> value
  jacobian: optional x -> row of partial derivatives
```

Semantics:

- Inequality constraint: feasible when `value >= 0`.
- Equality constraint: feasible when `value == 0` within tolerance.
- Box bounds live in `Bounds`, not in the constraint list. A method that supports bounds natively uses `Bounds` directly.
- Constraint evaluation failures are distinct from objective failures. A hard constraint failure aborts; a soft objective rejection returns a non-finite value.
- Every constrained result reports the maximum constraint violation. A result with a nonzero violation is not silently presented as feasible. SciPy's `trust-constr` exposes this as `constr_violation`.
- The policy for evaluating the objective at infeasible points must be documented per method. COBYLA, SLSQP, and `trust-constr` may evaluate outside the feasible region during iterations. COBYQA treats bounds as unrelaxable and never leaves them.
- Equality constraints need an explicit per-method story. COBYLA is inequality-oriented, so an equality must become two opposite inequalities with a tolerance, which weakens the guarantee. COBYQA, SLSQP, and `trust-constr` handle equalities natively.

COBYLA also needs an absolute feasibility tolerance, `catol` in SciPy terms, so near-feasible points are accepted. Powell himself reports no favourable convergence theory for COBYLA, so treat it as a robust practical method, not a provably convergent one.

---

## 6. Shared Seam Requirements

Every method above depends on the same contract. The full design lives in `2026-11-21-general-solver-api-roadmap.md`. The parts that constrain these three methods specifically:

**Objective**

```rust
pub trait Objective {
    fn dimension(&self) -> usize;
    fn bounds(&self) -> &Bounds;
    fn evaluate(&self, x: &[f64]) -> Result<f64, SolverError>;
}
```

**Solve status.** Distinguish converged, iteration limit reached, evaluation budget exhausted, line search failed, constraint infeasible, and hard error. Never report success after budget exhaustion. SciPy's `OptimizeResult` reports `success` and a `message`; the package needs the same split plus explicit budget states.

**Evaluation budget.** Every objective call counts, including calls inside a local solve, inside a perturbation step, and inside a migration or archive update. A basin hopping run that reports 100 iterations but 40,000 objective evaluations must say so.

**Determinism.** Equal seed, bounds, objective, and options must produce identical output on a single thread. Use the seeded `rand` generators already in the dependency set. No wall-clock decisions inside the solver loop.

**Bounds.** Every returned point is finite and inside bounds. Methods that cannot support infinite bounds return an explicit unsupported error. Perturbation and annealing proposals are generated inside bounds, never clipped silently from outside.

**Missing data.** Objectives receive numeric vectors. Missing-value semantics live in the adapter that builds the objective. The optimizer never imputes.

---

## 7. Implementation Order and Dependencies

### Phase 0 — Shared seam

1. `Objective`, `SmoothObjective`, `Bounds`, `Constraint`, `ConstraintKind`.
2. `OptimizationResult`, `SolveStatus`, evaluation and iteration budget.
3. `minimize(request)` routing entry point.
4. Deterministic seed handling and bounds validation.
5. Feasibility reporting: maximum constraint violation on every constrained result.

Nothing else starts before this lands. Without it, each new method reimplements bounds, budget, and status.

### Phase 1 — Random Search and Latin Hypercube

1. Deterministic uniform sampling inside bounds.
2. Stratified sampling with deterministic shuffle.
3. Best incumbent, evaluation count, seeded reproducibility.

This baseline lands before any global metaheuristic. Without it, there is no honest way to claim that simulated annealing or basin hopping adds value.

### Phase 2 — L-BFGS-B

1. Port the L-BFGS-B iteration as defined by Byrd, Lu, Nocedal, and Zhu (1995). Compute the generalized Cauchy point along the piecewise-linear projected-gradient path to identify the active set, then approximately minimize the quadratic model over the free variables using the compact limited-memory BFGS representation, then apply a line search. Use the existing `lbfgs.rs` two-loop recursion as the curvature memory base. Do not substitute a projected-gradient method with L-BFGS preconditioning.
2. Projected stopping test on the gradient norm over free variables.
3. Keep unbounded behaviour identical to the current `lbfgs.rs` result on the same objective.
4. Cross-validate against the original Fortran implementation and against the `basin` crate.

This is next because it is the smooth bounded local solver, it is the inner minimizer basin hopping needs, and it is the trustworthy baseline the population and trajectory methods must beat.

### Phase 3 — Bounded simulated annealing

1. Metropolis acceptance with geometric cooling.
2. Bounded proposal generation with a documented boundary policy.
3. Best-incumbent tracking, budget accounting, deterministic seeding.
4. Test against a known multimodal toy objective with a known global minimum, and against a fixed-seed reference run.
5. Benchmark against Phase 1 Random Search. If annealing cannot beat random search within the same evaluation budget on package workloads, keep it internal.

### Phase 4 — COBYLA

1. Linear model of objective and constraints over the interpolation set.
2. Trust-region radius that shrinks from `rhobeg` to `rhoend`.
3. Subproblem linear program over the linear models with non-negative constraint values.
4. Feasibility tolerance, bounds as constraints, explicit rejection of equality constraints.
5. Cross-validate against PRIMA, not only against Powell's original 1994 algorithm. SciPy replaced the original implementation with the bug-fixed PRIMA version in 1.16. The pure-Rust `cobyla` crate is also usable as an oracle.

### Phase 5 — Basin hopping

1. Perturbation, local solve, Metropolis acceptance, all built on the L-BFGS-B and annealing components.
2. Monotonic mode.
3. Adaptive step size toward a target acceptance rate.
4. Best incumbent across iterations, including failed local solves.
5. Benchmark against Random Search and annealing under equal evaluation budgets. The inner local solve consumes most of the budget, so compare on evaluations, not iterations.

### Phase 6 — Only with evidence

- SLSQP for equality constraints.
- COBYQA as a better-constrained successor to COBYLA.
- BFGS and Polak-Ribiere nonlinear CG as leaf-level competitors to L-BFGS, benchmarked on identical GLM objectives.
- NSGA-II for multiobjective search.
- Island orchestration over the population methods.

---

## 8. Dependency and Reuse Decision

Pure-Rust options already exist:

| Crate | Coverage | License | Downloads | Note |
|---|---|---|---|---|
| `basin` | L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, COBYLA, SLSQP, simulated annealing, basin hopping, barrier and augmented-Lagrangian adapters, first-class constraints | Apache-2.0 and MIT | roughly 30,000 in three months, current version 3.x | Broadest single match. Its paper reports the Powell-family solvers were developed against PRIMA and cross-validated against it, and L-BFGS-B was checked against the original Fortran code. |
| `cobyla` | COBYLA only | crates.io reports non-standard; repository states MIT | 364,000 total, roughly 36,000 recent | Pure Rust, transpiled from NLopt C code. Usable as a test oracle. Confirm license before any use beyond local testing. |
| `slsqp` | SLSQP only | crates.io reports non-standard; repository states MIT | 79,000 total, roughly 36,000 recent | Pure Rust, transpiled from NLopt C code. Deferred oracle for a future equality-constraint method. |
| `good_lp` | Linear and mixed-integer programming | MIT | 4.5M total, roughly 548,000 recent | Considered for the COBYLA linear subproblem. Rejected for the core crate because it pulls a solver stack. `nalgebra` and `faer` cover the subproblem. |
| `nlopt` | Broad catalog through a C binding | MIT | established | Requires a C toolchain. Rejected by the package stance on lean pure-Rust builds. |
| `gomez` | Derivative-free methods and nonlinear least squares | MIT | smaller | Supports box constraints only. |
| `argmin` | Broad numerical framework | MIT | established | Design reference for the executor and state model. |

Decision: **do not add any of these as runtime dependencies of `mltreelib-core`.** Reasons:

1. `mltreelib-core` already ships its own L-BFGS, CCD, and evolutionary solvers with package-specific semantics: `GlmFamily`, monotone projection, missing-value handling, and `LeafControl` integration. Adopting a second framework duplicates that surface and forces an adapter layer over every call site.
2. The package requires deterministic seeded behaviour, explicit evaluation budgets, and specific stop reasons. Those live in `mltreelib`'s shared seam, so a third-party solver would need a wrapper for each of its semantics anyway.
3. The package stance is a lean Rust core. A broad optimization framework with its own linear-algebra tier, state traits, and observer layer is a large surface to adopt.

Use `basin`, `cobyla`, and `slsqp` as **dev-dependencies and test oracles** instead. Cross-validating a Rust port against a reference implementation is the cheapest way to catch a sign error in a trust-region or line-search update. This matches the practice already visible in the `basin` project itself, which validated its Powell-family solvers against PRIMA and its L-BFGS-B against the original Fortran code. The `cobyla` and `slsqp` crates report non-standard licenses on crates.io while their repositories state MIT, so confirm licensing before any use beyond local testing.

Revisit this decision if the generic module grows past the roadmap scope, or if a hard requirement appears for SQP-level constraint handling that the package cannot implement well.

---

## 9. Risks

1. **False completeness.** Three global and constrained methods look like a full optimizer suite. They are not. The package has no general constrained solver with equality constraints, and no multiobjective method until NSGA-II lands.
2. **Temperature and step-size sensitivity.** Simulated annealing and basin hopping both look parameter-light and are not. Defaults tuned on one objective will mislead on another. Expose the parameters and document their meaning in objective units.
3. **Evaluation accounting drift.** Basin hopping nests local solves. If the inner solve does not report its evaluations to the outer budget, reported costs become fiction.
4. **Bound violations.** Clipping a perturbation into bounds changes the proposal distribution. Generate inside bounds instead.
5. **Constraint sign confusion.** COBYLA convention is feasibility when the constraint value is non-negative. A package convention of the opposite sign would silently invert feasibility. Pick one convention and test it.
6. **COBYLA equality misuse.** Encoding an equality as two inequalities gives an approximate answer with a weaker guarantee. Reject equality constraints explicitly instead.
7. **Cross-validation gap.** A hand-written Rust port can agree with itself and still be wrong. Keep a reference-oracle test for every numeric method.
8. **Bundle overlap.** The existing evolutionary solvers already cover much of the global-search role. Adding basin hopping and annealing gives redundancy, not coverage, unless the shared seam makes each method cheap to add.
9. **One multi-core job at a time.** Population methods multiply evaluations. The package rule against concurrent multi-core runs still applies.

---

## 10. Verification Requirements

**Simulated annealing**

- Deterministic output for a fixed seed.
- Best incumbent is never worse than the final iterate.
- Returned point is inside bounds.
- Known multimodal toy objective reaches the known global minimum within a documented budget.
- Evaluation count matches the number of objective calls.

**COBYLA**

- Feasible solution on a known constrained toy problem, verified independently.
- Equality constraints return an explicit unsupported error.
- Infeasible start returns a documented status, not a panic.
- Returned point is inside bounds and satisfies every inequality constraint within `catol`.
- Cross-validation against PRIMA or the `cobyla` crate on fixed problems.

**Basin hopping**

- Deterministic output for a fixed seed.
- Best incumbent is a local minimum under the supplied local solver.
- Local solve failures do not corrupt the incumbent.
- Evaluation budget counts inner local-solve evaluations.
- Returned point is inside bounds.

**Shared seam**

- Bounds validation rejects `lower > upper` and non-finite bounds.
- Methods that cannot support infinite bounds say so.
- Every status is reachable and tested.
- Equal seed, bounds, objective, and options give identical output.

---

## 11. Primary Sources

1. Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983). "Optimization by Simulated Annealing." *Science* 220(4598), 671-680. [DOI](https://doi.org/10.1126/science.220.4598.671).
2. Geman, S. and Geman, D. (1984). "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images." *IEEE TPAMI* 6(6), 721-741. [DOI](https://doi.org/10.1109/TPAMI.1984.4767270).
3. Bertsimas, D. and Tsitsiklis, J. (1993). "Simulated Annealing." *Statistical Science* 8(1), 10-15. [DOI](https://doi.org/10.1214/ss/1177011077).
4. Wales, D. J. and Doye, J. P. K. (1997). "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms." *J. Phys. Chem. A* 101(28), 5111-5116. [DOI](https://doi.org/10.1021/jp970984n).
5. Li, Z. and Scheraga, H. A. (1987). "Monte Carlo-minimization approach to the multiple-minima problem in protein folding." *PNAS* 84(19), 6611-6615. [DOI](https://doi.org/10.1073/pnas.84.19.6611).
6. Baioletti, M., Santucci, V., and Tomassini, M. (2024). "A Performance Analysis of Basin Hopping Compared to Established Metaheuristics for Global Optimization." *Journal of Global Optimization*. [arXiv:2403.05877](https://arxiv.org/abs/2403.05877). [DOI](https://doi.org/10.1007/s10898-024-01373-5).
7. Powell, M. J. D. (1994). "A Direct Search Optimization Method that Models the Objective and Constraint Functions by Linear Interpolation." In *Advances in Optimization and Numerical Analysis*, 51-67. [DOI](https://doi.org/10.1007/978-94-015-8330-5_4).
8. Powell, M. J. D. (1998). "Direct Search Algorithms for Optimization Calculations." *Acta Numerica* 7, 287-336. [DOI](https://doi.org/10.1017/S0962492900002834).
9. Zhang, Z. (2023). "PRIMA: Reference Implementation for Powell's Methods with Modernization and Amelioration." [DOI](https://doi.org/10.5281/zenodo.8052654).
10. Ragonneau, T. M. and Zhang, Z. (2024). "PDFO: A Cross-Platform Package for Powell's Derivative-Free Optimization Solvers." *Mathematical Programming Computation*. [arXiv:2302.13246](https://arxiv.org/abs/2302.13246). [DOI](https://doi.org/10.1007/s12532-024-00257-9).
11. Ragonneau, T. M. (2022). "Model-Based Derivative-Free Optimization Methods and Software." PhD thesis, Hong Kong Polytechnic University. [Record](https://theses.lib.polyu.edu.hk/handle/200/12294).
12. Kraft, D. (1988). "A Software Package for Sequential Quadratic Programming." DFVLR-FB 88-28. [Report PDF](https://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf).
13. Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). "A Limited Memory Algorithm for Bound Constrained Optimization." *SIAM J. Sci. Comput.* 16(5), 1190-1208. [DOI](https://doi.org/10.1137/0916069).
14. Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). "Algorithm 778: L-BFGS-B: Fortran Subroutines for Large-Scale Bound-Constrained Optimization." *ACM TOMS* 23(4), 550-560. [DOI](https://doi.org/10.1145/279232.279236).
15. Larsson, J. (2026). "Basin: Efficient and Extensible Numerical Optimization in Rust." [arXiv:2608.11279](https://arxiv.org/abs/2608.11279).
16. SciPy. "scipy.optimize.minimize." [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
17. SciPy. "scipy.optimize.basinhopping." [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html).
18. SciPy. "scipy.optimize.fmin_cobyla." [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cobyla.html).
19. COBYQA project documentation. [Official documentation](https://www.cobyqa.com/).
20. `basin` crate. [crates.io](https://crates.io/crates/basin). [Repository](https://github.com/jolars/basin).
21. `cobyla` crate. [crates.io](https://crates.io/crates/cobyla). [Repository](https://github.com/relf/cobyla/).
22. Hajek, B. (1988). "Cooling Schedules for Optimal Annealing." *Mathematics of Operations Research* 13(2), 311-329. [DOI](https://doi.org/10.1287/moor.13.2.311).
23. Powell, M. J. D. (2007). "A View of Algorithms for Optimization without Derivatives." Cambridge University Technical Report DAMTP 2007/NA03. [Report record](https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2007_03).
24. Ragonneau, T. M. (2022). "Model-Based Derivative-Free Optimization Methods and Software." PhD thesis, Hong Kong Polytechnic University. [arXiv:2210.12018](https://arxiv.org/abs/2210.12018).
25. Byrd, R. H., Hribar, M. E., and Nocedal, J. (1999). "An Interior Point Algorithm for Large-Scale Nonlinear Programming." *SIAM J. Optimization* 9(4), 877-900. [DOI](https://doi.org/10.1137/S1052623497325107).
26. `slsqp` crate. [crates.io](https://crates.io/crates/slsqp). [Repository](https://github.com/relf/slsqp/).
27. `good_lp` crate. [crates.io](https://crates.io/crates/good_lp). [Repository](https://github.com/rust-or/good_lp).

## 12. Source Notes

- Primary papers, official SciPy documentation, the COBYQA project documentation, and the `basin` and `cobyla` crate registries supplied the algorithm mechanics, the reference implementations, and the reuse options.
- alphaXiv supplied the `basin` Rust library paper and the 2024 basin-hopping performance comparison, plus recent basin-hopping and simulated-annealing candidates.
- The `basin` paper and its repository provided the cross-validation practice used as the reference standard: Powell-family solvers validated against PRIMA, L-BFGS-B validated against the original Fortran code.
- treg catalog exposed Perplexity, Brave, and Scholar endpoints, but live treg calls failed because the saved authentication expired. No claim in this note depends on a failed treg response.
- Current source was read directly for the baseline section. Where older notes disagree with current source, current source wins.
