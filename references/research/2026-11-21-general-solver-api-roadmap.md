# General Solver Interface and Implementation Roadmap for `mltreelib`

**Date:** 2026-11-21  
**Decision informed:** Which optimization algorithms should become public package capabilities, which should remain internal leaf solvers, and what interface should support future tree research without becoming a full-scale optimization platform.  
**Recommendation:** Build one small general optimization module. Reuse existing algorithms through adapters. Add six algorithms, not twenty.

## 1. Decision

`mltreelib` should expose a **general-purpose, bounded optimization module** for research and advanced construction workflows. It should not try to match SciPy, never-ending optimizer, or commercial solver suites.

Public portfolio:

| Method | Public status | Main `mltreelib` use |
|---|---|---|
| Random Search + LHS | Add now | Baseline, initialization, robust sampling |
| PSO | Add now | Bounded hyperparameter and leaf-parameter tuning |
| DE/rand/1/bin | Add now | Continuous black-box search and comparison baseline |
| CMA-ES + restart option | Refactor and expose | Ill-conditioned continuous tuning |
| Real-Coded GA | Refactor and expose | Mixed/discrete encodings and tree structure search |
| SHADE | Refactor and expose | Adaptive DE baseline |
| L-BFGS-B | Add now | Smooth bounded leaf and hyperparameter optimization |
| CCD / proximal CCD | Refactor and expose | Sparse and constrained leaf fitting |
| Bounded simulated annealing | Add after Random Search and L-BFGS-B | Derivative-free global search over non-smooth objectives |
| COBYLA | Add after annealing | One derivative-free nonlinear inequality constraint solver |
| Basin hopping | Add after L-BFGS-B | Global search composed of perturbation, local solve, and Metropolis acceptance |
| NSGA-II | Add after base interface | Accuracy, complexity, calibration, and latency trade-offs |
| Island model | Add after multiobjective support | Long searches and heterogeneous populations |

See `2026-11-21-generic-global-and-constrained-solvers.md` for the full comparison, dependency order, and constraint-model decision behind the simulated annealing, COBYLA, and basin hopping entries.

Do not expose every proposed algorithm as separate public concept. Public interfaces should express capabilities, not papers.

## 2. Product Principle

`mltreelib` needs optimization because trees create many small optimization problems:

- Leaf coefficient fitting.
- Split thresholds and categorical partitions.
- Hyperparameter search.
- Feature and interaction subset search.
- Ensemble weight and tree selection.
- Anomaly threshold calibration.
- Clustering initialization and model selection.
- Accuracy-versus-complexity Pareto search.

Optimization should therefore be a **core research module**, not a Python-only convenience. Rust core owns numerical behavior and deterministic reproducibility. Python exposes callable-objective convenience.

Scope target: **good-enough, composable optimization for tree research**. Not production-scale global optimization.

## 3. Why Not Expose All Twenty

The audit found only 5 of 20 named algorithms fully present, 2 partial, and 13 absent. More importantly, current solvers fit `DesignMatrix` GLM coefficients. They do not support arbitrary bounded objectives, mixed encodings, objective vectors, or evaluation budgets.

Adding twenty algorithms directly would create twenty shallow interfaces with duplicated control flow. It would also imply unsupported scalability and global-optimality claims.

Use three layers instead:

1. **General optimization module** — bounded black-box and smooth objectives.
2. **Model-specific adapters** — GLM leaves, tree hyperparameters, clustering, anomaly scores.
3. **Research workflows** — islands, multiobjective search, robust ensembles.

This preserves depth: one interface supports many algorithms and many callers.

## 4. Public Interface

### 4.1 Rust interface

Keep public surface small:

```rust
pub trait Objective {
    fn dimension(&self) -> usize;
    fn bounds(&self) -> &Bounds;
    fn evaluate(&self, x: &[f64]) -> Result<f64, SolverError>;
}

pub trait SmoothObjective: Objective {
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), SolverError>;
}

pub enum OptimizerKind {
    RandomSearch,
    LatinHypercube,
    Pso,
    DifferentialEvolution,
    Shade,
    CmaEs,
    GeneticAlgorithm,
    Lbfgs,
    LbfgsB,
    Ccd,
    Nsga2,
}

pub struct OptimizationRequest<'a> {
    pub objective: &'a dyn Objective,
    pub smooth: Option<&'a dyn SmoothObjective>,
    pub start: Vec<f64>,
    pub optimizer: OptimizerKind,
    pub options: SolverOptions,
}

pub struct OptimizationResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: SolveStatus,
}
```

One `minimize(request)` function should route algorithms. Callers should not learn algorithm-specific control flow.

`SolverOptions` should contain shared, meaningful controls:

- `max_iterations`
- `max_evaluations`
- `seed`
- `population_size`
- `tolerance`
- `initial_population`
- algorithm-specific nested options only when callers need them

Avoid one giant flat options struct. Use nested method-specific options or private defaults.

### 4.2 Bounds

Use one bounds representation:

```rust
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
```

Rules:

- `lower[i] <= upper[i]`.
- Infinite bounds allowed only where method supports them.
- Every returned point must be finite and within bounds.
- Unbounded methods report unsupported-bound errors instead of silently clipping.
- Initial points outside bounds are projected or rejected according to documented method behavior.

### 4.3 Python interface

Expose one callable-objective function:

```python
result = optimize(
    objective,
    x0,
    bounds,
    method="pso",
    max_iterations=1000,
    max_evaluations=20_000,
    seed=42,
    options=None,
)
```

`objective` receives a NumPy-like vector and returns one float.

Multiobjective mode should not overload scalar return with an array. Use a separate method or explicit `vector=True` path returning a Pareto result. Better: `minimize_many(objective, x0, bounds, ...)` for vector objectives.

Python must expose:

- Best vector.
- Best scalar objective.
- Evaluation count.
- Iteration count.
- Termination status.
- Deterministic seed.
- Optional convergence history.
- Pareto set for multiobjective methods.

## 5. Internal Organization

```text
solver/
  mod.rs                 public minimize interface and shared types
  objective.rs           Objective, SmoothObjective, Bounds
  result.rs              OptimizationResult, SolveStatus
  budget.rs              evaluation and iteration budgets
  population.rs          shared population state and migration
  black_box/
    random.rs            random search and LHS
    pso.rs
    de.rs                DE/rand/1/bin
    shade.rs             existing SHADE moved or wrapped
    cmaes.rs             existing CMA-ES moved or wrapped
    ga.rs                existing RCGA moved or wrapped
  smooth/
    lbfgs.rs             existing L-BFGS
    lbfgs_b.rs
    ccd.rs               existing CCD
  multiobjective/
    nsga2.rs
  adapters/
    glm.rs               DesignMatrix objective adapter
```

Prefer moving implementations only when it improves locality. Do not create wrapper modules that merely forward calls without adding behavior.

## 6. Algorithm Recommendations

### 6.1 Add now

#### Random Search and LHS

Random Search gives every future optimizer an honest baseline. LHS improves coverage when initial populations matter.

Use cases:

- Hyperparameter baseline.
- CMA-ES, DE, GA, and PSO initialization.
- Sparse feature-subset screening.
- Benchmarking a new objective before algorithm tuning.

Implementation must stratify dimensions, shuffle strata deterministically, and reproduce identical samples for equal seeds.

#### PSO

PSO provides strong continuous black-box search with little code. It complements CMA-ES:

- PSO explores broadly.
- CMA-ES adapts covariance for rotated landscapes.
- DE handles multimodal and noisy landscapes.
- GA handles mixed encodings.

Support global-best and ring/local topologies. Start with inertia-weight PSO and explicit velocity clamping.

#### DE/rand/1/bin

Plain DE is required as a transparent baseline and test oracle for SHADE. It is also easy to implement correctly.

Expose:

- Population size.
- Mutation factor.
- Crossover probability.
- Bounds handling.
- Optional external archive.

Do not alias plain DE to SHADE. `de` and `shade` must be distinct methods.

#### L-BFGS-B

Smooth bounded optimization belongs in the general module. L-BFGS-B enables:

- Bounded leaf coefficients.
- Hyperparameters with natural ranges.
- Penalty or calibration parameter fitting.
- Trustworthy smooth baseline against PSO and DE.

Implement the L-BFGS-B iteration as defined by Byrd, Lu, Nocedal, and Zhu (1995). Each iteration computes the generalized Cauchy point along the piecewise-linear projected-gradient path to identify the active set, approximately minimizes the quadratic model over the free variables with the compact limited-memory BFGS representation, then applies a line search. Do not substitute a projected-gradient method with L-BFGS preconditioning. Keep unbounded L-BFGS behavior compatible with current implementation.

### 6.2 Refactor and expose existing algorithms

#### CMA-ES

Existing CMA-ES already adapts mean, covariance, and step size. Expose it through general objective interface and add:

- Bound handling.
- Deterministic initialization.
- Evaluation budget.
- Incumbent preservation.
- Optional IPOP restart as an option, not separate public method.

IPOP should increase population after stagnation while preserving global incumbent. Restart count and population schedule belong in CMA-ES options.

#### Real-Coded GA

Existing RCGA already uses tournament selection, SBX, polynomial mutation, and elitism. Generalize encoding through adapters:

- Continuous vector.
- Integer vector.
- Categorical vector.
- Binary feature selection.
- Tree-structure encoding.

Keep encoding logic outside GA implementation. GA must operate on decoded candidate vectors and a shared fitness interface.

#### SHADE

Expose existing SHADE as adaptive DE. Keep DE and SHADE distinct. Shared mutation, archive, and population machinery should live in one module.

#### CCD and proximal CCD

CCD already implements active-set cyclic updates and L1 soft thresholding. General interface should expose it for bounded sparse objectives. Random-coordinate selection can become a CCD option later.

### 6.3 Add second wave

#### NSGA-II

NSGA-II has strong product value because tree research often has conflicting objectives:

- Predictive error versus tree size.
- Accuracy versus inference latency.
- Calibration versus coverage.
- Reconstruction error versus sparsity.
- Clustering compactness versus separation.

Implement nondominated sorting, crowding distance, elitist survivor selection, SBX, and polynomial mutation. Return Pareto archive and crowding metrics.

Do not implement MOEA/D until real workloads need four or more objectives. NSGA-II covers the common case with less machinery.

#### Island model

Implement islands as orchestration above population methods, not inside one algorithm.

Features:

- Number of islands.
- Ring or complete topology.
- Migration interval.
- Elite count.
- Per-island algorithm and seed.
- Global incumbent archive.
- Optional asynchronous execution later.

Serial deterministic islands must come first. Package rule forbids concurrent multi-core jobs unless scheduler controls oversubscription.

## 7. Explicit Non-Priorities

| Algorithm or feature | Decision | Reason |
|---|---|---|
| jDE as separate public method | Defer | SHADE already covers adaptive DE; per-individual mode can be later option |
| MOEA/D | Defer | NSGA-II covers common multiobjective case |
| Trust-region Newton / Steihaug CG | Defer | Large numerical burden; no current tree product need |
| Nonlinear CG | Defer | L-BFGS covers smooth large problems |
| SGD with momentum | Defer | Poor fit for deterministic small-leaf fits |
| AdamW | Reject for core | Neural optimizer assumptions do not fit package |
| General-purpose random-coordinate descent | Later option | Useful diagnostic, low differentiation |
| Fully scalable parallel optimizer platform | Reject | Contradicts package scope and maintenance budget |

## 8. Important Semantics

### Evaluation budget

Every stochastic method must report and enforce objective evaluation count. Migration, archive evaluation, and warm-start evaluation count must be unambiguous.

### Failure status

Statuses should distinguish:

- Converged.
- Iteration limit.
- Evaluation limit.
- Invalid objective value.
- Infeasible bounds.
- Numerical failure.
- Unsupported method or objective.
- User cancellation, if later supported.

Do not report success when stopping after budget exhaustion.

### Determinism

Equal seed, bounds, objective, options, and single-thread execution must produce identical output. Document any methods that cannot guarantee this.

### Constraints

Do not hide constraints inside penalty functions by default. Represent simple bounds directly. Expose penalty-based constraints as caller-defined objective behavior.

### Missing data

Objectives receive numeric vectors. Missing-data semantics belong in adapters and objective construction. Optimization module must not impute silently.

## 9. Use Cases and Expected Payoff

| Use case | Recommended method | Payoff |
|---|---|---|
| Leaf coefficient fitting, smooth bounded | L-BFGS-B | Better constrained leaf models |
| Sparse leaf fitting | CCD / proximal CCD | Existing strength exposed cleanly |
| Hyperparameter tuning | Random/LHS baseline, PSO, DE, CMA-ES | Strong research workflow without external optimizer dependency |
| Noisy objective | DE | Robust population search |
| Rotated continuous objective | CMA-ES | Covariance adaptation |
| Mixed discrete-continuous encoding | GA | Tree and feature search |
| Accuracy-complexity-latency frontier | NSGA-II | Pareto model selection |
| Long multimodal search | Islands | Diversity and restart resilience |
| Split or threshold tuning | Random/LHS, DE, PSO | Research beyond greedy splits |

This makes solver module useful beyond leaf fitting while keeping implementation moderate.

## 10. Delivery Sequence

### Phase 1 — Public core

1. Define `Objective`, `SmoothObjective`, `Bounds`, `OptimizationRequest`, `OptimizationResult`, and `SolveStatus`.
2. Add evaluation budget and deterministic seed semantics.
3. Add Random Search and LHS.
4. Add Python `optimize` and `minimize_many` interfaces.
5. Adapt existing CCD, L-BFGS, CMA-ES, SHADE, and RCGA without changing numerical behavior.

### Phase 2 — Missing high-value algorithms

1. Add PSO.
2. Add DE/rand/1/bin.
3. Add L-BFGS-B.
4. Add bound-aware CMA-ES with IPOP restart option.

### Phase 3 — Research breadth

1. Add NSGA-II.
2. Add generic encoding adapters for binary, integer, and categorical vectors.
3. Add serial island model.
4. Add model-specific adapters for tree, anomaly, and clustering objectives.

### Phase 4 — Evidence-driven expansion

Add jDE, MOEA/D, nonlinear CG, trust-region methods, or stochastic optimizers only after benchmark evidence shows current portfolio cannot solve a real package workload.

## 11. Acceptance Criteria

A public solver module is complete when:

- One Rust `minimize` interface supports scalar bounded objectives.
- One Python `optimize` function supports callable scalar objectives.
- Random Search, LHS, PSO, DE, SHADE, CMA-ES, GA, L-BFGS, L-BFGS-B, and CCD pass shared convergence and boundary tests.
- Every result reports evaluations, iterations, status, best objective, and best point.
- Equal seeds reproduce results.
- Bounds cannot be violated in returned solutions.
- Invalid objective values follow documented failure behavior.
- Multiobjective method returns a verified nondominated set.
- Existing GLM leaf behavior remains unchanged.
- Public docs state non-scalable research scope clearly.

## 12. Primary References

1. Storn, R. and Price, K. (1997). “Differential Evolution.” [DOI](https://doi.org/10.1023/A:1008202821328).
2. Tanabe, R. and Fukunaga, A. (2013). “Success-History Based Adaptive Differential Evolution.” [DOI](https://doi.org/10.1109/CEC.2013.6557555).
3. Hansen, N., Ostermeier, M., and Arnold, I. (2001). “The Covariance Matrix Adaptation Evolution Strategy.” Technical report, Technische Universität Berlin. No DOI.
4. Hansen, N. (2016). “The CMA Evolution Strategy: A Tutorial.” [arXiv:1604.00772](https://arxiv.org/abs/1604.00772).
5. Auger, A. and Hansen, N. (2005). “A Restart CMA Evolution Strategy.” [DOI](https://doi.org/10.1109/CEC.2005.1554904).
6. Kennedy, J. and Eberhart, R. (1995). “Particle Swarm Optimization.” [DOI](https://doi.org/10.1109/ICNN.1995.488968).
7. Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). “NSGA-II.” [DOI](https://doi.org/10.1109/4235.996017).
8. Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). “Algorithm 778: L-BFGS-B: Fortran Subroutines for Large-Scale Bound-Constrained Optimization.” *ACM TOMS* 23(4), 550-560. [DOI](https://doi.org/10.1145/279232.279236).
9. McKay, M. D., Beckman, R. S., and Conover, M. F. (1979). “A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code.” *Technometrics* 21(2), 239-245. [DOI](https://doi.org/10.1080/00401706.1979.10489755).
10. Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). “A Limited Memory Algorithm for Bound Constrained Optimization.” *SIAM J. Sci. Comput.* 16(5), 1190-1208. [DOI](https://doi.org/10.1137/0916069).

## 13. Relationship to Existing Research

This roadmap refines:

- `2026-11-21-optimization-solvers-implementation-landscape.md`
- `2026-11-21-anomaly-detection-implementation-landscape.md`
- `2026-11-21-clustering-implementation-landscape.md`
- `2026-11-17-modern-glm-solvers-and-optimization-landscape.md`

Current source wins where older notes conflict. In particular, L-BFGS, CCD, CMA-ES, SHADE, sNES, and RCGA already exist.
