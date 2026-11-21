# Optimization Solvers Implementation Landscape for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Tree optimization, GLM leaf fitting, convex clustering, first-order methods, constrained problems, and mixed-integer solvers usable in current `mltreelib`.  
**Decision informed:** Which solver capabilities belong in the Rust core and which require optional backends.  
**Confidence:** High for current repository state and classical solver fit. Medium for exact-tree scaling claims.

## 1. Executive Summary

Current solver module is broader than the 2026-11-17 GLM note suggests. It already contains:

- Exact Gaussian OLS and Ridge through normal equations and Cholesky.
- Elastic-net and monotone coordinate descent.
- L-BFGS for non-linear GLM deviance.
- Dense and family-aware solver paths.
- Evolutionary optimization: CMA-ES, SHADE, sNES, and genetic algorithms.

The stale statement “implement L-BFGS in `optim/lbfgs.rs`” must not be copied into new work. L-BFGS already exists at `crates/mltreelib-core/src/solver/lbfgs.rs`.

Highest-value solver additions:

1. **FISTA / proximal-gradient path for L1 GLM leaves.** Closes explicit non-Gaussian L1 limitation while reusing deviance gradients.
2. **Bounded exact subtree search.** DL8.5/GOSDT-style branch-and-bound can optimize small subtrees without an external solver.
3. **Convex clustering ADMM and tree-guided dynamic programming.** Adds convex cluster paths and a serious solver benchmark.
4. **Population-search expansion.** Add PSO, serial island orchestration, and NSGA-II around existing CMA-ES, SHADE, sNES, and GA.
5. **Frank-Wolfe simplex weights.** Sparse ensemble or leaf-mixture weights without projection.
6. **Optional mixed-integer backend.** Keep MIP and SAT exact trees outside default core.

No single universal solver should replace existing routing. Package already models different mathematical structure: dense linear systems, smooth deviance, nonsmooth penalties, and combinatorial tree choices.

---

## 2. Current Repository State

`crates/mltreelib-core/src/solver/mod.rs` routes work across:

| Solver | Problem structure | Current status |
|---|---|---|
| `normal` | Gaussian OLS/Ridge | Implemented |
| `ccd` | Elastic net and monotone constraints | Implemented |
| `lbfgs` | Smooth GLM deviance | Implemented |
| `dense` | Dense numerical solver support | Implemented |
| `family` | Distribution-family logic | Implemented |
| `evolutionary` | CMA-ES, SHADE, sNES, GA | Implemented |

Dependencies already available in core include `faer`, `nalgebra`, `statrs`, `rand`, `rayon`, and related utilities. Python package has no mandatory dependencies. Solver extras currently cover datasets, plotting, and tuning rather than an external numerical backend.

Known limitation from repository inspection: `fit_model` rejects L1 regularization for non-Gaussian families with the message:

```text
L1 regularization currently supported for Gaussian family
```

That is a concrete solver gap, not a speculative roadmap item.

---

## 3. Solver Taxonomy and Fit

### 3.1 Proximal-gradient solvers

For a smooth objective $f(\beta)$ plus an L1 penalty $g(\beta)$, proximal gradient performs

\[
\beta^{k+1} = \operatorname{prox}_{\alpha g}\left(\beta^k-\alpha\nabla f(\beta^k)\right).
\]

For L1, the proximal operator is soft thresholding:

\[
\operatorname{prox}_{\alpha\lambda\|\cdot\|_1}(v)_j
=\operatorname{sign}(v_j)\max(|v_j|-\alpha\lambda,0).
\]

**FISTA** adds momentum and $O(1/k^2)$ convergence for convex smooth-plus-proximal objectives.

**Package fit:** excellent for GLM leaves. Existing L-BFGS path already computes gradients and deviance. Add proximal handling around those gradients, a Lipschitz estimate with backtracking, and warm starts from ridge or CCD.

**Recommendation:** first new solver. It should support Binomial, Poisson, Gamma, and Tweedie once gradient and Hessian bounds are verified family by family. Add regression tests against reference solvers and finite-difference checks.

**Alternative:** OWL-QN handles L1 with L-BFGS by separating positive and negative variables. FISTA is easier to reason about when objective and gradient are already shared with current code.

### 3.2 Coordinate descent and proximal families

Coordinate descent already exists in `ccd`. It is ideal for sparse design matrices and exact L1 Gaussian leaves. Its weakness is slow convergence when features are strongly correlated.

**Covariance-updating coordinate descent** reduces repeated Gram-matrix work. **Block-coordinate descent** updates groups of features and can exploit grouped/hierarchical data.

**Package fit:** high for sparse or grouped leaves. Block-coordinate updates need explicit dependency grouping and deterministic ordering.

**Recommendation:** extend CCD only after FISTA provides non-Gaussian L1 support. Do not duplicate coordinate logic.

### 3.3 Convex clustering

Convex clustering is

\[
\min_\Theta \frac12\sum_i\|y_i-\theta_i\|_2^2
+\lambda\sum_{i<j}w_{ij}\|\theta_i-\theta_j\|_q.
\]

**ADMM** splits the problem into a data-fit step and a weighted fusion step. For sparse kNN weights, each iteration requires local weighted smoothers and a shrinkage-like fusion step.

**AMA** and second-order nonsmooth methods offer alternatives, but ADMM has the clearest implementation path.

**Package fit:** medium-high. `faer` can solve dense subproblems. Sparse graph weights are mandatory for scale. Full pairwise weights cost $O(n^2)$ memory.

**Recommendation:** implement small-data ADMM first, expose graph construction explicitly, and test against convex objective and cluster-path consistency. Then evaluate tree-guided fused-lasso dynamic programming, which fits the package's tree thesis better.

### 3.4 Tree-guided dynamic programming

Tree-Guided $L_1$-Convex Clustering decomposes the objective by coordinate and solves tree-structured fused lasso through dynamic programming. The 2025 paper reports a complete cluster path for $10^6$ two-dimensional points in about 60 seconds on a laptop and a constrained formulation without cluster splits.

**Package fit:** strong research fit. It avoids general quadratic weights and produces a hierarchy by design.

**Recommendation:** implement as a convex-clustering backend after ADMM baseline. Use dynamic programming for tree weights and ADMM for kNN or custom graph weights.

### 3.5 Frank-Wolfe and simplex optimization

Frank-Wolfe optimizes over a simplex without projection:

\[
s_t \in \arg\min_{s\in \Delta} \nabla f(x_t)^\top s,
\qquad
x_{t+1}=(1-\gamma_t)x_t+\gamma_t s_t.
\]

It naturally yields sparse convex combinations after dropping near-zero weights.

**Package fit:** excellent for ensemble weights, blend weights, or rule-selection mixtures. It does not replace L1 leaf fitting.

**Recommendation:** implement only when ensemble weighting or model blending exists as a product need. Convergence is slow near optimum, so use away-step or pairwise variants if accuracy matters.

### 3.6 Trust-region and quasi-Newton methods

L-BFGS already covers smooth leaf objectives. **Full-memory BFGS** is unsuitable for large leaf designs. **Trust-region methods** offer stronger global convergence but require quadratic subproblem machinery.

**Package fit:** BFGS and nonlinear conjugate gradient now have a concrete home. Spec 26 Phase 4 adds both as experimental leaf solvers that compete directly against L-BFGS on the same unconstrained L2 GLM objective. BFGS stays a small-leaf benchmark candidate because its inverse-Hessian memory is quadratic in parameter count. Nonlinear CG is the stronger default candidate because it keeps memory linear. Trust-region methods stay deferred.

#### 3.6.1 SciPy and statsmodels competitor audit

SciPy and statsmodels expose two different solver portfolios. SciPy provides general-purpose numerical methods. statsmodels selects SciPy methods for likelihood models and adds Newton-Raphson.

Current `mltreelib` already provides IRLS, exact Gaussian OLS/Ridge, CCD, L-BFGS, dense GLM fitting, and four evolutionary families. Against this baseline, SciPy and statsmodels add these credible competitors:

| Solver | Requirement | Direct competitor for current L-BFGS use? | `mltreelib` decision |
|---|---|---|---|
| BFGS | Objective and gradient | Yes | Benchmark candidate. Full-memory inverse Hessian is useful only for small leaves, but it gives a clean quasi-Newton comparison. |
| Nonlinear CG, Polak-Ribiere plus restart | Objective and gradient | Yes | Best additional leaf-solver candidate. Uses first derivatives, keeps memory linear, and avoids L-BFGS curvature history. |
| L-BFGS-B | Objective and gradient plus coefficient bounds | No, not yet | Add to general optimizer. Current leaf API has no box-bounded coefficients; one-sided coefficient bounds would also need an explicit distinction from feature-monotone constraints. |
| Newton / Newton-CG | Objective, gradient, and Hessian or Hessian-vector product | No | Defer. Canonical GLM families already have IRLS; exposing exact Hessian algebra across Tobit, Tweedie, and missing-value paths adds complexity. |
| Nelder-Mead | Objective only | No for smooth GLM leaves | Keep out of leaf dispatch. Useful only as a derivative-free generic fallback or benchmark. |
| Powell | Objective only | No for smooth GLM leaves | Keep out of leaf dispatch. Useful only in the general bounded black-box optimizer. |
| TNC | Objective, gradient, and bound support | No | Defer until bounded leaf coefficients become a real requirement. |
| SLSQP, COBYLA, COBYQA, trust-constr | Explicit constraints and bounds | No | Not leaf solvers. Add COBYLA to the generic bounded optimizer as the one nonlinear constraint solver. Defer SLSQP, COBYQA, and trust-constr. See `2026-11-21-generic-global-and-constrained-solvers.md`. |
| Basin hopping | Objective plus inner local optimizer | No | Add to the generic bounded optimizer after L-BFGS-B exists. Evolutionary searches already cover nonconvex tuning, so basin hopping needs benchmark evidence before public exposure. See `2026-11-21-generic-global-and-constrained-solvers.md`. |

statsmodels exposes `newton`, `nm`, `bfgs`, `lbfgs`, `cg`, `ncg`, `powell`, `basinhopping`, and a generic `minimize` bridge. Its presence confirms ecosystem precedent for BFGS, CG, Nelder-Mead, and Powell. It does not show that every method should become a public `mltreelib` leaf solver.

**Recommended direct competition set:** implement BFGS and Polak-Ribiere nonlinear CG as experimental leaf solvers, then benchmark them against L-BFGS on identical GLM objectives, weights, L2 penalties, warm starts, and iteration/evaluation budgets. Keep L-BFGS as default. Add L-BFGS-B to the general bounded optimizer, not leaf dispatch, because coefficient bounds do not currently exist.

**Needed benchmark dimensions:** convergence speed, final deviance, gradient norm, memory, evaluation count, sensitivity to conditioning, and failure behavior across Binomial, Poisson, Gamma, Tweedie, and Tobit leaves. Report that BFGS and CG solve the same unconstrained L2 objective as L-BFGS. Keep monotone and L1 paths separate because those objectives differ.

### 3.7 Stochastic gradient methods

**SAGA** gives composite-step stochastic gradients with L1 support. It can fit large leaf datasets without repeatedly visiting every row.

**Package fit:** medium for very large leaves, low for normal tree leaves. Tree leaves are often small, and full-batch deterministic behavior fits package requirements.

**Recommendation:** defer until leaf sizes make full-batch gradients materially slow.

### 3.8 Global, swarm, and evolutionary search

Current core already contains four useful population families:

| Algorithm | Current core | Best use in `mltreelib` |
|---|---|---|
| CMA-ES | Yes | Continuous non-convex tuning with learned covariance |
| SHADE | Yes | Self-adaptive differential evolution for difficult continuous search |
| sNES / Natural Gradient | Yes | Lightweight continuous evolutionary adaptation |
| Real-coded GA | Yes | Mixed continuous search with SBX and polynomial mutation |

These suit hyperparameter tuning, feature-order search, and non-smooth objectives without gradients. They do not provide global optimality guarantees. Their role remains meta-optimization rather than exact leaf fitting.

#### Differential Evolution family

**Differential Evolution (DE)** evolves candidate vectors from scaled differences between population members. Standard mutation crosses a donor vector with the target, then binomial crossover selects coordinates. DE/rand/1/bin is a strong general-purpose baseline because it needs few hyperparameters and handles non-smooth search spaces well.

Current SHADE already covers the most important adaptive DE family: success-history adaptation, current-to-pbest/1 mutation, and an external archive. Adding plain DE would mostly duplicate existing machinery. Later variants worth benchmarking for specific objectives are:

- **DE/current-to-pbest/1 with archive:** robust default for constrained or multimodal continuous search.
- **JADE:** DE with external archive and adaptive parameter learning.
- **L-SHADE:** linear population-size reduction and historical memory.
- **SHADE-RSP:** random-key permutation encoding for mixed discrete-continuous variables.
- **Multi-modal DE:** useful only when several equivalent optima matter, such as redundant feature subsets.

**Package fit:** SHADE is already implemented. Keep DE additions benchmark-driven. A distinct API should expose DE mutation strategy, crossover probability, archive policy, and population-size policy instead of adding many near-duplicate enums.

#### Evolution Strategies and CMA-ES

A **CMA-ES** adapts a full or diagonal covariance matrix from ranked sampling steps. It excels on continuous, ill-conditioned, rotated objectives when evaluations are expensive enough to justify covariance learning.

Current CMA-ES should remain primary for continuous model and leaf tuning. Later candidates:

- **sep-CMA-ES:** diagonal covariance; lower $O(P^2)$ memory for many features.
- **BIP-CMA-ES:** restarts with increasing population size.
- **CMA-ES with active noise handling:** useful for stochastic validation losses.
- **modular CMA-ES:** separates modules across related objective families.

**Package fit:** high, but first verify whether current implementation already provides diagonal covariance or restart behavior. Do not add variants by name without benchmark evidence.

**ES family:** **(mu, lambda)-ES** uses only offspring selection and permits many parallel evaluations. **(mu + lambda)-ES** includes parents and is more stable. Natural Evolution Strategies update distribution parameters through natural-gradient estimates and can match ES quality with simpler update rules.

Current sNES covers the natural-gradient ES family. Plain classical ES is less useful when CMA-ES and sNES exist unless evaluation-budget simplicity matters.

#### Particle Swarm Optimization

**PSO** moves particles using personal-best and global-best attraction. It is simple, has few parameters, and works well on continuous numeric objectives. Standard PSO has linear convergence near the optimum and can stagnate in high dimensions.

Variants worth later consideration:

- **Inertia-weight PSO:** safest baseline.
- **Constriction PSO:** fixed convergence coefficient.
- **Canonical versus social-only PSO:** useful ablation.
- **Fully informed PSO:** particles learn from all neighborhood bests.
- **Adaptive or time-varying inertia:** helps rugged or shifting objectives.
- **Differential Evolution / PSO hybrids:** can improve exploration but increase tuning burden.

**Package fit:** excellent for continuous hyperparameter and leaf-parameter search. PSO needs no gradients and maps cleanly to current `DesignMatrix` objective. It should support deterministic initialization and explicit topology. Start with global-best PSO and inertia weight before neighborhood topologies.

#### Island-model evolutionary algorithms

An **island model** runs several independent subpopulations and periodically migrates elites through a topology such as ring, complete, or random graph. Islands preserve diversity and reduce premature convergence. Migration rate and topology determine whether the method behaves like parallel restarts or a cooperative population.

Useful properties for `mltreelib`:

- Search different tree structures, feature subsets, or leaf parameters concurrently.
- Seed islands from different algorithms, for example greedy tree, CMA-ES, SHADE, and PSO.
- Preserve diversity across categorical partition encodings.
- Enable asynchronous parallel search when future runtime supports it.

**ICoEvoRDF** demonstrates direct tree relevance: independent islands evolve robust trees and adversarial perturbations, then migrate top trees. Across 20 benchmark datasets, paper reports improvements up to 0.010 adversarial accuracy and 0.064 minimax regret. Treat these as paper results, not reproduced `mltreelib` results.

Risks:

- Multiple islands multiply expensive objective evaluations.
- Migration can destroy useful specialization.
- A ring topology can circulate poor solutions around the graph.
- Parallel islands conflict with package rule: run one multi-core job at a time.

**Package fit:** high as an orchestration layer over existing optimizers, but low as another leaf-fitting algorithm. Implement islands first over scalar objective functions, with serial deterministic migration. Add parallel or asynchronous islands only after scheduling and oversubscription controls exist.

#### Multi-objective evolutionary algorithms

**NSGA-II** uses non-dominated sorting, crowding distance, elitist survivor selection, SBX, and polynomial mutation. It fits objectives such as predictive error, model complexity, calibration error, and runtime.

**NSGA-III** adds reference directions and niching for many objectives. **MOEA/D** decomposes a multi-objective problem into scalar subproblems and improves diversity for some Pareto fronts. **SMS-EMOA** uses hypervolume improvement and can be strong for low objective counts but costs more.

**Package fit:** medium-high for later product work. `mltreelib` has real multi-objective choices: accuracy versus tree size, anomaly detection versus false-positive burden, clustering compactness versus separation, and solver speed versus accuracy.

NSGA-II is first choice for two or three objectives. NSGA-III and MOEA/D should wait until real use needs four or more objectives. Every Pareto result needs nondominated sorting, crowding or niching, objective direction normalization, and a deterministic tie-break policy.

#### Other population methods

- **Artificial Bee Colony:** useful for continuous global search, but structurally close to DE/PSO and unlikely to earn a core algorithm before them.
- **Ant Colony Optimization:** suitable for discrete combinatorial routing or assignment, but expensive for vector-valued model parameters.
- **Memetic algorithms:** combine local search with evolutionary populations. Strong when each candidate can receive a cheap local improvement, such as a greedy tree repaired after mutation.
- **Quality Diversity algorithms:** useful when many diverse solutions matter, such as interpretable tree architectures or feature subsets. They require behavioral-descriptor design and archive management.

**Recommendation:** defer Ant Colony and Artificial Bee Colony. Memetic search and quality diversity are better future directions for tree-structure search because they exploit local tree improvement.

### 3.9 Exact decision-tree solvers

#### Branch-and-bound

**CORELS / Optimal Sparse Decision Trees** and **GOSDT** search globally over depth-bounded trees using lower bounds and caching. DL8.5 adds caching branch-and-bound for binary trees.

**Package fit:** excellent for bounded subtrees. A depth-3 or depth-4 post-pass on greedy trees can improve local structure without solving full datasets.

**Risk:** exponential search. Require explicit depth, leaf count, minimum support, time budget, and optimality status.

#### Mixed-integer programming

Bertsimas and Dunn's optimal classification tree formulation and later ICOT formulations express tree structure and assignments as mixed-integer programs.

**Package fit:** weak for default core. Requires MIP solver and license/dependency concerns. Gurobi, CPLEX, HiGHS, or SCIP have different performance and licensing profiles.

**Recommendation:** optional Python extra or separate integration crate. Core should expose candidate tree and objective data, not depend on one commercial solver.

#### SAT-based optimal trees

SAT encodings trade continuous split search for efficient Boolean structure search. They remain niche behind dedicated solvers.

**Recommendation:** defer until core branch-and-bound proves insufficient for real workloads.

### 3.10 Multi-way tree and feature-graph optimization

Recent optimal multi-way tree work uses binary linear optimization over paths in a feature graph. For interpretable clustering, ICOMT extends this idea.

**Package fit:** candidate generation fits Rust. Exact path selection needs an external binary linear solver or a specialized branch-and-bound implementation.

**Recommendation:** build candidate enumeration and greedy fallback first. Keep exact backend optional.

### 3.11 Constrained and conic solvers

Clarabel and other conic solvers handle second-order cone and semidefinite constraints. Such problems can express constrained regression, robust covariance, and some global tree bounds.

**Package fit:** low. No current core product requires semidefinite constraints. A conic dependency materially expands build and numerical-complexity surface.

**Recommendation:** defer until a concrete constrained model lands.

---

## 4. Implementation-Fit Matrix

| Solver or method | Pure Rust | Main use | Complexity | Risk | Tier |
|---|---:|---|---|---|---|
| FISTA / proximal gradient | Yes | L1 GLM leaves | $O(NP)$ per iteration plus backtracking | Low | 1 |
| Block-coordinate descent | Yes | Grouped/sparse leaves | Iteration-dependent | Medium | 2 |
| ADMM | Yes | Convex clustering | Iteration and graph dependent | Medium | 1 |
| Tree-guided fused-lasso DP | Yes | Convex cluster paths | Near-linear under tree structure | Medium-high | 1 |
| Frank-Wolfe | Yes | Sparse simplex weights | Iteration dependent | Low | 2 |
| Bounded DL8.5/GOSDT search | Yes | Exact small subtrees | Exponential in depth | Medium | 1 |
| MIP optimal tree | No | Globally optimal trees | Solver dependent | High | 3 |
| SAT optimal tree | External | Boolean tree search | Solver dependent | High | 3 |
| SAGA | Yes | Very large stochastic leaves | Stochastic | Medium | 2 |
| Conic solver | External | Advanced constrained models | Solver dependent | High | 4 |
| PSO | Yes | Continuous black-box tuning | $O(GPN)$ | Low | 2 |
| Island-model orchestrator | Yes | Parallel populations and migration | Multiplies evaluations | Medium | 2 |
| NSGA-II | Yes | Two or three conflicting objectives | Population-dependent | Medium | 2 |
| NSGA-III / MOEA/D | Yes | Four or more objectives | Population-dependent | Medium-high | 3 |
| Plain DE | Yes | Continuous global search | $O(GPN)$ | Low, but overlaps SHADE | 3 |
| sep-CMA-ES / BIP-CMA-ES | Yes | High-dimensional or restarted search | $O(GP^2)$ or restarts | Medium | 3 |
| Memetic / quality-diversity search | Yes | Diverse tree structures | Search-dependent | Medium-high | 3 |

---

## 5. Recommended Architecture

### 5.1 Common solver seam

Avoid one monolithic solver interface. Use a small set of mathematical seams:

- `SmoothObjective`: value, gradient, optional Hessian, Lipschitz estimate.
- `ProxOperator`: identity, L1 soft threshold, bound projection, monotone constraint.
- `IterativeSolver`: state, convergence, stop reason, iteration history.
- `ExactSolver`: incumbent, bound, optimality status, search statistics.
- `GlobalSolver`: population/individual states and budget for evolutionary methods.

This lets FISTA, L-BFGS, CCD, ADMM, and exact search report evidence consistently without pretending they solve the same problem class.

### 5.2 Phase sequence

**Phase 1 — L1 GLM leaves**

- Add FISTA or another proximal method around existing GLM gradients.
- Backtrack until step satisfies smoothness bound.
- Warm-start from ridge solution.
- Extend `fit_model` L1 routing to supported families.
- Compare against R `glmnet` or equivalent fixed-seed references.
- Report convergence, KKT residual, and coefficient support.

**Phase 2 — Bounded exact trees**

- Reuse split candidate generation.
- Implement branch-and-bound lower bounds for additive losses.
- Limit search to selected subtrees or small datasets.
- Return incumbent, lower bound, and whether optimality was proved.
- Benchmark against greedy tree and current GOSDT/DL8.5 implementations.

**Phase 3 — Convex clustering solvers**

- Implement ADMM with sparse graph weights.
- Implement tree-guided dynamic programming for tree weights.
- Compare cluster paths and objective values.
- Expose memory limits and approximation status.

**Phase 4 — Population and multi-objective search**

- Add PSO as a small deterministic continuous-search backend.
- Add serial island orchestration over CMA-ES, SHADE, sNES, GA, and PSO populations.
- Add migration policy, ring topology, elite count, and migration interval as explicit parameters.
- Add NSGA-II for accuracy-versus-complexity, calibration-versus-coverage, and runtime trade-offs.
- Keep NSGA-III, MOEA/D, memetic search, and quality diversity benchmark-driven.
- Preserve one multi-core job at a time; parallel islands require explicit scheduler support.

**Phase 5 — Optional exact backends**

- Define solver-neutral tree model and objective interface.
- Add optional HiGHS/Gurobi/SCIP adapter behind extra dependency.
- Keep MIP/SAT out of default Rust core.
- Preserve deterministic core behavior when external backend is absent.

---

## 6. Verification Requirements

Every solver addition needs more than “does not panic.”

### Generic

- Deterministic output for fixed seed.
- Finite objective and coefficients.
- Explicit stop reason.
- Correct behavior with empty, constant, singleton, and missing-heavy inputs.
- No silent fallback to a different objective or penalty.

### Proximal solver

- Compare Gaussian L1 result to CCD.
- Check KKT stationarity and subgradient conditions.
- Check warm-start equivalence.
- Compare non-Gaussian coefficients with fixed reference data.

### Exact tree search

- Return proof status when search exhausted.
- Match exhaustive enumeration on tiny instances.
- Respect depth, support, and leaf limits.
- Never claim optimality after timeout.

### Convex clustering

- Objective matches direct evaluation.
- Cluster path does not regress without documented algorithm behavior.
- Sparse graph and tree graph use same public objective semantics.
- Memory cap triggers explicit error or documented approximation.

### Population and island search

- Same seed and budget produce identical results.
- Island migration cannot duplicate elite population beyond configured size.
- Ring topology preserves or improves global best incumbent.
- Islands with different algorithms remain observably distinct before migration.
- PSO respects bounds and reports global best, not only final swarm position.
- NSGA-II returns nondominated set under known toy objectives.
- Objective directions and normalization are tested for minimization and maximization.
- Evaluation budget counts every objective call, including migration and archive evaluations.

---

## 7. Risks

1. **Solver sprawl.** Each method adds numerical edge cases. Common objective/prox seams reduce this risk.
2. **False optimality.** Greedy, evolutionary, and timed-out exact searches are not globally optimal.
3. **Scale illusion.** O(nd) ECOD-style scoring differs from repeated optimization passes.
4. **Penalty drift.** Objective, regularization, and normalization must be recorded in fitted result.
5. **Backend lock-in.** Core data structures must not mirror one MIP vendor.
6. **Parallel oversubscription.** Rayon-backed solvers must not run concurrently with other multi-core fits.
7. **Stale roadmap.** L-BFGS already exists. New research must inspect current source, not trust old notes.

---

## 8. Rejected or Deferred Options

| Option | Decision | Reason |
|---|---|---|
| Rewrite L-BFGS | Reject | Already implemented |
| Full BFGS in core | Add as experimental leaf solver | Spec 26 Phase 4. Small-leaf benchmark candidate only. Quadratic memory limits it to small leaves. |
| Adam for tree leaves | Reject | Exact statistical fits do not need adaptive neural optimization |
| MIP solver in default build | Reject | Heavy optional dependency and license complexity |
| SAT solver in default build | Defer | Niche exact-tree path |
| Semidefinite conic solver | Defer | No current product requirement |
| SAGA everywhere | Reject | Full-batch deterministic fitting fits normal leaf sizes |
| Evolutionary solver as “global solver” | Reject | No global guarantee |

---

## 9. Exact Coverage Audit Against the Proposed 20-Algorithm Portfolio

This audit compares named algorithms against current source, not mathematical adjacency. “Full” means core mechanics exist under the requested name. “Partial” means reusable mechanics exist but requested algorithm, bounds, restart, or public behavior does not.

### 9.1 Population and evolutionary portfolio

| # | Proposed algorithm | Status | Current evidence | Absolute gap |
|---:|---|---|---|---|
| 1 | Random Search + LHS | Absent | CMA-ES, SHADE, and GA initialize populations randomly | No random-search baseline and no Latin hypercube sampler |
| 2 | PSO | Absent | No particle state, velocity, inertia, or neighborhood update | Complete algorithm |
| 3 | DE/rand/1/bin | Partial | `SHADE` implements adaptive DE with archive and `current-to-pbest/1` mutation | No plain DE/rand/1/bin implementation; Python name `de` currently aliases SHADE |
| 4 | jDE | Absent | SHADE uses success-history memory shared by population | No per-individual self-adaptive F and CR |
| 5 | CMA-ES | Full | `CmaEs` adapts mean, covariance, and step size | No explicit bounds; full covariance costs $O(P^2)$ memory |
| 6 | IPOP-CMA-ES | Absent | CMA-ES stops without restart | No stagnation-triggered increasing-population restarts |
| 7 | Real-Coded GA | Full | `RCGA` uses tournament selection, SBX, polynomial mutation, and elitism | Current encoding is continuous coefficients, not generic discrete/integer encodings |
| 8 | Island-Model EA | Absent | No subpopulation, topology, migration, or archive layer | Complete framework |
| 9 | NSGA-II | Absent | No Pareto archive, nondominated sorting, crowding distance, or multiobjective result | Complete algorithm and objective-vector API |
| 10 | MOEA/D | Absent | No weighted scalar subproblems, neighborhoods, or reference vectors | Complete algorithm and decomposition API |

**Portfolio result:** 2 full, 1 partial, 7 absent. Named coverage is 20% full or 25% counting partial DE as covered.

### 9.2 First-order, quasi-Newton, and stochastic portfolio

| # | Proposed algorithm | Status | Current evidence | Absolute gap |
|---:|---|---|---|---|
| 11 | Cyclic Coordinate Descent | Full | `ccd.rs` cycles active coordinates and updates residuals incrementally | No generic block-coordinate mode |
| 12 | Random Coordinate Descent | Absent | CCD always uses fixed cyclic order | Random coordinate or block scheduler |
| 13 | Proximal Coordinate Descent | Full | CCD applies exact L1 soft thresholding per coordinate | General proximal objective API remains absent |
| 14 | Gradient Descent + Backtracking | Partial | L-BFGS implements Armijo backtracking | No standalone gradient-descent algorithm |
| 15 | Nonlinear Conjugate Gradient | Absent | No conjugate search-direction history | Complete algorithm and Hessian-vector product contract |
| 16 | L-BFGS | Full | `lbfgs.rs` uses two-loop recursion, memory history, and Armijo line search | No box-bound handling |
| 17 | L-BFGS-B | Absent | No lower or upper coefficient bounds anywhere in solver options | Complete projected-bound variant |
| 18 | Trust-Region Newton / Steihaug CG | Absent | No trust radius, quadratic subproblem, or CG trust-region update | Complete second-order machinery |
| 19 | SGD with Momentum | Absent | Solvers use full-batch deterministic objectives | Mini-batch sampling, stochastic gradients, and velocity state |
| 20 | AdamW | Absent | No adaptive moments or decoupled weight decay | Complete algorithm |

**Portfolio result:** 3 full, 1 partial, 6 absent. Named coverage is 30% full or 35% counting partial gradient descent as covered.

### 9.3 Combined result

- Full implementations: **5 of 20 = 25%**.
- Partial implementations: **2 of 20 = 10%**.
- Completely absent named algorithms: **13 of 20 = 65%**.
- Effective missing coverage: **14 of 20 = 70%** when partial implementations count as incomplete.
- First portfolio: **2 full, 1 partial, 7 absent**.
- Second portfolio: **3 full, 1 partial, 6 absent**.

This percentage understates functional overlap because present algorithms are tightly coupled to `DesignMatrix` GLM coefficient fitting. They are not generic bounded black-box optimizers for hyperparameter, tree-structure, discrete-encoding, or multiobjective search.

### 9.4 Architectural gap behind percentages

A generic optimization layer is still missing. Current solvers accept a `DesignMatrix`, response, penalties, and coefficient-sign constraints. They do not share one interface for:

- Arbitrary bounded objective functions.
- Mixed continuous, integer, categorical, and tree-structure encodings.
- Multiobjective result vectors and Pareto archives.
- Island populations and migration.
- Restart budgets and incumbent preservation.
- Evaluation-budget accounting across stochastic methods.

Adding more named leaf optimizers before this seam exists would duplicate control flow and make public APIs inconsistent.

### 9.5 Absolute implementation drop, ordered by value

| Priority | Addition | Drop from current state | Why first |
|---:|---|---|---|
| 1 | Generic bounded black-box optimizer API | New core seam | Enables every population method without duplicating routing |
| 2 | Random Search + LHS | New sampler plus baseline | Foundational fallback; deterministic and cheap |
| 3 | L-BFGS-B | New projected-bound quasi-Newton path | Box bounds affect every later optimizer |
| 4 | PSO | Complete new algorithm | Strong black-box baseline with little shared machinery |
| 5 | IPOP-CMA-ES | Restart layer over existing CMA-ES | High leverage because CMA-ES already exists |
| 6 | NSGA-II | New Pareto API and selection machinery | First practical multiobjective method for model-complexity trade-offs |
| 7 | Island model | New orchestration layer | Useful after generic objective and multiobjective APIs exist |
| 8 | Plain DE/rand/1/bin | Small addition over SHADE internals | Useful baseline, but lower incremental value than missing families |
| 9 | jDE | Per-individual parameter adaptation | Moderate value; SHADE already covers adaptive DE |
| 10 | RCD and standalone GD | Small first-order additions | Useful diagnostics, but less differentiated |
| 11 | MOEA/D | Full decomposition machinery | Wait until real many-objective need appears |
| 12 | Nonlinear CG | Add as experimental leaf solver | Spec 26 Phase 4. First-order competitor against L-BFGS with linear memory. Trust-region CG stays deferred. |
| 13 | SGD momentum and AdamW | Full stochastic stack | Poor fit for exact small-leaf statistical fits |

---

## 10. Primary Sources and Indexes

1. Beck, A. and Teboulle, M. (2009). “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.” *SIAM Journal on Imaging Sciences*. [DOI](https://doi.org/10.1137/080716542).
2. Friedman, J., Hastie, T., and Tibshirani, R. (2010). “Regularization Paths for Generalized Linear Models via Coordinate Descent.” *JSS*. [DOI](https://doi.org/10.18637/jss.v033.i01).
3. Byrd, R. H., Nocedal, J., and Wright, S. J. (1995). “Convergence of the Trust Region Algorithm for Large Scale Problems.” In *Numerical Optimization*. [DOI](https://doi.org/10.1007/978-1-4612-5559-2_5).
4. Defazio, A., Bach, F., and Lacoste-Julien, S. (2014). “SAGA: A Fast Incremental Gradient Method with Support for Non-Smooth Nonsmooth Composite Objectives.” *NeurIPS*. [arXiv:1407.0202](https://arxiv.org/abs/1407.0202).
5. Jaggi, M. (2013). “Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization.” *ICML*. [PMLR](https://proceedings.mlr.press/v28/jaggi13.html).
6. Chi, E. and Lange, S. (2015). “Splitting Methods for Convex Clustering.” *JCGS*. [arXiv:1304.0499](https://arxiv.org/abs/1304.0499).
7. Hocking, T., Joulin, D., Bach, F., and Vert, J.-P. (2011). “Clusterpath: An Algorithm for Clustering using Convex Fusion Penalties.” *ICML*. [Paper](https://icml.cc/2011/papers/419_icmlpaper.pdf).
8. Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. (2011). “Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.” *Foundations and Trends in ML*. [DOI](https://doi.org/10.1561/2200000016).
9. Zhang, B. and Terada, Y. (2025). “Tree-Guided $L_1$-Convex Clustering.” [arXiv:2503.24012](https://arxiv.org/abs/2503.24012).
10. Hu, X., Rudin, C., and Seltzer, M. (2019). “Optimal Sparse Decision Trees.” *NeurIPS*. [arXiv:1904.12847](https://arxiv.org/abs/1904.12847).
11. Lin, J., Zhong, C., Hu, D., Rudin, C., and Seltzer, M. (2020). “Generalized and Scalable Optimal Sparse Decision Trees.” *ICML*. [PMLR](https://proceedings.mlr.press/v119/lin20g.html).
12. Aglin, G., Nijssen, S., and Schaus, P. (2020). “Learning Optimal Decision Trees Using Caching Branch-and-Bound Search.” *AAAI*. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5711).
13. Verdonck, K., et al. (2020). “PyDL8.5, a Scalable Implementation of the DL8.5 Algorithm.” *IJCAI*. [Paper](https://ijcai.org/proceedings/2020/750).
14. Bertsimas, D. and Dunn, J. (2017). “Optimal Classification Trees.” *Machine Learning*. [DOI](https://doi.org/10.1007/s10994-017-5633-9).
15. Narodytska, I., Ignatiev, A., Pereira, F. M., and Marques-Silva, J. (2018). “Learning Optimal Decision Trees Using SAT Solving.” *IJCAI*. [Paper](https://ijcai.org/proceedings/2018/189).
16. Lampainen, J., Joki, K., Karmitsa, N., and Mäkelä, M. M. (2025). “Clust-Splitter.” [arXiv:2505.04389](https://arxiv.org/abs/2505.04389).
17. Suzuki, H., et al. (2026). “Interpretable Clustering via Optimal Multi-way Decision Trees.” [arXiv:2602.13586](https://arxiv.org/abs/2602.13586).
18. Storn, R. and Price, K. (1997). “Differential Evolution: A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.” *Journal of Global Optimization*. [DOI](https://doi.org/10.1023/A:1008202821328).
19. Tanabe, R. and Fukunaga, A. (2013/2014). “Success-History Based Adaptive Differential Evolution.” *IEEE CEC*. [DOI](https://doi.org/10.1109/CEC.2013.6557555).
20. Hansen, N., Ostermeier, M., and Arnold, I. (2001). “The Covariance Matrix Adaptation Evolution Strategy.” Technical report, Technische Universität Berlin. No DOI.
21. Kennedy, J. and Eberhart, R. (1995). “Particle Swarm Optimization.” *ICNN*. [DOI](https://doi.org/10.1109/ICNN.1995.488968).
22. Poli, R., Kennedy, J., and Blackwell, T. (2007). “Particle Swarm Optimization: An Overview.” *Swarm Intelligence*. [DOI](https://doi.org/10.1007/s11721-007-0002-0).
23. Tanese, R. (1989). “Distributed Genetic Algorithms for Function Optimization.” University of Michigan. [Report record](https://deepblue.lib.umich.edu/handle/2027.42/2267).
24. Skolicki, Z. and De Jong, K. (2004). “Improving Evolutionary Algorithms with Multi-Representation Island Models.” *PPSN*. [DOI](https://doi.org/10.1007/978-3-540-30217-9_38).
25. Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.” *IEEE TEC*. [DOI](https://doi.org/10.1109/4235.996017).
26. Deb, K., et al. (2013). “An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting: NSGA-III.” *IEEE TEC*. [DOI](https://doi.org/10.1109/TEVC.2013.2301485).
27. Zhang, J. and Li, M. (2009). “MOEA/D: A Decomposition-Based Multi-Objective Evolutionary Algorithm.” *IEEE TEC*. [DOI](https://doi.org/10.1109/TEVC.2009.2001037).
28. Żychowski, A., Perrault, A., and Mańdziuk, J. (2024). “Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution.” [arXiv:2412.13762](https://arxiv.org/abs/2412.13762).
29. SciPy. “scipy.optimize.minimize.” [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
30. statsmodels. “Optimization.” [Official documentation](https://www.statsmodels.org/stable/optimization.html).

## 11. Source Notes

- alphaXiv supplied recent search, convex-clustering, optimal-tree, and mixed-integer candidates.
- Papers With Code was used for historical method discovery and implementation metadata where available.
- treg catalog exposed Perplexity, Brave, and Scholar endpoints. Live treg calls failed because saved authentication expired. No failed response supports a claim.
- The 2026-11-17 GLM solver note is useful but stale on L-BFGS status. Current source wins.
