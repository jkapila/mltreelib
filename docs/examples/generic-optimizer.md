# Generic optimizer and real-data tuning

`mltreelib.solvers.optimize` minimizes a callable scalar objective inside
bounds. It is the public entry point for the shared optimization seam in the
Rust core. One objective, bounds, budget, request, and result contract backs
every method, so the result never changes shape when the method changes.

This guide covers every optimizer in the seam: the two that ship today and the
eight that land in the later waves of
[Spec 26](https://github.com/jkapila/mltreelib/blob/main/specs/26-generic-global-and-constrained-optimizers.md).
Methods that are not implemented yet raise one error naming the method. The
library never substitutes another algorithm silently.

## Python surface

```python
from mltreelib.solvers import optimize

def objective(point):
    return (point[0] - 0.3) ** 2 + (point[1] + 1.5) ** 2

result = optimize(
    objective,
    [(-2.0, 2.0), (-2.0, 2.0)],
    method="random_search",
    samples=500,
    max_evaluations=500,
    seed=42,
)
```

| Argument | Meaning |
|---|---|
| `objective` | Callable that takes one list of parameter values and returns one finite scalar. |
| `bounds` | One `(lower, upper)` pair per parameter. |
| `method` | Canonical method name. |
| `start` | Starting point. Warm-starts the search. |
| `max_evaluations` | Cap on objective calls. |
| `max_iterations` | Cap on solver iterations. |
| `tol` | Convergence tolerance. |
| `seed` | Seed for stochastic methods. |
| `gradient` | Optional callable returning one derivative per parameter. |
| `samples` | Candidate count for the sampling methods. |
| `initial_population` | Optional first candidates for population methods. |

The result reports:

| Field | Meaning |
|---|---|
| `best_x` | Best point found. Finite and inside the bounds. |
| `best_value` | Objective value at `best_x`. |
| `iterations` | Solver iterations performed. |
| `evaluations` | User objective evaluations performed. |
| `status` | Termination status. |
| `converged` | True only for `converged` and `tolerance`. |
| `history` | Optional convergence history. |
| `max_constraint_violation` | Largest constraint violation at `best_x`. |
| `message` | Failure detail when the run did not succeed. |

The API reference page [Solvers & optimization](../reference/api/solvers/)
holds the exact signatures and docstrings.

## Statuses

| Status | Meaning |
|---|---|
| `converged` | The method reached its convergence test. |
| `tolerance` | The tolerance test ended the run. |
| `iteration_limit` | The iteration budget ended the run. |
| `evaluation_limit` | The evaluation budget ended the run. |
| `invalid_objective` | The objective returned a NaN or infinite value. |
| `numerical_failure` | The numerical kernel failed. |
| `infeasible` | No feasible point was found. |
| `unsupported` | The method cannot solve the requested problem. |
| `hard_error` | An unrecoverable error ended the run. |

Budget exhaustion never reports success. A failed run never returns a
converged status.

## The optimizers

| Method | Status | Needs | Bounds | Memory | Use it for |
|---|---|---|---|---|---|
| `random_search` | available | objective only | finite box | linear | Honest baseline for every comparison. |
| `latin_hypercube` | available | objective only | finite box | linear | Stratified initialization for population methods. |
| `lbfgs_b` | planned | gradient | box bounds | linear | Smooth bounded objectives with many parameters. |
| `bfgs` | planned | gradient | unbounded only | quadratic | Small smooth problems where curvature pays off. |
| `nonlinear_cg` | planned | gradient | unbounded only | linear | First-order competitor with no curvature history. |
| `nelder_mead` | planned | objective only | finite box | linear | Black-box local search and basin-hopping inner solves. |
| `powell` | planned | objective only | finite box | linear | Derivative-free search with different basin behaviour. |
| `simulated_annealing` | planned | objective only | finite box | linear | Cheap global search over non-smooth objectives. |
| `cobyla` | planned | objective only | finite box | linear | Nonlinear inequality constraints without gradients. |
| `basin_hopping` | planned | inner local solver | finite box | linear | Rugged landscapes with separated local minima. |

### Random Search

Samples uniformly inside the bounds and keeps the best candidate. It needs no
gradient, so it runs against any objective. It is the baseline every other
method must beat under an equal evaluation budget.

### Latin hypercube

Stratifies every dimension independently into `samples` strata, draws one point
per stratum, and shuffles the strata with the seeded generator. The marginal
coverage of each dimension beats uniform sampling at the same budget, which
makes it the better initializer for population methods.

### L-BFGS-B

The bounded smooth local solver. Each iteration computes the generalized
Cauchy point along the piecewise-linear projected-gradient path, approximately
minimizes the quadratic model over the free variables, runs a line search,
projects onto the bounds, and tests the projected gradient norm over the free
variables only. It keeps the limited-memory curvature update, so large
parameter vectors stay practical. Without binding bounds its result matches
the existing `solve_lbfgs` path.

### BFGS

Full-memory BFGS with a dense inverse-Hessian approximation. Memory grows
quadratically with the parameter count, so it suits small problems. It solves
the same unconstrained objective as L-BFGS, which makes the two directly
comparable. It rejects finite bounds instead of clipping them.

### Nonlinear conjugate gradient

Polak-Ribiere direction with a documented restart rule. It needs only the
objective and the gradient, keeps memory linear in the parameter count, and
carries no curvature history. A negative Polak-Ribiere coefficient restarts
the search direction.

### Nelder-Mead

Derivative-free simplex search. It replaces the worst vertex each iteration
with reflection, expansion, contraction, or shrink. The initial simplex shape
and the two tolerances are configurable. It clips or reflects simplex vertices
into the box according to a documented policy.

### Powell

Derivative-free conjugate-direction search. It performs sequential
one-dimensional minimizations along a maintained direction set, and every
line-search evaluation stays inside the box.

### Simulated annealing

Global trajectory search over a bounded domain. The proposal is a per
dimension bounded random displacement, the cooling schedule is geometric by
default, and the boundary policy is either resampling or reflection. Clipping
is not offered because it piles proposal mass on bound-active points. The best
incumbent is tracked separately from the current point, so the result never
degrades. The method ships as a heuristic with no optimality claim.

### COBYLA

The one nonlinear constraint solver. It models the objective and the
constraints by linear interpolation over an interpolation set and solves a
small linear subproblem inside a shrinking trust region. It needs function
values only. Bounds act as unrelaxable constraints, so every evaluated point
stays inside the box. Equality constraints are rejected explicitly. The
result reports the largest constraint violation.

### Basin hopping

A composition, not a numerical kernel. It perturbs the incumbent, runs an
inner local solve through the same seam, and accepts the result with a
constant-temperature Metropolis test. The step size is either fixed or adapted
toward a target acceptance rate. Inner solve evaluations count against the
outer budget, and a failed inner solve never erases the incumbent.

## Budgets and determinism

`max_evaluations` caps the number of objective calls and `max_iterations` caps
the number of solver iterations. Both limits report their own status. Every
stochastic method takes an explicit `seed`, and the same inputs, bounds,
options, and seed produce the same result. An omitted seed uses the fixed
default seed, so omitted seeds stay reproducible.

## Constraint semantics

An inequality constraint is feasible when its value is `>= 0`. An equality
constraint is feasible when its value is zero within the feasibility
tolerance. Every constrained result reports the largest constraint violation,
so a slightly infeasible point never looks feasible. Bounds stay separate from
general constraints, and each method uses the strongest native bound support it
has.

## Leaf solver decoupling

Leaf building and solver selection are independent axes. A leaf declares the
optimization problem it poses. A solver declares what it can solve. One
compatibility lookup resolves a leaf, a response family, and a solver into a
supported pairing or one precise rejection reason.

| Solver | Gradients | Derivative-free | L1 | Monotone | Box bounds | Equality | Nonlinear inequality | Memory |
|---|---|---|---|---|---|---|---|---|
| `irls` | yes | no | yes | yes | no | no | no | quadratic |
| `lbfgs` | yes | no | no | no | no | no | no | linear |
| `ccd` | no | yes | yes | yes | no | no | no | linear |
| `cmaes`, `shade`, `nes`, `ga` | no | yes | yes | yes | no | no | no | population |

Rejected pairings name the missing capability, for example
`solver Lbfgs does not support L1 penalties`.

Every fitted leaf persists two diagnostics: `solver_used` names the solver that
ran, or that it did not apply, and `leaf_problem_type` names the objective the
fit minimized. A closed-form leaf such as the weighted mean accepts any solver
selection and reports the solver as not applicable instead of failing the fit.

## Datasets

The benchmark Parquet files under `datasets/` are not tracked in git. Fetch
the suite once per checkout:

```sh
uv run python scripts/fetch_datasets.py
uv run python scripts/fetch_datasets.py --only diabetes airfoil_self_noise
```

One dataset can also arrive on demand through the loader, which caches it
under `~/.cache/mltreelib/datasets/`:

```python
from mltreelib.datasets import load_dataset

data = load_dataset("airfoil_self_noise")
```

`datasets/MANIFEST.md` lists every dataset with its OpenML identifier, task,
target, size, and license.

## Tuning on real data

`examples/optimize_generic.py` tunes `ModelTree` hyperparameters on a local
benchmark dataset and reports the holdout RMSE against a default tree. Both
methods receive the same evaluation budget.

```sh
uv run python examples/optimize_generic.py --dataset airfoil_self_noise --budget 30
```

```json
{
  "dataset": "airfoil_self_noise",
  "rows": 1503,
  "train_rows": 1202,
  "holdout_rows": 301,
  "budget": 30,
  "baseline_rmse": 3.8525345431114677,
  "methods": [
    {
      "method": "random_search",
      "status": "converged",
      "evaluations": 30,
      "best_rmse": 3.341695653133229
    },
    {
      "method": "latin_hypercube",
      "status": "converged",
      "evaluations": 30,
      "best_rmse": 3.07072735209262
    }
  ]
}
```

The real-data tests in `tests/test_optimize_real_data.py` optimize a bounded
two-parameter least-squares objective on the same file and check both sampling
methods against the closed-form least-squares optimum. Those tests skip when
the suite is not downloaded locally.
