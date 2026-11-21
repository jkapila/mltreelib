# Generic optimizer and real-data tuning

`mltreelib.solvers.optimize` minimizes a callable scalar objective inside
bounds. It is the public entry point for the shared optimization seam in the
Rust core. One objective, bounds, budget, request, and result contract backs
every method, so results never change shape when the method changes.

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

The objective receives one list of parameter values and returns one finite
scalar. `bounds` holds one `(lower, upper)` pair per parameter. Pass `start`
to warm-start a search, `initial_population` to hand population methods their
first candidates, and `gradient` to supply exact derivatives. Smooth methods
fall back to central differences when no gradient is supplied.

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

## Methods

Available now:

- `random_search` samples uniformly inside the bounds.
- `latin_hypercube` stratifies every dimension independently and shuffles the
  strata with the seeded generator.

Later waves from
[Spec 26](https://github.com/jkapila/mltreelib/blob/main/specs/26-generic-global-and-constrained-optimizers.md) add
`lbfgs_b`, `bfgs`, `nonlinear_cg`, `nelder_mead`, `powell`,
`simulated_annealing`, `cobyla`, and `basin_hopping`. An unknown or
unimplemented method raises one error naming the method. The library never
substitutes another algorithm silently.

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

## Tuning on real data

`examples/optimize_generic.py` tunes `ModelTree` hyperparameters on a local
TabArena Parquet file and reports the holdout RMSE against a default tree.
Both methods receive the same evaluation budget.

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

The Parquet files under `datasets/` are not tracked in git. Download the
TabArena suite before running the script or the real-data tests.
