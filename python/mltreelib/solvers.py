"""Fit a weighted Gaussian or generalized linear model without growing a tree.

Pass a complete numeric matrix and one response per row. Use nonnegative
case weights with positive total weight. An intercept is always included
and excluded from the penalty.

Use `direct` for Gaussian OLS or ridge, and `ccd` for Gaussian fits with
L1 or coefficient signs. Use `lbfgs` for smooth GLMs with L2. Use `irls`
for GLMs with L1 or sign constraints. `cmaes`, `shade`, `nes`, and `ga`
also support L1, L2, and coefficient signs across all families. The
stochastic solvers use `seed` for reproducibility.

CCD and GLM IRLS apply L1 and L2 to scaled coefficients when L1 is positive.
The remaining solvers penalize coefficients on the original scale.

`optimize` minimizes a callable scalar objective inside bounds. The objective
receives one list of parameter values and returns one finite scalar. Pass
`bounds` as one `(lower, upper)` pair per parameter. The optional `gradient`
callable returns one value per parameter; smooth methods use a
central-difference fallback when no gradient is supplied.
"""

from mltreelib._core import (
    AskTell,
    FitResult,
    OptimalTreeResult,
    OptimizationResult,
    PruningResult,
    SparseRuleResult,
    minimize_many,
    optimize,
    prune_ensemble,
    select_sparse_rules,
    solve_optimal_tree,
)
from mltreelib._core import fit_glm as fit

__all__ = [
    "AskTell",
    "FitResult",
    "OptimalTreeResult",
    "OptimizationResult",
    "PruningResult",
    "SparseRuleResult",
    "fit",
    "minimize_many",
    "optimize",
    "prune_ensemble",
    "select_sparse_rules",
    "solve_optimal_tree",
]
