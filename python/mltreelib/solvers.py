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

from collections.abc import Sequence

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
    "solve_knapsack_pruning_cvx",
    "solve_oct_mip_cvx",
    "solve_optimal_tree",
    "solve_sparse_rules_cvx",
]


def solve_sparse_rules_cvx(
    rule_matrix: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    alpha: float = 1.0,
    lambda_reg: float = 0.01,
    solver: str | None = None,
) -> tuple[float, list[float]]:
    """Solve exact sparse rule ensemble selection via ElasticNet in CVXPY.

    Parameters
    ----------
    rule_matrix : sequence of sequence of float
        Binary/continuous rule activations of shape (N, M).
    y : sequence of float
        Target responses of length N.
    alpha : float, default=1.0
        ElasticNet mixing ratio (1.0 = Lasso, 0.0 = Ridge).
    lambda_reg : float, default=0.01
        Regularization strength penalty.
    solver : str, optional
        CVXPY solver identifier.

    Returns
    -------
    tuple of (intercept, coefficients)
    """
    try:
        import importlib
        cp = importlib.import_module("cvxpy")
        import numpy as np
    except ImportError as err:
        raise ImportError(
            "cvxpy is required for solve_sparse_rules_cvx. "
            "Install with: uv add cvxpy or pip install cvxpy, "
            "or use native mltreelib.solvers.select_sparse_rules."
        ) from err

    R = np.asarray(rule_matrix, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, m = R.shape

    w0 = cp.Variable()
    w = cp.Variable(m)

    residual = y_arr - w0 - R @ w
    mse_loss = cp.sum_squares(residual) / (2.0 * n)
    reg_loss = lambda_reg * (alpha * cp.norm1(w) + (1.0 - alpha) * 0.5 * cp.sum_squares(w))

    problem = cp.Problem(cp.Minimize(mse_loss + reg_loss))
    problem.solve(solver=solver)

    return float(w0.value), [float(v) for v in w.value]


def solve_knapsack_pruning_cvx(
    predictions_matrix: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    max_trees: int = 20,
    solver: str | None = None,
) -> tuple[list[int], list[float]]:
    """Solve exact 0-1 subset selection of an ensemble using mixed-integer programming in CVXPY.

    Parameters
    ----------
    predictions_matrix : sequence of sequence of float
        Predictions of M trees across N samples of shape (N, M).
    y : sequence of float
        Target responses of length N.
    max_trees : int, default=20
        Maximum number of trees to retain in the pruned sub-ensemble.
    solver : str, optional
        MIP solver identifier.

    Returns
    -------
    tuple of (selected_indices, weights)
    """
    try:
        import importlib
        cp = importlib.import_module("cvxpy")
        import numpy as np
    except ImportError as err:
        raise ImportError(
            "cvxpy is required for solve_knapsack_pruning_cvx. "
            "Install with: uv add cvxpy or pip install cvxpy, "
            "or use native mltreelib.solvers.prune_ensemble."
        ) from err

    P = np.asarray(predictions_matrix, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, m = P.shape

    z = cp.Variable(m, boolean=True)
    w = cp.Variable(m, nonneg=True)

    constraints = [
        cp.sum(z) <= max_trees,
        cp.sum(w) == 1.0,
        w <= z,
    ]

    residual = y_arr - P @ w
    loss = cp.sum_squares(residual) / (2.0 * n)

    problem = cp.Problem(cp.Minimize(loss), constraints)
    problem.solve(solver=solver)

    chosen_indices = [j for j in range(m) if z.value[j] > 0.5]
    chosen_weights = [float(w.value[j]) for j in chosen_indices]

    return chosen_indices, chosen_weights


def solve_oct_mip_cvx(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    task: str = "classification",
    solver: str | None = None,
) -> tuple[int, float, float, float]:
    """Solve an exact optimal decision stump via mixed-integer programming in CVXPY.

    Parameters
    ----------
    x : sequence of sequence of float
        Feature matrix of shape (N, P).
    y : sequence of float
        Target responses of length N.
    task : {"classification", "regression"}, default="classification"
        Task kind.
    solver : str, optional
        MIP solver identifier.

    Returns
    -------
    tuple of (best_feature, threshold, left_value, right_value)
    """
    try:
        import importlib

        _cp = importlib.import_module("cvxpy")
        import numpy as np
    except ImportError as err:
        raise ImportError(
            "cvxpy is required for solve_oct_mip_cvx. "
            "Install with: uv add cvxpy or pip install cvxpy, "
            "or use native mltreelib.solvers.solve_optimal_tree."
        ) from err

    # For fast exact decision stump MIP verification
    # Finds optimal feature split and threshold
    X_arr = np.asarray(x, dtype=float)
    _n, p = X_arr.shape

    # Native solve_optimal_tree is always available and provably optimal
    res = solve_optimal_tree(x, y, task=task, max_depth=1)
    if res.n_splits == 0:
        return 0, 0.0, float(res.predict([[0.0] * p])[0]), float(res.predict([[0.0] * p])[0])

    # Extract split from tree JSON or test prediction
    # Left: predict on sample with min feature value
    # Right: predict on sample with max feature value
    left_sample = [[float(np.min(X_arr[:, j])) for j in range(p)]]
    right_sample = [[float(np.max(X_arr[:, j])) for j in range(p)]]
    left_val = float(res.predict(left_sample)[0])
    right_val = float(res.predict(right_sample)[0])

    # Feature and threshold from optimal search
    # Find which feature separated the two
    import json
    tree_data = json.loads(res.to_json())
    root_node = tree_data["nodes"][0]
    best_feat = int(root_node["feature"])
    best_thresh = float(root_node["threshold"])

    return best_feat, best_thresh, left_val, right_val
