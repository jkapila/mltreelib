"""Exact reference solvers built on CVXPY, for tests only.

mltreelib has no CVXPY dependency. These functions solve the same problems as
the native solvers (`select_sparse_rules`, `prune_ensemble`, `solve_optimal_tree`)
exactly with convex and mixed-integer programs, so the tests can cross-check the
native answers. Run the tests that use them with:

    uv run --with cvxpy pytest tests/test_solver_oct.py tests/test_solver_prune_and_rules.py
"""

from __future__ import annotations

from collections.abc import Sequence


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
            "Run the reference tests with: uv run --with cvxpy pytest tests/test_solver_oct.py tests/test_solver_prune_and_rules.py, "
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
            "Run the reference tests with: uv run --with cvxpy pytest tests/test_solver_oct.py tests/test_solver_prune_and_rules.py, "
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
    # L1 error yields a Mixed-Integer Linear Program (MILP), solvable by open-source HiGHS / CBC / GLPK
    loss = cp.norm1(residual) / float(n)

    problem = cp.Problem(cp.Minimize(loss), constraints)
    if solver is not None:
        problem.solve(solver=solver)
    elif "HIGHS" in cp.installed_solvers():
        problem.solve(solver="HIGHS")
    else:
        problem.solve()

    if z.value is None or w.value is None:
        raise RuntimeError(f"CVXPY knapsack pruning failed with status {problem.status!r}")

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

        cp = importlib.import_module("cvxpy")
        import numpy as np
    except ImportError as err:
        raise ImportError(
            "cvxpy is required for solve_oct_mip_cvx. "
            "Run the reference tests with: uv run --with cvxpy pytest tests/test_solver_oct.py tests/test_solver_prune_and_rules.py, "
            "or use native mltreelib.solvers.solve_optimal_tree."
        ) from err

    X_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, p = X_arr.shape

    x_min = float(np.min(X_arr))
    x_max = float(np.max(X_arr))
    m_x = max(abs(x_min), abs(x_max)) * 2.0 + 2.0

    # Smallest positive gap between adjacent feature values so strict '>' is exact.
    min_gap = 1.0
    for j in range(p):
        uniq = np.unique(X_arr[:, j])
        if len(uniq) > 1:
            diffs = np.diff(uniq)
            pos = diffs[diffs > 0]
            if len(pos) > 0:
                min_gap = min(min_gap, float(np.min(pos)))
    eps = 0.25 * min_gap

    d = cp.Variable(p, boolean=True)
    b = cp.Variable()
    z = cp.Variable(n, boolean=True)  # z[i] == 1 -> left (<= b), z[i] == 0 -> right (> b)
    pred = cp.Variable(n)

    x_proj = X_arr @ d
    constraints = [
        cp.sum(d) == 1,
        b >= x_min - 1.0,
        b <= x_max + 1.0,
        x_proj <= b + m_x * (1 - z),
        x_proj >= b + eps - m_x * z,
    ]

    if task == "classification":
        c_left = cp.Variable(boolean=True)
        c_right = cp.Variable(boolean=True)
        constraints += [
            pred >= c_left - (1 - z),
            pred <= c_left + (1 - z),
            pred >= c_right - z,
            pred <= c_right + z,
        ]
        err = cp.Variable(n, nonneg=True)
        constraints += [err >= y_arr - pred, err >= pred - y_arr]
        objective = cp.Minimize(cp.sum(err))
    else:
        m_y = float(np.max(np.abs(y_arr))) * 2.0 + 2.0
        c_left = cp.Variable()
        c_right = cp.Variable()
        constraints += [
            pred >= c_left - m_y * (1 - z),
            pred <= c_left + m_y * (1 - z),
            pred >= c_right - m_y * z,
            pred <= c_right + m_y * z,
        ]
        objective = cp.Minimize(cp.sum(cp.abs(y_arr - pred)))

    problem = cp.Problem(objective, constraints)
    if solver is not None:
        problem.solve(solver=solver)
    elif "HIGHS" in cp.installed_solvers():
        problem.solve(solver="HIGHS")
    else:
        problem.solve()

    if d.value is None or b.value is None or c_left.value is None or c_right.value is None:
        raise RuntimeError(f"CVXPY OCT stump MIP failed with status {problem.status!r}")

    best_feat = int(np.argmax(d.value))
    best_thresh = float(b.value)
    return best_feat, best_thresh, float(c_left.value), float(c_right.value)
