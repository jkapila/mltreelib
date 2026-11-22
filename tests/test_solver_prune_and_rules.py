import pytest
import math
from mltreelib.solvers import (
    prune_ensemble,
    select_sparse_rules,
    solve_knapsack_pruning_cvx,
    solve_sparse_rules_cvx,
)


def test_prune_ensemble_stepwise():
    # 50 samples, 8 candidate models
    # Models 0 and 1 are correlated with y; Models 2..7 are random noise
    n = 60
    y = [10.0 + 2.0 * (i % 5) - 1.5 * (i % 3) for i in range(n)]

    # Model 0: y + slight noise
    m0 = [y[i] + 0.1 * ((i % 7) - 3) for i in range(n)]
    # Model 1: y + opposite noise
    m1 = [y[i] - 0.1 * ((i % 7) - 3) for i in range(n)]
    # Models 2..7: uncorrelated
    tree_preds = [m0, m1]
    for k in range(2, 8):
        tree_preds.append([float((i * k) % 17) for i in range(n)])

    # Transpose to N samples x M trees
    preds = [[tree_preds[m][i] for m in range(8)] for i in range(n)]

    res = prune_ensemble(preds, y, max_trees=4, tol=1e-5)

    assert len(res.selected_indices) > 0
    assert len(res.selected_indices) <= 4
    assert len(res.weights) == len(res.selected_indices)
    assert abs(sum(res.weights) - 1.0) < 1e-6
    # Model 0 and 1 should dominate selection
    assert 0 in res.selected_indices or 1 in res.selected_indices
    # Loss should decrease
    assert res.final_loss < res.loss_history[0] or math.isclose(res.final_loss, res.loss_history[0])


def test_select_sparse_rules_lasso():
    # 80 samples, 6 rules
    n = 80
    # Rule 0 active when i % 2 == 0
    r0 = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    # Rule 1 active when i % 3 == 0
    r1 = [1.0 if i % 3 == 0 else 0.0 for i in range(n)]
    # Rules 2..5 random or uninformative
    r2 = [1.0 if i % 5 == 0 else 0.0 for i in range(n)]
    r3 = [1.0 if i % 7 == 0 else 0.0 for i in range(n)]

    # y = 1.5 + 4.0 * r0 - 3.0 * r1
    y = [1.5 + 4.0 * r0[i] - 3.0 * r1[i] for i in range(n)]

    # rule_matrix: N rows x M cols
    rule_matrix = [[r0[i], r1[i], r2[i], r3[i]] for i in range(n)]

    res = select_sparse_rules(rule_matrix, y, alpha=1.0, lambda_reg=0.01)

    assert 0 in res.selected_indices
    assert 1 in res.selected_indices
    # Check intercept close to 1.5
    assert abs(res.intercept - 1.5) < 0.5

    # Check predictions
    preds = res.predict(rule_matrix)
    assert len(preds) == n
    mse = sum((preds[i] - y[i]) ** 2 for i in range(n)) / n
    assert mse < 0.2

    # Check importance is positive for selected rules
    for imp in res.importances:
        assert imp > 0.0


def test_cvxpy_solvers_or_importerror():
    try:
        import cvxpy  # noqa: F401

        has_cvxpy = True
    except ImportError:
        has_cvxpy = False

    rule_matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    y = [2.0, -1.0, 1.0, 0.0]

    if has_cvxpy:
        w0, w = solve_sparse_rules_cvx(rule_matrix, y, alpha=1.0, lambda_reg=0.01)
        assert isinstance(w0, float)
        assert len(w) == 2

        preds = [[2.0, 1.0], [-1.0, 0.0], [1.0, 2.0], [0.0, -1.0]]
        chosen, weights = solve_knapsack_pruning_cvx(preds, y, max_trees=2)
        assert len(chosen) <= 2
        assert len(weights) == len(chosen)
    else:
        with pytest.raises(ImportError, match="cvxpy is required"):
            solve_sparse_rules_cvx(rule_matrix, y)

        with pytest.raises(ImportError, match="cvxpy is required"):
            solve_knapsack_pruning_cvx([[1.0]], [1.0])
