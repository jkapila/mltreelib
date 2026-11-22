import pytest
from mltreelib.solvers import solve_optimal_tree, solve_oct_mip_cvx


def test_solve_optimal_tree_stump_classification():
    # 20 samples, 2 features
    # feature 0 < 10 -> class 0, feature 0 >= 10 -> class 1
    x = [[float(i), float(i % 3)] for i in range(20)]
    y = [0.0 if i < 10 else 1.0 for i in range(20)]

    tree = solve_optimal_tree(x, y, task="classification", max_depth=1)
    assert tree.max_depth == 1
    assert tree.n_splits == 1
    assert tree.total_loss == 0.0

    preds = tree.predict(x)
    assert preds == y

    # Test JSON serialization
    json_str = tree.to_json()
    assert "nodes" in json_str


def test_solve_optimal_tree_xor_depth2():
    # 2D XOR pattern: requires depth 2 tree
    grid = [
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (2.0, 8.0, 1.0),
        (3.0, 7.0, 1.0),
        (8.0, 2.0, 1.0),
        (7.0, 3.0, 1.0),
        (8.0, 8.0, 0.0),
        (7.0, 7.0, 0.0),
    ]
    x = [[x0, x1] for (x0, x1, _) in grid]
    y = [label for (_, _, label) in grid]

    tree = solve_optimal_tree(x, y, task="classification", max_depth=2, min_samples_leaf=1)
    assert tree.max_depth == 2
    assert tree.total_loss == 0.0

    preds = tree.predict(x)
    assert preds == y


def test_solve_optimal_tree_regression():
    # 30 samples, 3 levels
    x = [[float(i)] for i in range(30)]
    y = [2.0 if i < 10 else (10.0 if i < 20 else 25.0) for i in range(30)]

    tree = solve_optimal_tree(x, y, task="regression", max_depth=2, min_samples_leaf=2)
    assert tree.max_depth == 2
    assert tree.total_loss < 1e-4

    preds = tree.predict(x)
    for p, actual in zip(preds, y):
        assert abs(p - actual) < 1e-4


def test_solve_oct_mip_cvx_or_importerror():
    try:
        import cvxpy  # noqa: F401

        has_cvxpy = True
    except ImportError:
        has_cvxpy = False

    x = [[1.0], [2.0], [8.0], [9.0]]
    y = [0.0, 0.0, 1.0, 1.0]

    if has_cvxpy:
        feat, thresh, left_val, right_val = solve_oct_mip_cvx(x, y, task="classification")
        assert feat == 0
        assert 2.0 <= thresh <= 8.0
        assert left_val == 0.0
        assert right_val == 1.0
    else:
        with pytest.raises(ImportError, match="cvxpy is required"):
            solve_oct_mip_cvx(x, y)
