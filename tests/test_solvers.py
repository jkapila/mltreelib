import pytest

from mltreelib import LeafControl, ModelTree
from mltreelib.solvers import fit


def test_standalone_weighted_gaussian_fit():
    X = [[0.0], [1.0], [2.0], [3.0]]
    result = fit(X, [1.0, 3.0, 5.0, 7.0], sample_weight=[1.0, 3.0, 2.0, 1.0])
    assert abs(result.intercept - 1.0) < 1e-10
    assert abs(result.coef[0] - 2.0) < 1e-10
    assert result.converged
    assert result.predict([[4.0]])[0] == pytest.approx(9.0)


def test_standalone_weighted_binomial_fit():
    X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    result = fit(
        X,
        [0.0, 0.0, 0.0, 1.0, 1.0],
        family="binomial",
        solver="lbfgs",
        l2=0.1,
        sample_weight=[1.0, 2.0, 1.0, 2.0, 1.0],
    )
    predictions = result.predict(X)
    assert all(left < right for left, right in zip(predictions, predictions[1:]))
    assert all(0.0 <= value <= 1.0 for value in predictions)


def test_leaf_control_solver_property():
    ctrl_default = LeafControl(leaf="linear")
    assert ctrl_default.leaf_solver == "irls"

    ctrl_lbfgs = LeafControl(leaf="linear", leaf_family="binomial", leaf_solver="lbfgs")
    assert ctrl_lbfgs.leaf_solver == "lbfgs"

    ctrl_ccd = LeafControl(leaf="linear", leaf_solver="ccd")
    assert ctrl_ccd.leaf_solver == "ccd"

    for solver_name in ["cmaes", "shade", "nes", "ga"]:
        ctrl = LeafControl(leaf="linear", leaf_solver=solver_name)
        assert ctrl.leaf_solver == solver_name


def test_model_tree_leaf_solvers_evolutionary():
    # Synthetic linear regression data
    X = [[float(i), float(i % 3)] for i in range(50)]
    y = [2.0 + 1.5 * row[0] for row in X]

    for solver in ["cmaes", "shade", "nes", "ga"]:
        tree = ModelTree(
            max_depth=1,
            min_rows=15,
            leaf="linear",
            leaf_solver=solver,
        )
        tree.fit(X, y)
        assert tree.n_leaves() >= 1
        preds = tree.predict([[0.0, 0.0], [10.0, 0.0]])
        assert preds[1] > preds[0]


def test_model_tree_leaf_solver_lbfgs_binomial():
    # Synthetic binary classification data
    X = [[(i - 50.0) * 0.1, (i % 5) * 0.2] for i in range(100)]
    y = [1.0 if row[0] > 0 else 0.0 for row in X]

    tree = ModelTree(
        max_depth=2,
        min_rows=15,
        leaf="linear",
        leaf_family="binomial",
        leaf_solver="lbfgs",
        l2=1e-3,
    )
    tree.fit(X, y)
    assert tree.n_leaves() >= 1

    # Monotonic predictions
    preds = tree.predict([[-2.0, 0.0], [2.0, 0.0]])
    assert preds[0] < preds[1]


def test_model_tree_leaf_solver_ccd_sparse_gaussian():
    # Synthetic regression where feature 1 is signal, feature 2 is noise
    X = [[float(i), float(i % 7)] for i in range(80)]
    y = [3.0 + 2.0 * row[0] for row in X]

    tree = ModelTree(
        max_depth=1,
        min_rows=20,
        leaf="linear",
        leaf_solver="ccd",
        l1=10.0,
        l2=0.1,
    )
    tree.fit(X, y)
    assert tree.n_leaves() >= 1
