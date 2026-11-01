"""Tests for flexible retrain, control dictionary, mode (internal/external), and id_col in leaf_samples."""

import pytest
from mltreelib import ModelTree, LeafControl


@pytest.fixture
def cohort_micro_data():
    """Dataset with macro cohort features (x0, x1) and micro vehicle features (mileage, age)."""
    n = 100
    x = []
    y = []
    vins = []
    for i in range(n):
        x0 = (i - 50) / 25.0       # Split feature: regime -2.0 to 1.96
        x1 = float(i % 5)          # Macro category/cohort
        mileage = float(i * 100)   # Micro feature 1
        age = float(i % 10 + 1)    # Micro feature 2
        vin = f"VIN_{i:04d}"

        # Target: regime determines intercept, micro features determine slope
        if x0 < 0.0:
            target = 100.0 + 0.05 * mileage + 2.0 * age
        else:
            target = 500.0 + 0.10 * mileage + 5.0 * age

        x.append([x0, x1, mileage, age])
        y.append(target)
        vins.append(vin)

    col_names = ["x0", "x1", "mileage", "age"]
    return x, y, vins, col_names


def test_retrain_control_dict(cohort_micro_data):
    x, y, _, col_names = cohort_micro_data
    # Fit initial tree on macro features
    tree = ModelTree(max_depth=1, min_rows=10, leaf="mean", prune="none")
    tree.fit([r[:2] for r in x], y)

    assert tree.n_leaves() == 2

    # Retrain using control as a clean dictionary
    ctrl_dict = {
        "leaf": "linear",
        "family": "gaussian",
        "l2": 1e-4,
    }
    ret = tree.retrain([r[:2] for r in x], y, control=ctrl_dict)
    assert ret is tree  # internal mode returns self

    leaves = tree.leaves()
    assert leaves[0]["intercept"] is not None
    assert len(leaves[0]["coef"]) == 2


def test_retrain_mode_external_vs_internal(cohort_micro_data):
    x, y, _, col_names = cohort_micro_data
    # Base macro tree with constant mean leaves
    tree = ModelTree(max_depth=1, min_rows=10, leaf="mean", prune="none")
    tree.fit([r[:2] for r in x], y)

    orig_leaves = tree.leaves()
    assert orig_leaves[0]["intercept"] is None  # Mean leaf has no linear intercept

    # External mode: leaves retrained to linear on new data/columns, original tree untouched
    new_cols = ["x0", "x1", "mileage", "age"]
    new_model = tree.retrain(
        x,
        y,
        mode="external",
        col_names=new_cols,
        leaf_features=["mileage", "age"],
        leaf="linear",
        l2=1e-3,
    )

    # 1. Verify original tree is UNTOUCHED
    assert new_model is not tree
    leaves_orig_after = tree.leaves()
    assert leaves_orig_after[0]["intercept"] is None
    assert tree.splits()[0]["col"] == "x0"

    # 2. Verify new_model has retrained micro leaves using selected micro features
    new_leaves = new_model.leaves()
    assert new_leaves[0]["intercept"] is not None
    # Leaves use mileage and age (2 features)
    assert len(new_leaves[0]["coef"]) == 2
    assert new_leaves[0]["cols"] == ["mileage", "age"]
    assert new_leaves[1]["cols"] == ["mileage", "age"]

    # Predict with new model on full 4-feature test points
    test_pt_left = [[-1.0, 0.0, 1000.0, 5.0]]
    test_pt_right = [[1.0, 0.0, 1000.0, 5.0]]
    pred_left = new_model.predict(test_pt_left)[0]
    pred_right = new_model.predict(test_pt_right)[0]

    # Ground truth: left=100 + 0.05*1000 + 2*5 = 160. right=500 + 0.10*1000 + 5*5 = 625.
    assert pred_left == pytest.approx(160.0, abs=5.0)
    assert pred_right == pytest.approx(625.0, abs=5.0)


def test_retrain_leaf_features_subsets_and_aliases(cohort_micro_data):
    x, y, _, col_names = cohort_micro_data
    tree = ModelTree(max_depth=1, min_rows=10, leaf="mean", prune="none")
    tree.fit([r[:2] for r in x], y)

    # Use retrain_cols alias with integer column indices
    new_model = tree.retrain(
        x,
        y,
        mode="external",
        col_names=col_names,
        retrain_cols=[2],  # only mileage
        leaf="linear",
        l2=1e-3,
    )
    leaves = new_model.leaves()
    assert leaves[0]["cols"] == ["mileage"]
    assert len(leaves[0]["coef"]) == 1


def test_retrain_split_column_validation():
    # Tree splits on col_A
    x_train = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]] * 5
    y_train = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0] * 5
    tree = ModelTree(max_depth=1, min_rows=3, prune="none")
    tree.fit(x_train, y_train)

    # Retrain data is missing the split column x0
    x_new = [[10.0, 20.0]] * 30
    y_new = [1.0] * 30
    with pytest.raises(ValueError, match="required split column 'x0' missing"):
        tree.retrain(x_new, y_new, col_names=["other_col", "another_col"])


def test_leaf_samples_id_col(cohort_micro_data):
    x, y, vins, col_names = cohort_micro_data
    tree = ModelTree(max_depth=1, min_rows=10, leaf="mean", prune="none")
    tree.fit([r[:2] for r in x], y)

    # 1. Default: id_col is None -> integer row indices
    samples_default = tree.leaf_samples([r[:2] for r in x])
    assert set(samples_default.keys()) == {0, 1}
    assert isinstance(samples_default[0][0], int)
    assert samples_default[0] == list(range(50))
    assert samples_default[1] == list(range(50, 100))

    # 2. id_col supplied as external list/sequence of IDs
    samples_vins = tree.leaf_samples([r[:2] for r in x], id_col=vins)
    assert set(samples_vins.keys()) == {0, 1}
    assert samples_vins[0] == [f"VIN_{i:04d}" for i in range(50)]
    assert samples_vins[1] == [f"VIN_{i:04d}" for i in range(50, 100)]

    # 3. id_col supplied as column index in data
    # Create dataset where VIN is column 0 and features are 1 and 2
    x_with_vin = [[vins[i], r[0], r[1]] for i, r in enumerate(x)]
    samples_idx = tree.leaf_samples(
        [[r[1], r[2]] for r in x_with_vin],
        id_col=vins
    )
    assert samples_idx[0][:3] == ["VIN_0000", "VIN_0001", "VIN_0002"]


def test_fit_with_control_dict():
    x = [[float(i), float(i % 3)] for i in range(40)]
    y = [float(i * 2) for i in range(40)]

    tree = ModelTree(max_depth=1, min_rows=5, prune="none")
    tree.fit(x, y, control={"leaf": "mean"})

    leaves = tree.leaves()
    assert len(leaves) >= 1
    assert leaves[0]["intercept"] is None  # Mean leaf
