"""Tests for frozen-topology retraining on decision stumps."""

import pytest
from mltreelib import ModelTree, LeafControl


@pytest.fixture
def two_regime_data():
    """Synthetic 2-regime dataset partitioned by x0 at 0.0."""
    n = 200
    x = []
    y_reg = []
    for i in range(n):
        x0 = (i - 100) / 50.0  # Range -2.0 to +1.98
        x1 = (i % 10) * 0.5     # Range 0.0 to 4.5
        if x0 < 0.0:
            y_val = 10.0 + 2.0 * x1
        else:
            y_val = 50.0 + 5.0 * x1
        x.append([x0, x1])
        y_reg.append(y_val)
    return x, y_reg


def test_stump_retrain_frozen_topology(two_regime_data):
    x, y_reg = two_regime_data
    stump = ModelTree(max_depth=1, min_rows=10, leaf="linear", l2=1e-3, prune="none")
    stump.fit(x, y_reg)

    orig_splits = stump.splits()
    assert len(orig_splits) == 1
    split_col = orig_splits[0]["col"]
    split_thresh = orig_splits[0]["threshold"]
    assert split_col == "x0"

    test_points = [[-1.0, 2.0], [1.0, 2.0]]

    # 1. Retrain to Mean leaf
    stump.retrain(x, y_reg, control=LeafControl(leaf="mean"))
    splits_after = stump.splits()
    assert len(splits_after) == 1
    assert splits_after[0]["col"] == split_col
    assert splits_after[0]["threshold"] == split_thresh

    preds_mean = stump.predict(test_points)
    # Piecewise constant per leaf
    assert preds_mean[0] == pytest.approx(14.5, abs=0.1)
    assert preds_mean[1] == pytest.approx(61.25, abs=0.1)

    # 2. Retrain to Linear leaf with unpenalized OLS
    stump.retrain(x, y_reg, control=LeafControl(leaf="linear", l2=1e-5))
    leaves = stump.leaves()
    assert leaves[0]["intercept"] == pytest.approx(10.0, abs=1e-3)
    assert leaves[0]["coef"][1] == pytest.approx(2.0, abs=1e-3)
    assert leaves[1]["intercept"] == pytest.approx(50.0, abs=1e-3)
    assert leaves[1]["coef"][1] == pytest.approx(5.0, abs=1e-3)

    preds_linear = stump.predict(test_points)
    assert preds_linear[0] == pytest.approx(14.0, abs=1e-3)
    assert preds_linear[1] == pytest.approx(60.0, abs=1e-3)

    # 3. Retrain to heavy Ridge leaf (coefficients shrink)
    stump.retrain(x, y_reg, leaf="linear", l2=100.0)
    leaves_ridge = stump.leaves()
    assert leaves_ridge[0]["coef"][1] < leaves[0]["coef"][1]
    assert leaves_ridge[1]["coef"][1] < leaves[1]["coef"][1]

    # 4. Retrain with Quantiles and Leaf Digest
    stump.retrain(x, y_reg, leaf_digest=True)
    q10 = stump.quantile(test_points, 0.10)
    q50 = stump.quantile(test_points, 0.50)
    q90 = stump.quantile(test_points, 0.90)
    assert q10[0] <= q50[0] <= q90[0]
    assert q10[1] <= q50[1] <= q90[1]

    # 5. Retrain with Binomial family
    y_bin = [0.0 if i < 100 else 1.0 for i in range(len(x))]
    stump.retrain(x, y_bin, leaf_family="binomial", leaf="mean")
    probs = stump.predict_proba(test_points)
    labels = stump.predict_label(test_points)
    assert probs == [0.0, 1.0]
    assert labels == [0, 1]

    # 6. Retrain with Poisson family
    y_pois = [3.0 if i < 100 else 30.0 for i in range(len(x))]
    stump.retrain(x, y_pois, leaf_family="poisson", leaf="mean")
    rates = stump.predict(test_points)
    assert rates[0] == pytest.approx(3.0, abs=1e-3)
    assert rates[1] == pytest.approx(30.0, abs=1e-3)
