import numpy as np
import pytest
from mltreelib import ModelTree


def test_leaf_na_policy_invalid():
    with pytest.raises(ValueError, match="leaf_na_policy must be"):
        ModelTree(leaf_na_policy="non_existent_policy")


def test_leaf_na_policy_drop_fallback():
    # 20 samples: 16 complete, 4 NA with extreme y
    x = [[float(i)] for i in range(16)] + [[None]] * 4
    # complete rows: y = 2 + 3 * x
    y = [2.0 + 3.0 * float(i) for i in range(16)] + [100.0] * 4

    tree = ModelTree(max_depth=0, min_rows=5, l1=0.0, l2=0.0, leaf_na_policy="drop_fallback")
    tree.fit(x, y)

    leaves = tree.leaves()
    assert len(leaves) == 1
    # Check predictions:
    # Observed point evaluates linear fit (2 + 3 * 5 = 17)
    p_obs = tree.predict([[5.0]])[0]
    assert abs(p_obs - 17.0) < 1e-4

    # NA point falls back to 0D response mean: 792 / 20 = 39.6
    p_na = tree.predict([[None]])[0]
    assert abs(p_na - 39.6) < 1e-4


def test_leaf_na_policy_impute_local():
    # 20 samples:
    # Feature 0: 0, 2, ..., 28 (mean = 14.0) in 15 rows, 5 None
    # Feature 1: 100% None in all 20 rows (feature subsetting drops feature 1)
    x = [[float(i * 2), None] for i in range(15)] + [[None, None]] * 5
    # y = 5 + 2 * x0
    y = [5.0 + 2.0 * float(i * 2) for i in range(15)] + [33.0] * 5

    tree = ModelTree(max_depth=0, min_rows=5, l1=0.0, l2=0.0, leaf_na_policy="impute_local")
    tree.fit(x, y)

    # Prediction with NA in feature 0 imputes local leaf mean 14.0:
    # 5 + 2 * 14.0 = 33.0
    p_na = tree.predict([[None, None]])[0]
    assert abs(p_na - 33.0) < 1e-3


def test_leaf_na_policy_indicator():
    x = [[float(i)] for i in range(15)] + [[None]] * 5
    y = [2.0 + 3.0 * float(i) for i in range(15)] + [50.0] * 5

    tree = ModelTree(max_depth=0, min_rows=5, l1=0.0, l2=0.0, leaf_na_policy="indicator")
    tree.fit(x, y)

    # Indicator learns missingness displacement term
    p_na = tree.predict([[None]])[0]
    assert abs(p_na - 50.0) < 1e-3
