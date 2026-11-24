"""Tests for Spec 27 Phase 1: Cardinality-Adaptive Binned Splits in Python layer."""

import pytest
from mltreelib import GrowthControl, ModelTree


def test_growth_control_accepts_max_bins_and_min_bin_rows():
    gc = GrowthControl(max_bins=64, min_bin_rows=2, split_eval="histogram")
    assert gc.max_bins == 64
    assert gc.min_bin_rows == 2
    assert gc.split_eval == "histogram"

    tree = ModelTree(growth_control=gc, max_depth=2, min_rows=5, leaf="mean")
    assert tree.params["max_bins"] == 64
    assert tree.params["min_bin_rows"] == 2
    assert tree.params["split_eval"] == "histogram"


def test_flat_kwargs_accept_max_bins_and_min_bin_rows():
    tree = ModelTree(max_bins=128, min_bin_rows=5, split_eval="histogram")
    assert tree.params["max_bins"] == 128
    assert tree.params["min_bin_rows"] == 5
    assert tree.params["split_eval"] == "histogram"


def test_binned_histogram_fits_regression_gains():
    # 5 discrete feature values: 1, 2, 3, 4, 5
    X = [[float(i % 5 + 1)] for i in range(100)]
    y = [10.0 if row[0] <= 2.0 else 50.0 for row in X]

    for gain in ["sse", "friedman_mse", "sdr", "deviance"]:
        tree = ModelTree(
            split_eval="histogram",
            max_bins=64,
            gain=gain,
            leaf="mean",
            max_depth=2,
            min_rows=5,
        )
        tree.fit(X, y)
        assert tree.n_leaves() >= 2
        preds = tree.predict([[1.0], [5.0]])
        assert len(preds) == 2
        assert preds[0] < preds[1]


def test_binned_and_fast_equivalence_on_discrete_data():
    X = [[float(i % 5 + 1)] for i in range(100)]
    y = [10.0 if row[0] <= 2.0 else 50.0 for row in X]

    tree_fast = ModelTree(
        split_eval="fast",
        gain="sse",
        leaf="mean",
        max_depth=1,
        min_rows=5,
        split_precision=None,
    )
    tree_fast.fit(X, y)

    tree_hist = ModelTree(
        split_eval="histogram",
        max_bins=256,
        gain="sse",
        leaf="mean",
        max_depth=1,
        min_rows=5,
        split_precision=None,
    )
    tree_hist.fit(X, y)

    splits_fast = tree_fast.splits()
    splits_hist = tree_hist.splits()

    assert len(splits_fast) == 1
    assert len(splits_hist) == 1
    assert splits_fast[0]["col"] == splits_hist[0]["col"]
    assert splits_fast[0]["threshold"] == splits_hist[0]["threshold"]
    assert abs(splits_fast[0]["gain"] - splits_hist[0]["gain"]) < 1e-6
