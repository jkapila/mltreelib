"""Tests for fixes from the 2026-11-18 practical documentation and codebase audits."""

import mltreelib
from mltreelib import ModelTree


def test_package_version():
    """Verify package version is set."""
    assert mltreelib.__version__ == "0.4.14"


def test_break_down_and_shapley_accept_string_categories_and_nas():
    """ModelTree.break_down and shapley must handle raw string categories and None."""
    X = [
        ["north", 10.0],
        ["north", 20.0],
        ["south", 30.0],
        ["south", 40.0],
        ["south", None],
    ]
    y = [1.0, 2.0, 10.0, 12.0, 11.0]

    tree = ModelTree(max_depth=2, min_rows=2, leaf="mean")
    tree.fit(X, y, dtypes=["categorical", "numeric"])

    # Query with raw string category and numeric
    query_row = ["north", 15.0]
    bd = tree.break_down(query_row)
    assert "baseline" in bd
    assert "prediction" in bd
    assert "steps" in bd
    assert abs((bd["baseline"] + sum(s[1] for s in bd["steps"]) + bd["residual"]) - bd["prediction"]) < 1e-6

    # Test with None and another category
    query_row_na = ["south", None]
    bd_na = tree.break_down(query_row_na)
    assert "prediction" in bd_na

    # Shapley test
    sh = tree.shapley(query_row)
    assert "base_value" in sh
    assert "prediction" in sh
    assert "values" in sh
    assert abs((sh["base_value"] + sum(v[1] for v in sh["values"])) - sh["prediction"]) < 1e-6


def test_dot_split_boundary_operator():
    """DOT diagram must label left branch <= and right branch >."""
    X = [[float(i)] for i in range(20)]
    y = [0.0 if i < 10 else 10.0 for i in range(20)]
    tree = ModelTree(max_depth=1, min_rows=5, leaf="mean")
    tree.fit(X, y)

    dot = tree.to_dot()
    assert "<=" in dot
    assert ">" in dot
    assert "< " not in dot  # Should be <=, not <
