import pytest
from mltreelib import ModelTree
from mltreelib._core import MarsRegressor


def test_modeltree_telemetry_verbose_0():
    x = [[float(i), float(i % 5)] for i in range(100)]
    y = [10.0 if i < 50 else 50.0 for i in range(100)]
    tree = ModelTree(max_depth=2, min_rows=10, verbose=0)
    tree.fit(x, y)
    assert len(tree.training_log()) == 0


def test_modeltree_telemetry_verbose_1_and_2():
    x = [[float(i), float(i % 5)] for i in range(100)]
    y = [10.0 if i < 50 else 50.0 for i in range(100)]
    tree = ModelTree(max_depth=2, min_rows=10, verbose=2)
    tree.fit(x, y)
    log = tree.training_log()
    assert len(log) > 0
    assert any("[ModelTree] fit start" in line for line in log)
    assert any("[ModelTree] depth" in line for line in log)
    assert any("[node depth=" in line for line in log)
    assert any("[split]" in line for line in log)
    assert any("[leaf]" in line for line in log)
    assert any("[ModelTree] fit complete" in line for line in log)


def test_mars_telemetry_verbose():
    x = [[float(i), float(i % 5)] for i in range(100)]
    y = [10.0 if i < 50 else 50.0 for i in range(100)]
    mars = MarsRegressor(degree=1, max_terms=5, verbose=1)
    mars.fit(x, y)
    preds = mars.predict(x)
    assert len(preds) == 100
