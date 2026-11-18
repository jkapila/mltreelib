import pytest
from mltreelib import ModelTree, LeafControl

def test_preset_accurate():
    m = ModelTree(preset="accurate")
    assert m.max_depth == 8
    assert m.min_rows == 10
    assert m.shrinkage == 10.0
    assert m.prune == "none"

def test_preset_fast():
    m = ModelTree(preset="fast")
    assert m.split_eval == "fast"
    assert m.max_depth == 5
    assert m.min_rows == 20
    assert m.shrinkage == 0.0

def test_preset_interpretable():
    m = ModelTree(preset="interpretable")
    assert m.max_depth == 3
    assert m.min_rows == 30
    assert m.shrinkage == 0.0

def test_default_values():
    m = ModelTree()
    assert m.max_depth == 6
    assert m.min_rows == 15
    assert m.shrinkage == 0.0
    assert m.max_classes == 256

def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        ModelTree(preset="invalid_preset_name")

def test_accurate_preset_fit():
    # Synthetic piecewise data
    x = [[float(i)] for i in range(100)]
    y = [float(2 * i if i < 50 else 3 * i) for i in range(100)]
    m = ModelTree(preset="accurate")
    m.fit(x, y)
    preds = m.predict([[10.0], [40.0], [60.0], [90.0]])
    assert len(preds) == 4
    assert preds[0] < preds[1] < preds[2] < preds[3]

def test_multiclass_histogram_fit():
    # 3 classes: [0, 40) -> 0, [40, 80) -> 1, [80, 120) -> 2
    x = [[float(i)] for i in range(120)]
    y = [float(0 if i < 40 else (1 if i < 80 else 2)) for i in range(120)]
    m = ModelTree(gain="gini", n_classes=3, max_depth=3, min_rows=10, preset="fast")
    m.fit(x, y)
    preds = m.predict([[10.0], [60.0], [100.0]])
    assert preds[0] < preds[1] < preds[2]

def test_leaf_control_shrinkage():
    ctrl = LeafControl(shrinkage=5.0, max_classes=128)
    m = ModelTree(control=ctrl)
    assert m.shrinkage == 5.0
    assert m.max_classes == 128
