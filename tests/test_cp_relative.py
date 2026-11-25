"""Scale-free cost-complexity pruning: `cp` is a fraction of the root cost."""

import numpy as np
import pytest

from mltreelib import ModelTree


def _data(scale: float = 1.0, n: int = 600):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, 4))
    y = (2.0 * X[:, 0] + np.where(X[:, 1] > 0, 3.0, -3.0) + rng.normal(0, 1.0, n)) * scale
    return X.tolist(), y.tolist()


def _fit(X, y, **kw) -> ModelTree:
    model = ModelTree(**kw)
    model.fit(X, y)
    return model


def _leaves(model: ModelTree) -> int:
    return int(model.n_leaves())


def test_cp_is_invariant_to_target_scale():
    X, y = _data(1.0)
    _, y_big = _data(1000.0)
    a = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp=0.02)
    b = _fit(X, y_big, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp=0.02)
    assert _leaves(a) == _leaves(b)


def test_absolute_cp_alpha_depends_on_target_scale():
    X, y = _data(1.0)
    _, y_big = _data(1000.0)
    a = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp_alpha=50.0)
    b = _fit(X, y_big, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp_alpha=50.0)
    assert _leaves(b) > _leaves(a)


def test_cp_zero_does_not_prune_and_larger_cp_prunes_more():
    X, y = _data()
    none = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="none")
    zero = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp=0.0)
    small = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp=0.01)
    big = _fit(X, y, leaf="mean", max_depth=6, min_rows=10, prune="cp", cp=0.2)
    assert _leaves(zero) == _leaves(none)
    assert _leaves(big) <= _leaves(small) <= _leaves(zero)
    assert _leaves(big) < _leaves(zero)


def test_cp_overrides_cp_alpha_and_rejects_negative():
    X, y = _data()
    a = _fit(X, y, leaf="mean", max_depth=5, min_rows=10, prune="cp", cp=0.05, cp_alpha=0.0)
    b = _fit(X, y, leaf="mean", max_depth=5, min_rows=10, prune="cp", cp=0.05, cp_alpha=1e9)
    assert _leaves(a) == _leaves(b)
    with pytest.raises(ValueError):
        _fit(X, y, prune="cp", cp=-0.1)


def test_cp_survives_json_roundtrip():
    X, y = _data()
    m = _fit(X, y, leaf="mean", max_depth=5, min_rows=10, prune="cp", cp=0.05)
    m2 = ModelTree.from_json(m.to_json())
    assert m2.predict(X[:20]) == m.predict(X[:20])
