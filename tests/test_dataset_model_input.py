"""Models accept a Dataset directly: roles, names and dtypes travel with the data."""

import pytest
from mltreelib import ModelTree
from mltreelib.data import Dataset


def _columns(n=240):
    import random

    rng = random.Random(4)
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [None if i % 11 == 0 else rng.gauss(0, 1) for i in range(n)]
    c = [rng.choice(["u", "v"]) for _ in range(n)]
    y = [2 * a[i] + (1.0 if c[i] == "u" else 0.0) + rng.gauss(0, 0.1) for i in range(n)]
    return {"a": a, "b": b, "c": c, "y": y}


def test_fit_and_predict_accept_a_dataset_and_match_the_hand_path():
    cols = _columns()
    ds = Dataset(cols, target="y")
    train, test = ds.split(test_size=0.25, seed=3)
    by_hand = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    by_hand.fit(train.x, train.y, col_names=train.names, dtypes=train.dtypes)
    direct = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    direct.fit(train)
    assert str(direct) == str(by_hand)
    assert direct.predict(test) == by_hand.predict(test.x)


def test_explicit_arguments_override_dataset_values():
    ds = Dataset(_columns(), target="y")
    m = ModelTree(leaf="mean", max_depth=2, min_rows=10)
    m.fit(ds, col_names=["p", "q", "r"])
    assert "p" in str(m) or "q" in str(m) or "r" in str(m)


def test_missing_target_is_a_clear_error():
    ds = Dataset(_columns())  # no target role
    with pytest.raises(ValueError, match="needs a target"):
        ModelTree(leaf="mean").fit(ds)


def test_non_numeric_treatment_is_rejected_not_recoded():
    cols = _columns()
    cols["arm"] = ["Women" if i % 2 else "None" for i in range(len(cols["y"]))]
    ds = Dataset(cols, target="y", treatment="arm")
    with pytest.raises(ValueError, match="never recodes"):
        ModelTree(leaf="mean").fit(ds)


def test_numeric_treatment_flows_through():
    cols = _columns()
    cols["arm"] = [float(i % 2) for i in range(len(cols["y"]))]
    ds = Dataset(cols, target="y", treatment="arm")
    m = ModelTree(gain="causal", leaf="mean", max_depth=2, min_rows=20)
    m.fit(ds)
    assert len(m.predict(ds, kind="uplift")) == len(cols["y"])
