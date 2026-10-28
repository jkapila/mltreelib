"""Behavior tests for the dataset component through the Python bindings."""

import pytest

from mltreelib import IndexMapper, generate, grouped_transform, summarize


def test_generate_is_deterministic_per_seed():
    a = generate(n_rows=300, seed=7)
    b = generate(n_rows=300, seed=7)
    c = generate(n_rows=300, seed=8)
    assert a["target"] == b["target"]
    assert a["target"] != c["target"]


def test_generate_shapes_and_columns():
    g = generate(
        n_rows=500,
        n_numeric=3,
        cardinalities=[4, 20],
        hierarchy_levels=2,
        branching=[3, 5],
        seed=1,
    )
    cols = g["columns"]
    assert set(cols) == {
        "num_0", "num_1", "num_2",
        "cat_0", "cat_1",
        "hier_0", "hier_1",
    }
    assert all(len(v) == 500 for v in cols.values())
    assert len(g["target"]) == 500
    assert g["labels"] is None
    assert set(g["truth"]["variance"]) == {
        "numeric", "categorical", "hierarchy", "interaction", "noise", "total",
    }


def test_generate_rejects_bad_params():
    with pytest.raises(ValueError):
        generate(n_rows=0)
    with pytest.raises(ValueError):
        generate(cardinalities=[1])
    with pytest.raises(ValueError):
        generate(missingness=1.0)
    with pytest.raises(ValueError):
        generate(task="clustering")


def test_classification_labels_cover_all_classes():
    g = generate(n_rows=600, task="classification", n_classes=3, seed=2)
    assert g["labels"] is not None
    assert set(g["labels"]) == {0, 1, 2}


def test_missingness_scatters_features_only():
    g = generate(n_rows=1000, missingness=0.3, seed=3)
    assert any(v is None for v in g["columns"]["num_0"])
    assert all(t is not None for t in g["target"])


def test_variance_shares_follow_knobs():
    g = generate(
        n_rows=20000,
        n_numeric=0,
        cardinalities=[8],
        hierarchy_levels=1,
        branching=[2],
        group_effect_strength=1.0,
        hierarchy_effect_strength=0.0,
        interaction_strength=0.0,
        noise=0.05,
        seed=5,
    )
    v = g["truth"]["variance"]
    assert v["categorical"] > 0.8
    assert v["noise"] < 0.05


def test_index_mapper_roundtrip():
    m = IndexMapper()
    vals = ["a", "b", None, "a", "c"]
    m.fit("col", vals)
    codes = m.transform("col", vals)
    assert codes == [0, 1, None, 0, 2]
    assert m.inverse_transform("col", codes) == vals
    assert m.transform("col", ["z"]) == [None]  # unseen level -> missing
    assert m.get_all_names("col") == ["a", "b", "c"]
    with pytest.raises(ValueError):
        m.transform("nope", vals)


def test_grouped_transform():
    keys = ["g1", "g1", "g2", "g2", "g2"]
    vals = [1.0, 3.0, 10.0, 20.0, 30.0]
    assert grouped_transform(keys, vals, kind="mean") == [2.0, 2.0, 20.0, 20.0, 20.0]
    assert grouped_transform(keys, vals, kind="count") == [2.0, 2.0, 3.0, 3.0, 3.0]
    with pytest.raises(ValueError):
        grouped_transform(keys, vals, kind="nope")


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, None])
    assert s["count"] == 3
    assert s["mean"] == 2.0
    assert s["median"] == 2.0
    assert s["min"] == 1.0
    assert s["max"] == 3.0
