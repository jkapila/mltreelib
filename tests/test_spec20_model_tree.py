import json
import pickle

import pytest
from mltreelib import GrowthControl, LeafControl, ModelTree


def test_controls_accept_typed_dict_and_flat_options():
    x = [[i / 10] for i in range(30)]
    y = [1 + 3 * row[0] for row in x]
    models = [
        ModelTree(
            growth_control=GrowthControl(max_depth=0),
            leaf_control=LeafControl(solver="lbfgs", l2=0.2),
        ),
        ModelTree(
            growth_control={"max_depth": 0}, leaf_control={"solver": "lbfgs", "l2": 0.2}
        ),
        ModelTree(max_depth=0, leaf_solver="lbfgs", l2=0.2),
    ]
    predictions = []
    for model in models:
        model.fit(x, y)
        assert model.get_params()["leaf_solver"] == "lbfgs"
        predictions.append(model.predict(x))
    assert predictions[0] == predictions[1] == predictions[2]
    assert predictions[0] == pytest.approx(y, abs=0.04)


def test_control_validation_points_to_the_parameter_and_solver():
    with pytest.raises(ValueError, match="Did you mean 'max_depth'"):
        GrowthControl(max_dpeth=2)
    with pytest.raises(ValueError, match="supplied more than once"):
        ModelTree(2, growth_control={"max_depth": 3})
    with pytest.raises(ValueError, match="supplied more than once"):
        ModelTree(leaf_solver="irls", leaf_control={"solver": "lbfgs"})
    with pytest.raises(ValueError, match="requires zero L1"):
        LeafControl(solver="lbfgs", l1=0.5)
    with pytest.raises(ValueError, match="does not support"):
        ModelTree(leaf_control={"family": "aft", "solver": "lbfgs"})


def test_class_count_is_a_growth_control():
    model = ModelTree(
        growth_control=GrowthControl(max_depth=0, n_classes=3),
        leaf_control={"family": "softmax"},
    )
    model.fit([[i / 10] for i in range(30)], [i % 3 for i in range(30)])
    assert model.get_params()["n_classes"] == 3
    probabilities = model.predict([[0.5]], kind="proba")
    assert sum(probabilities[0]) == pytest.approx(1.0)
    assert len(probabilities[0]) == 3
    restored = ModelTree.from_json(model.to_json())
    assert restored.predict([[0.5]], kind="proba") == probabilities


def test_model_round_trips_as_json_bytes_file_and_pickle(tmp_path):
    x = [("a" if i % 2 else "b", i / 10) for i in range(40)]
    y = [row[1] + 10 * (row[0] == "b") for row in x]
    model = ModelTree(max_depth=2, min_rows=4, prune="none", leaf="mean")
    model.fit(x, y, dtypes=["categorical", "numeric"])
    sample = [["a", 2.0], ["b", 1.0], ["new level", 0.5], [None, 1.0]]
    expected = model.predict(sample)
    rules = str(model.extract_rules())
    path = tmp_path / "model.json"
    model.save(path)
    assert isinstance(json.loads(path.read_text()), dict)
    assert model.to_bytes().decode() == model.to_json()

    for loaded in (
        ModelTree.load(path),
        ModelTree.from_json(model.to_json()),
        ModelTree.from_bytes(model.to_bytes()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert loaded.predict(sample) == expected
        assert loaded.predict(
            sample, kind="interval", alpha=0.1
        ) == model.predict_interval(sample)
        assert loaded.predict(
            sample, kind="adjusted", neighbors=3
        ) == model.predict_adjusted(sample, 3)
        assert str(loaded.extract_rules()) == rules


def test_pruned_model_round_trip_preserves_live_tree_state():
    x = [[0.0], [0.1], [0.9], [1.0]]
    y = [0.0, 0.0, 10.0, 10.0]
    grown = ModelTree(max_depth=1, min_rows=2, prune="none", leaf="mean")
    pruned = ModelTree(
        max_depth=1,
        min_rows=2,
        prune="cp",
        cp_alpha=1e9,
        leaf="mean",
    )
    grown.fit(x, y)
    pruned.fit(x, y)
    assert grown.n_leaves() > pruned.n_leaves()

    restored = ModelTree.from_json(pruned.to_json())
    assert restored.n_leaves() == pruned.n_leaves()
    assert restored.predict(x) == pruned.predict(x)
    assert str(restored.extract_rules()) == str(pruned.extract_rules())


def test_persistence_rejects_unfitted_and_invalid_models(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("existing")
    with pytest.raises(ValueError, match="not fitted"):
        ModelTree().save(path)
    assert path.read_text() == "existing"
    with pytest.raises(ValueError):
        ModelTree.from_json("{}")
    with pytest.raises(ValueError):
        ModelTree.from_bytes(b"\xff")
    with pytest.raises(FileNotFoundError):
        ModelTree.load(tmp_path / "absent.json")
