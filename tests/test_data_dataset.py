"""Behaviour tests for the Dataset class: roles, views, delegation, splits and model hand-off."""

import copy
import math
import os
import subprocess
import sys

import pytest
from mltreelib import ModelTree
from mltreelib.data import (
    Dataset,
    assoc,
    class_balance,
    constant_columns,
    describe,
    duplicate_rows,
    information_value_table,
    missing_profile,
    near_zero_variance,
    schema_capture,
    select_rows,
    split,
    survival_xy,
    train_test_split,
    uplift_xy,
    xy,
)


def _same(a, b):
    """NaN-safe equality for nested plain values."""
    return repr(a) == repr(b)


def _real(name):
    """A load_dataset result from the local parquet files, or skip the test."""
    pytest.importorskip("pyarrow")
    from mltreelib.datasets import load_dataset

    try:
        return load_dataset(name, auto_download=False)
    except (ValueError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"dataset {name!r} not available locally: {exc}")


class _FrameStub:
    """Minimal data-frame stand-in: ``.columns`` and column ``__getitem__``."""

    def __init__(self, data):
        self._data = data
        self.columns = list(data)

    def __getitem__(self, name):
        return self._data[name]


def _table(n=40):
    """Small mixed table with missing values and a unique row id."""
    return {
        "id": list(range(n)),
        "a": [None if i % 7 == 0 else float(i) for i in range(n)],
        "b": [float("nan") if i % 5 == 0 else float(i % 3) for i in range(n)],
        "c": ["u" if i % 4 else None for i in range(n)],
        "g": [f"grp{i % 6}" for i in range(n)],
        "y": [1 if i % 3 == 0 else 0 for i in range(n)],
    }


# -- construction -----------------------------------------------------------


def test_construct_from_dict_matches_xy_and_roles_excluded():
    cols = {"a": [1.0, None, 3.0], "c": ["p", "q", None], "y": [0.0, 1.0, None]}
    ds = Dataset(cols, target="y")
    assert ds.names == ["a", "c"]
    assert ds.dtypes == ["numeric", "categorical"]
    assert ds.n_rows == 3
    assert _same(ds.x, xy(cols, "y")["x"])
    assert _same(ds.y, [0.0, 1.0, None])


def test_construct_from_load_result_uses_declared_dtypes_and_name():
    loaded = {
        "name": "toy",
        "columns": {"code": ["1", "2", "1"], "v": [1.0, None, 2.0], "y": [0, 1, 0]},
        "dtypes": {"code": "categorical", "v": "numeric", "y": "numeric"},
    }
    ds = Dataset(loaded, target="y")
    assert ds.dtypes == [
        "categorical",
        "numeric",
    ]  # declared, not inferred from '1'/'2'
    assert "Dataset toy" in str(ds)


def test_construct_from_data_frame_stand_in():
    frame = _FrameStub({"a": [1, 2, None], "b": ["x", "y", "x"], "y": [0, 1, 0]})
    ds = Dataset(frame, target="y")
    assert ds.names == ["a", "b"]
    assert ds.x == [[1, "x"], [2, "y"], [None, "x"]]


def test_load_passes_roles_and_uses_load_dataset(monkeypatch):
    from mltreelib import datasets

    seen = {}

    def fake_load(name, auto_download=True):
        seen["args"] = (name, auto_download)
        return {"name": name, "columns": {"a": [1, 2], "y": [0, 1]}, "dtypes": {}}

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    ds = Dataset.load("toy", "y", exclude=["a"], auto_download=False)
    assert seen["args"] == ("toy", False)
    assert ds.names == [] and ds.y == [0, 1]


# -- roles and validation ---------------------------------------------------


def test_unknown_role_column_raises_keyerror_naming_it():
    with pytest.raises(KeyError, match="'nope'"):
        Dataset({"a": [1, 2]}, target="nope")
    with pytest.raises(KeyError, match="'missing_t'"):
        Dataset({"a": [1, 2], "e": [0, 1]}, time="missing_t", event="e")


def test_feature_that_is_a_role_raises_value_error():
    with pytest.raises(ValueError, match="'y'"):
        Dataset({"a": [1, 2], "y": [0, 1]}, target="y", features=["a", "y"])


def test_target_and_outcome_must_agree_and_roles_cannot_share_a_column():
    cols = {"a": [1, 2], "y": [0, 1], "w": [0, 1]}
    with pytest.raises(ValueError, match="name different columns"):
        Dataset(cols, target="y", outcome="a")
    with pytest.raises(ValueError, match="both"):
        Dataset(cols, treatment="w", groups="w")
    same = Dataset(cols, target="y", outcome="y")
    assert same.names == ["a", "w"]


def test_roles_are_excluded_from_features_and_exclude_removes_more():
    cols = _table()
    ds = Dataset(cols, target="y", groups="g", weights="id", exclude=["b"])
    assert ds.names == ["a", "c"]
    explicit = Dataset(cols, target="y", features=["c", "a"])
    assert explicit.names == ["c", "a"]
    with pytest.raises(KeyError, match="'zz'"):
        Dataset(cols, target="y", exclude=["zz"])


def test_roles_views_return_the_column_values():
    cols = {"t": [5.0, None, 9.0], "e": [1, 0, 1], "tr": [0, 1, 1], "x": [1, 2, 3]}
    ds = Dataset(cols, time="t", event="e", treatment="tr")
    assert ds.time == [5.0, None, 9.0]
    assert ds.event == [1, 0, 1]
    assert ds.treatment == [0, 1, 1]
    assert ds.y is None and ds.groups is None and ds.weights is None
    assert ds.names == ["x"]


# -- properties equal the helper outputs -----------------------------------


def test_survival_view_equals_survival_xy():
    cols = {
        "t": [5.0, None, 9.0, 2.0],
        "died": [1, 2, 0, None],
        "age": [60, None, 50, 70],
        "sex": ["F", "M", None, "F"],
    }
    ds = Dataset(cols, time="t", event="died")
    ref = survival_xy(cols, "t", "died")
    assert ds.names == ref["names"]
    assert ds.dtypes == ref["dtypes"]
    assert _same(ds.x, ref["x"])
    assert _same(ds.time, ref["time"]) and _same(ds.event, ref["event"])


def test_uplift_view_equals_uplift_xy_with_outcome_as_target():
    cols = {
        "treat": [1, 0, None, 1],
        "conv": [0.0, 1.0, 1.0, None],
        "age": [30, None, 50, 40],
    }
    ds = Dataset(cols, treatment="treat", outcome="conv")
    ref = uplift_xy(cols, "treat", "conv")
    assert ds.names == ref["names"]
    assert _same(ds.x, ref["x"])
    assert _same(ds.treatment, ref["treatment"])
    assert _same(ds.y, ref["outcome"])


def test_xy_view_equals_xy_with_declared_dtypes_and_views_are_fresh():
    loaded = {
        "columns": {"a": [1, None], "y": [0, 1]},
        "dtypes": {"a": "numeric", "y": "numeric"},
    }
    ds = Dataset(loaded, target="y")
    ref = xy(loaded, "y")
    assert _same(ds.x, ref["x"]) and ds.dtypes == ref["dtypes"]
    ds.x[0].append("mutated")
    ds.columns["a"].append("mutated")
    assert _same(ds.x, ref["x"]) and ds.columns["a"] == [1, None]


# -- delegation of the printable methods ------------------------------------


def test_printable_methods_delegate_to_helper_functions():
    cols = _table()
    ds = Dataset(cols, target="y", groups="g")
    feats = {k: cols[k] for k in ds.names}
    assert _same(ds.describe(), describe(cols))
    assert _same(ds.missing(), missing_profile(cols))
    assert _same(ds.schema(), schema_capture(cols))
    checks = ds.checks()
    assert checks["constant_columns"] == constant_columns(feats)
    assert _same(checks["near_zero_variance"], near_zero_variance(feats))
    dups = duplicate_rows(feats)
    assert checks["duplicate_rows"]["groups"] == dups
    assert checks["duplicate_rows"]["n_extra_rows"] == sum(len(g) - 1 for g in dups)
    assert _same(checks["class_balance"], class_balance(cols["y"]))
    assoc_out = ds.associations()
    assert _same(assoc_out["correlation_matrix"], assoc.correlation_matrix(feats))
    assert _same(assoc_out["cramers_v_matrix"], assoc.cramers_v_matrix(feats))
    assert _same(assoc_out["vif"], assoc.vif(feats))
    assert _same(assoc_out["leakage_hints"], assoc.leakage_hints(feats, cols["y"]))
    assert _same(ds.iv(), information_value_table(feats, cols["y"]))
    assert _same(ds.missing_vs_target(), _mvt(feats, cols["y"]))


def _mvt(feats, y):
    from mltreelib.data import missing_vs_target

    return missing_vs_target(feats, y)


def test_class_balance_and_no_target_outputs():
    cols = {"a": [1.0, 2.0, 3.0, 4.0], "s": ["x", "x", "y", "x"], "y": [0, 1, 0, 1]}
    ds = Dataset(cols, target="y")
    assert ds.checks()["class_balance"] is not None
    assert ds.associations()["leakage_hints"] is not None
    bare = Dataset({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert bare.checks()["class_balance"] is None
    assert bare.associations()["leakage_hints"] is None
    with pytest.raises(ValueError, match="target"):
        bare.iv()
    with pytest.raises(ValueError, match="target"):
        bare.missing_vs_target()
    with pytest.raises(ValueError, match="target"):
        bare.fit_args()


def test_str_and_summary_snapshot_on_tiny_dataset():
    ds = Dataset(
        {"a": [1.0, 2.0, None, 4.0], "c": ["u", "v", "u", "u"], "y": [0, 1, 1, 0]},
        target="y",
        name="tiny",
    )
    assert str(ds) == (
        "Dataset tiny: 4 rows, 2 features\n"
        "roles: target=y\n"
        "features:\n"
        "  a  numeric      missing 1\n"
        "  c  categorical  missing 0"
    )
    assert ds.summary() == (
        "name  kind               n  missing%  mean   std     min  median  max  distinct  mode\n"
        "a     ordinal_candidate  4  25.0      2.333  1.528   1    2       4\n"
        "c     categorical        4  0.0                                        2         u\n"
        "y     ordinal_candidate  4  0.0       0.5    0.5774  0    0.5     1"
    )


# -- splitting ---------------------------------------------------------------


def _assert_rows_kept(parent, child):
    """Every child row equals the parent row it came from (id column), missing kept."""
    ids = child.columns["id"]
    for col in ("a", "b", "c", "g", "y"):
        expected = [parent.columns[col][i] for i in ids]
        assert _same(child.columns[col], expected), col


def test_split_plain_keeps_counts_values_and_is_a_partition():
    ds = Dataset(_table(), target="y")
    train, test = ds.split(test_size=0.25, seed=0)
    assert (train.n_rows, test.n_rows) == (30, 10)
    assert sorted(train.columns["id"] + test.columns["id"]) == list(range(40))
    _assert_rows_kept(ds, train)
    _assert_rows_kept(ds, test)


def test_split_with_validation_has_three_partitions():
    ds = Dataset(_table(), target="y")
    train, val, test = ds.split(test_size=0.2, val_size=0.2, seed=1)
    ids = sorted(train.columns["id"] + val.columns["id"] + test.columns["id"])
    assert ids == list(range(40))
    assert (train.n_rows, val.n_rows, test.n_rows) == (24, 8, 8)
    _assert_rows_kept(ds, val)


def test_split_stratify_on_column_and_list_and_groups_keep_whole_groups():
    ds = Dataset(_table(), target="y")
    train, test = ds.split(test_size=0.25, seed=0, stratify="c")
    assert train.n_rows + test.n_rows == 40
    train, test = ds.split(test_size=0.25, seed=0, stratify=["c", "y"])
    assert sorted(train.columns["id"] + test.columns["id"]) == list(range(40))
    train, test = ds.split(test_size=0.5, seed=0, groups="g")
    test_groups = set(test.columns["g"])
    assert test_groups.isdisjoint(set(train.columns["g"]))
    _assert_rows_kept(ds, test)


def test_split_time_true_and_column_are_ordered_not_shuffled():
    ds = Dataset(_table(), target="y")
    train, test = ds.split(test_size=10, time=True)
    assert test.columns["id"] == list(range(30, 40))
    assert train.columns["id"] == list(range(30))
    keyed = Dataset(
        {"id": [0, 1, 2, 3], "t": [3, None, 1, 2], "y": [0, 1, 0, 1]}, target="y"
    )
    train, test = keyed.split(test_size=2, time="t")
    assert sorted(test.columns["id"]) == [0, 1]  # latest keys, missing key last
    assert sorted(train.columns["id"]) == [2, 3]


def test_split_is_deterministic_across_processes_with_hash_seeds():
    code = (
        "from mltreelib.data import Dataset\n"
        "cols = {'id': list(range(60)), 'g': ['p%d' % (i % 7) for i in range(60)],"
        " 'y': [i % 2 for i in range(60)], 'a': [float(i) for i in range(60)]}\n"
        "ds = Dataset(cols, target='y')\n"
        "tr, va, te = ds.split(test_size=0.2, val_size=0.2, seed=5, stratify='g')\n"
        "kf = ds.kfold(k=4, seed=2, groups='g')\n"
        "print(tr.columns['id'], va.columns['id'], te.columns['id'],"
        " [f[0].columns['id'] for f in kf])\n"
    )
    outputs = set()
    for hash_seed in ("1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        run = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(run.stdout)
    assert len(outputs) == 1


def test_stratify_true_on_binary_target_matches_class_share():
    cols = {
        "id": list(range(100)),
        "a": [float(i) for i in range(100)],
        "y": [1 if i < 40 else 0 for i in range(100)],
    }
    _, test = Dataset(cols, target="y").split(test_size=0.25, seed=0, stratify=True)
    ones = test.columns["y"].count(1)
    assert abs(ones - 0.4 * 25) <= 1


def test_stratify_true_uses_treatment_for_uplift():
    treat = [1 if i < 80 else 0 for i in range(100)]
    cols = {
        "id": list(range(100)),
        "a": [float(i) for i in range(100)],
        "tr": treat,
        "conv": [i % 2 for i in range(100)],
    }
    ds = Dataset(cols, treatment="tr", outcome="conv")
    _, test = ds.split(test_size=20, seed=0, stratify=True)
    assert abs(test.columns["tr"].count(1) - 16) <= 1


def test_stratify_true_uses_event_for_survival():
    cols = {
        "id": list(range(100)),
        "a": [float(i) for i in range(100)],
        "t": [float(i) for i in range(100)],
        "e": [1 if i < 30 else 0 for i in range(100)],
    }
    ds = Dataset(cols, time="t", event="e")
    _, test = ds.split(test_size=20, seed=0, stratify=True)
    assert abs(test.columns["e"].count(1) - 6) <= 1


def test_stratify_true_errors_are_clear():
    continuous = {
        "a": [float(i) for i in range(50)],
        "y": [i * 0.37 for i in range(50)],
    }
    with pytest.raises(ValueError, match="classes of one"):
        Dataset(continuous, target="y").split(stratify=True)
    with pytest.raises(ValueError, match="needs a treatment, event or target"):
        Dataset({"a": [1, 2, 3, 4]}).split(stratify=True)


def test_kfold_covers_every_row_once_and_pairs_are_disjoint():
    ds = Dataset(_table(), target="y")
    folds = ds.kfold(k=4, seed=0, stratify="c")
    assert len(folds) == 4
    tests = [test.columns["id"] for _, test in folds]
    assert sorted(i for t in tests for i in t) == list(range(40))
    for train, test in folds:
        assert sorted(train.columns["id"] + test.columns["id"]) == list(range(40))
        _assert_rows_kept(ds, test)
    grouped = ds.kfold(k=3, seed=0, groups="g")
    for train, test in grouped:
        assert set(train.columns["g"]).isdisjoint(set(test.columns["g"]))


def test_take_keeps_order_repeats_roles_and_declared_kinds():
    cols = {"c": [None, None, "u", "v"], "a": [1.0, None, 3.0, 4.0], "y": [0, 1, 0, 1]}
    ds = Dataset(cols, target="y")
    child = ds.take([3, 0, 3])
    assert child.columns == {
        "c": ["v", None, "v"],
        "a": [4.0, 1.0, 4.0],
        "y": [1, 0, 1],
    }
    assert child.names == ["c", "a"]
    assert child.dtypes == ["categorical", "numeric"]  # frozen at construction
    assert child.y == [1, 0, 1]
    tr, te = ds.take([1, 2]), ds.take([0])
    assert tr.dtypes == ["categorical", "numeric"] and te.dtypes == [
        "categorical",
        "numeric",
    ]


def test_splits_never_change_values_of_inputs():
    cols = _table()
    before = copy.deepcopy(cols)
    ds = Dataset(cols, target="y", groups="g")
    ds.split(test_size=0.3, seed=0, stratify="c")
    ds.kfold(k=3, seed=0)
    ds.take([5, 1])
    ds.describe()
    ds.associations()
    assert _same(cols, before)


# -- model hand-off ----------------------------------------------------------


def test_fit_args_and_predict_args_shapes():
    cols = {"a": [1.0, None, 3.0], "t": [5.0, 6.0, None], "e": [1, 0, 1]}
    ds = Dataset(cols, time="t", event="e", weights="a")
    args = ds.fit_args()
    assert set(args) == {"x", "y", "col_names", "dtypes", "event", "sample_weight"}
    assert args["y"] == [5.0, 6.0, None] and args["event"] == [1, 0, 1]
    assert args["sample_weight"] == [1.0, None, 3.0]
    assert ds.predict_args() == {"x": [[]] * 3}
    up = Dataset(
        {"tr": [1, 0], "o": [0.5, 1.0], "a": [2.0, 3.0]}, treatment="tr", outcome="o"
    )
    assert up.fit_args()["treatment"] == [1, 0] and up.fit_args()["y"] == [0.5, 1.0]


def test_fit_args_drives_modeltree_regression_identical_to_hand_built_path():
    data = _real("concrete_compressive_strength")
    target = "ConcreteCompressiveStrength"
    ds = Dataset(data, target)
    train, test = ds.split(test_size=0.2, seed=0)
    model = ModelTree(leaf="mean", max_depth=3, min_rows=20)
    model.fit(**train.fit_args())
    got = model.predict(**test.predict_args())

    d = xy(data, target)
    tr_idx, te_idx = train_test_split(data["n_rows"], test_size=0.2, seed=0)
    tr, te = select_rows(d, tr_idx), select_rows(d, te_idx)
    hand = ModelTree(leaf="mean", max_depth=3, min_rows=20)
    hand.fit(tr["x"], tr["y"], col_names=d["names"], dtypes=d["dtypes"])
    assert len(got) == len(te["y"]) and all(math.isfinite(p) for p in got)
    assert got == hand.predict(te["x"])


def test_fit_args_drives_modeltree_binary_with_missing_features():
    raw = _real("GiveMeSomeCredit")
    target = "FinancialDistressNextTwoYears"
    cols = {k: list(v) for k, v in raw["columns"].items()}
    cols[target] = [1 if v == "Yes" else 0 for v in cols[target]]  # recode by hand
    dtypes = {**raw["dtypes"], target: "numeric"}
    data = {"name": "GiveMeSomeCredit", "columns": cols, "dtypes": dtypes}
    rows = 6000
    assert (
        sum(v is None for v in cols["MonthlyIncome"][:rows]) > 0
    )  # missing features kept

    sub = Dataset(data, target).take(range(rows))
    train, test = sub.split(test_size=0.2, seed=0)
    model = ModelTree(leaf_family="binomial", max_depth=3, min_rows=100)
    model.fit(**train.fit_args())
    got = model.predict(**test.predict_args())

    sub_cols = split.take(cols, range(rows))
    d = xy({"columns": sub_cols, "dtypes": dtypes}, target)
    tr_idx, te_idx = train_test_split(rows, test_size=0.2, seed=0)
    tr, te = select_rows(d, tr_idx), select_rows(d, te_idx)
    hand = ModelTree(leaf_family="binomial", max_depth=3, min_rows=100)
    hand.fit(tr["x"], tr["y"], col_names=d["names"], dtypes=d["dtypes"])
    assert got == hand.predict(te["x"])


def test_inputs_are_not_modified_by_construction_or_methods():
    cols = _table()
    before = copy.deepcopy(cols)
    ds = Dataset(cols, target="y", groups="g", exclude=["b"])
    ds.describe()
    ds.missing()
    ds.checks()
    ds.associations()
    ds.summary()
    ds.fit_args()
    assert _same(cols, before)
    assert _same(ds.x, Dataset(before, target="y", groups="g", exclude=["b"]).x)


def test_dataset_is_immutable_and_methods_return_new_objects():
    ds = Dataset(_table(), target="y")
    train, _ = ds.split(seed=0)
    assert train is not ds and ds.n_rows == 40
    with pytest.raises(AttributeError):
        ds.n_rows = 3  # read-only property
