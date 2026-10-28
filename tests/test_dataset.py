"""Behavior tests for the dataset component through the Python bindings."""

from datetime import date, timedelta

import pytest

from mltreelib import (
    DatasetBuilder,
    IndexMapper,
    VariableScaler,
    calendar_features,
    decomposed_transform,
    generate,
    grouped_scale,
    grouped_transform,
    lag_by_days,
    summarize,
    trended_transform,
    windowed_transform,
)


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


def test_dataset_builder_ingest():
    b = DatasetBuilder(n_category=10, na_policy="impute", digits=2, add_intercept=True)
    b.add_numeric("price", [1.234, None, 2.356])
    b.add_numeric("code", [1.0, 2.0, 1.0])
    b.add_categorical("grp", ["x", "x", None])
    ds = b.build()
    assert ds["n_rows"] == 3
    assert set(ds["columns"]) == {"price", "code", "grp", "intercept"}
    assert ds["dtypes"]["code"] == "categorical"  # n_category inference
    assert ds["dtypes"]["price"] == "numeric"
    assert ds["columns"]["price"][0] == 1.23  # digits rounding
    assert abs(ds["columns"]["price"][1] - 1.795) < 1e-9  # imputed mean of rounded
    assert ds["columns"]["intercept"] == [1.0, 1.0, 1.0]


def test_dataset_builder_drop_rows():
    b = DatasetBuilder(na_policy="drop")
    b.add_numeric("a", [1.0, None, 3.0])
    b.add_categorical("b", ["x", "y", "z"])
    ds = b.build()
    assert ds["n_rows"] == 2
    assert ds["columns"]["a"] == [1.0, 3.0]
    assert ds["columns"]["b"] == ["x", "z"]

    with pytest.raises(ValueError):
        bad = DatasetBuilder(na_policy="drop")
        bad.add_numeric("a", [None, None])
        bad.build()


def test_grouped_scale_and_inverse():
    from mltreelib import GroupedScaler

    keys = ["g", "g", "g"]
    vals = [1.0, 3.0, 5.0]
    out = grouped_scale(keys, vals, strategy="zscore")
    assert out[1] == 0.0
    assert abs(out[0] + out[2]) < 1e-9

    scaler = GroupedScaler()
    scaler.fit(keys, vals, "zscore")
    out = scaler.transform(keys, vals)
    back = scaler.inverse_transform(keys, out)
    for b, v in zip(back, vals):
        assert abs(b - v) < 1e-9

    out = grouped_scale(keys, vals, strategy="mean")
    assert abs(out[0] - 1.0 / 3.0) < 1e-9
    assert grouped_scale(["g"], [1.0], strategy="zscore") == [None]  # zero scale

    with pytest.raises(ValueError):
        grouped_scale(keys, vals, strategy="nope")


def test_calendar_features():
    out = calendar_features(["2026-10-28", None, "2026-10-28T14:30:00"])
    assert out["year"] == [2026.0, None, 2026.0]
    assert out["month"] == [10.0, None, 10.0]
    assert out["quarter"] == [4.0, None, 4.0]
    assert out["weekday"] == [2.0, None, 2.0]  # Wednesday
    assert out["season"] == [2.0, None, 2.0]  # fall
    assert out["weekend"] == [0.0, None, 0.0]
    assert out["day_of_year"] == [301.0, None, 301.0]

    southern = calendar_features(["2026-01-15"], hemisphere="south")
    assert southern["season"] == [1.0]  # southern summer

    with pytest.raises(ValueError):
        calendar_features(["nope"])


def test_lag_by_days():
    keys = ["a", "a", "a"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-05"]
    vals = [10.0, 20.0, 30.0]
    assert lag_by_days(keys, dates, vals, 1) == [None, 10.0, None]
    assert lag_by_days(keys, dates, vals, 4) == [None, None, 10.0]
    assert lag_by_days(keys, dates, vals, 0) == [10.0, 20.0, 30.0]


def test_trended_transform():
    keys = ["a", "a", "a", "b"]
    dates = ["2026-01-05", "2026-01-20", "2026-02-03", "2026-01-05"]
    vals = [1.0, 3.0, 10.0, 100.0]
    out = trended_transform(keys, dates, vals, granularity="year_month")
    assert out == [2.0, 2.0, 10.0, 100.0]  # per (key, month) means

    out = trended_transform(keys, dates, vals, granularity="year", kind="sum")
    assert out == [14.0, 14.0, 14.0, 100.0]

    with pytest.raises(ValueError):
        trended_transform(keys, dates, vals, granularity="nope")
    with pytest.raises(ValueError):
        trended_transform(keys, ["bad-date"] + dates[1:], vals)


def test_datasets_loader_feeds_builder():
    from mltreelib import list_datasets, load_dataset

    names = list_datasets()
    if not names:
        pytest.skip("datasets/ suite not present (see datasets/MANIFEST.md)")

    assert "blood-transfusion-service-center" in names
    ds = load_dataset("blood-transfusion-service-center")
    assert ds["n_rows"] == 748
    assert ds["n_cols"] == 5
    assert "na_counts" in ds
    assert "DonatedBloodInMarch2007" in ds["columns"]
    assert ds["dtypes"]["DonatedBloodInMarch2007"] == "categorical"  # target Yes/No
    assert ds["dtypes"]["MonthsSinceLastDonation"] == "numeric"

    # loader output feeds straight into the data layer
    b = DatasetBuilder(n_category=10)
    for col, values in ds["columns"].items():
        if ds["dtypes"][col] == "categorical":
            b.add_categorical(col, values)
        else:
            b.add_numeric(col, values)
    built = b.build()
    assert built["n_rows"] == 748

    with pytest.raises(ValueError):
        load_dataset("does-not-exist")


def test_summarize_shows_missing_by_default():
    s = summarize([1.0, None, float("nan"), 3.0])
    assert s["count"] == 2  # observed only
    assert s["n_missing"] == 2  # NA + NaN shown, not hidden
    assert s["mean"] == 2.0


def test_builder_reports_na_counts_and_nan_is_missing():
    b = DatasetBuilder()
    b.add_numeric("a", [1.0, float("nan"), None])
    b.add_categorical("b", ["x", None, "y"])
    ds = b.build()
    assert ds["na_counts"] == {"a": 2, "b": 1}  # NaN counted as missing


def test_variable_scaler():
    vs = VariableScaler()
    vs.fit({"a": [1.0, 3.0, 5.0], "b": [10.0, 10.0, 10.0]}, "min-max")
    assert vs.transform("a", [1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    assert vs.inverse_transform("a", [0.0, 0.5, 1.0]) == [1.0, 3.0, 5.0]
    assert vs.transform("b", [10.0]) == [None]  # zero scale -> missing
    assert vs.transform("a", [float("nan")]) == [None]  # missing in -> missing out
    with pytest.raises(ValueError):
        vs.transform("zz", [1.0])


def test_ts_transforms_drop_na_rows_and_warn():
    keys = ["a", "a", None]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    vals = [1.0, None, 3.0]

    with pytest.warns(UserWarning):
        out = lag_by_days(keys, dates, vals, 1)
    assert out == [None, None, None]

    with pytest.warns(UserWarning):
        out = windowed_transform(keys, dates, vals, 3, kind="sum")
    assert out == [1.0, None, None]

    with pytest.warns(UserWarning):
        out = trended_transform(keys, dates, vals)
    assert out == [1.0, None, None]


def test_categorical_encoders():
    from mltreelib import (
        CatBoostEncoder,
        HashingEncoder,
        OneHotEncoder,
        TargetEncoder,
    )

    ohe = OneHotEncoder()
    ohe.fit(["a", "b", "a"])
    assert ohe.levels() == ["a", "b"]
    assert ohe.transform(["a", "zz", None]) == {"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}

    he = HashingEncoder(4)
    cols = he.transform(["x", "y", "x"])
    assert len(cols) == 4
    assert sum(sum(v) for v in cols.values()) == 3.0

    te = TargetEncoder(2.0)
    te.fit(["a", "a", "b", "b"], [0.0, 0.0, 10.0, 10.0])
    assert te.transform(["a"]) == [2.5]
    assert te.transform(["zz", None]) == [5.0, None]
    assert te.prior() == 5.0

    cb = CatBoostEncoder(1.0, 4, 42)
    enc = cb.fit_transform(["a", "a", "a", "a"], [1.0, 3.0, 5.0, 7.0])
    assert len(set(enc)) > 1  # row-dependent values = leakage guard
    assert cb.transform(["a", "zz", None]) == [4.0, 4.0, None]


def test_class_space_encoders():
    from mltreelib import EmbeddingEncoder, TfidfEncoder, WoEEncoder

    tf = TfidfEncoder(normalize=False)
    tf.fit(["a", "a", "b", "b"], [0.0, 1.0, 0.0, 1.0], 2)
    out = tf.transform(["a", None])
    assert out[1] is None
    assert len(out[0]) == 2  # one weight per class

    woe = WoEEncoder(1.0)
    woe.fit(["a", "a", "b"], [1.0, 1.0, 0.0], 2)
    v = woe.transform(["a"])[0]
    assert v[1] > 0 > v[0]  # over-represented in class 1

    emb = EmbeddingEncoder(4, 0.5, 50, 7, "classification")
    emb.fit(["a", "a", "b", "b"], [0.0, 0.0, 1.0, 1.0], 2)
    assert emb.predict(["a", "b"]) == [0.0, 1.0]
    assert emb.transform(["zz"])[0] == [0.0, 0.0, 0.0, 0.0]

    reg = EmbeddingEncoder(3, 0.1, 300, 11, "regression")
    reg.fit(["a", "b", "a", "b"], [5.0, 0.0, 5.0, 0.0])
    assert all(p >= 0 for p in reg.predict(["a", "b"]))  # relu on all-positive targets


def test_quick_win_encoders():
    from mltreelib import (
        BaseNEncoder,
        CountEncoder,
        JamesSteinEncoder,
        LeaveOneOutEncoder,
        MEstimatorEncoder,
        QuantileEncoder,
        SumEncoder,
    )

    c = CountEncoder(False)
    c.fit(["a", "a", "b"])
    assert c.transform(["a", "zz", None]) == [2.0, 0.0, None]

    loo = LeaveOneOutEncoder()
    out = loo.fit_transform(["a", "a", "b"], [0.0, 10.0, 4.0])
    assert out == [10.0, 0.0, 14.0 / 3.0]  # own row excluded; single-obs -> global

    m = MEstimatorEncoder(2.0)
    m.fit(["a", "a"], [0.0, 10.0])
    assert m.transform(["a"]) == [5.0]  # (10 + 2*5) / (2+2)

    b = BaseNEncoder(2)
    b.fit(["a", "b", "c", "d"])
    assert b.transform(["d", "zz"]) == {"d0": [1.0, 0.0], "d1": [1.0, 0.0]}

    s = SumEncoder()
    s.fit(["a", "b", "c"])
    assert s.transform(["a", "c", None]) == [[1.0, 0.0], [-1.0, -1.0], None]

    q = QuantileEncoder(0.5, 0.0)
    q.fit(["a", "a", "b"], [0.0, 10.0, 100.0])
    assert q.transform(["a"]) == [5.0]  # median of [0, 10]

    js = JamesSteinEncoder()
    js.fit(["a", "b"], [0.0, 10.0])
    assert js.transform(["a", "b"]) == [5.0, 5.0]  # <3 levels -> grand mean


def test_encoding_demo_runs():
    import subprocess
    import sys
    from pathlib import Path

    from mltreelib import list_datasets

    if not list_datasets():
        pytest.skip("datasets/ suite not present (see datasets/MANIFEST.md)")
    script = Path(__file__).resolve().parents[1] / "examples" / "encoding_demo.py"
    proc = subprocess.run(
        [sys.executable, str(script), "churn"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "UNSUPERVISED ENCODERS" in proc.stdout
    assert "REGRESSION FAMILY" in proc.stdout


def test_metrics_single_class_computes_all():
    from mltreelib import Metrics

    m = Metrics()

    r = m.regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["mae"] == 0.0 and r["r2"] == 1.0 and r["n"] == 3
    assert len(r["deciles"]) == 10  # DAR bathtub profile included

    r = m.regression([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert r["mean_error"] == -1.0  # actual - fitted
    assert r["bias"] == 1.5 and r["rmse"] == 1.0 and r["mad"] == 0.0

    c = m.classification(
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        probabilities=[0.1, 0.6, 0.7, 0.9],
    )
    assert c["accuracy"] == 0.75
    assert c["auc"] == 1.0
    assert len(c["confusion"]) == 2 and len(c["per_class"]) == 2
    assert c["brier"] > 0

    g = m.regression_by_group(["a", "a", "b"], [1.0, 3.0, 10.0], [2.0, 2.0, 12.0])
    assert g["a"]["mae"] == 1.0 and g["b"]["mae"] == 2.0

    r = m.regression([1.0, None, 3.0], [1.0, 2.0, 5.0])
    assert r["n"] == 2 and r["n_missing"] == 1  # missing shown, not hidden


def test_decomposed_transform():
    dates = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(14)]
    vals = [0.4 if i % 7 < 5 else -1.0 for i in range(14)]
    keys = ["a"] * 14
    out = decomposed_transform(
        keys, dates, vals, season_granularity="weekday", trend_window=7
    )
    # zero-mean weekly seasonality, no trend -> clean removal in the interior
    for i in range(3, 11):
        assert abs(out["residual"][i]) < 1e-9
        assert abs(out["deseasonalized"][i]) < 1e-9


def test_demo_script_runs():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples" / "data_layer_demo.py"
    if not script.exists():
        pytest.skip("demo script not present")
    proc = subprocess.run(
        [sys.executable, str(script), "blood-transfusion-service-center"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    # sections 2-3 (generator, time features) always run; the dataset section
    # needs the downloaded suite (parquet files are not tracked in git — CI
    # checkouts and fresh clones only have datasets/MANIFEST.md)
    assert "GROUND TRUTH" in proc.stdout
    from mltreelib import list_datasets

    if list_datasets():
        assert "REAL DATASET" in proc.stdout
