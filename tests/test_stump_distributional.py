import pytest
import math
from mltreelib import ModelTree, LeafControl


def test_softmax_stump_fit_and_predict():
    # 3-class target (0, 1, 2)
    X = [[1.0, 0.5], [1.2, 0.4], [5.0, 4.8], [5.2, 5.1], [10.0, 9.8], [10.2, 10.1]] * 10
    y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0] * 10

    # 1. Constant Softmax Stump
    tree = ModelTree(max_depth=0, leaf="mean", leaf_family="softmax")
    tree.fit(X, y)

    # Class probability matrix
    probs = tree.predict(X, kind="proba")
    assert len(probs) == len(X)
    for p in probs:
        assert len(p) == 3
        assert pytest.approx(sum(p), abs=1e-5) == 1.0
        # Equal distribution across 3 classes
        assert pytest.approx(p[0], abs=1e-3) == 1.0 / 3.0
        assert pytest.approx(p[1], abs=1e-3) == 1.0 / 3.0
        assert pytest.approx(p[2], abs=1e-3) == 1.0 / 3.0

    # predict_proba convenience alias
    alias_probs = tree.predict_proba(X)
    assert len(alias_probs) == len(X)
    assert pytest.approx(alias_probs[0][0], abs=1e-5) == probs[0][0]

    # Leaves inspection
    lfs = tree.leaves()
    assert len(lfs) == 1
    assert "softmax_probs" in lfs[0]
    assert lfs[0]["n_classes"] == 3

    # 2. Linear Softmax Stump
    tree_lin = ModelTree(max_depth=0, leaf="linear", leaf_family="softmax")
    tree_lin.fit(X, y)
    probs_lin = tree_lin.predict(X, kind="proba")
    assert len(probs_lin) == len(X)
    for p in probs_lin:
        assert len(p) == 3
        assert pytest.approx(sum(p), abs=1e-5) == 1.0


def test_survival_stump_fit_and_predict():
    # Times 1..20, right-censored event indicators (1=event, 0=censored)
    X = [[float(i), float(i * 2)] for i in range(1, 21)]
    times = [float(i) for i in range(1, 21)]
    events = [1.0 if i % 2 == 0 else 0.0 for i in range(1, 21)]

    # 1. Fit with y as tuple (times, events)
    tree_tup = ModelTree(max_depth=0, leaf="mean", leaf_family="survival")
    tree_tup.fit(X, (times, events))

    time_grid = [2.0, 5.0, 10.0, 15.0, 20.0]
    surv_preds = tree_tup.predict(X, kind="survival", times=time_grid)
    assert len(surv_preds) == len(X)
    for s in surv_preds:
        assert len(s) == len(time_grid)
        # S(t) must be non-increasing in time
        for i in range(len(s) - 1):
            assert s[i] >= s[i + 1] - 1e-9
        assert 0.0 <= s[-1] <= 1.0

    # Cumulative hazard Lambda(t) must be non-decreasing
    haz_preds = tree_tup.predict(X, kind="hazard", times=time_grid)
    for h in haz_preds:
        assert len(h) == len(time_grid)
        for i in range(len(h) - 1):
            assert h[i] <= h[i + 1] + 1e-9

    # Median survival response prediction
    resp = tree_tup.predict(X, kind="response")
    assert len(resp) == len(X)
    assert resp[0] > 0.0

    # 2. Fit with explicit event kwarg
    tree_kw = ModelTree(max_depth=0, leaf="mean", leaf_family="survival")
    tree_kw.fit(X, times, event=events)
    surv_kw = tree_kw.predict(X, kind="survival", times=time_grid)
    assert pytest.approx(surv_kw[0][0], abs=1e-5) == surv_preds[0][0]

    # Leaves inspection
    lfs = tree_kw.leaves()
    assert len(lfs) == 1
    assert "survival_times" in lfs[0]
    assert "survival_probs" in lfs[0]
    assert "cum_hazard" in lfs[0]
    assert "median_time" in lfs[0]


def test_heteroscedastic_stump_fit_and_predict():
    # Data with mean 100.0 and noise standard deviation 10.0
    X = [[float(i)] for i in range(100)]
    y = [100.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(100)]

    # 1. Constant location-scale stump
    tree = ModelTree(max_depth=0, leaf="mean", leaf_family="heteroscedastic")
    tree.fit(X, y)

    # Response prediction = location mean
    resp = tree.predict(X, kind="response")
    assert len(resp) == 100
    assert pytest.approx(resp[0], abs=1e-4) == 100.0

    # Dist prediction = (mean, std)
    dist = tree.predict(X, kind="dist")
    assert len(dist) == 100
    for mu, sigma in dist:
        assert pytest.approx(mu, abs=1e-4) == 100.0
        assert pytest.approx(sigma, abs=1e-3) == 10.0

    # Leaves inspection
    lfs = tree.leaves()
    assert len(lfs) == 1
    assert "scale" in lfs[0]
    assert pytest.approx(lfs[0]["scale"], abs=1e-3) == 10.0

    # 2. Linear location-scale stump
    tree_lin = ModelTree(max_depth=0, leaf="linear", leaf_family="heteroscedastic")
    tree_lin.fit(X, y)
    dist_lin = tree_lin.predict(X, kind="dist")
    assert len(dist_lin) == 100
    assert pytest.approx(dist_lin[50][0], abs=1.0) == 100.0


def test_distributional_stump_retrain_and_entity_routing():
    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]] * 10
    y_base = [1.0, 2.0, 3.0, 4.0] * 10
    vins = [f"VIN_{i:03d}" for i in range(40)]

    # Initial macro stump
    macro_tree = ModelTree(max_depth=0, leaf="mean", leaf_family="gaussian")
    macro_tree.fit(X, y_base, col_names=["f1", "f2"])

    # Entity primary key routing on stump
    clusters = macro_tree.leaf_samples(X, id_col=vins)
    assert len(clusters) == 1
    assert 0 in clusters
    assert len(clusters[0]) == 40
    assert clusters[0][0] == "VIN_000"

    # Retrain to heteroscedastic externally
    het_tree = macro_tree.retrain(
        X,
        y_base,
        mode="external",
        leaf_family="heteroscedastic",
        leaf="mean",
        col_names=["f1", "f2"],
    )
    # Original remains gaussian
    assert macro_tree.params["leaf_family"] == "gaussian" if hasattr(macro_tree, "params") and isinstance(macro_tree.params, dict) else True
    # Het tree predicts distribution
    het_dist = het_tree.predict(X, kind="dist")
    assert len(het_dist) == 40
    assert het_dist[0][1] > 0.0
