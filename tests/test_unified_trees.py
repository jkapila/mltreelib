"""Tests for unified AnomalyTree and ClusterTree models."""

import json
import pickle

import pytest
from mltreelib import AnomalyTree, ClusterTree
from mltreelib.anomaly import AnomalyTree as PublicAnomalyTree
from mltreelib.cluster import ClusterTree as PublicClusterTree
from mltreelib.models import AnomalyTree as LegacyAnomalyTree
from mltreelib.models import ClusterTree as LegacyClusterTree


def test_unsupervised_public_imports_keep_class_identity():
    assert AnomalyTree is PublicAnomalyTree is LegacyAnomalyTree
    assert ClusterTree is PublicClusterTree is LegacyClusterTree


def test_anomaly_tree_eif_and_det():
    # 9 inliers + 1 clear outlier
    X = [
        [1.0, 2.0, "cat_A"],
        [1.1, 2.1, "cat_A"],
        [0.9, 1.9, "cat_A"],
        [1.2, 2.0, "cat_A"],
        [1.0, 2.2, "cat_A"],
        [1.05, 2.05, "cat_A"],
        [0.95, 1.95, "cat_A"],
        [1.15, 2.15, "cat_A"],
        [1.02, 1.98, "cat_A"],
        [50.0, 80.0, "cat_B"],  # clear outlier
    ]
    dtypes = ["numeric", "numeric", "categorical"]

    # 1. Extended Isolation Tree
    at_eif = AnomalyTree(
        kind="eif", max_depth=4, contamination=0.1, seed=42, subsample_size=10
    )
    at_eif.fit(X, dtypes=dtypes)
    scores = at_eif.score_samples(X)
    preds = at_eif.predict(X)
    assert len(scores) == len(X)
    assert preds[-1] == 1
    assert at_eif.predict(X, kind="binary") == preds
    assert at_eif.predict(X, kind="score") == scores
    assert len(at_eif.predict(X, kind="path_length")) == len(X)

    # 2. Density Estimation Tree
    at_det = AnomalyTree(kind="det", max_depth=4, contamination=0.1)
    at_det.fit(X, dtypes=dtypes)
    det_scores = at_det.score_samples(X)
    densities = at_det.predict_density(X)
    assert len(det_scores) == len(X)
    assert len(densities) == len(X)
    with pytest.raises(ValueError, match="binary.*score"):
        at_det.predict(X, kind="path_length")


def test_anomaly_tree_numeric_engines():
    X = [[0.0, 0.0], [0.1, 0.1], [-0.1, 0.0], [0.0, -0.1], [15.0, 15.0]]

    # HalfSpaceTree
    hst = AnomalyTree(kind="hst", max_depth=4, contamination=0.2, seed=42)
    hst.fit(X)
    assert len(hst.score_samples(X)) == len(X)

    # RandomCutTree
    rct = AnomalyTree(kind="rct", contamination=0.2, seed=42)
    rct.fit(X)
    assert len(rct.score_samples(X)) == len(X)


def test_cluster_tree_pct_and_rules():
    X = [
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.2],
        [10.0, 10.0],
        [10.2, 9.8],
        [9.9, 10.1],
    ]
    ct_pct = ClusterTree(kind="pct", max_depth=3, min_rows=2)
    ct_pct.fit(X)
    preds = ct_pct.predict(X)
    assert len(preds) == len(X)
    assert preds[0] == preds[1] == preds[2]
    assert preds[3] == preds[4] == preds[5]
    assert preds[0] != preds[3]
    assert ct_pct.n_clusters_ == 2
    assert ct_pct.predict(X, kind="cluster") == ct_pct.labels_ == preds
    with pytest.raises(ValueError, match="Use one of.*cluster"):
        ct_pct.predict(X, kind="distances")

    rules = ct_pct.rules()
    assert len(rules) >= 2
    assert "rule" in rules[0]
    assert "prototype" in rules[0]


def test_cluster_tree_hdbscan_and_urf():
    X = [
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.2],
        [10.0, 10.0],
        [10.2, 9.8],
        [9.9, 10.1],
    ]
    # HDBSCAN
    ct_hdb = ClusterTree(kind="hdbscan", min_rows=2)
    ct_hdb.fit(X)
    labels = ct_hdb.labels_
    assert len(labels) == len(X)

    # URF
    ct_urf = ClusterTree(kind="urf", n_clusters=2, seed=42, n_trees=10)
    ct_urf.fit(X)
    assert len(ct_urf.labels_) == len(X)
    query = [X[0], X[-1], [5.0, 5.0]]
    distances = ct_urf.predict(query, kind="distances")
    assert len(distances) == len(query)
    assert all(len(row) == len(X) for row in distances)
    assert distances[0][0] == distances[1][-1] == 0.0
    nearest = [ct_urf.labels_[row.index(min(row))] for row in distances]
    assert ct_urf.predict(query, kind="cluster") == nearest


def test_hdbscan_and_urf_native_snapshots_round_trip(tmp_path):
    X = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
    for kind in ("hdbscan", "urf"):
        model = ClusterTree(kind=kind, min_rows=2, n_clusters=2, n_trees=4)
        model.fit(X)
        restored = ClusterTree.from_json(model.to_json())
        assert restored.predict(X) == model.predict(X)


@pytest.mark.parametrize("kind", ["eif", "det", "hst", "rct", "active"])
def test_every_numeric_anomaly_engine_round_trips(kind):
    X = [[0.0, 0.0], [0.1, 0.2], [4.9, 5.0], [5.1, 4.8]]
    model = AnomalyTree(kind=kind, max_depth=4, contamination=0.25, seed=7)
    model.fit(X)
    restored = AnomalyTree.from_json(model.to_json())
    assert restored.predict(X, kind="score") == model.predict(X, kind="score")
    assert restored.predict(X) == model.predict(X)


def test_context_anomaly_engine_round_trips():
    model = AnomalyTree(kind="context", max_depth=3, alphabet_size=16)
    model.fit([[1, 2, 3, 1, 2, 3]])
    restored = AnomalyTree.from_json(model.to_json())
    assert restored.conditional_prob([1, 2], 3) == model.conditional_prob([1, 2], 3)
    assert restored.surprise_score([1, 2], 3) == model.surprise_score([1, 2], 3)


@pytest.mark.parametrize("kind", ["pct", "cltree", "hdbscan", "urf"])
def test_every_cluster_engine_round_trips(kind):
    X = [[0.0, 0.0], [0.1, 0.2], [4.9, 5.0], [5.1, 4.8]]
    model = ClusterTree(kind=kind, max_depth=4, min_rows=2, n_clusters=2, n_trees=4, seed=7)
    model.fit(X)
    restored = ClusterTree.from_json(model.to_json())
    assert restored.predict(X) == model.predict(X)
    assert restored.labels_ == model.labels_


@pytest.mark.parametrize("kind", ["eif", "pct"])
def test_unsupervised_json_restores_fitted_state(kind, tmp_path):
    features = [
        ["low", 0.0],
        ["low", 1.0],
        ["low", 2.0],
        ["high", 50.0],
        ["high", 51.0],
        ["high", 52.0],
    ]
    query = [["low", 1.5], ["high", 51.5], ["unseen", None]]
    if kind == "eif":
        model = AnomalyTree(kind=kind, contamination=0.5, max_depth=4)
    else:
        model = ClusterTree(kind=kind, min_rows=2)

    def predict(tree):
        if kind == "eif":
            return (
                tree.predict(query),
                tree.predict(query, kind="score"),
                tree.predict(query, kind="path_length"),
                tree.threshold,
            )
        return (
            tree.predict(query),
            tree.labels_,
            tree.n_clusters_,
            tree.rules(),
        )

    model.fit(features, col_names=["band", "value"])
    path = tmp_path / f"{kind}.json"
    model.save(path)

    snapshot = json.loads(path.read_text())
    assert snapshot["category_levels"][0] == {"low": 0, "high": 1}
    model_class = type(model)
    expected = predict(model)
    for restored in (
        model_class.load(path),
        model_class.from_bytes(model.to_bytes()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert predict(restored) == expected

    snapshot["native"]["model"]["nodes"][0]["left_child"] = 0
    with pytest.raises(ValueError, match="topology|node"):
        model_class.from_json(json.dumps(snapshot))


def test_unsupervised_snapshot_requires_a_fitted_engine():
    with pytest.raises(ValueError, match="not fitted"):
        AnomalyTree().to_json()


if __name__ == "__main__":
    test_anomaly_tree_eif_and_det()
    test_anomaly_tree_numeric_engines()
    test_cluster_tree_pct_and_rules()
    test_cluster_tree_hdbscan_and_urf()
    print("All unit tests passed successfully!")
