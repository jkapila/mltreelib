"""Tests for unified AnomalyTree and ClusterTree models."""

from mltreelib import AnomalyTree, ClusterTree


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
    at_eif = AnomalyTree(kind="eif", max_depth=4, contamination=0.1, seed=42)
    at_eif.fit(X, dtypes=dtypes)
    scores = at_eif.score_samples(X)
    preds = at_eif.predict(X)
    assert len(scores) == len(X)
    assert preds[-1] == 1

    # 2. Density Estimation Tree
    at_det = AnomalyTree(kind="det", max_depth=4, contamination=0.1)
    at_det.fit(X, dtypes=dtypes)
    det_scores = at_det.score_samples(X)
    densities = at_det.predict_density(X)
    assert len(det_scores) == len(X)
    assert len(densities) == len(X)


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
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.2],
        [10.0, 10.0], [10.2, 9.8], [9.9, 10.1],
    ]
    ct_pct = ClusterTree(kind="pct", max_depth=3, min_rows=2)
    ct_pct.fit(X)
    preds = ct_pct.predict(X)
    assert len(preds) == len(X)
    assert preds[0] == preds[1] == preds[2]
    assert preds[3] == preds[4] == preds[5]
    assert preds[0] != preds[3]
    assert ct_pct.n_clusters_ == 2

    rules = ct_pct.rules()
    assert len(rules) >= 2
    assert "rule" in rules[0]
    assert "prototype" in rules[0]


def test_cluster_tree_hdbscan_and_urf():
    X = [
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.2],
        [10.0, 10.0], [10.2, 9.8], [9.9, 10.1],
    ]
    # HDBSCAN
    ct_hdb = ClusterTree(kind="hdbscan", min_rows=2)
    ct_hdb.fit(X)
    labels = ct_hdb.labels_
    assert len(labels) == len(X)

    # URF
    ct_urf = ClusterTree(kind="urf", n_clusters=2, seed=42)
    ct_urf.fit(X)
    assert len(ct_urf.labels_) == len(X)


if __name__ == "__main__":
    test_anomaly_tree_eif_and_det()
    test_anomaly_tree_numeric_engines()
    test_cluster_tree_pct_and_rules()
    test_cluster_tree_hdbscan_and_urf()
    print("All unit tests passed successfully!")
