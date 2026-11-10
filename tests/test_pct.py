"""Tests for PredictiveClusteringTree (PCT / CLUS):
Multi-target variance reduction clustering tree with rule extraction.
"""

import pytest
from mltreelib import PredictiveClusteringTree


def test_pct_multi_target_clustering():
    pct = PredictiveClusteringTree(max_depth=5, min_samples_split=6, min_samples_leaf=3)

    # 2 distinct clusters in descriptive space X:
    # Group A: x0 in [0, 5], category "A" -> targets y0 ~ 10.0, y1 ~ 1.0
    # Group B: x0 in [50, 55], category "B" -> targets y0 ~ 500.0, y1 ~ 50.0
    x = []
    y = []

    for i in range(25):
        x.append([float(i * 0.2), "A"])
        y.append([10.0 + i * 0.05, 1.0])

    for i in range(25):
        x.append([50.0 + float(i * 0.2), "B"])
        y.append([500.0 + i * 0.05, 50.0])

    pct.fit(
        x,
        y=y,
        col_names=["val", "dept"],
        dtypes=["float", "cat"],
        target_names=["cost", "risk"],
    )

    assert pct.n_clusters >= 2

    # Predict cluster membership
    c_a = pct.predict_cluster([[2.0, "A"]])[0]
    c_b = pct.predict_cluster([[52.0, "B"]])[0]
    assert c_a != c_b, "Different data modes must map to different clusters"

    # Predict target prototypes
    proto_a = pct.predict_prototype([[2.0, "A"]])[0]
    proto_b = pct.predict_prototype([[52.0, "B"]])[0]
    assert proto_a[0] < 20.0 and proto_a[1] < 5.0
    assert proto_b[0] > 400.0 and proto_b[1] > 40.0

    # Verify rule extraction
    rules = pct.cluster_rules()
    assert len(rules) == pct.n_clusters
    for cid, rule_text, proto, size in rules:
        assert isinstance(cid, int)
        assert isinstance(rule_text, str)
        assert len(proto) == 2
        assert size >= 3


def test_pct_unsupervised_clustering():
    pct = PredictiveClusteringTree(max_depth=4, min_samples_split=6, min_samples_leaf=3)

    x = []
    for _ in range(20):
        x.append([1.0, 1.0])
    for _ in range(20):
        x.append([100.0, 100.0])

    # Unsupervised: target y is None
    pct.fit(x, col_names=["dim1", "dim2"], dtypes=["float", "float"])
    assert pct.n_clusters >= 2

    c1 = pct.predict_cluster([[1.0, 1.0]])[0]
    c2 = pct.predict_cluster([[100.0, 100.0]])[0]
    assert c1 != c2
