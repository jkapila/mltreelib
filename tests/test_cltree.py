"""Tests for CLTree (Clustering Trees - Liu et al. KDD 2000):
Decision tree clustering via analytical background non-data.
"""

import pytest
from mltreelib import CLTree


def test_cltree_dense_clusters_and_noise():
    clt = CLTree(max_depth=5, min_samples_leaf=5, min_density=0.55)

    # 2 dense clusters separated by empty space
    data = []
    for i in range(40):
        data.append([float(i * 0.05), 1.0])
    for i in range(40):
        data.append([10.0 + float(i * 0.05), 20.0])

    clt.fit(data, col_names=["x1", "x2"])
    assert clt.n_clusters >= 2

    # Cluster 1 query
    c1 = clt.predict_cluster([[0.5, 1.0]])[0]
    assert c1 is not None

    # Cluster 2 query
    c2 = clt.predict_cluster([[10.5, 20.0]])[0]
    assert c2 is not None
    assert c1 != c2

    # Noise query in empty gap
    noise = clt.predict_cluster([[5.0, 10.0]])[0]
    assert noise is None, "Sparse background space must be classified as None (noise/outlier)"

    # Check bounding boxes
    bboxes = clt.cluster_bounding_boxes()
    assert len(bboxes) == clt.n_clusters
    for cid, bounds, size, density in bboxes:
        assert isinstance(cid, int)
        assert len(bounds) == 2
        assert size >= 5
        assert density >= 0.55
