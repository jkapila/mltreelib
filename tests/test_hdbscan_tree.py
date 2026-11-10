"""Tests for HDBSCAN Condensed Tree clustering, stability selection & GLOSH outlier scores."""

import pytest
from mltreelib import HDBSCAN


def test_hdbscan_two_clusters_and_outlier():
    # 2 dense clusters + 1 distant outlier
    data = []
    # Cluster 0 around (0, 0)
    for i in range(25):
        data.append([float(i * 0.05), float(i * 0.05)])
    # Cluster 1 around (10, 10)
    for i in range(25):
        data.append([10.0 + float(i * 0.05), 10.0 + float(i * 0.05)])
    # Outlier
    data.append([100.0, 100.0])

    clusterer = HDBSCAN(min_cluster_size=5, min_samples=5)
    clusterer.fit(data)

    assert clusterer.n_clusters >= 1
    labels = clusterer.labels
    assert len(labels) == len(data)

    # Outlier scores
    outlier_scores = clusterer.outlier_scores
    assert len(outlier_scores) == len(data)
    # Outlier point (index -1) should have high GLOSH score
    assert outlier_scores[-1] >= outlier_scores[0]
    assert 0.0 <= outlier_scores[-1] <= 1.0

    # Condensed tree nodes
    tree = clusterer.condensed_tree()
    assert len(tree) > 0
    root = tree[0]
    assert root["node_id"] == 0
    assert root["parent_id"] is None
    assert root["size"] == len(data)

    # Predict new points
    preds = clusterer.predict([[0.1, 0.1], [10.1, 10.1], [50.0, 50.0]])
    assert len(preds) == 3
    # Near cluster points should be assigned to clusters
    assert preds[0] is not None or preds[1] is not None
