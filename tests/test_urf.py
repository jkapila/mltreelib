"""Tests for Unsupervised Random Forest (URF) proximity, agglomeration & outlier scoring."""

import pytest
from mltreelib import UnsupervisedRandomForest


def test_unsupervised_random_forest_two_clusters():
    data = []
    # Cluster 1 near (0, 0)
    for i in range(25):
        data.append([float(i * 0.05), float(-i * 0.05)])
    # Cluster 2 near (20, 20)
    for i in range(25):
        data.append([20.0 + float(i * 0.05), 20.0 - float(i * 0.05)])
    # Outlier point
    data.append([100.0, 100.0])

    urf = UnsupervisedRandomForest(n_trees=35, max_depth=6, min_samples_leaf=2, seed=42)
    urf.fit(data, n_clusters=2)

    assert urf.n_clusters == 2
    labels = urf.labels
    assert len(labels) == len(data)

    # Check proximity matrix
    prox = urf.proximity_matrix
    assert len(prox) == len(data)
    assert len(prox[0]) == len(data)
    # Self-proximity is 1.0
    for i in range(len(data)):
        assert abs(prox[i][i] - 1.0) < 1e-5

    # Within-cluster proximity > cross-cluster proximity
    within_cluster = prox[0][1]
    across_clusters = prox[0][30]
    assert within_cluster >= across_clusters

    # Check pairwise distances: D = sqrt(1 - P)
    dists = urf.pairwise_distances()
    assert len(dists) == len(data)
    assert abs(dists[0][0] - 0.0) < 1e-5

    # Outlier scores
    outlier_scores = urf.outlier_scores
    assert len(outlier_scores) == len(data)
    # Outlier should have higher score than normal points
    assert outlier_scores[-1] >= outlier_scores[0]

    # Leaf assignments
    leaf_ids = urf.apply([[0.0, 0.0], [20.0, 20.0]])
    assert len(leaf_ids) == 2
    assert len(leaf_ids[0]) == 35


def test_urf_knobs():
    data = []
    for i in range(15):
        data.append([float(i * 0.1), float(i * 0.1)])
    for i in range(15):
        data.append([10.0 + float(i * 0.1), 10.0 + float(i * 0.1)])

    # Single linkage
    urf_single = UnsupervisedRandomForest(n_trees=10, max_features=1, linkage="single", seed=42)
    urf_single.fit(data, n_clusters=2)
    assert urf_single.n_clusters == 2
    assert len(urf_single.labels) == len(data)

    # Complete linkage
    urf_complete = UnsupervisedRandomForest(n_trees=10, max_features=1, linkage="complete", seed=42)
    urf_complete.fit(data, n_clusters=2)
    assert urf_complete.n_clusters == 2
    assert len(urf_complete.labels) == len(data)

