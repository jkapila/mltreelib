"""Tests for unified task facades: AnomalyModel, ClusterModel, RuleModel, and absence of sklearn params."""

import pytest
from mltreelib.anomaly import AnomalyModel, AnomalyResult, AnomalyTree
from mltreelib.cluster import ClusterModel, ClusterResult, ClusterTree
from mltreelib.components import ComponentModel
from mltreelib import ModelTree
from mltreelib.report import ModelReport
from mltreelib.rules import RuleModel, RuleResult


def test_anomaly_model_pca_and_nmf_reconstruction():
    # 2D subspace embedded in 3D
    data = [
        [float(i) * 0.1, 2.0 * float(i) * 0.1, 0.01 * (i % 3)]
        for i in range(80)
    ]

    model_pca = AnomalyModel(kind="pca", contamination=0.1, n_components=2)
    assert not hasattr(model_pca, "get_params")
    assert not hasattr(model_pca, "set_params")
    assert model_pca.config()["kind"] == "pca"

    model_pca.fit(data)
    inlier = [[2.0, 4.0, 0.01]]
    outlier = [[2.0, 4.0, 20.0]]

    inlier_score = model_pca.score_samples(inlier)[0]
    outlier_score = model_pca.score_samples(outlier)[0]
    assert outlier_score > inlier_score * 50.0

    preds = model_pca.predict(inlier + outlier)
    assert preds[0] == 0  # inlier
    assert preds[1] == 1  # anomaly

    rep = model_pca.report
    assert isinstance(rep, ModelReport)
    assert "Objective" in rep.to_markdown()

    # NMF anomaly detection (strictly positive data)
    pos_data = [
        [float(i % 5 + 1), float(i % 3 + 2), float(i % 4 + 1)]
        for i in range(60)
    ]
    model_nmf = AnomalyModel(kind="nmf", contamination=0.1, n_components=2)
    model_nmf.fit(pos_data)
    scores = model_nmf.score_samples(pos_data)
    assert len(scores) == 60
    assert all(s >= 0.0 for s in scores)
    assert model_nmf.report.title != ""


def test_anomaly_model_tree_and_compatibility():
    data = [[float(i), float(i % 4)] for i in range(50)]
    model = AnomalyModel(kind="eif", max_depth=4, contamination=0.08)
    model.fit(data)
    scores = model.score_samples(data)
    assert len(scores) == 50
    preds = model.predict(data)
    assert len(preds) == 50
    assert sum(preds) > 0  # at least some anomalies flagged

    # Compatibility alias AnomalyTree
    tree = AnomalyTree(kind="eif", max_depth=4, contamination=0.08)
    tree.fit(data)
    tree_scores = tree.score_samples(data)
    assert len(tree_scores) == 50
    assert hasattr(tree, "config")
    assert not hasattr(tree, "get_params")


def test_cluster_model_facade():
    # Two distinct clusters
    c1 = [[float(i) * 0.1, float(i) * 0.1] for i in range(30)]
    c2 = [[10.0 + float(i) * 0.1, 10.0 + float(i) * 0.1] for i in range(30)]
    data = c1 + c2

    model = ClusterModel(kind="pct", n_clusters=2, max_depth=3)
    assert not hasattr(model, "get_params")
    assert not hasattr(model, "set_params")
    assert model.config()["kind"] == "pct"

    labels = model.fit_predict(data)
    assert len(labels) == 60
    assert model.n_clusters_ >= 2

    rep = model.report
    assert isinstance(rep, ModelReport)
    assert "Clustering" in rep.to_text()

    # Compatibility alias ClusterTree
    ctree = ClusterTree(kind="pct", n_clusters=2, max_depth=3)
    ctree_labels = ctree.fit_predict(data)
    assert len(ctree_labels) == 60
    assert ctree.config()["kind"] == "pct"


def test_rule_model_facade(tmp_path):
    # Step function dataset:
    # y = 0.0 if x[0] < 25 else 10.0
    x = [[float(i)] for i in range(50)]
    y = [0.0 if i < 25 else 10.0 for i in range(50)]

    rule_mod = RuleModel(kind="c50", max_depth=3, min_rows=5, prune=True, cf=0.25)
    assert not hasattr(rule_mod, "get_params")
    assert not hasattr(rule_mod, "set_params")
    assert rule_mod.config()["prune"] is True
    assert rule_mod.config()["cf"] == 0.25

    rule_mod.fit(x, y)
    preds = rule_mod.predict([[10.0], [40.0]])
    assert preds[0] == 0.0
    assert preds[1] == 10.0

    assert len(rule_mod.rules_) >= 2
    assert rule_mod.default_prediction_ is not None

    rep = rule_mod.report
    assert isinstance(rep, ModelReport)
    assert "Rule Set" in rep.to_text()

    # Fit from existing ModelTree
    tree = ModelTree(max_depth=3, min_rows=5)
    tree.fit(x, y)
    rule_from_tree = RuleModel().fit(tree)
    assert len(rule_from_tree.rules_) >= 2

    # Persistence round-trip
    save_path = tmp_path / "rule_model.json"
    rule_mod.save(save_path)
    loaded_mod = RuleModel.load(save_path)
    loaded_preds = loaded_mod.predict([[10.0], [40.0]])
    assert loaded_preds == preds
