"""Tests for mltreelib.experimental namespaces and full snapshot persistence round-trips."""

from mltreelib.experimental import anomaly, cluster, components, rules
from mltreelib.experimental.anomaly import AnomalyModel, AnomalyResult
from mltreelib.experimental.cluster import ClusterModel, ClusterResult
from mltreelib.experimental.components import ComponentModel, ComponentResult
from mltreelib.experimental.rules import RuleModel, RuleResult


def test_experimental_namespace_imports():
    assert anomaly.AnomalyModel is AnomalyModel
    assert cluster.ClusterModel is ClusterModel
    assert components.ComponentModel is ComponentModel
    assert rules.RuleModel is RuleModel


def test_snapshot_persistence_all_facades(tmp_path):
    # 1. ComponentModel
    data = [[float(i), 2.0 * float(i)] for i in range(20)]
    comp = ComponentModel(kind="pca", n_components=1).fit(data)
    p_comp = tmp_path / "comp.json"
    comp.save(p_comp)
    loaded_comp = ComponentModel.load(p_comp)
    assert loaded_comp.transform(data) == comp.transform(data)

    # 2. AnomalyModel (tree-based)
    anom = AnomalyModel(kind="eif", max_depth=3).fit(data)
    p_anom = tmp_path / "anom.json"
    anom.save(p_anom)
    loaded_anom = AnomalyModel.load(p_anom)
    assert loaded_anom.score_samples(data) == anom.score_samples(data)

    # 3. ClusterModel (tree-based)
    clust = ClusterModel(kind="pct", max_depth=3).fit(data)
    p_clust = tmp_path / "clust.json"
    clust.save(p_clust)
    loaded_clust = ClusterModel.load(p_clust)
    assert loaded_clust.predict(data) == clust.predict(data)

    # 4. RuleModel
    y = [1.0 if r[0] > 10.0 else 0.0 for r in data]
    rule = RuleModel(max_depth=3).fit(data, y)
    p_rule = tmp_path / "rule.json"
    rule.save(p_rule)
    loaded_rule = RuleModel.load(p_rule)
    assert loaded_rule.predict(data) == rule.predict(data)
