"""Unified Task Facades Showcase: AnomalyModel, ClusterModel, RuleModel, ComponentModel.

Demonstrates:
1. Primary task facades without universal dispatcher or scikit-learn parameter protocol.
2. AnomalyModel with PCA/NMF reconstruction anomaly scoring alongside tree engines.
3. ClusterModel for predictive clustering and density-based partitioning.
4. RuleModel for Quinlan C4.5/C5.0 rule induction and pessimistic pruning.
5. Inspection via .config() and ModelReport diagnostic outputs.
"""

from __future__ import annotations

import math
from mltreelib.anomaly import AnomalyModel
from mltreelib.cluster import ClusterModel
from mltreelib.components import ComponentModel
from mltreelib.rules import RuleModel


def run_facade_showcase() -> None:
    print("=" * 78)
    print("Unified Task Facades Showcase (Spec 25)")
    print("=" * 78)

    # 1. ComponentModel (Decomposition & Projection)
    print("\n[1] ComponentModel (PCA Subspace)")
    comp_data = [[float(i), 2.0 * float(i) + 0.1 * (i % 3)] for i in range(40)]
    comp = ComponentModel(kind="pca", n_components=1).fit(comp_data)
    scores = comp.transform(comp_data[:3])
    print(f"Transformed scores for 3 points: {scores}")
    print(f"Config: {comp.config()}")
    assert not hasattr(comp, "get_params")

    # 2. AnomalyModel (Reconstruction Error vs Tree)
    print("\n[2] AnomalyModel (PCA Reconstruction Anomaly Detection)")
    train_3d = [[float(i) * 0.1, 2.0 * float(i) * 0.1, 0.01 * (i % 3)] for i in range(80)]
    anom_pca = AnomalyModel(kind="pca", contamination=0.05, n_components=2)
    anom_pca.fit(train_3d)

    test_points = [
        [5.0, 10.0, 0.01],   # inlier
        [5.0, 10.0, 50.0],   # outlier (orthogonal spike)
    ]
    anom_scores = anom_pca.score_samples(test_points)
    anom_preds = anom_pca.predict(test_points)
    print(f"Inlier vs Outlier Scores: {anom_scores[0]:.4f} vs {anom_scores[1]:.4f}")
    print(f"Binary Predictions (0=Normal, 1=Anomaly): {anom_preds}")

    # 3. ClusterModel (Predictive Clustering Tree)
    print("\n[3] ClusterModel (Divisive Predictive Tree Clustering)")
    c1 = [[float(i) * 0.1, float(i) * 0.1] for i in range(25)]
    c2 = [[10.0 + float(i) * 0.1, 10.0 + float(i) * 0.1] for i in range(25)]
    clust_data = c1 + c2
    cluster_mod = ClusterModel(kind="pct", n_clusters=2, max_depth=3)
    labels = cluster_mod.fit_predict(clust_data)
    print(f"Discovered Clusters: {cluster_mod.n_clusters_}")
    print(f"Sample Cluster Labels: {labels[:5]} ... {labels[-5:]}")

    # 4. RuleModel (Quinlan C4.5 / C5.0 Rule Extraction)
    print("\n[4] RuleModel (Transparent Propositional Rules)")
    rule_x = [[float(i)] for i in range(50)]
    rule_y = [0.0 if i < 25 else 10.0 for i in range(50)]
    rule_mod = RuleModel(kind="c50", max_depth=3, min_rows=5, prune=True, cf=0.25)
    rule_mod.fit(rule_x, rule_y)

    print(f"Extracted Rules ({len(rule_mod.rules_)} rules):")
    for r in rule_mod.rules_:
        print(f"  {r}")
    rule_preds = rule_mod.predict([[10.0], [40.0]])
    print(f"Predictions for x=10 and x=40: {rule_preds}")

    # 5. Diagnostic ModelReport
    print("\n[5] ModelReport Text Output from RuleModel:")
    print(rule_mod.report.to_text())


if __name__ == "__main__":
    run_facade_showcase()
