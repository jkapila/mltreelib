import pytest
from mltreelib import AnomalyTree, ClusterTree


def test_anomaly_tree_context_surprise():
    ct = AnomalyTree(kind="context", max_depth=3)
    ct.fit([[1, 2, 3, 4, 1, 2, 3, 4]])
    surprise = ct.surprise_score([1, 2, 3], 4)
    prob = ct.conditional_prob([1, 2, 3], 4)
    assert surprise > 0.0
    assert 0.0 < prob <= 1.0

    # Non-context engine should raise ValueError
    at = AnomalyTree(kind="hst", max_depth=3)
    at.fit([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="surprise_score is only supported by 'context'"):
        at.surprise_score([1, 2], 3)

def test_anomaly_tree_active_feedback():
    at = AnomalyTree(kind="active", max_depth=4)
    at.fit([[1.0, 2.0], [1.1, 2.1], [10.0, 10.0]])
    at.score_samples([[10.0, 10.0]])
    at.feedback([10.0, 10.0], is_anomaly=True, eta=0.5)
    scores_after = at.score_samples([[10.0, 10.0]])
    assert len(scores_after) == 1

    # Non-active engine should raise ValueError
    ct = AnomalyTree(kind="context", max_depth=3)
    with pytest.raises(ValueError, match="update_feedback is only supported by 'active'"):
        ct.feedback([1.0, 2.0], is_anomaly=True)

def test_anomaly_tree_threshold_tie_behavior():
    # If all inliers have exact same score (e.g. 0.0), strict tie-breaking ensures inliers aren't marked anomalies
    data = [[0.0, 0.0]] * 95 + [[10.0, 10.0]] * 5
    at = AnomalyTree(kind="hst", contamination=0.05, seed=42)
    at.fit(data)
    preds = at.predict(data)
    # The anomalies should be flagged, but the 95 inliers should not all be 1
    assert sum(preds) <= 10

def test_anomaly_and_cluster_missing_value_handling():
    data = [[1.0, 2.0], [1.2, None], [0.9, 2.1], [10.0, 10.0]]
    at = AnomalyTree(kind="hst", contamination=0.25)
    at.fit(data)
    preds = at.predict(data)
    assert len(preds) == 4
    assert preds[-1] == 1

    clt = ClusterTree(kind="cltree", max_depth=3)
    clt.fit(data)
    preds_clt = clt.predict(data)
    assert len(preds_clt) == 4
