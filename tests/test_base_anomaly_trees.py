"""Tests for the base anomaly trees in mltreelib:
- DensityEstimationTree (DET)
- ExtendedIsolationTree (EIF)
- HalfSpaceTree (HST)
- RandomCutTree (RCT)
- ActiveAnomalyTree (AAD)
- ContextAnomalyTree (CAT)
"""

import math
import pytest
from mltreelib import (
    ActiveAnomalyTree,
    ContextAnomalyTree,
    DensityEstimationTree,
    ExtendedIsolationTree,
    HalfSpaceTree,
    RandomCutTree,
)


def test_density_estimation_tree():
    det = DensityEstimationTree(max_depth=5, min_samples_split=4, min_leaf_samples=2)
    # Dense cluster [0.0..1.0], outlier at 50.0
    x = [[i * 0.02, 1.0] for i in range(50)]
    det.fit(x, col_names=["x1", "x2"], dtypes=["float", "float"])

    d_in = det.predict_density([[0.5, 1.0]])[0]
    d_out = det.predict_density([[50.0, 1.0]])[0]
    assert d_in > d_out, f"Dense region {d_in} must exceed outlier region {d_out}"

    score_in = det.predict_anomaly_score([[0.5, 1.0]])[0]
    score_out = det.predict_anomaly_score([[50.0, 1.0]])[0]
    assert score_out > score_in, f"Outlier score {score_out} must exceed normal score {score_in}"


def test_extended_isolation_tree_categorical():
    eif = ExtendedIsolationTree(max_depth=6, extension_level=1, seed=42)
    # Cluster with numeric variation and categorical values 0 / 1
    x = [[i * 0.05, "cat_a" if i % 2 == 0 else "cat_b"] for i in range(50)]
    # Outlier with extreme numeric and unseen/rare categorical
    x.append([100.0, "cat_c"])

    eif.fit(x, col_names=["val", "grp"], dtypes=["float", "cat"])

    path_normal = eif.path_length([[1.0, "cat_a"]])[0]
    path_outlier = eif.path_length([[100.0, "cat_c"]])[0]
    assert path_outlier < path_normal, "Outlier isolates in fewer tree steps"

    score_normal = eif.anomaly_score([[1.0, "cat_a"]])[0]
    score_outlier = eif.anomaly_score([[100.0, "cat_c"]])[0]
    assert score_outlier > score_normal, "Outlier anomaly score must be higher"


def test_half_space_tree():
    hst = HalfSpaceTree(max_depth=6, seed=42)
    data = [[i * 0.05, 0.5] for i in range(40)]
    hst.fit(data)

    s_normal = hst.anomaly_score([[1.0, 0.5]])[0]
    s_outlier = hst.anomaly_score([[50.0, 50.0]])[0]
    assert s_outlier > s_normal, "HST anomaly score must be higher for low-mass space"

    m_normal = hst.mass_score([[1.0, 0.5]])[0]
    m_outlier = hst.mass_score([[50.0, 50.0]])[0]
    assert m_normal > m_outlier, "Mass must be higher inside the dense training cluster"

    # Streaming update
    hst.insert([50.0, 50.0])
    m_after = hst.mass_score([[50.0, 50.0]])[0]
    assert m_after > m_outlier, "Point mass should increase after streaming observation"


def test_random_cut_tree():
    rct = RandomCutTree(seed=42)
    data = [[i * 0.05, 0.0] for i in range(40)]
    rct.fit(data)

    codisp_normal = rct.codisp([[1.0, 0.0]])[0]
    codisp_outlier = rct.codisp([[100.0, 100.0]])[0]
    assert codisp_outlier > codisp_normal, "CoDisp must be higher for isolated point"


def test_active_anomaly_tree_feedback():
    aad = ActiveAnomalyTree(max_depth=6, seed=42)
    data = [[i * 0.05] for i in range(40)]
    aad.fit(data)

    probe = [1.5]
    score_before = aad.anomaly_score([probe])[0]

    # Human analyst labels point as true anomaly
    aad.update_feedback(probe, is_anomaly=True, eta=0.8)
    score_after = aad.anomaly_score([probe])[0]
    assert score_after > score_before, "True anomaly feedback must boost anomaly score"


def test_context_anomaly_tree():
    cat = ContextAnomalyTree(max_depth=4, alphabet_size=100)
    # Train on typical sequences e.g. [1, 2, 3], [1, 2, 3, 4]
    seqs = [[1, 2, 3, 4]] * 25
    cat.fit(seqs)

    p_expected = cat.conditional_prob([1, 2], 3)
    p_unexpected = cat.conditional_prob([1, 2], 99)
    assert p_expected > p_unexpected, "Common sequence symbol must have higher conditional prob"

    surprise_expected = cat.surprise_score([1, 2], 3)
    surprise_unexpected = cat.surprise_score([1, 2], 99)
    assert surprise_unexpected > surprise_expected, "Surprise in bits must be higher for rare transition"


def test_anomaly_knobs_configuration():
    # Test DET knobs
    det = DensityEstimationTree(max_depth=4, min_density_gain=0.01, alpha=0.05)
    det.fit([[1.0], [2.0], [3.0]], col_names=["x"], dtypes=["float"])
    assert len(det.predict_density([[2.0]])) == 1

    # Test EIF subsample_size
    eif = ExtendedIsolationTree(max_depth=4, extension_level=1, subsample_size=10, seed=42)
    eif.fit([[float(i)] for i in range(100)], col_names=["x"], dtypes=["float"])
    assert len(eif.anomaly_score([[50.0]])) == 1

    # Test HST window_size and initial_mass
    hst = HalfSpaceTree(max_depth=4, window_size=100, initial_mass=1.0, seed=42)
    hst.fit([[1.0, 2.0], [2.0, 3.0]])
    assert len(hst.anomaly_score([[1.5, 2.5]])) == 1

    # Test RCT sample_size
    rct = RandomCutTree(sample_size=32, seed=42)
    rct.fit([[float(i), float(i)] for i in range(100)])
    assert len(rct.codisp([[50.0, 50.0]])) == 1

    # Test AAD eta and prior_weight
    aad = ActiveAnomalyTree(max_depth=4, eta=0.2, prior_weight=2.0, seed=42)
    aad.fit([[1.0], [2.0], [3.0]])
    assert len(aad.anomaly_score([[2.0]])) == 1

    # Test CAT smoothing_discount
    cat = ContextAnomalyTree(max_depth=3, alphabet_size=20, smoothing_discount=0.2)
    cat.fit([[1, 2, 3]])
    prob = cat.conditional_prob([1, 2], 3)
    assert 0.0 < prob <= 1.0

