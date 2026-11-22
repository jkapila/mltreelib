"""Tests for ComponentModel and unified ModelReport."""

import pytest
from mltreelib.components import ComponentModel, ComponentResult
from mltreelib.report import ModelReport


def test_pca_component_model_lifecycle(tmp_path):
    # Linear dataset with small noise
    data = [
        [float(i), 2.0 * float(i) + 0.1 * (i % 3), 0.5 * float(i)]
        for i in range(50)
    ]

    model = ComponentModel(kind="pca", n_components=2, standardize=True)
    assert model.config()["kind"] == "pca"
    assert model.config()["n_components"] == 2

    # Verify absence of set_params
    assert not hasattr(model, "set_params")

    # Fit and transform
    model.fit(data)
    scores = model.transform(data)
    assert len(scores) == 50
    assert len(scores[0]) == 2

    # Direct properties
    comps = model.components_
    assert len(comps) == 2
    assert len(comps[0]) == 3

    ev = model.explained_variance_
    evr = model.explained_variance_ratio_
    assert ev is not None and len(ev) == 2
    assert evr is not None and len(evr) == 2
    assert sum(evr) > 0.95

    # Inverse transform and reconstruct
    reconstructed = model.reconstruct(data)
    assert len(reconstructed) == 50
    inv = model.inverse_transform(scores)
    assert len(inv) == 50

    # Reconstruction errors
    errs = model.score_samples(data)
    assert len(errs) == 50
    assert sum(errs) / 50.0 < 0.1

    # Report
    rep = model.report
    assert isinstance(rep, ModelReport)
    assert rep.title == "Principal Component Analysis"
    txt = rep.to_text()
    assert "1. OBJECTIVE" in txt
    assert "2. LEARNER" in txt
    assert "3. CONTROLLER" in txt
    assert "4. DIAGNOSTICS" in txt
    md = rep.to_markdown()
    assert "## 1. Objective" in md
    d = rep.to_dict()
    assert d["objective"]["name"] == "Maximum Variance Subspace Projection"

    # Persistence round trip (JSON)
    json_str = model.to_json()
    loaded_json = ComponentModel.from_json(json_str)
    loaded_scores = loaded_json.transform(data)
    for r1, r2 in zip(scores, loaded_scores):
        for v1, v2 in zip(r1, r2):
            assert abs(v1 - v2) < 1e-6

    # Persistence round trip (File)
    model_path = tmp_path / "pca_model.json"
    model.save(model_path)
    loaded_file = ComponentModel.load(model_path)
    file_scores = loaded_file.transform(data)
    for r1, r2 in zip(scores, file_scores):
        for v1, v2 in zip(r1, r2):
            assert abs(v1 - v2) < 1e-6


def test_svd_component_model():
    data = [[float(i), 3.0 * float(i)] for i in range(1, 30)]
    model = ComponentModel(kind="svd", n_components=1)
    scores = model.fit_transform(data)
    assert len(scores) == 29
    assert len(scores[0]) == 1

    recon = model.reconstruct(data)
    errs = model.score_samples(data)
    assert max(errs) < 1e-6


def test_whitening_component_model():
    data = [
        [float(i) * 0.1, 1.5 * float(i) * 0.1 + (i % 5) * 0.2]
        for i in range(60)
    ]
    model = ComponentModel(kind="whitening", whitening_kind="zca")
    scores = model.fit_transform(data)
    assert len(scores) == 60
    assert len(scores[0]) == 2

    # Invert
    restored = model.inverse_transform(scores)
    for row_orig, row_rest in zip(data, restored):
        for v1, v2 in zip(row_orig, row_rest):
            assert abs(v1 - v2) < 1e-4


def test_fastica_component_model():
    # Two independent mixed signals
    s1 = [1.0 if (i % 8) < 4 else -1.0 for i in range(100)]
    s2 = [(i % 10) / 5.0 - 1.0 for i in range(100)]
    mixed = [[0.7 * s1[i] + 0.3 * s2[i], 0.2 * s1[i] - 0.8 * s2[i]] for i in range(100)]

    model = ComponentModel(kind="ica", n_components=2, seed=42)
    sources = model.fit_transform(mixed)
    assert len(sources) == 100
    assert len(sources[0]) == 2

    recon = model.reconstruct(mixed)
    errs = model.score_samples(mixed)
    assert sum(errs) / 100.0 < 0.1


def test_nmf_component_model():
    # Non-negative combinations of basis factors
    data = [
        [float(i % 4 + 1), float(i % 3 + 2), float(i % 5 + 1)]
        for i in range(40)
    ]
    model = ComponentModel(kind="nmf", n_components=2, seed=42)
    w = model.fit_transform(data)
    assert len(w) == 40
    assert len(w[0]) == 2

    # All factors in components must be non-negative
    for row in model.components_:
        for val in row:
            assert val >= 0.0

    errs = model.score_samples(data)
    assert sum(errs) / 40.0 < 1.0


def test_invalid_input_handling():
    model = ComponentModel(kind="pca")
    with pytest.raises(ValueError, match="not fitted"):
        model.transform([[1.0, 2.0]])

    # Missing value handling
    with pytest.raises(ValueError, match="Missing or non-finite"):
        model.fit([[1.0, None], [2.0, 3.0]])

    # NMF negative values
    nmf = ComponentModel(kind="nmf")
    with pytest.raises(ValueError, match="non-negative"):
        nmf.fit([[1.0, -2.0], [3.0, 4.0]])
