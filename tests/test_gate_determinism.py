"""Significance-gate statistics must be bit-identical for identical inputs."""

import json

import numpy as np
import pytest

from mltreelib import ModelTree


def _data():
    rng = np.random.default_rng(2026)
    n = 300
    score = rng.uniform(-2, 2, n)
    zip_idx = rng.integers(0, 50, n)
    tier = rng.integers(0, 2, n).astype(float)
    y = np.where(score > 0, 2 * score, -score) + 0.5 * tier + rng.normal(0, 0.3, n)
    x = [[float(score[i]), f"z{zip_idx[i]:02d}", float(tier[i])] for i in range(n)]
    return x, y.tolist()


@pytest.mark.parametrize("test", ["mob", "ctree"])
def test_categorical_gate_statistic_is_deterministic(test):
    x, y = _data()
    seen = set()
    for _ in range(12):
        t = ModelTree(gain="ctree" if test == "ctree" else "sse", split_test=test, max_depth=1, min_rows=15, prune="none", seed=42)
        t.fit(x, y, col_names=["score", "zip_code", "tier"], dtypes=["numeric", "categorical", "numeric"])
        variables = t.root_split()["gate"]["variables"]
        seen.add(json.dumps([(v["statistic"], v.get("p_value")) for v in variables]))
    assert len(seen) == 1


def test_saved_model_is_byte_identical_for_identical_fits():
    """Timing diagnostics are not serialised, so the saved model is reproducible."""
    x, y = _data()
    saved = set()
    for _ in range(5):
        t = ModelTree(gain="ctree", split_test="ctree", max_depth=3, min_rows=15, prune="none", seed=42)
        t.fit(x, y, col_names=["score", "zip_code", "tier"], dtypes=["numeric", "categorical", "numeric"])
        saved.add(t.to_json())
    assert len(saved) == 1
