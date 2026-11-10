"""Tests for TrieKNN (Jitin Kapila 2018/2025):
Mixed-type categorical prefix trie with local numerical K-Nearest Neighbors.
"""

import pytest
from mltreelib import TrieKNN


def test_trie_knn_classification_mixed_data():
    model = TrieKNN(task="classification", k=3, weights="distance")

    # 2 categorical columns ("country", "tier"), 2 numerical columns ("age", "spend")
    # Group 1: US, Gold -> class 0
    # Group 2: IN, Platinum -> class 1
    x = []
    y = []

    for i in range(25):
        x.append(["US", "Gold", 20.0 + i * 0.5, 100.0])
        y.append(0)

    for i in range(25):
        x.append(["IN", "Platinum", 30.0 + i * 0.5, 5000.0])
        y.append(1)

    model.fit(x, y, col_names=["country", "tier", "age", "spend"], dtypes=["cat", "cat", "float", "float"])

    assert model.classes == [0, 1]

    # Query matching Group 1
    pred_us = model.predict([["US", "Gold", 25.0, 100.0]])[0]
    assert pred_us == 0

    probs_us = model.predict_proba([["US", "Gold", 25.0, 100.0]])[0]
    assert len(probs_us) == 2
    assert probs_us[0] > 0.9

    # Query matching Group 2
    pred_in = model.predict([["IN", "Platinum", 35.0, 5000.0]])[0]
    assert pred_in == 1

    # Hierarchical back-off test:
    # "US", "UnknownTier" -> backs off to "US" prefix node and accurately classifies as 0
    pred_fallback = model.predict([["US", "Silver", 25.0, 100.0]])[0]
    assert pred_fallback == 0, "Should back off to parent prefix node 'US'"


def test_trie_knn_regression():
    model = TrieKNN(task="regression", k=2, weights="uniform")

    x = [
        ["Dept_A", 10.0],
        ["Dept_A", 20.0],
        ["Dept_B", 100.0],
        ["Dept_B", 110.0],
    ]
    y = [1.0, 2.0, 10.0, 11.0]

    model.fit(x, y, col_names=["dept", "val"], dtypes=["cat", "float"])

    pred_a = model.predict([["Dept_A", 15.0]])[0]
    assert abs(pred_a - 1.5) < 1e-4

    pred_b = model.predict([["Dept_B", 105.0]])[0]
    assert abs(pred_b - 10.5) < 1e-4

    # Unseen department backs off to root global average
    pred_unseen = model.predict([["Dept_Unknown", 50.0]])[0]
    assert 1.0 <= pred_unseen <= 11.0
