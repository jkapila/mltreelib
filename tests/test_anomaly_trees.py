import pytest
from mltreelib import DensityEstimationTree, ExtendedIsolationTree

def test_det_continuous_density_and_anomaly_score():
    # 100 points clustered around (1.0, 1.0)
    data = []
    for i in range(100):
        data.append([1.0 + 0.1 * (i % 7), 1.0 + 0.1 * (i % 5)])

    # Plant outlier at (10.0, 10.0)
    data.append([10.0, 10.0])

    det = DensityEstimationTree(max_depth=5, min_samples_split=5, min_leaf_samples=2)
    det.fit(data, col_names=["x", "y"])

    densities = det.predict_density([[1.0, 1.0], [10.0, 10.0]])
    scores = det.predict_anomaly_score([[1.0, 1.0], [10.0, 10.0]])

    assert len(densities) == 2
    assert densities[0] > densities[1], "Cluster center must have higher density than outlier"
    assert scores[1] > scores[0], "Outlier must have higher negative log-density anomaly score"

def test_det_mixed_categorical():
    # Data with categorical column
    data = []
    for i in range(50):
        cat = "A" if i % 2 == 0 else "B"
        data.append([float(i % 10), cat])
    # Plant rare categorical category "Z" at extreme continuous value
    data.append([50.0, "Z"])

    det = DensityEstimationTree(max_depth=4, min_samples_split=4, min_leaf_samples=1)
    det.fit(data, col_names=["val", "group"], dtypes=["float", "cat"])

    scores = det.predict_anomaly_score([[5.0, "A"], [50.0, "Z"]])
    assert scores[1] > scores[0], "Rare category 'Z' at extreme value must have higher anomaly score"

def test_eif_tree_oblique_isolation_and_categorical():
    data = []
    # 80 normal points
    for i in range(80):
        cat = "group1" if i % 2 == 0 else "group2"
        data.append([2.0 + 0.1 * (i % 6), 3.0 + 0.1 * (i % 8), cat])

    # Outlier point far away
    data.append([25.0, 35.0, "rare_group"])

    eif = ExtendedIsolationTree(max_depth=8, extension_level=2, seed=123)
    eif.fit(data, col_names=["x1", "x2", "cat"], dtypes=["float", "float", "cat"])

    normal_len = eif.path_length([[2.0, 3.0, "group1"]])[0]
    outlier_len = eif.path_length([[25.0, 35.0, "rare_group"]])[0]

    normal_score = eif.anomaly_score([[2.0, 3.0, "group1"]])[0]
    outlier_score = eif.anomaly_score([[25.0, 35.0, "rare_group"]])[0]

    assert outlier_len < normal_len, "Outlier must be isolated earlier (shorter path length)"
    assert outlier_score > normal_score, "Outlier must have higher anomaly score"
