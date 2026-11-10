import pytest
from mltreelib import GritBot

def test_gritbot_python_basic_scan():
    # Build dataset with pregnant vs age contextual anomaly
    # 50 rows pregnant=1, age ~ 28
    # 50 rows pregnant=0, age ~ 65
    # Row 0: pregnant=1, age=80 (contextual anomaly)
    data = [[1.0, 80.0]]
    for i in range(50):
        data.append([1.0, 26.0 + float(i % 5)])
    for i in range(50):
        data.append([0.0, 60.0 + float(i % 12)])

    bot = GritBot(min_items=20, max_norm=3.0, sig_threshold=0.05)
    anomalies = bot.scan(data, col_names=["pregnant", "age"])

    assert len(anomalies) > 0
    case0 = next((a for a in anomalies if a["case_id"] == 0 and a["attribute_name"] == "age"), None)
    assert case0 is not None, "Case 0 must be flagged as anomalous on age"
    assert case0["actual_value"] == 80.0
    assert case0["z_score"] >= 3.0
    assert len(case0["conditions"]) > 0
    assert case0["conditions"][0]["col_name"] == "pregnant"
    assert "where pregnant" in case0["summary"]

def test_gritbot_python_global_outlier():
    data = []
    for i in range(50):
        data.append([float(i), 10.0 + float(i % 3)])
    data.append([25.0, 500.0])  # row 50: global anomaly

    bot = GritBot(min_items=20, max_norm=3.0, sig_threshold=0.05)
    anomalies = bot.scan(data, col_names=["x", "y"])

    found = next((a for a in anomalies if a["case_id"] == 50 and a["attribute_name"] == "y"), None)
    assert found is not None
    assert found["actual_value"] == 500.0
    assert len(found["conditions"]) == 0
