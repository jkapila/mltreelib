"""Checks on generated API reference pages under docs/reference/api/."""

from __future__ import annotations

import re
from pathlib import Path

from mltreelib import AnomalyModel, ClusterModel

API_DIR = Path(__file__).resolve().parent.parent / "docs" / "reference" / "api"
ALIASES = ("ClusterTree", "AnomalyTree")


def test_generated_api_pages_use_umbrella_names_never_aliases():
    pages = sorted(API_DIR.glob("*.md"))
    assert pages, f"no API markdown pages found under {API_DIR}"
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for alias in ALIASES:
            assert not re.search(rf"\b{alias}\b", text), (
                f"{page.name} must not mention alias {alias}"
            )


def test_generated_cluster_and_anomaly_defaults_match_constructors():
    cluster_cfg = ClusterModel().config()
    cluster_doc = (API_DIR / "cluster.md").read_text(encoding="utf-8")
    assert f"min_rows: 'int' = {cluster_cfg['min_rows']}" in cluster_doc

    anomaly_cfg = AnomalyModel().config()
    anomaly_doc = (API_DIR / "anomaly.md").read_text(encoding="utf-8")
    assert f"max_depth: 'int | None' = {anomaly_cfg['max_depth']}" in anomaly_doc
    assert f"min_rows: 'int' = {anomaly_cfg['min_rows']}" in anomaly_doc
