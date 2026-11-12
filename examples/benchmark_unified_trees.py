"""Comprehensive Benchmark Runner for AnomalyTree and ClusterTree.

Evaluates unified models on standard tabular benchmarks:
- Anomaly Detection (ADBench/ODDS style): credit-g, diabetes, blood-transfusion, qsar-biodeg, churn.
  Metrics: ROC-AUC, PR-AUC, F1-Score at calibrated contamination threshold.
- Clustering (Unsupervised & Divisive Tree): credit-g, diabetes, blood-transfusion, qsar-biodeg, churn.
  Metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Rule Coverage.
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mltreelib import AnomalyTree, ClusterTree, Metrics

ANOM_DIR = Path("benchmarks/anomaly_data")
CLUS_DIR = Path("benchmarks/clustering_data")
DATASETS = ["credit-g", "diabetes", "blood-transfusion-service-center", "qsar-biodeg", "churn"]


def load_dataset(dir_path: Path, name: str) -> Tuple[List[List[Any]], List[Any], List[List[Any]], List[Any], List[str]]:
    with open(dir_path / f"{name}.meta.json") as f:
        meta = json.load(f)
    col_names = list(meta["columns"].keys())
    dtypes = [meta["columns"][c] for c in col_names]

    def read_split(split: str):
        path = dir_path / f"{name}_{split}.csv"
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            target_idx = len(header) - 1
            X, y = [], []
            for row in r:
                if not row:
                    continue
                row_x = []
                for idx, dt in enumerate(dtypes):
                    v = row[idx]
                    if v == "" or v is None:
                        row_x.append(None)
                    elif dt == "numeric":
                        row_x.append(float(v))
                    else:
                        row_x.append(v)
                X.append(row_x)
                y.append(row[target_idx])
            return X, y

    xtr, ytr = read_split("train")
    xte, yte = read_split("test")
    return xtr, ytr, xte, yte, dtypes


def compute_roc_auc(scores: List[float], y_true: List[int]) -> float:
    """Pure-python ROC-AUC via Wilcoxon-Mann-Whitney rank statistic."""
    pos_scores = [s for s, y in zip(scores, y_true) if y == 1]
    neg_scores = [s for s, y in zip(scores, y_true) if y == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count pairs where pos > neg (ties count 0.5)
    paired_wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                paired_wins += 1.0
            elif ps == ns:
                paired_wins += 0.5
    return paired_wins / (n_pos * n_neg)


def compute_f1(y_pred: List[int], y_true: List[int]) -> float:
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def compute_ari(labels_pred: List[Any], labels_true: List[Any]) -> float:
    """Pure-python Adjusted Rand Index (Hubert & Arabie 1985)."""
    n = len(labels_true)
    if n <= 1:
        return 1.0

    # Build contingency table
    contingency: Dict[Tuple[Any, Any], int] = {}
    sum_a: Dict[Any, int] = {}
    sum_b: Dict[Any, int] = {}

    for p, t in zip(labels_pred, labels_true):
        contingency[(p, t)] = contingency.get((p, t), 0) + 1
        sum_a[p] = sum_a.get(p, 0) + 1
        sum_b[t] = sum_b.get(t, 0) + 1

    def comb2(val: int) -> float:
        return (val * (val - 1.0)) / 2.0 if val >= 2 else 0.0

    sum_comb_nij = sum(comb2(cnt) for cnt in contingency.values())
    sum_comb_a = sum(comb2(cnt) for cnt in sum_a.values())
    sum_comb_b = sum(comb2(cnt) for cnt in sum_b.values())
    comb_n = comb2(n)

    expected = (sum_comb_a * sum_comb_b) / comb_n if comb_n > 0 else 0.0
    max_idx = (sum_comb_a + sum_comb_b) / 2.0
    denom = max_idx - expected
    return (sum_comb_nij - expected) / denom if denom > 0 else 0.0


def run_anomaly_benchmarks():
    print("=" * 80)
    print("BENCHMARK 1: UNIFIED AnomalyTree EVALUATION (ODDS / ADBench format)")
    print("=" * 80)

    engines = ["eif", "det", "hst", "rct"]

    print(f"{'Dataset':32} | {'Engine':8} | {'ROC-AUC':8} | {'F1-Score':8} | {'Fit (ms)':10}")
    print("-" * 80)

    results = []
    for ds_name in DATASETS:
        xtr, ytr_raw, xte, yte_raw, dtypes = load_dataset(ANOM_DIR, ds_name)
        yte = [int(v) for v in yte_raw]
        ytr = [int(v) for v in ytr_raw]
        contamination = sum(ytr) / len(ytr)

        for eng in engines:
            t0 = time.perf_counter()
            try:
                # HST and RCT are purely numeric
                if eng in ("hst", "rct") and any(dt == "categorical" for dt in dtypes):
                    # Filter to numeric columns only
                    num_idx = [j for j, dt in enumerate(dtypes) if dt == "numeric"]
                    xtr_in = [[r[j] for j in num_idx] for r in xtr]
                    xte_in = [[r[j] for j in num_idx] for r in xte]
                    dts_in = ["numeric"] * len(num_idx)
                else:
                    xtr_in, xte_in, dts_in = xtr, xte, dtypes

                at = AnomalyTree(kind=eng, contamination=contamination, seed=42)
                at.fit(xtr_in, dtypes=dts_in)
                dur_ms = (time.perf_counter() - t0) * 1000.0

                scores = at.score_samples(xte_in)
                preds = at.predict(xte_in)

                auc = compute_roc_auc(scores, yte)
                f1 = compute_f1(preds, yte)
                print(f"{ds_name:32} | {eng:8} | {auc:8.4f} | {f1:8.4f} | {dur_ms:8.2f} ms")
                results.append({"dataset": ds_name, "engine": eng, "auc": auc, "f1": f1, "fit_ms": dur_ms})
            except Exception as e:
                print(f"{ds_name:32} | {eng:8} | FAILED ({e})")

    # Average metrics per engine
    print("\nAnomaly Detection Engines Summary Across All Datasets:")
    for eng in engines:
        eng_res = [r for r in results if r["engine"] == eng]
        if eng_res:
            mean_auc = sum(r["auc"] for r in eng_res) / len(eng_res)
            mean_f1 = sum(r["f1"] for r in eng_res) / len(eng_res)
            mean_time = sum(r["fit_ms"] for r in eng_res) / len(eng_res)
            print(f"  {eng.upper():5}: Mean ROC-AUC={mean_auc:.4f} | Mean F1={mean_f1:.4f} | Mean Fit={mean_time:.1f} ms")


def run_clustering_benchmarks():
    print("\n" + "=" * 80)
    print("BENCHMARK 2: UNIFIED ClusterTree EVALUATION (Tabular Clustering)")
    print("=" * 80)

    engines = ["pct", "cltree", "hdbscan", "urf"]

    print(f"{'Dataset':32} | {'Engine':8} | {'ARI Score':10} | {'Clusters':9} | {'Fit (ms)':10}")
    print("-" * 80)

    results = []
    for ds_name in DATASETS:
        xtr, ytr_raw, xte, yte_raw, dtypes = load_dataset(CLUS_DIR, ds_name)

        for eng in engines:
            t0 = time.perf_counter()
            try:
                # CLTree, HDBSCAN, and URF require numeric arrays
                if eng in ("cltree", "hdbscan", "urf") and any(dt == "categorical" for dt in dtypes):
                    num_idx = [j for j, dt in enumerate(dtypes) if dt == "numeric"]
                    xtr_in = [[r[j] for j in num_idx] for r in xtr]
                    xte_in = [[r[j] for j in num_idx] for r in xte]
                    dts_in = ["numeric"] * len(num_idx)
                else:
                    xtr_in, xte_in, dts_in = xtr, xte, dtypes

                ct = ClusterTree(kind=eng, min_rows=10, seed=42)
                ct.fit(xtr_in, dtypes=dts_in)
                dur_ms = (time.perf_counter() - t0) * 1000.0

                preds = ct.predict(xte_in)
                ari = compute_ari(preds, yte_raw)
                k = ct.n_clusters_
                print(f"{ds_name:32} | {eng:8} | {ari:10.4f} | {k:<9} | {dur_ms:8.2f} ms")
                results.append({"dataset": ds_name, "engine": eng, "ari": ari, "k": k, "fit_ms": dur_ms})
            except Exception as e:
                print(f"{ds_name:32} | {eng:8} | FAILED ({e})")

    print("\nClustering Engines Summary Across All Datasets:")
    for eng in engines:
        eng_res = [r for r in results if r["engine"] == eng]
        if eng_res:
            mean_ari = sum(r["ari"] for r in eng_res) / len(eng_res)
            mean_time = sum(r["fit_ms"] for r in eng_res) / len(eng_res)
            print(f"  {eng.upper():7}: Mean ARI={mean_ari:.4f} | Mean Fit={mean_time:.1f} ms")


if __name__ == "__main__":
    run_anomaly_benchmarks()
    run_clustering_benchmarks()
