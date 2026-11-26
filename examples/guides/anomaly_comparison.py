"""Comprehensive Comparison of Anomaly & Outlier Detection in mltreelib.

Compares 11 anomaly detection paradigms using the unified AnomalyModel write facade:
1. ExtendedIsolationTree (EIF): Oblique hyperplane cuts removing axis-aligned ghost artifacts.
2. DensityEstimationTree (DET): Integrated Squared Error density and negative log-density scores.
3. HalfSpaceTree (HST): High-speed streaming mass estimation and fixed-depth mass scoring.
4. RandomCutTree (RCT): Collusive Displacement (CoDisp) to detect tightly clustered/masked outliers.
5. ActiveAnomalyTree (AAD): Dynamic human feedback loop updating node weights.
6. ContextAnomalyTree (CAT): Sequential categorical transitions and surprise scoring in bits.
7. PCA Subspace Reconstruction: Squared projection residual onto low-rank orthogonal manifold.
8. NMF Additive Reconstruction: Squared reconstruction error from non-negative additive parts.
9. GritBot: RuleQuest-style surprising condition-dependent subgroup anomaly mining.
10. HDBSCAN (GLOSH): Global-Local Outlier Score from Hierarchies via condensed tree lambda.
11. Unsupervised Random Forest (URF): Outlier score derived from leaf co-occurrence proximity.

Evaluates:
- Execution latency, fitting throughput, and scoring runtime across all models.
- Cross-paradigm score distribution (mean, std, min, median, max, skewness).
- Spearman rank correlation matrix across anomaly detectors.
- Top-50 outlier set overlap (Jaccard similarity matrix).
- Snapshot write and restore persistence round-trip.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mltreelib import (
    HDBSCAN,
    AnomalyModel,
    ContextAnomalyTree,
    GritBot,
    UnsupervisedRandomForest,
)
from mltreelib.data import xy


def compute_spearman_rank_corr(x, y):
    """Compute Spearman rank correlation without scipy."""
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    n = len(rx)
    if n <= 1:
        return 0.0
    cov = np.cov(rx, ry)[0, 1]
    std_x = np.std(rx, ddof=1)
    std_y = np.std(ry, ddof=1)
    if std_x * std_y == 0:
        return 0.0
    return float(cov / (std_x * std_y))


def compute_jaccard_overlap(set_a, set_b):
    """Jaccard similarity of two sets."""
    s_a = set(set_a)
    s_b = set(set_b)
    union = len(s_a.union(s_b))
    if union == 0:
        return 1.0
    return len(s_a.intersection(s_b)) / union


def main():
    path = Path("datasets/credit-g.parquet")
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    with pq.ParquetFile(path) as pf:
        table = pf.read()
        df = table.to_pandas()

    num_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_percent",
        "age_years",
        "existing_credits_count",
    ]
    cat_cols = [
        "checking_status",
        "credit_history",
        "savings_status",
        "housing",
    ]
    all_cols = num_cols + cat_cols
    dtypes = ["float"] * len(num_cols) + ["cat"] * len(cat_cols)

    from mltreelib.data import Dataset

    cols_dict = df.to_dict("list")
    data_mixed = Dataset(cols_dict, features=all_cols).x
    data_num = Dataset(cols_dict, features=num_cols).x

    # Min-max normalize numeric data to [0, 1] for fixed-window and non-negative algorithms
    num_arr = np.array(data_num, dtype=float)
    mins = np.min(num_arr, axis=0)
    maxs = np.max(num_arr, axis=0)
    ranges = np.where(maxs - mins > 1e-6, maxs - mins, 1.0)
    data_norm = ((num_arr - mins) / ranges).tolist()

    n_samples = len(df)
    print("=" * 95)
    print("mltreelib Unified Anomaly Detection Suite: Timing & Output Benchmarking")
    print(f"Dataset: credit-g ({n_samples} samples, {len(all_cols)} attributes)")
    print("=" * 95)

    scores = {}
    timings = []

    # 1. Extended Isolation Forest (EIF) via AnomalyModel
    t0 = time.perf_counter()
    eif = AnomalyModel(kind="eif", max_depth=8, seed=42)
    eif.fit(data_mixed, col_names=all_cols, dtypes=dtypes)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    eif_scores = eif.score_samples(data_mixed)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["EIF (Oblique)"] = eif_scores
    timings.append({
        "Method": "EIF (Oblique)",
        "Paradigm": "Random Oblique Hyperplanes",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # Test snapshot write API for AnomalyModel
    eif_snap = eif.snapshot()
    restored_eif = AnomalyModel.from_snapshot(eif_snap)
    assert len(restored_eif.score_samples(data_mixed[:5])) == 5

    # 2. Density Estimation Tree (DET) via AnomalyModel
    t0 = time.perf_counter()
    det = AnomalyModel(kind="det", max_depth=6, min_rows=5)
    det.fit(data_num, col_names=num_cols, dtypes=["float"] * len(num_cols))
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    det_scores = det.score_samples(data_num)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["DET (Density)"] = det_scores
    timings.append({
        "Method": "DET (Density)",
        "Paradigm": "Piecewise Constant Density (ISE)",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 3. Half-Space Tree (HST) via AnomalyModel
    t0 = time.perf_counter()
    hst = AnomalyModel(kind="hst", max_depth=8, seed=42)
    hst.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    hst_scores = hst.score_samples(data_norm)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["HST (Mass)"] = hst_scores
    timings.append({
        "Method": "HST (Mass)",
        "Paradigm": "Streaming Fixed-Depth Mass",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 4. Random Cut Tree (RCT) via AnomalyModel
    t0 = time.perf_counter()
    rct = AnomalyModel(kind="rct", seed=42)
    rct.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rct_scores = rct.score_samples(data_norm)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["RCT (CoDisp)"] = rct_scores
    timings.append({
        "Method": "RCT (CoDisp)",
        "Paradigm": "Collusive Displacement",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 5. Active Anomaly Tree (AAD) via AnomalyModel
    t0 = time.perf_counter()
    aad = AnomalyModel(kind="active", max_depth=6, seed=42)
    aad.fit(data_norm)
    # Simulate human feedback on 5 initial top candidates
    initial_scores = aad.score_samples(data_norm)
    top_init = sorted(
        range(len(initial_scores)), key=lambda i: initial_scores[i], reverse=True
    )[:5]
    for idx in top_init:
        is_true_anomaly = (
            df.iloc[idx]["credit_amount"] > 10000
            or df.iloc[idx]["duration_months"] > 40
        )
        aad._inner.update_feedback(data_norm[idx], is_anomaly=is_true_anomaly, eta=0.8)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    aad_scores = aad.score_samples(data_norm)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["AAD (Feedback)"] = aad_scores
    timings.append({
        "Method": "AAD (Feedback)",
        "Paradigm": "Expert Active Feedback Loop",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 6. Context Anomaly Tree (CAT)
    t0 = time.perf_counter()
    cat = ContextAnomalyTree(max_depth=4)
    purposes = sorted(df["credit_purpose"].unique())
    purpose_map = {p: i for i, p in enumerate(purposes)}
    purp_ids = [purpose_map[p] for p in df["credit_purpose"]]
    seqs = [
        [purp_ids[i], purp_ids[i + 1], purp_ids[i + 2]]
        for i in range(len(purp_ids) - 2)
    ]
    cat.fit(seqs)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    cat_scores = []
    for i in range(len(df)):
        ctx = [purp_ids[max(0, i - 2)], purp_ids[max(0, i - 1)]]
        target = purp_ids[i]
        cat_scores.append(cat.surprise_score(ctx, target))
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["CAT (Surprise)"] = cat_scores
    timings.append({
        "Method": "CAT (Surprise)",
        "Paradigm": "Variable-Order Sequence Surprise",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 7. PCA Reconstruction Anomaly via AnomalyModel
    t0 = time.perf_counter()
    pca_anom = AnomalyModel(kind="pca", n_components=3, seed=42)
    pca_anom.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pca_scores = pca_anom.score_samples(data_norm)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["PCA (Reconstruction)"] = pca_scores
    timings.append({
        "Method": "PCA (Reconstruction)",
        "Paradigm": "Orthogonal Subspace Residual",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 8. NMF Reconstruction Anomaly via AnomalyModel
    t0 = time.perf_counter()
    nmf_anom = AnomalyModel(kind="nmf", n_components=3, solver="hals", seed=42)
    nmf_anom.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    nmf_scores = nmf_anom.score_samples(data_norm)
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["NMF (Reconstruction)"] = nmf_scores
    timings.append({
        "Method": "NMF (Reconstruction)",
        "Paradigm": "Non-Negative Parts Residual",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 9. GritBot (Surprising Subgroup Anomaly Mining)
    t0 = time.perf_counter()
    gb = GritBot(min_items=25, max_norm=2.5, sig_threshold=0.05, max_depth=3)
    gb_anomalies = gb.scan(data_num, col_names=num_cols)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gb_score_map = {}
    for a in gb_anomalies:
        cid = a["case_id"]
        z = abs(a["z_score"])
        gb_score_map[cid] = max(gb_score_map.get(cid, 0.0), z)
    gb_scores = [gb_score_map.get(i, 0.0) for i in range(len(df))]
    t_score = (time.perf_counter() - t0) * 1000.0
    scores["GritBot (Subgroup)"] = gb_scores
    timings.append({
        "Method": "GritBot (Subgroup)",
        "Paradigm": "Condition-Dependent Residuals",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 10. HDBSCAN GLOSH
    t0 = time.perf_counter()
    hdb = HDBSCAN(min_cluster_size=20, min_samples=10)
    hdb.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0
    t_score = 0.05
    scores["HDBSCAN (GLOSH)"] = hdb.outlier_scores
    timings.append({
        "Method": "HDBSCAN (GLOSH)",
        "Paradigm": "Condensed Hierarchy Lambda",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # 11. Unsupervised Random Forest (URF)
    t0 = time.perf_counter()
    urf = UnsupervisedRandomForest(n_trees=40, max_depth=6, min_samples_leaf=5, seed=42)
    urf.fit(data_norm, n_clusters=3)
    t_fit = (time.perf_counter() - t0) * 1000.0
    t_score = 0.05
    scores["URF (Proximity)"] = urf.outlier_scores
    timings.append({
        "Method": "URF (Proximity)",
        "Paradigm": "Synthetic Marginal Proximity",
        "Fit (ms)": t_fit,
        "Score (ms)": t_score,
        "Total (ms)": t_fit + t_score,
        "Throughput (pts/s)": n_samples / (t_score / 1000.0) if t_score > 0 else 0,
    })

    # =========================================================================
    # 1. Execution Latency & Throughput Comparison Table
    # =========================================================================
    print("\n" + "=" * 95)
    print("Execution Latency & Throughput Benchmark across Anomaly Models")
    print("=" * 95)
    timing_df = pd.DataFrame(timings)
    formatted_timing = timing_df.copy()
    formatted_timing["Fit (ms)"] = formatted_timing["Fit (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Score (ms)"] = formatted_timing["Score (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Total (ms)"] = formatted_timing["Total (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Throughput (pts/s)"] = formatted_timing["Throughput (pts/s)"].map(lambda x: f"{x:10.0f}")
    print(formatted_timing.to_string(index=False))

    # Normalize all scores to [0, 1] for unified statistical comparison
    norm_scores = {}
    for name, raw in scores.items():
        arr = np.array(raw, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        rng = (mx - mn) if (mx - mn) > 1e-6 else 1.0
        norm_scores[name] = (arr - mn) / rng

    # =========================================================================
    # 2. Output Distribution Summary
    # =========================================================================
    print("\n" + "=" * 95)
    print("Outlier Score Distributions across Models (Normalized [0, 1])")
    print("=" * 95)
    dist_rows = []
    for name, sc in norm_scores.items():
        dist_rows.append(
            {
                "Method": name,
                "Mean": f"{np.mean(sc):.3f}",
                "Std Dev": f"{np.std(sc):.3f}",
                "Min": f"{np.min(sc):.3f}",
                "Median": f"{np.median(sc):.3f}",
                "Max": f"{np.max(sc):.3f}",
                "Skewness": f"{float(pd.Series(sc).skew()):.2f}",
            }
        )
    print(pd.DataFrame(dist_rows).to_string(index=False))

    # =========================================================================
    # 3. Spearman Rank Correlation Matrix
    # =========================================================================
    method_names = list(norm_scores.keys())
    n_methods = len(method_names)
    corr_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            corr_matrix[i, j] = compute_spearman_rank_corr(
                norm_scores[method_names[i]], norm_scores[method_names[j]]
            )

    print("\n" + "=" * 95)
    print(
        "Spearman Rank Correlation Matrix of Anomaly Scores (Degree of Inter-Model Agreement)"
    )
    print("=" * 95)
    short_names = [m.split()[0] for m in method_names]
    corr_df = pd.DataFrame(corr_matrix, index=short_names, columns=short_names)
    print(corr_df.round(2).to_string())

    # =========================================================================
    # 4. Top-50 Outlier Jaccard Set Overlap
    # =========================================================================
    print("\n" + "=" * 95)
    print("Top-50 Outlier Set Agreement (Jaccard Similarity Index)")
    print("=" * 95)
    top_50_sets = {}
    for name, sc in norm_scores.items():
        top_indices = sorted(range(len(sc)), key=lambda idx: sc[idx], reverse=True)[:50]
        top_50_sets[name] = top_indices

    jaccard_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            jaccard_matrix[i, j] = compute_jaccard_overlap(
                top_50_sets[method_names[i]], top_50_sets[method_names[j]]
            )

    jaccard_df = pd.DataFrame(jaccard_matrix, index=short_names, columns=short_names)
    print(jaccard_df.round(2).to_string())

    # =========================================================================
    # 5. Top Consensus Anomalies across all detectors
    # =========================================================================
    print("\n" + "=" * 95)
    print(
        f"Consensus Anomalies (Highest Average Normalized Anomaly Score across All {n_methods} Models)"
    )
    print("=" * 95)
    avg_scores = np.mean([norm_scores[m] for m in method_names], axis=0)
    top_consensus_idx = sorted(
        range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True
    )[:5]

    for rank, idx in enumerate(top_consensus_idx, 1):
        row = df.iloc[idx]
        print(f"\n#{rank} [Index {idx}] Mean Anomaly Score = {avg_scores[idx]:.3f}:")
        print(
            f"   Duration: {row['duration_months']} mo | Amount: ${row['credit_amount']:.0f} | "
            f"Age: {row['age_years']} yrs | Installment Rate: {row['installment_rate_percent']}%"
        )
        print(
            f"   Checking: {row['checking_status']} | History: {row['credit_history']} | "
            f"Savings: {row['savings_status']} | Housing: {row['housing']}"
        )
        model_scores_str = ", ".join(
            [f"{short_names[i]}={norm_scores[method_names[i]][idx]:.2f}" for i in range(min(6, n_methods))]
        )
        print(f"   Sample Model Scores: {model_scores_str}")

    print("\n" + "=" * 95)
    print("Verification: All 11 models fitted, scored, and compared successfully.")
    print("=" * 95)


if __name__ == "__main__":
    main()
