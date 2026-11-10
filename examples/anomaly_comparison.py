"""Comprehensive Comparison of Tree-Based Anomaly & Outlier Detection in mltreelib.

Compares:
1. ExtendedIsolationTree (EIF): Oblique hyperplane cuts removing axis-aligned ghost artifacts.
2. DensityEstimationTree (DET): Integrated Squared Error density and negative log-density scores.
3. HalfSpaceTree (HST): High-speed streaming mass estimation and fixed-depth mass scoring.
4. RandomCutTree (RCT): Collusive Displacement (CoDisp) to detect tightly clustered/masked outliers.
5. ActiveAnomalyTree (AAD): Dynamic human feedback loop updating node weights.
6. ContextAnomalyTree (CAT): Sequential categorical transitions and surprise scoring in bits.
7. GritBot: RuleQuest-style surprising condition-dependent subgroup anomaly mining.
8. HDBSCAN (GLOSH): Global-Local Outlier Score from Hierarchies via condensed tree lambda.
9. Unsupervised Random Forest (URF): Outlier score derived from leaf co-occurrence proximity.

Evaluates:
- Cross-paradigm score distribution (mean, std, min, max).
- Spearman rank correlation matrix across anomaly detectors.
- Top-50 outlier set overlap (Jaccard similarity matrix).
- Method comparison guide: when to use which tree-based anomaly paradigm.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mltreelib import (
    ActiveAnomalyTree,
    ContextAnomalyTree,
    DensityEstimationTree,
    ExtendedIsolationTree,
    GritBot,
    HDBSCAN,
    HalfSpaceTree,
    RandomCutTree,
    UnsupervisedRandomForest,
)


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

    data_mixed = df[all_cols].values.tolist()
    data_num = df[num_cols].values.tolist()

    # Min-max normalize numeric data for fixed-window algorithms (HST, DET, etc.)
    num_arr = np.array(data_num, dtype=float)
    mins = np.min(num_arr, axis=0)
    maxs = np.max(num_arr, axis=0)
    ranges = np.where(maxs - mins > 1e-6, maxs - mins, 1.0)
    data_norm = ((num_arr - mins) / ranges).tolist()

    print("=" * 90)
    print("mltreelib Tree-Based Anomaly Detection Suite Comparison & Benchmarking")
    print(f"Dataset: credit-g ({len(df)} samples, {len(all_cols)} attributes)")
    print("=" * 90)

    scores = {}

    # 1. Extended Isolation Forest (EIF)
    print("\n[1/9] Fitting Extended Isolation Tree (Oblique Hyperplane Cuts)...")
    eif = ExtendedIsolationTree(max_depth=8, seed=42)
    eif.fit(data_mixed, col_names=all_cols, dtypes=dtypes)
    scores["EIF (Oblique)"] = eif.anomaly_score(data_mixed)

    # 2. Density Estimation Tree (DET)
    print("[2/9] Fitting Density Estimation Tree (ISE Density Optimization)...")
    det = DensityEstimationTree(max_depth=6, min_samples_split=10, min_leaf_samples=5)
    det.fit(data_num, col_names=num_cols, dtypes=["float"] * len(num_cols))
    scores["DET (Density)"] = det.predict_anomaly_score(data_num)

    # 3. Half-Space Tree (HST)
    print("[3/9] Fitting Half-Space Tree (Streaming Mass Estimation)...")
    hst = HalfSpaceTree(max_depth=8, seed=42)
    hst.fit(data_norm)
    scores["HST (Mass)"] = hst.anomaly_score(data_norm)

    # 4. Random Cut Tree (RCT)
    print("[4/9] Fitting Random Cut Tree (Collusive Displacement - CoDisp)...")
    rct = RandomCutTree(seed=42)
    rct.fit(data_norm)
    scores["RCT (CoDisp)"] = rct.codisp(data_norm)

    # 5. Active Anomaly Tree (AAD)
    print("[5/9] Fitting Active Anomaly Tree (Human-Feedback Adaptation)...")
    aad = ActiveAnomalyTree(max_depth=6, seed=42)
    aad.fit(data_norm)
    # Simulate human feedback on 5 initial top candidates
    initial_scores = aad.anomaly_score(data_norm)
    top_init = sorted(range(len(initial_scores)), key=lambda i: initial_scores[i], reverse=True)[:5]
    for idx in top_init:
        # Mark as confirmed anomaly if credit_amount > 10000 or duration > 40
        is_true_anomaly = df.iloc[idx]["credit_amount"] > 10000 or df.iloc[idx]["duration_months"] > 40
        aad.update_feedback(data_norm[idx], is_anomaly=is_true_anomaly, eta=0.8)
    scores["AAD (Feedback)"] = aad.anomaly_score(data_norm)

    # 6. Context Anomaly Tree (CAT)
    print("[6/9] Fitting Context Anomaly Tree (Categorical Sequence Transitions)...")
    cat = ContextAnomalyTree(max_depth=4)
    purposes = sorted(df["credit_purpose"].unique())
    purpose_map = {p: i for i, p in enumerate(purposes)}
    purp_ids = [purpose_map[p] for p in df["credit_purpose"]]
    seqs = [[purp_ids[i], purp_ids[i + 1], purp_ids[i + 2]] for i in range(len(purp_ids) - 2)]
    cat.fit(seqs)
    cat_scores = []
    for i in range(len(df)):
        ctx = [purp_ids[max(0, i - 2)], purp_ids[max(0, i - 1)]]
        target = purp_ids[i]
        cat_scores.append(cat.surprise_score(ctx, target))
    scores["CAT (Surprise)"] = cat_scores

    # 7. GritBot (Surprising Subgroup Anomaly Mining)
    print("[7/9] Mining Subgroup Outliers with GritBot...")
    gb = GritBot(min_items=25, max_norm=2.5, sig_threshold=0.05, max_depth=3)
    gb_anomalies = gb.scan(data_num, col_names=num_cols)
    gb_score_map = {}
    for a in gb_anomalies:
        cid = a["case_id"]
        z = abs(a["z_score"])
        gb_score_map[cid] = max(gb_score_map.get(cid, 0.0), z)
    scores["GritBot (Subgroup)"] = [gb_score_map.get(i, 0.0) for i in range(len(df))]

    # 8. HDBSCAN GLOSH (Global-Local Outlier Score from Hierarchies)
    print("[8/9] Computing HDBSCAN GLOSH Outlier Scores...")
    hdb = HDBSCAN(min_cluster_size=20, min_samples=10)
    hdb.fit(data_norm)
    scores["HDBSCAN (GLOSH)"] = hdb.outlier_scores

    # 9. Unsupervised Random Forest (Proximity Outlier Scoring)
    print("[9/9] Computing Unsupervised Random Forest Proximity Outlier Scores...")
    urf = UnsupervisedRandomForest(n_trees=40, max_depth=6, min_samples_leaf=5, seed=42)
    urf.fit(data_norm, n_clusters=3)
    scores["URF (Proximity)"] = urf.outlier_scores

    # Normalize all scores to [0, 1] for unified statistical comparison
    norm_scores = {}
    for name, raw in scores.items():
        arr = np.array(raw, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        rng = (mx - mn) if (mx - mn) > 1e-6 else 1.0
        norm_scores[name] = (arr - mn) / rng

    # 1. Distribution summary
    print("\n" + "=" * 90)
    print("Outlier Score Distributions across Models (Normalized [0, 1])")
    print("=" * 90)
    dist_rows = []
    for name, sc in norm_scores.items():
        dist_rows.append({
            "Method": name,
            "Mean": f"{np.mean(sc):.3f}",
            "Std Dev": f"{np.std(sc):.3f}",
            "Min": f"{np.min(sc):.3f}",
            "Median": f"{np.median(sc):.3f}",
            "Max": f"{np.max(sc):.3f}",
            "Skewness": f"{float(pd.Series(sc).skew()):.2f}",
        })
    print(pd.DataFrame(dist_rows).to_string(index=False))

    # 2. Spearman Rank Correlation Matrix
    method_names = list(norm_scores.keys())
    n_methods = len(method_names)
    corr_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            corr_matrix[i, j] = compute_spearman_rank_corr(
                norm_scores[method_names[i]], norm_scores[method_names[j]]
            )

    print("\n" + "=" * 90)
    print("Spearman Rank Correlation Matrix of Anomaly Scores (Degree of Inter-Model Agreement)")
    print("=" * 90)
    corr_df = pd.DataFrame(corr_matrix, index=method_names, columns=method_names)
    print(corr_df.round(2).to_string())

    # 3. Top-50 Outlier Jaccard Set Overlap
    print("\n" + "=" * 90)
    print("Top-50 Outlier Set Agreement (Jaccard Similarity Index)")
    print("=" * 90)
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

    jaccard_df = pd.DataFrame(jaccard_matrix, index=method_names, columns=method_names)
    print(jaccard_df.round(2).to_string())

    # 4. Top Consensus Anomalies across all 9 detectors
    print("\n" + "=" * 90)
    print("Consensus Anomalies (Highest Average Normalized Anomaly Score across All 9 Models)")
    print("=" * 90)
    avg_scores = np.mean([norm_scores[m] for m in method_names], axis=0)
    top_consensus_idx = sorted(range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True)[:5]

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
        model_scores_str = ", ".join([f"{m.split()[0]}={norm_scores[m][idx]:.2f}" for m in method_names[:5]])
        print(f"   Individual Scores: {model_scores_str}")

    # Architectural Comparison Summary
    print("\n" + "=" * 90)
    print("Architectural Comparison: When to Use Which Anomaly Paradigm")
    print("=" * 90)
    print("""
1. Extended Isolation Tree (EIF):
   - Best for: Tabular mixed data with oblique correlations where standard axis-aligned
     isolation trees fail due to ghost masking.

2. Density Estimation Tree (DET):
   - Best for: Fast, exact piecewise constant density estimation with theoretical ISE optimality;
     outliers scored by true log-density.

3. Half-Space Tree (HST):
   - Best for: Ultra high-throughput streaming environments where point arrival speed demands
     O(depth) updates with constant memory.

4. Random Cut Tree (RCT):
   - Best for: Streaming and batch workloads vulnerable to "collusion" (dense clusters of
     anomalies that mask one another from standard isolation).

5. Active Anomaly Tree (AAD):
   - Best for: Human-in-the-loop audit workflows where an analyst periodically confirms or
     rejects flagged outliers to eliminate repeated false positives.

6. Context Anomaly Tree (CAT):
   - Best for: Sequential or session-based events where an attribute value is not rare overall,
     but highly anomalous given the preceding sequence.

7. GritBot:
   - Best for: Contextual and conditional outliers that violate local rules (e.g. high credit
     amount for a young student with no checking balance).

8. HDBSCAN (GLOSH):
   - Best for: Outliers relative to local cluster density hierarchies; distinguishes global
     outliers from micro-clusters that shed at low lambda.

9. Unsupervised Random Forest (URF):
   - Best for: Complex multi-dimensional manifolds where proximity co-occurrence across
     scrambled marginal trees isolates non-conforming data points.
""")


if __name__ == "__main__":
    main()
