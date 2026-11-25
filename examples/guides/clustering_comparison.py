"""Comprehensive Comparison of Tree-Based Clustering Methods in mltreelib.

Compares:
1. Predictive Clustering Trees (PCT / CLUS): Multi-target variance reduction.
2. CLTree (Liu et al. KDD 2000): Hyper-rectangular partition against analytical noise.
3. HDBSCAN Condensed Tree: Mutual reachability MST + Excess of Mass stability.
4. Unsupervised Random Forest (URF): Breiman proximity kernel + Lance-Williams agglomeration.

Evaluates and compares all methods across:
- Execution latency, fitting throughput, and prediction runtime.
- Jitin Kapila's Crosstab Sparsity (CSP) metric (Segregation * Explanation).
- Cluster size distribution and noise handling.
- Extracted decision rules and bounding boxes.
- Snapshot write and restore persistence round-trip.
"""

from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mltreelib import ClusterModel


def crosstab_sparsity(
    df, clusters, numeric_bin="decile", method="fd", exhaustive=False
):
    """Crosstab Sparsity (CSP) metric calculation based on Jitin Kapila's paper & gist:

    https://gist.github.com/jkapila/83bb8f6461ec91bfced437762f2c9220

    :param df: DataFrame containing features to evaluate.
    :param clusters: Cluster assignment per instance (array-like, length N).
    :param numeric_bin: Binning strategy for numeric features ('decile', 'decile_20', 'hist_m', 'hist_n').
    :param method: Histogram binning method for 'hist_m' ('fd', 'rice', 'scott').
    :param exhaustive: Include individual contingency tables in result.
    """
    if isinstance(df, pd.DataFrame):
        col_names = df.columns
        df_flag = True
    else:
        col_names = np.arange(1, df.shape[1] + 1)
        df_flag = False

    if df.shape[0] != len(clusters):
        raise ValueError(
            "Clusters and number of observations are different. Cannot Proceed!"
        )

    # Filter out noise/unassigned instances (-1 or None) if present for valid contingency
    valid_mask = [c is not None and c != -1 for c in clusters]
    sub_df = df.iloc[valid_mask] if df_flag else df[valid_mask]
    sub_clusters = [clusters[i] for i in range(len(clusters)) if valid_mask[i]]

    n_d = len(sub_clusters)
    k = len(np.unique(sub_clusters))
    if k <= 1 or n_d == 0:
        return {
            "score": 0.0,
            "n_clusters": k,
            "n_d": n_d,
            "scores": pd.DataFrame({"Columns": col_names, "metric": 0.0}),
        }

    score_df = pd.DataFrame(
        {
            "Columns": col_names,
            "l": -1,
            "n_v": -1,
            "med_n_v": -1.0,
            "min_n_v": -1.0,
            "avg_n_v": -1.0,
            "max_n_v": -1.0,
            "k": k,
            "n_d": n_d,
            "seggregation": -1.0,
            "explaination": -1.0,
            "inv_seggregation": -1.0,
            "inv_explaination": -1.0,
        }
    )

    cstables = {} if exhaustive else None

    for col_ in col_names:
        if df_flag:
            col_val = sub_df[col_]
            if pd.api.types.is_numeric_dtype(col_val):
                clean_vals = col_val.dropna()
                if len(clean_vals) == 0:
                    continue
                if numeric_bin == "decile":
                    bins = np.unique(np.percentile(clean_vals, np.arange(0, 101, 10)))
                elif numeric_bin == "decile_20":
                    bins = np.unique(np.percentile(clean_vals, np.arange(0, 101, 5)))
                elif numeric_bin == "hist_m":
                    bins = np.histogram_bin_edges(clean_vals, bins=method)
                else:
                    bins = np.histogram_bin_edges(clean_vals, bins=10)

                if len(bins) <= 1:
                    bins = 2
                col_val = pd.cut(col_val, bins=bins, include_lowest=True)
        else:
            col_val = sub_df[:, col_ - 1]

        # Contingency table
        cstable = pd.crosstab(col_val, sub_clusters)
        vals = cstable.values
        med_ = float(np.median(vals.flatten()))
        l = cstable.shape[0]
        n_v = int(np.sum(vals > med_))

        seg = n_v / max(l, k) if max(l, k) > 0 else 0.0
        inv_seg = max(l, k) / n_v if n_v > 0 else 0.0
        exp = np.log(n_d / (l * k)) if (l * k) > 0 and n_d > 0 else 0.0
        inv_exp = np.log((l * k) / n_d) if n_d > 0 and (l * k) > 0 else 0.0

        score_df.loc[score_df["Columns"] == col_, "avg_n_v"] = float(np.mean(vals))
        score_df.loc[score_df["Columns"] == col_, "min_n_v"] = float(np.min(vals))
        score_df.loc[score_df["Columns"] == col_, "max_n_v"] = float(np.max(vals))
        score_df.loc[score_df["Columns"] == col_, "med_n_v"] = med_
        score_df.loc[score_df["Columns"] == col_, "l"] = l
        score_df.loc[score_df["Columns"] == col_, "n_v"] = n_v
        score_df.loc[score_df["Columns"] == col_, "seggregation"] = seg
        score_df.loc[score_df["Columns"] == col_, "inv_seggregation"] = inv_seg
        score_df.loc[score_df["Columns"] == col_, "explaination"] = exp
        score_df.loc[score_df["Columns"] == col_, "inv_explaination"] = inv_exp

        if exhaustive:
            cstables[col_] = cstable

    score_df["metric"] = score_df["seggregation"] * score_df["explaination"]
    score_df = score_df.sort_values(by="metric", ascending=False).reset_index(drop=True)

    res = {
        "score": float(score_df["metric"].sum()),
        "n_clusters": k,
        "n_d": n_d,
        "scores": score_df,
    }
    if exhaustive:
        res["cstables"] = cstables
    return res


def main():
    path = Path("datasets/credit-g.parquet")
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    with pq.ParquetFile(path) as pf:
        table = pf.read()
        df = table.to_pandas()

    eval_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_percent",
        "age_years",
        "existing_credits_count",
        "checking_status",
        "credit_history",
        "savings_status",
        "housing",
    ]
    numeric_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_percent",
        "age_years",
    ]
    data_num = df[numeric_cols].values.tolist()

    # Normalize numeric features for distance algorithms
    means = [np.mean(df[col]) for col in numeric_cols]
    stds = [np.std(df[col]) + 1e-6 for col in numeric_cols]
    data_norm = [
        [(row[j] - means[j]) / stds[j] for j in range(len(numeric_cols))]
        for row in data_num
    ]

    n_samples = len(df)
    print("=" * 95)
    print("mltreelib Unified ClusterModel Suite: Timing & Output Benchmarking")
    print(
        f"Dataset: credit-g ({n_samples} samples, {len(eval_cols)} evaluation features)"
    )
    print("=" * 95)

    results = {}
    timings = []

    # 1. Predictive Clustering Tree (PCT / CLUS) via ClusterModel
    print("\n[1/4] Fitting Predictive Clustering Tree (PCT / CLUS)...")
    t0 = time.perf_counter()
    pct = ClusterModel(kind="pct", max_depth=3, min_rows=15)
    pct.fit(data_num, col_names=numeric_cols, dtypes=["float"] * len(numeric_cols))
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pct_labels = pct.predict(data_num)
    t_pred = (time.perf_counter() - t0) * 1000.0

    results["PCT (CLUS)"] = {
        "model": pct,
        "labels": pct_labels,
        "paradigm": "Divisive Multi-Target Variance Reduction",
        "interpretability": "Extracted decision paths with centroid prototypes",
    }
    timings.append({
        "Method": "PCT (CLUS)",
        "Paradigm": "Multi-Target Variance Reduction",
        "Fit (ms)": t_fit,
        "Predict (ms)": t_pred,
        "Total (ms)": t_fit + t_pred,
        "Throughput (pts/s)": n_samples / (t_pred / 1000.0) if t_pred > 0 else 0,
    })

    # Test snapshot write API for ClusterModel
    pct_snap = pct.snapshot()
    restored_pct = ClusterModel.from_snapshot(pct_snap)
    assert len(restored_pct.predict(data_num[:5])) == 5

    # 2. CLTree (Liu et al. KDD 2000) via ClusterModel
    print("[2/4] Fitting CLTree (Decision Tree Clustering via Analytical Noise)...")
    t0 = time.perf_counter()
    cltree = ClusterModel(kind="cltree", max_depth=4, min_rows=15, min_density=0.60)
    cltree.fit(data_num, col_names=numeric_cols)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    clt_labels = cltree.predict(data_num)
    t_pred = (time.perf_counter() - t0) * 1000.0

    results["CLTree"] = {
        "model": cltree,
        "labels": clt_labels,
        "paradigm": "Decision Tree against Virtual Uniform Background",
        "interpretability": "Hyper-rectangular bounding boxes with density bounds",
    }
    timings.append({
        "Method": "CLTree",
        "Paradigm": "Virtual Uniform Background Split",
        "Fit (ms)": t_fit,
        "Predict (ms)": t_pred,
        "Total (ms)": t_fit + t_pred,
        "Throughput (pts/s)": n_samples / (t_pred / 1000.0) if t_pred > 0 else 0,
    })

    # 3. HDBSCAN Condensed Tree via ClusterModel
    print("[3/4] Fitting HDBSCAN Condensed Tree (Density Hierarchy & Excess of Mass)...")
    t0 = time.perf_counter()
    hdb = ClusterModel(kind="hdbscan", min_rows=25, min_samples=15)
    hdb.fit(data_norm)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    hdb_labels = hdb.labels_
    t_pred = (time.perf_counter() - t0) * 1000.0

    results["HDBSCAN"] = {
        "model": hdb,
        "labels": hdb_labels,
        "paradigm": "Mutual Reachability MST + Condensed Hierarchy",
        "interpretability": "Cluster birth/death stability + GLOSH outlier scoring",
    }
    timings.append({
        "Method": "HDBSCAN",
        "Paradigm": "Mutual Reachability MST Hierarchy",
        "Fit (ms)": t_fit,
        "Predict (ms)": t_pred,
        "Total (ms)": t_fit + t_pred,
        "Throughput (pts/s)": n_samples / (t_pred / 1000.0) if t_pred > 0 else 0,
    })

    # 4. Unsupervised Random Forest (URF) via ClusterModel
    print("[4/4] Fitting Unsupervised Random Forest (Breiman Proximity & Agglomeration)...")
    t0 = time.perf_counter()
    urf = ClusterModel(kind="urf", n_clusters=4, max_depth=5, min_rows=5, n_trees=50, seed=42)
    urf.fit(data_num)
    t_fit = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    urf_labels = urf.labels_
    t_pred = (time.perf_counter() - t0) * 1000.0

    results["URF"] = {
        "model": urf,
        "labels": urf_labels,
        "paradigm": "Synthetic Marginal Permutation + Tree Proximity Kernel",
        "interpretability": "Pairwise leaf co-occurrence matrix P(i, j) & distance",
    }
    timings.append({
        "Method": "URF",
        "Paradigm": "Synthetic Marginal Tree Proximity",
        "Fit (ms)": t_fit,
        "Predict (ms)": t_pred,
        "Total (ms)": t_fit + t_pred,
        "Throughput (pts/s)": n_samples / (t_pred / 1000.0) if t_pred > 0 else 0,
    })

    # =========================================================================
    # 1. Execution Latency & Throughput Benchmark
    # =========================================================================
    print("\n" + "=" * 95)
    print("Execution Latency & Throughput Benchmark across Cluster Models")
    print("=" * 95)
    timing_df = pd.DataFrame(timings)
    formatted_timing = timing_df.copy()
    formatted_timing["Fit (ms)"] = formatted_timing["Fit (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Predict (ms)"] = formatted_timing["Predict (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Total (ms)"] = formatted_timing["Total (ms)"].map(lambda x: f"{x:8.2f}")
    formatted_timing["Throughput (pts/s)"] = formatted_timing["Throughput (pts/s)"].map(lambda x: f"{x:10.0f}")
    print(formatted_timing.to_string(index=False))

    # =========================================================================
    # 2. Crosstab Sparsity (CSP) Metric Evaluation
    # =========================================================================
    print("\n" + "=" * 95)
    print("Crosstab Sparsity (CSP) Metric Evaluation across Models")
    print("=" * 95)

    summary_rows = []
    csp_details = {}

    for name, res in results.items():
        labels = res["labels"]
        valid = [l for l in labels if l is not None and l != -1]
        noise_cnt = len(labels) - len(valid)
        n_clusters = len(set(valid))

        csp = crosstab_sparsity(df[eval_cols], labels, numeric_bin="decile")
        csp_score = csp["score"]
        csp_details[name] = csp

        summary_rows.append(
            {
                "Method": name,
                "Paradigm": res["paradigm"],
                "Clusters (k)": n_clusters,
                "Assigned": len(valid),
                "Noise / Unassigned": f"{noise_cnt} ({noise_cnt / len(labels) * 100:.1f}%)",
                "CSP Metric Score": f"{csp_score:.3f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Summary Performance Table ---")
    print(summary_df.to_string(index=False))

    # =========================================================================
    # 3. Discovered Cluster Size Distribution & Support
    # =========================================================================
    print("\n" + "=" * 95)
    print("Discovered Cluster Size Distributions")
    print("=" * 95)
    for name, res in results.items():
        labels = res["labels"]
        counts = Counter(labels)
        sorted_counts = sorted(counts.items(), key=lambda x: (x[0] is None or x[0] == -1, x[0]))
        dist_strs = [
            f"Cluster {c if c is not None and c != -1 else 'Noise'}: n={n} ({n/n_samples*100:.1f}%)"
            for c, n in sorted_counts
        ]
        print(f"[{name}]: " + " | ".join(dist_strs[:6]))

    # =========================================================================
    # 4. Top Feature Drivers per Method by CSP Metric
    # =========================================================================
    print("\n" + "=" * 95)
    print("Top Feature Drivers of Cluster Separation (CSP Metric = Segregation * Explanation)")
    print("=" * 95)
    for name, csp in csp_details.items():
        print(
            f"\n[{name}] Total CSP: {csp['score']:.3f} across {csp['n_clusters']} clusters:"
        )
        top_cols = csp["scores"].head(4)
        for _, r in top_cols.iterrows():
            print(
                f"  - {r['Columns']:<25} | Score={r['metric']:.3f} "
                f"(Segregation={r['seggregation']:.3f}, Explanation={r['explaination']:.3f})"
            )

    # =========================================================================
    # 5. Sample Discovered Decision Rules (PCT Interpretability)
    # =========================================================================
    print("\n" + "=" * 95)
    print("Sample Discovered Decision Rules from Predictive Clustering Tree (PCT)")
    print("=" * 95)
    pct_rules = results["PCT (CLUS)"]["model"].rules()
    for idx, rule in enumerate(pct_rules[:3]):
        pred = rule.get("predicate", "root")
        n_pts = rule.get("n", 0)
        print(f"  Cluster {idx} (Support n={n_pts}): {pred}")

    print("\n" + "=" * 95)
    print("Verification: All 4 clustering models fitted, evaluated, and compared successfully.")
    print("=" * 95)


if __name__ == "__main__":
    main()
