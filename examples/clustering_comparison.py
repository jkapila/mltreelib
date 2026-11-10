"""Comprehensive Comparison of Tree-Based Clustering Methods in mltreelib.

Compares:
1. Predictive Clustering Trees (PCT / CLUS): Multi-target variance reduction.
2. CLTree (Liu et al. KDD 2000): Hyper-rectangular partition against analytical noise.
3. HDBSCAN Condensed Tree: Mutual reachability MST + Excess of Mass stability.
4. Unsupervised Random Forest (URF): Breiman proximity kernel + Lance-Williams agglomeration.

Evaluates and compares all methods using Jitin Kapila's Crosstab Sparsity (CSP)
metric (https://gist.github.com/jkapila/83bb8f6461ec91bfced437762f2c9220):
  CSP = sum_j [ Segregation(j) * Explanation(j) ]
where:
  Segregation = n_v / max(l, k)  (concentration above median contingency cell)
  Explanation = ln(N / (l * k))  (information content of the cross-tabulation)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mltreelib import (
    CLTree,
    HDBSCAN,
    PredictiveClusteringTree,
    UnsupervisedRandomForest,
)


def crosstab_sparsity(df, clusters, numeric_bin="decile", method="fd", exhaustive=False):
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
        raise ValueError("Clusters and number of observations are different. Cannot Proceed!")

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

    score_df = pd.DataFrame({
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
    })

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

    print("=" * 90)
    print("mltreelib Tree-Based Clustering Suite Comparison & Benchmarking")
    print(f"Dataset: credit-g ({len(df)} samples, {len(eval_cols)} evaluation features)")
    print("=" * 90)

    results = {}

    # 1. Predictive Clustering Tree (PCT / CLUS)
    print("\n[1/4] Fitting Predictive Clustering Tree (PCT / CLUS)...")
    pct = PredictiveClusteringTree(max_depth=3, min_samples_split=25, min_samples_leaf=15)
    pct.fit(data_num, col_names=numeric_cols, dtypes=["float"] * len(numeric_cols))
    pct_labels = pct.predict_cluster(data_num)
    results["PCT (CLUS)"] = {
        "labels": pct_labels,
        "paradigm": "Divisive Multi-Target Variance Reduction",
        "interpretability": "Extracted decision paths with centroid prototypes",
    }

    # 2. CLTree (Liu et al. KDD 2000)
    print("[2/4] Fitting CLTree (Decision Tree Clustering via Analytical Noise)...")
    cltree = CLTree(max_depth=4, min_samples_leaf=15, min_density=0.60)
    cltree.fit(data_num, col_names=numeric_cols)
    clt_labels = cltree.predict_cluster(data_num)
    results["CLTree"] = {
        "labels": clt_labels,
        "paradigm": "Decision Tree against Virtual Uniform Background",
        "interpretability": "Hyper-rectangular bounding boxes with density bounds",
    }

    # 3. HDBSCAN Condensed Tree
    print("[3/4] Fitting HDBSCAN Condensed Tree (Density Hierarchy & Excess of Mass)...")
    hdb = HDBSCAN(min_cluster_size=25, min_samples=15)
    hdb.fit(data_norm)
    results["HDBSCAN"] = {
        "labels": hdb.labels,
        "paradigm": "Mutual Reachability MST + Condensed Hierarchy",
        "interpretability": "Cluster birth/death stability + GLOSH outlier scoring",
    }

    # 4. Unsupervised Random Forest (URF)
    print("[4/4] Fitting Unsupervised Random Forest (Breiman Proximity & Agglomeration)...")
    urf = UnsupervisedRandomForest(n_trees=50, max_depth=5, min_samples_leaf=5, seed=42)
    urf.fit(data_num, n_clusters=4)
    results["URF"] = {
        "labels": urf.labels,
        "paradigm": "Synthetic Marginal Permutation + Tree Proximity Kernel",
        "interpretability": "Pairwise leaf co-occurrence matrix P(i, j) & distance",
    }

    # Evaluate Crosstab Sparsity (CSP) Metric for each model
    print("\n" + "=" * 90)
    print("Evaluating Crosstab Sparsity (CSP) Metric across Models")
    print("=" * 90)

    summary_rows = []
    csp_details = {}

    for name, res in results.items():
        labels = res["labels"]
        # Count clusters and noise
        valid = [l for l in labels if l is not None and l != -1]
        noise_cnt = len(labels) - len(valid)
        n_clusters = len(set(valid))

        csp = crosstab_sparsity(df[eval_cols], labels, numeric_bin="decile")
        csp_score = csp["score"]
        csp_details[name] = csp

        summary_rows.append({
            "Method": name,
            "Paradigm": res["paradigm"],
            "Clusters (k)": n_clusters,
            "Assigned": len(valid),
            "Noise / Unassigned": f"{noise_cnt} ({noise_cnt / len(labels) * 100:.1f}%)",
            "CSP Metric Score": f"{csp_score:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Summary Comparison Table ---")
    print(summary_df.to_string(index=False))

    # Top variables segregated per method
    print("\n" + "=" * 90)
    print("Top Feature Drivers of Cluster Separation (by CSP Metric = Segregation * Explanation)")
    print("=" * 90)
    for name, csp in csp_details.items():
        print(f"\n[{name}] Total CSP: {csp['score']:.3f} across {csp['n_clusters']} clusters:")
        top_cols = csp["scores"].head(4)
        for _, r in top_cols.iterrows():
            print(
                f"  - {r['Columns']:<25} | Score={r['metric']:.3f} "
                f"(Segregation={r['seggregation']:.3f}, Explanation={r['explaination']:.3f})"
            )

    # Architectural Comparison Summary
    print("\n" + "=" * 90)
    print("Architectural Comparison: How Tree Clustering Methods Differ")
    print("=" * 90)
    print("""
1. Predictive Clustering Trees (PCT / CLUS):
   - Philosophy: Top-down hierarchical variance reduction on one or multiple targets.
   - When to use: When cluster assignments must be directly interpretable via clear decision
     paths (if checking_status == 'no' and duration <= 24 then Cluster 3) and multi-objective targets.

2. CLTree (Liu et al. KDD 2000):
   - Philosophy: Discriminative decision tree separating data points from synthetic background space.
   - When to use: When hyper-rectangular boundaries are required, and when background empty space
     must be cleanly separated from dense clusters.

3. HDBSCAN Condensed Tree (Campello et al. 2013):
   - Philosophy: Non-parametric density hierarchy built from mutual reachability distance graph.
   - When to use: When clusters have arbitrary non-convex shapes, varying densities, and when
     explicit noise rejection (GLOSH outlier score) is critical.

4. Unsupervised Random Forest (URF - Breiman 2001, Shi & Horvath 2006):
   - Philosophy: Ensemble proximity metric P(i, j) = fraction of trees where instances share leaf.
   - When to use: When natural mixed-type proximity is needed without explicit distance metrics,
     and for feeding into manifold learning or hierarchical agglomeration.
""")


if __name__ == "__main__":
    main()
