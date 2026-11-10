"""Comprehensive Showcase of mltreelib Tree-Based Clustering & TrieKNN.

Demonstrates:
1. TrieKNN (Jitin Kapila 2018/2025):
   - Mixed-type tabular classification and regression.
   - Non-binary categorical prefix branching + leaf numerical KNN.
   - Hierarchical parent back-off on unseen categories.
2. Predictive Clustering Trees (PCT / CLUS):
   - Supervised multi-target clustering with transparent rule extraction.
   - Unsupervised divisive clustering over numeric features.
"""

from pathlib import Path
import pyarrow.parquet as pq

from mltreelib import (
    CLTree,
    HDBSCAN,
    PredictiveClusteringTree,
    TrieKNN,
    UnsupervisedRandomForest,
)


def showcase_trie_knn(df):
    print("=" * 80)
    print("1. TrieKNN on credit-g Dataset (Mixed Categorical & Numerical)")
    print("=" * 80)

    # Mixed columns: 3 categorical, 3 numerical
    cat_cols = ["checking_status", "credit_history", "housing"]
    num_cols = ["duration_months", "credit_amount", "age_years"]
    all_cols = cat_cols + num_cols
    dtypes = ["cat", "cat", "cat", "float", "float", "float"]

    # Target: good_or_bad_customer (1 = good, 0 = bad)
    y_raw = df["good_or_bad_customer"].tolist()
    y = [1.0 if str(v).lower() in ["1", "good", "true"] else 0.0 for v in y_raw]
    x = df[all_cols].values.tolist()

    # Split train / test
    split = 800
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = TrieKNN(task="classification", k=5, weights="distance")
    model.fit(x_train, y_train, col_names=all_cols, dtypes=dtypes)

    print(f"Trained TrieKNN on {len(x_train)} mixed records. Classes: {model.classes}")

    # Predict test cases
    preds = model.predict(x_test[:10])
    probs = model.predict_proba(x_test[:10])

    print("\nSample Test Predictions:")
    for i in range(5):
        row = df.iloc[split + i]
        actual = y_test[i]
        pred = preds[i]
        prob = probs[i]
        print(f"  Case {i+1}: Actual={actual:.0f} | Pred={pred:.0f} | P(Good)={prob[1]:.2f}")
        print(f"    Context: {row['checking_status']}, {row['credit_history']}, {row['housing']}")
        print(f"    Numeric: ${row['credit_amount']:.0f}, {row['duration_months']} mo, Age {row['age_years']}")

    # Demonstrate Hierarchical Back-off on Unseen Category
    print("\nDemonstrating Hierarchical Back-off:")
    synth_query = [["unseen_checking_val", "critical account / other credits existing", "own", 48.0, 10000.0, 35.0]]
    synth_pred = model.predict(synth_query)[0]
    synth_prob = model.predict_proba(synth_query)[0]
    print(f"  Unseen category query routed via parent back-off: Pred={synth_pred:.0f}, P(Good)={synth_prob[1]:.2f}")


def showcase_pct(df):
    print("\n" + "=" * 80)
    print("2. Predictive Clustering Trees (PCT / CLUS)")
    print("=" * 80)

    # Multi-target setup:
    # Descriptive X: checking_status, credit_history, housing, age_years
    # Target Y (multi-objective): duration_months, credit_amount
    descriptive_cols = ["checking_status", "credit_history", "housing", "age_years"]
    desc_dtypes = ["cat", "cat", "cat", "float"]
    target_cols = ["duration_months", "credit_amount"]

    x = df[descriptive_cols].values.tolist()
    y = df[target_cols].values.tolist()

    pct = PredictiveClusteringTree(max_depth=4, min_samples_split=20, min_samples_leaf=10)
    pct.fit(
        x,
        y=y,
        col_names=descriptive_cols,
        dtypes=desc_dtypes,
        target_names=target_cols,
    )

    print(f"PCT partitioned data into {pct.n_clusters} interpretable clusters.")
    print("\nCluster Prototypes and Descriptive Rules:")
    rules = pct.cluster_rules()
    for cid, rule, proto, size in rules[:5]:
        print(f"\n  [Cluster {cid}] (Size: {size} instances)")
        print(f"    Rule: {rule}")
        print(f"    Centroid: Duration = {proto[0]:.1f} mo | Credit Amount = ${proto[1]:.0f}")

    # Unsupervised Mode (Y omitted)
    print("\n--- Unsupervised PCT Clustering ---")
    numeric_cols = ["duration_months", "credit_amount", "installment_rate_percent", "age_years"]
    x_num = df[numeric_cols].values.tolist()
    pct_unsup = PredictiveClusteringTree(max_depth=3, min_samples_split=30, min_samples_leaf=15)
    pct_unsup.fit(x_num, col_names=numeric_cols, dtypes=["float"] * len(numeric_cols))
    print(f"Unsupervised PCT extracted {pct_unsup.n_clusters} clusters across numeric dimensions.")


def showcase_cltree(df):
    print("\n" + "=" * 80)
    print("3. CLTree (Clustering Trees - Liu et al. KDD 2000)")
    print("=" * 80)

    # Use continuous features to isolate dense clusters from background empty space
    cols = ["duration_months", "credit_amount", "age_years"]
    data = df[cols].values.tolist()

    clt = CLTree(max_depth=5, min_samples_leaf=15, min_density=0.60)
    clt.fit(data, col_names=cols)

    print(f"CLTree discovered {clt.n_clusters} dense clusters surrounded by sparse space.")
    bboxes = clt.cluster_bounding_boxes()
    for cid, bounds, size, density in bboxes[:5]:
        print(f"\n  [Cluster {cid}] Size: {size} instances | Data Density: {density * 100:.1f}%")
        print(f"    Duration range: [{bounds[0][0]:.1f}, {bounds[0][1]:.1f}] mo")
        print(f"    Credit range:   [${bounds[1][0]:.0f}, ${bounds[1][1]:.0f}]")
        print(f"    Age range:      [{bounds[2][0]:.1f}, {bounds[2][1]:.1f}] yrs")

    # Evaluate query points: dense vs sparse noise
    dense_pt = [12.0, 1500.0, 30.0]
    outlier_pt = [100.0, 50000.0, 99.0]
    c_dense = clt.predict_cluster([dense_pt])[0]
    c_outlier = clt.predict_cluster([outlier_pt])[0]
    print("\nPoint Assignments:")
    print(f"  Typical point {dense_pt} -> Cluster {c_dense}")
    print(f"  Extreme point {outlier_pt} -> {c_outlier} (None = identified as noise/outlier space)")


def showcase_hdbscan(df):
    print("\n" + "=" * 80)
    print("4. HDBSCAN Condensed Tree (Hierarchical Density-Based Clustering)")
    print("=" * 80)

    # Use continuous features: duration, credit_amount, age
    cols = ["duration_months", "credit_amount", "age_years"]
    data = df[cols].values.tolist()

    # Normalize data for distance-based reachability
    means = [sum(row[j] for row in data) / len(data) for j in range(len(cols))]
    stds = [
        (sum((row[j] - means[j]) ** 2 for row in data) / len(data)) ** 0.5
        for j in range(len(cols))
    ]
    norm_data = [
        [(row[j] - means[j]) / (stds[j] if stds[j] > 1e-6 else 1.0) for j in range(len(cols))]
        for row in data
    ]

    hdb = HDBSCAN(min_cluster_size=25, min_samples=15)
    hdb.fit(norm_data)

    print(f"HDBSCAN discovered {hdb.n_clusters} stable density clusters via Excess of Mass.")
    labels = hdb.labels
    cluster_counts = {}
    noise_count = 0
    for l in labels:
        if l is None:
            noise_count += 1
        else:
            cluster_counts[l] = cluster_counts.get(l, 0) + 1

    for cid in sorted(cluster_counts.keys()):
        print(f"  Cluster {cid}: {cluster_counts[cid]} points")
    print(f"  Noise / Outliers (lambda death): {noise_count} points ({noise_count / len(data) * 100:.1f}%)")

    # Condensed Tree inspection
    tree = hdb.condensed_tree()
    selected_nodes = [n for n in tree if n["selected"]]
    print(f"\nCondensed Tree has {len(tree)} cluster split/shedding events.")
    print(f"Excess of Mass selected {len(selected_nodes)} persistent cluster nodes.")
    for sn in selected_nodes[:3]:
        print(
            f"  Node {sn['node_id']}: Birth lambda={sn['lambda_birth']:.3f}, "
            f"Death lambda={sn['lambda_death']:.3f}, Stability={sn['stability']:.2f}, Size={sn['size']}"
        )

    # GLOSH Outlier scores
    outlier_scores = hdb.outlier_scores
    top_outlier_indices = sorted(range(len(outlier_scores)), key=lambda i: outlier_scores[i], reverse=True)[:5]
    print("\nTop 5 Outliers by GLOSH (Global-Local Outlier Score from Hierarchies):")
    for idx in top_outlier_indices:
        row = df.iloc[idx]
        print(
            f"  Index {idx}: GLOSH={outlier_scores[idx]:.3f} | Duration={row['duration_months']} mo, "
            f"Amount=${row['credit_amount']:.0f}, Age={row['age_years']}"
        )


def showcase_urf(df):
    print("\n" + "=" * 80)
    print("5. Unsupervised Random Forest (Breiman Proximity & Agglomeration)")
    print("=" * 80)

    cols = ["duration_months", "credit_amount", "age_years", "installment_rate_percent"]
    data = df[cols].values.tolist()

    urf = UnsupervisedRandomForest(n_trees=50, max_depth=6, min_samples_leaf=5, seed=42)
    urf.fit(data, n_clusters=3)

    print(f"Unsupervised Forest grew 50 synthetic-scrambled margin-separator trees.")
    print(f"Agglomerative clustering on D = sqrt(1 - P) yielded {urf.n_clusters} clusters.")

    # Cluster distribution
    labels = urf.labels
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    for cid in sorted(counts.keys()):
        print(f"  Cluster {cid}: {counts[cid]} instances")

    # Proximity matrix inspection
    prox = urf.proximity_matrix
    sample_prox = [prox[i][j] for i in range(10) for j in range(i + 1, 10)]
    avg_prox = sum(sample_prox) / len(sample_prox) if sample_prox else 0.0
    print(f"\nProximity matrix shape: [{len(prox)} x {len(prox)}]")
    print(f"Sample off-diagonal average co-occurrence: {avg_prox * 100:.1f}%")

    # Proximity outlier scores
    outlier_scores = urf.outlier_scores
    top_outliers = sorted(range(len(outlier_scores)), key=lambda i: outlier_scores[i], reverse=True)[:5]
    print("\nTop 5 Outliers by Proximity Isolation:")
    for idx in top_outliers:
        row = df.iloc[idx]
        print(
            f"  Index {idx}: Outlier Score={outlier_scores[idx]:.3f} | Duration={row['duration_months']} mo, "
            f"Amount=${row['credit_amount']:.0f}, Age={row['age_years']}"
        )


def main():
    path = Path("datasets/credit-g.parquet")
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    with pq.ParquetFile(path) as pf:
        table = pf.read()
        df = table.to_pandas()

    showcase_trie_knn(df)
    showcase_pct(df)
    showcase_cltree(df)
    showcase_hdbscan(df)
    showcase_urf(df)
    print("\n" + "=" * 80)
    print("Showcase completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
