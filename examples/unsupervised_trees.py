"""Suite 8: Unsupervised Trees: Clustering & Anomaly Triage.

Demonstrates:
1. Predictive Clustering Trees (PCT) for interpretable multi-target clustering with rule extraction.
2. Extended Isolation Forest (EIF) with oblique hyperplanes eliminating axis-aligned ghost anomalies.
3. Comparative 2D anomaly scoring contour landscape.

Usage:
    uv run python examples/unsupervised_trees.py
"""

import math
import random

from mltreelib import AnomalyTree, ClusterTree, load_dataset


def main():
    # =========================================================================
    # Part 1: Predictive Clustering Trees (PCT) on Wine Quality
    # =========================================================================
    print("=== 1. Predictive Clustering Trees (PCT) on Wine Quality ===")
    data = load_dataset("wine_quality")
    features = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "alcohol"]
    n_sample = 1500

    X_wine = [[float(data["columns"][c][i]) for c in features] for i in range(n_sample)]
    print(f"Loaded Wine Quality dataset: {n_sample} samples, features: {features}")

    pct = ClusterTree(kind="pct", max_depth=3, min_rows=30)
    pct.fit(X_wine, col_names=features)

    print(f"Discovered clusters: {pct.n_clusters_}")
    rules = pct.rules()
    print("Sample discovered cluster rules:")
    for idx, rule in enumerate(rules[:3]):
        pred = rule.get("predicate", "root")
        n_pts = rule.get("n", 0)
        print(f"  Cluster {idx} (n={n_pts}): {pred}")
    print()

    # =========================================================================
    # Part 2: Extended Isolation Forest (EIF) vs Standard Isolation Forest
    # =========================================================================
    print("=== 2. Extended Isolation Forest vs Axis-Aligned Isolation Forest ===")
    # Generate 2D multimodal normal mixture + peripheral anomalies
    random.seed(42)
    n_inliers = 600
    n_outliers = 40

    X_2d = []
    # Cluster 1: center (-2.5, -2.5)
    for _ in range(n_inliers // 2):
        X_2d.append([random.gauss(-2.5, 0.65), random.gauss(-2.5, 0.65)])
    # Cluster 2: center (2.5, 2.5)
    for _ in range(n_inliers // 2):
        X_2d.append([random.gauss(2.5, 0.65), random.gauss(2.5, 0.65)])
    # Outliers scattered uniformly in outer box
    for _ in range(n_outliers):
        X_2d.append([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)])

    print(f"Generated 2D mixture: {n_inliers} inliers (2 clusters), {n_outliers} peripheral anomalies.")

    # 1. Standard Isolation Forest (axis-aligned splits)
    iforest = AnomalyTree(kind="isolation", max_depth=8, seed=42)
    iforest.fit(X_2d, col_names=["x1", "x2"])

    # 2. Extended Isolation Forest (oblique random hyperplanes)
    eif = AnomalyTree(kind="eif", max_depth=8, extension_level=1, seed=42)
    eif.fit(X_2d, col_names=["x1", "x2"])

    scores_if = iforest.score_samples(X_2d[-5:])
    scores_eif = eif.score_samples(X_2d[-5:])
    print(f"Outlier sample anomaly scores:")
    print(f"  Axis-Aligned IF : {[round(s, 3) for s in scores_if]}")
    print(f"  Extended IF     : {[round(s, 3) for s in scores_eif]}\n")

    # =========================================================================
    # Part 3: Generate Publication Multi-Panel Figure
    # =========================================================================
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

        c_teal = "#00897b"
        c_orange = "#f4511e"
        c_blue = "#0288d1"
        c_dark = "#263238"

        # Panel 1: Wine Quality Feature Scatter (Alcohol vs Volatile Acidity) colored by PCT Clusters
        alc_idx = features.index("alcohol")
        vol_idx = features.index("volatile_acidity")
        alc_vals = [row[alc_idx] for row in X_wine]
        vol_vals = [row[vol_idx] for row in X_wine]
        labels = pct.labels_ if pct.labels_ is not None else [0] * len(X_wine)

        scatter = ax1.scatter(alc_vals, vol_vals, c=labels, cmap="viridis", alpha=0.65, s=24, edgecolors="none")
        ax1.set_title("Predictive Clustering Tree (PCT) Partitions", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax1.set_xlabel("Alcohol Content (%)", fontsize=10, color=c_dark)
        ax1.set_ylabel("Volatile Acidity (g/dm³)", fontsize=10, color=c_dark)
        ax1.grid(True, linestyle=":", alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Assigned Cluster ID", fontsize=9, color=c_dark)

        # Panel 2: Cluster Sizes and Alcohol Content
        cluster_ids = sorted(list(set(labels)))
        cluster_counts = [labels.count(c) for c in cluster_ids]
        mean_alcs = [sum(alc_vals[i] for i, l in enumerate(labels) if l == c) / max(1, labels.count(c)) for c in cluster_ids]

        ax2.bar(cluster_ids, cluster_counts, color=c_teal, alpha=0.75, edgecolor=c_dark, width=0.55, label="Cluster Size (n)")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(cluster_ids, mean_alcs, color=c_orange, marker="o", linewidth=2.0, label="Mean Alcohol (%)")
        ax2.set_title("Discovered Cluster Profiles (Size & Mean Alcohol)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax2.set_xlabel("Cluster ID", fontsize=10, color=c_dark)
        ax2.set_ylabel("Sample Support (n)", fontsize=10, color=c_dark)
        ax2_twin.set_ylabel("Mean Alcohol (%)", fontsize=10, color=c_orange)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Compute 2D grid for contour scoring
        gx = np.linspace(-5.0, 5.0, 60)
        gy = np.linspace(-5.0, 5.0, 60)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = [[float(x), float(y)] for x, y in zip(GX.ravel(), GY.ravel())]

        Z_if = np.array(iforest.score_samples(grid_pts)).reshape(GX.shape)
        Z_eif = np.array(eif.score_samples(grid_pts)).reshape(GX.shape)

        # Panel 3: Axis-Aligned Isolation Forest Contours
        cp3 = ax3.contourf(GX, GY, Z_if, levels=15, cmap="YlOrRd_r", alpha=0.85)
        ax3.scatter([p[0] for p in X_2d[:n_inliers]], [p[1] for p in X_2d[:n_inliers]], color=c_dark, s=12, alpha=0.4, label="Inliers")
        ax3.scatter([p[0] for p in X_2d[n_inliers:]], [p[1] for p in X_2d[n_inliers:]], color=c_blue, s=28, marker="x", label="Anomalies")
        ax3.set_title("Standard Isolation Forest (Axis-Aligned Ghost Artifacts)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax3.set_xlabel("Dimension 1", fontsize=10, color=c_dark)
        ax3.set_ylabel("Dimension 2", fontsize=10, color=c_dark)
        plt.colorbar(cp3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=8, loc="upper left")

        # Panel 4: Extended Isolation Forest Contours
        cp4 = ax4.contourf(GX, GY, Z_eif, levels=15, cmap="YlOrRd_r", alpha=0.85)
        ax4.scatter([p[0] for p in X_2d[:n_inliers]], [p[1] for p in X_2d[:n_inliers]], color=c_dark, s=12, alpha=0.4, label="Inliers")
        ax4.scatter([p[0] for p in X_2d[n_inliers:]], [p[1] for p in X_2d[n_inliers:]], color=c_blue, s=28, marker="x", label="Anomalies")
        ax4.set_title("Extended Isolation Forest (Smooth Oblique Contours)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax4.set_xlabel("Dimension 1", fontsize=10, color=c_dark)
        ax4.set_ylabel("Dimension 2", fontsize=10, color=c_dark)
        plt.colorbar(cp4, ax=ax4, fraction=0.046, pad=0.04)
        ax4.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=8, loc="upper left")

        plt.suptitle("mltreelib Suite 8: Unsupervised Trees: Clustering & Anomaly Triage", fontsize=15, fontweight="bold", color=c_dark, y=0.98)
        plt.tight_layout()
        out_fig = "docs/assets/unsupervised_clusters_anomalies.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"Generated publication asset: {out_fig}")
    except ImportError:
        print("Note: matplotlib not found; install mltreelib[plotting] to generate figures.")


if __name__ == "__main__":
    main()
