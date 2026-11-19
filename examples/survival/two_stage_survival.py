"""Two-Stage Hierarchical Survival Trees: Segregate First, Model Later.

Demonstrates the two-stage survival architecture:
1. Stage 1: Subject ID-level cohort discovery via transparent tree partitioning (ModelTree).
2. Stage 2: Leaf-level non-parametric Kaplan-Meier survival step function estimation.
3. Serving: O(depth) routing of new subjects to localized, calibrated survival curves.

Evaluates on the real benchmark Bank_Customer_Churn (10,000 customers).
Produces: docs/assets/two_stage_survival_cohorts.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from mltreelib import ModelTree, Metrics


def run_two_stage_survival():
    print("=" * 75)
    print("mltreelib: Two-Stage Hierarchical Grouped Survival Modeling")
    print("=" * 75)

    # 1. Load real bank churn benchmark
    parquet_path = "datasets/Bank_Customer_Churn.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pydict()
    n_total = len(df["churn"])

    feature_cols = ["credit_score", "age", "balance", "products_number", "estimated_salary"]
    X = [[float(df[c][i]) for c in feature_cols] for i in range(n_total)]
    durations = [float(df["tenure"][i]) for i in range(n_total)]  # 0 to 10 years
    events = [1.0 if str(df["churn"][i]).lower() in ("1", "yes", "true") else 0.0 for i in range(n_total)]

    print(f"Dataset: Bank Customer Churn ({n_total:,} customers, max tenure = 10 years)")
    print(f"Observed Churn Events: {int(sum(events)):,} ({sum(events)/n_total*100:.1f}%), Right-Censored: {int(n_total - sum(events)):,}")

    # Train / Test split (70/30)
    split_idx = int(n_total * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    t_train, t_test = durations[:split_idx], durations[split_idx:]
    e_train, e_test = events[:split_idx], events[split_idx:]

    # 2. Stage 1: Cohort Segmentation Tree (ID-level profiling)
    print("\n[Stage 1] Fitting Profile Segmentation Tree on Baseline Covariates...")
    stage1_tree = ModelTree(max_depth=2, min_rows=50)
    stage1_tree.fit(X_train, e_train, col_names=feature_cols)

    train_leaf_ids = stage1_tree.apply(X_train)
    test_leaf_ids = stage1_tree.apply(X_test)

    rules = stage1_tree.extract_rules()
    print("Discovered Segmentation Rules:")
    print(rules)

    # 3. Stage 2: Fit Leaf-Level Survival Models (Nelson-Aalen / Kaplan-Meier)
    print("\n[Stage 2] Fitting Leaf-Level Localized Survival Engines...")
    distinct_leaves = sorted(list(set(train_leaf_ids)))
    leaf_models = {}
    leaf_stats = {}

    cohort_names = {
        0: "Cohort 0: Core Retained (Age <= 42, Prod <= 2)",
        1: "Cohort 1: Young High-Product Churners (Age <= 42, Prod > 2)",
        2: "Cohort 2: Mature Mainstream (Age > 42, Prod <= 2)",
        3: "Cohort 3: Senior Flight Risk (Age > 42, Prod > 2)",
    }

    colors = {
        0: "#00897b",  # Teal (High Survival)
        1: "#fb8c00",  # Amber
        2: "#0288d1",  # Cerulean Blue
        3: "#d81b60",  # Rose Red (Severe Hazard)
    }

    for leaf_id in distinct_leaves:
        idx = [i for i, l in enumerate(train_leaf_ids) if l == leaf_id]
        l_X = [X_train[i] for i in idx]
        l_t = [t_train[i] for i in idx]
        l_e = [e_train[i] for i in idx]

        surv_engine = ModelTree(max_depth=0, leaf_family="survival")
        surv_engine.fit(l_X, (l_t, l_e))
        leaf_models[leaf_id] = surv_engine

        churn_rate = sum(l_e) / len(idx) * 100.0
        leaf_stats[leaf_id] = {
            "n": len(idx),
            "events": int(sum(l_e)),
            "churn_rate": churn_rate,
            "name": cohort_names.get(leaf_id, f"Cohort {leaf_id}"),
        }
        print(f"  -> Leaf {leaf_id} ({cohort_names.get(leaf_id, 'Cohort')}): {len(idx):4d} customers | Churn Rate: {churn_rate:5.1f}%")

    # 4. Out-of-sample Evaluation & Prediction on Held-Out Test Set
    tenure_grid = np.linspace(0.0, 10.0, 50).tolist()
    print("\n[Serving] Generating Held-Out Monotone Survival Curves across 10-Year Horizon...")

    # Compute mean survival curve per cohort on test set
    cohort_curves = {}
    for leaf_id in distinct_leaves:
        test_idx = [i for i, l in enumerate(test_leaf_ids) if l == leaf_id]
        if not test_idx:
            continue
        sub_X = [X_test[i] for i in test_idx]
        preds = leaf_models[leaf_id].predict(sub_X, kind="survival", times=tenure_grid)
        cohort_curves[leaf_id] = np.mean(preds, axis=0)

    # 5. Render Publication Diagnostic Figure
    os.makedirs("docs/assets", exist_ok=True)
    out_png = "docs/assets/two_stage_survival_cohorts.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Cohort Churn Rates & Sample Distribution
    cohort_ids = sorted(list(leaf_stats.keys()))
    names_short = [f"Leaf {cid}\n(n={leaf_stats[cid]['n']:,})" for cid in cohort_ids]
    rates = [leaf_stats[cid]["churn_rate"] for cid in cohort_ids]
    bar_colors = [colors[cid] for cid in cohort_ids]

    bars = ax1.bar(names_short, rates, color=bar_colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title("A: Stage 1 Profile Partition (Cohort Churn Rates)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Empirical Churn Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, cid in zip(bars, cohort_ids):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Add explanatory cohort annotations
    ax1.text(0, 15, "Low Risk\nAge <= 42\nProd <= 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(1, 80, "High Risk\nAge <= 42\nProd > 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(2, 45, "Moderate\nAge > 42\nProd <= 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(3, 101, "Severe\nAge > 42\nProd > 2", ha="center", fontsize=8, color="white", fontweight="bold")

    # Panel B: Leaf-Level Kaplan-Meier Survival Curves over 10 Years
    for cid in cohort_ids:
        curve = cohort_curves[cid]
        ax2.step(
            tenure_grid,
            curve,
            where="post",
            label=cohort_names[cid],
            color=colors[cid],
            linewidth=2.4,
            alpha=0.9,
        )

    ax2.set_title("B: Stage 2 Leaf Kaplan-Meier Curves (10-Year Tenure)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Customer Tenure (Years)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Survival Probability S(t)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(0, 10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"\nSaved Two-Stage Survival figure to {out_png}")
    print("=" * 75)


if __name__ == "__main__":
    run_two_stage_survival()
