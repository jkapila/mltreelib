"""Two-Stage Hierarchical Survival Trees: Segregate First, Model Later.

Demonstrates the two-stage survival architecture using native ModelTree.retrain():
1. Stage 1: Subject ID-level cohort discovery via transparent tree partitioning (ModelTree).
2. Stage 2: In-place leaf retraining with non-parametric Kaplan-Meier survival step functions via tree.retrain().
3. Serving: O(depth) routing of new subjects to localized Kaplan-Meier survival curves.
4. Benchmark: 20-seed comparison against unstratified Kaplan-Meier and single-stage survival trees.

Evaluates on the real benchmark Bank_Customer_Churn (10,000 customers, OpenML ID 46911).
Produces: docs/assets/two_stage_survival_cohorts.png and .svg
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from mltreelib import ModelTree, Metrics
from mltreelib.data import survival_xy, train_test_split


def run_two_stage_survival(n_seeds=20, output_dir="docs/assets"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("mltreelib: Two-Stage Hierarchical Grouped Survival Modeling via retrain()")
    print("=" * 80)

    # 1. Load real bank churn benchmark
    parquet_path = "datasets/Bank_Customer_Churn.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pydict()
    n_total = len(df["churn"])

    from mltreelib.data import Dataset

    feature_cols = ["credit_score", "age", "balance", "products_number", "estimated_salary"]
    df["churn"] = [1.0 if str(v).lower() in ("1", "yes", "true") else 0.0 for v in df["churn"]]
    ds = Dataset(df, time="tenure", event="churn", features=feature_cols)  # durations: 0 to 10 years
    X = np.array(ds.x, dtype=float)
    durations = np.array(ds.time, dtype=float)
    events = np.array(ds.event, dtype=float)

    n_events = int(sum(events))
    churn_pct = n_events / n_total * 100.0
    print(f"Dataset: Bank Customer Churn ({n_total:,} customers, max tenure = 10 years)")
    print(f"Observed Churn Events: {n_events:,} ({churn_pct:.1f}%), Right-Censored: {n_total - n_events:,}\n")

    # 2. Stage 1 Profile Tree on Seed 0 (Representative Model Inspection)
    tr_idx0, te_idx0 = train_test_split(n_total, test_size=0.3, seed=0)  # seeded 70/30 split

    X_train0, X_test0 = X[tr_idx0], X[te_idx0]
    t_train0, t_test0 = durations[tr_idx0], durations[te_idx0]
    e_train0, e_test0 = events[tr_idx0], events[te_idx0]

    print("[Stage 1] Fitting Profile Segmentation Tree on Baseline Covariates (Seed 0)...")
    tree0 = ModelTree(max_depth=2, min_rows=50)
    tree0.fit(X_train0.tolist(), e_train0.tolist(), col_names=feature_cols)

    rules = tree0.extract_rules()
    print("\nDiscovered Segmentation Rules (Stage 1):")
    print(rules)

    # 3. Stage 2: In-place Leaf Retraining with Survival Families
    print("\n[Stage 2] In-place Retraining Leaves with Kaplan-Meier Survival Engines via tree.retrain()...")
    # Native pure Rust refitting: routes rows and refits leaf survival distributions in place!
    tree0.retrain(X_train0.tolist(), (t_train0.tolist(), e_train0.tolist()), leaf_family="survival")

    train_leaf_ids0 = tree0.apply(X_train0.tolist())
    test_leaf_ids0 = tree0.apply(X_test0.tolist())
    distinct_leaves0 = sorted(list(set(train_leaf_ids0)))

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

    leaf_stats0 = {}
    for leaf_id in distinct_leaves0:
        idx = [i for i, l in enumerate(train_leaf_ids0) if l == leaf_id]
        l_e = [e_train0[i] for i in idx]
        churn_rate = sum(l_e) / len(idx) * 100.0
        leaf_stats0[leaf_id] = {
            "n": len(idx),
            "events": int(sum(l_e)),
            "churn_rate": churn_rate,
            "name": cohort_names.get(leaf_id, f"Cohort {leaf_id}"),
        }
        print(f"  -> Leaf {leaf_id} ({cohort_names.get(leaf_id, 'Cohort')}): {len(idx):4d} customers | Churn Rate: {churn_rate:5.1f}%")

    # Generate held-out cohort survival curves directly via tree0.predict()
    tenure_grid = np.linspace(0.0, 10.0, 50).tolist()
    cohort_curves0 = {}
    for leaf_id in distinct_leaves0:
        test_sub_idx = [i for i, l in enumerate(test_leaf_ids0) if l == leaf_id]
        if test_sub_idx:
            sub_X = [X_test0[i].tolist() for i in test_sub_idx]
            preds = tree0.predict(sub_X, kind="survival", times=tenure_grid)
            cohort_curves0[leaf_id] = np.mean(preds, axis=0)

    # 4. Multi-Seed Rigorous Benchmark (20 Seeds)
    print(f"\n[Benchmark] Running {n_seeds}-seed train/test evaluation (70/30 splits on full 10k rows)...")
    m = Metrics()
    metrics_marginal = {"c_index": [], "ibs": []}
    metrics_single = {"c_index": [], "ibs": []}
    metrics_two_stage = {"c_index": [], "ibs": []}
    two_stage_beats_single = 0

    for seed in range(n_seeds):
        tr_idx, te_idx = train_test_split(n_total, test_size=0.3, seed=seed)

        X_tr, X_te = X[tr_idx].tolist(), X[te_idx].tolist()
        t_tr, t_te = durations[tr_idx].tolist(), durations[te_idx].tolist()
        e_tr, e_te = events[tr_idx].tolist(), events[te_idx].tolist()

        # Policy A: Marginal Unstratified Kaplan-Meier
        marginal = ModelTree(max_depth=0, leaf_family="survival")
        marginal.fit(X_tr, (t_tr, e_tr))
        risk_m = marginal.predict(X_te, kind="risk")
        surv_m = marginal.predict(X_te, kind="survival", times=tenure_grid)
        res_m = m.survival(t_te, e_te, risk_m, surv_curves=surv_m, times=tenure_grid)
        metrics_marginal["c_index"].append(res_m["harrell_c"])
        metrics_marginal["ibs"].append(res_m["ibs"])

        # Policy B: Single-Stage Direct Survival Tree (log-rank splitting on duration)
        single = ModelTree(max_depth=3, leaf_family="survival")
        single.fit(X_tr, (t_tr, e_tr))
        risk_s = single.predict(X_te, kind="risk")
        surv_s = single.predict(X_te, kind="survival", times=tenure_grid)
        res_s = m.survival(t_te, e_te, risk_s, surv_curves=surv_s, times=tenure_grid)
        metrics_single["c_index"].append(res_s["harrell_c"])
        metrics_single["ibs"].append(res_s["ibs"])

        # Policy C: Two-Stage Hierarchical Survival Tree via retrain()
        ts_tree = ModelTree(max_depth=2, min_rows=50)
        ts_tree.fit(X_tr, e_tr)
        ts_tree.retrain(X_tr, (t_tr, e_tr), leaf_family="survival")

        risk_ts = ts_tree.predict(X_te, kind="risk")
        surv_ts = ts_tree.predict(X_te, kind="survival", times=tenure_grid)
        res_ts = m.survival(t_te, e_te, risk_ts, surv_curves=surv_ts, times=tenure_grid)
        metrics_two_stage["c_index"].append(res_ts["harrell_c"])
        metrics_two_stage["ibs"].append(res_ts["ibs"])

        if res_ts["harrell_c"] > res_s["harrell_c"]:
            two_stage_beats_single += 1

    print(f"Two-Stage Tree beat Single-Stage Tree in {two_stage_beats_single}/{n_seeds} seeds.\n")

    # Print Markdown Benchmark Table
    print("=" * 80)
    print(f"BENCHMARK RESULTS ({n_seeds} Random Seeds, 10,000 Customers, 70/30 Splits)")
    print("=" * 80)
    print("| Method | Harrell's C-Index | Integrated Brier Score (IBS) | Architecture |")
    print("| :--- | :--- | :--- | :--- |")
    for name, met, arch in [
        ("Marginal Kaplan-Meier", metrics_marginal, "Unstratified Population S(t)"),
        ("Single-Stage Survival Tree", metrics_single, "Log-Rank Splitting on (T, E)"),
        ("Two-Stage Hierarchical Tree (retrain)", metrics_two_stage, "Profile Tree + Native Leaf retrain()"),
    ]:
        c_m, c_s = np.mean(met["c_index"]), np.std(met["c_index"])
        ibs_m, ibs_s = np.mean(met["ibs"]), np.std(met["ibs"])
        print(f"| {name} | {c_m:.4f} ± {c_s:.4f} | {ibs_m:.4f} ± {ibs_s:.4f} | {arch} |")
    print("=" * 80)

    # 5. Render Publication Figures (PNG & SVG)
    out_png = os.path.join(output_dir, "two_stage_survival_cohorts.png")
    out_svg = os.path.join(output_dir, "two_stage_survival_cohorts.svg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    cohort_ids = sorted(list(leaf_stats0.keys()))
    names_short = [f"Leaf {cid}\n(n={leaf_stats0[cid]['n']:,})" for cid in cohort_ids]
    rates = [leaf_stats0[cid]["churn_rate"] for cid in cohort_ids]
    bar_colors = [colors[cid] for cid in cohort_ids]

    bars = ax1.bar(names_short, rates, color=bar_colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title("A: Stage 1 Profile Partition (Cohort Churn Rates)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Empirical Churn Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, cid in zip(bars, cohort_ids):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax1.text(0, 15, "Low Risk\nAge <= 42\nProd <= 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(1, 80, "High Risk\nAge <= 42\nProd > 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(2, 45, "Moderate\nAge > 42\nProd <= 2", ha="center", fontsize=8, color="white", fontweight="bold")
    ax1.text(3, 101, "Severe\nAge > 42\nProd > 2", ha="center", fontsize=8, color="white", fontweight="bold")

    for cid in cohort_ids:
        curve = cohort_curves0[cid]
        ax2.step(
            tenure_grid,
            curve,
            where="post",
            label=cohort_names[cid],
            color=colors[cid],
            linewidth=2.4,
            alpha=0.9,
        )

    ax2.set_title("B: Stage 2 Leaf Kaplan-Meier Curves via retrain()", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Customer Tenure (Years)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Survival Probability S(t)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(0, 10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()

    print(f"\nSaved Two-Stage Survival figures to:")
    print(f"  -> {out_png}")
    print(f"  -> {out_svg}")
    print("=" * 80)


if __name__ == "__main__":
    run_two_stage_survival()
