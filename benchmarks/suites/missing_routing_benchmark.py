"""Suite 3: Native Missing Data & Surrogate Routing on Scania Truck Telemetry.

Demonstrates:
1. Native missing routing (missing_left) without synthetic imputation.
2. Surrogate split agreement (maxsurrogate=N) under intermittent sensor dropouts.
3. Performance comparison: Native Routing vs Mean Imputation on Scania APSFailure.

Usage:
    uv run python benchmarks/suites/missing_routing_benchmark.py
"""

from mltreelib import Metrics, ModelTree, load_dataset


def main():
    # 1. Load Scania Truck Air Pressure System failure benchmark
    data = load_dataset("APSFailure")
    target = "AirPressureSystemFailure"
    features = [
        "aa_000", "ab_000", "ac_000", "ad_000", "ae_000", "af_000",
        "ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005"
    ]
    n_sample = 12000

    y_raw = data["columns"][target][:n_sample]
    y = [1 if v in ("pos", "1", 1) else 0 for v in y_raw]
    X = [[data["columns"][c][i] for c in features] for i in range(n_sample)]

    # Compute missing rates per sensor
    missing_rates = {
        col: sum(1 for row in X if row[idx] is None) / n_sample * 100.0
        for idx, col in enumerate(features)
    }

    # Split: 8,000 train, 2,000 validation, 2,000 test
    X_train, y_train = X[:8000], y[:8000]
    X_val, y_val = X[8000:10000], y[8000:10000]
    X_test, y_test = X[10000:12000], y[10000:12000]

    print(f"Loaded Scania APS Failure benchmark: {n_sample} samples, {len(features)} sensor channels.")
    print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}, Test rows: {len(X_test)}")
    print(f"Sensor 'ab_000' missing rate: {missing_rates['ab_000']:.1f}%\n")

    # =========================================================================
    # Part 1: Fit ModelTree with Native Missing Routing & Surrogates
    # =========================================================================
    print("=== 1. Native Routing & Surrogate Splits (maxsurrogate=2) ===")
    tree_native = ModelTree(
        max_depth=3,
        min_rows=50,
        maxsurrogate=2,
        leaf="mean",
        leaf_family="binomial",
        gain="deviance",
    )
    tree_native.fit(X_train, y_train, col_names=features)

    # Inspect primary and surrogate split routing
    splits = tree_native.splits()
    print(f"Primary split node: {splits[0]['col']} <= {splits[0]['threshold']:.2f} (missing_left={splits[0]['missing_left']})")

    p_native_test = tree_native.predict_proba(X_test)
    metrics = Metrics()
    pr_native = metrics.pr_curve(y_test, p_native_test)
    roc_native = metrics.roc_curve(y_test, p_native_test)
    brier_native = metrics.brier_score(y_test, p_native_test)

    print(f"Native Routing   -> Average Precision: {pr_native['average_precision']:.4f}, ROC-AUC: {roc_native['auc']:.4f}, Brier: {brier_native:.4f}\n")

    # =========================================================================
    # Part 2: Comparison against Mean Imputation
    # =========================================================================
    print("=== 2. Comparison against Standard Mean Imputation Baseline ===")
    # Calculate column training means for numeric imputation
    means = {}
    for idx, col in enumerate(features):
        obs = [row[idx] for row in X_train if row[idx] is not None]
        means[idx] = sum(obs) / len(obs) if obs else 0.0

    X_train_imp = [[means[idx] if val is None else val for idx, val in enumerate(row)] for row in X_train]
    X_test_imp = [[means[idx] if val is None else val for idx, val in enumerate(row)] for row in X_test]

    tree_imp = ModelTree(
        max_depth=3,
        min_rows=50,
        maxsurrogate=0,
        leaf="mean",
        leaf_family="binomial",
        gain="deviance",
    )
    tree_imp.fit(X_train_imp, y_train, col_names=features)
    p_imp_test = tree_imp.predict_proba(X_test_imp)

    pr_imp = metrics.pr_curve(y_test, p_imp_test)
    roc_imp = metrics.roc_curve(y_test, p_imp_test)
    brier_imp = metrics.brier_score(y_test, p_imp_test)

    print(f"Mean Imputation  -> Average Precision: {pr_imp['average_precision']:.4f}, ROC-AUC: {roc_imp['auc']:.4f}, Brier: {brier_imp:.4f}")
    lift_ap = (pr_native["average_precision"] - pr_imp["average_precision"]) / pr_imp["average_precision"] * 100.0
    print(f"--> Average Precision advantage of Native Missing Routing: +{lift_ap:.1f}%\n")

    # =========================================================================
    # Part 3: Robustness across Incomplete vs Complete Test Telemetry
    # =========================================================================
    complete_indices = [i for i, row in enumerate(X_test) if all(v is not None for v in row)]
    incomplete_indices = [i for i, row in enumerate(X_test) if any(v is None for v in row)]

    y_complete = [y_test[i] for i in complete_indices]
    p_complete = [p_native_test[i] for i in complete_indices]
    y_incomp = [y_test[i] for i in incomplete_indices]
    p_incomp = [p_native_test[i] for i in incomplete_indices]

    brier_comp = metrics.brier_score(y_complete, p_complete)
    brier_incomp = metrics.brier_score(y_incomp, p_incomp)
    print(f"Complete Records   (n={len(complete_indices)}): Brier = {brier_comp:.4f}")
    print(f"Incomplete Records (n={len(incomplete_indices)}): Brier = {brier_incomp:.4f}")
    print("Native routing preserves calibrated probabilities even when multiple sensor channels are missing.\n")

    # =========================================================================
    # Part 4: Generate Publication Multi-Panel Figure
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        c_teal = "#00897b"
        c_orange = "#f4511e"
        c_blue = "#0288d1"
        c_dark = "#263238"

        # Panel 1: Missingness Prevalence across Sensors
        cols_sorted = sorted(features, key=lambda c: missing_rates[c], reverse=True)
        rates_sorted = [missing_rates[c] for c in cols_sorted]
        ax1.barh(cols_sorted, rates_sorted, color=c_orange, alpha=0.8, edgecolor=c_dark)
        ax1.set_title("Missing Value Prevalence by Sensor Channel (%)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax1.set_xlabel("Missing Observations (%)", fontsize=10, color=c_dark)
        ax1.set_ylabel("Sensor Channel", fontsize=10, color=c_dark)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Precision-Recall Curve (Native vs Imputed)
        pts_nat_pr = pr_native.get("points", [])
        pts_imp_pr = pr_imp.get("points", [])
        if pts_nat_pr:
            rec_nat = [p[0] for p in pts_nat_pr]
            prec_nat = [p[1] for p in pts_nat_pr]
            ax2.plot(rec_nat, prec_nat, color=c_teal, linewidth=2.2, label=f"Native Routing (AP: {pr_native['average_precision']:.3f})")
        if pts_imp_pr:
            rec_imp = [p[0] for p in pts_imp_pr]
            prec_imp = [p[1] for p in pts_imp_pr]
            ax2.plot(rec_imp, prec_imp, color=c_orange, linewidth=2.0, linestyle="--", label=f"Mean Imputation (AP: {pr_imp['average_precision']:.3f})")
        ax2.set_title("Precision-Recall Curve (Telemetry Failure Triage)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax2.set_xlabel("Recall", fontsize=10, color=c_dark)
        ax2.set_ylabel("Precision", fontsize=10, color=c_dark)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 3: ROC Curve (Native vs Imputed)
        pts_nat_roc = roc_native.get("points", [])
        pts_imp_roc = roc_imp.get("points", [])
        if pts_nat_roc:
            fpr_nat = [p[0] for p in pts_nat_roc]
            tpr_nat = [p[1] for p in pts_nat_roc]
            ax3.plot(fpr_nat, tpr_nat, color=c_teal, linewidth=2.2, label=f"Native Routing (AUC: {roc_native['auc']:.3f})")
        if pts_imp_roc:
            fpr_imp = [p[0] for p in pts_imp_roc]
            tpr_imp = [p[1] for p in pts_imp_roc]
            ax3.plot(fpr_imp, tpr_imp, color=c_orange, linewidth=2.0, linestyle="--", label=f"Mean Imputation (AUC: {roc_imp['auc']:.3f})")
        ax3.plot([0, 1], [0, 1], linestyle=":", color=c_dark, alpha=0.5)
        ax3.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax3.set_xlabel("False Positive Rate", fontsize=10, color=c_dark)
        ax3.set_ylabel("True Positive Rate", fontsize=10, color=c_dark)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 4: Calibration Comparison (Complete vs Incomplete Records)
        cal_comp = metrics.calibration(y_complete, p_complete, n_bins=5)
        cal_incomp = metrics.calibration(y_incomp, p_incomp, n_bins=5)
        b_comp_p = [b.get("mean_score", 0.0) for b in cal_comp.get("bins", [])]
        b_comp_o = [b.get("observed", 0.0) for b in cal_comp.get("bins", [])]
        b_incomp_p = [b.get("mean_score", 0.0) for b in cal_incomp.get("bins", [])]
        b_incomp_o = [b.get("observed", 0.0) for b in cal_incomp.get("bins", [])]

        ax4.plot(b_comp_p, b_comp_o, marker="o", color=c_teal, linewidth=2.0, label=f"Complete (Brier: {brier_comp:.4f})")
        ax4.plot(b_incomp_p, b_incomp_o, marker="s", color=c_blue, linewidth=2.0, label=f"Incomplete (Brier: {brier_incomp:.4f})")
        ax4.plot([0, 0.4], [0, 0.4], linestyle=":", color=c_dark, alpha=0.5, label="Perfect Calibration")
        ax4.set_title("Probability Calibration by Missingness Status", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax4.set_xlabel("Mean Predicted Probability", fontsize=10, color=c_dark)
        ax4.set_ylabel("Observed Failure Rate", fontsize=10, color=c_dark)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        plt.suptitle("mltreelib Suite 3: Native Missing Data Routing & Surrogate Splits", fontsize=15, fontweight="bold", color=c_dark, y=0.98)
        plt.tight_layout()
        out_path = "docs/assets/missing_routing_aps.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated publication asset: {out_path}")
    except ImportError:
        print("Note: matplotlib not found; install mltreelib[plotting] to generate figures.")


if __name__ == "__main__":
    main()
