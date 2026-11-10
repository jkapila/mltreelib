"""Model explanation & evaluation surface showcase on real data.

Demonstrates:
1. Model training on real TabArena benchmark (Bank_Customer_Churn, 10,000 rows).
2. DALEX-style prediction break-down attribution (decision path decomposition).
3. Exact TreeSHAP feature attribution (satisfying efficiency invariant).
4. Three-way feature importance comparison (gain, TreeSHAP, permutation).
5. Comprehensive evaluation metrics & curves (ROC, PR, Calibration, Lift, Gain, KS)
   rendered in a coherent, publication-grade theme.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from mltreelib import Metrics, ModelTree
from mltreelib.datasets import load_dataset
from mltreelib import plots


def main():
    print("=" * 70)
    print("mltreelib: Model Explanation & Evaluation Surface Showcase")
    print("=" * 70)

    # 1. Load real benchmark data
    ds = load_dataset("Bank_Customer_Churn")
    print(f"Loaded dataset: {ds['name']} ({ds['n_rows']} rows, {ds['n_cols']} cols)")

    # Feature columns and target
    feature_cols = [c for c in ds["columns"].keys() if c != "churn"]
    n_rows = ds["n_rows"]

    # Target conversion to float 0.0 / 1.0 ('Yes' -> 1.0, 'No' -> 0.0)
    raw_y = ds["columns"]["churn"]
    y = [1.0 if str(v) in ("Yes", "yes", "1", "1.0", "True", "true") else 0.0 for v in raw_y]

    # Row-major matrix X
    # Convert categorical strings to numeric levels via simple hash or index mapping
    cat_cols = [c for c, dt in ds["dtypes"].items() if dt == "categorical" and c != "churn"]
    col_vocabs = {}
    for c in cat_cols:
        col_vocabs[c] = {v: i for i, v in enumerate(sorted(set(ds["columns"][c])))}

    X = []
    for r in range(n_rows):
        row = []
        for c in feature_cols:
            val = ds["columns"][c][r]
            if c in cat_cols:
                row.append(float(col_vocabs[c].get(val, 0)))
            else:
                row.append(float(val) if val is not None else None)
        X.append(row)

    # Train / test split (80 / 20)
    split_idx = int(0.8 * n_rows)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # 2. Fit ModelTree (binomial probability tree)
    tree = ModelTree(
        max_depth=4,
        min_rows=20,
        leaf="mean",
        leaf_family="binomial",
    )
    tree.fit(X_train, y_train, col_names=feature_cols)
    print(f"Fitted tree topology:\n{tree}\n")

    # 3. DALEX-style break-down attribution on a customer
    sample_idx = 12
    sample_row = X_test[sample_idx]
    actual_churn = y_test[sample_idx]

    bd = tree.break_down(sample_row)
    print("-" * 50)
    print(f"Sample customer #{sample_idx} (Actual Churn: {actual_churn}):")
    print(f"  Baseline (overall train rate): {bd['baseline']:.4f}")
    for col, step in bd["steps"]:
        sign = "+" if step >= 0 else ""
        print(f"    -> Split on {col:18s}: {sign}{step:.4f}")
    print(f"  Residual leaf adjustment     : {bd['residual']:+.4f}")
    print(f"  Final predicted probability  : {bd['prediction']:.4f}")
    check_sum = bd["baseline"] + sum(s for _, s in bd["steps"]) + bd["residual"]
    print(f"  Audit check (telescoping sum): {check_sum:.4f} (diff = {abs(check_sum - bd['prediction']):.2e})")

    # 4. Exact TreeSHAP local attribution
    sh = tree.shapley(sample_row)
    print("-" * 50)
    print("Exact TreeSHAP Values (efficiency invariant: base + sum(phi) == prediction):")
    print(f"  TreeSHAP base value: {sh['base_value']:.4f}")
    total_phi = 0.0
    for col, phi in sorted(sh["values"], key=lambda kv: abs(kv[1]), reverse=True):
        if abs(phi) > 1e-4:
            sign = "+" if phi >= 0 else ""
            print(f"    phi({col:18s}) = {sign}{phi:.4f}")
        total_phi += phi
    print(f"  Sum of all phi attributions : {total_phi:+.4f}")
    print(f"  TreeSHAP prediction check    : {sh['base_value'] + total_phi:.4f} == {sh['prediction']:.4f}")

    # 5. Three-way Feature Importance Comparison
    print("-" * 50)
    print("Feature Importance Comparison (Top 5 features):")
    gain_imp = dict(tree.feature_importance(method="gain"))
    shap_imp = dict(tree.feature_importance(method="shapley", x=X_test[:500]))
    perm_imp = dict(tree.feature_importance(method="permutation", x=X_test, y=y_test, metric="rmse"))

    print(f"{'Feature':<20} | {'Gain Share':<12} | {'Mean |SHAP|':<12} | {'Perm. RMSE Degradation':<20}")
    print("-" * 75)
    for feat in feature_cols[:6]:
        g = gain_imp.get(feat, 0.0)
        s = shap_imp.get(feat, 0.0)
        p = perm_imp.get(feat, 0.0)
        print(f"{feat:<20} | {g:<12.4f} | {s:<12.4f} | {p:<20.4f}")

    # 6. Evaluation Curves on Real Test Set
    test_scores = tree.predict_proba(X_test)
    metrics = Metrics()

    roc_res = metrics.roc_curve(y_test, test_scores)
    pr_res = metrics.pr_curve(y_test, test_scores)
    ks_res = metrics.ks_statistic(y_test, test_scores)

    print("-" * 50)
    print(f"Test Set Metrics:")
    print(f"  ROC AUC            : {roc_res['auc']:.4f}")
    print(f"  Average Precision  : {pr_res['average_precision']:.4f}")
    print(f"  KS Statistic       : {ks_res['ks']:.4f} (at threshold {ks_res['score_at_ks']:.4f})")

    # 7. Render 6-Panel Cohesive Visual Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("mltreelib Model Explanation & Diagnostic Dashboard (Bank Churn)", fontsize=14, fontweight="bold", y=0.98)

    # Panel (0, 0): Break-down waterfall
    plots.plot_breakdown(bd, ax=axes[0, 0])
    axes[0, 0].set_title("Customer #12 Prediction Break-Down")

    # Panel (0, 1): ROC Curve
    plots.plot_roc(y_test, test_scores, ax=axes[0, 1])

    # Panel (0, 2): Precision-Recall Curve
    plots.plot_pr(y_test, test_scores, ax=axes[0, 2])

    # Panel (1, 0): Calibration Curve
    plots.plot_calibration(y_test, test_scores, n_bins=10, ax=axes[1, 0])

    # Panel (1, 1): Cumulative Gains
    plots.plot_cumulative_gain(y_test, test_scores, ax=axes[1, 1])

    # Panel (1, 2): Feature Importance (Gain vs Permutation)
    top_feats = sorted(gain_imp.items(), key=lambda kv: kv[1], reverse=True)[:7]
    plots.plot_importance(top_feats, ax=axes[1, 2])
    axes[1, 2].set_title("Top Feature Importances (Gain Share)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = Path("churn_evaluation_dashboard.png")
    fig.savefig(out_img, dpi=180)
    plt.close(fig)
    print(f"\n[Saved Dashboard Image]: {out_img.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
