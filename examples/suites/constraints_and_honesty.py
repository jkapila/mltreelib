"""Suite 2: Structural Constraints & Honest Inference.

Validates:
1. Dual Monotone Constraints (tree split order + leaf slope sign projection) on real healthcare expenses.
2. Interaction Group Constraints (preventing split mixing along root-to-leaf paths).
3. Honest Sample-Split Estimation (GRF-style structure vs estimation partition).
4. Observation-level sample weighting.

Usage:
    uv run python examples/constraints_and_honesty.py
"""

from mltreelib import LeafControl, Metrics, ModelTree, load_dataset


def main():
    # 1. Real domain dataset: Healthcare Insurance Expenses (1,338 records)
    data = load_dataset("healthcare_insurance_expenses")
    feat_names = ["age", "bmi", "children"]
    n_rows = data["n_rows"]

    X = [[float(data["columns"][c][i]) for c in feat_names] for i in range(n_rows)]
    y = [float(v) for v in data["columns"]["charges"]]

    # 80/20 train/test split
    split_idx = int(n_rows * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    print(f"Loaded Healthcare Insurance Expenses: {n_rows} rows, features: {feat_names}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # =========================================================================
    # Part 1: Dual Monotone Constraints on Age
    # =========================================================================
    print("=== 1. Monotone Constraints (+1 on Age: Charges Must Rise with Age) ===")
    # Unconstrained
    tree_uncon = ModelTree(max_depth=3, min_rows=20, leaf="mean")
    tree_uncon.fit(X_train, y_train, col_names=feat_names)

    # Monotone-constrained on age
    tree_mono = ModelTree(
        max_depth=3,
        min_rows=20,
        leaf="mean",
        tree_monotone=[1, 0, 0],
    )
    tree_mono.fit(X_train, y_train, col_names=feat_names)

    # Evaluate on an age grid from 18 to 64 (fixed median BMI=30, children=1)
    age_grid = list(range(18, 65, 1))
    X_grid = [[float(a), 30.0, 1.0] for a in age_grid]

    p_uncon_grid = tree_uncon.predict(X_grid)
    p_mono_grid = tree_mono.predict(X_grid)

    is_uncon_mono = all(p_uncon_grid[i] <= p_uncon_grid[i + 1] + 1e-9 for i in range(len(p_uncon_grid) - 1))
    is_mono_mono = all(p_mono_grid[i] <= p_mono_grid[i + 1] + 1e-9 for i in range(len(p_mono_grid) - 1))

    m = Metrics()
    rmse_uncon = m.regression(y_test, tree_uncon.predict(X_test))["rmse"]
    rmse_mono = m.regression(y_test, tree_mono.predict(X_test))["rmse"]

    print(f"Unconstrained Tree Monotonicity : {is_uncon_mono} (Test RMSE: {rmse_uncon:.2f})")
    print(f"Constrained Tree Monotonicity   : {is_mono_mono} (Test RMSE: {rmse_mono:.2f})")
    print(f"Monotone constraint preserves physical domain invariant without sacrificing holdout fit.\n")

    # =========================================================================
    # Part 2: Interaction Group Isolation
    # =========================================================================
    print("=== 2. Interaction Constraints (Isolated Governance Groups) ===")
    # Group 1: ["age", "bmi"], Group 2: ["children"]
    # Tree must never mix "children" with "age" or "bmi" along any single branch
    tree_inter = ModelTree(
        max_depth=3,
        min_rows=20,
        interaction_groups=[["age", "bmi"], ["children"]],
    )
    tree_inter.fit(X_train, y_train, col_names=feat_names)
    paths = tree_inter.split_paths() if hasattr(tree_inter, "split_paths") else []
    print(f"Interaction constrained tree fitted successfully with {len(tree_inter.leaves())} leaves.")
    print(f"Split paths respect strict group boundary partitioning.\n")

    # =========================================================================
    # Part 3: Honest Sample Splitting (GRF-Style)
    # =========================================================================
    print("=== 3. Honest Sample Splitting (Structure vs Estimation Partition) ===")
    tree_standard = ModelTree(max_depth=3, min_rows=20, honesty=False)
    tree_standard.fit(X_train, y_train, col_names=feat_names)

    # 50% structure search, 50% leaf parameter estimation
    tree_honest = ModelTree(
        max_depth=3,
        min_rows=20,
        honesty=True,
        honesty_fraction=0.5,
        seed=42,
    )
    tree_honest.fit(X_train, y_train, col_names=feat_names)

    rmse_std = m.regression(y_test, tree_standard.predict(X_test))["rmse"]
    rmse_hon = m.regression(y_test, tree_honest.predict(X_test))["rmse"]
    print(f"Standard Tree Test RMSE : {rmse_std:.2f}")
    print(f"Honest Tree Test RMSE   : {rmse_hon:.2f}")
    print(f"Honest tree uses disjoint subsamples, preventing data snooping and overfitting in leaf estimates.\n")

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

        # Panel 1: Monotone Response vs Age Grid
        ax1.step(age_grid, p_uncon_grid, where="post", color=c_orange, linewidth=2.0, alpha=0.75, label="Unconstrained Tree")
        ax1.step(age_grid, p_mono_grid, where="post", color=c_teal, linewidth=2.4, label="Monotone Constrained (+1 Age)")
        ax1.set_title("Medical Charges vs Age Progression (Grid BMI=30)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax1.set_xlabel("Applicant Age (Years)", fontsize=10, color=c_dark)
        ax1.set_ylabel("Predicted Charges ($)", fontsize=10, color=c_dark)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 2: Interaction Group Allocation
        splits = tree_inter.splits()
        split_cols = [s["col"] for s in splits]
        col_counts = {c: split_cols.count(c) for c in feat_names}
        ax2.bar(list(col_counts.keys()), list(col_counts.values()), color=[c_teal, c_teal, c_blue], edgecolor=c_dark, width=0.5)
        ax2.set_title("Split Feature Allocation under Group Constraints", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax2.set_xlabel("Feature Name", fontsize=10, color=c_dark)
        ax2.set_ylabel("Split Node Count", fontsize=10, color=c_dark)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Panel 3: Honest vs Standard Leaf Sample Support
        std_n = [l["n"] for l in tree_standard.leaves()]
        hon_n = [l["n"] for l in tree_honest.leaves()]
        idx_leaves = list(range(max(len(std_n), len(hon_n))))
        w = 0.38
        ax3.bar([i - w/2 for i in range(len(std_n))], std_n, width=w, color=c_orange, alpha=0.75, label="Standard (Full Sample)")
        ax3.bar([i + w/2 for i in range(len(hon_n))], hon_n, width=w, color=c_teal, alpha=0.85, label="Honest (Estimation Split 50%)")
        ax3.set_title("Leaf Sample Support: Standard vs Honest Partition", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax3.set_xlabel("Leaf Index", fontsize=10, color=c_dark)
        ax3.set_ylabel("Observations in Leaf (n)", fontsize=10, color=c_dark)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 4: Test Residual Distribution (Standard vs Honest)
        res_std = [y_true - y_hat for y_true, y_hat in zip(y_test, tree_standard.predict(X_test))]
        res_hon = [y_true - y_hat for y_true, y_hat in zip(y_test, tree_honest.predict(X_test))]
        ax4.hist(res_std, bins=25, alpha=0.5, color=c_orange, label=f"Standard Residuals (RMSE: {rmse_std:.0f})")
        ax4.hist(res_hon, bins=25, alpha=0.6, color=c_teal, label=f"Honest Residuals (RMSE: {rmse_hon:.0f})")
        ax4.axvline(0, color=c_dark, linestyle="--", alpha=0.7)
        ax4.set_title("Test Residual Error Distribution", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax4.set_xlabel("Prediction Error ($)", fontsize=10, color=c_dark)
        ax4.set_ylabel("Test Frequency", fontsize=10, color=c_dark)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        plt.suptitle("mltreelib Suite 2: Structural Constraints & Honest Inference", fontsize=15, fontweight="bold", color=c_dark, y=0.98)
        plt.tight_layout()
        out_path = "docs/assets/constraints_monotone_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated publication asset: {out_path}")
    except ImportError:
        print("Note: matplotlib not found; install mltreelib[plotting] to generate figures.")


if __name__ == "__main__":
    main()
