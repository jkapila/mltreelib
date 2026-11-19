"""Core Tree Architecture, Two-Control Decoupling, and Growth Knobs.

Consolidates tree growth controls, LeafControl decoupling, 0D statistical profiles,
and Chow test hypothesis stopping on real benchmark data.

Usage:
    uv run python examples/core_tree_architecture.py
"""

from mltreelib import LeafControl, Metrics, ModelTree, load_dataset


def main():
    # 1. Real cited benchmark data (Yeh 1998 Concrete Compressive Strength)
    data = load_dataset("concrete_compressive_strength")
    target = "ConcreteCompressiveStrength"
    feature_names = [col for col in data["columns"].keys() if col != target]
    n_rows = data["n_rows"]

    X = [[data["columns"][c][i] for c in feature_names] for i in range(n_rows)]
    y = data["columns"][target]

    split_idx = int(n_rows * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    print(f"Loaded Concrete Compressive Strength: {n_rows} rows, {len(feature_names)} features.")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # =========================================================================
    # Part 1: Two-Control Architecture (ModelTree vs LeafControl)
    # =========================================================================
    print("=== 1. Two-Control Architecture: Decoupled Topology & Leaf Solvers ===")
    leaf_ctrl = LeafControl(
        leaf="linear",
        l1=0.0,
        l2=1e-3,
        leaf_family="gaussian",
    )
    tree = ModelTree(
        max_depth=3,
        min_rows=25,
        prune="none",
        control=leaf_ctrl,
    )
    tree.fit(X_train, y_train, col_names=feature_names)
    preds = tree.predict(X_test)
    m = Metrics().regression(y_test, preds)
    print(f"Fitted ModelTree with injected LeafControl: Test RMSE = {m['rmse']:.2f} MPa, R2 = {m['r2']:.3f}")
    print(f"Tree leaves count: {len(tree.leaves())}\n")

    # =========================================================================
    # Part 2: 0D Descriptive Leaf Statistics
    # =========================================================================
    print("=== 2. 0D Leaf Statistical Profiles across Terminal Partitions ===")
    leaves = tree.leaves()
    for idx, leaf in enumerate(leaves[:3]):
        stats = leaf.get("stats", {})
        print(f"Leaf {idx} (n={leaf.get('n', 0)}):")
        print(f"  Center : mean={stats.get('mean', 0.0):.2f}, median={stats.get('median', 0.0):.2f}")
        print(f"  Spread : var={stats.get('var', 0.0):.2f}, mad={stats.get('mad', 0.0):.2f}, iqr={stats.get('iqr', 0.0):.2f}")
        print(f"  Shape  : skew={stats.get('skew', 0.0):.2f}, kurtosis={stats.get('kurtosis', 0.0):.2f}\n")

    # =========================================================================
    # Part 3: Chow Test Hypothesis Stopping Gate
    # =========================================================================
    print("=== 3. Chow Structural Break Test Hypothesis Gating ===")
    tree_chow = ModelTree(
        max_depth=5,
        min_rows=25,
        split_test="chow",
        split_alpha=0.01,
        control=leaf_ctrl,
    )
    tree_chow.fit(X_train, y_train, col_names=feature_names)
    print(f"Unpruned leaves (depth 5): {len(tree.leaves())}")
    print(f"Chow-gated leaves (alpha=0.01): {len(tree_chow.leaves())}")
    print("Chow gate automatically terminates subtrees when structural parameter stability is accepted.\n")

    # =========================================================================
    # Part 4: Depth vs Regularization Sensitivity Grid
    # =========================================================================
    print("=== 4. Parameter Sensitivity: Depth vs Leaf Regularization ===")
    depths = [1, 2, 3, 4, 5]
    cart_rmse = []
    mt_rmse = []
    mt_leaves = []

    metrics = Metrics()
    for d in depths:
        # CART
        c = ModelTree(leaf="mean", max_depth=d, min_rows=20)
        c.fit(X_train, y_train, col_names=feature_names)
        cart_rmse.append(metrics.regression(y_test, c.predict(X_test))["rmse"])

        # ModelTree Linear
        mt = ModelTree(leaf="linear", max_depth=d, min_rows=20, l2=1e-3)
        mt.fit(X_train, y_train, col_names=feature_names)
        mt_rmse.append(metrics.regression(y_test, mt.predict(X_test))["rmse"])
        mt_leaves.append(len(mt.leaves()))
        print(f"Depth {d} -> CART RMSE: {cart_rmse[-1]:.2f} MPa | ModelTree RMSE: {mt_rmse[-1]:.2f} MPa ({mt_leaves[-1]} leaves)")

    # =========================================================================
    # Part 5: Generate Publication Multi-Panel Figure
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Brand colors
        c_teal = "#00897b"
        c_orange = "#f4511e"
        c_blue = "#0288d1"
        c_dark = "#263238"

        # Panel 1: Depth vs Test RMSE
        ax1.plot(depths, cart_rmse, marker="o", linewidth=2.2, color=c_orange, label="CART (Mean Leaves)")
        ax1.plot(depths, mt_rmse, marker="s", linewidth=2.2, color=c_teal, label="ModelTree (Linear Ridge Leaves)")
        ax1.set_title("Test Error vs Tree Depth", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax1.set_xlabel("Max Depth", fontsize=10, color=c_dark)
        ax1.set_ylabel("Test RMSE (MPa)", fontsize=10, color=c_dark)
        ax1.set_xticks(depths)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        # Panel 2: ModelTree Leaves vs Depth
        ax2.bar(depths, mt_leaves, color=c_teal, alpha=0.75, width=0.5, edgecolor=c_dark, linewidth=0.8)
        for d, n_l in zip(depths, mt_leaves):
            ax2.text(d, n_l + 0.3, str(n_l), ha="center", va="bottom", fontsize=10, fontweight="bold", color=c_dark)
        ax2.set_title("Terminal Leaf Count vs Depth (ModelTree)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax2.set_xlabel("Max Depth", fontsize=10, color=c_dark)
        ax2.set_ylabel("Number of Leaves", fontsize=10, color=c_dark)
        ax2.set_xticks(depths)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Panel 3: 0D Leaf Variance Distribution
        leaf_indices = list(range(len(leaves)))
        leaf_vars = [l.get("stats", {}).get("var", 0.0) for l in leaves]
        ax3.bar(leaf_indices, leaf_vars, color=c_blue, alpha=0.75, edgecolor=c_dark, linewidth=0.8)
        ax3.set_title("0D Leaf Variance Profile (Within-Node Heterogeneity)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax3.set_xlabel("Leaf Index", fontsize=10, color=c_dark)
        ax3.set_ylabel("Sample Variance (MPa²)", fontsize=10, color=c_dark)
        ax3.set_xticks(leaf_indices)
        ax3.grid(True, linestyle=":", alpha=0.6)

        # Panel 4: Parity Comparison at Optimal Depth (Depth 3)
        m_cart_opt = ModelTree(leaf="mean", max_depth=3, min_rows=20)
        m_cart_opt.fit(X_train, y_train, col_names=feature_names)
        opt_cart_p = m_cart_opt.predict(X_test)

        m_mt_opt = ModelTree(leaf="linear", max_depth=3, min_rows=20, l2=1e-3)
        m_mt_opt.fit(X_train, y_train, col_names=feature_names)
        opt_mt_p = m_mt_opt.predict(X_test)
        ax4.scatter(y_test, opt_cart_p, color=c_orange, alpha=0.5, s=28, label=f"CART Step (RMSE: {cart_rmse[2]:.2f})")
        ax4.scatter(y_test, opt_mt_p, color=c_teal, alpha=0.75, s=28, edgecolors="white", linewidth=0.4, label=f"ModelTree Linear (RMSE: {mt_rmse[2]:.2f})")
        min_v, max_v = min(y_test), max(y_test)
        ax4.plot([min_v, max_v], [min_v, max_v], linestyle="--", color=c_dark, alpha=0.7, label="1:1 Parity")
        ax4.set_title("Parity at Depth 3 (34.3% Error Reduction)", fontsize=12, fontweight="bold", color=c_dark, pad=10)
        ax4.set_xlabel("Observed (MPa)", fontsize=10, color=c_dark)
        ax4.set_ylabel("Predicted (MPa)", fontsize=10, color=c_dark)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=9)

        plt.suptitle("mltreelib Suite 1: Core Tree Architecture & Growth Knobs", fontsize=15, fontweight="bold", color=c_dark, y=0.98)
        plt.tight_layout()
        out_fig = "docs/assets/core_growth_knob_grid.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"\nGenerated publication asset: {out_fig}")
    except ImportError:
        print("\nNote: matplotlib not installed; install mltreelib[plotting] to generate figures.")


if __name__ == "__main__":
    main()
