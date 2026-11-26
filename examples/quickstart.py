"""mltreelib Quickstart: Fit a ModelTree on real benchmark data in seconds.

Demonstrates why piecewise linear model trees outperform standard CART
step functions on continuous tabular dynamics (Yeh 1998 Concrete Compressive Strength).

Usage:
    uv run python examples/quickstart.py
    (optional: add `--with pygraphviz` after `uv run` to also render the tree as SVG; needs Graphviz)
"""

import numpy as np
from mltreelib import MetricsReport, ModelTree
from mltreelib.data import Dataset


def main():
    # 1. Load standard real benchmark dataset (1,030 industrial concrete mixtures)
    ds = Dataset.load(
        "concrete_compressive_strength", target="ConcreteCompressiveStrength"
    )

    # 80/20 train/test split, no shuffling: first 80% of rows train, last 20% test
    train, test = ds.split(test_size=0.2, time=True)

    print(f"Loaded Concrete Compressive Strength: {ds.n_rows} rows, {len(ds.names)} features.")
    print(f"Train rows: {train.n_rows}, Test rows: {test.n_rows}\n")

    # 2. Fit standard CART decision tree (mean leaves, depth=3)
    cart = ModelTree(leaf="mean", max_depth=3, min_rows=20)
    cart.fit(train)
    pred_cart = cart.predict(test)

    # 3. Fit mltreelib ModelTree (piecewise linear ridge leaves, depth=3)
    model_tree = ModelTree(leaf="linear", max_depth=3, min_rows=20, l2=1e-3)
    model_tree.fit(train)
    pred_mt = model_tree.predict(test)

    # 4. Evaluate metrics
    m_cart = MetricsReport("regression", y_true=test.y, y_pred=pred_cart)
    m_mt = MetricsReport("regression", y_true=test.y, y_pred=pred_mt)

    print("=== Model Performance on Test Set ===")
    print(f"Standard Decision Tree (CART)  : RMSE = {m_cart['rmse']:.2f} MPa, R2 = {m_cart['r2']:.3f}, MAE = {m_cart['mae']:.2f} MPa")
    print(f"mltreelib ModelTree (Linear)   : RMSE = {m_mt['rmse']:.2f} MPa, R2 = {m_mt['r2']:.3f}, MAE = {m_mt['mae']:.2f} MPa")
    rmse_reduction = (1.0 - m_mt['rmse'] / m_cart['rmse']) * 100.0
    print(f"--> Error reduction with linear leaves: {rmse_reduction:.1f}%\n")

    # 5. Display rpart-style transparent tree structure
    print("=== Fitted ModelTree Structure (rpart-style) ===")
    print(model_tree)

    # 6. Render Graphviz DOT
    dot_str = model_tree.to_dot()
    try:
        import pygraphviz as pgv
        pgv.AGraph(string=dot_str).draw("docs/assets/quickstart_tree.svg", prog="dot")
        print("Rendered tree graph to: docs/assets/quickstart_tree.svg")
    except ImportError:
        with open("docs/assets/quickstart_tree.dot", "w") as f:
            f.write(dot_str)
        print("Saved DOT representation to: docs/assets/quickstart_tree.dot")

    # 7. Generate publication comparison plot (CART step function vs ModelTree piecewise linear)
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Palette
        c_cart = "#f4511e"   # Deep Orange
        c_mt = "#00897b"     # Teal
        c_ideal = "#263238"  # Dark Slate

        # Panel 1: Parity Plot (Actual vs Predicted)
        ax1.scatter(test.y, pred_cart, color=c_cart, alpha=0.55, s=36, label=f"CART Step (RMSE: {m_cart['rmse']:.2f}, R²: {m_cart['r2']:.3f})")
        ax1.scatter(test.y, pred_mt, color=c_mt, alpha=0.75, s=36, edgecolors="white", linewidth=0.5, label=f"ModelTree Linear (RMSE: {m_mt['rmse']:.2f}, R²: {m_mt['r2']:.3f})")
        min_v = min(min(test.y), min(pred_mt), min(pred_cart))
        max_v = max(max(test.y), max(pred_mt), max(pred_cart))
        ax1.plot([min_v, max_v], [min_v, max_v], linestyle="--", color=c_ideal, alpha=0.7, label="Perfect Parity (1:1)")
        ax1.set_title("Actual vs Predicted Compressive Strength (MPa)", fontsize=13, fontweight="bold", color=c_ideal, pad=12)
        ax1.set_xlabel("Observed Strength (MPa)", fontsize=11, color=c_ideal)
        ax1.set_ylabel("Predicted Strength (MPa)", fontsize=11, color=c_ideal)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=10)

        # Panel 2: Continuous Response Profile across Cement Content
        cement_idx = ds.names.index("Cement")
        cement_vals = [row[cement_idx] for row in train.x]
        c_grid = np.linspace(min(cement_vals), max(cement_vals), 200)

        # Baseline median vector for other ingredients
        medians = [float(np.median([row[j] for row in train.x])) for j in range(len(ds.names))]
        X_grid = []
        for c in c_grid:
            row = list(medians)
            row[cement_idx] = float(c)
            X_grid.append(row)

        cart_grid_preds = cart.predict(X_grid)
        mt_grid_preds = model_tree.predict(X_grid)

        cement_test = [row[cement_idx] for row in test.x]
        ax2.scatter(cement_test, test.y, color="#78909c", alpha=0.45, s=28, label="Test Observations")
        ax2.step(c_grid, cart_grid_preds, where="post", color=c_cart, linewidth=2.4, alpha=0.9, label="CART Step Function (Staircase, Depth 3)")
        ax2.plot(c_grid, mt_grid_preds, color=c_mt, linewidth=2.6, label="ModelTree Piecewise Linear (Depth 3)")
        ax2.set_title("Response Profile vs Cement Content (kg/m³)", fontsize=13, fontweight="bold", color=c_ideal, pad=12)
        ax2.set_xlabel("Cement Content (kg/m³)", fontsize=11, color=c_ideal)
        ax2.set_ylabel("Compressive Strength (MPa)", fontsize=11, color=c_ideal)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(frameon=True, facecolor="#eceff1", edgecolor="#cfd8dc", fontsize=10)

        plt.suptitle("mltreelib Quickstart: Piecewise Linear ModelTree vs CART", fontsize=15, fontweight="bold", color=c_ideal, y=0.98)
        plt.tight_layout()
        out_path = "docs/assets/quickstart_tree_vs_flat.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated comparison figure: {out_path}")
    except ImportError:
        print("Note: matplotlib not found; install mltreelib[plotting] to generate comparison figure.")


if __name__ == "__main__":
    main()
