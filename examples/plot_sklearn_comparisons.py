"""Generate scikit-learn style benchmark comparison figures using matplotlib.

Builds on:
1. https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
   Comparing decision boundaries across 2D datasets:
   - Input data
   - Scikit-Learn DecisionTree
   - Scikit-Learn RandomForest
   - LightGBM
   - CatBoost
   - mltreelib ModelTree (Linear Leaf)
   - mltreelib ModelTree (Mean Leaf)

2. https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
   Comparing 1D continuous regression curves:
   - Ground Truth
   - Scikit-Learn DecisionTree (Step function)
   - LightGBM (Ensemble step function)
   - CatBoost (Symmetric tree steps)
   - mltreelib ModelTree (Piecewise linear)

Outputs both PNG and SVG to docs/assets/.
"""

import sys
from pathlib import Path
import numpy as np

# Use Agg backend for headless generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

import mltreelib as mt
from mltreelib import ModelTree

DOCS_ASSETS = Path("docs/assets")
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)


def plot_classifier_comparison():
    print("Generating 2D Classifier Comparison (scikit-learn style)...")

    # Classifiers to compare
    names = [
        "Decision Tree",
        "Random Forest",
        "LightGBM",
        "CatBoost",
        "ModelTree (Linear)",
        "ModelTree (Mean)",
    ]

    classifiers = [
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=50, random_state=42),
        lgb.LGBMClassifier(max_depth=5, n_estimators=50, random_state=42, verbosity=-1),
        CatBoostClassifier(depth=5, iterations=50, random_seed=42, verbose=0),
        ModelTree(leaf="linear", leaf_family="binomial", gain="deviance", max_depth=3, seed=42),
        ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=3, seed=42),
    ]

    # Generate 3 benchmark datasets
    X_lin, y_lin = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X_lin += 2 * rng.uniform(size=X_lin.shape)
    linearly_separable = (X_lin, y_lin)

    datasets = [
        ("Moons", make_moons(n_samples=250, noise=0.25, random_state=42)),
        ("Circles", make_circles(n_samples=250, noise=0.15, factor=0.5, random_state=42)),
        ("Linear Separable", linearly_separable),
    ]

    figure = plt.figure(figsize=(21, 9))
    h = 0.03  # step size in the mesh

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#EF4444", "#3B82F6"])

    i = 1
    # Iterate over datasets
    for ds_cnt, (ds_name, ds) in enumerate(datasets):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_pts = np.c_[xx.ravel(), yy.ravel()]

        # Plot the input dataset first
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data", fontsize=12, fontweight="bold")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=0.6)
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.9)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylabel(ds_name, fontsize=12, fontweight="bold")
        i += 1

        # Iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            # Fit classifier
            if isinstance(clf, ModelTree):
                clf.fit(X_train.tolist(), [float(v) for v in y_train])
                probs_test = clf.predict_proba(X_test.tolist())
                score = np.mean((np.array(probs_test) >= 0.5).astype(int) == y_test)
                probs_grid = clf.predict_proba(grid_pts.tolist())
                Z = np.array(probs_grid).reshape(xx.shape)
            else:
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                if hasattr(clf, "predict_proba"):
                    Z = clf.predict_proba(grid_pts)[:, 1]
                else:
                    Z = clf.predict(grid_pts)
                Z = Z.reshape(xx.shape)

            # Plot the decision boundary
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.75, levels=np.linspace(0, 1, 21))

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=0.4, s=20
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.9, s=25
            )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name, fontsize=11, fontweight="bold")
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                f"{score:.2f}",
                size=14,
                fontweight="bold",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
            i += 1

    plt.tight_layout()
    out_png = DOCS_ASSETS / "plot_classifier_comparison.png"
    out_svg = DOCS_ASSETS / "plot_classifier_comparison.svg"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"  [Output] Saved {out_png} and {out_svg}")


def plot_tree_regression_comparison():
    print("\nGenerating 1D Tree Regression Comparison (scikit-learn style)...")

    # Create a non-linear continuous physics dataset (Sine with decay + noise)
    rng = np.random.RandomState(42)
    X = np.sort(5 * rng.rand(160, 1), axis=0)
    y = np.sin(X).ravel() + 0.2 * np.cos(3 * X).ravel() + rng.normal(0, 0.15, X.shape[0])
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_true = np.sin(X_test).ravel() + 0.2 * np.cos(3 * X_test).ravel()

    # Fit models
    regr_dt_shallow = DecisionTreeRegressor(max_depth=2, random_state=42)
    regr_dt_deep = DecisionTreeRegressor(max_depth=5, random_state=42)
    regr_lgb = lgb.LGBMRegressor(n_estimators=30, max_depth=3, random_state=42, verbosity=-1)
    regr_catb = CatBoostRegressor(iterations=30, depth=3, random_seed=42, verbose=0)
    regr_mt = ModelTree(leaf="linear", max_depth=2, min_rows=15, l2=1e-3, seed=42)

    regr_dt_shallow.fit(X, y)
    regr_dt_deep.fit(X, y)
    regr_lgb.fit(X, y)
    regr_catb.fit(X, y)
    regr_mt.fit(X.tolist(), y.tolist())

    # Predict
    y_dt_shallow = regr_dt_shallow.predict(X_test)
    y_dt_deep = regr_dt_deep.predict(X_test)
    y_lgb = regr_lgb.predict(X_test)
    y_catb = regr_catb.predict(X_test)
    y_mt = regr_mt.predict(X_test.tolist())

    # Calculate test RMSE
    rmse_dt_shallow = np.sqrt(np.mean((y_true - y_dt_shallow) ** 2))
    rmse_dt_deep = np.sqrt(np.mean((y_true - y_dt_deep) ** 2))
    rmse_lgb = np.sqrt(np.mean((y_true - y_lgb) ** 2))
    rmse_catb = np.sqrt(np.mean((y_true - y_catb) ** 2))
    rmse_mt = np.sqrt(np.mean((y_true - y_mt) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Panel 1: Decision Tree (Step Function Staircase) vs ModelTree
    ax = axes[0]
    ax.scatter(X, y, s=25, edgecolor="black", c="#94A3B8", alpha=0.7, label="Actual Data ($N=160$)")
    ax.plot(X_test, y_true, color="#64748B", linestyle="--", linewidth=1.8, label="True Underlying Curve")
    ax.plot(X_test, y_dt_shallow, color="#EF4444", linewidth=2.2, label=f"DecisionTree (Depth 2, RMSE={rmse_dt_shallow:.3f})")
    ax.plot(X_test, y_dt_deep, color="#F59E0B", linewidth=1.8, linestyle=":", label=f"DecisionTree (Depth 5, RMSE={rmse_dt_deep:.3f})")
    ax.plot(X_test, y_mt, color="#2563EB", linewidth=2.5, label=f"ModelTree Linear (Depth 2, RMSE={rmse_mt:.3f})")
    ax.set_xlabel("Feature $x$", fontsize=12)
    ax.set_ylabel("Target $y$", fontsize=12)
    ax.set_title("Step Staircase (CART) vs Piecewise Linear (ModelTree)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Ensembles (GBDT / CatBoost) vs ModelTree
    ax = axes[1]
    ax.scatter(X, y, s=25, edgecolor="black", c="#94A3B8", alpha=0.7, label="Actual Data ($N=160$)")
    ax.plot(X_test, y_true, color="#64748B", linestyle="--", linewidth=1.8, label="True Underlying Curve")
    ax.plot(X_test, y_lgb, color="#10B981", linewidth=2.2, label=f"LightGBM 30 Trees (RMSE={rmse_lgb:.3f})")
    ax.plot(X_test, y_catb, color="#8B5CF6", linewidth=2.0, linestyle="-.", label=f"CatBoost 30 Trees (RMSE={rmse_catb:.3f})")
    ax.plot(X_test, y_mt, color="#2563EB", linewidth=2.5, label=f"ModelTree 1 Single Tree (RMSE={rmse_mt:.3f})")
    ax.set_xlabel("Feature $x$", fontsize=12)
    ax.set_ylabel("Target $y$", fontsize=12)
    ax.set_title("GBDT Ensemble Steps vs Single ModelTree", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_png = DOCS_ASSETS / "plot_tree_regression_comparison.png"
    out_svg = DOCS_ASSETS / "plot_tree_regression_comparison.svg"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"  [Output] Saved {out_png} and {out_svg}")


if __name__ == "__main__":
    plot_classifier_comparison()
    plot_tree_regression_comparison()
    print("\nAll matplotlib benchmark plots generated successfully.")
