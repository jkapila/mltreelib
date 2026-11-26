"""Visualization for real work: graphviz tree diagrams + diagnostic plots.

Layout and rendering belong to graphviz (pygraphviz wraps the same `dot`
engine the `dot` binary uses) - we only emit DOT. Figures land in
docs/assets/ for the documentation pages.

    uv run --with pygraphviz python examples/guides/visualization.py
"""

from pathlib import Path

import pandas as pd
from mltreelib import MetricsReport, ModelTree, OneHotEncoder
from mltreelib.data import Dataset

ASSETS = Path("docs/assets")
ASSETS.mkdir(exist_ok=True)

df = pd.read_parquet("datasets/airfoil_self_noise.parquet")
y = df["scaled-sound-pressure"].astype(float).tolist()
x = df.drop(columns=["scaled-sound-pressure"]).copy()
for col in x.select_dtypes(include="category").columns:
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col].astype(object)])
    dummies = enc.transform(
        [None if v is None else str(v) for v in x[col].astype(object)]
    )
    x = x.drop(columns=[col])
    for level, values in dummies.items():
        x[f"{col}_{level}"] = values
ds = Dataset(x.assign(**{"scaled-sound-pressure": y}).to_dict("list"), target="scaled-sound-pressure")  # NaN stays missing

# Deterministic 80/20 split, no shuffling: first 80% of rows train, last 20% test.
train, test = ds.split(test_size=0.2, time=True)

# depth 3 on purpose: the diagram must read at a glance, not score
model = ModelTree(max_depth=3, min_rows=30)
model.fit(train)
pred = model.predict(test)

# --- tree diagram via graphviz -----------------------------------------
import pygraphviz as pgv

dot = model.to_dot()
(ASSETS / "airfoil_tree.dot").write_text(dot)
g = pgv.AGraph(string=dot)
g.layout(prog="dot")
g.draw(str(ASSETS / "airfoil_tree.svg"))
print(
    f"tree diagram: {ASSETS / 'airfoil_tree.svg'} "
    f"({len(model.leaves())} leaves, depth 3, rendered by graphviz)"
)

# --- diagnostic plots via matplotlib (dev extra, guarded) ---------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mltreelib import plots as mplots

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mplots._style_axis(axes[0])
    mplots._style_axis(axes[1])
    sample = list(range(0, len(pred), max(1, len(pred) // 2000)))
    axes[0].scatter(
        [test.y[i] for i in sample],
        [pred[i] for i in sample],
        s=14,
        alpha=0.55,
        color=mplots.PALETTE["primary"],
        zorder=3,
    )
    lo = min(min(pred), min(test.y))
    hi = max(max(pred), max(test.y))
    axes[0].plot(
        [lo, hi], [lo, hi], color=mplots.PALETTE["accent"], lw=1.2, ls="--", zorder=2
    )
    axes[0].set_xlabel("Actual Scaled Sound Pressure")
    axes[0].set_ylabel("Predicted Value")
    axes[0].set_title("Airfoil: Predicted vs. Actual (Test)")

    resid = [p - t for p, t in zip(pred, test.y)]
    axes[1].scatter(
        [pred[i] for i in sample],
        [resid[i] for i in sample],
        s=14,
        alpha=0.55,
        color=mplots.PALETTE["secondary"],
        zorder=3,
    )
    axes[1].axhline(0, color=mplots.PALETTE["accent"], lw=1.2, ls="--", zorder=2)
    axes[1].set_xlabel("Predicted Value")
    axes[1].set_ylabel("Residual (Predicted − Actual)")
    axes[1].set_title("Airfoil: Residuals vs. Fitted")
    fig.tight_layout()
    fig.savefig(ASSETS / "airfoil_diagnostics.png", dpi=110)
    plt.close(fig)
    print(f"diagnostics: {ASSETS / 'airfoil_diagnostics.png'}")

    ax = mplots.plot_importance(model.feature_importance())
    ax.figure.tight_layout()
    ax.figure.savefig(ASSETS / "airfoil_importance.png", dpi=110)
    plt.close(ax.figure)
    print(f"feature importance: {ASSETS / 'airfoil_importance.png'}")

    bd = model.break_down(test.x[0])
    ax = mplots.plot_breakdown(bd)
    ax.figure.tight_layout()
    ax.figure.savefig(ASSETS / "airfoil_breakdown.png", dpi=110)
    plt.close(ax.figure)
    print(f"breakdown of one test row: {ASSETS / 'airfoil_breakdown.png'}")
except ImportError:
    print(
        "matplotlib not installed - skipping diagnostic plots "
        "(uv run --with matplotlib python examples/guides/visualization.py)"
    )

print(f"test rmse: {MetricsReport('regression', y_true=test.y, y_pred=pred)['rmse']:.1f}")
