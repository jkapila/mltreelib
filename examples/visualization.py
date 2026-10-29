"""Visualization for real work: graphviz tree diagrams + diagnostic plots.

Layout and rendering belong to graphviz (pygraphviz wraps the same `dot`
engine the `dot` binary uses) - we only emit DOT. Figures land in
docs/assets/ for the documentation pages.

    uv run --with pandas --with pygraphviz --with matplotlib python examples/visualization.py
"""

from pathlib import Path

import pandas as pd

from mltreelib import Metrics, ModelTree, OneHotEncoder

ASSETS = Path("docs/assets")
ASSETS.mkdir(exist_ok=True)

df = pd.read_parquet("datasets/airfoil_self_noise.parquet")
y = df["scaled-sound-pressure"].astype(float).tolist()
x = df.drop(columns=["scaled-sound-pressure"]).copy()
for col in x.select_dtypes(include="category").columns:
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col].astype(object)])
    dummies = enc.transform([None if v is None else str(v) for v in x[col].astype(object)])
    x = x.drop(columns=[col])
    for level, values in dummies.items():
        x[f"{col}_{level}"] = values
X = [[None if pd.isna(v) else float(v) for v in row] for row in x.values.tolist()]
names = list(x.columns)
n_tr = int(0.8 * len(X))

# depth 3 on purpose: the diagram must read at a glance, not score
model = ModelTree(max_depth=3, min_rows=30)
model.fit(X[:n_tr], y[:n_tr])
pred = model.predict(X[n_tr:])

# --- tree diagram via graphviz -----------------------------------------
import pygraphviz as pgv

dot = model.to_dot()
(ASSETS / "airfoil_tree.dot").write_text(dot)
g = pgv.AGraph(string=dot)
g.layout(prog="dot")
g.draw(str(ASSETS / "airfoil_tree.svg"))
print(f"tree diagram: {ASSETS / 'airfoil_tree.svg'} "
      f"({len(model.leaves())} leaves, depth 3, rendered by graphviz)")

# --- diagnostic plots via matplotlib (dev extra, guarded) ---------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mltreelib import plots as mplots

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sample = list(range(0, len(pred), max(1, len(pred) // 2000)))
    axes[0].scatter([y[n_tr + i] for i in sample], [pred[i] for i in sample],
                    s=4, alpha=0.4, color="#2a7")
    lo = min(min(pred), min(y[n_tr:]))
    hi = max(max(pred), max(y[n_tr:]))
    axes[0].plot([lo, hi], [lo, hi], color="#333", lw=1)
    axes[0].set_xlabel("actual scaled sound pressure")
    axes[0].set_ylabel("predicted")
    axes[0].set_title("airfoil: predicted vs actual (test)")

    resid = [p - t for p, t in zip(pred, y[n_tr:])]
    axes[1].scatter([pred[i] for i in sample], [resid[i] for i in sample],
                    s=4, alpha=0.4, color="#c63")
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("residual")
    axes[1].set_title("residuals")
    fig.tight_layout()
    fig.savefig(ASSETS / "airfoil_diagnostics.png", dpi=110)
    plt.close(fig)
    print(f"diagnostics: {ASSETS / 'airfoil_diagnostics.png'}")

    ax = mplots.plot_importance(model.feature_importance())
    ax.figure.tight_layout()
    ax.figure.savefig(ASSETS / "airfoil_importance.png", dpi=110)
    plt.close(ax.figure)
    print(f"feature importance: {ASSETS / 'airfoil_importance.png'}")

    bd = model.break_down(X[n_tr])
    ax = mplots.plot_breakdown(bd)
    ax.figure.tight_layout()
    ax.figure.savefig(ASSETS / "airfoil_breakdown.png", dpi=110)
    plt.close(ax.figure)
    print(f"breakdown of one test row: {ASSETS / 'airfoil_breakdown.png'}")
except ImportError:
    print("matplotlib not installed - skipping diagnostic plots "
          "(uv run --with matplotlib python examples/visualization.py)")

print(f"test rmse: {Metrics().regression(y[n_tr:], pred)['rmse']:.1f}")
