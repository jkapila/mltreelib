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

df = pd.read_parquet("datasets/diamonds.parquet")
y = df["price"].astype(float).tolist()
x = df.drop(columns=["price"]).copy()
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

model = ModelTree(max_depth=6, min_rows=200)  # modest: the diagram must fit
model.fit(X[:n_tr], y[:n_tr])
pred = model.predict(X[n_tr:])

# --- tree diagram via graphviz -----------------------------------------
import pygraphviz as pgv

dot = model.to_dot()
(Path("docs/assets") / "diamonds_tree.dot").write_text(dot)
g = pgv.AGraph(string=dot)
g.layout(prog="dot")
g.draw(str(ASSETS / "diamonds_tree.svg"))
print(f"tree diagram: {ASSETS / 'diamonds_tree.svg'} "
      f"({len(model.leaves())} leaves, rendered by graphviz)")

# --- diagnostic plots via matplotlib (dev extra, guarded) ---------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sample = list(range(0, len(pred), max(1, len(pred) // 2000)))
    axes[0].scatter([y[n_tr + i] for i in sample], [pred[i] for i in sample],
                    s=4, alpha=0.4, color="#2a7")
    lo = min(min(pred), min(y[n_tr:]))
    hi = max(max(pred), max(y[n_tr:]))
    axes[0].plot([lo, hi], [lo, hi], color="#333", lw=1)
    axes[0].set_xlabel("actual price")
    axes[0].set_ylabel("predicted")
    axes[0].set_title("diamonds: predicted vs actual (test)")

    resid = [p - t for p, t in zip(pred, y[n_tr:])]
    axes[1].scatter([pred[i] for i in sample], [resid[i] for i in sample],
                    s=4, alpha=0.4, color="#c63")
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("residual")
    axes[1].set_title("residuals")
    fig.tight_layout()
    fig.savefig(ASSETS / "diamonds_diagnostics.png", dpi=110)
    print(f"diagnostics: {ASSETS / 'diamonds_diagnostics.png'}")
except ImportError:
    print("matplotlib not installed - skipping diagnostic plots "
          "(uv run --with matplotlib python examples/visualization.py)")

print(f"test rmse: {Metrics().regression(y[n_tr:], pred)['rmse']:.1f}")
