"""Native mixed inputs and held-out evaluation with visual diagnostics.

Run: uv run python examples/guides/data_metrics_pathway.py
Requires matplotlib. Graphviz is optional for rendering the tree.
All outcomes and categories are synthetic.
"""
from math import exp
from pathlib import Path
from random import Random
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import Metrics, ModelTree, plots

OUT = Path("docs/assets/guides")
OUT.mkdir(parents=True, exist_ok=True)
rng = Random(101)
rows, reg_y, class_y = [], [], []
for _ in range(4000):
    region = rng.choice(["north", "south"])
    volume = rng.uniform(0, 10)
    observed = None if rng.random() < 0.12 else volume
    rows.append([region, observed])
    reg_y.append(3 + 2 * volume + (3 if region == "south" else 0) + rng.gauss(0, 2))
    p = 1 / (1 + exp(-(-1.5 + .25 * volume + (.7 if region == "south" else 0))))
    class_y.append(float(rng.random() < p))

names, types = ["region", "volume"], ["categorical", "numeric"]
x_train, x_val, x_test = rows[:2000], rows[2000:3000], rows[3000:]
metrics = Metrics()

# Missingness counts are shown before fitting; no value is filled or encoded.
missing = [sum(row[j] is None for row in x_train) for j in range(len(names))]
print("training missing counts:", dict(zip(names, missing)))
fig, ax = plt.subplots(figsize=(5.6, 3.5))
ax.bar(names, missing, color="#168378")
ax.set(ylabel="Missing training rows", title="Inspect input missingness")
fig.tight_layout(); fig.savefig(OUT / "data-missingness.svg"); plt.close(fig)

reg = ModelTree(leaf="mean", max_depth=3, min_rows=15)
reg.fit(x_train, reg_y[:2000], dtypes=types, col_names=names)
reg_pred = reg.predict(x_test)
reg_base = [sum(reg_y[:2000]) / 2000] * len(x_test)
print("regression test MAE model/baseline:",
      round(metrics.regression(reg_y[3000:], reg_pred)["mae"], 3),
      round(metrics.regression(reg_y[3000:], reg_base)["mae"], 3))
fig, ax = plt.subplots(figsize=(5.6, 4))
plots.plot_residuals(reg_y[3000:], reg_pred, ax=ax)
ax.set_title("Regression: test residuals")
fig.tight_layout(); fig.savefig(OUT / "data-residuals.svg"); plt.close(fig)

clf = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=3, min_rows=15)
clf.fit(x_train, class_y[:2000], dtypes=types, col_names=names)
val_scores = clf.predict_proba(x_val)
# Example cost: each missed event costs 3; each false alert costs 1.
def decision_cost(y, scores, threshold):
    return sum(3 if label == 1 and p < threshold else 1 if label == 0 and p >= threshold else 0
               for label, p in zip(y, scores))
threshold = min((.2, .4, .6, .8), key=lambda t: decision_cost(class_y[2000:3000], val_scores, t))
scores = clf.predict_proba(x_test)
hard = [int(p >= threshold) for p in scores]
report = metrics.classification([int(v) for v in class_y[3000:]], hard)
print("classification validation threshold:", threshold)
print("classification test confusion:", report["confusion"])
print("classification test average precision:", round(metrics.pr_curve(class_y[3000:], scores)["average_precision"], 3))
for filename, title, render in [
    ("data-pr.svg", "Classification: test precision–recall", lambda ax: plots.plot_pr(class_y[3000:], scores, ax=ax)),
    ("data-calibration.svg", "Classification: test calibration", lambda ax: plots.plot_calibration(class_y[3000:], scores, n_bins=5, ax=ax)),
    ("data-confusion.svg", "Classification: test decisions", lambda ax: plots.plot_confusion_matrix(report["confusion"], class_names=["No event", "Event"], ax=ax)),
    ("data-importance.svg", "Regression: split importance", lambda ax: plots.plot_importance(reg.feature_importance(), ax=ax)),
]:
    fig, ax = plt.subplots(figsize=(5.6, 4))
    render(ax); ax.set_title(title); fig.tight_layout(); fig.savefig(OUT / filename); plt.close(fig)

# Graphviz is optional; DOT remains available without the renderer.
dot = shutil.which("dot")
if dot:
    subprocess.run([dot, "-Tsvg", "-o", str(OUT / "data-tree.svg")], input=reg.to_dot(), text=True, check=True)
