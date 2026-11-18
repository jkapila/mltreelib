"""Real-data model and review example using TabArena Bank Customer Churn.

Download OpenML dataset 46911 to datasets/Bank_Customer_Churn.parquet.
Run: uv run python examples/real_data_workflow.py
Requires pandas, pyarrow and matplotlib. The data is not bundled in Git.
"""
from collections import Counter
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mltreelib import AnomalyTree, ClusterTree, Metrics, ModelTree, plots

path = Path("datasets/Bank_Customer_Churn.parquet")
if not path.exists():
    raise FileNotFoundError("Download OpenML 46911 to " + str(path))
df = pd.read_parquet(path)
features = ["country", "gender", "age", "tenure", "balance", "products_number", "credit_score"]
types = ["categorical", "categorical", "numeric", "numeric", "numeric", "numeric", "numeric"]
x = [[None if pd.isna(v) else v for v in row] for row in df[features].itertuples(index=False, name=None)]
y = [float(str(v) == "Yes") for v in df["churn"]]
order = list(range(len(x)))
Random(2026).shuffle(order)
train, val, test = order[:6000], order[6000:8000], order[8000:]
def pick(values, indices):
    return [values[i] for i in indices]
xtr, xval, xte = (pick(x, ids) for ids in (train, val, test))
ytr, yval, yte = (pick(y, ids) for ids in (train, val, test))
print("rows train/validation/test:", len(train), len(val), len(test))
print("feature missing counts:", dict(zip(features, [sum(row[j] is None for row in xtr) for j in range(len(features))])))

model = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=3, min_rows=50)
model.fit(xtr, ytr, dtypes=types, col_names=features)
val_scores = model.predict_proba(xval)
def cost(scores, truth, threshold):
    return sum(3 if y == 1 and p < threshold else 1 if y == 0 and p >= threshold else 0 for p, y in zip(scores, truth))
threshold = min([.2, .4, .6, .8], key=lambda t: cost(val_scores, yval, t))
scores = model.predict_proba(xte)
metrics = Metrics()
labels = [int(p >= threshold) for p in scores]
report = metrics.classification([int(v) for v in yte], labels)
print("training event rate:", round(sum(ytr) / len(ytr), 4))
print("validation threshold:", threshold)
print("test confusion:", report["confusion"])
print("test average precision:", round(metrics.pr_curve(yte, scores)["average_precision"], 4))
print("test brier model/baseline:", round(metrics.brier_score(yte, scores), 4), round(metrics.brier_score(yte, [sum(ytr)/len(ytr)] * len(yte)), 4))

# Unsupervised fits do not use the churn labels.
clusters = ClusterTree(kind="pct", max_depth=2, min_rows=100)
clusters.fit(xtr, dtypes=types, col_names=features)
print("cluster held-out counts:", dict(Counter(clusters.predict(xte))))
print("cluster rule count:", len(clusters.rules()))
anomaly = AnomalyTree(kind="eif", max_depth=8, extension_level=1, seed=42)
anomaly.fit(xtr, dtypes=types, col_names=features)
anomaly_scores = anomaly.score_samples(xte)
print("anomaly rows scored:", len(anomaly_scores))
print("anomaly top scores:", [round(v, 3) for v in sorted(anomaly_scores, reverse=True)[:5]])

out = Path("docs/assets/guides")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plots.plot_pr(yte, scores, ax=axes[0]); axes[0].set_title("Churn: held-out precision–recall")
plots.plot_calibration(yte, scores, n_bins=10, ax=axes[1]); axes[1].set_title("Churn: held-out calibration")
fig.tight_layout(); fig.savefig(out / "real-data-churn.svg"); plt.close(fig)
