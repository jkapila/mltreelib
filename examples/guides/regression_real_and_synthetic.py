"""Compare regression configurations on synthetic dirty data, then fit real data.

Run: uv run python examples/guides/regression_real_and_synthetic.py
Requires pandas, pyarrow and matplotlib, and OpenML 46923 as datasets/diamonds.parquet.
The model selected on the synthetic problem is an illustration, not a universal winner.
"""
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mltreelib import Metrics, ModelTree, plots

rng = Random(77)
x, y = [], []
for _ in range(5000):
    category = rng.choice(["north", "south"])
    volume = rng.uniform(0, 10)
    observed = None if rng.random() < .15 else volume
    x.append([category, observed])
    y.append(3 + 2 * volume + (4 if category == "south" else 0) + rng.gauss(0, 1.5))

names, dtypes = ["region", "volume"], ["categorical", "numeric"]
metrics = Metrics()
choices = [("mean", 2), ("mean", 4), ("linear", 2), ("linear", 4)]
results = []
for leaf, depth in choices:
    model = ModelTree(leaf=leaf, max_depth=depth, min_rows=30)
    model.fit(x[:3000], y[:3000], col_names=names, dtypes=dtypes)
    rmse = metrics.regression(y[3000:4000], model.predict(x[3000:4000]))["rmse"]
    results.append((rmse, leaf, depth, model))
    print("synthetic validation", leaf, depth, "RMSE", round(rmse, 3))
_, leaf, depth, best = min(results, key=lambda row: row[0])
test = metrics.regression(y[4000:], best.predict(x[4000:]))
baseline = metrics.regression(y[4000:], [sum(y[:3000]) / 3000] * 1000)
print("synthetic selected:", leaf, depth)
print("synthetic test RMSE model/baseline:", round(test["rmse"], 3), round(baseline["rmse"], 3))
print("synthetic test missing predictors:", sum(row[1] is None for row in x[4000:]))

path = Path("datasets/diamonds.parquet")
if not path.exists():
    raise FileNotFoundError("Download OpenML 46923 to " + str(path))
df = pd.read_parquet(path)
features = [c for c in df.columns if c != "price"]
types = ["categorical" if str(df[c].dtype) in ("category", "object", "str") else "numeric" for c in features]
real_x = [[None if pd.isna(v) else v for v in row] for row in df[features].itertuples(index=False, name=None)]
real_y = [float(v) for v in df["price"]]
indices = list(range(len(real_x)))
Random(2026).shuffle(indices)
tr, te = indices[:6000], indices[6000:8000]
xtr, xte = [[real_x[i] for i in ids] for ids in (tr, te)]
ytr, yte = [[real_y[i] for i in ids] for ids in (tr, te)]
real = ModelTree(leaf=leaf, max_depth=depth, min_rows=30)
real.fit(xtr, ytr, col_names=features, dtypes=types)
pred = real.predict(xte)
report = metrics.regression(yte, pred)
base = metrics.regression(yte, [sum(ytr) / len(ytr)] * len(yte))
print("real rows train/test:", len(tr), len(te))
print("real missing in training:", dict(zip(features, [sum(row[j] is None for row in xtr) for j in range(len(features))])))
print("real test RMSE model/baseline:", round(report["rmse"], 3), round(base["rmse"], 3))
print("real test MAE model/baseline:", round(report["mae"], 3), round(base["mae"], 3))
print("real splits:", real.splits())
print("real first leaf:", real.leaves()[0])

out = Path("docs/assets/guides")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar([f"{leaf_name}, d={d}" for _, leaf_name, d, _ in results], [r for r, _, _, _ in results], color="#168378")
axes[0].set(xlabel="Leaf and max depth", ylabel="Validation RMSE", title="Synthetic configuration check")
plots.plot_residuals(yte, pred, ax=axes[1]); axes[1].set_title("Diamonds: held-out residuals")
fig.tight_layout(); fig.savefig(out / "regression-real-synthetic.svg"); plt.close(fig)
