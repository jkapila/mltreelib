"""Inspection: metrics reports and every structural view of a fitted tree.

Same big diamonds dataset as the model-building example.

    uv run --with pandas python examples/inspection.py
"""

import pandas as pd

from mltreelib import Metrics, ModelTree, OneHotEncoder

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
model = ModelTree(max_depth=8, min_rows=50)
model.fit(X[:n_tr], y[:n_tr])
pred = model.predict(X[n_tr:])

# --- metrics reports ----------------------------------------------------
reg = Metrics().regression(y[n_tr:], pred)
print("regression report (test set):")
for k in ("n", "rmse", "mae", "mad", "r2", "bias", "max_error", "explained_variance"):
    print(f"  {k:<20} {reg[k]:.3f}")
print("  decile calibration (first 3):")
for d in reg["deciles"][:3]:
    print(f"    decile {d['decile']}: n={d['n']}  mean_error={d['mean_error']:.1f}  mae={d['mae']:.1f}")

# --- structural inspection ---------------------------------------------
print(f"\nn_leaves: {model.n_leaves()}")

splits = model.splits()
print(f"splits: {len(splits)} total; first 5:")
for s in splits[:5]:
    print(f"  {s['col']}  < {s['threshold']:.3f}  missing_left={s['missing_left']}")

print("\nsplit_paths() - split columns along each root-to-leaf path:")
for p in model.split_paths()[:5]:
    print("  ", " -> ".join(p))

leaf = model.leaves()[0]
print(f"\nfirst leaf: n={leaf['n']}  mean={leaf['mean']:.1f}  std={leaf['std']:.1f}")
print(f"  range: [{leaf['min']:.1f}, {leaf['max']:.1f}]  quantiles: {[round(q) for q in leaf['quantiles']]}")
print(f"  linear coefs on: {leaf['cols'][:6]}  intercept={leaf['intercept']:.1f}")

prof = model.profile()
print(f"\nfit profile: {len(prof)} depth levels; slowest depth's split scan: "
      f"{max(p['scan_ns'] for p in prof) / 1e6:.1f} ms")

# --- the rpart-style text ---------------------------------------------
print("\nthe tree itself (top of the print):")
print("\n".join(str(model).splitlines()[:12]))
