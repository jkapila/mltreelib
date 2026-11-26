"""Inspection: metrics reports and every structural view of a fitted tree.

Same big diamonds dataset as the model-building example.

    uv run --with pandas python examples/guides/inspection.py
"""

import pandas as pd
from mltreelib import MetricsReport, ModelTree, OneHotEncoder
from mltreelib.data import Dataset

df = pd.read_parquet("datasets/diamonds.parquet")
y = df["price"].astype(float).tolist()
x = df.drop(columns=["price"]).copy()
for col in x.select_dtypes(include="category").columns:
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col].astype(object)])
    dummies = enc.transform(
        [None if v is None else str(v) for v in x[col].astype(object)]
    )
    x = x.drop(columns=[col])
    for level, values in dummies.items():
        x[f"{col}_{level}"] = values
ds = Dataset(x.assign(price=y).to_dict("list"), target="price")  # missing cells stay missing (NaN)

# Deterministic 80/20 split, no shuffling: first 80% of rows train, last 20% test.
train, test = ds.split(test_size=0.2, time=True)
model = ModelTree(max_depth=8, min_rows=50)
model.fit(train)
pred = model.predict(test)

# --- metrics reports ----------------------------------------------------
reg = MetricsReport("regression", y_true=test.y, y_pred=pred)
print("regression report (test set):")
for k in ("n", "rmse", "mae", "mad", "r2", "bias", "max_error", "explained_variance"):
    print(f"  {k:<20} {reg[k]:.3f}")
print("  decile calibration (first 3):")
for d in reg.tables["error_deciles"][:3]:
    print(
        f"    decile {d['decile']}: n={d['n']}  mean_error={d['mean_error']:.1f}  mae={d['mae']:.1f}"
    )

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
print(
    f"  range: [{leaf['min']:.1f}, {leaf['max']:.1f}]  quantiles: {[round(q) for q in leaf['quantiles']]}"
)
print(f"  linear coefs on: {leaf['cols'][:6]}  intercept={leaf['intercept']:.1f}")

prof = model.profile()
print(
    f"\nfit profile: {len(prof)} depth levels; slowest depth's split scan: "
    f"{max(p['scan_ns'] for p in prof) / 1e6:.1f} ms"
)

# --- explaining one prediction (DALEX-style break-down) -----------------
row = test["x"][0]
bd = model.break_down(row)
total = sum(d for _, d in bd["steps"])
print(
    f"\nbreak_down of one row: baseline {bd['baseline']:.1f} -> "
    f"prediction {bd['prediction']:.1f} ({len(bd['steps'])} steps)"
)
for col, delta in bd["steps"][:6]:
    print(f"  {col:<6} {delta:+.1f}")
if len(bd["steps"]) > 6:
    print(f"  ... {len(bd['steps']) - 6} more steps")
print(
    f"  residual {bd['residual']:+.1f}  "
    f"(sum check: {bd['baseline'] + total + bd['residual']:.1f})"
)

# --- feature importance (share of total split gain) --------------------
imp = model.feature_importance()
print("\nfeature importance (top 5 of %d):" % len(imp))
for name, share in imp[:5]:
    print(f"  {name:<6} {share:.3f}")

# --- the rpart-style text ---------------------------------------------
print("\nthe tree itself (top of the print):")
print("\n".join(str(model).splitlines()[:12]))
