"""Model building & predictions: every parameter and nuance the tree handles.

One big, tedious dataset throughout: the bundled diamonds parquet (53,940
rows, numeric + categorical columns, missing cells), with deeper trees than
the quickstart. Fits run with NO regularization (l1=l2=0) so the tour is fast
and the numbers are clean - regularization gets its own dedicated example.

    uv run --with pandas python examples/model_building.py
"""

import pandas as pd

from mltreelib import Metrics, ModelTree, OneHotEncoder, winnow

df = pd.read_parquet("datasets/diamonds.parquet")
y = df["price"].astype(float).tolist()
x = df.drop(columns=["price"]).copy()
for col in x.select_dtypes(include="category").columns:  # categoricals -> dummies
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col].astype(object)])
    dummies = enc.transform([None if v is None else str(v) for v in x[col].astype(object)])
    x = x.drop(columns=[col])
    for level, values in dummies.items():
        x[f"{col}_{level}"] = values
X = [[None if pd.isna(v) else float(v) for v in row] for row in x.values.tolist()]
names = list(x.columns)
print(f"diamonds: {len(X)} rows x {len(names)} cols (deeper trees below)")

n_tr = int(0.8 * len(X))
xtr, ytr, xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

# Fits use the SHIPPED DEFAULTS (linear leaves + small ridge, closed-form
# fast). Regularization gets its own dedicated example; the one thing to know
# now is why the default keeps the ridge - shown at the end of section 2.


def show(label, model, w=None):
    model.fit(xtr, ytr, w)
    pred = model.predict(xte)
    m = Metrics().regression(yte, pred)
    print(f"{label:<38} rmse={m['rmse']:>8.1f}  r2={m['r2']:.3f}  leaves={model.n_leaves()}")


# 1) depth / min_rows / min_gain - the growth knobs
show("baseline depth=8", ModelTree(max_depth=8, min_rows=50))
show("deeper depth=12", ModelTree(max_depth=12, min_rows=50))
show("coarser leaves min_rows=500", ModelTree(max_depth=12, min_rows=500))

# 2) leaf model kinds: mean (piecewise constant) vs linear (piecewise slopes)
show('leaf="mean"', ModelTree(max_depth=8, min_rows=50, leaf="mean"))
show('leaf="linear", leaf_cols="path"', ModelTree(max_depth=8, min_rows=50, leaf_cols="path"))
show("linear, l1=0 l2=0 (no ridge!)", ModelTree(max_depth=8, min_rows=50, l1=0.0, l2=0.0))

# 3) row weights: upweight the small stones, the fit follows the weight
w = [2.0 if carat is not None and carat < 1.0 else 1.0
     for carat in (r[names.index("carat")] for r in xtr)]
show("row weights (2x carat<1)", ModelTree(max_depth=8, min_rows=50, leaf="mean"), w)

# 4) winnowing: screen junk columns before the split search
res = winnow(xtr, ytr, min_relevance=0.05)
print(f"winnow kept {len(res['kept'])}/{len(names)} columns "
      f"(dropped: {sorted(set(range(len(names))) - set(res['kept']))})")
show("min_relevance=0.05 (winnowed fit)", ModelTree(max_depth=8, min_rows=50, min_relevance=0.05))

# 5) monotone constraints - a domain fact: price never falls as carat rises
mono = [0] * len(names)
mono[names.index("carat")] = 1
show("monotone +1 on carat", ModelTree(max_depth=8, min_rows=50, monotone=mono))

# 6) interaction groups: limit which columns may split along one path
groups = [[names.index(c) for c in names if c.startswith("carat")],
          [names.index(c) for c in names if not c.startswith("carat")]]
show("interaction_groups [[carat],[rest]]", ModelTree(max_depth=8, min_rows=50, interaction_groups=groups))

# 7) missing values are first-class: None cells train and predict fine
m = ModelTree(max_depth=8, min_rows=50)
m.fit(xtr, ytr)
probe = [dict(zip(names, r)) for r in xte[:3]]
sparse = [[r.get(n) for n in names] for r in probe]
for r in sparse:
    r[names.index("carat")] = None  # pretend the carat cell is missing
print("predictions with missing carat:", [round(v, 1) for v in m.predict(sparse)])
