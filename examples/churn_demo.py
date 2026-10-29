"""End-to-end churn walkthrough on the bundled Bank Customer Churn dataset.

Predict who leaves the bank — typed ingest: categoricals enter the tree as
LEVELS (a distinct value is a level; the split search partitions level sets,
nothing is ever encoded), NAs route as-is. Strings are coded once by
IndexMapper because the fit takes numbers; that mapping is a pipeline step,
not a tree step. Fit, evaluate, interpret.

    uv run python examples/churn_demo.py
"""

import pandas as pd

from mltreelib import IndexMapper, Metrics, ModelTree

df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float).tolist()  # churn: No/Yes -> 0/1
print(f"rows: {len(df)}   churn rate: {sum(y) / len(y):.3f}")

# Typed ingest: country and gender are categorical — level codes go to the
# tree as-is. IndexMapper only turns the strings into codes (its one job);
# the tree never sees dummies.
cols = [c for c in df.columns if c != "churn"]
mappers = {}
for col in ("country", "gender"):
    m = IndexMapper()
    m.fit(col, [None if pd.isna(v) else str(v) for v in df[col]])
    mappers[col] = m

dtypes, columns = [], []
for col in cols:
    raw = [None if pd.isna(v) else v for v in df[col]]
    if col in mappers:
        dtypes.append("categorical")
        columns.append(mappers[col].transform(col, [None if v is None else str(v) for v in raw]))
    else:
        dtypes.append("numeric")
        columns.append([None if v is None else float(v) for v in raw])
X = [list(row) for row in zip(*columns)]
print("columns (xN index -> name, dtype):")
print("  " + ", ".join(f"x{i}={n}({d[:4]})" for i, (n, d) in enumerate(zip(cols, dtypes))))

# Deterministic 80/20 split.
n_train = int(0.8 * len(X))
xtr, ytr = X[:n_train], y[:n_train]
xte, yte = X[n_train:], y[n_train:]

model = ModelTree(max_depth=4, min_rows=40)
model.fit(xtr, ytr, dtypes=dtypes)
pred = model.predict(xte)

reg = Metrics().regression(yte, pred)
hard = [1.0 if p >= 0.5 else 0.0 for p in pred]
clf = Metrics().classification([int(v) for v in yte], [int(v) for v in hard])
print(f"test rmse: {reg['rmse']:.3f}  r2: {reg['r2']:.3f}")
print(f"accuracy @0.5: {clf['accuracy']:.3f}  macro f1: {clf['macro_f1']:.3f}")
print(f"confusion: {clf['confusion']}")

print()
print(model)  # the story: which customers churn (levels as-is)

dot = model.to_dot()
try:
    import pygraphviz as pgv

    pgv.AGraph(string=dot).draw("docs/assets/churn_tree.svg", prog="dot")
    print("wrote docs/assets/churn_tree.svg (rendered by graphviz)")
except ImportError:
    open("docs/assets/churn_tree.dot", "w").write(dot)
    print("wrote docs/assets/churn_tree.dot (render with graphviz `dot -Tsvg`)")
