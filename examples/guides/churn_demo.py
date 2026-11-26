"""End-to-end churn walkthrough on the bundled Bank Customer Churn dataset.

Predict who leaves the bank — typed ingest: categoricals enter the tree as
LEVELS (a distinct value is a level; the split search partitions level sets,
nothing is ever encoded), NAs route as-is. Raw categorical labels and missing predictor values go directly to ModelTree.
Fit, evaluate, interpret.

    uv run python examples/guides/churn_demo.py
    (optional: add `--with pygraphviz` after `uv run` to render the tree; needs Graphviz)
"""

import pandas as pd
from mltreelib import MetricsReport, ModelTree
from mltreelib.data import Dataset

df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
cols = df.to_dict("list")
cols["churn"] = [1.0 if v == "Yes" else 0.0 for v in cols["churn"]]
print(f"rows: {len(df)}   churn rate: {sum(cols['churn']) / len(df):.3f}")

# Raw categories and missing predictors go straight to ModelTree (NaN is missing, kept as is).
ds = Dataset(cols, target="churn")
print("columns (xN index -> name, dtype):")
print(
    "  " + ", ".join(f"x{i}={n}({d[:4]})" for i, (n, d) in enumerate(zip(ds.names, ds.dtypes)))
)

# Deterministic 80/20 split, no shuffling: first 80% of rows train, last 20% test.
train, test = ds.split(test_size=0.2, time=True)

model = ModelTree(max_depth=4, min_rows=40)
model.fit(train)
pred = model.predict(test)

reg = MetricsReport("regression", y_true=test.y, y_pred=pred)
hard = [1 if p >= 0.5 else 0 for p in pred]
clf = MetricsReport("classification", y_true=[int(v) for v in test.y], y_pred=hard)
print(f"test rmse: {reg['rmse']:.3f}  r2: {reg['r2']:.3f}")
print(f"accuracy @0.5: {clf['accuracy']:.3f}  macro f1: {clf['f1_macro']:.3f}")
print(f"confusion: {clf.tables['confusion']}")

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
