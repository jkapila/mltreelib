"""End-to-end churn walkthrough on the bundled Bank Customer Churn dataset.

Predict who leaves the bank: encode categoricals, fit, evaluate, interpret.
The tree is a regressor - fit on a 0/1 churn target it makes a solid,
fully interpretable baseline (threshold predictions at 0.5).

    uv run python examples/churn_demo.py
"""

import pandas as pd

from mltreelib import Metrics, ModelTree, OneHotEncoder

df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float)          # churn: No/Yes -> 0/1
print(f"rows: {len(df)}   churn rate: {y.mean():.3f}")

# Encode the two high-value categoricals with the package's OneHotEncoder;
# the rest of the schema is already numeric.
x = df.drop(columns=["churn"]).copy()
ohe_cols = {}
for col in ("country", "gender"):
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col]])
    ohe_cols[col] = enc.transform([None if v is None else str(v) for v in x[col]])

frame = x.drop(columns=["country", "gender"]).astype(float)
for col, dummies in ohe_cols.items():
    for level, values in dummies.items():
        frame[f"{col}_{level}"] = values
X = frame.values.tolist()
y_list = y.tolist()
print("feature columns (the tree prints xN indices):")
print("  " + ", ".join(f"x{i}={n}" for i, n in enumerate(frame.columns)))

# Deterministic 80/20 split.
n_train = int(0.8 * len(X))
xtr, ytr = X[:n_train], y_list[:n_train]
xte, yte = X[n_train:], y_list[n_train:]

model = ModelTree(max_depth=4, min_rows=40)
model.fit(xtr, ytr)
pred = model.predict(xte)

reg = Metrics().regression(yte, pred)
hard = [1.0 if p >= 0.5 else 0.0 for p in pred]
clf = Metrics().classification([int(v) for v in yte], [int(v) for v in hard])
print(f"test rmse: {reg['rmse']:.3f}  r2: {reg['r2']:.3f}")
print(f"accuracy @0.5: {clf['accuracy']:.3f}  macro f1: {clf['macro_f1']:.3f}")
print(f"confusion: {clf['confusion']}")

print()
print(model)  # the story: which customers churn

open("churn_tree.svg", "w").write(model.to_svg())
print("wrote churn_tree.svg")
