# End-to-end walkthrough — churn

Predict bank churn using raw category labels and missing predictors. `ModelTree` accepts them directly with declared types. This example uses a regression score as a baseline; it is not necessarily a calibrated probability.

Script: [`examples/churn_demo.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/churn_demo.py>) (`uv run --with pandas --with pyarrow --with matplotlib python examples/churn_demo.py`).

## Typed ingest

The script requires pandas and the local parquet dataset. It uses a deterministic row split for demonstration. For production, choose a time-aware or group-aware split when the data requires it. No category encoder or missing-value imputation is used. The model is a regression baseline; its scores are not guaranteed calibrated probabilities. Choose thresholds on validation data using decision costs.

```
import pandas as pd
from mltreelib import Metrics, ModelTree

df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float).tolist()  # churn: No/Yes -> 0/1
print(f"rows: {len(df)}   churn rate: {sum(y) / len(y):.3f}")

# Pass raw categories and missing predictors directly to ModelTree.
cols = [c for c in df.columns if c != "churn"]
n_train = int(0.8 * len(df))
dtypes = ["categorical" if col in ("country", "gender", "credit_card", "active_member") else "numeric" for col in cols]
X = [[None if pd.isna(v) else v for v in row] for row in df[cols].itertuples(index=False, name=None)]
print("columns (xN index -> name, dtype):")
print(
    "  " + ", ".join(f"x{i}={n}({d[:4]})" for i, (n, d) in enumerate(zip(cols, dtypes)))
)

# Deterministic 80/20 split.
xtr, ytr = X[:n_train], y[:n_train]
xte, yte = X[n_train:], y[n_train:]

model = ModelTree(max_depth=4, min_rows=40)
model.fit(xtr, ytr, dtypes=dtypes, col_names=cols)
pred = model.predict(xte)

reg = Metrics().regression(yte, pred)
hard = [1.0 if p >= 0.5 else 0.0 for p in pred]
clf = Metrics().classification([int(v) for v in yte], [int(v) for v in hard])
print(f"test rmse: {reg['rmse']:.3f}  r2: {reg['r2']:.3f}")
print(f"accuracy @0.5: {clf['accuracy']:.3f}  macro f1: {clf['macro_f1']:.3f}")
print(f"confusion: {clf['confusion']}")
```

## Interpretation

Inspect the named splits and leaf support before using a segment for decisions. A predictive association does not establish the effect of an intervention. The full script prints the fitted tree and writes a DOT file or diagram.
