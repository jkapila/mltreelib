# Churn

Who leaves the bank? An end-to-end run on the bundled **Bank Customer Churn** dataset (10,000 customers): encode categoricals, fit, evaluate, interpret. The full script: [`examples/churn_demo.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/churn_demo.py>).

The tree is a regressor — fit on a 0/1 churn target it makes a solid, fully interpretable baseline (threshold predictions at 0.5).

## Data

```
df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float)     # churn: No/Yes -> 0/1

# OneHotEncoder for the two categoricals; the rest is already numeric.
for col in ("country", "gender"):
    enc = OneHotEncoder()
    enc.fit(...)
    dummies = enc.transform(...)
```

```
rows: 10000   churn rate: 0.204
feature columns (the tree prints xN indices):
  x0=credit_score, x1=age, x2=tenure, x3=balance, x4=products_number,
  x5=credit_card, x6=active_member, x7=estimated_salary, x8=country_France,
  x9=country_Germany, x10=country_Spain, x11=gender_Female, x12=gender_Male
```

## Fit &amp; evaluate

```
model = ModelTree(max_depth=4, min_rows=40)
model.fit(xtr, ytr)
pred = model.predict(xte)
```

Deterministic 80/20 split. The evaluation uses the package's `Metrics`:

```
test rmse: 0.330  r2: 0.330
accuracy @0.5: 0.851  macro f1: 0.730
confusion: [[1520.0, 70.0], [228.0, 182.0]]
```

85% accuracy with 20% churn — and the confusion matrix shows the model finds most churners (182 of 410) while rarely crying wolf (70 false alarms).

## The story the tree tells

```
 1) root 8000 1296.108875 0.20337500
   2) x4< 1.5 4045 809.442274 0.27663782        # one product
     4) x1< 45.5 3050 465.351148 0.18786885     # younger
       8) x3< 97778.31 1073 167.679404 0.19384902
        16) x1< 39.5 785 90.955414 0.13375796 *
        17) x1>=39.5 288 66.163194 0.35763889 *
       9) x3>=97778.31 1977 297.612544 0.18462317
        18) x8< 0.5 1354 124.730428 0.10265879 *   # not France
        19) x8>=0.5 623 144.016051 0.36276083 *
     5) x1>=45.5 995 246.385930 0.54874372      # older, one product: 55% churn
   ...
   3) x4>=1.5 3955 442.749937 0.12844501        # 2+ products: 13% churn
     ...
      15) x3>=50210 168 6.708333 0.95833333     # wealthy, 3+ products: 96% churn!
```

Business reading: **product count is the first split** — one-product customers churn far more. Among them, **age** and **balance** segment the risk, and **country** matters (non-France higher). On the multi-product side there is a small pocket (3+ products, high balance) that is almost certain to leave — the classic "too many products" signal.

`print(model)` shows the full tree in rpart format; `model.to_svg()` writes `churn_tree.svg` for slides or docs.

Next: [Data Processing](<https://jkapila.github.io/mltreelib/examples/data-processing/index.md>) for the encoder catalog, or [Model build &amp; Tree Visualization](<https://jkapila.github.io/mltreelib/examples/model-build-viz/index.md>) to read every part of the output.
