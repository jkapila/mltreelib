# End-to-end walkthrough — churn

The whole story on one real problem: predict who leaves the bank. **Typed
ingest** throughout — `country` and `gender` enter the tree as **levels**
(a distinct value is a level; the split search partitions level sets), NAs
route as-is, and nothing is ever encoded for the tree. Strings are coded
once by `IndexMapper` because the fit takes numbers — that mapping is a
pipeline step, not a tree step. The tree is a regressor; fit on a 0/1 churn
target it is a solid, fully interpretable baseline (threshold at 0.5).

Script:
[`examples/churn_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/churn_demo.py)
(`uv run --with pandas --with pyarrow --with matplotlib python examples/churn_demo.py`).

## Typed ingest

```python
from mltreelib import IndexMapper, Metrics, ModelTree

df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float).tolist()

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

model = ModelTree(max_depth=4, min_rows=40)
model.fit(xtr, ytr, dtypes=dtypes)
```

## Output

```text
rows: 10000   churn rate: 0.204
columns (xN index -> name, dtype):
  x0=credit_score(nume), x1=country(cate), x2=gender(cate), x3=age(nume), x4=tenure(nume), x5=balance(nume), x6=products_number(nume), x7=credit_card(nume), x8=active_member(nume), x9=estimated_salary(nume)
test rmse: 0.344  r2: 0.275
accuracy @0.5: 0.852  macro f1: 0.721
confusion: [[1535.0, 55.0], [242.0, 168.0]]

n= 8000

node), split, n, deviance, yval
      * denotes terminal node

 1) root 8000 1296.108875 0.20337500
   2) x8< 0 3881 756.173151 0.26513785
     4) x6< 2 3737 682.729997 0.24056730
       8) x6< 1 1978 461.367543 0.37057634
        16) x5< 98635.77 713 171.279102 0.40112202 *
        17) x5>=98635.77 1265 289.048221 0.35335968 *
       9) x6>=1 1759 150.334281 0.09437180
        18) x3< 44 1552 84.780928 0.05798969 *
        19) x3>=44 207 48.096618 0.36714976 *
     5) x6>=2 144 12.638889 0.90277778
      10) x3< 42 65 10.984615 0.78461538 *
      11) x3>=42 79 0.000000 1.00000000 *
   3) x8>=0 4119 511.181840 0.14518087
     6) x6< 2 3996 437.437437 0.12512513
      12) x0< 768 3509 388.566543 0.12681676
        24) x3< 45 2666 208.501125 0.08552138 *
        25) x3>=45 843 161.141163 0.25741400 *
      13) x0>=768 487 48.788501 0.11293634
        26) x2 in {0} 272 18.529412 0.07352941 *
        27) x2 not-in {0} 215 29.302326 0.16279070 *
     7) x6>=2 123 19.918699 0.79674797
      14) x0< 614 43 6.511628 0.81395349 *
      15) x0>=614 80 13.387500 0.78750000 *

wrote docs/assets/churn_tree.svg (rendered by graphviz)
```

Read the tree top-down: `x8` is `active_member` (the strongest signal),
`x6` is `products_number` (2+ products almost always churns — leaves 5/7
at 0.78–0.90), `x3` is `age`, `x5` is `balance`. And the categorical split
renders as level sets: `x2 in {0}` is `gender` — one level against the
rest, exactly as the split search treats it (no dummies anywhere).

The tree diagram (left-to-right) is `churn_tree.svg`; the explanation
charts for the same fit — importance and the breakdown waterfall of one
row — live in [Visualization](visualization.md).
