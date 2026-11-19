# Prepare a model for prediction

This guide starts the deployment and schema workflow. It covers a fitted `ModelTree` in a running Python process. Persistence and cross-version compatibility need a separate implementation and contract check.

## Record the prediction contract

| Item | Record before handing off the model |
|---|---|
| Decision | Outcome, units, prediction horizon and when features are available. |
| Inputs | Feature names, order, types, units and allowed values. |
| Missingness | Which fields can be missing and how the fitted model routes them. |
| Categories | Training vocabulary and how the workflow handles an unseen value. |
| Output | Prediction kind, scale, class ordering and any decision threshold. |
| Evidence | Training data period, split, baseline, metrics and known model gaps. |

```mermaid
sequenceDiagram
    autonumber
    actor Client as Ingestion Client / API Gateway
    participant Guard as Schema Guard & Validator
    participant Normalizer as Sentinel Normalizer (None / NaN)
    participant Core as mltreelib-core Tree Route
    participant Leaf as Local Leaf Solver (Linear / GLM)
    participant Decision as Threshold & Policy Post-Processor

    Client->>Guard: Submit Raw Record (JSON Payload)
    Note over Guard: Check required keys, field order, and valid types
    Guard->>Normalizer: Verified Named Features
    Note over Normalizer: Map unobserved keys & NaNs to None; map unseen category levels to missing
    Normalizer->>Core: Fast Vector Traversal [x1, x2, ..., xk]
    Note over Core: Traverse internal split nodes using learned missing_left and surrogates
    Core->>Leaf: Deliver to Terminal Leaf Node
    Note over Leaf: Evaluate local linear model: y_hat = b0 + sum(bj * xj)
    Leaf->>Decision: Continuous Raw Score / Probability
    Note over Decision: Apply calibrated threshold (e.g. p >= 0.20 -> Review Queue)
    Decision-->>Client: Final Actionable Prediction & Audit Path
```

`ModelTree` accepts raw categorical strings and missing predictor values. Preserve the fitted feature order. A row-based input does not carry column names at prediction time, so your application needs to check the schema before constructing rows.

## Fit and predict with named records

The following example checks names before converting a record to the fitted order. It passes category labels and missing values directly to the tree.

```python
from mltreelib import ModelTree

names = ["region", "volume"]
dtypes = ["categorical", "numeric"]
x_train = [["north", 1.0], ["north", None], ["south", 1.0], ["south", None]] * 10
y_train = [1.0, 2.0, 9.0, 10.0] * 10
model = ModelTree(max_depth=2, min_rows=4, leaf="mean")
model.fit(x_train, y_train, col_names=names, dtypes=dtypes)

record = {"region": "north", "volume": None}
if set(record) != set(names):
    raise ValueError("Input fields do not match the model schema")
row = [record[name] for name in names]
print("Prediction:", model.predict([row])[0])
```

Check types, units, ranges and batch sizes at the application boundary as well. The short example checks field names only. It does not provide a complete serving validator.

## Checks before release

1. Compare a known in-process prediction with the intended serving path.
2. Check missing values, unseen categories, invalid types and changed column order.
3. Check the output scale and confirm the decision threshold uses independent validation data.
4. Record the model and package versions, data period and evaluation results.
5. Monitor input missingness, category changes, output distributions and observed outcomes.

The following gaps were reproduced in the current worktree and are tracked for investigation:

- A short or extra-long row was accepted in one test. Do not assume the model rejects a schema mismatch.
- An unseen category routed differently from a missing category in a tested tree, despite the stated unseen-as-missing rule.
- Python `pickle` raised `TypeError` for a fitted `ModelTree` in a test. A saved-model and reload contract needs implementation or a separately checked supported method.
- `break_down` rejected a raw string category in a tested row, although `predict` accepted it.

Do not present the checks above as complete deployment coverage. See the [data guide](data.md), [evaluation guide](evaluation.md), and [documentation audit](https://github.com/jkapila/mltreelib/tree/main/references/research) for further work.

## Check split boundaries

The DOT export in the checked build labels numeric branches as `<` and `>=`. In a one-feature reproduction with a threshold of zero, the fitted tree routed zero with lower values. It routed positive values to the other leaf. Use the fitted model to check predictions below, at, and above a threshold before translating a diagram into executable rules.
