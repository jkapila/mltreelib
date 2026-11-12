# Two-Control Architecture & 0D Leaf Formulations

`ModelTree` decouples tree growth from leaf modeling and provides rich 0D statistical profiles across terminal nodes:

1. **Two-Control Structure**:
   - `ModelTree`: Governs greedy tree partitioning, stopping, and pruning.
   - `LeafControl`: Encapsulates leaf model family, regression kind, penalties, and local slope constraints.
   - Inject `control=LeafControl(...)` directly into `ModelTree(...)` to prevent parameter bloat.
2. **Rich 0D Formulations**:
   - Produce exact 0D statistics across leaves: `prob`, `odds`, `log_odds`, `majority_class`, `var`, `skew`, `kurtosis`, `median`, `mad`, `iqr`, `entropy`, `gini`, `zero_fraction`, `dispersion`.
   - Access directly via `leaf["stats"]` or `tree.leaf_stats()`.
   - Predict 0D formulations row-wise on test data: `predict(x, kind="prob"|"odds"|"log_odds"|"median"|"stats")`.
3. **Dual Monotonicity**:
   - `tree_monotone`: Enforces directional ordering across tree splits: $(\bar{y}_L - \bar{y}_R) \cdot m \le 0$.
   - `leaf_monotone`: Enforces directional sign bounds on local linear leaf slopes ($\beta_j \ge 0$ for `+1`, $\le 0$ for `-1`).
4. **Row & Column Weights**:
   - `sample_weight` / `row_weights` / `w`: Row-level importance weighting.
   - `col_weights`: Feature-level split priorities scaling candidate split gains and weighting feature subsampling during `col_sample`.
5. **Interaction Groups with Column Names**:
   - Restrict root-to-leaf paths using semantic column names (e.g. `interaction_groups=[["age", "income"], ["city_tier"]]`).

Script: [`examples/two_controls_and_0d_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/two_controls_and_0d_showcase.py) (`python examples/two_controls_and_0d_showcase.py`).

---

## 1. Two-Control Architecture: `LeafControl`

Separate tree topology from leaf model complexity:

```python
from mltreelib import ModelTree, LeafControl

# Step 1: Configure leaf models independently
leaf_ctrl = LeafControl(
    leaf="linear",
    leaf_family="gaussian",
    l1=0.0,
    l2=1e-3,
    leaf_monotone=[1, -1],  # local feature slope constraints
)

# Step 2: Configure tree partitioning and inject leaf controls
model = ModelTree(
    max_depth=3,
    min_rows=10,
    tree_monotone=[1, 0],   # global split ordering
    col_weights=[2.0, 0.5], # feature split priorities
    control=leaf_ctrl,
)
```

---

## 2. 0D Leaf Formulations & Stats Dictionary

Train a tree with 0D probability leaves:

```python
clf = ModelTree(leaf="prob", max_depth=2, min_rows=10)
clf.fit(X, y, sample_weight=weights)

# Inspect the full 0D statistics dictionary on every leaf:
for i, leaf in enumerate(clf.leaves()):
    st = leaf["stats"]
    print(f"Leaf {i}: prob={st['prob']:.4f}, odds={st['odds']:.4f}, entropy={st['entropy']:.4f}")
    print(f"         median={st['median']:.4f}, mad={st['mad']:.4f}, iqr={st['iqr']:.4f}")

# Or get the list of leaf stats directly:
all_stats = clf.leaf_stats()
```

---

## 3. Flexible Prediction Kinds

Predict response values, probabilities, odds ratios, log-odds, medians, or full per-row stats dictionaries:

```python
# Specific 0D formulations:
preds_prob = clf.predict(X_test, kind="prob")
preds_odds = clf.predict(X_test, kind="odds")
preds_lo   = clf.predict(X_test, kind="log_odds")
preds_med  = clf.predict(X_test, kind="median")

# Full 0D stats profile per observation:
preds_stats = clf.predict(X_test, kind="stats")
# Returns a list of dicts: [{"prob": ..., "odds": ..., "entropy": ..., ...}, ...]
```

---

## 4. Interaction Groups with Column Names

Enforce LightGBM-style interaction constraints using column name lists:

```python
tree = ModelTree(
    max_depth=3,
    min_rows=5,
    interaction_groups=[["age", "income"], ["city_tier"]],
)

# Column names resolve automatically during fit:
tree.fit(X, y, col_names=["age", "income", "city_tier"])

# Verify split paths respect the groups:
paths = tree.split_paths()
for path in paths:
    print("Path:", path)
```
