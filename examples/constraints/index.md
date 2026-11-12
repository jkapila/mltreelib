# Monotone and interaction constraints

`ModelTree` supports domain-enforced structural constraints:

1. **Dual Monotone constraints**:
2. `tree_monotone=[...]`: enforces that split child node target means are monotonically ordered across split thresholds: $(\\bar{y}\_L - \\bar{y}\_R) \\cdot m \\le 0$.
3. `leaf_monotone=[...]`: enforces that local linear leaf model coefficients satisfy directional sign constraints ($\\beta\_j \\ge 0$ for `+1`, $\\beta\_j \\le 0$ for `-1`).
4. `monotone=[...]`: sets both tree and leaf monotonicity simultaneously.
5. **Interaction constraints** (`interaction_groups=[[...], [...]]`): restricts trees from mixing split variables across specified feature groups along any root-to-leaf path. Supports column names as well as column indices.
6. **Column Weights** (`col_weights=[...]`): scales candidate split gains by feature importance and biases feature subsampling during `col_sample`.

Script: [`examples/constraints_check.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/constraints_check.py>) (`uv run python examples/constraints_check.py`).

## Output

```
== monotone (+1 on age, healthcare: charges rise with age) ==
  (R gbm oracle has constant leaves -> compare at mean-leaf semantics;
   linear leaves: within-leaf slopes are sign-projected and boundaries
   at splits on age gate on the children's fitted outputs at the
  ours/mean   unconstrained monotone: False   constrained: True
  ours/mean   rmse uncon 4791.068 -> constrained 4779.912
  ours/linear unconstrained monotone: False   constrained: False
  ours/linear rmse uncon 4986.580 -> constrained 5130.727

== interaction groups [[0,1],[2,3]] (property: no mixed paths) ==
  ours      mixed paths constrained: 0 (want 0)   unconstrained: 2 (want > 0)
```

## How to use

### Dual Monotone constraints

Pass a list of integers with length matching the number of features: - `+1`: monotonically non-decreasing in this feature. - `-1`: monotonically non-increasing in this feature. - `0`: unconstrained.

```
from mltreelib import ModelTree, LeafControl

# Constrain tree splits on column 0 to be non-decreasing, and leaf slopes on column 1 to be non-increasing:
tree = ModelTree(
    tree_monotone=[1, 0],
    leaf_monotone=[0, -1],
    leaf="linear",
    max_depth=3,
)
tree.fit(X, y)

# Or encapsulate leaf monotonicity inside LeafControl:
leaf_ctrl = LeafControl(leaf="linear", leaf_monotone=[0, -1])
tree = ModelTree(tree_monotone=[1, 0], control=leaf_ctrl)
tree.fit(X, y)
```

### Interaction constraints with column names

Define variable groups as lists of feature column names or column indices:

```
from mltreelib import ModelTree

# Group features by domain semantics using column names:
groups = [["age", "income"], ["city", "state"]]
tree = ModelTree(interaction_groups=groups, max_depth=6)

# Column names passed at fit time resolve automatically:
tree.fit(X, y, col_names=["age", "income", "city", "state"])

# Inspect paths to verify constraint satisfaction:
paths = tree.split_paths()
print("Split paths:", paths)
```

### Feature / Column Weights (`col_weights`)

Pass feature importance weights to scale candidate split gains and weight `col_sample`:

```
# Double priority for feature 0, half priority for feature 1:
tree = ModelTree(col_weights=[2.0, 0.5], col_sample=0.8, seed=42)
tree.fit(X, y)
```
