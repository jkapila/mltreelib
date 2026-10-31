# Monotone and interaction constraints

`ModelTree` supports domain-enforced structural constraints:

1. **Monotone constraints** (`monotone=[...]`): enforce that predictions never decrease (`+1`) or never increase (`-1`) along specified feature dimensions.
2. **Interaction constraints** (`interaction_groups=[[...], [...]]`): restrict trees from mixing split variables across specified feature groups along any root-to-leaf path.

Script: [`examples/constraints_check.py`](https://github.com/jkapila/mltreelib/blob/main/examples/constraints_check.py) (`uv run python examples/constraints_check.py`).

## Output

```text
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

### Monotone constraints

Pass a list of integers with length matching the number of features:
- `+1`: predictions are monotonically non-decreasing in this feature.
- `-1`: predictions are monotonically non-increasing in this feature.
- `0`: unconstrained.

```python
from mltreelib import ModelTree

# Constrain column 0 to be non-decreasing, column 1 unconstrained
tree = ModelTree(monotone=[1, 0], leaf="mean")
tree.fit(X, y)
```

For constant leaves (`leaf="mean"`, `leaf="median"`), child leaf values are bounded to preserve the monotonicity condition across split thresholds.

### Interaction constraints

Define variable groups as lists of feature column indices:

```python
from mltreelib import ModelTree

# Features 0 and 1 may interact with each other; features 2 and 3 may interact with each other;
# but no root-to-leaf path may split on both {0, 1} and {2, 3}.
groups = [[0, 1], [2, 3]]
tree = ModelTree(interaction_groups=groups, max_depth=6)
tree.fit(X, y)

# Inspect paths to verify constraint satisfaction:
paths = tree.split_paths()
```
