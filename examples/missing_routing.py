"""Missing data routing, learned split directions, and surrogate splits.

Demonstrates mltreelib's first-class missing data contract:
1. Learned missing directions (missing_left): the tree places missing values
   on the branch that minimizes deviance/SSE.
2. Surrogate splits (maxsurrogate=1): when the primary split feature is missing
   at inference time, the best surrogate feature routes the row.
3. NA coefficients in linear leaves: linear leaves estimate dedicated NA-indicator
   terms so missing values in leaves do not fail or require imputation.
"""

from __future__ import annotations

import math
from mltreelib import ModelTree


def run_learned_direction():
    # Target is high when x0 >= 5. When x0 is missing, target is also high (y=10).
    # The tree learns that missing x0 belongs on the right branch (missing_left=False).
    x, y = [], []
    for i in range(200):
        val = float(i % 10)
        y_val = 10.0 if val >= 5.0 else 0.0
        x.append([val])
        y.append(y_val)
    # Add 40 rows where x0 is missing and y is 10.0
    for _ in range(40):
        x.append([None])
        y.append(10.0)

    tree = ModelTree(max_depth=1, leaf="mean")
    tree.fit(x, y)

    # Predict on missing vs observed
    pred_missing = tree.predict([[None]])[0]
    pred_low = tree.predict([[1.0]])[0]
    pred_high = tree.predict([[8.0]])[0]
    print("== 1. Learned missing direction (missing_left) ==")
    print(f"  split details: {tree.splits()}")
    print(f"  pred(x=None): {pred_missing:.3f} (matches pred(x=8.0): {pred_high:.3f}, pred(x=1.0): {pred_low:.3f})")


def run_surrogate_splits():
    # x0 is the primary predictor with a clean step at 0.5 (occasionally missing).
    # x1 is a noisy correlated mirror of x0 (always observed).
    # Primary split is on x0.
    # When x0 is missing at prediction time:
    #   - maxsurrogate=0: both rows fall back to missing_left (identical output).
    #   - maxsurrogate=1: x1 acts as surrogate, correctly routing low vs high.
    def noise(i: int) -> float:
        return math.sin(i * 12.9898 + 78.233)

    x = [[None if i % 20 == 0 else i / 300.0, i / 300.0 + 0.2 * noise(i)] for i in range(300)]
    y = [5.0 if i / 300.0 >= 0.5 else 0.0 for i in range(300)]

    tree_no_surr = ModelTree(max_depth=1, leaf="mean", min_rows=10, l2=0.0, maxsurrogate=0)
    tree_no_surr.fit(x, y)

    tree_surr = ModelTree(max_depth=1, leaf="mean", min_rows=10, l2=0.0, maxsurrogate=1)
    tree_surr.fit(x, y)

    p_lo_nosurr = tree_no_surr.predict([[None, 0.1]])[0]
    p_hi_nosurr = tree_no_surr.predict([[None, 0.9]])[0]

    p_lo_surr = tree_surr.predict([[None, 0.1]])[0]
    p_hi_surr = tree_surr.predict([[None, 0.9]])[0]

    print("\n== 2. Surrogate splits (maxsurrogate=1) ==")
    print(f"  primary split : {tree_surr.splits()[0]['col']} (threshold={tree_surr.splits()[0]['threshold']:.3f})")
    print(f"  maxsurrogate=0 (fallback only): pred(None, 0.1)={p_lo_nosurr:.3f}, pred(None, 0.9)={p_hi_nosurr:.3f}")
    print(f"  maxsurrogate=1 (routed by x1) : pred(None, 0.1)={p_lo_surr:.3f}, pred(None, 0.9)={p_hi_surr:.3f}")


def run_linear_leaf_na():
    # Linear leaf: target is 2 * x0 + 3 * x1. When x1 is missing, target has an offset of +5.
    x, y = [], []
    for i in range(200):
        x0 = float(i % 10)
        x1 = float((i * 3) % 7) if i % 5 != 0 else None
        y_val = 2.0 * x0 + (3.0 * x1 if x1 is not None else 5.0)
        x.append([x0, x1])
        y.append(y_val)

    tree = ModelTree(max_depth=1, leaf="linear", l2=0.01)
    tree.fit(x, y)

    leaves = tree.leaves()
    print("\n== 3. Linear leaves with NA indicator coefficients ==")
    for idx, lf in enumerate(leaves):
        print(f"  leaf {idx}: cols={lf['cols']} intercept={lf['intercept']:.3f} na_coef={[round(v, 3) for v in lf['na_coef']]}")

    p_observed = tree.predict([[5.0, 2.0]])[0]
    p_missing = tree.predict([[5.0, None]])[0]
    print(f"  predict([5.0, 2.0]) : {p_observed:.3f}")
    print(f"  predict([5.0, None]): {p_missing:.3f}")


def main():
    run_learned_direction()
    run_surrogate_splits()
    run_linear_leaf_na()


if __name__ == "__main__":
    main()
