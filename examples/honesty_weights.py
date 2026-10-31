"""Honest tree estimation and sample weights in ModelTree.

Demonstrates two advanced estimation capabilities:
1. Honest estimation (honesty=True, honesty_fraction=...):
   Partitions the data into a structure sample (used exclusively to evaluate
   splits and build tree architecture) and an estimation sample (used exclusively
   to fit leaf parameters and empirical target distributions). This eliminates
   in-sample optimism bias (grf-style honest trees).
2. Sample weights (fit(x, y, w=weights)):
   Allows assigning per-observation weights for importance weighting, survey sampling,
   or recency weighting in non-stationary series.
"""

from __future__ import annotations

import math
from mltreelib import ModelTree


def run_honesty_demo():
    print("== 1. Honest tree estimation (honesty=True) ==")
    # 300 observations with noisy observations
    def noise(i: int) -> float:
        return math.sin(i * 12.9898 + 78.233)

    x = [[float(i % 2), float(i % 5) / 5.0] for i in range(300)]
    # True target depends only on x1: 3.0 * x1, with substantial noise
    y = [3.0 * x[i][1] + 0.5 * noise(i) for i in range(300)]

    # Standard tree: uses all 300 rows for both split search and leaf fitting
    standard = ModelTree(max_depth=2, min_rows=10, l2=0.0, honesty=False)
    standard.fit(x, y)

    # Honest tree: 40% (120 rows) for structure search, 60% (180 rows) for leaf estimation
    honest = ModelTree(max_depth=2, min_rows=10, l2=0.0, honesty=True, honesty_fraction=0.4)
    honest.fit(x, y)

    std_leaves = standard.leaves()
    hon_leaves = honest.leaves()

    print(f"  standard tree: {len(std_leaves)} leaves, total leaf rows: {sum(l['n'] for l in std_leaves)}")
    print(f"  honest tree  : {len(hon_leaves)} leaves, estimation leaf rows: {sum(l['n'] for l in hon_leaves)} (structure used 40%)")

    for idx, (sl, hl) in enumerate(zip(std_leaves, hon_leaves)):
        print(f"    leaf {idx}: standard mean={sl['mean']:.3f} (n={sl['n']}) | honest mean={hl['mean']:.3f} (n={hl['n']})")


def run_sample_weights_demo():
    print("\n== 2. Weighted tree fitting (fit(x, y, w=...)) ==")
    # Regime: x0 in {0, 1}. Target has two conflicting clusters at x0=0:
    # 8 observations at y=2.0, and 2 observations at y=20.0
    x, y = [], []
    for _ in range(8):
        x.append([0.0])
        y.append(2.0)
    for _ in range(2):
        x.append([0.0])
        y.append(20.0)
    for _ in range(10):
        x.append([1.0])
        y.append(50.0)

    # Unweighted: mean at x0=0 is (8*2 + 2*20)/10 = 56/10 = 5.6
    unweighted = ModelTree(max_depth=1, leaf="mean", l2=0.0, min_rows=1)
    unweighted.fit(x, y)
    p_unweighted = unweighted.predict([[0.0]])[0]

    # Weighted: assign 10x weight to the rare y=20.0 events (cost-sensitive / high-consequence)
    # Weights: 1.0 for y=2.0, 10.0 for y=20.0
    w = [1.0] * 8 + [10.0] * 2 + [1.0] * 10
    # Expected weighted mean at x0=0: (8*2*1 + 2*20*10) / (8*1 + 2*10) = (16 + 400) / 28 = 416 / 28 = 14.857
    weighted = ModelTree(max_depth=1, leaf="mean", l2=0.0, min_rows=1)
    weighted.fit(x, y, w=w)
    p_weighted = weighted.predict([[0.0]])[0]

    print(f"  x0=0 target: eight rows at y=2.0, two rows at y=20.0")
    print(f"  unweighted prediction (equal weight): {p_unweighted:.3f}")
    print(f"  weighted prediction (10x rare weight) : {p_weighted:.3f}")


def main():
    run_honesty_demo()
    run_sample_weights_demo()


if __name__ == "__main__":
    main()
