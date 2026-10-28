"""Grounded validation of monotone + interaction constraints.

Oracles (ground truth = reference implementations' observable behavior):
- monotone: R `gbm`'s `monotone` argument (same -1/0/+1 convention) via
  examples/constraints_check.R, on real domain data where the monotone
  direction is a documented fact (healthcare: charges rise with age);
- interaction groups: the shared property (no root-to-leaf path mixes
  groups), verified structurally via ModelTree.split_paths() and
  cross-checked against LightGBM's per-tree split features.

Run:
    uv run --with pandas --with numpy --with lightgbm python examples/constraints_check.py
"""

import math
import subprocess
from pathlib import Path

from mltreelib import ModelTree

DATA_DIR = Path("benchmarks/elo_data")
ENV = {"R_LIBS_USER": str(Path.home() / "R/library")}


def read_xy(path):
    import csv

    x, y = [], []
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if row:
                x.append([None if v == "" else float(v) for v in row[:-1]])
                y.append(float(row[-1]))
    return x, y, (header or [])


def rmse(pred, y):
    err = [p - t for p, t in zip(pred, y)]
    return math.sqrt(sum(e * e for e in err) / len(err))


def monotone_ok(preds):
    return all(b >= a - 1e-9 for a, b in zip(preds, preds[1:]))


def main():
    xtr, ytr, hdr = read_xy(DATA_DIR / "ta_healthcare_train.csv")
    xte, yte, _ = read_xy(DATA_DIR / "ta_healthcare_test.csv")
    age_col = hdr.index("age")
    mono = [0] * (len(hdr) - 1)
    mono[age_col] = 1

    # Probe grid: sweep age, hold others at medians.
    med = []
    for j in range(len(xtr[0])):
        vals = sorted(v for v in (r[j] for r in xtr) if v is not None)
        med.append(vals[len(vals) // 2] if vals else 0.0)
    grid = []
    for i in range(24):
        r = list(med)
        r[age_col] = 18.0 + i * (64.0 - 18.0) / 23.0
        grid.append(r)

    print("== monotone (+1 on age, healthcare: charges rise with age) ==")
    print("  (R gbm oracle has constant leaves -> compare at mean-leaf semantics;")
    print("   linear leaves: within-leaf slopes are sign-projected and boundaries")
    print("   at splits on age gate on the children's fitted outputs at the)")
    for leaf in ("mean", "linear"):
        ours_u, ours_c = ModelTree(leaf=leaf, l1=0.0, l2=0.0), ModelTree(
            monotone=mono, leaf=leaf, l1=0.0, l2=0.0
        )
        ours_u.fit(xtr, ytr)
        ours_c.fit(xtr, ytr)
        pu, pc = ours_u.predict(grid), ours_c.predict(grid)
        print(f"  ours/{leaf:<6} unconstrained monotone: {monotone_ok(pu)}"
              f"   constrained: {monotone_ok(pc)}")
        print(f"  ours/{leaf:<6} rmse uncon {rmse(ours_u.predict(xte), yte):.3f}"
              f" -> constrained {rmse(ours_c.predict(xte), yte):.3f}")

    import os

    r = subprocess.run(
        ["Rscript", "examples/constraints_check.R"],
        capture_output=True, text=True,
        env={**os.environ, "R_LIBS_USER": str(Path.home() / "R/library")},
    )
    if r.returncode == 0:
        for line in r.stdout.strip().splitlines():
            print("  " + line)
    else:
        print(f"  R gbm oracle skipped: {r.stderr.strip().splitlines()[-1:]}")

    print("\n== interaction groups [[0,1],[2,3]] (property: no mixed paths) ==")
    import numpy as np

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(2000, 4))
    Y = 3 * X[:, 0] - 2 * X[:, 1] + 8.0 * ((X[:, 0] > 0.5) & (X[:, 2] > 0.0))
    x_rows = [[float(v) for v in r] for r in X]
    y_list = [float(v) for v in Y]
    groups = [[0, 1], [2, 3]]
    # split_paths returns column names (x0..x3 for default-named fits)
    g_of = lambda name: 0 if int(name[1:]) in (0, 1) else 1

    ours_g = ModelTree(interaction_groups=groups, max_depth=6, min_rows=20, l1=0.0, l2=0.0)
    ours_g.fit(x_rows, y_list)
    mixed = [
        p for p in ours_g.split_paths()
        if len({g_of(c) for c in p}) > 1
    ]
    ours_u2 = ModelTree(max_depth=6, min_rows=20, l1=0.0, l2=0.0)
    ours_u2.fit(x_rows, y_list)
    mixed_u = [
        p for p in ours_u2.split_paths()
        if len({g_of(c) for c in p}) > 1
    ]
    print(f"  ours      mixed paths constrained: {len(mixed)} (want 0)"
          f"   unconstrained: {len(mixed_u)} (want > 0)")

    try:
        import lightgbm as lgb

        b = lgb.train(
            {"objective": "regression", "verbosity": -1, "num_leaves": 31,
             "interaction_constraints": [[0, 1], [2, 3]]},
            lgb.Dataset(X, label=Y), num_boost_round=200,
        )
        bad = 0
        trees = b.dump_model()["tree_info"]

        def feats(node):
            out = []
            if "split_feature" in node:
                out.append(node["split_feature"])
                out += feats(node["left_child"]) + feats(node["right_child"])
            return out

        for t in trees:
            f = set(feats(t["tree_structure"]))
            if len(f) > 1 and not (f <= {0, 1} or f <= {2, 3}):
                bad += 1
        print(f"  lightgbm  trees mixing groups: {bad}/{len(trees)} (want 0)")
    except Exception as e:
        print(f"  lightgbm  skipped ({type(e).__name__})")


if __name__ == "__main__":
    main()
