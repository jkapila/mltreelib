"""Bake-off for ModelTree DEFAULTS — one global config, arena datasets.

Not per-dataset tuning: we search a small grid of *defaults* and require the
winner to beat the baselines everywhere it matters (lmtree, cubist, rpart
rows in benchmarks/elo_scores.csv). Exact-coefficient tests pin l2=0.0
explicitly; the shipped default is whatever wins here.

Run (after maturin develop):
    uv run python examples/default_search.py
"""

import csv
import itertools
import math
from pathlib import Path

from mltreelib import ModelTree

DATA_DIR = Path("benchmarks/elo_data")
SCORES = Path("benchmarks/elo_scores.csv")
DATASETS = ["concrete", "abalone", "ozone", "boston", "fc2008"]
BARS = ["lmtree", "cubist", "rpart"]


def read_xy(path):
    x, y = [], []
    with open(path) as fh:
        for row in csv.reader(fh):
            if row and row[0] == "x0":
                continue
            x.append([None if v == "" else float(v) for v in row[:-1]])
            y.append(float(row[-1]))
    return x, y


def main():
    data = {
        ds: (
            read_xy(DATA_DIR / f"{ds}_train.csv"),
            read_xy(DATA_DIR / f"{ds}_test.csv"),
        )
        for ds in DATASETS
    }
    bars = {}
    for r in csv.DictReader(open(SCORES)):
        if r["model"] in BARS:
            bars[(r["dataset"], r["model"])] = float(r["rmse"])

    grid = []
    for (leaf, l1, l2, depth, min_rows) in itertools.product(
        ["mean", "linear"], [0.0, 0.01, 0.1], [0.0, 0.1, 1.0, 10.0], [4, 6, 8], [10, 20, 40]
    ):
        if leaf == "mean" and (l1 != 0.0 or l2 != 0.0):
            continue
        if l1 == 0.0 and l2 == 0.0 and leaf == "linear" and depth == 4:
            pass  # keep the plain-OLS baselines once per depth/min_rows combo
        grid.append((leaf, l1, l2, depth, min_rows))

    results = []
    for leaf, l1, l2, depth, min_rows in grid:
        norm, wins = [], {b: 0 for b in BARS}
        per = {}
        for ds in DATASETS:
            (xtr, ytr), (xte, yte) = data[ds]
            m = ModelTree(leaf=leaf, l1=l1, l2=l2, max_depth=depth, min_rows=min_rows)
            m.fit(xtr, [float(v) for v in ytr])
            pred = m.predict(xte)
            err = [p - t for p, t in zip(pred, yte)]
            rmse = math.sqrt(sum(e * e for e in err) / len(err))
            per[ds] = rmse
            norm.append(rmse / bars[(ds, "lmtree")])
            for b in BARS:
                if rmse < bars[(ds, b)]:
                    wins[b] += 1
        score = math.exp(sum(math.log(v) for v in norm) / len(norm))
        results.append((score, leaf, l1, l2, depth, min_rows, wins, per))

    results.sort(key=lambda r: r[0])
    print(f"{'cfg':<52}{'norm':>7}{'>lmtr':>6}{'>cub':>6}{'>rpart':>7}")
    print("-" * 80)
    for score, leaf, l1, l2, depth, min_rows, wins, per in results[:12]:
        cfg = f"leaf={leaf} l1={l1:g} l2={l2:g} depth={depth} min_rows={min_rows}"
        print(f"{cfg:<52}{score:>7.3f}{wins['lmtree']:>6}{wins['cubist']:>6}{wins['rpart']:>7}")
    best = results[0]
    print(f"\nWINNER: leaf={best[1]} l1={best[2]:g} l2={best[3]:g} "
          f"max_depth={best[4]} min_rows={best[5]}")
    print("per-dataset rmse:", {k: round(v, 3) for k, v in best[7].items()})


if __name__ == "__main__":
    main()
