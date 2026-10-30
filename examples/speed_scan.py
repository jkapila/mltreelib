"""Speed scan: linear-leaf fit cost across representative shapes.

The incremental-factorization speed ticket (#7) changed how the linear
scan scores candidates: maintained Cholesky factors (rank-1 updates per
row) instead of a fresh least-squares fit per candidate. This example
prints real fit times and held-out rmse across three shapes — wide, long
and categorical — so the speed claim is verifiable output, not marketing.

Run:  uv run python examples/speed_scan.py
"""

import csv
import os
import random
import time

from mltreelib import ModelTree

ELO = "benchmarks/elo_data"


def rows_from_csv(path):
    with open(path) as f:
        r = list(csv.reader(f))
    head, body = r[0], r[1:]
    yi = head.index("y")
    x = [[float(v) if v != "" else None for j, v in enumerate(row) if j != yi] for row in body]
    y = [float(row[yi]) for row in body]
    return x, y


def rmse(a, b):
    return (sum((u - v) ** 2 for u, v in zip(a, b)) / len(b)) ** 0.5


def synthetic(shape, n, d, seed=7):
    rng = random.Random(seed)
    x, y = [], []
    signal = [rng.uniform(-2, 2) for _ in range(d)]
    for _ in range(n):
        row = []
        s = 0.0
        for j in range(d):
            if shape == "categorical" and j % 3 == 0:
                v = float(rng.randrange(4))
            else:
                v = rng.uniform(-1, 1)
            row.append(v)
            s += signal[j] * v
        x.append(row)
        y.append(s + rng.gauss(0, 0.5))
    return x, y


def bench(name, xtr, ytr, xte, yte):
    dts = ["numeric"] * len(xtr[0])
    for leaf in ("linear", "mean"):
        m = ModelTree(max_depth=6, min_rows=50, leaf=leaf)
        t0 = time.perf_counter()
        m.fit(xtr, ytr, dtypes=dts)
        dt = time.perf_counter() - t0
        score = rmse(m.predict(xte), yte)
        print(f"{name:<24}{leaf:<8}{dt:>9.3f}s{score:>10.4f}{m.n_leaves():>7}")


print(f"{'shape':<24}{'leaf':<8}{'fit':>10}{'rmse':>10}{'leaves':>7}")
for shape, n, d in [("wide 2k x 40", 2000, 40), ("long 20k x 8", 20000, 8), ("categorical 10k x 12", 10000, 12)]:
    x, y = synthetic(shape.split()[0], n, d)
    split = int(0.8 * n)
    bench(shape, x[:split], y[:split], x[split:], y[split:])

# Real data when the prepared splits are present (superconductivity: the
# shape where the per-candidate refits used to dominate).
train = os.path.join(ELO, "ta_superconductivity_train.csv")
test = os.path.join(ELO, "ta_superconductivity_test.csv")
if os.path.exists(train) and os.path.exists(test):
    (xtr, ytr), (xte, yte) = rows_from_csv(train), rows_from_csv(test)
    bench("superconductivity", xtr, ytr, xte, yte)
else:
    print("(benchmark data not present — synthetic shapes only)")
