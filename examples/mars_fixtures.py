"""Generate the shared MARS fixture suite ONCE (codex plan: fixture data is
generated here and shared across languages - matching seeds do not give
matching RNG streams across R/Python/Rust).

Writes one CSV per fixture to benchmarks/mars_fixtures/ with columns
x0..xk,y (plus a manifest.txt recording generation parameters). The R oracle
(examples/mars_oracle.R) and the future Rust implementation both consume
these files verbatim.

    uv run python examples/mars_fixtures.py
"""

import csv
import random
from pathlib import Path

OUT = Path("benchmarks/mars_fixtures")
SEED = 20261029
random.seed(SEED)


def hinge(v):
    return max(0.0, v)


def write(name, rows, k, note):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"x{j}" for j in range(k)] + ["y"])
        for r in rows:
            w.writerow([f"{v:.6f}" for v in r])
    (OUT / "manifest.txt").open("a").write(f"{name}\tx{k}\tn={len(rows)}\t{note}\n")


def r(n, k):
    return [[random.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.txt").write_text(f"seed={SEED}\nname\tk\tn\tnote\n")

    # f01: constant target
    xs = r(120, 3)
    write("f01_constant", [row + [3.0] for row in xs], 3, "y=3 exactly")

    # f02: exact linear
    xs = r(120, 3)
    write("f02_linear", [row + [2.0 * row[0] - row[1] + 1.0] for row in xs], 3,
          "y=2x0-x1+1")

    # f03: one hinge, no noise
    xs = r(150, 2)
    write("f03_one_hinge", [row + [3.0 * hinge(row[0] - 0.4)] for row in xs], 2,
          "y=3*max(0,x0-0.4)")

    # f04: multi-knot 1D spline in x0 (three hinges)
    xs = r(200, 2)
    write("f04_multiknot_1d",
          [row + [2.0 * hinge(row[0] + 0.5) - 3.0 * hinge(row[0])
                  + 1.5 * hinge(row[0] - 0.5)] for row in xs], 2,
          "three-hinge spline in x0")

    # f05: additive hinges + irrelevant predictors + noise
    xs = r(300, 4)
    write("f05_additive_noisy",
          [row + [4.0 * hinge(row[0] - 0.2) - 2.0 * hinge(-row[1] - 0.3)
                  + random.gauss(0.0, 0.2)] for row in xs], 4,
          "hinges on x0,x1; x2,x3 irrelevant; noise 0.2")

    # f06: two-variable hinge product (degree 2 needed)
    xs = r(300, 3)
    write("f06_product",
          [row + [5.0 * hinge(row[0] - 0.5) * hinge(row[1] + 0.3)] for row in xs], 3,
          "y=5*h(x0-.5)*h(x1+.3); degree 2 must beat degree 1")

    # f07/f08: Friedman-1 clean and noisy (10 features)
    def friedman(row, noise):
        import math
        y = (10.0 * math.sin(math.pi * row[0] * row[1])
             + 20.0 * (row[2] - 0.5) ** 2 + 10.0 * row[3] + 5.0 * row[4])
        return y + (random.gauss(0.0, 1.0) if noise else 0.0)
    xs = r(250, 10)
    write("f07_friedman1_clean", [row + [friedman(row, False)] for row in xs], 10,
          "Friedman-1 noiseless")
    xs = r(250, 10)
    write("f08_friedman1_noisy", [row + [friedman(row, True)] for row in xs], 10,
          "Friedman-1 noise sigma=1")

    # f09: edge cases in one set: repeated values, constant column,
    # near-collinear pair, extreme scale
    rows = []
    for i in range(160):
        x0 = -1.0 + 2.0 * ((i // 10) % 2)          # repeated values
        x1 = 7.0                                     # constant column
        x2 = random.uniform(-1.0, 1.0)
        x3 = x2 + 1e-9 * random.uniform(-1, 1)       # near-collinear with x2
        x4 = 1e6 * random.uniform(-1.0, 1.0)         # extreme scale
        yv = 2.0 * hinge(x0 - 0.0) + x2 + random.gauss(0, 0.05)
        rows.append([x0, x1, x2, x3, x4, yv])
    write("f09_edges", rows, 5, "repeated/constant/collinear/extreme-scale")

    # f10: tiny sample
    xs = r(12, 2)
    write("f10_tiny", [row + [1.0 + 2.0 * hinge(row[0])] for row in xs], 2,
          "n=12")

    print(f"wrote fixtures to {OUT} (seed {SEED})")


if __name__ == "__main__":
    main()
