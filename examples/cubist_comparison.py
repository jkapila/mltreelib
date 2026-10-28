"""ModelTree vs Cubist (RuleQuest demo, Release 2.10) on the boston data.

Runs our model tree on exactly the rows and predictors Cubist's demo used
(demo caps training at 200 cases) and compares training error. Cubist's
numbers come from running `bin/cubistdemo -f Data/boston` (see
`references/research/ours-vs-c5-cubist.md` for the test plan).

Run (after extracting references/sources/cubist-demo/):
    uv run python examples/cubist_comparison.py
"""

import math
from pathlib import Path

from mltreelib import ModelTree

DATA = (
    Path("references/sources/cubist-demo/CubistR2/Data/boston.data")
)
# Column layout from boston.names: OBS, TOWN, TOWN#, TRACT, LON, LAT, MEDV,
# CMEDV(target), CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO,
# B, LSTAT. Cubist's "Attributes included" drops TOWN#/TRACT/LON/LAT/MEDV
# (ignored) and B — we match its 12 predictors exactly.
TARGET = 7
PREDS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
PRED_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT",
]

# Cubist 2.10 demo, `bin/cubistdemo -f Data/boston` (200 training cases):
CUBIST_AVG_ABS_ERR = 1.41
CUBIST_REL_ERR = 0.25
CUBIST_CORR = 0.97


def load():
    rows, y = [], []
    for line in DATA.read_text().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        y.append(float(parts[TARGET]))
        rows.append([float(parts[j]) for j in PREDS])
    return rows[:200], y[:200]  # demo cap: first 200 cases


def report(name, preds, y):
    n = len(y)
    errs = [p - t for p, t in zip(preds, y)]
    avg_abs = sum(abs(e) for e in errs) / n
    mean_y = sum(y) / n
    rel = sum(abs(p - t) for p, t in zip(preds, y)) / sum(abs(t - mean_y) for t in y)
    mp = sum(preds) / n
    cov = sum((p - mp) * (t - mean_y) for p, t in zip(preds, y))
    sp = math.sqrt(sum((p - mp) ** 2 for p in preds))
    st = math.sqrt(sum((t - mean_y) ** 2 for t in y))
    corr = cov / (sp * st) if sp and st else float("nan")
    print(f"{name:<28}{avg_abs:>10.3f}{rel:>10.3f}{corr:>10.3f}")


def main():
    x, y = load()
    print(f"{len(y)} training cases, {len(PREDS)} predictors (Cubist's own set)")
    print(f"{'model':<28}{'avg|err|':>10}{'rel err':>10}{'corr':>10}")
    print("-" * 58)
    print(f"{'Cubist 2.10 (5 rules)':<28}{CUBIST_AVG_ABS_ERR:>10.2f}"
          f"{CUBIST_REL_ERR:>10.2f}{CUBIST_CORR:>10.2f}")
    for name, kw in [
        ("ModelTree linear (All)", {"leaf": "linear"}),
        ("ModelTree linear (Path)", {"leaf": "linear", "leaf_cols": "path"}),
        ("ModelTree mean (CART)", {"leaf": "mean"}),
    ]:
        t = ModelTree(max_depth=6, min_rows=10, **kw)
        t.fit(x, [float(v) for v in y])
        report(name, t.predict(x), y)


if __name__ == "__main__":
    main()
