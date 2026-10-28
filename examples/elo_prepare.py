"""Prepare the Elo arena data ONCE: five real regression datasets, one
80/20 split each (seed 7), identical one-hot encoding for every scorer.

Writes benchmarks/elo_data/<ds>_train.csv and <ds>_test.csv (predictors as
x0..xN, target as y; "N/A"/"?" become empty cells = missing). Every scorer
(R / Python / Rust) reads these files, so no environment can drift.

Run:
    uv run python examples/elo_prepare.py
"""

import random
from pathlib import Path

DATA = Path("references/sources/cubist-demo/CubistR2/Data")
OUT = Path("benchmarks/elo_data")
SEED = 7

# target column, dropped columns, {categorical column: levels} for one-hot
DATASETS = {
    "concrete": {"target": 8, "drop": [], "cats": {}},
    "abalone": {"target": 8, "drop": [], "cats": {0: ["I", "M", "F"]}},
    "ozone": {"target": 8, "drop": [], "cats": {}},
    "boston": {
        "target": 7,
        "drop": [0, 1, 2, 3, 4, 5, 6, 19],  # OBS/TOWN/TOWN#/TRACT/LON/LAT/MEDV/B
        "cats": {11: ["0", "1"]},           # CHAS
    },
    "fc2008": {
        "target": 9,
        "drop": [1],  # make = label
        "cats": {
            0: ["CAR", "STATION WAGON", "SUV", "PICKUP TRUCK", "VAN"],
            4: ["Auto", "Manual"],
            6: ["F", "R", "4"],
            7: ["R", "P", "D", "E", "C"],
            8: ["T", "S", "-"],
        },
    },
}


def load(name):
    cfg = DATASETS[name]
    rows, y = [], []
    for line in (DATA / f"{name}.data").read_text().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        y.append(float(parts[cfg["target"]]))
        feats = []
        for i, p in enumerate(parts):
            if i == cfg["target"] or i in cfg["drop"]:
                continue
            if i in cfg["cats"]:
                for lev in cfg["cats"][i]:
                    feats.append(1.0 if p == lev else 0.0)
            else:
                # "N/A"/"?" are missing values — empty CSV cell.
                feats.append(None if p in ("N/A", "?", "") else float(p))
        rows.append(feats)
    return rows, y


def write_csv(path, x, y):
    with open(path, "w") as fh:
        fh.write(",".join(f"x{i}" for i in range(len(x[0]))) + ",y\n")
        for row, t in zip(x, y):
            cells = ["" if v is None else str(v) for v in row]
            fh.write(",".join(cells) + f",{t}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        rows, y = load(ds)
        idx = list(range(len(y)))
        random.Random(SEED).shuffle(idx)
        n_tr = int(0.8 * len(y))
        tr, te = idx[:n_tr], idx[n_tr:]

        def pick(a, which):
            return [a[k] for k in which]

        xtr, xte = pick(rows, tr), pick(rows, te)
        ytr, yte = pick(y, tr), pick(y, te)
        write_csv(OUT / f"{ds}_train.csv", xtr, ytr)
        write_csv(OUT / f"{ds}_test.csv", xte, yte)
        na = sum(v is None for row in rows for v in row)
        print(f"{ds:<10} {len(ytr):>5} train / {len(yte):>4} test, "
              f"{len(rows[0]):>2} cols, {na} missing cells")


if __name__ == "__main__":
    main()
