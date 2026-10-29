"""Fit one (algorithm x dataset) as a secondary command-line process.

The orchestrator runs this under examples/elo_budget.run_with_budget so the
whole process group dies at the wall-clock budget. The worker re-reads the
prepared split (CSV, or the .npy stage if present) and prints exactly one
machine-parseable line:

    RESULT <dataset>,<model>,<rmse>,<mae>,<fit_s>

usage: python elo_fit_worker.py <dataset> <model>
"""

import csv
import sys
import time
from pathlib import Path

DATA_DIR = Path("benchmarks/elo_data")
STAGE = DATA_DIR / ".stage"


def read_xy(path):
    x, y = [], []
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            x.append([None if v in ("", "N/A", "?") else float(v) for v in row[:-1]])
            y.append(float(row[-1]))
    return x, y


def load_split(ds):
    import numpy as np

    stage = STAGE / ds
    if (stage / "xtr.npy").exists():  # staged once per dataset by the orchestrator
        xtr = np.load(stage / "xtr.npy", allow_pickle=False)
        ytr = np.load(stage / "ytr.npy", allow_pickle=False)
        xte = np.load(stage / "xte.npy", allow_pickle=False)
        yte = np.load(stage / "yte.npy", allow_pickle=False)
        to_rows = lambda a: [[None if v != v else float(v) for v in r] for r in a]
        return to_rows(xtr), ytr.tolist(), to_rows(xte), yte.tolist()
    xtr, ytr = read_xy(DATA_DIR / f"{ds}_train.csv")
    xte, yte = read_xy(DATA_DIR / f"{ds}_test.csv")
    return xtr, ytr, xte, yte


def registry():
    import elo_score_python
    import elo_score_rust

    ms = dict(elo_score_python.make_models())
    ms.update(elo_score_rust.make_models())
    return ms


def scores(pred, yte):
    import math

    pairs = [(float(p), t) for p, t in zip(pred, yte) if p == p]
    if not pairs:
        raise RuntimeError("no finite predictions")
    err = [p - t for p, t in pairs]
    rmse = math.sqrt(sum(e * e for e in err) / len(err))
    mae = sum(abs(e) for e in err) / len(err)
    return rmse, mae


def main():
    ds, name = sys.argv[1], sys.argv[2]
    models = registry()
    if name not in models:
        raise SystemExit(f"unknown model: {name}")
    xtr, ytr, xte, yte = load_split(ds)
    pred, fit = models[name](xtr, ytr, xte)
    rmse, mae = scores(pred, yte)
    print(f"RESULT {ds},{name},{rmse:.4f},{mae:.4f},{fit:.3f}", flush=True)


if __name__ == "__main__":
    main()
