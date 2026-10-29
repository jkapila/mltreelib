"""One development round of the Elo board: re-score OUR models only.

The external field (R packages, LightGBM, CatBoost, CTBoost, forust,
perpetual, MARS, sklearn, modeltrees) is scored once and frozen in
benchmarks/elo_baselines.csv - "where we started". Every iteration re-runs the
ModelTree variants on the same prepared splits, upserts their rows in
benchmarks/elo_scores.csv, and prints started -> reached.

    uv run python examples/elo_round.py                      # all datasets
    ELO_DATASETS=abalone,boston uv run python examples/elo_round.py
    uv run python examples/elo_round.py --freeze             # re-anchor baseline
"""

import csv
import math
import os
import sys
import time
from pathlib import Path

from mltreelib import Metrics, ModelTree

OUT = Path("benchmarks/elo_scores.csv")
BASE = Path("benchmarks/elo_baselines.csv")
DATA = Path("benchmarks/elo_data")
OURS = ["ModelTree(all)", "ModelTree(path)", "ModelTree(mean)", "ModelTree(winnow)"]


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


def variants():
    return {
        "ModelTree(all)": ModelTree(),
        "ModelTree(path)": ModelTree(leaf_cols="path"),
        "ModelTree(mean)": ModelTree(leaf="mean"),
        "ModelTree(winnow)": ModelTree(min_relevance=0.05),
    }


def write_scores(rows):
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "rmse", "mae", "fit_s"])
        for r in rows:
            w.writerow([r["dataset"], r["model"], r["rmse"], r["mae"], r["fit_s"]])


def elo(rows):
    """Round-robin on per-dataset RMSE, K=32, start 1000 - the shared protocol."""
    by = {(r["dataset"], r["model"]): float(r["rmse"]) for r in rows}
    ratings = {m: 1000.0 for (_, m) in by}
    for ds in sorted({d for (d, _) in by}):
        here = {m: v for (d, m), v in by.items() if d == ds}
        names = sorted(here)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
                ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
                ratings[a] += 32.0 * (sa - ea)
                ratings[b] += 32.0 * ((1.0 - sa) - (1.0 - ea))
    return ratings


def main():
    rows = list(csv.DictReader(open(OUT)))
    if not rows:
        raise SystemExit("elo_scores.csv is empty - run examples/elo_run_groups.sh first")

    # Freeze the anchor exactly once (or on demand with --freeze).
    if not BASE.exists() or "--freeze" in sys.argv:
        with open(BASE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "model", "rmse", "mae", "fit_s"])
            for r in rows:
                w.writerow([r["dataset"], r["model"], r["rmse"], r["mae"], r["fit_s"]])
        print(f"baseline frozen ({len(rows)} rows) -> {BASE}")

    datasets = sorted({r["dataset"] for r in rows})
    if os.environ.get("ELO_DATASETS"):
        only = set(os.environ["ELO_DATASETS"].split(","))
        datasets = [d for d in datasets if d in only]

    by = {(r["dataset"], r["model"]): r for r in rows}
    for ds in datasets:
        xtr, ytr = read_xy(DATA / f"{ds}_train.csv")
        xte, yte = read_xy(DATA / f"{ds}_test.csv")
        for name, m in variants().items():
            t0 = time.perf_counter()
            m.fit(xtr, ytr)
            pred = m.predict(xte)
            fit_s = time.perf_counter() - t0
            reg = Metrics().regression(yte, pred)
            by[(ds, name)] = {
                "dataset": ds,
                "model": name,
                "rmse": f"{reg['rmse']:.4f}",
                "mae": f"{reg['mae']:.4f}",
                "fit_s": f"{fit_s:.3f}",
            }
            print(f"{ds:<24}{name:<20}rmse={reg['rmse']:>10.3f} fit={fit_s:7.2f}s")

    write_scores(list(by.values()))

    # started -> reached
    base = list(csv.DictReader(open(BASE)))
    b_elo, c_elo = elo(base), elo(list(by.values()))
    print("\n== ours: started (frozen baseline) -> reached ==")
    print(f"{'model':<20}{'ELO before':>12}{'ELO now':>10}{'delta':>8}")
    for m in OURS:
        b, c = b_elo.get(m), c_elo.get(m)
        if b is not None and c is not None:
            print(f"{m:<20}{b:>12.1f}{c:>10.1f}{c - b:>+8.1f}")


if __name__ == "__main__":
    main()
