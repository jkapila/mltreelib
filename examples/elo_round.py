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
import datetime
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from mltreelib import Metrics, ModelTree

OUT = Path("benchmarks/elo_scores.csv")
BASE = Path("benchmarks/elo_baselines.csv")
DATA = Path("benchmarks/elo_data")
LOG = Path("benchmarks/elo_round.log")

log = logging.getLogger("elo.round")


def setup_logging():
    """Standard file+console logging for Elo runs (the pattern the rest of
    the codebase adopts later)."""
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.handlers = [fh, sh]
OURS = [
    "ModelTree(all)", "ModelTree(path)", "ModelTree(mean)", "ModelTree(winnow)",
    "ModelTree(alpha0)", "ModelTree(alpha.03)",
]

# The exact settings this round scores — the round manifest records this
# dict verbatim, and `variants()` builds from it, so the record can never
# drift from what produced the numbers.
_BASE = dict(
    max_depth=4, min_rows=10, min_gain=1e-9, leaf_cols="all", leaf="linear",
    l2=0.1, l1=0.0, monotone=[], interaction_groups=[], min_relevance=None,
    gain="sse", split_precision=0.001, cat_order="frequency_desc",
    cat_search="cumulative", leaf_digest=True, verbose=0, alpha=0.0,
    imbalance_penalty=0.0, col_sample=None, seed=42, prune="none",
)
SETTINGS = {
    "ModelTree(all)": dict(_BASE),
    "ModelTree(path)": dict(_BASE, leaf_cols="path"),
    "ModelTree(mean)": dict(_BASE, leaf="mean"),
    "ModelTree(winnow)": dict(_BASE, min_relevance=0.05),
    # Candidate defaults under evaluation (user spec 2026-10-29), run as two
    # alpha variants; the round decides which becomes the default.
    "ModelTree(alpha0)": dict(
        _BASE,
        max_depth=12, min_rows=6, l2=6.0, split_precision=None,
        leaf="linear", gain="sse", prune="bic",
        imbalance_penalty=0.06, alpha=0.0,
    ),
    "ModelTree(alpha.03)": dict(
        _BASE,
        max_depth=12, min_rows=6, l2=6.0, split_precision=None,
        leaf="linear", gain="sse", prune="bic",
        imbalance_penalty=0.06, alpha=0.03,
    ),
}


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
    return {name: ModelTree(**cfg) for name, cfg in SETTINGS.items()}


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
    setup_logging()
    log.info("round start: %d flavors, settings in this run's manifest", len(SETTINGS))
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
        log.info("baseline frozen (%d rows) -> %s", len(rows), BASE)

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
            log.info("%-24s%-20srmse=%10.3f fit=%7.2fs", ds, name, reg["rmse"], fit_s)

    write_scores(list(by.values()))

    # started -> reached
    base = list(csv.DictReader(open(BASE)))
    b_elo, c_elo = elo(base), elo(list(by.values()))
    log.info("== ours: started (frozen baseline) -> reached ==")
    log.info("%-20s%12s%10s%8s", "model", "ELO before", "ELO now", "delta")
    for m in OURS:
        b, c = b_elo.get(m), c_elo.get(m)
        if b is not None and c is not None:
            log.info("%-20s%12.1f%10.1f%+8.1f", m, b, c, c - b)

    # Provenance manifest: what settings created these numbers.
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = Path("benchmarks/rounds") / stamp
    rdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "round": stamp,
        "git_sha": sha,
        "settings": SETTINGS,
        "datasets": datasets,
        "results": list(by.values()),
        "elo_before": {m: b_elo.get(m) for m in OURS},
        "elo_after": {m: c_elo.get(m) for m in OURS},
    }
    (rdir / "settings.json").write_text(json.dumps(manifest, indent=2))
    deltas = " ".join(
        f"{m}:{b_elo.get(m, 0):.0f}->{c_elo.get(m, 0):.0f}" for m in OURS
    )
    log.info("round %s sha=%s datasets=%d %s", stamp, sha, len(datasets), deltas)
    log.info("round manifest: %s", rdir / "settings.json")


if __name__ == "__main__":
    main()
