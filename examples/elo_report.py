"""Elo report: reads benchmarks/elo_scores.csv (written by the R, Python and
Rust scorers) and prints per-dataset tables plus our Elo ranking.

Protocol (ours, documented): deterministic round-robin over per-dataset RMSE
matchups, standard Elo K=32, start 1000. TabArena numbers are a reference
anchor only - their protocol differs.

Run:
    uv run python examples/elo_report.py
"""

import csv
import itertools
import statistics
from pathlib import Path

import logging

# Standard run logging: file + console (the AGENTS.md logging pattern).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("benchmarks/elo_run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("elo.report")

OUT = Path("benchmarks/elo_scores.csv")
DATA_DIR = Path("benchmarks/elo_data")
# Datasets are discovered from the prepared files - add data without
# touching the scorers.
DATASETS = sorted(p.name[: -len("_train.csv")] for p in DATA_DIR.glob("*_train.csv"))


def main():
    rows = [r for r in csv.DictReader(open(OUT)) if r["model"] != "model"]
    by = {(r["dataset"], r["model"]): (float(r["rmse"]), float(r["mae"]), float(r["fit_s"]))
          for r in rows}

    for ds in DATASETS:
        here = {m: v for (d, m), v in by.items() if d == ds}
        log.info(f"\n== {ds} ==")
        log.info(f"{'model':<24}{'RMSE':>9}{'MAE':>9}{'fit s':>8}")
        log.info("-" * 52)
        for m, (rmse, mae, fit) in sorted(here.items(), key=lambda kv: kv[1][0]):
            log.info(f"{m:<24}{rmse:>9.3f}{mae:>9.3f}{fit:>8.2f}")

    ratings = {m: 1000.0 for (_, m) in by}
    k = 32.0
    for ds in DATASETS:
        here = {m: v[0] for (d, m), v in by.items() if d == ds}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (sa - ea)
            ratings[b] += k * ((1.0 - sa) - (1.0 - ea))

    log.info("\n== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==")
    total = len({d for (d, _) in by})
    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    w = (11, 21, 26, 13)

    def row(*cells):
        return "|" + "|".join(c.center(x) for c, x in zip(cells, w)) + "|"

    line = "+" + "+".join("-" * x for x in w) + "+"
    log.info(line)
    log.info(row("ELO", "FIT TIME (±sd)", "MODEL", "DATA"))
    log.info(line)
    for m, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        n = sum(1 for (d, mm) in by if mm == m)
        xs = fits.get(m, [0.0])
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        log.info(row(f"{r:.1f}", f"{statistics.mean(xs):>6.2f} s (±{sd:.2f})", m,
                     f"{n}/{total}"))
    log.info(line)
    log.info("\nTabArena anchor (different protocol - reference only):")
    log.info("  1955.0  Kumo-Tabular (best overall)")
    log.info("  1411.0  LightGBM tuned+ens (best tree-based)")


if __name__ == "__main__":
    main()
