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
from pathlib import Path

OUT = Path("benchmarks/elo_scores.csv")
DATASETS = ["concrete", "abalone", "ozone", "boston", "fc2008"]


def main():
    rows = list(csv.DictReader(open(OUT)))
    by = {(r["dataset"], r["model"]): (float(r["rmse"]), float(r["mae"]), float(r["fit_s"]))
          for r in rows}

    for ds in DATASETS:
        here = {m: v for (d, m), v in by.items() if d == ds}
        print(f"\n== {ds} ==")
        print(f"{'model':<24}{'RMSE':>9}{'MAE':>9}{'fit s':>8}")
        print("-" * 52)
        for m, (rmse, mae, fit) in sorted(here.items(), key=lambda kv: kv[1][0]):
            print(f"{m:<24}{rmse:>9.3f}{mae:>9.3f}{fit:>8.2f}")

    ratings = {m: 1000.0 for (_, m) in by}
    k = 32.0
    for ds in DATASETS:
        here = {m: v[0] for (d, m), v in by.items() if d == ds}
        for a, b in itertools.combinations(sorted(here), 2):
            sa = 1.0 if here[a] < here[b] else (0.0 if here[a] > here[b] else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (sa - ea)
            ratings[b] += k * ((1.0 - sa) - (1.0 - ea))

    print("\n== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==")
    for m, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        n = sum(1 for (_, mm) in by if mm == m)
        print(f"{r:7.1f}  {m}  ({n}/5 datasets)")
    print("\nTabArena anchor (different protocol - reference only):")
    print("  1955.0  Kumo-Tabular (best overall)")
    print("  1411.0  LightGBM tuned+ens (best tree-based)")


if __name__ == "__main__":
    main()
