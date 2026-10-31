"""Elo report: reads benchmarks/elo_scores.csv and prints per-dataset tables plus Elo ranking.

Protocol: deterministic round-robin over per-dataset RMSE matchups, Elo K=32, start 1000.
"""

import csv
from pathlib import Path
import statistics
import sys

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from elo_common import calculate_elo, discover_datasets, get_cat_handling, get_model_lang, setup_logging

log = setup_logging("elo.report")

OUT = Path("benchmarks/elo_scores.csv")
DATA_DIR = Path("benchmarks/elo_data")


def main():
    if not OUT.exists():
        log.warning(f"No score file found at {OUT}")
        return

    with open(OUT, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("model") and r.get("model") != "model" and r.get("rmse")]

    if not rows:
        log.info(f"No scores present in {OUT}")
        return

    datasets = discover_datasets(DATA_DIR)
    by = {(r["dataset"], r["model"]): (float(r["rmse"]), float(r["mae"]), float(r["fit_s"]), r.get("cat_handling") or get_cat_handling(r["model"]), r.get("lang") or get_model_lang(r["model"])) for r in rows}

    for ds in datasets:
        here = {m: v for (d, m), v in by.items() if d == ds}
        if not here:
            continue
        log.info(f"\n== {ds} ==")
        log.info(f"{'model':<24}{'lang':<8}{'handling':<10}{'RMSE':>9}{'MAE':>9}{'fit s':>8}")
        log.info("-" * 72)
        for m, (rmse, mae, fit, cat, lang) in sorted(here.items(), key=lambda kv: kv[1][0]):
            log.info(f"{m:<24}{lang:<8}{cat:<10}{rmse:>9.3f}{mae:>9.3f}{fit:>8.2f}")

    ratings = calculate_elo(rows, metric_col="rmse", lower_is_better=True)

    log.info("\n== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==")
    total = len({r["dataset"] for r in rows})
    fits = {}
    for r in rows:
        fits.setdefault(r["model"], []).append(float(r["fit_s"]))
    w = (11, 21, 24, 8, 12, 11)

    def row(*cells):
        return "|" + "|".join(c.center(x) for c, x in zip(cells, w)) + "|"

    line = "+" + "+".join("-" * x for x in w) + "+"
    log.info(line)
    log.info(row("ELO", "FIT TIME (±sd)", "MODEL", "LANG", "CAT", "DATA"))
    log.info(line)
    for m, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        n = sum(1 for (d, mm) in by if mm == m)
        xs = fits.get(m, [0.0])
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        cat = get_cat_handling(m)
        lang = get_model_lang(m)
        log.info(row(f"{r:.1f}", f"{statistics.mean(xs):>6.2f} s (±{sd:.2f})", m, lang, cat, f"{n}/{total}"))
    log.info(line)


if __name__ == "__main__":
    main()
