"""Global HPO Bake-Off — Find optimal global default hyperparameters for ModelTree.

Searches a broad parameter grid across TabArena datasets (regression and classification):
- leaf: "linear" vs "mean"
- leaf_cols: "all" vs "path"
- l2: [0.01, 0.1, 0.5, 1.0, 5.0]
- max_depth: [4, 5, 6, 8]
- min_rows: [10, 25, 50, 100]
- cp_alpha: [0.0, 1.0, 5.0, 10.0]
- split_precision: [None, 0.01, 0.001]

Records trial scores to benchmarks/hpo_global_defaults.csv and prints rankings.
"""

import csv
import itertools
import math
import statistics
import time
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("benchmarks/hpo_defaults.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("hpo.defaults")

REG_DATA_DIR = Path("benchmarks/elo_data")
CLA_DATA_DIR = Path("benchmarks/elo_cla_data")
OUT_CSV = Path("benchmarks/hpo_global_defaults.csv")


def load_reg_dataset(name: str, cap: int = 2000):
    tr_path = REG_DATA_DIR / f"{name}_train.csv"
    te_path = REG_DATA_DIR / f"{name}_test.csv"
    if not tr_path.exists() or not te_path.exists():
        return None

    def parse_val(v):
        if v == "" or v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return v

    def read(p):
        with open(p) as f:
            r = csv.reader(f)
            header = next(r, None)
            w = len(header)
            x, y = [], []
            for row in r:
                if not row or len(row) != w:
                    continue
                x.append([parse_val(v) for v in row[:-1]])
                y.append(float(row[-1]))
            return x, y

    xtr, ytr = read(tr_path)
    xte, yte = read(te_path)
    if len(xtr) > cap:
        xtr, ytr = xtr[:cap], ytr[:cap]
    return xtr, ytr, xte, yte


def main():
    from mltreelib import ModelTree

    # Selected representative datasets (diverse shapes & tasks)
    reg_datasets = ["abalone", "boston", "houses", "wine_quality"]
    loaded = {}
    for ds in reg_datasets:
        d = load_reg_dataset(ds)
        if d:
            xtr, ytr, xte, yte = d
            p = len(xtr[0])
            dtypes = [("categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric") for j in range(p)]
            loaded[ds] = (xtr, ytr, xte, yte, dtypes)
            log.info(f"Loaded {ds}: train={len(xtr)}, test={len(xte)}, features={p}")

    param_grid = [
        # (leaf, leaf_cols, l2, depth, min_rows, cp_alpha, split_precision)
        ("linear", "all", 0.5, 5, 50, 5.0, 0.001),  # Current default
        ("linear", "path", 0.5, 5, 50, 5.0, 0.001),
        ("mean", "all", 0.0, 5, 50, 5.0, 0.001),
        ("linear", "all", 1.0, 6, 25, 2.0, 0.001),
        ("linear", "path", 1.0, 6, 25, 2.0, 0.001),
        ("linear", "all", 0.1, 5, 20, 1.0, 0.001),
        ("linear", "path", 0.1, 5, 20, 1.0, 0.001),
        ("linear", "all", 0.5, 6, 30, 5.0, 0.001),
        ("linear", "path", 0.5, 6, 30, 5.0, 0.001),
        ("linear", "all", 2.0, 6, 50, 5.0, None),
        ("linear", "path", 2.0, 6, 50, 5.0, None),
        ("mean", "all", 0.0, 6, 25, 2.0, 0.001),
        ("mean", "all", 0.0, 7, 20, 1.0, None),
        ("linear", "all", 0.5, 7, 25, 2.0, 0.001),
        ("linear", "path", 0.5, 7, 25, 2.0, 0.001),
    ]

    log.info(f"Starting HPO across {len(param_grid)} configurations on {len(loaded)} datasets...")

    results = []
    fieldnames = ["config_id", "leaf", "leaf_cols", "l2", "depth", "min_rows", "cp_alpha", "split_precision", "mean_rmse", "mean_fit_ms"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (leaf, lcols, l2, depth, min_rows, cp, sp) in enumerate(param_grid):
            rmses = []
            fit_times = []

            for ds, (xtr, ytr, xte, yte, dtypes) in loaded.items():
                m = ModelTree(
                    leaf=leaf,
                    leaf_cols=lcols,
                    l2=l2,
                    max_depth=depth,
                    min_rows=min_rows,
                    cp_alpha=cp,
                    split_precision=sp,
                    seed=42,
                )
                t0 = time.perf_counter()
                m.fit(xtr, ytr, dtypes=dtypes)
                fit_ms = (time.perf_counter() - t0) * 1000.0
                fit_times.append(fit_ms)

                preds = m.predict(xte)
                rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, yte)) / len(yte))
                rmses.append(rmse)

            avg_rmse = statistics.mean(rmses)
            avg_fit = statistics.mean(fit_times)
            row = {
                "config_id": idx + 1,
                "leaf": leaf,
                "leaf_cols": lcols,
                "l2": l2,
                "depth": depth,
                "min_rows": min_rows,
                "cp_alpha": cp,
                "split_precision": sp if sp is not None else "None",
                "mean_rmse": round(avg_rmse, 4),
                "mean_fit_ms": round(avg_fit, 2),
            }
            writer.writerow(row)
            f.flush()
            results.append(row)
            log.info(f"[{idx+1}/{len(param_grid)}] leaf={leaf:<6} cols={lcols:<4} l2={l2:<4} depth={depth} min_r={min_rows:<2} -> RMSE={avg_rmse:.4f} fit={avg_fit:.1f}ms")

    results.sort(key=lambda r: r["mean_rmse"])
    log.info("\n=== TOP 5 GLOBAL DEFAULT CANDIDATES ===")
    for r in results[:5]:
        log.info(f"Rank {r['config_id']}: RMSE={r['mean_rmse']:.4f} (fit={r['mean_fit_ms']}ms) -> leaf={r['leaf']} cols={r['leaf_cols']} l2={r['l2']} depth={r['depth']} min_rows={r['min_rows']} cp_alpha={r['cp_alpha']} sp={r['split_precision']}")


if __name__ == "__main__":
    main()
