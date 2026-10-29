"""Rust-model scorer for the Elo arena. Run THIRD:
    uv run --with pandas --with numpy --with forust --with perpetual \
           python examples/elo_score_rust.py
Reads benchmarks/elo_data/<ds>_{train,test}.csv, appends
(dataset,model,rmse,mae,fit_s) rows to benchmarks/elo_scores.csv.

Models: forust (jinlow/forust, XGBoost-parity GBDT in Rust) and perpetual
(self-generalizing GBM in Rust) - stock configs.
"""

import csv
import math
import time
from pathlib import Path

DATA_DIR = Path("benchmarks/elo_data")
OUT = Path("benchmarks/elo_scores.csv")
# Datasets are discovered from the prepared files - add data without
# touching the scorers.
DATASETS = sorted(p.name[: -len("_train.csv")] for p in DATA_DIR.glob("*_train.csv"))
import os
if os.environ.get("ELO_DATASETS"):  # groups-of-5 runs: score a subset
    _only = set(os.environ["ELO_DATASETS"].split(","))
    DATASETS = [d for d in DATASETS if d in _only]


def read_xy(path):
    x, y = [], []
    width = [0]
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)  # first row is the header, always
        width[0] = len(header) if header else 0
        for row in reader:
            if not row:
                continue
            if len(row) != width[0]:
                raise ValueError(f"{path}: row width {len(row)} != {width[0]}")
            x.append([None if v == "" else float(v) for v in row[:-1]])
            y.append(float(row[-1]))
    return x, y


def scores(pred, yte):
    err = [p - t for p, t in zip(pred, yte)]
    return (
        math.sqrt(sum(e * e for e in err) / len(err)),
        sum(abs(e) for e in err) / len(err),
    )


def append_row(ds, model, rmse, mae, fit):
    """Upsert: one row per (dataset, model); rerunning is incremental."""
    rows = []
    if OUT.exists():
        with open(OUT) as fh:
            rdr = csv.reader(fh)
            next(rdr, None)
            rows = [r for r in rdr if not (r[0] == ds and r[1] == model)]
    with open(OUT, "w") as fh:
        fh.write("dataset,model,rmse,mae,fit_s\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
        fh.write(f"{ds},{model},{rmse:.4f},{mae:.4f},{fit:.3f}\n")


def make_models():
    import numpy as np

    ms = {}

    def attempt(name, builder):
        try:
            ms[name] = builder()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

            def f(*_a, **_k):
                raise RuntimeError(err)

            ms[name] = f

    def forust_model():
        from forust import GradientBooster

        def run(xtr, ytr, xte):
            import pandas as pd

            cols = [f"x{i}" for i in range(len(xtr[0]))]
            m = GradientBooster(objective_type="SquaredLoss")  # regression
            t0 = time.perf_counter()
            m.fit(pd.DataFrame(xtr, columns=cols), pd.Series(ytr))
            return m.predict(pd.DataFrame(xte, columns=cols)), time.perf_counter() - t0

        return run

    def perpetual_model():
        from perpetual import PerpetualBooster

        def run(xtr, ytr, xte):
            m = PerpetualBooster(objective="SquaredLoss", budget=1.0)
            t0 = time.perf_counter()
            m.fit(np.asarray(xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64))
            return m.predict(np.asarray(xte, dtype=np.float64)), time.perf_counter() - t0

        return run

    attempt("forust (Rust)", forust_model)
    attempt("perpetual (Rust)", perpetual_model)
    return ms


def main():
    models = make_models()
    for ds in DATASETS:
        xtr, ytr = read_xy(DATA_DIR / f"{ds}_train.csv")
        xte, yte = read_xy(DATA_DIR / f"{ds}_test.csv")
        for name, fn in models.items():
            try:
                pred, fit = fn(xtr, ytr, xte)
                rmse, mae = scores([float(p) for p in pred], yte)
                append_row(ds, name, rmse, mae, fit)
                print(f"{ds:<9} {name:<16} rmse={rmse:8.3f} mae={mae:8.3f} fit={fit:6.2f}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"{ds:<9} {name:<16} FAILED ({msg[:70]})")
    print("Rust scorer done ->", OUT)


if __name__ == "__main__":
    main()
