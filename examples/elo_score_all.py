"""Budgeted Elo orchestrator: datasets outer, algorithms inner.

Every (algorithm x dataset) runs as a SECONDARY command-line process in its
own session under examples/elo_budget.run_with_budget (180s hard wall-clock
budget; on lapse the whole process group dies - main and secondary PIDs
together). Already-scored pairs are skipped; every attempt is recorded in a
separate ledger so timeouts count as attempts (set ELO_RETRY=1 to retry
them). Logs are unbuffered: tail -f benchmarks/elo_run.log shows exactly
which algorithm is at which dataset.

    uv run --with pandas --with numpy --with lightgbm --with catboost \
      --with ctboost --with scikit-learn --with mars-earth --with forust \
      --with perpetual python -u examples/elo_score_all.py
    ELO_DATASETS=ta_healthcare,ta_protein python -u examples/elo_score_all.py
"""

import csv
import os
import sys
import time
from pathlib import Path

from elo_budget import run_with_budget
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("benchmarks/elo_run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("elo.orchestrator")

DATA_DIR = Path("benchmarks/elo_data")
OUT = Path("benchmarks/elo_scores.csv")
ATTEMPTS = Path("benchmarks/elo_attempts.csv")
STAGE = DATA_DIR / ".stage"
# Per-(model x dataset) time tiers: 30s = TabArena-toppers line (slow),
# 180s = CRITICAL, 300s = hard kill (process group dies).
SLOW_S = 30.0
CRITICAL_S = 180.0
BUDGET_S = 300.0

R_MODELS = ["rpart", "cubist", "lmtree", "glmtree", "ctree", "grf_llf",
            "grf_rf", "earth", "mboost"]


def datasets():
    ds = sorted(p.name[: -len("_train.csv")] for p in DATA_DIR.glob("*_train.csv"))
    if os.environ.get("ELO_DATASETS"):
        only = set(os.environ["ELO_DATASETS"].split(","))
        ds = [d for d in ds if d in only]
    return ds


def done_pairs():
    """(dataset, model) already committed OR already attempted (unless retry)."""
    done = set()
    if OUT.exists():
        with open(OUT) as fh:
            done |= {(r["dataset"], r["model"]) for r in csv.DictReader(fh)}
    if not os.environ.get("ELO_RETRY") and ATTEMPTS.exists():
        with open(ATTEMPTS) as fh:
            done |= {(r["dataset"], r["model"]) for r in csv.DictReader(fh)}
    return done


def record_attempt(ds, name, status, seconds):
    new = not ATTEMPTS.exists()
    with open(ATTEMPTS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "model", "status", "seconds"])
        w.writerow([ds, name, status, f"{seconds:.1f}"])


def stage_dataset(ds):
    """Stage the prepared split once per dataset (.npy) for the workers."""
    import numpy as np

    STAGE.mkdir(exist_ok=True)
    out = STAGE / ds
    if (out / "xtr.npy").exists():
        return
    out.mkdir(exist_ok=True)

    def read_xy(path):
        x, y = [], []
        with open(path) as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if not row:
                    continue
                x.append([float("nan") if v in ("", "N/A", "?") else float(v)
                          for v in row[:-1]])
                y.append(float(row[-1]))
        return np.asarray(x), np.asarray(y)

    xtr, ytr = read_xy(DATA_DIR / f"{ds}_train.csv")
    xte, yte = read_xy(DATA_DIR / f"{ds}_test.csv")
    np.save(out / "xtr.npy", xtr)
    np.save(out / "ytr.npy", ytr)
    np.save(out / "xte.npy", xte)
    np.save(out / "yte.npy", yte)


def parse_result(text, ds, name):
    for line in text.splitlines():
        if line.startswith("RESULT "):
            parts = line[len("RESULT "):].split(",")
            if len(parts) == 5 and parts[0] == ds and parts[1] == name:
                return float(parts[2]), float(parts[3]), float(parts[4])
    return None


def main():
    import elo_score_python
    import elo_score_rust

    py_models = sorted(set(elo_score_python.make_models())
                       | set(elo_score_rust.make_models()))
    done = done_pairs()
    env = dict(os.environ)
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))

    for ds in datasets():
        log.info(f"=== dataset: {ds} ===", flush=True)
        stage_dataset(ds)
        for name in R_MODELS + py_models:
            if (ds, name) in done:
                continue
            log.info(f"{ds:<24}{name:<20} RUNNING", flush=True)
            if name in R_MODELS:
                argv = ["Rscript", "--vanilla", "examples/elo_r_worker.R", ds, name]
            else:
                argv = [sys.executable, "-u", "examples/elo_fit_worker.py", ds, name]
            t0 = time.monotonic()
            out, code, timed_out = run_with_budget(argv, BUDGET_S, env=env)
            dt = time.monotonic() - t0
            if dt > CRITICAL_S:
                log.warning("%-24s%-20s CRITICAL fit=%.1fs (budget %.0fs)",
                            ds, name, dt, BUDGET_S)
            elif dt > SLOW_S:
                log.info("%-24s%-20s slow fit=%.1fs (TabArena line %.0fs)",
                         ds, name, dt, SLOW_S)
            if timed_out:
                record_attempt(ds, name, "timeout", dt)
                log.warning("%-24s%-20s TIMEOUT (%.0fs, killed)", ds, name, BUDGET_S)
                continue
            res = parse_result(out, ds, name)
            if code == 0 and res:
                rmse, mae, fit = res
                elo_score_python.append_row(ds, name, rmse, mae, fit)
                record_attempt(ds, name, "ok", dt)
                log.info(f"{ds:<24}{name:<20} rmse={rmse:9.3f} mae={mae:9.3f} fit={fit:6.2f}",
                      flush=True)
            else:
                record_attempt(ds, name, "error", dt)
                tail = [l for l in out.splitlines() if l.strip()][-1:]
                log.info(f"{ds:<24}{name:<20} FAILED (exit {code}"
                      f"{': ' + tail[0][:60] if tail else ''})", flush=True)
        log.info(f"--- dataset complete: {ds} ---", flush=True)
    print("=== orchestrator done ->", OUT, "===", flush=True)


if __name__ == "__main__":
    main()
