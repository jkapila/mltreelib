"""Budgeted Elo orchestrator: datasets outer, algorithms inner.

Every (algorithm x dataset) runs as a SECONDARY command-line process in its
own session under examples/elo_budget.run_with_budget (300s hard wall-clock
budget; on lapse the whole process group dies - main and secondary PIDs
together). Already-scored pairs are skipped; every attempt is recorded in a
separate ledger so timeouts count as attempts (set ELO_RETRY=1 to retry them).

Supports:
    python examples/elo_score_all.py --task reg
    python examples/elo_score_all.py --task cla
    ELO_DATASETS=ta_fiat500 python examples/elo_score_all.py --task reg
"""

import argparse
import os
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from elo_budget import run_with_budget
from elo_common import (
    AttemptsLedger,
    BUDGET_S,
    CRITICAL_S,
    SLOW_S,
    ScoreSheet,
    discover_datasets,
    setup_logging,
)
import elo_models_reg
import elo_models_cla

log = setup_logging("elo.orchestrator")

R_MODELS_REG = [
    "rpart", "cubist", "lmtree", "glmtree", "ctree", "grf_llf",
    "grf_rf", "earth", "mboost", "ranger", "bart", "fru",
]

R_MODELS_CLA = [
    "rpart", "c50_tree", "c50_rules", "ctree", "glmtree", "earth", "mboost", "ranger", "bart", "grf_rf",
]


def parse_result_reg(text: str, ds: str, name: str):
    for line in text.splitlines():
        if line.startswith("RESULT "):
            parts = line[len("RESULT "):].split(",")
            if len(parts) == 5 and parts[0] == ds and parts[1] == name:
                try:
                    return float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    pass
    return None


def parse_result_cla(text: str, ds: str, name: str):
    for line in text.splitlines():
        if line.startswith("RESULT "):
            parts = line[len("RESULT "):].split(",")
            if len(parts) == 8 and parts[0] == ds and parts[1] == name:
                try:
                    logloss = float(parts[2])
                    acc = float(parts[3])
                    bal_acc = float(parts[4])
                    auc = float(parts[5]) if parts[5] != "" else None
                    macro_f1 = float(parts[6])
                    fit_s = float(parts[7])
                    return logloss, acc, bal_acc, auc, macro_f1, fit_s
                except ValueError:
                    pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Budgeted Elo Benchmark Orchestrator")
    parser.add_argument("--task", choices=["reg", "cla"], default="reg",
                        help="Benchmark task type ('reg' for regression, 'cla' for classification)")
    parser.add_argument("--budget", type=float, default=BUDGET_S,
                        help=f"Per-fit hard timeout budget in seconds (default {BUDGET_S}s)")
    args = parser.parse_args()

    task = args.task
    budget_s = args.budget

    if task == "reg":
        data_dir = Path("benchmarks/elo_data")
        scores_csv = Path("benchmarks/elo_scores.csv")
        columns = ["dataset", "model", "lang", "cat_handling", "rmse", "mae", "fit_s"]
        py_models = sorted(elo_models_reg.make_models().keys())
        r_models = R_MODELS_REG
        models_to_run = r_models + py_models
    else:
        data_dir = Path("benchmarks/elo_cla_data")
        scores_csv = Path("benchmarks/elo_cla_scores.csv")
        columns = ["dataset", "model", "lang", "cat_handling", "logloss", "accuracy", "bal_acc", "auc", "macro_f1", "fit_s"]
        py_models = sorted(elo_models_cla.make_models().keys())
        r_models = R_MODELS_CLA
        models_to_run = r_models + py_models

    exclude = [s.strip() for s in os.environ.get("ELO_EXCLUDE", "").split(",") if s.strip()]
    if exclude:
        models_to_run = [m for m in models_to_run if not any(x in m for x in exclude)]

    sheet = ScoreSheet(scores_csv, columns)
    attempts = AttemptsLedger()

    done = sheet.done_pairs()
    if not os.environ.get("ELO_RETRY"):
        done |= attempts.attempted_pairs()

    env = dict(os.environ)
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))

    datasets = discover_datasets(data_dir)
    log.info(f"Starting Elo run: task={task}, datasets={len(datasets)}, models={len(models_to_run)}")

    for ds in datasets:
        log.info(f"=== dataset: {ds} ===")
        for name in models_to_run:
            if (ds, name) in done:
                continue

            log.info(f"{ds:<24}{name:<20} RUNNING")
            if name in r_models:
                argv = ["Rscript", "--vanilla", str(EXAMPLES_DIR / "elo_r_worker.R"), task, ds, name]
            else:
                argv = [sys.executable, "-u", str(EXAMPLES_DIR / "elo_fit_worker.py"), task, ds, name]

            t0 = time.monotonic()
            out, code, timed_out = run_with_budget(argv, budget_s=budget_s, env=env)
            dt = time.monotonic() - t0

            if dt > CRITICAL_S:
                log.warning("%-24s%-20s CRITICAL fit=%.1fs (budget %.0fs)", ds, name, dt, budget_s)
            elif dt > SLOW_S:
                log.info("%-24s%-20s slow fit=%.1fs (TabArena line %.0fs)", ds, name, dt, SLOW_S)

            if timed_out:
                attempts.record(ds, name, "timeout", dt)
                log.warning("%-24s%-20s TIMEOUT (%.0fs, killed)", ds, name, budget_s)
                continue

            if task == "reg":
                res = parse_result_reg(out, ds, name)
                if code == 0 and res:
                    rmse, mae, fit = res
                    sheet.upsert(ds, name, rmse=rmse, mae=mae, fit_s=fit)
                    attempts.record(ds, name, "ok", dt)
                    log.info(f"{ds:<24}{name:<20} rmse={rmse:9.3f} mae={mae:9.3f} fit={fit:6.2f}s")
                else:
                    attempts.record(ds, name, "error", dt)
                    tail = [l for l in out.splitlines() if l.strip()][-1:]
                    log.info(f"{ds:<24}{name:<20} FAILED (exit {code}{': ' + tail[0][:60] if tail else ''})")
            else:
                res = parse_result_cla(out, ds, name)
                if code == 0 and res:
                    logloss, acc, bal_acc, auc, macro_f1, fit = res
                    auc_s = f"{auc:.4f}" if auc is not None else ""
                    sheet.upsert(ds, name, logloss=logloss, accuracy=acc, bal_acc=bal_acc, auc=auc_s, macro_f1=macro_f1, fit_s=fit)
                    attempts.record(ds, name, "ok", dt)
                    log.info(f"{ds:<24}{name:<20} logloss={logloss:7.4f} acc={acc:6.4f} fit={fit:6.2f}s")
                else:
                    attempts.record(ds, name, "error", dt)
                    tail = [l for l in out.splitlines() if l.strip()][-1:]
                    log.info(f"{ds:<24}{name:<20} FAILED (exit {code}{': ' + tail[0][:60] if tail else ''})")

        log.info(f"--- dataset complete: {ds} ---")

    log.info(f"=== Orchestrator finished ({task}) -> {scores_csv} ===")


if __name__ == "__main__":
    main()
