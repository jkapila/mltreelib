"""Reproduce the published board and replace one entrant with the rerun."""
import ast
import csv
import itertools
import json
import logging
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAME = "ModelTree(all)"
cases = json.loads((HERE / "manifest.json").read_text())["cases"]


def read(name):
    with (HERE / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def rows_from_board(board):
    return {
        model: dict(rank=int(rank), overall=float(elo), regression=reg, classification=cla, coverage=coverage)
        for rank, elo, model, reg, cla, coverage in re.findall(
            r"<tr><td [^>]+>(\d+)</td><td [^>]+><b>([\d.]+)</b></td>"
            r"<td [^>]+><b>([^<]+)</b></td><td [^>]+>([^<]+)</td>"
            r"<td [^>]+>([^<]+)</td><td [^>]+>([^<]+)</td>", board,
        )
    }


board_code = ast.parse((HERE / "examples" / "elo_table.py").read_text())
board_funcs = [n for n in board_code.body if isinstance(n, ast.FunctionDef)
               and n.name in {"compute_elo_ratings", "generate_overall"}]
namespace = dict(csv=csv, itertools=itertools, statistics=statistics, log=logging.getLogger("elo.report"))
exec(compile(ast.Module(body=board_funcs, type_ignores=[]), "archived_elo_table", "exec"), namespace)


def render(label, reg, cla):
    target = HERE / f"{label}.md"
    namespace.update(SCORES_REG=HERE / f"{reg}.csv", SCORES_CLA=HERE / f"{cla}.csv", DOC_OVERALL=target)
    namespace["generate_overall"]()
    board = target.read_text()
    if label.startswith("comparison_"):
        board = (board
                 .replace("**56 benchmark datasets**", "**55 benchmark datasets**")
                 .replace("**18 Regression**", "**17 Regression**")
                 .replace("Leaderboard (56 Datasets)", "Leaderboard (55 Datasets)")
                 .replace("/56</td>", "/55</td>"))
        board = board.replace(
            "Evaluates models across **Rust, Python, and R** under identical 300s wall-clock isolation.",
            "Excludes Diamonds because the frozen splits use the wrong target. Peer scores are frozen.",
        )
        target.write_text(board)
    return rows_from_board(board)


old = render("frozen_overall", "frozen_reg", "frozen_cla")
published = subprocess.check_output(["git", "show", "v0.4.11:docs/benchmarks/elo_overall.md"], cwd=ROOT, text=True)
assert old == rows_from_board(published), "Archived score snapshots do not reproduce the published board"
print("Historical board reproduced:", old[NAME])
if sys.argv[1:] == ["--check-frozen"]:
    sys.exit()

coverage = {}
for task, metric in (("reg", "rmse"), ("cla", "accuracy")):
    previous = read(f"frozen_{task}")
    baseline_pairs = {r["dataset"] for r in previous if r["model"] == NAME and math.isfinite(float(r[metric]))}
    new = read(f"{task}_scores")
    assert all(r["model"] == NAME and math.isfinite(float(r[metric])) for r in new)
    assert sorted(r["dataset"] for r in new) == cases[task]
    attempts = {
        r["dataset"] for ledger in HERE.glob(f"{task}_attempts_*s.csv")
        for r in csv.DictReader(ledger.read_text().splitlines())
        if r["model"] == NAME and r["status"] == "ok"
    }
    assert attempts == set(cases[task])
    coverage[task] = dict(historical=len(baseline_pairs), scored=len(new), expected=len(cases[task]))

    def comparable(rows):
        return [r for r in rows if (task, r["dataset"]) != ("reg", "ta_diamonds")]

    for label, rows in (
        ("comparison_historical", comparable(previous)),
        ("comparison_rerun", comparable(
            [r for r in previous if r["model"] != NAME]
            + [dict(r, lang="Rust", cat_handling="internal") for r in new]
        )),
    ):
        with (HERE / f"{label}_{task}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(previous[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

old_comparable = render("comparison_historical", "comparison_historical_reg", "comparison_historical_cla")
new = render("comparison_rerun", "comparison_rerun_reg", "comparison_rerun_cla")
summary = dict(coverage=coverage, published=old[NAME], historical_comparable=old_comparable[NAME], rerun=new[NAME],
               catboost=new["CatBoost(200)"], lightgbm=new["LightGBM(200)"])
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
