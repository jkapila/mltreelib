"""Rebuild the local comparison with the corrected Diamonds peer field."""
import ast
import csv
import itertools
import json
import logging
import math
from pathlib import Path
import re
import statistics
import sys

PEERS = Path(__file__).resolve().parent
RUN = PEERS.parent
OWN = "ModelTree(all)"
FOCUS = ["CatBoost(200)", "LightGBM(200)", "ChimeraBoost"]


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


old_reg = read(RUN / "frozen_reg.csv")
new_reg = read(RUN / "comparison_rerun_reg.csv")
diamonds = read(PEERS / "scores.csv")
expected = {r["model"] for r in old_reg if r["dataset"] == "ta_diamonds"}
assert len(diamonds) == len(expected) and {r["model"] for r in diamonds} == expected
assert all(r["dataset"] == "ta_diamonds" and math.isfinite(float(r["rmse"])) for r in diamonds)
reg = new_reg + diamonds
cla = read(RUN / "comparison_rerun_cla.csv")
write(PEERS / "corrected_reg_field.csv", reg)

source = ast.parse((RUN / "examples" / "elo_table.py").read_text())
functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
             and n.name in {"compute_elo_ratings", "generate_overall"}]
board = PEERS / "corrected_overall.md"
namespace = dict(csv=csv, itertools=itertools, statistics=statistics, log=logging.getLogger("elo"),
                 SCORES_REG=PEERS / "corrected_reg_field.csv",
                 SCORES_CLA=RUN / "comparison_rerun_cla.csv", DOC_OVERALL=board)
exec(compile(ast.Module(body=functions, type_ignores=[]), "archived_elo_table", "exec"), namespace)
namespace["generate_overall"]()
content = board.read_text().replace(
    "Evaluates models across **Rust, Python, and R** under identical 300s wall-clock isolation.",
    "Uses fresh v0.4.11 ModelTree(all) scores and the corrected Diamonds field. Other peer scores are frozen.",
)
board.write_text(content)
elo = {
    model: dict(rank=int(rank), overall=float(rating))
    for rank, rating, model in re.findall(
        r"<tr><td [^>]+>(\d+)</td><td [^>]+><b>([\d.]+)</b></td><td [^>]+><b>([^<]+)</b></td>",
        content,
    )
}

summary = {}
for model in FOCUS:
    entry = {}
    for metric, rows in (("rmse", reg), ("accuracy", cla), ("logloss", cla)):
        scores = {}
        for row in rows:
            if row["model"] in (OWN, model):
                loss = float(row[metric])
                if math.isfinite(loss):
                    scores.setdefault(row["dataset"], {})[row["model"]] = loss
        win = tie = loss = 0
        for pair in scores.values():
            if len(pair) != 2:
                continue
            diff = pair[model] - pair[OWN]
            diff *= -1 if metric == "accuracy" else 1
            win += diff > 0
            tie += diff == 0
            loss += diff < 0
        entry[metric] = dict(wins=win, ties=tie, losses=loss)
    summary[model] = entry

result = dict(elo={name: elo[name] for name in [OWN, *FOCUS]}, modeltree_head_to_head=summary)
(PEERS / "focused_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
if sys.argv[1:] == ["--table"]:
    print(f"{'Rank':>4} {'Model':<32} {'Elo':>8}")
    for name, entry in sorted(elo.items(), key=lambda pair: pair[1]["rank"]):
        print(f"{entry['rank']:>4} {name:<32} {entry['overall']:>8.1f}")
else:
    print(json.dumps(result, indent=2))
