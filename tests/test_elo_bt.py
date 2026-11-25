"""Tests for the Bradley-Terry Elo engine in benchmarks/arena/elo_bt.py."""

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ARENA_DIR = REPO_ROOT / "benchmarks" / "arena"
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

from elo_bt import Round, fit_bt, pair_outcome, rate_rounds, round_wins

STRENGTH = {"A": 4.0, "B": 2.0, "C": 1.0}


def _plackett_luce_rounds(n_rounds: int, seed: int) -> list[Round]:
    """Each round is a random ranking drawn from Plackett-Luce; score = -position."""
    rng = np.random.default_rng(seed)
    names = list(STRENGTH)
    rounds = []
    for r in range(n_rounds):
        weights = np.array([STRENGTH[n] for n in names])
        order, remaining = [], list(range(len(names)))
        while remaining:
            w = weights[remaining]
            pick = remaining.pop(rng.choice(len(remaining), p=w / w.sum()))
            order.append(pick)
        scores = {names[idx]: float(-pos) for pos, idx in enumerate(order)}
        rounds.append(Round(cluster=f"ds{r % 10}", scores=scores, higher_better=True))
    return rounds


def test_three_model_recovery_and_anchor():
    rounds = _plackett_luce_rounds(300, seed=1)
    rows, _ = rate_rounds(rounds, anchor="B", boot=0, seed=7)
    by = {r["model"]: r for r in rows}
    assert [r["model"] for r in rows] == ["A", "B", "C"]
    assert by["B"]["elo"] == 1000.0
    assert by["A"]["elo"] > by["B"]["elo"] > by["C"]["elo"]
    assert by["A"]["rank"] == 1 and by["C"]["rank"] == 3

    # Noise-free expected counts: the MLE must recover strength ratios closely.
    index = {"A": 0, "B": 1, "C": 2}
    p = np.array([STRENGTH[n] for n in index])
    wins = np.array(
        [
            [p[i] / (p[i] + p[j]) * 1000 if i != j else 0.0 for j in range(3)]
            for i in range(3)
        ]
    )
    theta = fit_bt(wins)
    assert math.isclose(theta[0] - theta[1], math.log(2.0), abs_tol=1e-3)
    assert math.isclose(theta[1] - theta[2], math.log(2.0), abs_tol=1e-3)


def test_order_independence():
    rounds = _plackett_luce_rounds(120, seed=2)
    base, _ = rate_rounds(rounds, anchor="A", boot=0, seed=7)
    shuffled = [
        Round(
            cluster=r.cluster,
            scores=dict(reversed(list(r.scores.items()))),
            higher_better=r.higher_better,
        )
        for r in reversed(rounds)
    ]
    again, _ = rate_rounds(shuffled, anchor="A", boot=0, seed=7)
    assert base == again


def test_bootstrap_is_deterministic_for_fixed_seed():
    rounds = _plackett_luce_rounds(60, seed=3)
    first, _ = rate_rounds(rounds, anchor="B", boot=25, seed=11)
    second, _ = rate_rounds(rounds, anchor="B", boot=25, seed=11)
    assert first == second


def test_tie_handling():
    # Exact tie and tie within 1e-12 relative count as half a win each.
    assert pair_outcome(1.0, 1.0, higher_better=False) == 0.5
    assert pair_outcome(1.0, 1.0 + 1e-14, higher_better=False) == 0.5
    # Larger differences are decided by direction of the metric.
    assert pair_outcome(1.0, 1.0 + 1e-6, higher_better=False) == 1.0
    assert pair_outcome(1.0, 1.0 + 1e-6, higher_better=True) == 0.0

    rnd = Round(cluster="d", scores={"x": 0.5, "y": 0.5}, higher_better=False)
    wins = round_wins(rnd, {"x": 0, "y": 1})
    assert wins[0, 1] == 0.5 and wins[1, 0] == 0.5

    tied = [
        Round(cluster=f"d{i}", scores={"A": 0.3, "B": 0.3}, higher_better=False)
        for i in range(5)
    ]
    rows, _ = rate_rounds(tied, anchor="A", boot=0, seed=1)
    assert {r["model"]: r["elo"] for r in rows} == {"A": 1000.0, "B": 1000.0}


def test_non_finite_and_missing_scores_lose_and_fit_stays_finite():
    assert pair_outcome(math.inf, 2.0, higher_better=False) == 0.0
    assert pair_outcome(None, 2.0, higher_better=False) == 0.0
    assert pair_outcome(2.0, None, higher_better=False) == 1.0
    assert pair_outcome(None, None, higher_better=False) is None

    rounds = [
        Round(
            cluster=f"d{i}",
            scores={"good": 1.0, "mid": 2.0, "inf": math.inf, "miss": None},
            higher_better=False,
        )
        for i in range(8)
    ]
    rows, _ = rate_rounds(rounds, anchor="mid", boot=10, seed=5)
    by = {r["model"]: r for r in rows}
    assert all(math.isfinite(r[k]) for r in rows for k in ("elo", "ci_low", "ci_high"))
    assert by["good"]["elo"] > by["mid"]["elo"]
    assert by["mid"]["elo"] > by["inf"]["elo"] and by["mid"]["elo"] > by["miss"]["elo"]
    assert by["inf"]["rank"] >= 3 and by["miss"]["rank"] >= 3
