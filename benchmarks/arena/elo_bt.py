"""Order-independent Elo ratings for the arena boards.

Replaces sequential Elo (K=32) with a Bradley-Terry maximum-likelihood fit on
all pairwise battles, plus two-level cluster-bootstrap confidence intervals.

Battles are formed inside each round (a dataset, or one repeat of a dataset).
Datasets are the bootstrap clusters. Ratings are anchored so the baseline model
(``KNN``) sits at 1000 Elo in every fit and every bootstrap replicate.

Usage:
    uv run python benchmarks/arena/elo_bt.py [--boot 200] [--seed 7] [--anchor KNN]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REG_CSV = REPO_ROOT / "benchmarks" / "elo_scores.csv"
CLA_CSV = REPO_ROOT / "benchmarks" / "elo_cla_scores.csv"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

ANCHOR_ELO = 1000.0
ELO_PER_NAT = 400.0 / math.log(10.0)  # 400 * log10(ratio) == ELO_PER_NAT * ln(ratio)
REG_PSEUDO = (
    1e-3  # pseudo win and pseudo loss against a virtual average player (theta 0)
)
TIE_RTOL = 1e-12
MIN_DATASETS = 10
REPEAT_COLUMNS = ("repeat", "fold", "seed")

# board name -> list of (csv path, metric column, higher_is_better)
BOARDS: dict[str, list[tuple[Path, str, bool]]] = {
    "regression": [(REG_CSV, "rmse", False)],
    "classification": [(CLA_CSV, "accuracy", True)],
    "classification_logloss": [(CLA_CSV, "logloss", False)],
    "overall": [(REG_CSV, "rmse", False), (CLA_CSV, "accuracy", True)],
}


@dataclass(frozen=True)
class Round:
    """One game round: a dataset (or one repeat of it) with each model's score.

    ``scores`` maps model -> score, ``None`` meaning missing.
    """

    cluster: str
    scores: dict[str, float | None]
    higher_better: bool


def _clean(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return value


def pair_outcome(a: float | None, b: float | None, higher_better: bool) -> float | None:
    """Result for model a against model b: 1.0 a wins, 0.5 tie, 0.0 b wins, None skip.

    Finite scores compare by metric direction (ties within TIE_RTOL relative).
    A finite score beats an infinite or missing one. Two missing scores are
    skipped; any other pair involving a non-finite score is a tie.
    """
    a, b = _clean(a), _clean(b)
    fa = a is not None and math.isfinite(a)
    fb = b is not None and math.isfinite(b)
    if fa and fb:
        ga = a if higher_better else -a
        gb = b if higher_better else -b
        if abs(ga - gb) <= TIE_RTOL * max(abs(ga), abs(gb)):
            return 0.5
        return 1.0 if ga > gb else 0.0
    if fa:
        return 1.0
    if fb:
        return 0.0
    if a is None and b is None:
        return None
    return 0.5


def round_wins(rnd: Round, index: dict[str, int]) -> np.ndarray:
    """Matrix W with W[i, j] = battle wins of model i over model j in this round."""
    wins = np.zeros((len(index), len(index)))
    names = sorted(rnd.scores)
    for pos, name_a in enumerate(names):
        for name_b in names[pos + 1 :]:
            result = pair_outcome(
                rnd.scores[name_a], rnd.scores[name_b], rnd.higher_better
            )
            if result is None:
                continue
            i, j = index[name_a], index[name_b]
            wins[i, j] += result
            wins[j, i] += 1.0 - result
    return wins


def fit_bt(
    wins: np.ndarray,
    pseudo: float = REG_PSEUDO,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> np.ndarray:
    """Bradley-Terry MLE with a pseudo win and pseudo loss per model.

    The pseudo games are against a virtual player with strength 1 (log-strength
    0), which keeps unbeaten or winless models finite. Maximizes the penalized
    log-likelihood by Newton steps with step-halving; the Hessian is always
    negative definite because of the pseudo games. Returns natural-log strengths.
    """
    games = wins + wins.T
    wins_pen = wins  # pairwise wins, diagonal is zero

    def loglik(theta: np.ndarray) -> float:
        diff = theta[:, None] - theta[None, :]
        pair = wins_pen * (-np.logaddexp(0.0, -diff))
        np.fill_diagonal(pair, 0.0)
        pseudo_term = pseudo * (-np.logaddexp(0.0, -theta) - np.logaddexp(0.0, theta))
        return float(pair.sum() + pseudo_term.sum())

    theta = np.zeros(wins.shape[0])
    for _ in range(max_iter):
        prob = 1.0 / (1.0 + np.exp(-(theta[:, None] - theta[None, :])))  # P(i beats j)
        grad = (wins_pen - games * prob).sum(axis=1)
        grad += pseudo * (1.0 - 2.0 / (1.0 + np.exp(-theta)))
        weight = games * prob * (1.0 - prob)
        np.fill_diagonal(weight, 0.0)
        hess = weight.copy()
        hess[np.diag_indices_from(hess)] = -weight.sum(axis=1)
        own = 1.0 / (1.0 + np.exp(-theta))
        hess[np.diag_indices_from(hess)] -= 2.0 * pseudo * own * (1.0 - own)
        step = np.linalg.solve(hess, -grad)  # Newton ascent direction
        current = loglik(theta)
        scale = 1.0
        while scale > 1e-8 and loglik(theta + scale * step) < current:
            scale *= 0.5
        theta = theta + scale * step
        if np.max(np.abs(scale * step)) < tol:
            return theta
    warnings.warn(f"BT fit did not converge in {max_iter} Newton steps", stacklevel=2)
    return theta


def to_elo(theta: np.ndarray, anchor: int) -> np.ndarray:
    """Log-strengths -> Elo scale with the anchor model at ANCHOR_ELO."""
    return ANCHOR_ELO + ELO_PER_NAT * (theta - theta[anchor])


def _rank(elo: np.ndarray) -> np.ndarray:
    """1-based rank along the last axis, 1 = highest Elo."""
    return (elo[..., None, :] > elo[..., :, None]).sum(axis=-1) + 1


def rate_rounds(
    rounds: list[Round],
    anchor: str,
    boot: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Fit BT on all rounds and bootstrap CIs by resampling clusters, then rounds."""
    models = sorted({m for r in rounds for m in r.scores})
    if anchor not in models:
        raise ValueError(f"anchor model {anchor!r} not found in board")
    index = {m: i for i, m in enumerate(models)}
    a = index[anchor]
    mats = np.stack([round_wins(r, index) for r in rounds])

    clusters = sorted({r.cluster for r in rounds})
    members = [np.flatnonzero([r.cluster == c for r in rounds]) for c in clusters]

    elo = to_elo(fit_bt(mats.sum(axis=0)), a)
    rank = _rank(elo)

    rng = np.random.default_rng(seed)
    boot_elo = np.empty((boot, len(models)))
    for b in range(boot):
        chosen = []
        for c in rng.integers(0, len(clusters), len(clusters)):
            mem = members[c]
            chosen.append(mem[rng.integers(0, len(mem), len(mem))])
        boot_elo[b] = to_elo(fit_bt(mats[np.concatenate(chosen)].sum(axis=0)), a)
    if boot == 0:  # no replicates: report the point fit as its own interval
        boot_elo = elo[None, :]

    boot_diff = boot_elo - boot_elo[:, [a]]
    boot_rank = _rank(boot_elo)

    played = np.zeros(len(models), dtype=int)
    for mem in members:
        side = mats[mem].sum(axis=0)
        played += ((side + side.T).sum(axis=1) > 0).astype(int)

    def pct(values: np.ndarray, q: tuple[float, float]) -> tuple[float, float]:
        lo, hi = np.percentile(values, q)
        return float(lo), float(hi)

    rows = []
    for m, i in index.items():
        ci_lo, ci_hi = pct(boot_elo[:, i], (2.5, 97.5))
        vs_lo, vs_hi = pct(boot_diff[:, i], (2.5, 97.5))
        rk_lo, rk_hi = pct(boot_rank[:, i], (2.5, 97.5))
        rows.append(
            {
                "model": m,
                "n_datasets": int(played[i]),
                "elo": round(float(elo[i]), 2),
                "ci_low": round(ci_lo, 2),
                "ci_high": round(ci_hi, 2),
                "elo_vs_knn": round(float(elo[i] - elo[a]), 2),
                "vs_knn_low": round(vs_lo, 2),
                "vs_knn_high": round(vs_hi, 2),
                "rank": int(rank[i]),
                "rank_low": round(rk_lo, 1),
                "rank_high": round(rk_hi, 1),
            }
        )
    rows.sort(key=lambda r: (r["rank"], r["model"]))
    summary = {
        "n_datasets": len(clusters),
        "n_rounds": len(rounds),
        "n_battles": int(mats.sum()),
        "n_models": len(models),
    }
    return rows, summary


# --------------------------------------------------------------------------- CSV


def _parse_score(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return _clean(float(text))
    except ValueError:
        return None


def load_rounds(path: Path, metric: str, higher_better: bool) -> list[Round]:
    """Group rows into rounds by (dataset, repeat); one round per dataset if no repeat column."""
    groups: dict[tuple[str, str], dict[str, float | None]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        repeat_col = next(
            (c for c in REPEAT_COLUMNS if c in (reader.fieldnames or [])), None
        )
        for row in reader:
            model = (row.get("model") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            if not model or not dataset:
                continue
            key = (dataset, (row.get(repeat_col) or "").strip() if repeat_col else "")
            scores = groups.setdefault(key, {})
            if model in scores:
                raise ValueError(f"duplicate model {model!r} in round {key} of {path}")
            scores[model] = _parse_score(row.get(metric))
    return [
        Round(cluster=ds, scores=sc, higher_better=higher_better)
        for (ds, _), sc in groups.items()
    ]


def count_langs(path: Path, counts: dict[str, Counter]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model, lang = (
                (row.get("model") or "").strip(),
                (row.get("lang") or "").strip(),
            )
            if model:
                counts.setdefault(model, Counter())[lang] += 1


def modal_lang(counts: Counter) -> str:
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


# --------------------------------------------------------------------------- CLI

FIELDS = [
    "model",
    "lang",
    "n_datasets",
    "elo",
    "ci_low",
    "ci_high",
    "elo_vs_knn",
    "vs_knn_low",
    "vs_knn_high",
    "rank",
    "rank_low",
    "rank_high",
]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def run_board(name: str, anchor: str, boot: int, seed: int) -> tuple[list[dict], dict]:
    rounds: list[Round] = []
    lang_counts: dict[str, Counter] = {}
    for path, metric, higher_better in BOARDS[name]:
        rounds += load_rounds(path, metric, higher_better)
        count_langs(path, lang_counts)
    rows, summary = rate_rounds(rounds, anchor, boot, seed)
    for row in rows:
        row["lang"] = modal_lang(lang_counts[row["model"]])
    return rows, summary


def write_board(name: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"elo_bt_{name}.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})
    return out


def print_top(name: str, rows: list[dict], top: int = 15) -> None:
    print(f"\n== {name} (top {top}) ==")
    print(
        f"{'rank':>4}  {'model':<26} {'elo':>8}  {'95% CI':<17} {'vs KNN':>8}  {'vs KNN 95% CI':<19} rank 95% CI"
    )
    for r in rows[:top]:
        ci = f"[{r['ci_low']:.0f}, {r['ci_high']:.0f}]"
        vs = f"[{r['vs_knn_low']:+.0f}, {r['vs_knn_high']:+.0f}]"
        rk = f"[{r['rank_low']:.1f}, {r['rank_high']:.1f}]"
        print(
            f"{r['rank']:>4}  {r['model']:<26} {r['elo']:>8.1f}  {ci:<17} "
            f"{r['elo_vs_knn']:>+8.1f}  {vs:<19} {rk}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--boot", type=int, default=200, help="bootstrap replicates (default 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="bootstrap seed (default 7)"
    )
    parser.add_argument(
        "--anchor", default="KNN", help="model pinned at 1000 Elo (default KNN)"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    meta_boards: dict[str, dict] = {}
    for name in BOARDS:
        rows, summary = run_board(name, args.anchor, args.boot, args.seed)
        path = write_board(name, rows)
        print(f"wrote {path}")
        meta_boards[name] = summary
        print_top(name, rows)
        if summary["n_datasets"] < MIN_DATASETS:
            print(
                f"WARNING: board {name!r} has only {summary['n_datasets']} datasets "
                f"(< {MIN_DATASETS}); bootstrap CIs are unreliable",
                file=sys.stderr,
            )

    meta = {
        "method": "Bradley-Terry MLE (Newton, pseudo-regularized); two-level cluster bootstrap",
        "seed": args.seed,
        "B": args.boot,
        "anchor": args.anchor,
        "anchor_elo": ANCHOR_ELO,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "boards": meta_boards,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "elo_bt_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nruntime {time.perf_counter() - started:.1f}s -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
