"""Canonical optimizer benchmark.

Runs every method in the generic optimizer surface over a fixed problem set
with equal evaluation budgets and writes a JSON payload plus a Markdown
summary.

    uv run python examples/benchmark_optimizers.py
    uv run python examples/benchmark_optimizers.py --budget 200 --seed 7
    uv run python examples/benchmark_optimizers.py --output-dir reports/optimizers

Each record holds the method, the problem, the seed, the budget, the
evaluation and iteration counts, the wall time, the best objective, the gap to
the known optimum, the constraint violation, and the termination status.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from mltreelib.solvers import optimize
from optimize_generic import BOUNDS as TREE_BOUNDS
from optimize_generic import fit_tree, load_numeric_table, rmse

# -- Problems -------------------------------------------------------------


def sphere(point):
    """Sum of squares. The optimum is zero."""
    return sum(value * value for value in point)


def rotated_ellipsoid(point):
    """Weighted squares after a fixed orthogonal rotation."""
    first, second, *rest = point
    return (
        0.5 * (first + second) ** 2
        + 50.0 * (first - second) ** 2
        + sum((index + 1) * value**2 for index, value in enumerate(rest, start=2))
    )


def rosenbrock(point):
    """Rosenbrock valley. The optimum is zero at (1, 1)."""
    total = 0.0
    for index in range(len(point) - 1):
        total += 100.0 * (point[index + 1] - point[index] ** 2) ** 2
        total += (1.0 - point[index]) ** 2
    return total


def rastrigin(point):
    """Rastrigin function. The optimum is zero at the origin."""
    total = 10.0 * len(point)
    for value in point:
        total += value * value - 10.0 * math.cos(2.0 * math.pi * value)
    return total


def double_well(point):
    """Double-well function. The minima are zero at plus or minus one."""
    squared = point[0] * point[0] - 1.0
    return squared * squared


def constrained_circle(point):
    """Circle objective with one inequality constraint."""
    return point[0] * point[0] + point[1] * point[1]


def circle_constraint(point):
    """Constraint that keeps the sum of the parameters above one."""
    return [point[0] + point[1] - 1.0]


def regression_problem():
    """Weighted least squares with a closed-form normal-equation reference."""
    rows = [(-2.0, 1.0, 1.0), (-1.0, 1.5, 2.0), (0.0, 3.0, 1.0), (2.0, 5.0, 3.0)]

    def objective(point):
        intercept, slope = point
        return sum(w * (y - intercept - slope * x) ** 2 for x, y, w in rows)

    weight = sum(w for _, _, w in rows)
    mean_x = sum(w * x for x, _, w in rows) / weight
    mean_y = sum(w * y for _, y, w in rows) / weight
    slope = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in rows) / sum(
        w * (x - mean_x) ** 2 for x, _, w in rows
    )
    return {
        "name": "weighted_regression",
        "objective": objective,
        "bounds": [(-10.0, 10.0)] * 2,
        "optimum": objective([mean_y - slope * mean_x, slope]),
        "start": [0.0, 0.0],
    }


def binomial_problem():
    """Binomial negative log-likelihood with a closed-form log-odds reference."""
    successes, trials = 17, 64

    def objective(point):
        logit = point[0]
        log_partition = max(0.0, logit) + math.log1p(math.exp(-abs(logit)))
        return trials * log_partition - successes * logit

    return {
        "name": "binomial_likelihood",
        "objective": objective,
        "bounds": [(-8.0, 8.0)],
        "optimum": objective([math.log(successes / (trials - successes))]),
        "start": [0.0],
    }


def tree_problems(dataset, offline, seed):
    """Build a real-data tree objective and a scalar split-score objective."""
    _, rows, target = load_numeric_table(dataset, offline)
    split = len(rows) * 4 // 5
    x_train, x_holdout = rows[:split], rows[split:]
    y_train, y_holdout = target[:split], target[split:]

    def tree_objective(point):
        depth, minimum, l2 = point
        model = fit_tree(x_train, y_train, round(depth), round(minimum), l2, seed)
        return rmse(model, x_holdout, y_holdout)

    # Piecewise-constant split score: evaluate each distinct partition for the oracle.
    values = [(0.0, 0.0), (1.0, 0.2), (2.0, -0.1), (3.0, 4.0), (4.0, 4.3), (5.0, 4.1)]

    def split_objective(point):
        left = [y for x, y in values if x <= point[0]]
        right = [y for x, y in values if x > point[0]]
        return sum(
            sum((y - sum(group) / len(group)) ** 2 for y in group)
            for group in (left, right)
            if group
        )

    methods = {
        "random_search",
        "latin_hypercube",
        "nelder_mead",
        "powell",
        "simulated_annealing",
    }

    return [
        {
            "name": f"tree_hyperparameters_{dataset}",
            "objective": tree_objective,
            "bounds": TREE_BOUNDS,
            "optimum": None,
            "start": [3.0, 15.0, 0.001],
            "methods": methods,
        },
        {
            "name": "tree_split_threshold",
            "objective": split_objective,
            "bounds": [(0.0, 5.0)],
            "optimum": min(split_objective([x]) for x, _ in values),
            "start": [1.5],
            "methods": methods,
        },
    ]


# -- Problem set ----------------------------------------------------------


def build_problems(dimension: int):
    """Return the fixed problem set with bounds and known optima."""
    box = [(-5.0, 5.0)] * dimension
    return [
        {
            "name": "sphere",
            "objective": sphere,
            "bounds": box,
            "optimum": 0.0,
            "start": [2.0] * dimension,
        },
        {
            "name": "rotated_ellipsoid",
            "objective": rotated_ellipsoid,
            "bounds": box,
            "optimum": 0.0,
            "start": [1.5] * dimension,
        },
        {
            "name": "rosenbrock",
            "objective": rosenbrock,
            "bounds": [(-5.0, 5.0)] * 2,
            "optimum": 0.0,
            "start": [-1.2, 1.0],
        },
        {
            "name": "rastrigin",
            "objective": rastrigin,
            "bounds": [(-5.0, 5.0)] * 3,
            "optimum": 0.0,
            "start": [1.0, 1.0, 1.0],
        },
        {
            "name": "double_well",
            "objective": double_well,
            "bounds": [(-2.0, 2.0)],
            "optimum": 0.0,
            "start": [0.0],
            "local_methods": ("bfgs", "nonlinear_cg", "nelder_mead", "powell"),
        },
        {
            "name": "constrained_circle",
            "objective": constrained_circle,
            "bounds": [(-5.0, 5.0)] * 2,
            "optimum": 0.5,
            "start": [0.0, 0.0],
            "constraints": circle_constraint,
            "methods": {"cobyla"},
        },
        {
            "name": "nonlinear_disk",
            "objective": lambda point: (point[0] - 2.0) ** 2 + point[1] ** 2,
            "bounds": [(-5.0, 5.0)] * 2,
            "optimum": 1.0,
            "start": [1.5, 0.0],
            "constraints": lambda point: [1.0 - sphere(point)],
            "methods": {"cobyla"},
        },
        regression_problem(),
        binomial_problem(),
    ]


# -- Runner ---------------------------------------------------------------

METHODS = [
    "random_search",
    "latin_hypercube",
    "lbfgs_b",
    "bfgs",
    "nonlinear_cg",
    "nelder_mead",
    "powell",
    "simulated_annealing",
    "cobyla",
    "basin_hopping",
    "cmaes",
    "shade",
    "nes",
    "ga",
]

UNBOUNDED_METHODS = {"bfgs", "nonlinear_cg"}


def run_one(method, problem, budget, seed, local_method=None):
    """Run one method on one problem and collect the record."""
    bounds = problem["bounds"]
    unbounded = method in UNBOUNDED_METHODS or local_method in UNBOUNDED_METHODS
    if unbounded:
        bounds = [(-math.inf, math.inf)] * len(bounds)
    record = {
        "method": method,
        "problem": problem["name"],
        "seed": seed,
        "budget": budget,
        "domain": "unbounded" if unbounded else "box",
    }
    if method == "basin_hopping":
        record["local_method"] = local_method or "lbfgs_b"
    started = time.perf_counter()
    result = optimize(
        problem["objective"],
        bounds,
        method=method,
        start=problem["start"],
        max_evaluations=budget,
        seed=seed,
        samples=budget if method in {"random_search", "latin_hypercube"} else None,
        constraints=problem.get("constraints"),
        local_method=local_method,
    )

    record["status"] = result.status
    record["evaluations"] = result.evaluations
    record["iterations"] = result.iterations
    best_value = finite(result.best_value)
    violation = finite(result.max_constraint_violation)
    optimum = problem["optimum"]
    comparable = "constraints" not in problem or (
        violation is not None and violation <= 1e-8
    )
    record["best_objective"] = best_value
    record["best_point"] = [finite(value) for value in result.best_x]
    record["seconds"] = time.perf_counter() - started
    record["objective_gap"] = (
        best_value - optimum
        if comparable and best_value is not None and optimum is not None
        else None
    )
    record["constraint_violation"] = violation
    return record


def finite(value):
    """Return the value when it is finite, otherwise None."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def markdown_table(records):
    """Render the records as a compact Markdown table."""
    header = (
        "| problem | method | status | evaluations | best objective | gap | seconds |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for record in records:
        best = record.get("best_objective")
        gap = record.get("objective_gap")
        method = record["method"]
        if "local_method" in record:
            method += f" ({record['local_method']})"
        rows.append(
            "| {problem} | {method} | {status} | {evaluations} | {best} | {gap} | {seconds:.3f} |".format(
                problem=record["problem"],
                method=method,
                status=record["status"],
                evaluations=record.get("evaluations", "-"),
                best="-" if best is None else f"{best:.6g}",
                gap="-" if gap is None else f"{gap:.3g}",
                seconds=record.get("seconds", 0.0),
            )
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=200, help="evaluation budget per run."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for stochastic methods."
    )
    parser.add_argument(
        "--dimension", type=int, default=5, help="dimension of the smooth problems."
    )
    parser.add_argument(
        "--dataset", default="airfoil_self_noise", help="dataset for tree tuning."
    )
    parser.add_argument(
        "--offline", action="store_true", help="do not download the dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/optimizers"),
        help="folder for the JSON payload and the Markdown summary.",
    )
    args = parser.parse_args()

    if args.budget < 1 or args.dimension < 2:
        parser.error("--budget must be positive and --dimension must be at least 2")
    problems = build_problems(args.dimension) + tree_problems(
        args.dataset, args.offline, args.seed
    )
    records = [
        run_one(method, problem, args.budget, args.seed)
        for problem in problems
        for method in METHODS
        if method in problem.get("methods", METHODS)
    ]
    records.extend(
        run_one("basin_hopping", problem, args.budget, args.seed, local_method)
        for problem in problems
        for local_method in problem.get("local_methods", ())
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "budget": args.budget,
        "seed": args.seed,
        "methods": METHODS,
        "records": records,
    }
    json_path = args.output_dir / "optimizer_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    summary = "# Optimizer benchmark\n\n" + markdown_table(records)
    summary_path = args.output_dir / "optimizer_benchmark.md"
    summary_path.write_text(summary)

    print(summary)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
