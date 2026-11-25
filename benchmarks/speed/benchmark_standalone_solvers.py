"""Benchmark standalone optimizers on fixed weighted Gaussian and binomial data."""

import json
from math import exp
from random import Random
from time import perf_counter

from mltreelib.solvers import fit


def sample(family, size, seed):
    rng = Random(seed)
    x, y, weights = [], [], []
    for _ in range(size):
        row = [rng.uniform(-1, 1) for _ in range(3)]
        eta = 0.4 + 1.2 * row[0] - 0.8 * row[1]
        target = (
            eta + rng.gauss(0, 0.3)
            if family == "gaussian"
            else float(rng.random() < 1.0 / (1.0 + exp(-eta)))
        )
        x.append(row)
        y.append(target)
        weights.append(rng.uniform(0.5, 2.0))
    return x, y, weights


def main():
    results = []
    for family in ("gaussian", "binomial"):
        train_x, train_y, weights = sample(family, 192, seed=21)
        test_x, test_y, test_weights = sample(family, 256, seed=22)
        baselines = ["irls", "lbfgs", "shade", "cmaes", "ga", "nes"]
        if family == "gaussian":
            baselines.extend(["direct", "ccd"])
        for solver in baselines:
            result = {"family": family, "solver": solver}
            started = perf_counter()
            try:
                model = fit(
                    train_x,
                    train_y,
                    family=family,
                    solver=solver,
                    sample_weight=weights,
                    l2=0.1,
                    max_iter=150,
                    seed=42,
                )
            except ValueError as error:
                result["fit_seconds"] = perf_counter() - started
                result["error"] = str(error)
            else:
                result.update(
                    fit_seconds=perf_counter() - started,
                    objective=model.objective,
                    held_out_deviance=model.mean_deviance(
                        test_x, test_y, sample_weight=test_weights
                    ),
                    stop_reason=model.stop_reason,
                    iterations=model.iterations,
                    converged=model.converged,
                )
            results.append(result)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
