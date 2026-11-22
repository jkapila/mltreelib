import pytest
from mltreelib.solvers import AskTell, minimize_many


def test_ask_tell_cmaes_sphere():
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    opt = AskTell("cmaes", bounds, population_size=12, max_iterations=60, seed=42)

    while not opt.should_stop:
        cands = opt.ask()
        if not cands:
            break
        # Sphere centered at (1, -2, 0)
        fits = [(x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2 + x[2] ** 2 for x in cands]
        opt.tell(fits)

    assert opt.best_x is not None
    assert opt.best_value is not None
    assert opt.best_value < 0.1
    assert abs(opt.best_x[0] - 1.0) < 0.5
    assert abs(opt.best_x[1] + 2.0) < 0.5
    assert opt.evaluations > 0
    assert opt.iterations > 0


def test_ask_tell_pso_and_de():
    bounds = [(-4.0, 4.0), (-4.0, 4.0)]

    # Test DE
    de_opt = AskTell("de", bounds, population_size=20, max_iterations=40, seed=123)
    while not de_opt.should_stop:
        cands = de_opt.ask()
        fits = [x[0] ** 2 + x[1] ** 2 for x in cands]
        de_opt.tell(fits)

    assert de_opt.best_value < 0.05

    # Test PSO
    pso_opt = AskTell("pso", bounds, population_size=20, max_iterations=40, seed=456)
    while not pso_opt.should_stop:
        cands = pso_opt.ask()
        fits = [x[0] ** 2 + x[1] ** 2 for x in cands]
        pso_opt.tell(fits)

    assert pso_opt.best_value < 0.05


def test_ask_tell_shade_snes_rcga():
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    methods = ["shade", "snes", "rcga"]

    for m in methods:
        opt = AskTell(m, bounds, population_size=20, max_iterations=30, seed=77)
        while not opt.should_stop:
            cands = opt.ask()
            fits = [x[0] ** 2 + x[1] ** 2 for x in cands]
            opt.tell(fits)
        assert opt.best_value < 0.2, f"{m} failed on sphere: {opt.best_value}"


def test_ask_tell_nsga2_multi_objective():
    # Two conflicting objectives: f1(x) = x^2, f2(x) = (x - 2)^2
    bounds = [(-3.0, 5.0)]
    opt = AskTell("nsga2", bounds, population_size=30, max_iterations=30, seed=789)

    while not opt.should_stop:
        cands = opt.ask()
        # Vector of 2 objectives
        objs = [[x[0] ** 2, (x[0] - 2.0) ** 2] for x in cands]
        opt.tell(objs)

    front = opt.pareto_front
    assert len(front) > 0
    for pt, vals in front:
        assert len(vals) == 2
        # Pareto optimal points are in [0, 2]
        assert -0.1 <= pt[0] <= 2.1


def test_minimize_many():
    # Multi-objective convenience wrapper
    def multi_obj(x):
        return [x[0] ** 2, (x[0] - 3.0) ** 2]

    front = minimize_many(
        multi_obj,
        [(-2.0, 5.0)],
        method="nsga2",
        population_size=24,
        max_iterations=25,
        seed=101,
    )
    assert len(front) > 0
    for pt, vals in front:
        assert len(vals) == 2
        assert -0.1 <= pt[0] <= 3.1


def test_ask_tell_invalid_feedback():
    opt = AskTell("cmaes", [(-2.0, 2.0)], population_size=8)
    _ = opt.ask()
    with pytest.raises(ValueError):
        # Mismatched length (expects 8, pass 3)
        opt.tell([1.0, 2.0, 3.0])


def test_ask_tell_moead_multi_objective():
    bounds = [(-3.0, 5.0)]
    opt = AskTell("moead", bounds, population_size=20, max_iterations=30, seed=12)

    while not opt.should_stop:
        cands = opt.ask()
        objs = [[x[0] ** 2, (x[0] - 2.0) ** 2] for x in cands]
        opt.tell(objs)

    front = opt.pareto_front
    assert len(front) > 0
    for pt, vals in front:
        assert len(vals) == 2
        assert -0.2 <= pt[0] <= 2.2


def test_ask_tell_nsga3_many_objective():
    bounds = [(-2.0, 3.0), (-2.0, 3.0)]
    opt = AskTell("nsga3", bounds, population_size=25, max_iterations=25, seed=24, nsga3_partitions=8)

    while not opt.should_stop:
        cands = opt.ask()
        objs = [
            [
                x[0] ** 2 + x[1] ** 2,
                (x[0] - 1.0) ** 2 + x[1] ** 2,
                x[0] ** 2 + (x[1] - 1.0) ** 2,
            ]
            for x in cands
        ]
        opt.tell(objs)

    front = opt.pareto_front
    assert len(front) > 0
    for pt, vals in front:
        assert len(vals) == 3
        assert -0.2 <= pt[0] <= 1.2
        assert -0.2 <= pt[1] <= 1.2


def test_ask_tell_island_model():
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    opt = AskTell("island", bounds, population_size=24, max_iterations=25, seed=36, island_count=3)

    while not opt.should_stop:
        cands = opt.ask()
        fits = [(x[0] - 0.5) ** 2 + (x[1] + 0.5) ** 2 for x in cands]
        opt.tell(fits)

    assert opt.best_value is not None
    assert opt.best_value < 0.1
    assert abs(opt.best_x[0] - 0.5) < 0.4
    assert abs(opt.best_x[1] + 0.5) < 0.4


def test_minimize_many_moead_nsga3():
    def multi_obj(x):
        return [x[0] ** 2, (x[0] - 2.0) ** 2]

    f_moead = minimize_many(multi_obj, [(-2.0, 4.0)], method="moead", population_size=20, max_iterations=20)
    assert len(f_moead) > 0

    f_nsga3 = minimize_many(multi_obj, [(-2.0, 4.0)], method="nsga3", population_size=20, max_iterations=20)
    assert len(f_nsga3) > 0
