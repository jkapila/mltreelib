use mltreelib_core::solver::{AskTell, AskTellKind, AskTellOptions};

#[test]
fn test_ask_tell_single_objective_cmaes() {
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let options = AskTellOptions {
        population_size: Some(12),
        max_iterations: Some(50),
        seed: Some(42),
        ..Default::default()
    };
    let mut opt = AskTell::new(AskTellKind::CmaEs, bounds, options).unwrap();

    while !opt.should_stop() {
        let candidates = opt.ask().unwrap();
        if candidates.is_empty() {
            break;
        }
        let fits: Vec<f64> = candidates
            .iter()
            .map(|x| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2) + x[2].powi(2))
            .collect();
        opt.tell_scalar(&fits).unwrap();
    }

    let (best_x, best_v) = opt.best_solution().expect("must find solution");
    assert!(best_v < 0.1, "CMA-ES best value was {}", best_v);
    assert!((best_x[0] - 1.0).abs() < 0.5);
    assert!((best_x[1] + 2.0).abs() < 0.5);
}

#[test]
fn test_ask_tell_pso_and_de() {
    let bounds = vec![(-4.0, 4.0), (-4.0, 4.0)];

    // Test DE
    let mut de_opt = AskTell::new(
        AskTellKind::De,
        bounds.clone(),
        AskTellOptions {
            population_size: Some(20),
            max_iterations: Some(40),
            seed: Some(123),
            ..Default::default()
        },
    )
    .unwrap();

    while !de_opt.should_stop() {
        let cands = de_opt.ask().unwrap();
        let fits: Vec<f64> = cands.iter().map(|x| x[0].powi(2) + x[1].powi(2)).collect();
        de_opt.tell_scalar(&fits).unwrap();
    }
    let (_, de_val) = de_opt.best_solution().unwrap();
    assert!(de_val < 0.05, "DE failed on sphere: {}", de_val);

    // Test PSO
    let mut pso_opt = AskTell::new(
        AskTellKind::Pso,
        bounds,
        AskTellOptions {
            population_size: Some(20),
            max_iterations: Some(40),
            seed: Some(456),
            ..Default::default()
        },
    )
    .unwrap();

    while !pso_opt.should_stop() {
        let cands = pso_opt.ask().unwrap();
        let fits: Vec<f64> = cands.iter().map(|x| x[0].powi(2) + x[1].powi(2)).collect();
        pso_opt.tell_scalar(&fits).unwrap();
    }
    let (_, pso_val) = pso_opt.best_solution().unwrap();
    assert!(pso_val < 0.05, "PSO failed on sphere: {}", pso_val);
}

#[test]
fn test_ask_tell_multi_objective_nsga2() {
    // Two conflicting objectives: f1(x) = x^2, f2(x) = (x - 2)^2
    let bounds = vec![(-3.0, 5.0)];
    let options = AskTellOptions {
        population_size: Some(30),
        max_iterations: Some(30),
        seed: Some(789),
        ..Default::default()
    };
    let mut opt = AskTell::new(AskTellKind::Nsga2, bounds, options).unwrap();

    while !opt.should_stop() {
        let cands = opt.ask().unwrap();
        let objs: Vec<Vec<f64>> = cands
            .iter()
            .map(|x| vec![x[0].powi(2), (x[0] - 2.0).powi(2)])
            .collect();
        opt.tell_multi(&objs).unwrap();
    }

    let front = opt.pareto_front();
    assert!(!front.is_empty(), "Pareto front must not be empty");
    // All Pareto optimal solutions for f1=x^2 and f2=(x-2)^2 lie in x in [0, 2]
    for sol in &front {
        assert!(sol.point[0] >= -0.1 && sol.point[0] <= 2.1, "x out of Pareto range: {:?}", sol.point);
        assert_eq!(sol.objectives.len(), 2);
    }
}
