//! Behavior tests for the shared generic optimization seam.

use mltreelib_core::solver::generic::{
    is_feasible, latin_hypercube_samples, latin_hypercube_search, max_violation, random_search,
    violation, AnnealingOptions, Bounds, Budget, BudgetStatus, Constraint, ConstraintKind,
    LatinHypercubeOptions, MethodOptions, Objective, ObjectiveError, OptimizationRequest,
    OptimizerKind, RandomSearchOptions, SharedOptions, SolveStatus,
};

/// Quadratic objective over a bounded box.
struct Quadratic {
    bounds: Bounds,
}

impl Objective for Quadratic {
    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        if x.len() != self.bounds.dim() {
            return Err(ObjectiveError::DimensionMismatch {
                expected: self.bounds.dim(),
                actual: x.len(),
            });
        }
        Ok(x.iter().map(|v| v * v).sum())
    }
}

#[test]
fn bounds_reject_invalid_configuration() {
    assert!(Bounds::new(vec![0.0, 0.0], vec![1.0]).is_err());
    assert!(Bounds::new(vec![1.0], vec![0.0]).is_err());
    assert!(Bounds::new(vec![f64::NAN], vec![1.0]).is_err());
    assert!(Bounds::new(vec![f64::INFINITY], vec![1.0]).is_err());
    assert!(Bounds::new(vec![0.0], vec![f64::NEG_INFINITY]).is_err());
    assert!(Bounds::new(vec![0.0], vec![0.0]).is_ok());
}

#[test]
fn bounds_report_domain_and_projection() {
    let bounds = Bounds::new(vec![-1.0, 2.0], vec![1.0, 4.0]).expect("valid bounds");
    assert_eq!(bounds.dim(), 2);
    assert!(bounds.is_bounded());
    assert!(bounds.contains(&[0.0, 3.0]));
    assert!(!bounds.contains(&[2.0, 3.0]));
    assert!(!bounds.contains(&[0.0, 3.0, 1.0]));
    assert_eq!(bounds.project(&[3.0, 1.0]), vec![1.0, 2.0]);

    let unbounded = Bounds::unbounded(2);
    assert!(!unbounded.is_bounded());
    assert!(!unbounded.has_finite_limit());
}

#[test]
fn budget_counts_evaluations_and_iterations() {
    let mut budget = Budget::new(Some(2), Some(1));
    assert_eq!(budget.remaining_evaluations(), Some(2));
    assert!(budget.charge_evaluation().is_ok());
    assert!(budget.charge_evaluation().is_ok());
    assert_eq!(
        budget.charge_evaluation(),
        Err(BudgetStatus::EvaluationLimit)
    );
    assert_eq!(budget.evaluations(), 2);
    assert_eq!(budget.remaining_evaluations(), Some(0));

    assert!(budget.charge_iteration().is_ok());
    assert_eq!(budget.charge_iteration(), Err(BudgetStatus::IterationLimit));
    assert_eq!(budget.iterations(), 1);
}

#[test]
fn unlimited_budget_never_stops() {
    let mut budget = Budget::unlimited();
    for _ in 0..1000 {
        assert!(budget.charge_evaluation().is_ok());
        assert!(budget.charge_iteration().is_ok());
    }
    assert_eq!(budget.evaluations(), 1000);
    assert_eq!(budget.remaining_evaluations(), None);
}

#[test]
fn statuses_stay_distinct() {
    let statuses = [
        SolveStatus::Converged,
        SolveStatus::IterationLimit,
        SolveStatus::EvaluationLimit,
        SolveStatus::Tolerance,
        SolveStatus::InvalidObjective,
        SolveStatus::NumericalFailure,
        SolveStatus::Infeasible,
        SolveStatus::Unsupported,
        SolveStatus::HardError,
    ];
    let mut names: Vec<&str> = statuses.iter().map(|s| s.as_str()).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(names.len(), count);

    assert!(SolveStatus::Converged.is_success());
    assert!(SolveStatus::Tolerance.is_success());
    assert!(!SolveStatus::IterationLimit.is_success());
    assert!(!SolveStatus::EvaluationLimit.is_success());
    assert!(!SolveStatus::InvalidObjective.is_success());
    assert!(!SolveStatus::NumericalFailure.is_success());
    assert!(!SolveStatus::Infeasible.is_success());
    assert!(!SolveStatus::Unsupported.is_success());
    assert!(!SolveStatus::HardError.is_success());
}

#[test]
fn inequality_feasibility_uses_nonnegative_values() {
    assert!(is_feasible(ConstraintKind::Inequality, &[0.0, 2.0], 1e-8));
    assert!(!is_feasible(ConstraintKind::Inequality, &[-0.5], 1e-8));
    assert_eq!(violation(ConstraintKind::Inequality, -0.5), 0.5);
    assert_eq!(violation(ConstraintKind::Inequality, 0.5), 0.0);
    assert_eq!(max_violation(ConstraintKind::Inequality, &[-0.5, 0.5]), 0.5);

    let constraint = Constraint::new(ConstraintKind::Inequality, vec![-0.25, 1.0]);
    assert_eq!(constraint.max_violation(), 0.25);
    assert!(!constraint.is_feasible(1e-8));
}

#[test]
fn equality_feasibility_uses_absolute_tolerance() {
    assert!(is_feasible(ConstraintKind::Equality, &[1e-10], 1e-8));
    assert!(!is_feasible(ConstraintKind::Equality, &[1e-6], 1e-8));
    assert_eq!(violation(ConstraintKind::Equality, -2.0), 2.0);
    assert_eq!(max_violation(ConstraintKind::Equality, &[0.1, -0.3]), 0.3);
}

#[test]
fn request_validation_rejects_bad_configuration() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-1.0], vec![1.0]).expect("valid bounds"),
    };
    let shared = SharedOptions::default();

    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::RandomSearch,
        shared.clone(),
    );
    assert!(request.validate().is_ok());

    let wrong_start = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::RandomSearch,
        shared.clone(),
    )
    .with_start(vec![0.0, 1.0]);
    assert!(wrong_start.validate().is_err());

    let outside = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::RandomSearch,
        shared.clone(),
    )
    .with_start(vec![5.0]);
    assert!(outside.validate().is_err());

    let mismatch = OptimizationRequest {
        objective: &objective,
        smooth: None,
        start: None,
        method: OptimizerKind::LbfgsB,
        options: MethodOptions::Annealing(AnnealingOptions::default()),
        shared: shared.clone(),
    };
    assert!(mismatch.validate().is_err());

    let needs_gradient = OptimizationRequest::new(&objective, None, OptimizerKind::LbfgsB, shared);
    assert!(needs_gradient.validate().is_err());
}

#[test]
fn method_names_round_trip() {
    for kind in [
        OptimizerKind::RandomSearch,
        OptimizerKind::LatinHypercube,
        OptimizerKind::LbfgsB,
        OptimizerKind::Bfgs,
        OptimizerKind::NonlinearConjugateGradient,
        OptimizerKind::NelderMead,
        OptimizerKind::Powell,
        OptimizerKind::SimulatedAnnealing,
        OptimizerKind::Cobyla,
        OptimizerKind::BasinHopping,
    ] {
        assert_eq!(OptimizerKind::from_name(kind.as_str()), Some(kind));
    }
    assert_eq!(OptimizerKind::from_name("nope"), None);
}

/// Objective that always returns a non-finite value.
struct BrokenObjective {
    bounds: Bounds,
}

impl Objective for BrokenObjective {
    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, _x: &[f64]) -> Result<f64, ObjectiveError> {
        Ok(f64::NAN)
    }
}

#[test]
fn random_search_finds_a_known_optimum() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
    };
    let mut budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &mut budget,
        &RandomSearchOptions { samples: 4000 },
        7,
        None,
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert_eq!(result.evaluations, 4000);
    assert!(result.best_value < 0.1, "best value was {}", result.best_value);
    assert!(objective.bounds().contains(&result.best_x));
}

#[test]
fn random_search_stops_at_the_evaluation_budget() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
    };
    let mut budget = Budget::new(Some(10), None);
    let result = random_search(
        &objective,
        &mut budget,
        &RandomSearchOptions { samples: 100 },
        7,
        None,
    );
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 10);
    assert!(!result.status.is_success());
}

#[test]
fn random_search_rejects_unbounded_domains() {
    let objective = Quadratic {
        bounds: Bounds::unbounded(2),
    };
    let mut budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &mut budget,
        &RandomSearchOptions { samples: 10 },
        7,
        None,
    );
    assert_eq!(result.status, SolveStatus::Unsupported);
    assert_eq!(result.evaluations, 0);
}

#[test]
fn non_finite_objective_values_end_the_run() {
    let objective = BrokenObjective {
        bounds: Bounds::new(vec![-1.0], vec![1.0]).expect("valid bounds"),
    };
    let mut budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &mut budget,
        &RandomSearchOptions { samples: 10 },
        7,
        None,
    );
    assert_eq!(result.status, SolveStatus::InvalidObjective);
    assert_eq!(result.evaluations, 1);
}

#[test]
fn latin_hypercube_is_reproducible_and_stratified() {
    let bounds = Bounds::new(vec![-2.0, 0.0], vec![2.0, 1.0]).expect("valid bounds");
    let first = latin_hypercube_samples(&bounds, 16, 3).expect("finite bounds");
    let second = latin_hypercube_samples(&bounds, 16, 3).expect("finite bounds");
    let other = latin_hypercube_samples(&bounds, 16, 4).expect("finite bounds");

    assert_eq!(first, second);
    assert_ne!(first, other);
    assert_eq!(first.len(), 16);
    assert!(first.iter().all(|point| bounds.contains(point)));

    for dim in 0..bounds.dim() {
        let width = bounds.width(dim) / 16.0;
        let mut strata: Vec<usize> = first
            .iter()
            .map(|point| (((point[dim] - bounds.lower()[dim]) / width).floor() as usize).min(15))
            .collect();
        strata.sort_unstable();
        strata.dedup();
        assert_eq!(strata.len(), 16, "dimension {dim} missed a stratum");
    }
}

#[test]
fn latin_hypercube_search_reports_its_evaluations() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-1.0], vec![1.0]).expect("valid bounds"),
    };
    let mut budget = Budget::new(Some(5), None);
    let result = latin_hypercube_search(
        &objective,
        &mut budget,
        &LatinHypercubeOptions { samples: 20 },
        5,
    );
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 5);
    assert!(objective.bounds().contains(&result.best_x));
}

#[test]
fn generic_entry_point_runs_a_method() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds"),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::RandomSearch,
        SharedOptions {
            max_evaluations: Some(50),
            seed: 11,
            ..SharedOptions::default()
        },
    );
    let result = mltreelib_core::solver::generic::optimize(&request).expect("method runs");
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 50);
}

#[test]
fn generic_entry_point_names_unimplemented_methods() {
    let objective = Quadratic {
        bounds: Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds"),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::Powell,
        SharedOptions::default(),
    );
    let error = mltreelib_core::solver::generic::optimize(&request).expect_err("method missing");
    assert!(error.contains("powell"), "error was {error}");
}
