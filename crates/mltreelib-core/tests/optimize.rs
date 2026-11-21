//! Behavior tests for the shared generic optimization seam.

use mltreelib_core::solver::generic::{
    is_feasible, max_violation, violation, AnnealingOptions, Bounds, Budget, BudgetStatus,
    Constraint, ConstraintKind, MethodOptions, Objective, ObjectiveError, OptimizationRequest,
    OptimizerKind, SharedOptions, SolveStatus,
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
