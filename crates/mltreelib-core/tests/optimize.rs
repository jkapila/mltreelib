//! Behavior tests for the generic optimization seam and every optimizer method.
//!
//! One file holds the whole surface: the shared contracts first, then one
//! section per method. Shared helpers live at the top so every section reads
//! the same way.

use std::cell::Cell;

use mltreelib_core::solver::generic::annealing::simulated_annealing;
use mltreelib_core::solver::generic::basin_hopping::basin_hopping;
use mltreelib_core::solver::generic::bfgs::bfgs;
use mltreelib_core::solver::generic::conjugate_gradient::nonlinear_conjugate_gradient;
use mltreelib_core::solver::generic::nelder_mead::nelder_mead;
use mltreelib_core::solver::generic::powell::powell;
use mltreelib_core::solver::generic::{
    is_feasible, latin_hypercube_samples, latin_hypercube_search, max_violation, optimize,
    random_search, violation, AnnealingOptions, BasinHoppingOptions, BfgsOptions, BoundaryPolicy,
    Bounds, Budget, BudgetStatus, Constraint, ConstraintKind, CountedObjective,
    FiniteDifferenceObjective, LatinHypercubeOptions, LbfgsBOptions, MethodOptions,
    NelderMeadOptions, NonlinearCgOptions, Objective, ObjectiveError, OptimizationRequest,
    OptimizerKind, PowellOptions, RandomSearchOptions, SharedOptions, SmoothObjective, SolveStatus,
};
use mltreelib_core::solver::{solve_lbfgs, DenseView, DesignMatrix, GlmFamily, LbfgsOptions};

/// Scaled quadratic objective with an exact gradient.
#[derive(Clone)]
struct Quadratic {
    bounds: Bounds,
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl Quadratic {
    fn new(bounds: Bounds, center: Vec<f64>, scale: Vec<f64>) -> Self {
        Self {
            bounds,
            center,
            scale,
        }
    }

    /// Quadratic that sums the squared parameters.
    fn sum_of_squares(bounds: Bounds) -> Self {
        let dim = bounds.dim();
        Self::new(bounds, vec![0.0; dim], vec![1.0; dim])
    }

    /// Weighted quadratic with its minimum at (0.3, -1.5).
    fn weighted(bounds: Bounds) -> Self {
        Self::new(bounds, vec![0.3, -1.5], vec![1.0, 0.5])
    }
}

impl Objective for Quadratic {
    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        Ok(x.iter()
            .zip(&self.center)
            .zip(&self.scale)
            .map(|((value, center), scale)| (value - center) * (value - center) / scale)
            .sum())
    }
}

impl SmoothObjective for Quadratic {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        Ok(x.iter()
            .zip(&self.center)
            .zip(&self.scale)
            .map(|((value, center), scale)| 2.0 * (value - center) / scale)
            .collect())
    }
}

/// Quadratic objective that counts every call.
struct CountingQuadratic {
    quadratic: Quadratic,
    calls: Cell<usize>,
}

impl CountingQuadratic {
    fn new(bounds: Bounds, center: Vec<f64>, scale: Vec<f64>) -> Self {
        Self {
            quadratic: Quadratic::new(bounds, center, scale),
            calls: Cell::new(0),
        }
    }
}

impl Objective for CountingQuadratic {
    fn dim(&self) -> usize {
        self.quadratic.dim()
    }

    fn bounds(&self) -> &Bounds {
        self.quadratic.bounds()
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.calls.set(self.calls.get() + 1);
        self.quadratic.value(x)
    }
}

/// One-dimensional double-well objective with a call counter.
struct DoubleWell {
    bounds: Bounds,
    calls: Cell<usize>,
}

impl Objective for DoubleWell {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        self.calls.set(self.calls.get() + 1);
        let squared = x[0] * x[0] - 1.0;
        Ok(squared * squared)
    }
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

/// Gaussian least-squares objective over a dense design.
struct GaussianObjective<'a> {
    design: DenseView<'a>,
    bounds: Bounds,
}

impl Objective for GaussianObjective<'_> {
    fn dim(&self) -> usize {
        self.design.n_features()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn value(&self, x: &[f64]) -> Result<f64, ObjectiveError> {
        let mut value = 0.0;
        for row in 0..self.design.n_rows() {
            let weight = self.design.get_weight(row);
            let residual = self.design.get_y(row) - self.design.eval_row(row, x);
            value += 0.5 * weight * residual * residual;
        }
        Ok(value)
    }
}

impl SmoothObjective for GaussianObjective<'_> {
    fn gradient(&self, x: &[f64]) -> Result<Vec<f64>, ObjectiveError> {
        let mut gradient = vec![0.0; self.dim()];
        let mut row = vec![0.0; self.dim()];
        for index in 0..self.design.n_rows() {
            let weight = self.design.get_weight(index);
            let residual = self.design.eval_row(index, x) - self.design.get_y(index);
            self.design.copy_design_row(index, &mut row);
            for column in 0..self.dim() {
                gradient[column] += weight * residual * row[column];
            }
        }
        Ok(gradient)
    }
}

/// Bounds that hold the optimum of the weighted quadratic.
fn weighted_bounds() -> Bounds {
    Bounds::new(vec![-2.0, -2.0], vec![2.0, 2.0]).expect("valid bounds")
}

/// Bounds for the double-well objective.
fn well_bounds() -> Bounds {
    Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds")
}

// -- Shared seam -----------------------------------------------------------

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
    let budget = Budget::new(Some(2), Some(1));
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
    let budget = Budget::unlimited();
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
    let mut names: Vec<&str> = statuses.iter().map(|status| status.as_str()).collect();
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
    let objective =
        Quadratic::sum_of_squares(Bounds::new(vec![-1.0], vec![1.0]).expect("valid bounds"));
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

    // A smooth method without a gradient source stays valid: the seam falls
    // back to a central-difference gradient over the counted objective.
    let without_gradient =
        OptimizationRequest::new(&objective, None, OptimizerKind::LbfgsB, shared);
    assert!(without_gradient.validate().is_ok());
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

#[test]
fn random_search_finds_a_known_optimum() {
    let objective = Quadratic::sum_of_squares(
        Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
    );
    let budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &budget,
        &RandomSearchOptions { samples: 4000 },
        7,
        None,
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert_eq!(result.evaluations, 4000);
    assert!(
        result.best_value < 0.1,
        "best value was {}",
        result.best_value
    );
    assert!(objective.bounds().contains(&result.best_x));
}

#[test]
fn random_search_stops_at_the_evaluation_budget() {
    let objective = Quadratic::sum_of_squares(
        Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
    );
    let budget = Budget::new(Some(10), None);
    let result = random_search(
        &objective,
        &budget,
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
    let objective = Quadratic::sum_of_squares(Bounds::unbounded(2));
    let budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &budget,
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
    let budget = Budget::unlimited();
    let result = random_search(
        &objective,
        &budget,
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
    let objective =
        Quadratic::sum_of_squares(Bounds::new(vec![-1.0], vec![1.0]).expect("valid bounds"));
    let budget = Budget::new(Some(5), None);
    let result = latin_hypercube_search(
        &objective,
        &budget,
        &LatinHypercubeOptions { samples: 20 },
        5,
    );
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 5);
    assert!(objective.bounds().contains(&result.best_x));
}

#[test]
fn generic_entry_point_runs_a_method() {
    let objective =
        Quadratic::sum_of_squares(Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds"));
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
    let result = optimize(&request).expect("method runs");
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 50);
}

#[test]
fn generic_entry_point_names_unimplemented_methods() {
    let objective =
        Quadratic::sum_of_squares(Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds"));
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::Cobyla,
        SharedOptions::default(),
    );
    let error = optimize(&request).expect_err("method missing");
    assert!(error.contains("cobyla"), "error was {error}");
}

#[test]
fn finite_difference_gradient_charges_the_shared_budget() {
    let objective = Quadratic::sum_of_squares(
        Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
    );
    let budget = Budget::new(Some(10), None);
    let counted = CountedObjective::new(&objective, &budget);
    let finite_difference = FiniteDifferenceObjective::new(&counted, 1e-6);
    let gradient = finite_difference
        .gradient(&[1.0, 2.0])
        .expect("central difference gradient");
    assert!(
        (gradient[0] - 2.0).abs() < 1e-6,
        "gradient was {gradient:?}"
    );
    assert!(
        (gradient[1] - 4.0).abs() < 1e-6,
        "gradient was {gradient:?}"
    );
    assert_eq!(budget.evaluations(), 4);
}

// -- L-BFGS-B ---------------------------------------------------------

#[test]
fn lbfgs_b_bounded_optimum_matches_analytic_solution() {
    let objective = Quadratic::new(
        Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"),
        vec![1.5, -2.0],
        vec![1.0, 2.0],
    );
    let request = OptimizationRequest::new(
        &objective,
        Some(&objective),
        OptimizerKind::LbfgsB,
        SharedOptions::default(),
    )
    .with_start(vec![0.0, 0.0])
    .with_options(MethodOptions::LbfgsB(LbfgsBOptions {
        m: 5,
        max_iter: 100,
        ftol: 1e-10,
        gtol: 1e-8,
    }));
    let result = optimize(&request).expect("method runs");
    assert!(result.status.is_success(), "status was {:?}", result.status);
    assert!(
        (result.best_x[0] - 1.5).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(
        (result.best_x[1] + 2.0).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
}

#[test]
fn lbfgs_b_bound_active_optimum_is_returned() {
    let objective = Quadratic::new(
        Bounds::new(vec![0.0, -1.0], vec![10.0, 1.0]).expect("valid bounds"),
        vec![20.0, 0.0],
        vec![1.0, 1.0],
    );
    let request = OptimizationRequest::new(
        &objective,
        Some(&objective),
        OptimizerKind::LbfgsB,
        SharedOptions::default(),
    )
    .with_start(vec![1.0, 0.0])
    .with_options(MethodOptions::LbfgsB(LbfgsBOptions {
        m: 5,
        max_iter: 100,
        ftol: 1e-10,
        gtol: 1e-8,
    }));
    let result = optimize(&request).expect("method runs");
    assert!(result.status.is_success(), "status was {:?}", result.status);
    assert!(
        (result.best_x[0] - 10.0).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(
        result.best_x[1].abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
}

#[test]
fn lbfgs_b_finite_difference_run_is_deterministic_and_feasible() {
    let objective = Quadratic::new(
        Bounds::new(vec![-3.0, -3.0], vec![3.0, 3.0]).expect("valid bounds"),
        vec![0.4, -0.7],
        vec![1.0, 1.0],
    );
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::LbfgsB,
        SharedOptions::default(),
    )
    .with_start(vec![2.0, -2.0]);
    let first = optimize(&request).expect("method runs");
    let second = optimize(&request).expect("method runs");
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);
    assert_eq!(first.evaluations, second.evaluations);
    assert!(objective.bounds().contains(&first.best_x));
}

#[test]
fn lbfgs_b_evaluations_match_objective_calls() {
    let objective = CountingQuadratic::new(
        Bounds::new(vec![-2.0], vec![2.0]).expect("valid bounds"),
        vec![0.0],
        vec![1.0],
    );
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::LbfgsB,
        SharedOptions::default(),
    )
    .with_start(vec![1.0]);
    let result = optimize(&request).expect("method runs");
    assert_eq!(result.evaluations, objective.calls.get());
}

#[test]
fn lbfgs_b_unbounded_gaussian_matches_leaf_lbfgs_objective() {
    let x = vec![vec![0.2], vec![0.4], vec![0.9], vec![1.3], vec![1.7]];
    let y = vec![0.4, 0.9, 1.7, 2.2, 2.8];
    let weights = vec![1.0, 1.5, 0.8, 1.2, 1.0];
    let design = DenseView::new(&x, &y, Some(&weights), true);
    let objective = GaussianObjective {
        design: design.clone(),
        bounds: Bounds::unbounded(design.n_features()),
    };
    let start = vec![y.iter().sum::<f64>() / y.len() as f64, 0.0];
    let request = OptimizationRequest::new(
        &objective,
        Some(&objective),
        OptimizerKind::LbfgsB,
        SharedOptions::default(),
    )
    .with_start(start.clone())
    .with_options(MethodOptions::LbfgsB(LbfgsBOptions {
        m: 6,
        max_iter: 300,
        ftol: 1e-12,
        gtol: 1e-8,
    }));
    let generic = optimize(&request).expect("method runs");
    let leaf = solve_lbfgs(
        &design,
        &LbfgsOptions {
            family: GlmFamily::Gaussian,
            l2: 0.0,
            max_iter: 300,
            m: 6,
            tol: 1e-8,
            verbose: 0,
        },
        Some(&start),
    )
    .expect("leaf solver runs");
    let leaf_value = objective.value(&leaf).expect("objective value");
    assert!(
        generic.best_value <= leaf_value + 1e-6,
        "generic={} leaf={}",
        generic.best_value,
        leaf_value
    );
    assert!((generic.best_value - leaf_value).abs() <= 1e-5);
}

// -- BFGS and nonlinear conjugate gradient --------------------------------

#[test]
fn bfgs_reaches_the_unbounded_optimum() {
    let objective = Quadratic::weighted(Bounds::unbounded(2));
    let budget = Budget::unlimited();
    let result = bfgs(
        &objective,
        &objective,
        &budget,
        &BfgsOptions::default(),
        Some(&[1.0, 1.0]),
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert!(
        (result.best_x[0] - 0.3).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(
        (result.best_x[1] + 1.5).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(result.best_value < 1e-8, "value was {}", result.best_value);
}

#[test]
fn nonlinear_conjugate_gradient_reaches_the_unbounded_optimum() {
    let objective = Quadratic::weighted(Bounds::unbounded(2));
    let budget = Budget::unlimited();
    let result = nonlinear_conjugate_gradient(
        &objective,
        &objective,
        &budget,
        &NonlinearCgOptions::default(),
        Some(&[1.0, 1.0]),
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert!(
        (result.best_x[0] - 0.3).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(
        (result.best_x[1] + 1.5).abs() < 1e-5,
        "point was {:?}",
        result.best_x
    );
    assert!(result.best_value < 1e-8, "value was {}", result.best_value);
}

#[test]
fn gradient_methods_reach_the_same_objective_quality() {
    let objective = Quadratic::weighted(Bounds::unbounded(2));
    let budget = Budget::unlimited();
    let full_memory = bfgs(
        &objective,
        &objective,
        &budget,
        &BfgsOptions::default(),
        Some(&[1.0, 1.0]),
    );
    let first_order = nonlinear_conjugate_gradient(
        &objective,
        &objective,
        &budget,
        &NonlinearCgOptions::default(),
        Some(&[1.0, 1.0]),
    );
    assert!(
        (full_memory.best_value - first_order.best_value).abs() < 1e-8,
        "bfgs {} versus cg {}",
        full_memory.best_value,
        first_order.best_value
    );
}

#[test]
fn gradient_methods_reject_finite_bounds() {
    let objective =
        Quadratic::weighted(Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).expect("valid bounds"));
    let budget = Budget::unlimited();
    let full_memory = bfgs(
        &objective,
        &objective,
        &budget,
        &BfgsOptions::default(),
        Some(&[0.0, 0.0]),
    );
    assert_eq!(full_memory.status, SolveStatus::Unsupported);
    let first_order = nonlinear_conjugate_gradient(
        &objective,
        &objective,
        &budget,
        &NonlinearCgOptions::default(),
        Some(&[0.0, 0.0]),
    );
    assert_eq!(first_order.status, SolveStatus::Unsupported);
}

#[test]
fn finite_difference_gradient_is_charged_to_the_budget() {
    let objective = CountingQuadratic::new(Bounds::unbounded(2), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::unlimited();
    let counted = CountedObjective::new(&objective, &budget);
    let finite_difference = FiniteDifferenceObjective::new(&counted, 1e-6);
    let result = bfgs(
        &counted,
        &finite_difference,
        &budget,
        &BfgsOptions::default(),
        Some(&[1.0, 1.0]),
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert_eq!(result.evaluations, budget.evaluations());
    assert!(budget.evaluations() > 0);
}

#[test]
fn gradient_methods_are_reproducible() {
    let first = bfgs(
        &Quadratic::weighted(Bounds::unbounded(2)),
        &Quadratic::weighted(Bounds::unbounded(2)),
        &Budget::unlimited(),
        &BfgsOptions::default(),
        Some(&[1.0, 1.0]),
    );
    let second = bfgs(
        &Quadratic::weighted(Bounds::unbounded(2)),
        &Quadratic::weighted(Bounds::unbounded(2)),
        &Budget::unlimited(),
        &BfgsOptions::default(),
        Some(&[1.0, 1.0]),
    );
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);
}

// -- Nelder-Mead and Powell ---------------------------------------------

#[test]
fn nelder_mead_reaches_the_bounded_optimum() {
    let objective = CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::unlimited();
    let result = nelder_mead(
        &objective,
        &budget,
        &NelderMeadOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert!(
        (result.best_x[0] - 0.3).abs() < 1e-3,
        "point was {:?}",
        result.best_x
    );
    assert!(
        (result.best_x[1] + 1.5).abs() < 1e-3,
        "point was {:?}",
        result.best_x
    );
    assert!(result.best_value < 1e-5, "value was {}", result.best_value);
}

#[test]
fn powell_reaches_the_bounded_optimum() {
    let objective = CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::unlimited();
    let result = powell(
        &objective,
        &budget,
        &PowellOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(result.status, SolveStatus::Converged);
    assert!(
        (result.best_x[0] - 0.3).abs() < 1e-3,
        "point was {:?}",
        result.best_x
    );
    assert!(
        (result.best_x[1] + 1.5).abs() < 1e-3,
        "point was {:?}",
        result.best_x
    );
    assert!(result.best_value < 1e-5, "value was {}", result.best_value);
}

#[test]
fn derivative_free_solvers_keep_points_inside_the_bounds() {
    let bounds = Bounds::new(vec![0.0, -1.0], vec![0.5, 1.0]).expect("valid bounds");
    let objective = CountingQuadratic::new(bounds.clone(), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::unlimited();

    let nelder = nelder_mead(
        &objective,
        &budget,
        &NelderMeadOptions::default(),
        Some(&[0.25, 0.0]),
    );
    assert!(bounds.contains(&nelder.best_x));

    let powell = powell(
        &objective,
        &budget,
        &PowellOptions::default(),
        Some(&[0.25, 0.0]),
    );
    assert!(bounds.contains(&powell.best_x));
}

#[test]
fn derivative_free_solvers_report_every_objective_call() {
    let objective = CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::unlimited();

    let nelder = nelder_mead(
        &objective,
        &budget,
        &NelderMeadOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(nelder.evaluations, objective.calls.get());

    let calls_after_nelder = objective.calls.get();
    let evaluations_after_nelder = budget.evaluations();
    let powell = powell(
        &objective,
        &budget,
        &PowellOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(
        objective.calls.get() - calls_after_nelder,
        budget.evaluations() - evaluations_after_nelder
    );
}

#[test]
fn derivative_free_solvers_stop_at_the_evaluation_budget() {
    let objective = CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]);
    let budget = Budget::new(Some(5), None);
    let result = nelder_mead(
        &objective,
        &budget,
        &NelderMeadOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 5);
    assert!(!result.status.is_success());
}

#[test]
fn derivative_free_solvers_are_reproducible() {
    let first = nelder_mead(
        &CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]),
        &Budget::unlimited(),
        &NelderMeadOptions::default(),
        Some(&[1.5, 1.5]),
    );
    let second = nelder_mead(
        &CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]),
        &Budget::unlimited(),
        &NelderMeadOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);

    let first = powell(
        &CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]),
        &Budget::unlimited(),
        &PowellOptions::default(),
        Some(&[1.5, 1.5]),
    );
    let second = powell(
        &CountingQuadratic::new(weighted_bounds(), vec![0.3, -1.5], vec![1.0, 0.5]),
        &Budget::unlimited(),
        &PowellOptions::default(),
        Some(&[1.5, 1.5]),
    );
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);
}

// -- Simulated annealing ------------------------------------------------

/// Annealing options used by the tests.
fn annealing_options() -> AnnealingOptions {
    AnnealingOptions {
        initial_temperature: Some(1.0),
        alpha: 0.9,
        min_temperature: 1e-6,
        proposal_scale: 0.4,
        boundary: BoundaryPolicy::Resample,
        max_stagnation: 500,
    }
}

#[test]
fn annealing_finds_a_known_multimodal_optimum() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let budget = Budget::unlimited();
    let result = simulated_annealing(&objective, &budget, &annealing_options(), 7, Some(&[0.0]));
    assert!(result.best_value < 0.05, "value was {}", result.best_value);
    assert!(
        (result.best_x[0].abs() - 1.0).abs() < 0.2,
        "point was {:?}",
        result.best_x
    );
}

#[test]
fn annealing_keeps_the_incumbent_and_the_bounds() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let budget = Budget::unlimited();
    let result = simulated_annealing(&objective, &budget, &annealing_options(), 9, Some(&[0.0]));
    assert!(objective.bounds().contains(&result.best_x));
    let history = result.history.expect("history");
    let mut previous = f64::INFINITY;
    for entry in &history {
        assert!(entry.best_value <= previous + 1e-12, "incumbent degraded");
        previous = entry.best_value;
    }
}

#[test]
fn annealing_is_reproducible_for_one_seed() {
    let first = simulated_annealing(
        &DoubleWell {
            bounds: well_bounds(),
            calls: Cell::new(0),
        },
        &Budget::unlimited(),
        &annealing_options(),
        11,
        Some(&[0.0]),
    );
    let second = simulated_annealing(
        &DoubleWell {
            bounds: well_bounds(),
            calls: Cell::new(0),
        },
        &Budget::unlimited(),
        &annealing_options(),
        11,
        Some(&[0.0]),
    );
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);
}

#[test]
fn annealing_reports_every_objective_call() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let budget = Budget::unlimited();
    let result = simulated_annealing(&objective, &budget, &annealing_options(), 3, Some(&[0.0]));
    assert_eq!(result.evaluations, objective.calls.get());
}

#[test]
fn annealing_stops_at_the_evaluation_budget() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let budget = Budget::new(Some(20), None);
    let result = simulated_annealing(&objective, &budget, &annealing_options(), 3, Some(&[0.0]));
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 20);
}

#[test]
fn annealing_refuses_unbounded_domains() {
    let objective = DoubleWell {
        bounds: Bounds::unbounded(1),
        calls: Cell::new(0),
    };
    let budget = Budget::unlimited();
    let result = simulated_annealing(&objective, &budget, &annealing_options(), 3, Some(&[0.0]));
    assert_eq!(result.status, SolveStatus::Unsupported);
    assert_eq!(result.evaluations, 0);
}

// -- Basin hopping -----------------------------------------------------

/// Basin hopping options used by the tests.
fn basin_options() -> MethodOptions {
    MethodOptions::BasinHopping(BasinHoppingOptions {
        inner: OptimizerKind::NelderMead,
        step_size: 0.5,
        temperature: 1.0,
        target_acceptance: Some(0.5),
        monotonic: false,
        iterations: 40,
    })
}

#[test]
fn basin_hopping_finds_a_known_multimodal_optimum() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::BasinHopping,
        SharedOptions::default(),
    )
    .with_options(basin_options());
    let result = optimize(&request).expect("method runs");
    assert!(result.best_value < 0.05, "value was {}", result.best_value);
    assert!(
        result.best_x[0].abs() > 0.5,
        "point was {:?}",
        result.best_x
    );
}

#[test]
fn basin_hopping_counts_the_inner_solves() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::BasinHopping,
        SharedOptions {
            max_evaluations: Some(200),
            ..SharedOptions::default()
        },
    )
    .with_options(basin_options());
    let result = optimize(&request).expect("method runs");
    assert_eq!(result.evaluations, objective.calls.get());
    assert!(result.local_solve_count.unwrap_or(0) > 0);
}

#[test]
fn basin_hopping_preserves_the_incumbent() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::BasinHopping,
        SharedOptions::default(),
    )
    .with_options(basin_options());
    let result = optimize(&request).expect("method runs");
    let history = result.history.expect("history");
    let mut previous = f64::INFINITY;
    for entry in &history {
        assert!(entry.best_value <= previous + 1e-12, "incumbent degraded");
        previous = entry.best_value;
    }
}

#[test]
fn basin_hopping_is_reproducible_for_one_seed() {
    let run = || {
        let objective = DoubleWell {
            bounds: well_bounds(),
            calls: Cell::new(0),
        };
        let request = OptimizationRequest::new(
            &objective,
            None,
            OptimizerKind::BasinHopping,
            SharedOptions {
                seed: 21,
                ..SharedOptions::default()
            },
        )
        .with_options(basin_options());
        optimize(&request).expect("method runs")
    };
    let first = run();
    let second = run();
    assert_eq!(first.best_x, second.best_x);
    assert_eq!(first.best_value, second.best_value);
}

#[test]
fn basin_hopping_stops_at_the_evaluation_budget() {
    let objective = DoubleWell {
        bounds: well_bounds(),
        calls: Cell::new(0),
    };
    let request = OptimizationRequest::new(
        &objective,
        None,
        OptimizerKind::BasinHopping,
        SharedOptions {
            max_evaluations: Some(30),
            ..SharedOptions::default()
        },
    )
    .with_options(basin_options());
    let result = optimize(&request).expect("method runs");
    assert_eq!(result.status, SolveStatus::EvaluationLimit);
    assert_eq!(result.evaluations, 30);
}
