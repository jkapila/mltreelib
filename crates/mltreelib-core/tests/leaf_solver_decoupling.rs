use mltreelib_core::family::LeafFamily;
use mltreelib_core::leaf::{
    fit_leaf_model, LeafControl, LeafKind, LeafObjective, LeafProblemType, LeafSolver,
    LeafSolverDiagnostic,
};
use mltreelib_core::solver::{
    check_leaf_solver_compatibility, solver_capabilities, LeafSolverRequirements, Objective,
    SmoothObjective,
};

fn linear_data() -> (Vec<Vec<Option<f64>>>, Vec<f64>, Vec<f64>, Vec<usize>) {
    (
        vec![vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]],
        vec![2.0, 4.0, 6.0, 8.0],
        vec![1.0; 4],
        vec![0, 1, 2, 3],
    )
}

#[test]
fn one_solver_fits_different_leaf_kinds_under_same_settings() {
    let (cols, y, w, rows) = linear_data();
    let mut control = LeafControl {
        leaf: LeafKind::Linear,
        solver: LeafSolver::Lbfgs,
        family: LeafFamily::Gaussian,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };

    let linear = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);
    let linear_model = linear.linear.as_ref().expect("linear model");
    assert!(linear_model.intercept.is_finite());
    assert!(linear_model.coef[0].is_finite());
    assert_eq!(
        linear.solver_used,
        LeafSolverDiagnostic::Applied(LeafSolver::Lbfgs)
    );
    assert_eq!(
        linear.leaf_problem_type,
        LeafProblemType::PenalizedGlm(LeafFamily::Gaussian)
    );

    control.leaf = LeafKind::Huber;
    let huber = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);
    assert!((huber.center.expect("Huber center") - 5.0).abs() < 1e-7);
    assert_eq!(
        huber.solver_used,
        LeafSolverDiagnostic::Applied(LeafSolver::Lbfgs)
    );
    assert_eq!(
        huber.leaf_problem_type,
        LeafProblemType::HuberLocation(LeafFamily::Gaussian)
    );
}

#[test]
fn unsupported_pairing_names_missing_capability() {
    let error = check_leaf_solver_compatibility(
        LeafKind::Linear,
        LeafFamily::Gaussian,
        LeafSolver::Lbfgs,
        LeafSolverRequirements {
            l1: true,
            monotone: false,
        },
    )
    .expect_err("L1 is unsupported by L-BFGS");
    assert!(error.contains("does not support L1 penalties"), "{error}");

    let error = check_leaf_solver_compatibility(
        LeafKind::Linear,
        LeafFamily::Gaussian,
        LeafSolver::Lbfgs,
        LeafSolverRequirements {
            l1: false,
            monotone: true,
        },
    )
    .expect_err("monotone constraints are unsupported by L-BFGS");
    assert!(
        error.contains("does not support monotone coefficient constraints"),
        "{error}"
    );

    let (cols, y, w, rows) = linear_data();
    let control = LeafControl {
        leaf: LeafKind::Linear,
        solver: LeafSolver::Lbfgs,
        family: LeafFamily::Gaussian,
        l1: 0.1,
        l2: 0.0,
        ..Default::default()
    };
    let leaf = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);
    match leaf.solver_used {
        LeafSolverDiagnostic::Rejected { reason, .. } => {
            assert!(reason.contains("does not support L1 penalties"), "{reason}");
        }
        diagnostic => panic!("expected rejected L1 pairing, got {diagnostic:?}"),
    }
    assert!(leaf
        .linear
        .as_ref()
        .expect("rejected linear model")
        .intercept
        .is_nan());
}

#[test]
fn closed_form_leaf_accepts_solver_and_reports_not_applicable() {
    let (cols, y, w, rows) = linear_data();
    let control = LeafControl {
        leaf: LeafKind::Mean,
        solver: LeafSolver::Ccd,
        family: LeafFamily::Gaussian,
        ..Default::default()
    };
    let leaf = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);

    assert!((leaf.center.expect("mean center") - 5.0).abs() < 1e-12);
    assert_eq!(
        leaf.solver_used,
        LeafSolverDiagnostic::NotApplicable(LeafSolver::Ccd)
    );
    assert_eq!(
        leaf.leaf_problem_type,
        LeafProblemType::ClosedFormMean(LeafFamily::Gaussian)
    );
    assert!(control.validate_solver().is_ok());
}

#[test]
fn existing_linear_fit_stays_numerically_unchanged() {
    let (cols, y, w, rows) = linear_data();
    let control = LeafControl {
        leaf: LeafKind::Linear,
        solver: LeafSolver::Irls,
        family: LeafFamily::Gaussian,
        l1: 0.0,
        l2: 0.0,
        ..Default::default()
    };
    let leaf = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);
    let model = leaf.linear.as_ref().expect("linear model");

    // Analytic OLS oracle for y = 2x.
    assert!((model.intercept - 0.0).abs() < 1e-12);
    assert!((model.coef[0] - 2.0).abs() < 1e-12);
    assert_eq!(
        leaf.solver_used,
        LeafSolverDiagnostic::Applied(LeafSolver::Irls)
    );
}

#[test]
fn old_leaf_snapshots_load_with_safe_diagnostic_defaults() {
    let (cols, y, w, rows) = linear_data();
    let control = LeafControl {
        leaf: LeafKind::Linear,
        solver: LeafSolver::Irls,
        family: LeafFamily::Gaussian,
        ..Default::default()
    };
    let leaf = fit_leaf_model(&cols, &y, &w, &rows, &[], &[0], &control);
    let mut value = serde_json::to_value(&leaf).expect("serialize leaf");
    let object = value.as_object_mut().expect("leaf object");
    object.remove("solver_used");
    object.remove("leaf_problem_type");

    let restored: mltreelib_core::node_stats::LeafModel =
        serde_json::from_value(value).expect("deserialize old leaf");
    assert_eq!(restored.solver_used, LeafSolverDiagnostic::Unknown);
    assert_eq!(restored.leaf_problem_type, LeafProblemType::Unknown);
}

#[test]
fn robust_objectives_expose_generic_seam() {
    let median = LeafObjective::median(vec![1.0, 2.0, 100.0], vec![1.0; 3]).expect("median");
    assert_eq!(median.dim(), 1);
    assert!(median.value(&[3.0]).expect("median value").is_finite());

    let quantile =
        LeafObjective::quantile(vec![1.0, 2.0, 100.0], vec![1.0; 3], 0.25).expect("quantile");
    assert_eq!(quantile.dim(), 1);
    assert!(quantile.value(&[3.0]).expect("quantile value").is_finite());

    let huber = LeafObjective::huber(vec![1.0, 2.0, 3.0], vec![1.0; 3]).expect("Huber");
    assert_eq!(huber.dim(), 1);
    assert!(huber.value(&[2.0]).expect("Huber value").is_finite());
    match huber {
        LeafObjective::Huber(objective) => {
            let gradient = objective.gradient(&[2.0]).expect("Huber gradient");
            assert!(gradient[0].abs() < 1e-12);
        }
        _ => panic!("expected Huber objective"),
    }
}

#[test]
fn capability_table_declares_solver_properties_once() {
    let lbfgs = solver_capabilities(LeafSolver::Lbfgs);
    assert!(lbfgs.supports_gradients);
    assert!(!lbfgs.supports_derivative_free);
    assert!(!lbfgs.supports_l1);
    assert!(!lbfgs.supports_monotone);
    assert!(!lbfgs.supports_finite_bounds);
    assert!(lbfgs.supports_unbounded);
    assert!(!lbfgs.supports_equality_constraints);
    assert!(!lbfgs.supports_nonlinear_inequality_constraints);

    let ccd = solver_capabilities(LeafSolver::Ccd);
    assert!(!ccd.supports_gradients);
    assert!(ccd.supports_derivative_free);
    assert!(ccd.supports_l1);
    assert!(ccd.supports_monotone);
}
