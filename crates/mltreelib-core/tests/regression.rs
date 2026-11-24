//! Tests for Spec 27 Phase 0: Group metadata substrate, validation, indicator inheritance, and penalty helpers.

use mltreelib_core::solver::{
    canonicalize_groups, expand_missing_indicators, group_l2_norm, group_lasso_penalty,
    group_soft_threshold, sparse_group_lasso_penalty, validate_groups, ColumnSliceView, DenseView,
    DesignMatrix, GroupKind, GroupSpec, GroupValidationError,
};

#[test]
fn test_group_spec_validation_success() {
    let groups = vec![
        GroupSpec::new(GroupKind::CoefficientGroup, "demographics", vec![1, 2]),
        GroupSpec::new(GroupKind::CoefficientGroup, "financial", vec![3, 4, 5]),
    ];
    // Design with intercept (col 0) + 5 features => n_features = 6
    assert!(validate_groups(&groups, 6, true, Some(100)).is_ok());
}

#[test]
fn test_group_spec_intercept_exclusion() {
    let groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "intercept_violator",
        vec![0, 1],
    )];
    let err = validate_groups(&groups, 4, true, Some(50)).unwrap_err();
    assert_eq!(
        err,
        GroupValidationError::InterceptIncluded {
            group: "intercept_violator".to_string()
        }
    );
}

#[test]
fn test_group_spec_index_out_of_range() {
    let groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "overflow",
        vec![1, 6],
    )];
    let err = validate_groups(&groups, 6, true, Some(50)).unwrap_err();
    assert_eq!(
        err,
        GroupValidationError::IndexOutOfRange {
            group: "overflow".to_string(),
            index: 6,
            max_allowed: 5
        }
    );
}

#[test]
fn test_group_spec_overlapping_rejection() {
    let groups = vec![
        GroupSpec::new(GroupKind::CoefficientGroup, "group_a", vec![1, 2]),
        GroupSpec::new(GroupKind::CoefficientGroup, "group_b", vec![2, 3]),
    ];
    let err = validate_groups(&groups, 5, true, Some(50)).unwrap_err();
    assert_eq!(
        err,
        GroupValidationError::OverlappingCoefficientGroups {
            column: 2,
            group1: "group_a".to_string(),
            group2: "group_b".to_string()
        }
    );
}

#[test]
fn test_group_spec_empty_group_rejection() {
    let groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "empty_group",
        vec![],
    )];
    let err = validate_groups(&groups, 5, true, Some(50)).unwrap_err();
    assert_eq!(
        err,
        GroupValidationError::EmptyGroup {
            group: "empty_group".to_string()
        }
    );
}

#[test]
fn test_group_spec_canonical_ordering() {
    let mut groups = vec![
        GroupSpec::new(GroupKind::CoefficientGroup, "beta", vec![4, 2, 2]),
        GroupSpec::new(GroupKind::CoefficientGroup, "alpha", vec![3, 1]),
    ];
    canonicalize_groups(&mut groups);
    assert_eq!(groups[0].name, "alpha");
    assert_eq!(groups[0].members, vec![1, 3]);
    assert_eq!(groups[1].name, "beta");
    assert_eq!(groups[1].members, vec![2, 4]);
}

#[test]
fn test_group_spec_serialization_roundtrip() {
    let group = GroupSpec::new(GroupKind::CoefficientGroup, "sector", vec![1, 2, 3])
        .with_weight(1.75);
    let serialized = serde_json::to_string(&group).expect("serialization failed");
    let deserialized: GroupSpec =
        serde_json::from_str(&serialized).expect("deserialization failed");
    assert_eq!(group, deserialized);
    assert_eq!(deserialized.effective_weight(), 1.75);
}

#[test]
fn test_dense_view_with_groups() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let y = vec![10.0, 20.0];
    let groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "pair",
        vec![1, 2],
    )];

    let view = DenseView::new(&x, &y, None, true)
        .with_groups(groups)
        .expect("valid groups should attach");

    assert!(view.groups().is_some());
    assert_eq!(view.groups().unwrap().len(), 1);
    assert_eq!(view.groups().unwrap()[0].name, "pair");
}

#[test]
fn test_column_slice_view_indicator_expansion() {
    // 2 features (chosen = [0, 1]), rows 0..2
    let col0 = vec![Some(1.0), None];
    let col1 = vec![Some(2.0), Some(3.0)];
    let cols = vec![col0, col1];
    let y = vec![5.0, 8.0];
    let w = vec![1.0, 1.0];
    let rows = vec![0, 1];
    let chosen = vec![0, 1];

    // Group specifies feature 1 (index 1 in design: col0 is chosen[0] => design col 1)
    let raw_groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "feature0_grp",
        vec![1],
    )];

    let direct_expanded = expand_missing_indicators(&raw_groups, 2);
    assert_eq!(direct_expanded[0].members, vec![1, 3]);

    // with_indicators = true => n_features = 1 (intercept) + 2 (features) + 2 (indicators) = 5
    // feature 0 is design col 1. Its indicator is design col 1 + 2 = 3.
    let slice_view = ColumnSliceView::new(&cols, &y, &w, &rows, &chosen, true)
        .with_groups(raw_groups)
        .expect("indicator group inheritance should succeed");

    let attached = slice_view.groups().expect("groups present");
    assert_eq!(attached.len(), 1);
    assert_eq!(attached[0].name, "feature0_grp");
    // Should inherit member 3 (the indicator column)
    assert_eq!(attached[0].members, vec![1, 3]);
}

#[test]
fn test_group_penalty_helpers() {
    // coef: [intercept=10.0, beta_1=3.0, beta_2=4.0, beta_3=0.0]
    let mut coef = vec![10.0, 3.0, 4.0, 0.0];
    let members = vec![1, 2];

    // L2 norm of [3.0, 4.0] is 5.0
    let norm = group_l2_norm(&coef, &members);
    assert!((norm - 5.0).abs() < 1e-12);

    let groups = vec![GroupSpec::new(
        GroupKind::CoefficientGroup,
        "g1",
        vec![1, 2],
    )];
    // Default weight is sqrt(2)
    let expected_penalty = 0.5 * (2.0f64).sqrt() * 5.0;
    let penalty = group_lasso_penalty(&coef, &groups, 0.5);
    assert!((penalty - expected_penalty).abs() < 1e-12);

    // Group soft-threshold with threshold 2.5: scale should be (1 - 2.5/5.0) = 0.5
    group_soft_threshold(&mut coef, &members, 2.5);
    assert!((coef[1] - 1.5).abs() < 1e-12);
    assert!((coef[2] - 2.0).abs() < 1e-12);
    assert_eq!(coef[0], 10.0); // intercept untouched

    // Group soft-threshold with threshold 10.0 (> current norm 2.5) => zeroes members
    group_soft_threshold(&mut coef, &members, 10.0);
    assert_eq!(coef[1], 0.0);
    assert_eq!(coef[2], 0.0);

    // Test sparse group lasso penalty helper
    let sgl_coef = vec![5.0, 3.0, 4.0];
    let sgl_penalty = sparse_group_lasso_penalty(&sgl_coef, &groups, 1.0, 0.5, 0.5);
    // L1 norm on non-intercept: |3| + |4| = 7.0. L2 on group [1, 2]: sqrt(3^2 + 4^2) = 5.0 * sqrt(2)
    let expected_sgl = 1.0 * (0.5 * 7.0 + 0.5 * (2.0f64).sqrt() * 5.0);
    assert!((sgl_penalty - expected_sgl).abs() < 1e-12);
}

#[test]
fn test_cardinality_adaptive_cuts_exact_low_cardinality() {
    use mltreelib_core::model_tree::ModelTreeParams;
    use mltreelib_core::split::histogram_cuts;

    let col0 = vec![
        Some(1.0),
        Some(2.0),
        Some(3.0),
        Some(4.0),
        Some(5.0),
        Some(2.0),
        Some(3.0),
    ];
    let cols = vec![col0];
    let rows: Vec<usize> = (0..7).collect();
    let mut params = ModelTreeParams::default();
    params.max_bins = 256;
    params.split_precision = None;

    let cuts = histogram_cuts(&cols, &rows, &params);
    assert_eq!(cuts.len(), 1);
    // 5 unique values: [1.0, 2.0, 3.0, 4.0, 5.0] => 4 exact midpoint cuts
    assert_eq!(cuts[0], vec![1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_cardinality_adaptive_cuts_high_cardinality_quantiles() {
    use mltreelib_core::model_tree::ModelTreeParams;
    use mltreelib_core::split::histogram_cuts;

    let col0: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
    let cols = vec![col0];
    let rows: Vec<usize> = (0..100).collect();
    let mut params = ModelTreeParams::default();
    params.max_bins = 8;
    params.min_bin_rows = 5;
    params.split_precision = None;

    let cuts = histogram_cuts(&cols, &rows, &params);
    assert_eq!(cuts.len(), 1);
    // 100 unique values with max_bins=8 => cuts.len() <= 7 strictly increasing
    assert!(cuts[0].len() <= 7);
    assert!(cuts[0].windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn test_binned_and_fast_split_equivalence_on_discrete_data() {
    use mltreelib_core::criteria::GainFn;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams, SplitEval};

    // Planted split at x=2.5: x in {1, 2} => y=10.0, x in {3, 4, 5} => y=50.0
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let val = (i % 5 + 1) as f64; // 1.0, 2.0, 3.0, 4.0, 5.0
        x.push(vec![Some(val)]);
        let target = if val <= 2.0 { 10.0 } else { 50.0 };
        y.push(Some(target));
    }

    let p_fast = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        leaf: LeafKind::Mean,
        gain: GainFn::Sse,
        split_eval: SplitEval::Fast,
        split_precision: None,
        ..Default::default()
    };
    let tree_fast = ModelTree::fit(&x, &y, p_fast).expect("fast fit succeeded");

    let p_hist = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        leaf: LeafKind::Mean,
        gain: GainFn::Sse,
        split_eval: SplitEval::Histogram,
        max_bins: 256,
        split_precision: None,
        ..Default::default()
    };
    let tree_hist = ModelTree::fit(&x, &y, p_hist).expect("histogram fit succeeded");

    assert_eq!(tree_fast.splits().len(), 1);
    assert_eq!(tree_hist.splits().len(), 1);

    let sp_fast = &tree_fast.splits()[0];
    let sp_hist = &tree_hist.splits()[0];

    assert_eq!(sp_fast.col, sp_hist.col);
    assert_eq!(sp_fast.threshold, sp_hist.threshold);
    assert!((sp_fast.gain - sp_hist.gain).abs() < 1e-6);
}

#[test]
fn test_binned_histogram_supports_regression_gains() {
    use mltreelib_core::criteria::GainFn;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams, SplitEval};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let val = (i % 6) as f64;
        x.push(vec![Some(val)]);
        y.push(Some(val * 2.0 + 1.0));
    }

    for gain in [
        GainFn::Sse,
        GainFn::FriedmanMse,
        GainFn::Sdr,
        GainFn::Deviance,
    ] {
        let p = ModelTreeParams {
            max_depth: 2,
            min_rows: 5,
            leaf: LeafKind::Mean,
            gain,
            split_eval: SplitEval::Histogram,
            max_bins: 64,
            split_precision: None,
            ..Default::default()
        };
        let tree = ModelTree::fit(&x, &y, p);
        assert!(tree.is_ok(), "Fit failed for regression gain in histogram mode");
        assert!(tree.unwrap().n_leaves() >= 2);
    }
}

// ---------------------------------------------------------------------------
// Spec 27 Phase 2: Quantile Regression Tests
// ---------------------------------------------------------------------------

use mltreelib_core::solver::{
    asymmetric_weight, detect_quantile_crossings, pinball_loss, smoothed_pinball_loss,
    solve_quantile_lbfgs, SmoothedQuantileObjective,
};

#[test]
fn test_asymmetric_weight() {
    assert_eq!(asymmetric_weight(2.5, 0.75), 0.75);
    assert_eq!(asymmetric_weight(0.0, 0.75), 0.75);
    assert_eq!(asymmetric_weight(-1.5, 0.75), 0.25);
    assert_eq!(asymmetric_weight(1.0, 0.5), 0.5);
    assert_eq!(asymmetric_weight(-1.0, 0.5), 0.5);
}

#[test]
fn test_exact_pinball_loss() {
    let tau = 0.8;
    // r >= 0: tau * r
    assert!((pinball_loss(2.0, tau) - 1.6).abs() < 1e-12);
    assert_eq!(pinball_loss(0.0, tau), 0.0);
    // r < 0: (1 - tau) * (-r)
    assert!((pinball_loss(-2.0, tau) - 0.4).abs() < 1e-12);
}

#[test]
fn test_smoothed_pinball_surrogate_and_gradient() {
    let tau = 0.7;
    let eps = 1e-3;

    // Far positive: matches exact pinball loss
    let (loss_pos, grad_pos) = smoothed_pinball_loss(2.0, tau, eps);
    assert!((loss_pos - pinball_loss(2.0, tau)).abs() < 1e-12);
    assert!((grad_pos - tau).abs() < 1e-12);

    // Far negative: matches exact pinball loss
    let (loss_neg, grad_neg) = smoothed_pinball_loss(-2.0, tau, eps);
    assert!((loss_neg - pinball_loss(-2.0, tau)).abs() < 1e-12);
    assert!((grad_neg - (tau - 1.0)).abs() < 1e-12);

    // Inside transition [-eps, eps]: verify derivative via finite difference
    for r in [-0.0008, -0.0002, 0.0, 0.0003, 0.0009] {
        let (loss, grad) = smoothed_pinball_loss(r, tau, eps);
        let h = 1e-7;
        let (loss_plus, _) = smoothed_pinball_loss(r + h, tau, eps);
        let (loss_minus, _) = smoothed_pinball_loss(r - h, tau, eps);
        let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
        assert!(
            (grad - fd_grad).abs() < 1e-5,
            "Gradient mismatch at r={r}: analytical={grad}, fd={fd_grad}"
        );
        assert!(loss >= 0.0, "Loss must be non-negative");
    }
}

#[test]
fn test_smoothed_quantile_objective_gradient_conformance() {
    use mltreelib_core::solver::generic::{Objective, SmoothObjective};

    // Design matrix with intercept + 2 features on 20 rows
    let mut x_dense = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        let x1 = (i as f64) * 0.1;
        let x2 = ((i % 5) as f64) * 0.2;
        x_dense.push(vec![x1, x2]);
        y.push(1.5 + 2.0 * x1 - 0.5 * x2);
    }
    let design = DenseView::new(&x_dense, &y, None, true);
    let obj = SmoothedQuantileObjective::new(&design, 0.6, 1e-4, 0.01);

    let beta = vec![1.2, 1.8, -0.3];
    let grad = obj.gradient(&beta).expect("Gradient computation failed");
    let h = 1e-7;
    for j in 0..beta.len() {
        let mut beta_plus = beta.clone();
        let mut beta_minus = beta.clone();
        beta_plus[j] += h;
        beta_minus[j] -= h;
        let val_plus = obj.value(&beta_plus).unwrap();
        let val_minus = obj.value(&beta_minus).unwrap();
        let fd = (val_plus - val_minus) / (2.0 * h);
        assert!(
            (grad[j] - fd).abs() < 1e-4,
            "Finite difference gradient mismatch in dimension {j}: analytic={}, fd={}",
            grad[j],
            fd
        );
    }
}

#[test]
fn test_solve_quantile_lbfgs_parameter_recovery() {
    let mut x_dense = Vec::new();
    let mut y = Vec::new();
    // Planted model: y = 2.0 + 3.0 * x1 + uniform(-0.1, 0.1)
    for i in 0..100 {
        let x1 = (i as f64) / 50.0;
        let noise = ((i as f64 * 7.0).sin()) * 0.05;
        x_dense.push(vec![x1]);
        y.push(2.0 + 3.0 * x1 + noise);
    }
    let design = DenseView::new(&x_dense, &y, None, true);

    // Median tau = 0.5
    let sol_med = solve_quantile_lbfgs(&design, 0.5, 1e-4, 1e-5, 250, 1e-7, None)
        .expect("L-BFGS solve failed");
    assert_eq!(sol_med.len(), 2);
    assert!((sol_med[0] - 2.0).abs() < 0.15, "Intercept recovery failed: {}", sol_med[0]);
    assert!((sol_med[1] - 3.0).abs() < 0.15, "Slope recovery failed: {}", sol_med[1]);

    // Upper quantile tau = 0.9 should have a higher intercept than median
    let sol_hi = solve_quantile_lbfgs(&design, 0.9, 1e-4, 1e-5, 250, 1e-7, None)
        .expect("L-BFGS solve failed");
    assert!(
        sol_hi[0] >= sol_med[0] - 0.01,
        "tau=0.9 intercept {} should be >= tau=0.5 intercept {}",
        sol_hi[0],
        sol_med[0]
    );
}

#[test]
fn test_quantile_linear_leaf_model_tree_fit_and_predict() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..120 {
        let x1 = (i as f64) * 0.05;
        x.push(vec![Some(x1)]);
        y.push(Some(1.0 + 2.5 * x1 + ((i % 7) as f64) * 0.1));
    }

    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 20,
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Quantile,
        tau: 0.75,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree quantile fit failed");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 120);
    assert!(preds.iter().all(|p| p.is_finite()));

    // Verify invalid OLS summaries are omitted
    for (leaf, _) in tree.leaves() {
        if let Some(lm) = &leaf.linear {
            assert!(lm.summary.is_none(), "Quantile leaf must not contain OLS standard errors");
        }
    }
}

#[test]
fn test_quantile_minimum_support_fallback() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    // Small dataset where leaf will have few rows relative to features
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..15 {
        x.push(vec![Some(i as f64), Some((i * 2) as f64), Some((i * 3) as f64)]);
        y.push(Some(i as f64));
    }

    let params = ModelTreeParams {
        max_depth: 3,
        min_rows: 3, // leaves can have 3 rows, but P=3 features (+1 intercept => requires 4 rows)
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Quantile,
        tau: 0.5,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit should not fail on small leaves");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 15);
    for p in preds {
        assert!(p.is_finite(), "Prediction should be finite from 0D quantile center fallback");
    }
}

#[test]
fn test_detect_quantile_crossings_diagnostics() {
    // 3 quantiles: tau = [0.1, 0.5, 0.9]
    let taus = vec![0.1, 0.5, 0.9];

    // Case 1: Monotone predictions (no crossings)
    let preds_clean = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.5, 1.5, 2.5],
        vec![2.0, 3.0, 4.0],
    ];
    let rep_clean = detect_quantile_crossings(&preds_clean, &taus);
    assert_eq!(rep_clean.n_crossings, 0);
    assert_eq!(rep_clean.n_violating_rows, 0);
    assert_eq!(rep_clean.crossing_rate, 0.0);
    assert_eq!(rep_clean.max_violation, 0.0);

    // Case 2: Crossing at row 1: Q_0.5 = 2.5 > Q_0.9 = 2.0
    let preds_crossed = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.5, 2.5, 2.0], // crossing!
        vec![2.0, 3.0, 4.0],
    ];
    let rep_crossed = detect_quantile_crossings(&preds_crossed, &taus);
    assert_eq!(rep_crossed.n_crossings, 1);
    assert_eq!(rep_crossed.n_violating_rows, 1);
    assert!((rep_crossed.max_violation - 0.5).abs() < 1e-9);
    assert_eq!(rep_crossed.crossings, vec![(1, 1, 2)]);
}

// ---------------------------------------------------------------------------
// Spec 27 Phase 3: Huber Regression with Robust Scale
// ---------------------------------------------------------------------------

#[test]
fn test_huber_loss_and_psi_properties() {
    use mltreelib_core::solver::problems::huber::{huber_loss, huber_psi};

    let delta = 1.345;
    // Quadratic region (|r| <= delta)
    let r1 = 0.5;
    assert!((huber_loss(r1, delta) - 0.5 * r1 * r1).abs() < 1e-12);
    assert!((huber_psi(r1, delta) - r1).abs() < 1e-12);

    // Linear region (|r| > delta)
    let r2 = 3.0;
    let expected_loss = delta * (r2 - 0.5 * delta);
    assert!((huber_loss(r2, delta) - expected_loss).abs() < 1e-12);
    assert!((huber_psi(r2, delta) - delta).abs() < 1e-12);

    let r3 = -4.0;
    let expected_loss_neg = delta * (4.0 - 0.5 * delta);
    assert!((huber_loss(r3, delta) - expected_loss_neg).abs() < 1e-12);
    assert!((huber_psi(r3, delta) - (-delta)).abs() < 1e-12);
}

#[test]
fn test_weighted_mad_scale_recovery() {
    use mltreelib_core::solver::problems::huber::weighted_mad_scale;

    // Normal samples scaled by sigma = 3.0 with unit weights
    // Quantiles of standard normal: median of |Z| is approx 0.67449
    // Multiplying by 1.4826 recovers sigma
    let mut rng_state = 123456789u64;
    let mut lcg = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    let sigma_true = 3.0;
    let mut residuals = Vec::with_capacity(500);
    for _ in 0..250 {
        // Box-Muller transform
        let u1 = lcg().max(1e-10);
        let u2 = lcg();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        residuals.push((z0 * sigma_true, 1.0));
        residuals.push((z1 * sigma_true, 1.0));
    }

    let estimated_scale = weighted_mad_scale(&residuals);
    // Should recover sigma = 3.0 within statistical sample tolerance (~10%)
    assert!(
        (estimated_scale - sigma_true).abs() < 0.4,
        "Expected scale close to 3.0, got {estimated_scale}"
    );
}

#[test]
fn test_huber_regression_outlier_resistance() {
    use mltreelib_core::solver::problems::huber::solve_huber_regression;
    use mltreelib_core::solver::DenseView;

    // Model: y = 2.0 * x1 - 1.5 * x2 + 4.0
    let n = 200;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);

    let mut rng_state = 987654321u64;
    let mut lcg = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    for i in 0..n {
        let x1 = lcg() * 4.0 - 2.0;
        let x2 = lcg() * 4.0 - 2.0;
        x_mat.push(vec![x1, x2]);

        // Standard small noise
        let noise = (lcg() - 0.5) * 0.2;
        let mut y = 4.0 + 2.0 * x1 - 1.5 * x2 + noise;

        // 10% severe contamination outliers: add +/- 200.0
        if i % 10 == 0 {
            y += if i % 20 == 0 { 200.0 } else { -200.0 };
        }
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);

    let result = solve_huber_regression(&view, 1.345, 1e-4, 30, 200, 1e-5)
        .expect("Huber regression should solve cleanly on contaminated data");

    assert!(result.converged, "Huber regression should converge");
    assert_eq!(result.coefficients.len(), 3); // [intercept, beta1, beta2]

    let intercept = result.coefficients[0];
    let beta1 = result.coefficients[1];
    let beta2 = result.coefficients[2];

    // Check recovered parameters against ground truth [4.0, 2.0, -1.5]
    assert!(
        (intercept - 4.0).abs() < 0.35,
        "Intercept expected ~4.0, got {intercept}"
    );
    assert!(
        (beta1 - 2.0).abs() < 0.25,
        "beta1 expected ~2.0, got {beta1}"
    );
    assert!(
        (beta2 - (-1.5)).abs() < 0.25,
        "beta2 expected ~-1.5, got {beta2}"
    );
}

#[test]
fn test_huber_model_tree_leaf_and_support_rule() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    // Synthesize clean linear data with 2 outliers
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let x1 = (i as f64) * 0.1;
        let x2 = (i as f64) * 0.2;
        x.push(vec![Some(x1), Some(x2)]);
        let mut yi = 1.0 + 2.0 * x1 + 0.5 * x2;
        if i == 5 {
            yi += 100.0; // Outlier
        }
        if i == 45 {
            yi -= 100.0; // Outlier
        }
        y.push(Some(yi));
    }

    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Huber,
        huber_delta: 1.345,
        l2: 0.1,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit with Huber leaf should succeed");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 50);

    for (leaf_model, _) in tree.leaves() {
        if let Some(ref lm) = leaf_model.linear {
            assert!(
                lm.summary.is_none(),
                "Multivariate Huber leaf must not report OLS standard errors"
            );
        }
    }

    // Check minimum support fallback when leaf rows < features + 1
    let mut x_tiny = Vec::new();
    let mut y_tiny = Vec::new();
    for i in 0..12 {
        x_tiny.push(vec![Some(i as f64), Some((i * 2) as f64), Some((i * 3) as f64)]);
        y_tiny.push(Some(i as f64));
    }
    let params_tiny = ModelTreeParams {
        max_depth: 3,
        min_rows: 3, // P=3 => requires 4 rows for linear fit
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Huber,
        huber_delta: 1.345,
        ..Default::default()
    };
    let tree_tiny = ModelTree::fit(&x_tiny, &y_tiny, params_tiny).expect("Fit should succeed with 0D center fallback");
    let preds_tiny = tree_tiny.predict(&x_tiny);
    assert_eq!(preds_tiny.len(), 12);
    for p in preds_tiny {
        assert!(p.is_finite(), "Prediction should be finite from 0D Huber center fallback");
    }
}

// =========================================================================
// Spec 27 Phase 4: Expectile Regression Tests
// =========================================================================

#[test]
fn test_expectile_loss_and_deriv() {
    use mltreelib_core::solver::problems::expectile::{expectile_deriv, expectile_loss};

    // For tau = 0.5, symmetric squared error: loss = 0.5 * r^2, deriv = r
    let r_pos = 2.0;
    let r_neg = -2.0;
    assert!((expectile_loss(r_pos, 0.5) - 2.0).abs() < 1e-10);
    assert!((expectile_loss(r_neg, 0.5) - 2.0).abs() < 1e-10);
    assert!((expectile_deriv(r_pos, 0.5) - 2.0).abs() < 1e-10);
    assert!((expectile_deriv(r_neg, 0.5) - (-2.0)).abs() < 1e-10);

    // For tau = 0.8:
    // r > 0 => weight = 0.8, loss = 0.8 * 4.0 = 3.2, deriv = 2 * 0.8 * 2 = 3.2
    // r < 0 => weight = 0.2, loss = 0.2 * 4.0 = 0.8, deriv = 2 * 0.2 * (-2) = -0.8
    assert!((expectile_loss(r_pos, 0.8) - 3.2).abs() < 1e-10);
    assert!((expectile_loss(r_neg, 0.8) - 0.8).abs() < 1e-10);
    assert!((expectile_deriv(r_pos, 0.8) - 3.2).abs() < 1e-10);
    assert!((expectile_deriv(r_neg, 0.8) - (-0.8)).abs() < 1e-10);
}

#[test]
fn test_expectile_mean_recovery_tau_half() {
    use mltreelib_core::solver::problems::expectile::solve_expectile_lbfgs;
    use mltreelib_core::solver::DenseView;

    // Symmetric clean data: y = 3.0 + 2.0 * x1 - 1.0 * x2
    let n = 100;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut rng = 424242u64;
    let mut lcg = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    };

    for _ in 0..n {
        let x1 = lcg() * 2.0 - 1.0;
        let x2 = lcg() * 2.0 - 1.0;
        let y = 3.0 + 2.0 * x1 - 1.0 * x2;
        x_mat.push(vec![x1, x2]);
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);
    let res = solve_expectile_lbfgs(&view, 0.5, 1e-4, 100, 1e-5, None)
        .expect("Expectile L-BFGS should solve");

    assert!((res[0] - 3.0).abs() < 1e-2, "Intercept should be ~3.0, got {}", res[0]);
    assert!((res[1] - 2.0).abs() < 1e-2, "Beta1 should be ~2.0, got {}", res[1]);
    assert!((res[2] - (-1.0)).abs() < 1e-2, "Beta2 should be ~ -1.0, got {}", res[2]);
}

#[test]
fn test_expectile_ordering_monotonicity() {
    use mltreelib_core::solver::problems::expectile::solve_expectile_lbfgs;
    use mltreelib_core::solver::DenseView;

    let n = 200;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut rng = 123456789u64;
    let mut lcg = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    };

    for _ in 0..n {
        let x1 = lcg() * 4.0 - 2.0;
        let noise = (lcg() - 0.5) * 10.0;
        let y = 5.0 + 1.5 * x1 + noise;
        x_mat.push(vec![x1]);
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);
    let res_10 = solve_expectile_lbfgs(&view, 0.1, 1e-4, 100, 1e-5, None).unwrap();
    let res_50 = solve_expectile_lbfgs(&view, 0.5, 1e-4, 100, 1e-5, None).unwrap();
    let res_90 = solve_expectile_lbfgs(&view, 0.9, 1e-4, 100, 1e-5, None).unwrap();

    assert!(
        res_10[0] < res_50[0],
        "tau=0.1 intercept ({}) must be < tau=0.5 intercept ({})",
        res_10[0],
        res_50[0]
    );
    assert!(
        res_50[0] < res_90[0],
        "tau=0.5 intercept ({}) must be < tau=0.9 intercept ({})",
        res_50[0],
        res_90[0]
    );
}

#[test]
fn test_expectile_model_tree_fit() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let val = (i as f64) * 0.1;
        x.push(vec![Some(val), Some(val.sin())]);
        y.push(Some(2.0 + 1.5 * val));
    }

    let params = ModelTreeParams {
        max_depth: 3,
        min_rows: 10,
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Expectile,
        expectile_tau: 0.75,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit with expectile leaf should succeed");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 60);
    for p in preds {
        assert!(p.is_finite());
    }
}

// =========================================================================
// Spec 27 Phase 5: Non-Negative Least Squares (NNLS) Tests
// =========================================================================

#[test]
fn test_solve_nnls_positive_coefficients() {
    use mltreelib_core::solver::problems::nnls::solve_nnls;
    use mltreelib_core::solver::DenseView;

    // Negative relationship: y = 10.0 - 3.0 * x1 + 4.0 * x2
    // Unconstrained OLS would give beta1 < 0.
    // NNLS with non_negative_intercept = false must clamp beta1 to 0.0 and beta2 > 0.
    let n = 80;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) / 10.0;
        let x2 = ((n - i) as f64) / 10.0;
        let y = 10.0 - 3.0 * x1 + 4.0 * x2;
        x_mat.push(vec![x1, x2]);
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);
    let res = solve_nnls(&view, 1e-6, false, 100, 1e-8).expect("NNLS should solve");

    // beta1 must be 0 (in active set) or >= 0
    assert!(res.coefficients[1] >= -1e-10, "beta1 must be non-negative, got {}", res.coefficients[1]);
    assert!(res.coefficients[2] > 0.0, "beta2 must be strictly positive, got {}", res.coefficients[2]);
}

#[test]
fn test_nnls_model_tree_fit() {
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let x1 = i as f64;
        let x2 = (60 - i) as f64;
        x.push(vec![Some(x1), Some(x2)]);
        y.push(Some(5.0 - 2.0 * x1 + 3.0 * x2));
    }

    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        leaf: mltreelib_core::leaf::LeafKind::Linear,
        non_negative: true,
        non_negative_intercept: false,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit with NNLS should succeed");
    for (leaf_model, _) in tree.leaves() {
        if let Some(ref lm) = leaf_model.linear {
            // Slope coefficients must all be >= 0
            for (idx, &coef) in lm.coef.iter().enumerate() {
                assert!(
                    coef >= -1e-8,
                    "Leaf slope coefficient {} must be >= 0, got {}",
                    idx,
                    coef
                );
            }
        }
    }
}

// =========================================================================
// Spec 27 Phase 7: Negative Binomial & Beta Regression Tests
// =========================================================================

#[test]
fn test_negative_binomial_dispersion_and_recovery() {
    use mltreelib_core::solver::problems::glm_extra::solve_negative_binomial_regression;
    use mltreelib_core::solver::DenseView;

    let n = 120;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) / 30.0;
        let mu = (1.0 + 0.5 * x1).exp();
        // Add artificial overdispersion variance
        let y = (mu + ((i % 5) as f64) * 2.0).round().max(0.0);
        x_mat.push(vec![x1]);
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);
    let res = solve_negative_binomial_regression(&view, None, 1e-4, 50, 1e-6)
        .expect("Negative Binomial GLM should solve");

    assert_eq!(res.coefficients.len(), 2); // [intercept, beta1]
    assert!(res.alpha >= 0.0, "Dispersion alpha must be non-negative, got {}", res.alpha);
}

#[test]
fn test_negative_binomial_poisson_boundary() {
    use mltreelib_core::solver::problems::glm_extra::nb2_unit_deviance;

    // At alpha -> 0, NB2 unit deviance should match Poisson unit deviance: 2 * (y * ln(y/mu) - (y - mu))
    let y = 10.0f64;
    let mu = 8.0f64;
    let poisson_dev = 2.0 * (y * (y / mu).ln() - (y - mu));
    let nb_dev = nb2_unit_deviance(y, mu, 1e-8);
    assert!(
        (nb_dev - poisson_dev).abs() < 1e-4,
        "NB deviance at alpha=0 ({nb_dev}) should match Poisson ({poisson_dev})"
    );
}

#[test]
fn test_negative_binomial_model_tree_fit() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let x1 = (i as f64) * 0.1;
        x.push(vec![Some(x1)]);
        y.push(Some(((1.0 + 0.3 * x1).exp()).round()));
    }

    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::NegativeBinomial,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit with NegativeBinomial should succeed");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 60);
    for p in preds {
        assert!(p.is_finite() && p >= 0.0, "Predictions should be non-negative counts");
    }
}

#[test]
fn test_beta_regression_bounded_and_recovery() {
    use mltreelib_core::solver::problems::glm_extra::solve_beta_regression;
    use mltreelib_core::solver::DenseView;

    let n = 100;
    let mut x_mat = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) / 50.0 - 1.0;
        // logit link: eta = 0.5 * x1
        let eta = 0.5 * x1;
        let mu = 1.0 / (1.0 + (-eta).exp());
        // clip strictly inside (0, 1)
        let y = mu.clamp(0.01, 0.99);
        x_mat.push(vec![x1]);
        y_vec.push(y);
    }

    let view = DenseView::new(&x_mat, &y_vec, None, true);
    let res = solve_beta_regression(&view, None, 1e-4, 50, 1e-6)
        .expect("Beta regression should solve");

    assert!(res.phi > 0.0, "Precision phi must be positive, got {}", res.phi);
    assert_eq!(res.coefficients.len(), 2);
}

#[test]
fn test_beta_model_tree_fit() {
    use mltreelib_core::family::LeafFamily;
    use mltreelib_core::leaf::LeafKind;
    use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let x1 = (i as f64) * 0.1 - 3.0;
        let eta = 0.4 * x1;
        let mu = 1.0 / (1.0 + (-eta).exp());
        x.push(vec![Some(x1)]);
        y.push(Some(mu.clamp(0.02, 0.98)));
    }

    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        leaf: LeafKind::Linear,
        leaf_family: LeafFamily::Beta,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("ModelTree fit with Beta family should succeed");
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 60);
    for p in preds {
        assert!(p > 0.0 && p < 1.0, "Predictions should be strictly inside (0, 1), got {p}");
    }
}

