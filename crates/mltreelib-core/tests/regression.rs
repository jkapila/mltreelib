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
