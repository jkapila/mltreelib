//! Model tree tests: honest-OLS leaves, structure recovery (house signature),
//! the two leaf-column policies, and NA-first behavior (learnable missing
//! direction + NA-indicator leaf terms).

use mltreelib_core::model_tree::{LeafCols, ModelTree, ModelTreeParams};

#[test]
fn stump_is_global_ols() {
    // y = 2 + 3a - b exactly; depth 0 => one leaf recovering the coefficients.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let a = i as f64 / 200.0;
        let b = ((i * 7) % 50) as f64 / 50.0;
        x.push(vec![Some(a), Some(b)]);
        y.push(Some(2.0 + 3.0 * a - b));
    }
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.n_leaves(), 1);
    assert!(tree.splits().is_empty());
    let (leaf, n) = tree.leaves()[0].clone();
    let m = leaf.linear.as_ref().unwrap();
    assert_eq!(n, 200);
    assert_eq!(m.cols, vec![0, 1]);
    assert!(m.na_coef.is_empty(), "no NAs -> no indicator terms");
    assert!((m.intercept - 2.0).abs() < 1e-6, "{}", m.intercept);
    assert!((m.coef[0] - 3.0).abs() < 1e-6, "{:?}", m.coef);
    assert!((m.coef[1] + 1.0).abs() < 1e-6, "{:?}", m.coef);
    // Predictions are always produced.
    let preds = tree.predict(&[vec![Some(0.5), Some(0.25)]]);
    assert!((preds[0] - (2.0 + 1.5 - 0.25)).abs() < 1e-6);
}

#[test]
fn recovers_planted_split_and_leaf_models() {
    // Two linear regimes split on z at 0: y = 1 + 2x (z<0), y = 5 - 3x (z>=0).
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let xv = ((i * 13) % 100) as f64 / 100.0;
        let zv = (i as f64 / 200.0) - 0.5;
        let yv = if zv < 0.0 {
            1.0 + 2.0 * xv
        } else {
            5.0 - 3.0 * xv
        };
        x.push(vec![Some(xv), Some(zv)]);
        y.push(Some(yv));
    }
    let p = ModelTreeParams {
        max_depth: 3,
        min_rows: 20,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();

    // Structure recovery: the split must be on z (column 1) near 0.
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected exactly one planted split");
    assert_eq!(splits[0].col, 1);
    assert!(splits[0].threshold.abs() < 0.05, "{}", splits[0].threshold);

    // Leaf models recover the planted coefficients (z-coef ~ 0 inside leaves).
    assert_eq!(tree.n_leaves(), 2);
    let mut leaves = tree.leaves();
    leaves.sort_by(|a, b| {
        a.0.linear
            .as_ref()
            .unwrap()
            .intercept
            .partial_cmp(&b.0.linear.as_ref().unwrap().intercept)
            .unwrap()
    });
    let lo = leaves[0].0.linear.as_ref().unwrap();
    let hi = leaves[1].0.linear.as_ref().unwrap();
    assert!((lo.intercept - 1.0).abs() < 1e-6, "{}", lo.intercept);
    assert!((lo.coef[0] - 2.0).abs() < 1e-6, "{:?}", lo.coef);
    assert!((hi.intercept - 5.0).abs() < 1e-6, "{}", hi.intercept);
    assert!((hi.coef[0] + 3.0).abs() < 1e-6, "{:?}", hi.coef);

    // Every row (NA-free here) gets a prediction.
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 200);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn path_used_leaves_are_piecewise_linear() {
    // y is piecewise linear in z itself: y = 2z (z<0), y = 4z+1 (z>=0).
    // PathUsed leaves regress on the split column only -> the two segments.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..200 {
        let z = (i as f64 / 200.0) - 0.5;
        let yv = if z < 0.0 { 2.0 * z } else { 4.0 * z + 1.0 };
        x.push(vec![Some(z)]);
        y.push(Some(yv));
    }
    let p = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        leaf_cols: LeafCols::PathUsed,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.splits().len(), 1);
    let mut leaves = tree.leaves();
    leaves.sort_by(|a, b| {
        a.0.linear
            .as_ref()
            .unwrap()
            .intercept
            .partial_cmp(&b.0.linear.as_ref().unwrap().intercept)
            .unwrap()
    });
    let lo = leaves[0].0.linear.as_ref().unwrap();
    let hi = leaves[1].0.linear.as_ref().unwrap();
    assert_eq!(lo.cols, vec![0]); // the split column is the leaf regressor
    assert_eq!(hi.cols, vec![0]);
    assert!((lo.intercept).abs() < 1e-6, "{}", lo.intercept);
    assert!((lo.coef[0] - 2.0).abs() < 1e-6, "{:?}", lo.coef);
    assert!((hi.intercept - 1.0).abs() < 1e-6, "{}", hi.intercept);
    assert!((hi.coef[0] - 4.0).abs() < 1e-6, "{:?}", hi.coef);
}

#[test]
fn na_direction_is_learned_from_the_gain() {
    // y = z^2 on observed rows (z = 0..99) — quadratic, so splitting helps;
    // 30 missing-z rows have y = 0, sitting with the low-z regime. The learned
    // default direction must send them left (missing_left = true).
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let z = i as f64;
        x.push(vec![Some(z)]);
        y.push(Some(z * z));
    }
    for _ in 0..30 {
        x.push(vec![None]);
        y.push(Some(0.0));
    }
    let p = ModelTreeParams {
        max_depth: 2,
        min_rows: 10,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert!(!tree.splits().is_empty(), "quadratic signal must split");
    let sp = tree.splits()[0];
    assert_eq!(sp.col, 0);
    assert!(sp.missing_left, "missing rows belong with the low-z side");
    // All 130 observed-target rows got a prediction, NAs included.
    let preds = tree.predict(&x);
    assert_eq!(preds.len(), 130);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn na_indicator_makes_missingness_predictive() {
    // y = 2z on observed rows (z in [-0.5, 0.5)), y = 5 exactly when z is
    // missing. A stump leaf must learn value slope 2 AND the NA-indicator
    // term -> missing rows predict ~5, observed rows ~2z.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let z = (i as f64 / 100.0) - 0.5;
        x.push(vec![Some(z)]);
        y.push(Some(2.0 * z));
    }
    for _ in 0..30 {
        x.push(vec![None]);
        y.push(Some(5.0));
    }
    x.push(vec![None]);
    y.push(None); // missing target -> the only dropped row
    let p = ModelTreeParams {
        max_depth: 0,
        min_rows: 5,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    let (leaf, n) = tree.leaves()[0].clone();
    let m = leaf.linear.as_ref().unwrap();
    assert_eq!(n, 130, "NA rows train; only missing-y rows drop");
    assert_eq!(m.na_coef.len(), 1, "missingness is a learned term");
    let preds = tree.predict(&[vec![None], vec![Some(0.25)]]);
    assert!((preds[0] - 5.0).abs() < 1e-6, "NA row predicts its regime: {}", preds[0]);
    assert!((preds[1] - 0.5).abs() < 1e-6, "observed row keeps the slope: {}", preds[1]);
}

#[test]
fn min_rows_blocks_tiny_leaves() {
    // Without enough rows per side the tree stays a single mean leaf.
    let x: Vec<Vec<Option<f64>>> = (0..10).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<Option<f64>> = (0..10).map(|i| Some(i as f64)).collect();
    let p = ModelTreeParams {
        max_depth: 4,
        min_rows: 20,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, p).unwrap();
    assert_eq!(tree.n_leaves(), 1);
    assert!(tree.splits().is_empty());
    let preds = tree.predict(&x);
    assert!(preds.iter().all(|p| p.is_finite()), "mean leaf still predicts");
}
