use mltreelib_core::solver::fit_optimal_tree;

#[test]
fn test_optimal_decision_stump_classification() {
    // 20 samples: x0 in 0..10 -> class 0, x0 in 11..20 -> class 1
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        x.push(vec![i as f64, (i % 3) as f64]);
        y.push(if i < 10 { 0.0 } else { 1.0 });
    }

    let res = fit_optimal_tree(&x, &y, "classification", 1, 1, 0.0).expect("stump fit");
    assert_eq!(res.max_depth, 1);
    assert_eq!(res.n_splits, 1);
    assert_eq!(res.total_loss, 0.0); // 0 misclassifications

    let preds = res.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert_eq!(p, y[i]);
    }
}

#[test]
fn test_optimal_tree_depth2_xor_classification() {
    // 2D XOR pattern
    // (x0 <= 5, x1 <= 5) -> 0
    // (x0 <= 5, x1 > 5) -> 1
    // (x0 > 5, x1 <= 5) -> 1
    // (x0 > 5, x1 > 5) -> 0
    let mut x = Vec::new();
    let mut y = Vec::new();

    let grid = [
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (2.0, 8.0, 1.0),
        (3.0, 7.0, 1.0),
        (8.0, 2.0, 1.0),
        (7.0, 3.0, 1.0),
        (8.0, 8.0, 0.0),
        (7.0, 7.0, 0.0),
    ];

    for &(x0, x1, label) in &grid {
        x.push(vec![x0, x1]);
        y.push(label);
    }

    let res = fit_optimal_tree(&x, &y, "classification", 2, 1, 0.0).expect("depth 2 fit");
    assert_eq!(res.max_depth, 2);
    // XOR should be perfectly separated by depth 2 tree (3 splits)
    assert_eq!(res.total_loss, 0.0);

    let preds = res.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert_eq!(p, y[i]);
    }
}

#[test]
fn test_optimal_tree_regression() {
    // 30 samples, 1D piecewise constant:
    // x < 5 -> y = 2.0
    // x >= 5 and x < 15 -> y = 10.0
    // x >= 15 -> y = 20.0
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..30 {
        let val = i as f64;
        let target = if val < 5.0 {
            2.0
        } else if val < 15.0 {
            10.0
        } else {
            20.0
        };
        x.push(vec![val]);
        y.push(target);
    }

    let res = fit_optimal_tree(&x, &y, "regression", 2, 2, 0.0).expect("reg tree fit");
    assert_eq!(res.max_depth, 2);
    assert!(res.total_loss < 1e-6);

    let preds = res.predict(&x);
    for (i, &p) in preds.iter().enumerate() {
        assert!((p - y[i]).abs() < 1e-4);
    }
}
