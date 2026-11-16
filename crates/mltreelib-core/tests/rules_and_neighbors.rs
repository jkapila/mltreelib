use mltreelib_core::leaf::{LeafCols, LeafKind};
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

#[test]
fn test_rules_extraction_and_prediction() {
    // Generate synthetic 2D data:
    // if x0 > 0.5 and x1 > 0.5 => y = 10.0 else y = 1.0
    let n = 200;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x0 = (i as f64) / (n as f64);
        let x1 = ((i * 7) % n) as f64 / (n as f64);
        let val = if x0 > 0.5 && x1 > 0.5 { 10.0 } else { 1.0 };
        x.push(vec![Some(x0), Some(x1)]);
        y.push(val);
    }

    let y_opt: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    let col_names = vec!["x0".to_string(), "x1".to_string()];

    let mut params = ModelTreeParams::default();
    params.max_depth = 4;
    params.min_rows = 5;
    params.leaf = LeafKind::Mean;

    let tree = ModelTree::fit_named(col_names, &x, &y_opt, params).expect("fit should succeed");

    // Extract C5.0 rule set with pessimistic pruning
    let ruleset = tree.extract_rules(Some(&x), Some(&y), true, 0.25);

    assert!(!ruleset.rules.is_empty(), "Extracted rule set must not be empty");
    assert!(ruleset.rules.len() <= tree.n_leaves(), "Pruned rules should be compact");

    // Check predictions from rules
    let preds = ruleset.predict(&x);
    assert_eq!(preds.len(), n);

    let mut mae = 0.0;
    for i in 0..n {
        mae += (preds[i] - y[i]).abs();
    }
    mae /= n as f64;
    assert!(mae < 1.0, "Rule-based MAE {} should be small on clean step data", mae);

    // Rule string representation
    let rule_str = format!("{}", ruleset);
    assert!(rule_str.contains("Rule"));
    assert!(rule_str.contains("Default Rule:"));
}

#[test]
fn test_regression_summary_r_format() {
    // Linear regression data: y = 2.0 + 3.0*x0 - 1.5*x1 + noise
    let n = 100;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let x0 = (i as f64) / (n as f64);
        let x1 = ((i * 3) % n) as f64 / (n as f64);
        let target = 2.0 + 3.0 * x0 - 1.5 * x1 + ((i % 5) as f64 - 2.0) * 0.05;
        x.push(vec![Some(x0), Some(x1)]);
        y.push(Some(target));
    }

    let col_names = vec!["x0".to_string(), "x1".to_string()];
    let mut params = ModelTreeParams::default();
    params.max_depth = 1;
    params.min_rows = 10;
    params.leaf = LeafKind::Linear;
    params.leaf_cols = LeafCols::All;

    let tree = ModelTree::fit_named(col_names, &x, &y, params).expect("fit should succeed");

    let summary_str = tree.summary();
    assert!(summary_str.contains("mltreelib ModelTree Summary"));
    assert!(summary_str.contains("Residual standard error:"));
    assert!(summary_str.contains("Multiple R-squared:"));
    assert!(summary_str.contains("F-statistic:"));
    assert!(summary_str.contains("Coefficients:"));
    assert!(summary_str.contains("Estimate"));
    assert!(summary_str.contains("Std. Error"));
    assert!(summary_str.contains("t value"));
    assert!(summary_str.contains("Pr(>|t|)"));

    let leaf0_summary = tree.leaf_summary(0).expect("leaf 0 summary should succeed");
    assert!(leaf0_summary.contains("Estimate"));
    assert!(leaf0_summary.contains("(Intercept)"));
}

#[test]
fn test_knn_neighbor_adjustment() {
    // Generate data where tree makes systematic residual errors
    let n = 150;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let v = (i as f64) / (n as f64) * 10.0;
        // Non-linear sine wave that piece-wise constant / shallow linear tree approximates
        let target = (v * 0.8).sin() * 5.0 + v * 0.5;
        x.push(vec![Some(v)]);
        y.push(Some(target));
    }

    let col_names = vec!["x0".to_string()];
    let mut params = ModelTreeParams::default();
    params.max_depth = 2;
    params.min_rows = 15;
    params.leaf = LeafKind::Mean;

    let tree = ModelTree::fit_named(col_names, &x, &y, params).expect("fit should succeed");

    // Standard tree prediction
    let unadj = tree.predict(&x);

    // Cubist-style k-NN adjusted prediction with 5 neighbors
    let adj = tree.predict_adjusted(&x, 5);

    assert_eq!(unadj.len(), n);
    assert_eq!(adj.len(), n);

    let mut unadj_sse = 0.0;
    let mut adj_sse = 0.0;
    for i in 0..n {
        let true_y = y[i].unwrap();
        unadj_sse += (unadj[i] - true_y).powi(2);
        adj_sse += (adj[i] - true_y).powi(2);
    }

    // k-NN sample neighbor adjustment uses training residuals to correct local errors,
    // so on training samples it substantially lowers error
    assert!(
        adj_sse < unadj_sse,
        "k-NN neighbor adjustment should reduce error: adj_sse={} vs unadj_sse={}",
        adj_sse,
        unadj_sse
    );

    // With 0 neighbors, predict_adjusted must equal predict
    let zero_neighbors = tree.predict_adjusted(&x, 0);
    assert_eq!(zero_neighbors, unadj);
}
