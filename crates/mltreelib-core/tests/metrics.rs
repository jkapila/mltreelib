use mltreelib_core::metrics::{
    classification_report, grouped_regression_report, regression_report,
};

fn s(v: &str) -> Option<String> {
    Some(v.to_string())
}

#[test]
fn regression_perfect_and_known_values() {
    let r = regression_report(&[Some(1.0), Some(2.0)], &[Some(1.0), Some(2.0)]).unwrap();
    assert_eq!(r.mae, 0.0);
    assert!((r.r2 - 1.0).abs() < 1e-12);
    assert_eq!(r.max_error, 0.0);
    assert_eq!(r.deciles.len(), 10);

    // off-by-one everywhere: error = actual - fitted = -1 on every row;
    // rmse 1, mad 0, bias 3/2, mape 61.1.., r2 -0.5 (worse than the mean)
    let r = regression_report(
        &[Some(1.0), Some(2.0), Some(3.0)],
        &[Some(2.0), Some(3.0), Some(4.0)],
    )
    .unwrap();
    assert_eq!(r.mean_error, -1.0);
    assert_eq!(r.rmse, 1.0);
    assert_eq!(r.mae, 1.0);
    assert_eq!(r.mad, 0.0);
    assert_eq!(r.bias, 1.5);
    assert!((r.mape - 100.0 * (1.0 + 0.5 + 1.0 / 3.0) / 3.0).abs() < 1e-9);
    assert_eq!(r.max_error, 1.0);
    assert!((r.r2 + 0.5).abs() < 1e-12);
    assert!((r.explained_variance - 1.0).abs() < 1e-12); // constant error
}

#[test]
fn regression_decile_profile_and_missing() {
    // ten rows, one per decile; errors grow with the target -> bathtub shape
    let t: Vec<Option<f64>> = (0..10).map(|i| Some(i as f64)).collect();
    let p: Vec<Option<f64>> = (0..10).map(|i| Some(i as f64 + 1.0)).collect();
    let r = regression_report(&t, &p).unwrap();
    assert_eq!(r.deciles.len(), 10);
    for (i, d) in r.deciles.iter().enumerate() {
        assert_eq!(d.decile, i + 1);
        assert_eq!(d.n, 1);
        assert_eq!(d.mean_error, -1.0); // actual - fitted
        assert_eq!(d.mae, 1.0);
    }

    // missing pairs dropped and counted, never hidden
    let r = regression_report(&[Some(1.0), None, Some(3.0)], &[Some(1.0), Some(2.0), Some(5.0)]).unwrap();
    assert_eq!(r.n, 2);
    assert_eq!(r.n_missing, 1);
    assert_eq!(r.mae, 1.0);
    assert!(regression_report(&[], &[]).is_err());
    assert!(regression_report(&[Some(1.0)], &[Some(1.0), Some(2.0)]).is_err());
}

#[test]
fn classification_report_all_at_once() {
    let y_true = vec![Some(0.0), Some(0.0), Some(1.0), Some(1.0)];
    let y_pred = vec![Some(0.0), Some(1.0), Some(1.0), Some(1.0)];
    let probs = vec![Some(0.1), Some(0.6), Some(0.7), Some(0.9)];
    let c = classification_report(&y_true, &y_pred, Some(&probs)).unwrap();
    assert_eq!(c.n, 4);
    assert!((c.accuracy - 0.75).abs() < 1e-12);
    assert_eq!(c.confusion[0], vec![1.0, 1.0]);
    assert_eq!(c.confusion[1], vec![0.0, 2.0]);
    assert!((c.per_class[1].precision - 2.0 / 3.0).abs() < 1e-12);
    assert!((c.per_class[1].recall - 1.0).abs() < 1e-12);
    assert!((c.auc - 1.0).abs() < 1e-12); // perfect ranking
    let expected_ll = -(0.9f64.ln() + 0.4f64.ln() + 0.7f64.ln() + 0.9f64.ln()) / 4.0;
    assert!((c.logloss - expected_ll).abs() < 1e-9);
    assert!((c.brier - 0.1175).abs() < 1e-9);

    // no probabilities -> probabilistic metrics are NaN
    let c = classification_report(&y_true, &y_pred, None).unwrap();
    assert!(c.logloss.is_nan() && c.brier.is_nan() && c.auc.is_nan());

    // missing pairs counted
    let c = classification_report(
        &[Some(0.0), None, Some(1.0)],
        &[Some(0.0), Some(1.0), Some(1.0)],
        None,
    )
    .unwrap();
    assert_eq!(c.n, 2);
    assert_eq!(c.n_missing, 1);
}

#[test]
fn grouped_regression_matches_ungrouped() {
    let keys = vec![s("a"), s("a"), s("b"), None];
    let t = vec![Some(1.0), Some(3.0), Some(10.0), Some(99.0)];
    let p = vec![Some(2.0), Some(2.0), Some(12.0), Some(0.0)];
    let g = grouped_regression_report(&keys, &t, &p).unwrap();
    assert_eq!(g.len(), 2); // missing key excluded
    assert_eq!(g[0].0, "a");
    assert_eq!(g[0].1.mae, 1.0); // |1-2|, |3-2| -> mean 1
    assert_eq!(g[1].0, "b");
    assert_eq!(g[1].1.mae, 2.0);
    assert!(grouped_regression_report(&keys[..1], &t, &p).is_err());
}
