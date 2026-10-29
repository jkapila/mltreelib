use mltreelib_core::metrics::{
    brier_score, calibration_bins, classification_report, classification_scores, decile_table,
    double_lift, gain_curve, grouped_regression_report, hinge_loss, ks_statistic, lift_curve,
    log_loss, multilabel_scores, pairwise_distances, pr_curve, regression_report, roc_curve,
    stability_index, weighted_kappa,
};

#[test]
fn roc_auc_orders_perfectly() {
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let (pts, auc) = roc_curve(&y, &[0.1, 0.2, 0.8, 0.9]).unwrap();
    assert!((auc - 1.0).abs() < 1e-12, "{auc}");
    assert_eq!(pts.first().unwrap(), &(0.0, 0.0));
    assert_eq!(pts.last().unwrap(), &(1.0, 1.0));
    let (_, auc_rev) = roc_curve(&y, &[0.9, 0.8, 0.2, 0.1]).unwrap();
    assert!(auc_rev.abs() < 1e-12, "{auc_rev}");
}

#[test]
fn pr_average_precision_perfect_and_gains_reach_one() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let s = vec![0.1, 0.9, 0.2, 0.8];
    let (pts, ap) = pr_curve(&y, &s).unwrap();
    assert!((ap - 1.0).abs() < 1e-12, "{ap}");
    // perfect ranking: precision 1.0 until recall reaches 1.0; the trailing
    // points are the negative tail at recall 1 (precision falls to prevalence)
    assert!(pts.iter().filter(|(r, _)| *r < 1.0).all(|(_, p)| (*p - 1.0).abs() < 1e-12), "{pts:?}");
    let g = gain_curve(&y, &s).unwrap();
    assert_eq!(g.first().unwrap(), &(0.0, 0.0));
    assert_eq!(g.last().unwrap(), &(1.0, 1.0));
}

#[test]
fn calibration_of_a_perfect_forecaster_is_diagonal() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let s = vec![0.0, 1.0, 0.0, 1.0];
    let bins = calibration_bins(&y, &s, 4).unwrap();
    assert!(!bins.is_empty());
    for b in bins {
        assert!((b.mean_score - b.observed).abs() < 1e-12, "{b:?}");
    }
}

#[test]
fn curves_reject_bad_input() {
    assert!(roc_curve(&[0.0, 1.0], &[0.5]).is_err());
    assert!(roc_curve(&[1.0, 1.0], &[0.2, 0.9]).is_err());
    assert!(pr_curve(&[0.0, 0.0], &[0.2, 0.9]).is_err());
    assert!(calibration_bins(&[0.0], &[], 4).is_err());
}

#[test]
fn ks_perfect_separation_is_one() {
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let r = ks_statistic(&y, &[0.1, 0.2, 0.8, 0.9]).unwrap();
    assert!((r.ks - 1.0).abs() < 1e-12, "{}", r.ks);
    assert!(r.score_at_ks > 0.2 && r.score_at_ks <= 0.8, "{}", r.score_at_ks);
    assert_eq!(r.tpr_at_ks, 1.0);
    assert_eq!(r.fpr_at_ks, 0.0);
}

#[test]
fn decile_profile_tracks_the_ranking() {
    // 20 rows, 4 events all at the top scores -> decile 1 holds every event.
    let y: Vec<f64> = (0..20).map(|i| if i < 4 { 1.0 } else { 0.0 }).collect();
    let s: Vec<f64> = (0..20).map(|i| (20 - i) as f64 / 20.0).collect();
    let rows = decile_table(&y, &s, 4).unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].events, 4.0);
    assert!((rows[0].cum_capture - 1.0).abs() < 1e-12);
    assert!((rows[0].lift - 4.0).abs() < 1e-12, "{}", rows[0].lift);
    assert_eq!(rows.iter().map(|r| r.n).sum::<usize>(), 20);
    // cumulative response is non-increasing for a perfect ranking
    for w in rows.windows(2) {
        assert!(w[0].cum_event_rate >= w[1].cum_event_rate - 1e-12);
    }
}

#[test]
fn lift_starts_high_and_ends_one() {
    let y: Vec<f64> = (0..20).map(|i| if i < 4 { 1.0 } else { 0.0 }).collect();
    let s: Vec<f64> = (0..20).map(|i| (20 - i) as f64 / 20.0).collect();
    let pts = lift_curve(&y, &s).unwrap();
    assert!((pts.last().unwrap().1 - 1.0).abs() < 1e-12, "{pts:?}");
    assert!(pts[1].1 > 1.5, "{pts:?}");
}

#[test]
fn psi_zero_for_same_distribution_and_positive_for_shift() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
    let (same, _) = stability_index(&a, &a, 10).unwrap();
    assert!(same.abs() < 1e-12, "{same}");
    let b: Vec<f64> = (50..150).map(|i| i as f64 / 100.0).collect();
    let (shift, bins) = stability_index(&a, &b, 10).unwrap();
    assert!(shift > 0.1, "{shift}");
    assert_eq!(bins.len(), 10);
}

#[test]
fn double_lift_covers_every_row() {
    let y: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let a: Vec<f64> = (0..30).map(|i| (i as f64 * 2.0) + (i % 3) as f64).collect();
    let b: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let rows = double_lift(&y, &a, &b, 5).unwrap();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows.iter().map(|r| r.n).sum::<usize>(), 30);
    // ranking by A descends -> bucket 1 holds the largest A predictions
    assert!(rows[0].pred_a > rows[4].pred_a);
}

#[test]
fn kappa_quadratic_matches_perfect_and_punishes_reversal() {
    let y = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    let perfect = weighted_kappa(&y, &y, "quadratic").unwrap();
    assert!((perfect - 1.0).abs() < 1e-12, "{perfect}");
    let rev: Vec<f64> = y.iter().map(|v| 3.0 - v).collect();
    let bad = weighted_kappa(&y, &rev, "quadratic").unwrap();
    assert!(bad < 0.5, "{bad}");
    let none = weighted_kappa(&y, &rev, "none").unwrap();
    assert!(none < 0.5, "{none}");
    // linear == none for two distinct labels (weights coincide)
    let two_y = vec![0.0, 1.0, 1.0, 0.0];
    let two_p = vec![0.0, 1.0, 0.0, 0.0];
    let lin = weighted_kappa(&two_y, &two_p, "linear").unwrap();
    let non = weighted_kappa(&two_y, &two_p, "none").unwrap();
    assert!((lin - non).abs() < 1e-12, "linear {lin} != none {non}");
}

#[test]
fn probability_scores_reward_confidence() {
    let y = vec![0.0, 1.0];
    let good = vec![0.01, 0.99];
    let bad = vec![0.99, 0.01];
    assert!(log_loss(&y, &good).unwrap() < log_loss(&y, &bad).unwrap());
    assert!(brier_score(&y, &good).unwrap() < brier_score(&y, &bad).unwrap());
    assert!(brier_score(&y, &vec![0.0, 1.0]).unwrap().abs() < 1e-12);
}

#[test]
fn classification_scores_match_hand_computation() {
    // truth [0,0,1,1], pred [0,1,1,1]: tp=2, fp=1, fn=0, tn=1 per class 1
    let s = classification_scores(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 1.0]).unwrap();
    assert!((s.accuracy - 0.75).abs() < 1e-12);
    assert!((s.balanced_accuracy - 0.75).abs() < 1e-12);
    assert!((s.f1_macro - (2.0 / 3.0 + 0.8) / 2.0).abs() < 1e-12);
    assert!((s.mcc - (4.0 / (8.0f64 * 6.0).sqrt())).abs() < 1e-12);
    assert!((s.hamming_loss - 0.25).abs() < 1e-12);
    assert!((s.zero_one_loss - 0.25).abs() < 1e-12);
    assert!((s.jaccard_macro - (0.5 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
}

#[test]
fn multilabel_scores_cover_the_surface() {
    let y = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
    ];
    let perfect = vec![
        vec![0.9, 0.2, 0.8],
        vec![0.3, 0.7, 0.1],
        vec![0.6, 0.55, 0.4],
    ];
    let s = multilabel_scores(&y, &perfect).unwrap();
    assert!((s.subset_accuracy - 1.0).abs() < 1e-12);
    assert!(s.hamming_loss.abs() < 1e-12);
    assert!((s.f1_samples - 1.0).abs() < 1e-12);
    assert!((s.label_ranking_ap - 1.0).abs() < 1e-12);
    assert!((s.coverage_error - 5.0 / 3.0).abs() < 1e-12, "{}", s.coverage_error);

    let one_miss = vec![
        vec![0.9, 0.2, 0.8],
        vec![0.3, 0.7, 0.1],
        vec![0.6, 0.55, 0.9], // false positive on label 2
    ];
    let s2 = multilabel_scores(&y, &one_miss).unwrap();
    assert!(s2.subset_accuracy < 1.0);
    assert!((s2.hamming_loss - 1.0 / 9.0).abs() < 1e-12);
}

#[test]
fn hinge_loss_zero_for_confident_margins() {
    assert!(hinge_loss(&[1.0, 0.0], &[2.0, -3.0]).unwrap().abs() < 1e-12);
    let bad = hinge_loss(&[1.0, 0.0], &[-2.0, 3.0]).unwrap();
    assert!((bad - 3.5).abs() < 1e-12, "{bad}"); // (3 + 4) / 2
}

#[test]
fn pairwise_distances_match_geometry() {
    let a = vec![vec![0.0, 0.0]];
    let b = vec![vec![3.0, 4.0]];
    let e = pairwise_distances(&a, &b, "euclidean", 2.0).unwrap();
    assert!((e[0][0] - 5.0).abs() < 1e-12);
    let m = pairwise_distances(&a, &b, "manhattan", 2.0).unwrap();
    assert!((m[0][0] - 7.0).abs() < 1e-12);
    let c = pairwise_distances(&a, &b, "chebyshev", 2.0).unwrap();
    assert!((c[0][0] - 4.0).abs() < 1e-12);
    let mi = pairwise_distances(&a, &b, "minkowski", 3.0).unwrap();
    assert!((mi[0][0] - 91.0f64.powf(1.0 / 3.0)).abs() < 1e-9);

    let u = vec![vec![1.0, 0.0]];
    let v = vec![vec![0.0, 1.0]];
    let cos = pairwise_distances(&u, &v, "cosine", 2.0).unwrap();
    assert!((cos[0][0] - 1.0).abs() < 1e-12);
    let ham = pairwise_distances(&[vec![0.0, 1.0, 1.0]], &[vec![0.0, 1.0, 0.0]], "hamming", 2.0).unwrap();
    assert!((ham[0][0] - 1.0 / 3.0).abs() < 1e-12);
    let jac = pairwise_distances(&[vec![1.0, 1.0, 0.0]], &[vec![1.0, 0.0, 0.0]], "jaccard", 2.0).unwrap();
    assert!((jac[0][0] - 0.5).abs() < 1e-12);
    assert!(pairwise_distances(&a, &b, "nope", 2.0).is_err());
}

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
