use mltreelib_core::builder::{DatasetBuilder, NaPolicy};
use mltreelib_core::timefeat::{decomposed_transform, lag_by_days, trended_transform, windowed_transform};
use mltreelib_core::transform::{summarize, GroupedStrategy, SummaryKind, VariableScaler};

#[test]
fn variable_scaler_minmax() {
    let mut vs = VariableScaler::new();
    let vals = [Some(1.0), Some(3.0), Some(5.0)];
    vs.fit(&[("a", &vals)], GroupedStrategy::MinMax).unwrap();
    let out = vs.transform("a", &vals).unwrap();
    assert_eq!(out, vec![Some(0.0), Some(0.5), Some(1.0)]);
}

#[test]
fn variable_scaler_roundtrip_and_unknown_column() {
    let mut vs = VariableScaler::new();
    let vals = [Some(1.0), Some(3.0), Some(5.0), None];
    vs.fit(
        &[("a", &vals), ("b", &[Some(10.0)])],
        GroupedStrategy::Zscore,
    )
    .unwrap();

    let out = vs.transform("a", &vals).unwrap();
    // zscore of 1,3,5: mean 3, std sqrt(8/3); first value must be negative of third
    assert!((out[1].unwrap()).abs() < 1e-12);
    assert!((out[0].unwrap() + out[2].unwrap()).abs() < 1e-12);
    assert_eq!(out[3], None);

    let back = vs.inverse_transform("a", &out).unwrap();
    for (b, v) in back.iter().zip(&vals) {
        if let (Some(b), Some(v)) = (b, v) {
            assert!((b - v).abs() < 1e-9);
        } else {
            assert_eq!(b, v);
        }
    }

    // degenerate column b (single value -> std 0) -> None, and unknown -> error
    assert_eq!(vs.transform("b", &[Some(10.0)]).unwrap(), vec![None]);
    assert!(vs.transform("zz", &[Some(1.0)]).is_err());
    assert_eq!(vs.columns(), vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn decomposed_removes_trend_and_seasonality() {
    // 14 daily rows from Monday 2026-01-05: zero-mean weekly seasonality, no trend
    let dates: Vec<Option<String>> = (0..14)
        .map(|i| Some(format!("2026-01-{:02}", 5 + i)))
        .collect();
    let vals: Vec<Option<f64>> = (0..14)
        .map(|i| Some(if i % 7 < 5 { 0.4 } else { -1.0 }))
        .collect();
    let keys = vec![Some("a".to_string()); 14];

    let d = decomposed_transform(&keys, &dates, &vals, "weekday", 7, false).unwrap();
    // interior rows (window fully inside): trend 0, seasonal == value, residual 0
    for i in 3..11 {
        assert!((d.detrended[i].unwrap() - vals[i].unwrap()).abs() < 1e-9);
        assert!(d.deseasonalized[i].unwrap().abs() < 1e-9);
        assert!(d.residual[i].unwrap().abs() < 1e-9);
    }
}

#[test]
fn decomposed_linear_trend_yields_zero_residual() {
    // pure linear series: centered MA reproduces it exactly in the interior
    let dates: Vec<Option<String>> = (0..10)
        .map(|i| Some(format!("2026-02-{:02}", 1 + i)))
        .collect();
    let vals: Vec<Option<f64>> = (0..10).map(|i| Some(3.0 + 2.0 * i as f64)).collect();
    let keys = vec![Some("a".to_string()); 10];

    let d = decomposed_transform(&keys, &dates, &vals, "month", 3, false).unwrap();
    for i in 1..9 {
        assert!(d.detrended[i].unwrap().abs() < 1e-9);
        assert!(d.residual[i].unwrap().abs() < 1e-9);
    }
}

#[test]
fn decomposed_rejects_bad_params() {
    let k = vec![Some("a".to_string())];
    let d = vec![Some("2026-01-01".to_string())];
    let v = vec![Some(1.0)];
    assert!(decomposed_transform(&k, &d, &v, "month", 0, false).is_err());
    assert!(decomposed_transform(&k, &d, &v, "nope", 3, false).is_err());
}

#[test]
fn windowed_trailing_sums_and_means() {
    let keys = vec![Some("a".to_string()); 5];
    let dates: Vec<Option<String>> = (0..5)
        .map(|i| Some(format!("2026-01-{:02}", 1 + i)))
        .collect();
    let vals = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];

    // inclusive: window covers 2 days ending at the row's date -> sums 1,3,5,7,9
    let out = windowed_transform(&keys, &dates, &vals, 2, SummaryKind::Sum, true).unwrap();
    assert_eq!(out, vec![Some(1.0), Some(3.0), Some(5.0), Some(7.0), Some(9.0)]);

    // exclusive: strictly earlier days in (d-2, d) -> None,1,1.5,2.5,3.5
    let out = windowed_transform(&keys, &dates, &vals, 2, SummaryKind::Mean, false).unwrap();
    assert_eq!(out[0], None);
    assert_eq!(out[1], Some(1.0));
    assert_eq!(out[2], Some(1.5));
    assert_eq!(out[3], Some(2.5));
    assert_eq!(out[4], Some(3.5));
}

#[test]
fn windowed_per_key_isolation_and_bad_params() {
    let keys = vec![Some("a".into()), Some("b".into()), Some("a".into())];
    let dates = vec![
        Some("2026-01-01".into()),
        Some("2026-01-01".into()),
        Some("2026-01-02".into()),
    ];
    let vals = vec![Some(1.0), Some(100.0), Some(3.0)];

    let out = windowed_transform(&keys, &dates, &vals, 2, SummaryKind::Sum, true).unwrap();
    assert_eq!(out, vec![Some(1.0), Some(100.0), Some(4.0)]); // key a: 1 then 1+3

    assert!(windowed_transform(&keys, &dates, &vals, 0, SummaryKind::Sum, true).is_err());
}

#[test]
fn summarize_shows_missing_by_default() {
    let s = summarize(&[Some(1.0), None, Some(f64::NAN), Some(3.0)]);
    assert_eq!(s.count, 2); // observed only
    assert_eq!(s.n_missing, 2); // NA + NaN, reported not hidden
    assert_eq!(s.mean, 2.0);
    let s = summarize(&[None, None]);
    assert_eq!(s.count, 0);
    assert_eq!(s.n_missing, 2);
}

#[test]
fn nan_is_missing_at_ingest() {
    // NaN is normalized to missing; filling is optional (opt-in impute)
    let ds = DatasetBuilder::new()
        .add_numeric("a", vec![Some(1.0), Some(f64::NAN)])
        .build()
        .unwrap();
    assert_eq!(ds.column("a").unwrap().na_count(), 1);

    let ds = DatasetBuilder::new()
        .na_policy(NaPolicy::Impute)
        .add_numeric("a", vec![Some(1.0), Some(f64::NAN), Some(3.0)])
        .build()
        .unwrap();
    assert_eq!(ds.numeric("a").unwrap(), &[Some(1.0), Some(2.0), Some(3.0)]);
}

#[test]
fn ts_transforms_drop_incomplete_rows() {
    let keys = vec![Some("a".into()), Some("a".into()), None];
    let dates = vec![
        Some("2026-01-01".into()),
        Some("2026-01-02".into()),
        Some("2026-01-03".into()),
    ];
    let vals = vec![Some(1.0), None, Some(3.0)];

    // row 1 (missing value) and row 2 (missing key) are dropped from estimation
    let out = lag_by_days(&keys, &dates, &vals, 1).unwrap();
    assert_eq!(out, vec![None, None, None]);

    let out = windowed_transform(&keys, &dates, &vals, 3, SummaryKind::Sum, true).unwrap();
    assert_eq!(out, vec![Some(1.0), None, None]);

    let out = trended_transform(&keys, &dates, &vals, "year_month", SummaryKind::Mean, false).unwrap();
    assert_eq!(out, vec![Some(1.0), None, None]);
}
