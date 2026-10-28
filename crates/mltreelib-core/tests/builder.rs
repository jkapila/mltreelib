use mltreelib_core::builder::{DatasetBuilder, NaPolicy};
use mltreelib_core::schema::NCategoryRule;
use mltreelib_core::transform::{GroupedScaler, GroupedStrategy};

#[test]
fn builder_rounds_digits_and_infers_categories() {
    let ds = DatasetBuilder::new()
        .digits(Some(2))
        .n_category(Some(NCategoryRule::Absolute(10)))
        .add_numeric("price", vec![Some(1.234), Some(2.356)])
        .add_numeric("code", vec![Some(1.0), Some(2.0)])
        .build()
        .unwrap();
    assert_eq!(ds.numeric("price").unwrap(), &[Some(1.23), Some(2.36)]);
    let (codes, vocab) = ds.categorical("code").unwrap();
    assert_eq!(vocab.len(), 2);
    assert_eq!(codes, &[Some(0), Some(1)]);
}

#[test]
fn builder_drop_rows_removes_any_na() {
    let ds = DatasetBuilder::new()
        .na_policy(NaPolicy::DropRows)
        .add_numeric("a", vec![Some(1.0), None, Some(3.0), Some(4.0)])
        .add_categorical(
            "b",
            vec![
                Some("x".into()),
                Some("y".into()),
                None,
                Some("x".into()),
            ],
        )
        .build()
        .unwrap();
    assert_eq!(ds.n_rows(), 2);
    assert_eq!(ds.numeric("a").unwrap(), &[Some(1.0), Some(4.0)]);
}

#[test]
fn builder_imputes_mean_and_mode() {
    let ds = DatasetBuilder::new()
        .na_policy(NaPolicy::Impute)
        .add_numeric("a", vec![Some(1.0), None, Some(3.0)])
        .add_categorical("b", vec![Some("x".into()), Some("x".into()), None])
        .build()
        .unwrap();
    assert_eq!(ds.numeric("a").unwrap(), &[Some(1.0), Some(2.0), Some(3.0)]);
    let (codes, _) = ds.categorical("b").unwrap();
    assert_eq!(codes, &[Some(0), Some(0), Some(0)]);
}

#[test]
fn builder_intercept_and_drop_all() {
    let ds = DatasetBuilder::new()
        .add_intercept(true)
        .add_numeric("a", vec![Some(1.0)])
        .build()
        .unwrap();
    assert_eq!(ds.column("intercept").unwrap().len(), 1);

    let err = DatasetBuilder::new()
        .na_policy(NaPolicy::DropRows)
        .add_numeric("a", vec![None, None])
        .build();
    assert!(err.is_err());
}

#[test]
fn grouped_scaler_zscore_and_inverse() {
    let keys = vec![Some("g".into()), Some("g".into()), Some("g".into())];
    let vals = vec![Some(1.0), Some(3.0), Some(5.0)]; // mean 3, population std sqrt(8/3)
    let mut gs = GroupedScaler::new();
    gs.fit(&keys, &vals, GroupedStrategy::Zscore).unwrap();
    let out = gs.transform(&keys, &vals).unwrap();
    assert!((out[1].unwrap()).abs() < 1e-12);
    assert!((out[0].unwrap() + out[2].unwrap()).abs() < 1e-12);
    let back = gs.inverse_transform(&keys, &out).unwrap();
    for (b, v) in back.iter().zip(&vals) {
        assert!((b.unwrap() - v.unwrap()).abs() < 1e-9);
    }
}

#[test]
fn grouped_scaler_ratio_unseen_and_degenerate() {
    let keys = vec![
        Some("g".into()),
        Some("g".into()),
        Some("h".into()),
        None,
    ];
    let vals = vec![Some(2.0), Some(4.0), Some(10.0), Some(99.0)];
    let mut gs = GroupedScaler::new();
    gs.fit(&keys, &vals, GroupedStrategy::MeanRatio).unwrap();
    let out = gs.transform(&keys, &vals).unwrap();
    assert_eq!(out[0], Some(2.0 / 3.0));
    assert_eq!(out[1], Some(4.0 / 3.0));
    assert_eq!(out[2], Some(1.0));
    assert_eq!(out[3], None); // missing key

    // unseen group -> None
    let out = gs
        .transform(&[Some("zz".into())], &[Some(1.0)])
        .unwrap();
    assert_eq!(out, vec![None]);

    // zero scale (single-value group) -> None instead of inf
    let k1 = vec![Some("s".into())];
    let v1 = vec![Some(5.0)];
    let mut gs2 = GroupedScaler::new();
    gs2.fit(&k1, &v1, GroupedStrategy::Zscore).unwrap();
    assert_eq!(gs2.transform(&k1, &v1).unwrap(), vec![None]);
}

#[test]
fn grouped_strategy_names_parse() {
    for name in [
        "zscore", "min-max", "median-iqr", "mean-iqr", "median-qd", "median-std",
        "median-range", "mean-range", "min-qd", "min-iqr", "min", "max", "mean",
        "median", "iqr", "qd", "std", "range",
    ] {
        assert!(GroupedStrategy::parse(name).is_ok(), "{name}");
    }
    assert!(GroupedStrategy::parse("nope").is_err());
}
