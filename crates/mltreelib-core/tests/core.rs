use mltreelib_core::frame::{Column, Dataset};
use mltreelib_core::generator::{generate, GeneratorParams, TailBehavior, Task};
use mltreelib_core::index::IndexMapper;
use mltreelib_core::schema::{apply_n_category, NCategoryRule};
use mltreelib_core::transform::{summarize, GroupedTransform, SummaryKind};

#[test]
fn dataset_validates_lengths_and_names() {
    let ok = Dataset::new(vec![
        Column::numeric("a", vec![Some(1.0), Some(2.0)]),
        Column::categorical("b", vec![Some("x".into()), None]),
    ]);
    assert!(ok.is_ok());

    let bad = Dataset::new(vec![
        Column::numeric("a", vec![Some(1.0)]),
        Column::numeric("b", vec![Some(1.0), Some(2.0)]),
    ]);
    assert!(bad.is_err());

    let dup = Dataset::new(vec![
        Column::numeric("a", vec![Some(1.0)]),
        Column::numeric("a", vec![Some(2.0)]),
    ]);
    assert!(dup.is_err());
}

#[test]
fn generator_is_deterministic_per_seed() {
    let p = GeneratorParams {
        n_rows: 500,
        seed: 7,
        ..Default::default()
    };
    let a = generate(&p).unwrap();
    let b = generate(&p).unwrap();
    assert_eq!(a.target, b.target);
    assert_eq!(a.dataset.n_rows(), 500);

    let c = generate(&GeneratorParams {
        seed: 8,
        ..p.clone()
    })
    .unwrap();
    assert_ne!(a.target, c.target);
}

#[test]
fn generator_validation_rejects_bad_params() {
    assert!(generate(&GeneratorParams {
        n_rows: 0,
        ..Default::default()
    })
    .is_err());
    assert!(generate(&GeneratorParams {
        cardinalities: vec![1],
        ..Default::default()
    })
    .is_err());
    assert!(generate(&GeneratorParams {
        missingness: 1.0,
        ..Default::default()
    })
    .is_err());
    assert!(generate(&GeneratorParams {
        task: Task::Classification { n_classes: 1 },
        ..Default::default()
    })
    .is_err());
}

#[test]
fn generator_variance_shares_follow_knobs() {
    let g = generate(&GeneratorParams {
        n_rows: 20_000,
        n_numeric: 0,
        cardinalities: vec![8],
        hierarchy_levels: 1,
        branching: vec![2],
        group_effect_strength: 1.0,
        hierarchy_effect_strength: 0.0,
        interaction_strength: 0.0,
        noise: 0.05,
        missingness: 0.0,
        seed: 5,
        ..Default::default()
    })
    .unwrap();
    assert!(
        g.truth.variance.categorical > 0.8,
        "expected categorical-dominated variance, got {:?}",
        g.truth.variance
    );
    assert!(g.truth.variance.noise < 0.05);
}

#[test]
fn generator_bathtub_makes_tails_noisier() {
    let g = generate(&GeneratorParams {
        n_rows: 20_000,
        tail_behavior: TailBehavior::Bathtub,
        seed: 11,
        ..Default::default()
    })
    .unwrap();
    // residual spread in the outer target deciles must exceed the middle
    let mut pairs: Vec<(f64, f64)> = g
        .target
        .iter()
        .zip(g.dataset.numeric("num_0").unwrap())
        .filter_map(|(t, x)| x.map(|x| (*t, x)))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = pairs.len();
    let residual = |slice: &[(f64, f64)]| {
        let mean_x = slice.iter().map(|p| p.1).sum::<f64>() / slice.len() as f64;
        slice.iter().map(|p| (p.1 - mean_x).abs()).sum::<f64>() / slice.len() as f64
    };
    let tails = residual(&pairs[..n / 10]) + residual(&pairs[n - n / 10..]);
    let middle = residual(&pairs[n / 2..n / 2 + n / 10]);
    assert!(tails > middle, "tails {tails} vs middle {middle}");
}

#[test]
fn classification_produces_all_classes() {
    let g = generate(&GeneratorParams {
        n_rows: 3_000,
        task: Task::Classification { n_classes: 3 },
        seed: 2,
        ..Default::default()
    })
    .unwrap();
    let labels = g.labels.expect("classification must produce labels");
    let mut seen = [false; 3];
    for l in labels {
        seen[l as usize] = true;
    }
    assert!(seen.iter().all(|&s| s), "all classes must appear");
}

#[test]
fn missingness_scatters_features_only() {
    let g = generate(&GeneratorParams {
        n_rows: 2_000,
        missingness: 0.3,
        seed: 3,
        ..Default::default()
    })
    .unwrap();
    let vals = g.dataset.numeric("num_0").unwrap();
    let nas = vals.iter().filter(|v| v.is_none()).count();
    assert!(nas > 0, "expected some missing values");
    assert_eq!(g.target.iter().filter(|t| t.is_nan()).count(), 0);
    assert_eq!(g.dataset.n_rows(), 2_000);
}

#[test]
fn n_category_rule_converts_integral_columns() {
    let col = Column::numeric("x", (0..100).map(|i| Some((i % 4) as f64)).collect());
    let ds = Dataset::new(vec![col]).unwrap();
    let out = apply_n_category(&ds, Some(NCategoryRule::Absolute(10)));
    match &out.columns()[0] {
        Column::Categorical { vocab, codes, .. } => {
            assert_eq!(vocab.len(), 4);
            assert_eq!(codes.len(), 100);
        }
        _ => panic!("expected categorical conversion"),
    }

    // above the threshold the column stays numeric
    let out = apply_n_category(&ds, Some(NCategoryRule::Absolute(3)));
    assert!(matches!(out.columns()[0], Column::Numeric { .. }));

    // fraction rule: 4 levels > 2% of 100 rows -> stays numeric
    let out = apply_n_category(&ds, Some(NCategoryRule::Fraction(0.02)));
    assert!(matches!(out.columns()[0], Column::Numeric { .. }));
}

#[test]
fn index_mapper_roundtrips_and_maps_unseen_to_none() {
    let vals = vec![
        Some("a".to_string()),
        Some("b".to_string()),
        None,
        Some("a".to_string()),
        Some("c".to_string()),
    ];
    let mut m = IndexMapper::new();
    m.fit("col", &vals);
    let codes = m.transform("col", &vals).unwrap();
    assert_eq!(codes, vec![Some(0), Some(1), None, Some(0), Some(2)]);
    let back = m.inverse_transform("col", &codes).unwrap();
    assert_eq!(back, vals);
    assert_eq!(m.transform("col", &[Some("z".to_string())]).unwrap(), vec![None]);
    assert_eq!(m.get_all_names("col").unwrap(), vec!["a", "b", "c"]);
    assert!(m.transform("nope", &vals).is_err());
}

#[test]
fn grouped_transform_matches_manual_mean() {
    let keys = vec![
        Some("g1".to_string()),
        Some("g1".to_string()),
        Some("g2".to_string()),
        None,
    ];
    let vals = vec![Some(1.0), Some(3.0), Some(10.0), Some(99.0)];
    let mut gt = GroupedTransform::new();
    gt.fit(&keys, &vals, SummaryKind::Mean).unwrap();
    let out = gt.transform(&keys).unwrap();
    assert_eq!(out, vec![Some(2.0), Some(2.0), Some(10.0), None]);

    gt.fit(&keys, &vals, SummaryKind::Count).unwrap();
    let out = gt.transform(&keys).unwrap();
    assert_eq!(out, vec![Some(2.0), Some(2.0), Some(1.0), None]);

    gt.fit(&keys, &vals, SummaryKind::Sum).unwrap();
    let out = gt.transform(&keys).unwrap();
    assert_eq!(out, vec![Some(4.0), Some(4.0), Some(10.0), None]);
}

#[test]
fn summarize_skips_nas() {
    let s = summarize(&[Some(1.0), None, Some(3.0), Some(2.0)]);
    assert_eq!(s.count, 3);
    assert_eq!(s.mean, 2.0);
    assert_eq!(s.median, 2.0);
    assert_eq!(s.min, 1.0);
    assert_eq!(s.max, 3.0);
    assert!(summarize(&[]).count == 0);
}
