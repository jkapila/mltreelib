use std::collections::BTreeMap;
use std::fs;

use mltreelib_core::{
    criteria::{CustomGain, GainFn},
    family::LeafFamily,
    leaf::LeafKind,
    model_tree::{ModelTree, ModelTreeParams},
    prune::PruneCost,
    split::ColType,
    PersistenceError,
};

fn sample() -> (Vec<Vec<Option<f64>>>, Vec<Option<f64>>) {
    let x = (0..60)
        .map(|i| vec![Some((i % 2) as f64), Some(i as f64)])
        .collect();
    let y = (0..60).map(|i| Some(10.0 * (i % 2) as f64)).collect();
    (x, y)
}

#[test]
fn restores_categorical_splits_digests_and_neighbor_adjustments() {
    let (x, y) = sample();
    let mut tree = ModelTree::fit(
        &x,
        &y,
        ModelTreeParams {
            max_depth: 2,
            min_rows: 5,
            leaf: LeafKind::Mean,
            dtypes: vec![ColType::Categorical, ColType::Float],
            prune: PruneCost::None,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(tree.n_leaves() > 1);
    tree.category_levels = vec![
        Some(BTreeMap::from([
            ("north".into(), 0.0),
            ("south".into(), 1.0),
        ])),
        None,
    ];

    let bytes = tree.to_bytes().unwrap();
    let restored = ModelTree::from_bytes(&bytes).unwrap();
    assert_eq!(restored.category_levels, tree.category_levels);
    let query = [vec![Some(1.0), Some(12.3)], vec![None, None]];
    assert_eq!(restored.predict(&query), tree.predict(&query));
    assert_eq!(restored.quantile(&query, 0.75), tree.quantile(&query, 0.75));
    assert_eq!(
        restored.predict_adjusted(&query, 3),
        tree.predict_adjusted(&query, 3)
    );
    assert_eq!(
        restored.extract_rules(None, None, false, 0.25).to_string(),
        tree.extract_rules(None, None, false, 0.25).to_string(),
    );
    assert_eq!(restored.to_bytes().unwrap(), bytes);
}

#[test]
fn restores_survival_incidence_and_honest_uplift() {
    let (x, _) = sample();
    let time = (1..=60).map(|i| Some(i as f64)).collect::<Vec<_>>();
    let model = ModelTree::fit(
        &x,
        &time,
        ModelTreeParams {
            max_depth: 0,
            leaf_family: LeafFamily::Survival,
            events: Some((0..60).map(|i| (i % 3) as f64).collect()),
            gain: GainFn::LogRank,
            ..Default::default()
        },
    )
    .unwrap();
    let restored = ModelTree::from_json(&model.to_json().unwrap()).unwrap();
    let times = [0.0, 10.0, 20.0];
    let query = &x[..2];
    assert_eq!(
        restored.predict_survival(query, &times).unwrap(),
        model.predict_survival(query, &times).unwrap(),
    );
    assert_eq!(
        restored.predict_cif(query, &times, 2).unwrap(),
        model.predict_cif(query, &times, 2).unwrap(),
    );

    let model = ModelTree::fit(
        &x,
        &time,
        ModelTreeParams {
            max_depth: 0,
            leaf: LeafKind::Mean,
            treatments: Some((0..60).map(|i| (i % 2) as f64).collect()),
            gain: GainFn::Causal,
            honesty: true,
            ..Default::default()
        },
    )
    .unwrap();
    let restored = ModelTree::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(
        restored.predict_uplift(query).unwrap(),
        model.predict_uplift(query).unwrap(),
    );
}

#[test]
fn preserves_nonfinite_sentinels_and_rejects_invalid_snapshots() {
    let (x, y) = sample();
    let mut tree = ModelTree::fit(&x, &y, ModelTreeParams::default()).unwrap();
    tree.params.min_gain = f64::NAN;
    tree.params.tobit_threshold = f64::NEG_INFINITY;
    tree.params.l2 = 0.5000000000000001;
    let json = tree.to_json().unwrap();

    let restored = ModelTree::from_json(&json).unwrap();
    assert!(restored.params.min_gain.is_nan());
    assert_eq!(restored.params.tobit_threshold, f64::NEG_INFINITY);
    assert_eq!(restored.params.l2.to_bits(), tree.params.l2.to_bits());
    assert!(ModelTree::from_json(&json[..json.len() - 1]).is_err());
    assert!(ModelTree::from_json(&(json.clone() + "{}")).is_err());
    assert!(ModelTree::from_bytes(&[255]).is_err());

    let mut snapshot: serde_json::Value = serde_json::from_str(&json).unwrap();
    snapshot["format_version"] = 9.into();
    snapshot["model"] = serde_json::Value::Null;
    assert!(matches!(
        ModelTree::from_json(&snapshot.to_string()),
        Err(PersistenceError::Version(9)),
    ));
    let mut snapshot: serde_json::Value = serde_json::from_str(&json).unwrap();
    snapshot["model"]["root"] = usize::MAX.into();
    assert!(matches!(
        ModelTree::from_json(&snapshot.to_string()),
        Err(PersistenceError::Invalid(_)),
    ));
}

#[test]
fn saves_json_and_preserves_the_previous_file_on_serialization_error() {
    struct ZeroGain;
    impl CustomGain for ZeroGain {
        fn gain(&self, _: &[(f64, f64)], _: &[(f64, f64)]) -> f64 {
            0.0
        }
    }

    let (x, y) = sample();
    let mut tree = ModelTree::fit(&x, &y, ModelTreeParams::default()).unwrap();
    let path = std::env::temp_dir().join(format!(
        "mltreelib-persistence-test-{}.json",
        std::process::id()
    ));
    tree.save(&path).unwrap();
    let expected = tree.predict(&x);
    let contents = fs::read_to_string(&path).unwrap();
    assert_eq!(ModelTree::load(&path).unwrap().predict(&x), expected);

    tree.params.gain = GainFn::Custom(std::sync::Arc::new(ZeroGain));
    assert!(matches!(
        tree.save(&path),
        Err(PersistenceError::Invalid(_))
    ));
    assert_eq!(fs::read_to_string(&path).unwrap(), contents);
    fs::remove_file(path).unwrap();
}
