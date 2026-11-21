use mltreelib_core::criteria::GainFn;
use mltreelib_core::leaf::LeafKind;
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};
use mltreelib_core::prune::PruneCost;
use mltreelib_core::split::SplitEval;

#[test]
fn numeric_split_counts_a_class_observed_only_with_missing_features() {
    let x = [
        vec![Some(0.0)],
        vec![Some(1.0)],
        vec![Some(2.0)],
        vec![None],
    ];
    let weights = [1.0, 1.0, 2.0, 4.0];
    for (mode, labels) in [
        (SplitEval::Fast, [0.0, 1.0, 0.0, 2.0]),
        (SplitEval::Fast, [0.0, -3.5, 0.0, 500.0]),
        (SplitEval::Histogram, [0.0, 1.0, 0.0, 2.0]),
    ] {
        let params = ModelTreeParams {
            max_depth: 1,
            min_rows: 1,
            min_gain: 0.0,
            split_precision: None,
            leaf: LeafKind::Mean,
            prune: PruneCost::None,
            gain: GainFn::Gini,
            split_eval: mode,
            ..Default::default()
        };
        let tree = ModelTree::fit_weighted(&x, &labels.map(Some), &weights, params).unwrap();
        let split = tree.root_split().unwrap();
        assert_eq!(split.threshold, 0.5);
        assert!(split.missing_left);
        assert!((split.gain - 109.0 / 480.0).abs() < 1e-12);
    }
}

#[test]
fn histogram_mode_limits_candidates_and_preserves_aligned_class_boundary() {
    let x: Vec<_> = (0..1024).map(|i| vec![Some(i as f64)]).collect();
    let y: Vec<_> = (0..1024).map(|i| Some((i / 256) as f64)).collect();
    let params = ModelTreeParams {
        max_depth: 2,
        min_rows: 1,
        split_precision: None,
        leaf: LeafKind::Mean,
        prune: PruneCost::None,
        gain: GainFn::Entropy,
        split_eval: SplitEval::Histogram,
        ..Default::default()
    };
    let tree = ModelTree::fit(&x, &y, params).unwrap();
    assert_eq!(tree.root_split().unwrap().threshold, 511.5);
    assert_eq!(tree.predict(&x), y.into_iter().flatten().collect::<Vec<_>>());
    assert!(tree.profiles().iter().all(|node| node.candidates <= 255));
}
