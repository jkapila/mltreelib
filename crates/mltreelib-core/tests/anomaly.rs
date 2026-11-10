use mltreelib_core::anomaly::{
    ActiveAnomalyTree, ContextAnomalyTree, DensityEstimationTree, ExtendedIsolationTree,
    HalfSpaceTree, RandomCutTree,
};
use mltreelib_core::split::ColType;

#[test]
fn test_det_density_and_outlier_separation() {
    let mut det = DensityEstimationTree::new().with_max_depth(5).with_min_leaf_samples(2);
    // Cluster near 0.0, outlier near 100.0
    let mut data = Vec::new();
    for i in 0..50 {
        data.push(vec![i as f64 * 0.05, 1.0]);
    }
    let col_names = vec!["x1".to_string(), "x2".to_string()];
    let dtypes = vec![ColType::Float, ColType::Float];
    det.fit(&data, &col_names, &dtypes);

    let dense_density = det.predict_density(&[0.5, 1.0]);
    let outlier_density = det.predict_density(&[100.0, 1.0]);
    assert!(dense_density > outlier_density, "Dense area must have higher density than outlier");
    assert!(det.predict_anomaly_score(&[100.0, 1.0]) > det.predict_anomaly_score(&[0.5, 1.0]));
}

#[test]
fn test_eif_tree_categorical_and_numeric() {
    let mut eif = ExtendedIsolationTree::new(42);
    let mut data = Vec::new();
    // Dense cluster of 60 points
    for i in 0..60 {
        data.push(vec![i as f64 * 0.05, (i % 2) as f64]);
    }
    // Single distant outlier
    data.push(vec![100.0, 0.0]);

    let col_names = vec!["num".to_string(), "cat".to_string()];
    let dtypes = vec![ColType::Float, ColType::Categorical];
    eif.fit(&data, &col_names, &dtypes);

    let cluster_path = eif.path_length(&[1.5, 1.0]);
    let outlier_path = eif.path_length(&[100.0, 0.0]);
    assert!(outlier_path < cluster_path, "Outlier isolates faster than cluster points");
    assert!(eif.anomaly_score(&[100.0, 0.0]) > eif.anomaly_score(&[1.5, 1.0]));
}

#[test]
fn test_half_space_tree_mass_scoring() {
    let mut hst = HalfSpaceTree::new(6, 42);
    let mut data = Vec::new();
    for i in 0..50 {
        data.push(vec![i as f64 * 0.1, 0.5]);
    }
    hst.fit(&data);

    let score_in_cluster = hst.anomaly_score(&[1.0, 0.5]);
    let score_outlier = hst.anomaly_score(&[50.0, 50.0]);
    assert!(score_outlier > score_in_cluster, "HST score must be higher for low-mass region");
}

#[test]
fn test_random_cut_tree_codisp() {
    let mut rct = RandomCutTree::new(42);
    let mut data = Vec::new();
    for i in 0..40 {
        data.push(vec![0.0 + (i as f64 * 0.01), 0.0]);
    }
    rct.fit(&data);

    let normal_codisp = rct.codisp(&[0.05, 0.0]);
    let outlier_codisp = rct.codisp(&[50.0, 50.0]);
    assert!(outlier_codisp > normal_codisp, "CoDisp must be significantly higher for outlier");
}

#[test]
fn test_active_anomaly_tree_feedback() {
    let mut aad = ActiveAnomalyTree::new(5, 42);
    let mut data = Vec::new();
    for i in 0..40 {
        data.push(vec![i as f64 * 0.1]);
    }
    aad.fit(&data);

    let probe = vec![2.0];
    let score_before = aad.score(&probe);
    // User confirms probe is an anomaly (true feedback)
    aad.update_feedback(&probe, true, 0.5);
    let score_after = aad.score(&probe);
    assert!(score_after > score_before, "Positive anomaly feedback must increase score");
}

#[test]
fn test_context_anomaly_tree_sequence_surprise() {
    let mut cat = ContextAnomalyTree::new(3, 100);
    // Frequent pattern: [1, 2] -> 3
    for _ in 0..20 {
        cat.insert(&[1, 2], 3);
    }
    // Rare transition: [1, 2] -> 99
    let normal_surprise = cat.surprise_score(&[1, 2], 3);
    let rare_surprise = cat.surprise_score(&[1, 2], 99);
    assert!(rare_surprise > normal_surprise, "Rare transition must have higher surprise in bits");
}
