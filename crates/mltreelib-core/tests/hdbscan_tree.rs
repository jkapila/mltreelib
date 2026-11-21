use mltreelib_core::cluster::HdbscanTree;

#[test]
fn test_hdbscan_tree_two_clusters() {
    // Generate two tight clusters separated in 2D space
    // Cluster 0: near (0, 0)
    // Cluster 1: near (10, 10)
    // Noise point: (100, 100)
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..15 {
        let offset = (i as f64) * 0.05;
        data.push(vec![0.0 + offset, 0.0 + offset]);
    }
    for i in 0..15 {
        let offset = (i as f64) * 0.05;
        data.push(vec![10.0 + offset, 10.0 + offset]);
    }
    // Outlier / noise point
    data.push(vec![100.0, 100.0]);

    let mut model = HdbscanTree::new(5, 5);
    model.fit(&data);

    assert!(model.n_clusters >= 1);
    assert_eq!(model.labels.len(), data.len());
    assert_eq!(model.outlier_scores.len(), data.len());

    // Outlier point at (100, 100) should have high outlier score
    let outlier_idx = data.len() - 1;
    let outlier_score = model.outlier_scores[outlier_idx];
    let cluster_pt_score = model.outlier_scores[0];

    assert!(
        outlier_score >= cluster_pt_score,
        "Outlier score {} should be >= cluster point score {}",
        outlier_score,
        cluster_pt_score
    );

    // Condensed nodes should exist
    assert!(!model.condensed_nodes.is_empty());
    let root = &model.condensed_nodes[0];
    assert_eq!(root.size, data.len());

    // Out-of-sample prediction
    let pred_near_0 = model.predict(&[0.1, 0.1]);
    let pred_near_10 = model.predict(&[10.1, 10.1]);
    assert!(pred_near_0.is_some() || pred_near_10.is_some());
}

#[test]
fn test_hdbscan_tree_empty_and_single() {
    let mut model = HdbscanTree::new(3, 3);
    model.fit(&[]);
    assert_eq!(model.n_clusters, 0);
    assert!(model.labels.is_empty());

    let single = vec![vec![1.0, 2.0]];
    model.fit(&single);
    assert_eq!(model.labels.len(), 1);
}

#[test]
fn test_hdbscan_distance_uses_jointly_observed_coordinates() {
    let model = HdbscanTree::new(2, 1);
    assert!((model.compute_dist(&[0.0, f64::NAN], &[3.0, 4.0]) - 18.0_f64.sqrt()).abs() < 1e-12);
    assert!(model.compute_dist(&[f64::NAN], &[1.0]).is_infinite());
}

#[test]
fn test_hdbscan_snapshot_round_trip_and_rejects_bad_reference() {
    let data: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64, (i % 2) as f64]).collect();
    let mut model = HdbscanTree::new(2, 2);
    model.fit(&data);
    let snapshot = model.to_json().unwrap();
    let restored = HdbscanTree::from_json(&snapshot).unwrap();
    assert_eq!(restored.labels, model.labels);
    let mut broken: serde_json::Value = serde_json::from_str(&snapshot).unwrap();
    broken["model"]["condensed_nodes"][0]["node_id"] = serde_json::json!(999999);
    assert!(HdbscanTree::from_json(&broken.to_string()).is_err());
}
