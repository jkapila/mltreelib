use mltreelib_core::cluster::UnsupervisedProximityForest;

#[test]
fn test_unsupervised_random_forest_two_blobs() {
    let mut data: Vec<Vec<f64>> = Vec::new();
    // Blob 1 around (0, 0)
    for i in 0..20 {
        let offset = (i as f64) * 0.05;
        data.push(vec![0.0 + offset, 0.0 - offset]);
    }
    // Blob 2 around (20, 20)
    for i in 0..20 {
        let offset = (i as f64) * 0.05;
        data.push(vec![20.0 + offset, 20.0 - offset]);
    }
    // Outlier point
    data.push(vec![100.0, 100.0]);

    let mut urf = UnsupervisedProximityForest::new(30, 6, 2, 42);
    urf.fit(&data, 2);

    assert_eq!(urf.proximity_matrix.len(), data.len());
    assert_eq!(urf.proximity_matrix[0].len(), data.len());

    // Diagonal elements of proximity matrix must be 1.0 (a point shares leaf with itself)
    for i in 0..data.len() {
        assert!(
            (urf.proximity_matrix[i][i] - 1.0).abs() < 1e-6,
            "Self-proximity should be 1.0"
        );
    }

    // Proximity between points within Blob 1 should be much higher than between Blob 1 and Blob 2
    let within_blob1 = urf.proximity_matrix[0][1];
    let across_blobs = urf.proximity_matrix[0][25];
    assert!(
        within_blob1 >= across_blobs,
        "Within blob proximity {} should be >= cross blob proximity {}",
        within_blob1,
        across_blobs
    );

    // Cluster labels should have length N and target 2 clusters
    assert_eq!(urf.cluster_labels.len(), data.len());
    let mut unique_labels = urf.cluster_labels.clone();
    unique_labels.sort();
    unique_labels.dedup();
    assert_eq!(unique_labels.len(), 2);

    // Points in Blob 1 should mostly share the same cluster label
    let label_blob1 = urf.cluster_labels[0];
    let mut count_same = 0;
    for &l in &urf.cluster_labels[0..20] {
        if l == label_blob1 {
            count_same += 1;
        }
    }
    assert!(count_same >= 16, "Most points in blob 1 should share cluster label");

    // Outlier score test: outlier at (100, 100) should have low total proximity / high outlier score
    let outlier_idx = data.len() - 1;
    let outlier_score = urf.outlier_scores[outlier_idx];
    let normal_score = urf.outlier_scores[0];
    assert!(
        outlier_score >= normal_score,
        "Outlier score {} should be >= normal score {}",
        outlier_score,
        normal_score
    );

    // Pairwise distance matrix check
    let dist_matrix = urf.pairwise_distances();
    assert_eq!(dist_matrix.len(), data.len());
    assert!((dist_matrix[0][0] - 0.0).abs() < 1e-6);
}
