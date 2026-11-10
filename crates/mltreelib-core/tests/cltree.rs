use mltreelib_core::cluster::ClTree;

#[test]
fn test_cltree_separates_clusters_from_noise() {
    let mut clt = ClTree::new(5, 5, 0.55);

    // Two dense clusters in 2D surrounded by sparse space:
    // Cluster 1: [0.0..1.0, 0.0..1.0]
    // Cluster 2: [10.0..11.0, 10.0..11.0]
    let mut data = Vec::new();
    for i in 0..40 {
        data.push(vec![0.0 + (i as f64 * 0.02), 0.5]);
    }
    for i in 0..40 {
        data.push(vec![10.0 + (i as f64 * 0.02), 10.5]);
    }

    clt.fit(&data, None);

    assert!(clt.n_clusters >= 2, "CLTree must isolate at least 2 dense clusters");

    // Inside Cluster 1
    let c1 = clt.predict_cluster(&[0.5, 0.5]);
    assert!(c1.is_some(), "Dense cluster 1 point must be assigned a cluster ID");

    // Inside Cluster 2
    let c2 = clt.predict_cluster(&[10.5, 10.5]);
    assert!(c2.is_some(), "Dense cluster 2 point must be assigned a cluster ID");
    assert_ne!(c1, c2, "Different dense regions must have different cluster IDs");

    // In sparse empty space between clusters
    let noise = clt.predict_cluster(&[5.0, 5.0]);
    assert!(noise.is_none(), "Point in empty space must be classified as noise/outlier (None)");

    let bboxes = clt.cluster_bounding_boxes();
    assert_eq!(bboxes.len(), clt.n_clusters);
    for (cid, bounds, size, density) in bboxes {
        assert!(size >= 5);
        assert!(density >= 0.55);
        println!("Cluster {}: size={}, density={:.2}, bounds={:?}", cid, size, density, bounds);
    }
}
