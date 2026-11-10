use mltreelib_core::cluster::PredictiveClusteringTree;
use mltreelib_core::split::ColType;

#[test]
fn test_pct_multi_target_variance_reduction() {
    let mut pct = PredictiveClusteringTree::new(5, 4, 2);

    // 2 clusters in descriptive X (x0 <= 5.0 vs x0 > 5.0)
    // Cluster A: target_0 ~ 1.0, target_1 ~ 10.0
    // Cluster B: target_0 ~ 100.0, target_1 ~ 500.0
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..30 {
        x.push(vec![i as f64 * 0.1]); // 0.0 .. 3.0
        y.push(vec![1.0 + (i as f64 * 0.01), 10.0]);
    }
    for i in 0..30 {
        x.push(vec![10.0 + (i as f64 * 0.1)]); // 10.0 .. 13.0
        y.push(vec![100.0 + (i as f64 * 0.01), 500.0]);
    }

    let col_names = vec!["feat_x".to_string()];
    let dtypes = vec![ColType::Float];
    let target_names = vec!["t0".to_string(), "t1".to_string()];

    pct.fit(&x, Some(&y), Some(&col_names), Some(&dtypes), Some(&target_names));

    assert!(pct.n_clusters >= 2, "PCT must find at least 2 distinct clusters");

    let cluster_a = pct.predict_cluster(&[1.5]);
    let cluster_b = pct.predict_cluster(&[11.5]);
    assert_ne!(cluster_a, cluster_b, "Different regions must map to different clusters");

    let proto_a = pct.predict_prototype(&[1.5]);
    let proto_b = pct.predict_prototype(&[11.5]);
    assert!(proto_a[0] < 5.0 && proto_a[1] < 20.0);
    assert!(proto_b[0] > 90.0 && proto_b[1] > 400.0);

    let rules = pct.cluster_rules();
    assert_eq!(rules.len(), pct.n_clusters);
    for (cid, rule, proto, size) in rules {
        assert!(size >= 2);
        assert_eq!(proto.len(), 2);
        println!("Cluster {}: rule '{}', size {}", cid, rule, size);
    }
}

#[test]
fn test_pct_unsupervised_mode() {
    let mut pct = PredictiveClusteringTree::new(4, 6, 2);

    // Unsupervised data (no y supplied)
    let mut x = Vec::new();
    for _ in 0..25 {
        x.push(vec![1.0, 2.0]);
    }
    for _ in 0..25 {
        x.push(vec![50.0, 60.0]);
    }

    pct.fit(&x, None, None, None, None);
    assert!(pct.n_clusters >= 2);

    let c1 = pct.predict_cluster(&[1.0, 2.0]);
    let c2 = pct.predict_cluster(&[50.0, 60.0]);
    assert_ne!(c1, c2, "Unsupervised clusters must separate dense groups");
}
