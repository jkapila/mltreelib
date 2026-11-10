use mltreelib_core::trie_knn::{KnnWeights, TrieKnn, TrieKnnTask};

#[test]
fn test_trie_knn_classification_mixed_types() {
    // 2 categorical columns (col 0, col 1), 2 numeric columns (col 2, col 3)
    let cat_cols = vec![0, 1];
    let num_cols = vec![2, 3];
    let mut model = TrieKnn::new(TrieKnnTask::Classification, 3, KnnWeights::Distance, cat_cols, num_cols);

    // Group 0: (catA, cat1) -> class 0
    // Group 1: (catB, cat2) -> class 1
    let mut data = Vec::new();
    let mut targets = Vec::new();

    for i in 0..20 {
        data.push(vec![0.0, 1.0, i as f64 * 0.1, 1.0]);
        targets.push(0.0);
    }
    for i in 0..20 {
        data.push(vec![1.0, 2.0, i as f64 * 0.1, 10.0]);
        targets.push(1.0);
    }

    model.fit(&data, &targets);

    // Query matching Group 0
    let pred_0 = model.predict(&[0.0, 1.0, 0.5, 1.0]);
    assert_eq!(pred_0, 0.0);

    // Query matching Group 1
    let pred_1 = model.predict(&[1.0, 2.0, 0.5, 10.0]);
    assert_eq!(pred_1, 1.0);

    // Query with unseen categorical sub-branch (catA, cat99) -> backs off to parent catA
    let pred_fallback = model.predict(&[0.0, 99.0, 0.5, 1.0]);
    assert_eq!(pred_fallback, 0.0, "Should back off to parent prefix node catA");

    // Check predict_proba
    let probs = model.predict_proba(&[0.0, 1.0, 0.5, 1.0]);
    assert_eq!(probs.len(), 2);
    assert!(probs[0] > 0.9);
}

#[test]
fn test_trie_knn_regression() {
    let cat_cols = vec![0];
    let num_cols = vec![1];
    let mut model = TrieKnn::new(TrieKnnTask::Regression, 2, KnnWeights::Uniform, cat_cols, num_cols);

    let data = vec![
        vec![0.0, 10.0],
        vec![0.0, 20.0],
        vec![1.0, 100.0],
        vec![1.0, 110.0],
    ];
    let targets = vec![1.0, 2.0, 10.0, 11.0];

    model.fit(&data, &targets);

    let pred_cat0 = model.predict(&[0.0, 15.0]);
    assert!((pred_cat0 - 1.5).abs() < 1e-4);

    let pred_cat1 = model.predict(&[1.0, 105.0]);
    assert!((pred_cat1 - 10.5).abs() < 1e-4);
}
