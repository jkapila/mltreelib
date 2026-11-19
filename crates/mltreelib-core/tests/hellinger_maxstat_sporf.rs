use mltreelib_core::criteria::{GainFn, KernelKind};
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

#[test]
fn test_hellinger_extreme_imbalance_recovery() {
    let n = 300;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    // 10 minority positives in first 15 samples, remaining 285 are negative
    for i in 0..n {
        let feat = i as f64;
        let label = if i < 15 { 1.0 } else { 0.0 };
        x.push(vec![Some(feat)]);
        y.push(Some(label));
    }

    let params = ModelTreeParams {
        max_depth: 1,
        min_rows: 5,
        gain: GainFn::Hellinger,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit hellinger");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected root split");
    let root = &splits[0];
    assert_eq!(root.col, 0);
    assert!(
        root.threshold >= 10.0 && root.threshold <= 20.0,
        "Hellinger split threshold must isolate minority cluster, got {}",
        root.threshold
    );
}

#[test]
fn test_maxstat_recovers_rank_cutpoint() {
    let n = 200;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let v = -2.0 + 4.0 * (i as f64) / (n as f64);
        // Monotone cubic transformation with sharp step at 0
        let target = if v < 0.0 { -10.0 + v * v } else { 10.0 + v * v };
        x.push(vec![Some(v)]);
        y.push(Some(target));
    }

    let params = ModelTreeParams {
        max_depth: 1,
        min_rows: 10,
        gain: GainFn::MaxStat,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit maxstat");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected root split");
    let root = &splits[0];
    assert_eq!(root.col, 0);
    assert!(
        root.threshold.abs() < 0.2,
        "MaxStat cutpoint should be near 0, got {}",
        root.threshold
    );
}

#[test]
fn test_sporf_recovers_diagonal_hyperplane() {
    let n = 300;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut lcg: u64 = 12345;
    let mut rand_uniform = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    for _ in 0..n {
        let x0 = rand_uniform();
        let x1 = rand_uniform();
        let x2 = rand_uniform(); // noise
        let target = if x0 + x1 > 0.0 { 50.0 } else { -50.0 };
        x.push(vec![Some(x0), Some(x1), Some(x2)]);
        y.push(Some(target));
    }

    let params = ModelTreeParams {
        max_depth: 1,
        min_rows: 10,
        sporf_d: 2,
        sporf_projections: 40,
        seed: 42,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit sporf");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected root split");
    let root = &splits[0];
    assert!(
        root.oblique.is_some(),
        "SPORF must select an oblique projection over univariate axes"
    );

    let ob_parts = tree.oblique_parts();
    let proj = &ob_parts[root.oblique.unwrap()];
    assert_eq!(proj.cols.len(), 2, "must combine exactly 2 features");

    // Predictions should be highly correlated with planted diagonal target
    let preds = tree.predict(&x);
    let mut correct = 0;
    for (i, &p) in preds.iter().enumerate() {
        let true_y = y[i].unwrap();
        if (p > 0.0 && true_y > 0.0) || (p < 0.0 && true_y < 0.0) {
            correct += 1;
        }
    }
    let acc = (correct as f64) / (n as f64);
    assert!(
        acc > 0.85,
        "SPORF oblique cut should achieve >85% accuracy on diagonal boundary, got {:.2}",
        acc
    );
}

#[test]
fn test_rff_mmd_large_sample_scaling() {
    let n = 1200; // >= 1000 triggers RFF approximation
    let d = 5;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y_vecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_dummy: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut lcg: u64 = 999;
    let mut rand_uniform = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    for _ in 0..n {
        let x0 = rand_uniform();
        let x1 = rand_uniform();
        x.push(vec![Some(x0), Some(x1)]);

        // 5D cluster vectors
        let center = if x0 < 0.0 { -3.0 } else { 3.0 };
        let vec: Vec<f64> = (0..d).map(|_| center + rand_uniform() * 0.2).collect();
        y_vecs.push(vec);
        y_dummy.push(Some(center));
    }

    let mut params = ModelTreeParams {
        max_depth: 1,
        min_rows: 50,
        gain: GainFn::Mmd {
            kernel: KernelKind::Rbf,
            bandwidth: 1.0,
        },
        ..Default::default()
    };
    params.target_vectors = Some(y_vecs);

    let tree = ModelTree::fit(&x, &y_dummy, params).expect("fit rff mmd");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected root split");
    let root = &splits[0];
    assert_eq!(root.col, 0, "RFF MMD must split on signal column 0");
    assert!(
        root.threshold.abs() < 0.3,
        "RFF MMD cutpoint should be near 0, got {}",
        root.threshold
    );
}
