use mltreelib_core::component::{FastIcaModel, NmfModel, PcaModel, TruncatedSvdModel, WhiteningKind, WhiteningModel};

#[test]
fn test_pca_variance_and_reconstruction() {
    // Generate synthetic 2D line with noise: y = 2x + noise
    let mut data = Vec::new();
    for i in 0..100 {
        let x = i as f64 * 0.1;
        let y = 2.0 * x + (i % 5) as f64 * 0.05;
        let z = 0.5 * x - 0.2 * y;
        data.push(vec![x, y, z]);
    }

    let pca = PcaModel::fit(&data, Some(2), false).expect("fit PCA");
    assert_eq!(pca.n_components, 2);
    assert_eq!(pca.components.len(), 2);
    assert_eq!(pca.components[0].len(), 3);

    // First component should explain overwhelming majority of variance
    assert!(pca.explained_variance_ratio[0] > 0.85);
    assert!(pca.cumulative_explained_variance[1] > 0.99);

    // Transform and inverse transform
    let scores = pca.transform(&data).expect("transform");
    assert_eq!(scores.len(), 100);
    assert_eq!(scores[0].len(), 2);

    let reconstructed = pca.inverse_transform(&scores).expect("inverse_transform");
    assert_eq!(reconstructed.len(), 100);

    // Reconstruction error on 2 components should be tiny
    let errors = pca.reconstruction_errors(&data).expect("errors");
    assert_eq!(errors.len(), 100);
    let mean_err: f64 = errors.iter().sum::<f64>() / 100.0;
    assert!(mean_err < 0.01, "Mean reconstruction error should be very small, got {}", mean_err);

    // Report generation
    let report = pca.report();
    assert_eq!(report.learner.family, "Component Analysis");
    let txt = report.to_text();
    assert!(txt.contains("Principal Component Analysis"));
    let md = report.to_markdown();
    assert!(md.contains("## 1. Objective"));
}

#[test]
fn test_pca_reconstruction_anomaly_detection() {
    // 3D data living primarily in 2D plane
    let mut train = Vec::new();
    for i in 0..80 {
        let x1 = (i as f64) * 0.2;
        let x2 = (i as f64) * -0.5;
        let x3 = 0.01 * ((i % 3) as f64);
        train.push(vec![x1, x2, x3]);
    }

    let pca = PcaModel::fit(&train, Some(2), false).expect("fit");

    // Inlier test row: follows the subspace
    let inlier = vec![vec![10.0, -25.0, 0.01]];
    let inlier_err = pca.reconstruction_errors(&inlier).expect("inlier error")[0];

    // Outlier test row: breaks the subspace (orthogonal spike)
    let outlier = vec![vec![10.0, -25.0, 50.0]];
    let outlier_err = pca.reconstruction_errors(&outlier).expect("outlier error")[0];

    assert!(
        outlier_err > inlier_err * 100.0,
        "Outlier error ({}) should vastly exceed inlier error ({})",
        outlier_err,
        inlier_err
    );
}

#[test]
fn test_truncated_svd_uncentered() {
    let mut data = Vec::new();
    for i in 0..50 {
        let x = (i + 1) as f64;
        data.push(vec![x, 2.0 * x, 3.0 * x]);
    }

    let svd = TruncatedSvdModel::fit(&data, Some(1)).expect("fit SVD");
    assert_eq!(svd.n_components, 1);
    let scores = svd.transform(&data).expect("transform");
    let _recon = svd.inverse_transform(&scores).expect("inverse");

    // Because rank is 1, reconstruction of [x, 2x, 3x] with 1 component should be exact
    let errs = svd.reconstruction_errors(&data).expect("errors");
    let max_err = errs.iter().cloned().fold(0.0, f64::max);
    assert!(max_err < 1e-6, "Max rank-1 reconstruction error: {}", max_err);
}

#[test]
fn test_whitening_covariance_identity() {
    let mut data = Vec::new();
    for i in 0..100 {
        let x = i as f64 * 0.1;
        let y = 1.5 * x + ((i % 7) as f64) * 0.2;
        data.push(vec![x, y]);
    }

    let zca = WhiteningModel::fit(&data, WhiteningKind::Zca, None).expect("fit ZCA");
    let whitened = zca.transform(&data).expect("transform");

    // Verify covariance of whitened data has approximately unit variance
    let n = whitened.len() as f64;
    let var0: f64 = whitened.iter().map(|r| r[0].powi(2)).sum::<f64>() / (n - 1.0);
    let var1: f64 = whitened.iter().map(|r| r[1].powi(2)).sum::<f64>() / (n - 1.0);
    assert!((var0 - 1.0).abs() < 0.15, "ZCA var0: {}", var0);
    assert!((var1 - 1.0).abs() < 0.15, "ZCA var1: {}", var1);

    // Invert back
    let restored = zca.inverse_transform(&whitened).expect("invert");
    for i in 0..data.len() {
        for j in 0..2 {
            assert!((data[i][j] - restored[i][j]).abs() < 1e-4);
        }
    }
}

#[test]
fn test_fastica_source_separation() {
    // Create two mixed signals: uniform random and sawtooth
    let n = 200;
    let mut s1 = Vec::with_capacity(n);
    let mut s2 = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * 0.05;
        s1.push((t.sin()).signum()); // square wave
        s2.push((i % 10) as f64 / 5.0 - 1.0); // sawtooth wave
    }

    // Mix them: X = S * A^T
    let mut mixed = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = 0.8 * s1[i] + 0.6 * s2[i];
        let x2 = 0.5 * s1[i] - 0.7 * s2[i];
        mixed.push(vec![x1, x2]);
    }

    let ica = FastIcaModel::fit(&mixed, Some(2), Some(150), Some(1e-4), Some(42)).expect("fit ICA");
    assert_eq!(ica.n_components, 2);

    let sources = ica.transform(&mixed).expect("transform");
    assert_eq!(sources.len(), n);
    assert_eq!(sources[0].len(), 2);

    // Reconstruct
    let _recon = ica.reconstruct(&mixed).expect("reconstruct");
    let errs = ica.reconstruction_errors(&mixed).expect("errors");
    let mean_err: f64 = errs.iter().sum::<f64>() / (n as f64);
    assert!(mean_err < 0.1, "FastICA reconstruction error: {}", mean_err);
}

#[test]
fn test_nmf_multiplicative_updates() {
    // Synthetic non-negative rank-2 matrix
    let mut data = Vec::new();
    for i in 0..40 {
        let a = (i % 5) as f64 + 1.0;
        let b = (i % 3) as f64 + 2.0;
        // combination of basis [1, 2, 0] and [0, 1, 3]
        let r = vec![a * 1.0, a * 2.0 + b * 1.0, b * 3.0];
        data.push(r);
    }

    let nmf = NmfModel::fit(&data, Some(2), Some(100), Some(1e-4), Some(42)).expect("fit NMF");
    assert_eq!(nmf.n_components, 2);
    assert_eq!(nmf.components.len(), 2);
    assert_eq!(nmf.components[0].len(), 3);

    // Verify all components are non-negative
    for row in &nmf.components {
        for &val in row {
            assert!(val >= 0.0);
        }
    }

    // Transform and reconstruct
    let w = nmf.transform(&data).expect("transform");
    assert_eq!(w.len(), 40);
    assert_eq!(w[0].len(), 2);

    let errs = nmf.reconstruction_errors(&data).expect("errors");
    let mean_err: f64 = errs.iter().sum::<f64>() / 40.0;
    assert!(mean_err < 1.0, "NMF reconstruction error should be small: {}", mean_err);

    // Report
    let rep = nmf.report();
    assert_eq!(rep.learner.kind, "NMF");
    assert!(rep.to_text().contains("Non-negative Matrix Factorization"));
}
