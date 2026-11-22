use mltreelib_core::component::{
    FactorAnalysisModel, FastIcaModel, NmfModel, NmfSolver, PcaModel, RotationMethod,
    TruncatedSvdModel, WhiteningKind, WhiteningModel,
};

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

    let pca = PcaModel::fit(&data, Some(2), false, None).expect("fit PCA");
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

    let pca = PcaModel::fit(&train, Some(2), false, None).expect("fit");

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

    let nmf = NmfModel::fit(&data, Some(2), Some(NmfSolver::Mu), Some(100), Some(1e-4), Some(42), None, None).expect("fit NMF");
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

#[test]
fn test_pca_rotations() {
    // Generate 4-feature data with two distinct clusters of features:
    // (f1, f2) driven by factor 1; (f3, f4) driven by factor 2
    let mut data = Vec::new();
    for i in 0..120 {
        let f1 = (i as f64) * 0.1;
        let f2 = 2.0 * f1 + (i % 3) as f64 * 0.05;
        let f3 = (i % 7) as f64 * 1.5;
        let f4 = -1.2 * f3 + (i % 5) as f64 * 0.02;
        data.push(vec![f1, f2, f3, f4]);
    }

    // Unrotated PCA
    let pca_raw = PcaModel::fit(&data, Some(2), true, None).expect("fit raw PCA");
    assert!(pca_raw.rotation_matrix.is_none());

    // Varimax rotation
    let pca_varimax = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Varimax)).expect("fit varimax");
    assert!(pca_varimax.rotation_matrix.is_some());
    let rot_m = pca_varimax.rotation_matrix.as_ref().unwrap();
    assert_eq!(rot_m.len(), 2);
    assert_eq!(rot_m[0].len(), 2);

    // Check orthogonality of T: T^T T approx I
    let t00 = rot_m[0][0];
    let t01 = rot_m[0][1];
    let t10 = rot_m[1][0];
    let t11 = rot_m[1][1];
    let dot_col0 = t00 * t00 + t10 * t10;
    let dot_col1 = t01 * t01 + t11 * t11;
    let dot_cross = t00 * t01 + t10 * t11;
    assert!((dot_col0 - 1.0).abs() < 1e-4, "Orthogonal col 0 norm: {}", dot_col0);
    assert!((dot_col1 - 1.0).abs() < 1e-4, "Orthogonal col 1 norm: {}", dot_col1);
    assert!(dot_cross.abs() < 1e-4, "Orthogonal cross product: {}", dot_cross);

    // Quartimax, Equamax, Parsimax, Biquartimax
    let _pca_quarti = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Quartimax)).expect("fit quartimax");
    let _pca_equa = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Equamax)).expect("fit equamax");
    let _pca_parsi = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Parsimax)).expect("fit parsimax");
    let _pca_biquarti = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Biquartimax)).expect("fit biquartimax");

    // Promax oblique rotation
    let pca_promax = PcaModel::fit(&data, Some(2), true, Some(RotationMethod::Promax(4.0))).expect("fit promax");
    assert!(pca_promax.rotation_matrix.is_some());

    // Reconstruction error with rotated loadings should remain comparable
    let raw_errs = pca_raw.reconstruction_errors(&data).expect("raw errors");
    let varimax_errs = pca_varimax.reconstruction_errors(&data).expect("varimax errors");
    let mean_raw: f64 = raw_errs.iter().sum::<f64>() / (data.len() as f64);
    let mean_varimax: f64 = varimax_errs.iter().sum::<f64>() / (data.len() as f64);
    assert!(
        (mean_raw - mean_varimax).abs() < 0.5,
        "Rotated subspace reconstruction error ({}) should match raw ({})",
        mean_varimax,
        mean_raw
    );
}

#[test]
fn test_nmf_als_and_hals_solvers() {
    let mut data = Vec::new();
    for i in 0..50 {
        let a = (i % 4) as f64 + 1.0;
        let b = (i % 6) as f64 + 0.5;
        data.push(vec![2.0 * a, a + 3.0 * b, 4.0 * b]);
    }

    // Projected ALS solver
    let nmf_als = NmfModel::fit(
        &data,
        Some(2),
        Some(NmfSolver::Als),
        Some(100),
        Some(1e-4),
        Some(42),
        Some(0.0),
        Some(0.01),
    ).expect("fit ALS");
    assert_eq!(nmf_als.solver, NmfSolver::Als);
    let als_errs = nmf_als.reconstruction_errors(&data).expect("als errors");
    let mean_als: f64 = als_errs.iter().sum::<f64>() / 50.0;
    assert!(mean_als < 1.0, "ALS error: {}", mean_als);

    // Hierarchical ALS (HALS) solver with L1 and L2 regularization
    let nmf_hals = NmfModel::fit(
        &data,
        Some(2),
        Some(NmfSolver::Hals),
        Some(150),
        Some(1e-4),
        Some(42),
        Some(0.005),
        Some(0.01),
    ).expect("fit HALS");
    assert_eq!(nmf_hals.solver, NmfSolver::Hals);
    let hals_errs = nmf_hals.reconstruction_errors(&data).expect("hals errors");
    let mean_hals: f64 = hals_errs.iter().sum::<f64>() / 50.0;
    assert!(mean_hals < 1.0, "HALS error: {}", mean_hals);
}

#[test]
fn test_factor_analysis_paf_and_schmid_leiman() {
    let mut data = Vec::new();
    // 2 latent factors g1, g2 driving 4 observed indicators with noise
    for i in 0..100 {
        let g1 = (i as f64) * 0.1 - 5.0;
        let g2 = ((i % 11) as f64) * 0.5 - 2.5;
        let x1 = 0.8 * g1 + 0.1 * ((i % 3) as f64);
        let x2 = 0.7 * g1 + 0.15 * ((i % 5) as f64);
        let x3 = 0.9 * g2 + 0.1 * ((i % 4) as f64);
        let x4 = 0.75 * g2 + 0.2 * ((i % 2) as f64);
        data.push(vec![x1, x2, x3, x4]);
    }

    // Fit PAF Factor Analysis with Varimax rotation
    let fa = FactorAnalysisModel::fit(
        &data,
        Some(2),
        Some(100),
        Some(1e-4),
        Some(RotationMethod::Varimax),
        "paf",
    )
    .expect("fit FA PAF");

    assert_eq!(fa.n_components, 2);
    assert_eq!(fa.components.len(), 2);
    assert_eq!(fa.components[0].len(), 4);
    assert_eq!(fa.communalities.len(), 4);
    assert_eq!(fa.uniquenesses.len(), 4);

    // Communalities + Uniquenesses must sum to approximately 1.0 (on standardized correlation scale)
    for j in 0..4 {
        let sum_var = fa.communalities[j] + fa.uniquenesses[j];
        assert!(
            (sum_var - 1.0).abs() < 1e-3,
            "Comm + Uniq sum ({}) at feature {}",
            sum_var,
            j
        );
    }

    // Transform and reconstruct
    let scores = fa.transform(&data).expect("transform FA");
    assert_eq!(scores.len(), 100);
    assert_eq!(scores[0].len(), 2);

    let recon = fa.reconstruct(&data).expect("reconstruct FA");
    assert_eq!(recon.len(), 100);
    assert_eq!(recon[0].len(), 4);

    let errs = fa.reconstruction_errors(&data).expect("reconstruction errors FA");
    assert_eq!(errs.len(), 100);

    // Test Schmid-Leiman hierarchical bifactor decomposition with Promax rotation
    let fa_oblique = FactorAnalysisModel::fit(
        &data,
        Some(2),
        Some(100),
        Some(1e-4),
        Some(RotationMethod::Promax(4.0)),
        "paf",
    )
    .expect("fit FA Promax");

    assert!(fa_oblique.factor_correlations.is_some());
    let (bifactor_loadings, second_order_g) = fa_oblique
        .schmid_leiman()
        .expect("schmid_leiman bifactor");
    assert_eq!(bifactor_loadings.len(), 4);
    // 1 general factor + 2 group factors = 3 columns
    assert_eq!(bifactor_loadings[0].len(), 3);
    assert_eq!(second_order_g.len(), 2);

    // Report
    let report = fa.report();
    assert_eq!(report.learner.family, "Component Analysis");
    assert!(report.to_markdown().contains("mean_communality"));
}
