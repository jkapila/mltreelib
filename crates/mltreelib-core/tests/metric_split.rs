use mltreelib_core::criteria::{GainFn, KernelKind};
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};
use mltreelib_core::prune::PruneCost;
use mltreelib_core::split::ColType;

#[test]
fn test_mmd_orthogonal_covariances_identical_means() {
    // Spec 12 requirement:
    // Generate two 10-dimensional Gaussian clusters with identical means (0.0)
    // but orthogonal covariances. Standard SSE cannot split (delta mean = 0).
    // MMD tree separates the regimes along X0.
    let n = 240;
    let d = 10;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y_vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_scalar: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 54321;
    let next_f = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let rand_norm = |s: &mut u64| -> f64 {
        let u1 = next_f(s).max(1e-9);
        let u2 = next_f(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    for i in 0..n {
        let x0 = (i as f64 / (n as f64)) * 4.0 - 2.0; // [-2, +2]
        let x1_noise = next_f(&mut seed);
        x.push(vec![Some(x0), Some(x1_noise)]);

        let mut y_vec = vec![0.0; d];
        if x0 <= 0.0 {
            // Regime 1: High variance on dimensions 0..5, tiny variance on 5..10
            for k in 0..5 {
                y_vec[k] = rand_norm(&mut seed) * 2.5;
            }
            for k in 5..10 {
                y_vec[k] = rand_norm(&mut seed) * 0.1;
            }
        } else {
            // Regime 2: Tiny variance on dimensions 0..5, high variance on 5..10
            for k in 0..5 {
                y_vec[k] = rand_norm(&mut seed) * 0.1;
            }
            for k in 5..10 {
                y_vec[k] = rand_norm(&mut seed) * 2.5;
            }
        }
        y_scalar.push(Some(y_vec[0]));
        y_vectors.push(y_vec);
    }

    let params = ModelTreeParams {
        max_depth: 2,
        gain: GainFn::Mmd {
            kernel: KernelKind::Rbf,
            bandwidth: 0.0, // Automatic median heuristic bandwidth
        },
        min_rows: 20,
        prune: PruneCost::None,
        target_vectors: Some(y_vectors),
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y_scalar, params).expect("fit mmd tree");
    let root = tree.root_split().expect("MMD root must split on orthogonal covariances");

    // Must split on X0 (the regime partition), NOT X1 (noise)
    assert_eq!(root.col, 0, "MMD split must choose X0, got {}", root.col);
    assert!(
        (root.threshold - 0.0).abs() < 0.35,
        "Threshold must be near 0.0, got {}",
        root.threshold
    );

    // Predict vector test
    let test_x = vec![
        vec![Some(-1.0), Some(0.5)],
        vec![Some(1.0), Some(0.5)],
    ];
    let preds = tree.predict_vector(&test_x).expect("predict vector");
    assert_eq!(preds.len(), 2);
    assert_eq!(preds[0].len(), d);
    assert_eq!(preds[1].len(), d);
}

#[test]
fn test_energy_distance_vector_targets() {
    let n = 200;
    let d = 5;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y_vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_scalar: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 99999;
    let mut lcg = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for i in 0..n {
        let x0 = (i as f64 / (n as f64)) * 4.0 - 2.0;
        let x1_cat = (i % 10) as f64; // 10-level categorical noise
        x.push(vec![Some(x0), Some(x1_cat)]);

        let center = if x0 <= 0.0 { -3.0 } else { 3.0 };
        let mut y_vec = Vec::with_capacity(d);
        for _ in 0..d {
            let eps = (lcg() - 0.5) * 0.2;
            y_vec.push(center + eps);
        }
        y_scalar.push(Some(y_vec[0]));
        y_vectors.push(y_vec);
    }

    let dtypes = vec![ColType::Float, ColType::Categorical];
    let params = ModelTreeParams {
        max_depth: 2,
        gain: GainFn::EnergyDistance,
        min_rows: 15,
        dtypes,
        prune: PruneCost::None,
        target_vectors: Some(y_vectors),
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y_scalar, params).expect("fit energy tree");
    let root = tree.root_split().expect("Energy distance must split");
    assert_eq!(root.col, 0, "Must choose column 0, got {}", root.col);
    assert!(
        (root.threshold - 0.0).abs() < 0.25,
        "Threshold must be near 0.0, got {}",
        root.threshold
    );

    let test_x = vec![
        vec![Some(-1.0), Some(0.0)],
        vec![Some(1.0), Some(0.0)],
    ];
    let preds = tree.predict_vector(&test_x).expect("predict vector");
    assert_eq!(preds.len(), 2);
    for val in &preds[0] {
        assert!((val - (-3.0)).abs() < 0.5);
    }
    for val in &preds[1] {
        assert!((val - 3.0).abs() < 0.5);
    }
}

#[test]
fn test_scalar_energy_distance_and_mmd_gain_of() {
    let n = 150;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let x0 = (i as f64 / (n as f64)) * 2.0 - 1.0;
        let y_val = if x0 <= 0.0 { -5.0 } else { 5.0 };
        x.push(vec![Some(x0)]);
        y.push(Some(y_val));
    }

    // Scalar EnergyDistance test
    let params_energy = ModelTreeParams {
        max_depth: 1,
        gain: GainFn::EnergyDistance,
        min_rows: 10,
        prune: PruneCost::None,
        ..Default::default()
    };
    let tree_energy = ModelTree::fit(&x, &y, params_energy).expect("fit 1D energy");
    assert_eq!(tree_energy.root_split().unwrap().col, 0);

    // Scalar MMD test
    let params_mmd = ModelTreeParams {
        max_depth: 1,
        gain: GainFn::Mmd {
            kernel: KernelKind::Rbf,
            bandwidth: 1.0,
        },
        min_rows: 10,
        prune: PruneCost::None,
        ..Default::default()
    };
    let tree_mmd = ModelTree::fit(&x, &y, params_mmd).expect("fit 1D mmd");
    assert_eq!(tree_mmd.root_split().unwrap().col, 0);
}
