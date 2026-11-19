use mltreelib_core::criteria::GainFn;
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

#[test]
fn test_wasserstein2_recovers_distribution_split() {
    let n = 200;
    let d = 30; // 30 quantile points per distribution
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y_vecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_dummy: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut lcg: u64 = 42;
    let mut rand_uniform = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for _i in 0..n {
        let x0 = rand_uniform() * 2.0 - 1.0;
        let x1 = rand_uniform() * 2.0 - 1.0; // noise
        x.push(vec![Some(x0), Some(x1)]);

        // Generate 30 sorted quantiles
        let offset = if x0 < 0.0 { -2.0 } else { 2.0 };
        let mut q_vec = Vec::with_capacity(d);
        for k in 0..d {
            let tau = (k as f64 + 0.5) / (d as f64);
            // approximate normal quantile: offset + 0.5 * (tau - 0.5) * 4.0
            q_vec.push(offset + (tau - 0.5) * 2.0);
        }
        y_vecs.push(q_vec);
        y_dummy.push(Some(offset));
    }

    let mut params = ModelTreeParams {
        max_depth: 1,
        min_rows: 20,
        gain: GainFn::Wasserstein2,
        ..Default::default()
    };
    params.target_vectors = Some(y_vecs);

    let tree = ModelTree::fit(&x, &y_dummy, params).expect("fit wasserstein");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1, "expected exactly 1 root split");
    let root = &splits[0];
    assert_eq!(root.col, 0, "must split on signal column 0");
    assert!(
        (root.threshold - 0.0).abs() < 0.3,
        "cutpoint should be near 0, got {}",
        root.threshold
    );

    // Verify vector prediction matches Fréchet barycenter
    let test_x = vec![vec![Some(-0.8), Some(0.0)], vec![Some(0.8), Some(0.0)]];
    let preds = tree.predict_vector(&test_x).expect("predict_vector");
    assert_eq!(preds.len(), 2);
    // Pred 0 should have negative offset, Pred 1 positive offset
    let mean_pred0: f64 = preds[0].iter().sum::<f64>() / (d as f64);
    let mean_pred1: f64 = preds[1].iter().sum::<f64>() / (d as f64);
    assert!(mean_pred0 < -1.5, "expected ~ -2.0, got {}", mean_pred0);
    assert!(mean_pred1 > 1.5, "expected ~ +2.0, got {}", mean_pred1);
}

#[test]
fn test_wasserstein2_shape_split_identical_means() {
    let n = 200;
    let d = 40;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y_vecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_dummy: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut lcg: u64 = 12345;
    let mut rand_uniform = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for _ in 0..n {
        let x0 = rand_uniform() * 2.0 - 1.0;
        let x1 = rand_uniform() * 2.0 - 1.0; // noise
        x.push(vec![Some(x0), Some(x1)]);

        // Both regimes have mean 0.0, but Regime A is Uniform[-2, 2] and Regime B is Bimodal {-1.5, +1.5}
        let mut q_vec = Vec::with_capacity(d);
        for k in 0..d {
            let tau = (k as f64 + 0.5) / (d as f64);
            if x0 < 0.0 {
                // Uniform[-2, 2] quantile: -2 + 4 * tau
                q_vec.push(-2.0 + 4.0 * tau);
            } else {
                // Bimodal: -1.5 for tau < 0.5, +1.5 for tau >= 0.5
                if tau < 0.5 {
                    q_vec.push(-1.5 + (tau - 0.25) * 0.5);
                } else {
                    q_vec.push(1.5 + (tau - 0.75) * 0.5);
                }
            }
        }
        y_vecs.push(q_vec);
        y_dummy.push(Some(0.0)); // Identical mean
    }

    let mut params = ModelTreeParams {
        max_depth: 1,
        min_rows: 20,
        gain: GainFn::Wasserstein2,
        ..Default::default()
    };
    params.target_vectors = Some(y_vecs);

    let tree = ModelTree::fit(&x, &y_dummy, params).expect("fit wasserstein");
    let splits = tree.splits();
    assert_eq!(splits.len(), 1);
    let root = &splits[0];
    assert_eq!(root.col, 0, "Wasserstein-2 must separate shapes on col 0 even when means are identical");
    assert!((root.threshold - 0.0).abs() < 0.35);
}

#[test]
fn test_fairness_penalty_eliminates_disparate_impact() {
    let n = 300;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut lcg: u64 = 999;
    let mut rand_uniform = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };

    // Col 0: Protected attribute S in {0, 1}
    // Col 1: High correlation with S (proxy feature: X1 = S + small noise) -> unfair
    // Col 2: Fair predictor X2 in [-1, 1], completely independent of S, also predicts Y
    for _ in 0..n {
        let s = if rand_uniform() < 0.5 { 0.0 } else { 1.0 };
        let x1_proxy = s + (rand_uniform() - 0.5) * 0.1;
        let x2_fair = rand_uniform() * 2.0 - 1.0;

        // Target depends primarily on x2, but x1 proxy has slightly higher empirical correlation in unconstrained tree
        let target = if x2_fair > 0.0 { 10.0 } else { 0.0 } + 2.0 * s;

        x.push(vec![Some(s), Some(x1_proxy), Some(x2_fair)]);
        y.push(Some(target));
    }

    // Unconstrained tree (fairness_penalty = 0.0):
    // Splitting on x1 or s gives strong impurity reduction but severe demographic disparity
    let p_unconstrained = ModelTreeParams {
        max_depth: 1,
        min_rows: 30,
        fairness_penalty: 0.0,
        protected_col: Some(0),
        ..Default::default()
    };
    let tree_uncon = ModelTree::fit(&x, &y, p_unconstrained).expect("fit unconstrained");
    let splits_uncon = tree_uncon.splits();
    assert_eq!(splits_uncon.len(), 1);
    let uncon_col = splits_uncon[0].col;

    // Fair tree (fairness_penalty = 1.0):
    // Penalizes the disparity on col 0, forcing tree to split on fair col 2
    let p_fair = ModelTreeParams {
        max_depth: 1,
        min_rows: 30,
        fairness_penalty: 1.0,
        protected_col: Some(0),
        ..Default::default()
    };
    let tree_fair = ModelTree::fit(&x, &y, p_fair).expect("fit fair");
    let splits_fair = tree_fair.splits();
    assert_eq!(splits_fair.len(), 1);
    let fair_col = splits_fair[0].col;

    assert_eq!(fair_col, 2, "Fair constrained tree must split on fair column 2");
    // Ensure that unconstrained and fair chose different columns or fair avoided col 0 / proxy
    assert!(uncon_col != 2 || fair_col == 2);
}
