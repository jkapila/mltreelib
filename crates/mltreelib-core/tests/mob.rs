use mltreelib_core::gate::SplitTest;
use mltreelib_core::leaf::LeafKind;
use mltreelib_core::mob::invert_symmetric_psd;
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};

#[test]
fn test_mob_invert_matrix() {
    let eye = [1.0, 0.0, 0.0, 1.0];
    let inv = invert_symmetric_psd(&eye, 2).expect("invert identity");
    assert!((inv[0] - 1.0).abs() < 1e-4);
    assert!(inv[1].abs() < 1e-4);
    assert!(inv[2].abs() < 1e-4);
    assert!((inv[3] - 1.0).abs() < 1e-4);

    let mat = [4.0, 1.0, 1.0, 3.0];
    let inv = invert_symmetric_psd(&mat, 2).expect("invert 2x2");
    // Det ~ 11, inv ~ [3/11, -1/11, -1/11, 4/11] = [0.2727, -0.0909, -0.0909, 0.3636]
    assert!((inv[0] - 3.0 / 11.0).abs() < 1e-3);
    assert!((inv[1] - (-1.0 / 11.0)).abs() < 1e-3);
    assert!((inv[3] - 4.0 / 11.0).abs() < 1e-3);
}

#[test]
fn test_mob_linear_structural_break() {
    // Structural break in linear slope along X0 at X0 = 0:
    // Left: y = 2.0 * X1 + noise
    // Right: y = -2.0 * X1 + noise
    // X2 is an independent noise variable
    let n = 200;
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    let mut y: Vec<Option<f64>> = Vec::new();

    let mut rng: u64 = 42;
    let mut rand_f = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    for i in 0..n {
        let x0 = -1.0 + 2.0 * (i as f64 / (n - 1) as f64);
        let x1 = -1.0 + 2.0 * rand_f();
        let x2 = -1.0 + 2.0 * rand_f();
        let eps = 0.05 * (-1.0 + 2.0 * rand_f());

        let y_val = if x0 <= 0.0 {
            2.0 * x1 + eps
        } else {
            -2.0 * x1 + eps
        };

        x.push(vec![Some(x0), Some(x1), Some(x2)]);
        y.push(Some(y_val));
    }

    use mltreelib_core::leaf::LeafCols;
    use mltreelib_core::prune::PruneCost;
    let params = ModelTreeParams {
        leaf: LeafKind::Linear,
        leaf_cols: LeafCols::All,
        split_cols: Some(vec![0, 2]),
        split_test: SplitTest::Mob,
        split_alpha: 0.05,
        bonferroni: true,
        max_depth: 2,
        min_rows: 15,
        prune: PruneCost::None,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit mob tree");

    // The root node must split on column 0 (x0)
    let profs = tree.profiles();
    let gate = profs
        .iter()
        .find(|p| p.depth == 0)
        .and_then(|p| p.gate.as_ref())
        .expect("GateNote on root");
    println!("var_p_values: {:?}", gate.var_p_values);

    let root_split = tree.root_split().expect("MOB tree must split root on structural break");
    assert_eq!(root_split.col, 0, "MOB must select X0 as the break variable");
    assert!(
        root_split.threshold.abs() < 0.25,
        "MOB split threshold must be near 0.0, got {}",
        root_split.threshold
    );

    assert_eq!(gate.test_name, "mob");
    assert!(gate.passed, "MOB test must pass on planted structural break");
    assert!(gate.p < 0.05, "MOB p-value must be significant");

    // Predictions on left side (x0 = -0.5, x1 = 1.0) -> ~2.0
    let pred_left = tree.predict(&[vec![Some(-0.5), Some(1.0), Some(0.0)]])[0];
    assert!(
        (pred_left - 2.0).abs() < 0.3,
        "Left prediction should be near 2.0, got {pred_left}"
    );

    // Predictions on right side (x0 = 0.5, x1 = 1.0) -> ~-2.0
    let pred_right = tree.predict(&[vec![Some(0.5), Some(1.0), Some(0.0)]])[0];
    assert!(
        (pred_right - (-2.0)).abs() < 0.3,
        "Right prediction should be near -2.0, got {pred_right}"
    );
}

#[test]
fn test_mob_mean_structural_break() {
    // Mean structural break along X0 at X0 = 0:
    // Left: y = 10.0 + noise
    // Right: y = -10.0 + noise
    let n = 150;
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    let mut y: Vec<Option<f64>> = Vec::new();

    let mut rng: u64 = 123;
    let mut rand_f = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    for i in 0..n {
        let x0 = -2.0 + 4.0 * (i as f64 / (n - 1) as f64);
        let x1 = -1.0 + 2.0 * rand_f();
        let eps = 0.1 * (-1.0 + 2.0 * rand_f());

        let y_val = if x0 <= 0.0 { 10.0 + eps } else { -10.0 + eps };

        x.push(vec![Some(x0), Some(x1)]);
        y.push(Some(y_val));
    }

    let params = ModelTreeParams {
        leaf: LeafKind::Mean,
        split_test: SplitTest::Mob,
        split_alpha: 0.01,
        bonferroni: true,
        max_depth: 2,
        min_rows: 15,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit mob mean tree");

    let profs = tree.profiles();
    let gate = profs
        .iter()
        .find(|p| p.depth == 0)
        .and_then(|p| p.gate.as_ref())
        .expect("GateNote on root");
    println!("var_p_values: {:?}", gate.var_p_values);

    let root_split = tree.root_split().expect("root must split on mean break");
    assert_eq!(root_split.col, 0, "MOB must select X0 as the break variable");
    assert!(
        root_split.threshold.abs() < 0.3,
        "MOB split threshold must be near 0.0, got {}",
        root_split.threshold
    );
}

#[test]
fn test_mob_null_stability_stops_at_root() {
    // Data with NO parameter instability: constant slope y = 3.0 * X1 + 1.0 across all X0
    let n = 150;
    let mut x: Vec<Vec<Option<f64>>> = Vec::new();
    let mut y: Vec<Option<f64>> = Vec::new();

    let mut rng: u64 = 777;
    let mut rand_f = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    for _ in 0..n {
        let x0 = -1.0 + 2.0 * rand_f();
        let x1 = -1.0 + 2.0 * rand_f();
        let eps = 0.05 * (-1.0 + 2.0 * rand_f());
        let y_val = 3.0 * x1 + 1.0 + eps;

        x.push(vec![Some(x0), Some(x1)]);
        y.push(Some(y_val));
    }

    let params = ModelTreeParams {
        leaf: LeafKind::Linear,
        split_test: SplitTest::Mob,
        split_alpha: 0.05,
        bonferroni: true,
        max_depth: 3,
        min_rows: 15,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit mob null tree");

    // The root node must NOT split because no instability is detected
    assert!(
        tree.splits().is_empty(),
        "MOB must stay as a single terminal leaf when parameters are stable under H0"
    );
    assert_eq!(tree.n_leaves(), 1, "Tree must have exactly 1 leaf");
}

#[test]
fn test_mob_unbiased_p_values_under_null() {
    // Under H0 (pure noise y), p-values for independent covariates of differing
    // cardinality/scale must be uniform [0, 1], avoiding Loh's selection bias.
    let n = 120;
    let mut rng: u64 = 999;
    let mut rand_f = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    let mut low_card_p_sum = 0.0;
    let mut high_card_p_sum = 0.0;
    let n_trials = 30;

    for _ in 0..n_trials {
        let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
        let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

        for _ in 0..n {
            let x_binary = if rand_f() > 0.5 { 1.0 } else { 0.0 };
            let x_cont = rand_f(); // continuous 100+ unique values
            let y_noise = rand_f(); // independent noise

            x.push(vec![Some(x_binary), Some(x_cont)]);
            y.push(Some(y_noise));
        }

        use mltreelib_core::split::ColType;
        let params = ModelTreeParams {
            leaf: LeafKind::Mean,
            split_test: SplitTest::Mob,
            split_alpha: 0.05,
            dtypes: vec![ColType::Categorical, ColType::Float],
            bonferroni: false, // evaluate raw marginal p-values
            max_depth: 1,
            min_rows: 10,
            ..Default::default()
        };

        let tree = ModelTree::fit(&x, &y, params).expect("fit null mob");
        let profs = tree.profiles();
        let root_prof = profs.iter().find(|p| p.depth == 0).expect("root prof");
        if let Some(ref gate) = root_prof.gate {
            for &(col, _, p) in &gate.var_p_values {
                if col == 0 {
                    low_card_p_sum += p;
                } else if col == 1 {
                    high_card_p_sum += p;
                }
            }
        }
    }

    let mean_p_low = low_card_p_sum / n_trials as f64;
    let mean_p_high = high_card_p_sum / n_trials as f64;

    // Under null, p-values are roughly uniform -> mean around 0.50 (+/- 0.20)
    // High-cardinality variable is NOT unfairly favored (mean p is NOT near 0).
    assert!(
        mean_p_high > 0.25,
        "High cardinality variable must not have deflated p-values (Loh bias), got mean p = {mean_p_high}"
    );
    assert!(
        (mean_p_low - mean_p_high).abs() < 0.25,
        "Low and high cardinality variables must have comparable p-value distributions under H0: {mean_p_low} vs {mean_p_high}"
    );
}
