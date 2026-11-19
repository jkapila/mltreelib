use mltreelib_core::fast_interaction::fast_interaction_screen;
use mltreelib_core::model_tree::{FairnessMetric, ModelTree, ModelTreeParams};

#[test]
fn test_equalized_odds_fairness_penalty() {
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

        let target = if x2_fair > 0.0 { 6.0 } else { 0.0 } + 10.0 * x1_proxy;

        x.push(vec![Some(s), Some(x1_proxy), Some(x2_fair)]);
        y.push(Some(target));
    }

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

    let p_eo = ModelTreeParams {
        max_depth: 1,
        min_rows: 30,
        fairness_penalty: 0.85,
        protected_col: Some(0),
        fairness_metric: FairnessMetric::EqualizedOdds,
        ..Default::default()
    };
    let tree_eo = ModelTree::fit(&x, &y, p_eo).expect("fit equalized odds");
    let splits_eo = tree_eo.splits();
    assert_eq!(splits_eo.len(), 1);
    let eo_col = splits_eo[0].col;

    println!("uncon splits: {:?}", tree_uncon.splits());
    println!("eo splits: {:?}", tree_eo.splits());
    assert_eq!(eo_col, 2, "Equalized Odds fair tree must split on fair column 2");
    assert_ne!(uncon_col, 2, "Unconstrained tree splits on proxy col 0 or 1");
}

#[test]
fn test_extratrees_random_cuts_deterministic() {
    let n = 200;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let val = (i as f64) / 20.0;
        x.push(vec![Some(val), Some(val * val)]);
        y.push(Some(if val > 5.0 { 1.0 } else { 0.0 }));
    }

    // Standard CART: deterministic cut at ~5.0
    let mut cart_params = ModelTreeParams::default();
    cart_params.min_rows = 10;
    cart_params.extratrees = false;
    let t_cart = ModelTree::fit(&x, &y, cart_params).unwrap();
    let root_cart = t_cart.root_split().unwrap();

    // ExtraTrees: uniformly randomized cut
    let mut et_params = ModelTreeParams::default();
    et_params.min_rows = 10;
    et_params.extratrees = true;
    et_params.seed = 12345;
    let t_et1 = ModelTree::fit(&x, &y, et_params.clone()).unwrap();
    let root_et1 = t_et1.root_split().unwrap();

    // Verify determinism with same seed
    let t_et2 = ModelTree::fit(&x, &y, et_params).unwrap();
    let root_et2 = t_et2.root_split().unwrap();

    assert_eq!(root_et1.col, root_et2.col);
    assert!((root_et1.threshold - root_et2.threshold).abs() < 1e-10);
    // Randomized threshold differs from exact CART optimal threshold
    assert_ne!(root_cart.threshold, root_et1.threshold);
}

#[test]
fn test_fast_interaction_screening_xor() {
    let n = 300;
    let mut cols: Vec<Vec<Option<f64>>> = vec![
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let w = vec![1.0; n];
    let col_names = vec![
        "feature_a".to_string(),
        "feature_b".to_string(),
        "noise_c".to_string(),
        "noise_d".to_string(),
    ];

    let mut state: u64 = 42;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (state >> 33) as f64 / 2147483648.0;
        -1.0 + 2.0 * x
    };

    for _ in 0..n {
        let x0 = next_rand();
        let x1 = next_rand();
        let x2 = next_rand();
        let x3 = next_rand();

        // Main effects plus strong non-linear interaction between x0 and x1
        let y_val = 2.0 * x0 + 2.0 * x1 + 8.0 * (x0 * x1);

        cols[0].push(Some(x0));
        cols[1].push(Some(x1));
        cols[2].push(Some(x2));
        cols[3].push(Some(x3));
        y.push(y_val);
    }

    let interactions = fast_interaction_screen(&cols, &y, &w, &col_names, 5, 5);
    assert!(!interactions.is_empty());

    let top = &interactions[0];
    // Top interaction must be between feature_a (0) and feature_b (1)
    assert_eq!(top.col1, 0);
    assert_eq!(top.col2, 1);
    assert_eq!(top.col1_name, "feature_a");
    assert_eq!(top.col2_name, "feature_b");
    assert!(top.score > 0.05, "Interaction score must be strong: got {}", top.score);

    // Any noise pairs must have much lower interaction scores
    for inter in &interactions[1..] {
        assert!(top.score > inter.score);
    }
}
