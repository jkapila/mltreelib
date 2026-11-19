use mltreelib_core::criteria::GainFn;
use mltreelib_core::ctree::{compute_ctree_test_categorical, compute_ctree_test_numeric};
use mltreelib_core::gate::SplitTest;
use mltreelib_core::model_tree::{ModelTree, ModelTreeParams};
use mltreelib_core::prune::PruneCost;
use mltreelib_core::split::ColType;

#[test]
fn test_ctree_numeric_and_categorical_null_calibration() {
    // Under H_0 (independence), p-values should be well-behaved and not favor 100-level over 2-level.
    let n = 2000;
    // Generate deterministic pseudo-random sequences
    let mut y = Vec::with_capacity(n);
    let mut col_bin: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut col_cat100: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 123456789;
    let mut lcg = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut sum_y = 0.0;
    for i in 0..n {
        let u1 = lcg().max(1e-9);
        let u2 = lcg();
        let val_y = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        y.push(val_y);
        sum_y += val_y;

        col_bin.push(Some((i % 2) as f64));
        col_cat100.push(Some((i % 100) as f64));
    }

    let rows: Vec<usize> = (0..n).collect();
    let w = vec![1.0; n];
    let y_mean = sum_y / (n as f64);
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let vy = sst / ((n - 1) as f64);

    let (_c_bin, df_bin, p_bin) =
        compute_ctree_test_categorical(&col_bin, &y, &w, &rows, y_mean, vy).expect("bin test");
    let (_c_cat100, df_cat100, p_cat100) =
        compute_ctree_test_categorical(&col_cat100, &y, &w, &rows, y_mean, vy).expect("cat100 test");

    assert_eq!(df_bin, 1.0);
    assert_eq!(df_cat100, 99.0);

    // Neither should reject null at alpha = 0.01 under pure noise
    assert!(p_bin > 0.01, "bin p-value under null: {p_bin}");
    assert!(p_cat100 > 0.01, "cat100 p-value under null: {p_cat100}");

    // Test numeric column as well
    let col_num: Vec<Option<f64>> = (0..n).map(|_| Some(lcg())).collect();
    let (_c_num, df_num, p_num) =
        compute_ctree_test_numeric(&col_num, &y, &w, &rows, y_mean, vy).expect("num test");
    assert_eq!(df_num, 1.0);
    assert!(p_num > 0.01, "num p-value under null: {p_num}");
}

#[test]
fn test_ctree_strong_signal_recovery() {
    let n = 400;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 987654321;
    let mut lcg = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for i in 0..n {
        let val_x0 = (i as f64 / (n as f64)) * 4.0 - 2.0; // [-2, +2]
        let val_x1 = (i % 20) as f64;
        let val_x2 = lcg();

        let eps = (lcg() - 0.5) * 0.2;
        let val_y = if val_x0 <= 0.0 { -3.0 } else { 3.0 } + eps;

        x.push(vec![Some(val_x0), Some(val_x1), Some(val_x2)]);
        y.push(Some(val_y));
    }

    let dtypes = vec![ColType::Float, ColType::Categorical, ColType::Float];

    let params = ModelTreeParams {
        max_depth: 2,
        gain: GainFn::Ctree,
        split_test: SplitTest::Ctree,
        split_alpha: 0.05,
        bonferroni: true,
        dtypes,
        prune: PruneCost::None,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit ctree");
    let root = tree.root_split().expect("root must split on strong signal");

    // The chosen variable must be column 0 (the true signal), NOT column 1 (20-level noise)
    assert_eq!(root.col, 0, "CTree must pick column 0, got {}", root.col);
    assert!(
        (root.threshold - 0.0).abs() < 0.2,
        "Threshold must be near 0.0, got {}",
        root.threshold
    );

    let root_prof = tree.profiles().iter().find(|p| p.depth == 0).unwrap();
    let gate = root_prof.gate.as_ref().expect("gate note must be recorded");
    assert_eq!(gate.test_name, "ctree");
    assert!(gate.passed);
    assert!(gate.p < 0.001, "Significance on strong signal must be tiny, got {}", gate.p);
}

#[test]
fn test_ctree_null_stopping() {
    let n = 200;
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 424242;
    let mut lcg = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for i in 0..n {
        x.push(vec![
            Some((i % 2) as f64),
            Some((i % 10) as f64),
            Some(lcg()),
        ]);
        y.push(Some(lcg() * 2.0 - 1.0));
    }

    let dtypes = vec![ColType::Categorical, ColType::Categorical, ColType::Float];

    let params = ModelTreeParams {
        max_depth: 3,
        gain: GainFn::Ctree,
        split_test: SplitTest::Ctree,
        split_alpha: 0.01,
        bonferroni: true,
        dtypes,
        prune: PruneCost::None,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit null ctree");
    // Under pure noise with alpha = 0.01, tree should not split at root
    assert!(
        tree.root_split().is_none(),
        "CTree under null with alpha=0.01 must not split"
    );
}

#[test]
fn test_ctree_categorical_grouping_recovery() {
    let n = 400;
    // Categorical X0 has 4 levels: 0 and 1 have Y ~ -2.0, 2 and 3 have Y ~ +2.0
    let mut x: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);

    let mut seed: u64 = 112233;
    let mut lcg = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for i in 0..n {
        let level = (i % 4) as f64;
        let eps = (lcg() - 0.5) * 0.1;
        let val_y = if level < 2.0 { -2.0 } else { 2.0 } + eps;

        x.push(vec![Some(level), Some(lcg())]);
        y.push(Some(val_y));
    }

    let dtypes = vec![ColType::Categorical, ColType::Float];

    let params = ModelTreeParams {
        max_depth: 2,
        gain: GainFn::Ctree,
        split_test: SplitTest::Ctree,
        split_alpha: 0.05,
        bonferroni: true,
        dtypes,
        prune: PruneCost::None,
        ..Default::default()
    };

    let tree = ModelTree::fit(&x, &y, params).expect("fit categorical ctree");
    let root = tree.root_split().expect("root must split");
    assert_eq!(root.col, 0, "Must choose categorical column 0");
    assert!(root.cat.is_some(), "Must have categorical partition");
}
