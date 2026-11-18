use mltreelib_core::solver::{
    fit_model, solve_ccd, solve_cmaes, solve_lbfgs, solve_normal, solve_rcga,
    solve_shade, solve_snes, CcdOptions, ColumnSliceView, DenseView, DesignMatrix, EvolutionaryKind,
    EvolutionaryOptions, GlmFamily, LbfgsOptions, SolverOptions,
};

#[test]
fn test_ols_normal_exact_and_summary() {
    // Ground truth: y = 2.5 + 1.8 * x1 - 3.2 * x2
    let n = 200;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) * 0.05;
        let x2 = ((i * 7 % 23) as f64) * 0.1;
        let noise = ((i * 13 % 17) as f64 - 8.0) * 0.02;
        let yi = 2.5 + 1.8 * x1 - 3.2 * x2 + noise;
        x.push(vec![x1, x2]);
        y.push(yi);
    }

    let view = DenseView::new(&x, &y, None, true);
    let (beta, summary_opt) = solve_normal(&view, 0.0, None).expect("solve_normal failed");

    assert_eq!(beta.len(), 3);
    assert!((beta[0] - 2.5).abs() < 0.1, "beta[0]={}", beta[0]);
    assert!((beta[1] - 1.8).abs() < 0.05, "beta[1]={}", beta[1]);
    assert!((beta[2] - (-3.2)).abs() < 0.05, "beta[2]={}", beta[2]);

    let summary = summary_opt.expect("summary should be present");
    assert!(summary.r_squared.unwrap() > 0.99);
    assert!(summary.f_statistic.unwrap() > 1000.0);
    assert_eq!(summary.coefficients.len(), 3);
    assert!(summary.coefficients[1].p_value < 1e-5);
    assert!(summary.coefficients[2].p_value < 1e-5);
}

#[test]
fn test_ridge_penalty_shrinks_coefficients() {
    let n = 100;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        x.push(vec![x1]);
        y.push(3.0 + 5.0 * x1);
    }

    let view = DenseView::new(&x, &y, None, true);
    let (beta_ols, _) = solve_normal(&view, 0.0, None).unwrap();
    let (beta_ridge, _) = solve_normal(&view, 100.0, None).unwrap();

    assert!(beta_ridge[1] < beta_ols[1], "Ridge should shrink slope");
    assert!(beta_ridge[1] > 0.0);
}

#[test]
fn test_ccd_lasso_sparsity() {
    let n = 150;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        let x_noise1 = ((i * 3 % 11) as f64) * 0.1;
        let x_noise2 = ((i * 5 % 13) as f64) * 0.1;
        // y depends ONLY on x1
        let yi = 1.0 + 4.0 * x1;
        x.push(vec![x1, x_noise1, x_noise2]);
        y.push(yi);
    }

    let view = DenseView::new(&x, &y, None, true);
    let opts = CcdOptions {
        l1: 20.0,
        l2: 0.0,
        max_iter: 1000,
        tol: 1e-7,
        monotone: Vec::new(),
    };
    let beta = solve_ccd(&view, &opts, None).expect("solve_ccd failed");

    assert_eq!(beta.len(), 4);
    assert!(beta[1] > 2.0, "Informative feature preserved: beta[1]={}", beta[1]);
    assert_eq!(beta[2], 0.0, "Noise feature 1 shrunk to exactly 0.0");
    assert_eq!(beta[3], 0.0, "Noise feature 2 shrunk to exactly 0.0");
}

#[test]
fn test_ccd_monotone_constraints() {
    let n = 100;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        // Unconstrained slope would be negative (-2.0)
        let yi = 5.0 - 2.0 * x1;
        x.push(vec![x1]);
        y.push(yi);
    }

    let view = DenseView::new(&x, &y, None, true);
    // Enforce positive monotonicity (+1) on feature 1
    let opts = CcdOptions {
        l1: 0.0,
        l2: 0.0,
        max_iter: 1000,
        tol: 1e-7,
        monotone: vec![1],
    };
    let beta = solve_ccd(&view, &opts, None).unwrap();

    assert!(beta[1] >= 0.0, "Positive monotone violated: beta[1]={}", beta[1]);
}

#[test]
fn test_lbfgs_binomial_logistic() {
    let n = 200;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = ((i as f64) - 100.0) * 0.05;
        let prob = 1.0 / (1.0 + (-2.0 * x1).exp());
        let yi = if prob > 0.5 { 1.0 } else { 0.0 };
        x.push(vec![x1]);
        y.push(yi);
    }

    let view = DenseView::new(&x, &y, None, true);
    let opts = LbfgsOptions {
        family: GlmFamily::Binomial,
        l2: 1e-2,
        max_iter: 100,
        m: 6,
        tol: 1e-5,
        verbose: 0,
    };
    let beta = solve_lbfgs(&view, &opts, None).expect("solve_lbfgs failed");

    assert_eq!(beta.len(), 2);
    assert!(beta[1] > 0.5, "Slope should be positive: beta[1]={}", beta[1]);
}

#[test]
fn test_lbfgs_poisson_count() {
    let n = 150;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) * 0.02;
        let mu = (0.5 + 1.2 * x1).exp();
        x.push(vec![x1]);
        y.push(mu.round());
    }

    let view = DenseView::new(&x, &y, None, true);
    let opts = LbfgsOptions {
        family: GlmFamily::Poisson,
        l2: 1e-4,
        max_iter: 100,
        m: 6,
        tol: 1e-5,
        verbose: 0,
    };
    let beta = solve_lbfgs(&view, &opts, None).expect("poisson solve failed");

    assert_eq!(beta.len(), 2);
    assert!((beta[0] - 0.5).abs() < 0.3, "beta[0]={}", beta[0]);
    assert!((beta[1] - 1.2).abs() < 0.3, "beta[1]={}", beta[1]);
}

#[test]
fn test_column_slice_view_equivalence() {
    let n = 100;
    let mut col1 = Vec::with_capacity(n);
    let mut col2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);

    let mut dense_x = Vec::with_capacity(n / 2);
    let mut dense_y = Vec::with_capacity(n / 2);
    let mut rows = Vec::new();

    for i in 0..n {
        let v1 = (i as f64) * 0.1;
        let v2 = ((i * 3 % 7) as f64) * 0.2;
        let yi = 1.5 + 2.0 * v1 - 1.0 * v2;
        col1.push(Some(v1));
        col2.push(Some(v2));
        y.push(yi);
        w.push(1.0);

        if i % 2 == 0 {
            rows.push(i);
            dense_x.push(vec![v1, v2]);
            dense_y.push(yi);
        }
    }

    let cols = vec![col1, col2];
    let chosen = vec![0, 1];
    let slice_view = ColumnSliceView::new(&cols, &y, &w, &rows, &chosen, false);
    let dense_view = DenseView::new(&dense_x, &dense_y, None, true);

    let (beta_slice, _) = solve_normal(&slice_view, 0.0, None).unwrap();
    let (beta_dense, _) = solve_normal(&dense_view, 0.0, None).unwrap();

    assert_eq!(beta_slice.len(), beta_dense.len());
    for j in 0..beta_slice.len() {
        assert!(
            (beta_slice[j] - beta_dense[j]).abs() < 1e-10,
            "Mismatch at j={}: slice={} dense={}",
            j,
            beta_slice[j],
            beta_dense[j]
        );
    }
}

#[test]
fn test_column_slice_view_missing_indicators() {
    let n = 60;
    let mut col = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let w = vec![1.0; n];
    let rows: Vec<usize> = (0..n).collect();

    for i in 0..n {
        if i % 3 == 0 {
            col.push(None);
            y.push(10.0); // missing level has distinct offset
        } else {
            let v = i as f64;
            col.push(Some(v));
            y.push(2.0 + 3.0 * v);
        }
    }

    let cols = vec![col];
    let chosen = vec![0];
    let slice_view = ColumnSliceView::new(&cols, &y, &w, &rows, &chosen, true);

    // Dimension: intercept + 1 feature + 1 indicator = 3
    assert_eq!(slice_view.n_features(), 3);
    let (beta, _) = solve_normal(&slice_view, 1e-4, None).unwrap();
    assert_eq!(beta.len(), 3);
    assert!((beta[1] - 3.0).abs() < 0.2, "Slope on observed values: {}", beta[1]);
}

#[test]
fn test_fit_model_unified_dispatch() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0];
    let view = DenseView::new(&x, &y, None, true);

    let opts = SolverOptions::default();
    let res = fit_model(&view, &opts, None, None).expect("fit_model failed");

    assert_eq!(res.coef.len(), 1);
    assert!((res.coef[0] - 2.0).abs() < 1e-6);
    assert!(res.summary.is_some());
    let s = res.summary.unwrap();
    assert!(s.r_squared.unwrap() > 0.999);
}

#[test]
fn test_cmaes_recovers_linear_coefficients() {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        x.push(vec![x1]);
        y.push(1.5 + 2.5 * x1);
    }
    let view = DenseView::new(&x, &y, None, true);
    let opts = EvolutionaryOptions {
        kind: EvolutionaryKind::CmaEs,
        family: GlmFamily::Gaussian,
        l1: 0.0,
        l2: 0.0,
        monotone: Vec::new(),
        population_size: Some(15),
        max_generations: 80,
        tol: 1e-5,
        seed: 42,
        verbose: 0,
    };
    let beta = solve_cmaes(&view, &opts, None).expect("CMA-ES solve failed");
    assert_eq!(beta.len(), 2);
    assert!((beta[0] - 1.5).abs() < 0.25, "intercept={}", beta[0]);
    assert!((beta[1] - 2.5).abs() < 0.15, "slope={}", beta[1]);
}

#[test]
fn test_shade_adaptive_de_convergence() {
    let n = 80;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        x.push(vec![x1]);
        y.push(3.0 + 1.2 * x1);
    }
    let view = DenseView::new(&x, &y, None, true);
    let opts = EvolutionaryOptions {
        kind: EvolutionaryKind::Shade,
        family: GlmFamily::Gaussian,
        l1: 0.0,
        l2: 1e-3,
        monotone: Vec::new(),
        population_size: Some(20),
        max_generations: 60,
        tol: 1e-5,
        seed: 123,
        verbose: 0,
    };
    let beta = solve_shade(&view, &opts, None).expect("SHADE solve failed");
    assert_eq!(beta.len(), 2);
    assert!((beta[0] - 3.0).abs() < 0.3, "intercept={}", beta[0]);
    assert!((beta[1] - 1.2).abs() < 0.15, "slope={}", beta[1]);
}

#[test]
fn test_snes_natural_gradient_learning() {
    let n = 90;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) * 0.05;
        x.push(vec![x1]);
        y.push(2.0 + 4.0 * x1);
    }
    let view = DenseView::new(&x, &y, None, true);
    let opts = EvolutionaryOptions {
        kind: EvolutionaryKind::NaturalGradient,
        family: GlmFamily::Gaussian,
        l1: 0.0,
        l2: 1e-4,
        monotone: Vec::new(),
        population_size: Some(20),
        max_generations: 70,
        tol: 1e-5,
        seed: 999,
        verbose: 0,
    };
    let beta = solve_snes(&view, &opts, None).expect("sNES solve failed");
    assert_eq!(beta.len(), 2);
    assert!((beta[0] - 2.0).abs() < 0.35, "intercept={}", beta[0]);
    assert!((beta[1] - 4.0).abs() < 0.25, "slope={}", beta[1]);
}

#[test]
fn test_rcga_monotone_constrained() {
    let n = 70;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) * 0.1;
        x.push(vec![x1]);
        y.push(5.0 - 2.0 * x1);
    }
    let view = DenseView::new(&x, &y, None, true);
    // Enforce positive slope constraint (mono = 1)
    let opts = EvolutionaryOptions {
        kind: EvolutionaryKind::GeneticAlgorithm,
        family: GlmFamily::Gaussian,
        l1: 0.0,
        l2: 0.0,
        monotone: vec![1],
        population_size: Some(30),
        max_generations: 40,
        tol: 1e-4,
        seed: 42,
        verbose: 0,
    };
    let beta = solve_rcga(&view, &opts, None).expect("RCGA solve failed");
    assert_eq!(beta.len(), 2);
    assert!(beta[1] >= 0.0, "Monotone constraint must be satisfied: {}", beta[1]);
}

