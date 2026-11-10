use mltreelib_core::mars::{MarsGlmFamily, MarsPruningMethod, MarsRegressor};

#[test]
fn test_mars_1d_hinge_recovery() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..100 {
        let x_val = (i as f64) * 0.1; // 0.0 to 10.0
        // True function: bent at x = 5.0
        // if x < 5: 10 - 2 * (5 - x) = 2x
        // if x >= 5: 10 + 4 * (x - 5) = 4x - 10
        let h_left = (5.0 - x_val).max(0.0);
        let h_right = (x_val - 5.0).max(0.0);
        let y_val = 10.0 - 2.0 * h_left + 4.0 * h_right;
        x.push(vec![x_val]);
        y.push(y_val);
    }

    let mut mars = MarsRegressor::new(1, 15, 2.0);
    mars.fit(&x, &y).unwrap();

    let model = mars.model.as_ref().unwrap();
    assert!(model.n_terms() >= 2, "MARS should find at least one hinge pair");
    assert!(model.rss < 1.0, "RSS should be near 0 on clean hinge: got {}", model.rss);

    let preds = mars.predict(&[vec![0.0], vec![5.0], vec![10.0]]).unwrap();
    assert!((preds[0] - 0.0).abs() < 0.5, "pred at 0.0: got {}", preds[0]);
    assert!((preds[1] - 10.0).abs() < 0.5, "pred at 5.0: got {}", preds[1]);
    assert!((preds[2] - 30.0).abs() < 0.5, "pred at 10.0: got {}", preds[2]);
}

#[test]
fn test_mars_2d_interaction_recovery() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            let x0 = i as f64;
            let x1 = j as f64;
            // Product interaction: h(x0 - 7) * h(x1 - 7)
            let h0 = (x0 - 7.0).max(0.0);
            let h1 = (x1 - 7.0).max(0.0);
            let y_val = 5.0 + 2.0 * h0 * h1;
            x.push(vec![x0, x1]);
            y.push(y_val);
        }
    }

    let mut mars = MarsRegressor::new(2, 21, 2.0);
    mars.fit(&x, &y).unwrap();

    let model = mars.model.as_ref().unwrap();
    assert!(model.rss < 5.0, "Degree 2 MARS should fit interaction: RSS {}", model.rss);

    let test_flat = mars.predict(&[vec![3.0, 3.0], vec![3.0, 10.0], vec![10.0, 3.0]]).unwrap();
    for &p in &test_flat {
        assert!((p - 5.0).abs() < 0.5, "Flat region expected ~5.0, got {}", p);
    }
}

#[test]
fn test_mars_linpreds_linear_only_feature() {
    // Feature 0 is pure linear (y = 3.0 * x0)
    // Feature 1 is independent hinged: max(0, x1 - 5.0)
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            let x0 = i as f64 * 0.5;
            let x1 = j as f64 * 0.5;
            let y_val = 2.0 + 3.0 * x0 + 4.0 * (x1 - 2.5).max(0.0);
            x.push(vec![x0, x1]);
            y.push(y_val);
        }
    }

    let mut mars = MarsRegressor::new(1, 10, 2.0);
    // Force feature 0 to be linear only (no hinge knots)
    mars.params.linpreds = vec![0];
    mars.fit(&x, &y).unwrap();

    let model = mars.model.as_ref().unwrap();
    let eq = model.equation(Some(&["f0".to_string(), "f1".to_string()]));
    assert!(eq.contains("f0"), "Equation should contain linear f0 term: got {}", eq);
    // Ensure no hinge knot on f0 like h(f0 - ...)
    assert!(!eq.contains("h(f0 - "), "f0 was specified as linpreds, should not have knots: got {}", eq);
    assert!(model.rss < 0.5, "RSS should be near 0: got {}", model.rss);
}

#[test]
fn test_mars_sample_weights() {
    // Two groups of points:
    // x = 0..10, y = 2 * x (weight 1.0)
    // 5 outlier points at x = 5.0, y = 100.0 (weight 0.0)
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();

    for i in 0..20 {
        let xv = i as f64 * 0.5;
        x.push(vec![xv]);
        y.push(2.0 * xv);
        w.push(1.0);
    }
    for _ in 0..5 {
        x.push(vec![5.0]);
        y.push(100.0);
        w.push(0.0); // Zero weight outliers
    }

    let mut mars = MarsRegressor::new(1, 10, 2.0);
    mars.fit_weighted(&x, &y, Some(&w)).unwrap();

    let pred = mars.predict(&[vec![5.0]]).unwrap();
    assert!((pred[0] - 10.0).abs() < 1.0, "Zero-weight outliers should be ignored, expected ~10.0, got {}", pred[0]);
}

#[test]
fn test_mars_glm_binomial_logistic_classification() {
    // Binary target: 1 if x0 > 5.0, 0 otherwise
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let xv = i as f64 * 0.2; // 0 to 12
        x.push(vec![xv]);
        y.push(if xv > 5.0 { 1.0 } else { 0.0 });
    }

    let mut mars = MarsRegressor::new(1, 10, 2.0);
    mars.params.glm = MarsGlmFamily::Binomial;
    mars.fit(&x, &y).unwrap();

    let probas = mars.predict_proba(&[vec![1.0], vec![5.0], vec![10.0]]).unwrap();
    assert!(probas[0] < 0.3, "p(x=1) should be low, got {}", probas[0]);
    assert!(probas[2] > 0.7, "p(x=10) should be high, got {}", probas[2]);

    let eq = mars.model.as_ref().unwrap().equation(None);
    assert!(eq.starts_with("logit(p) = "), "Binomial model equation should have logit prefix: got {}", eq);
}

#[test]
fn test_mars_pmethod_cv() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let xv = i as f64 * 0.2;
        x.push(vec![xv]);
        y.push(3.0 + 2.0 * (xv - 4.0).max(0.0));
    }

    let mut mars = MarsRegressor::new(1, 10, 2.0);
    mars.params.pmethod = MarsPruningMethod::Cv(5);
    mars.fit(&x, &y).unwrap();

    let model = mars.model.as_ref().unwrap();
    assert!(model.n_terms() >= 2);
    assert!(model.rss < 1.0);
}
