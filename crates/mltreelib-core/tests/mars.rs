use mltreelib_core::mars::MarsRegressor;

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
    println!("Fitted equation: {}", model.equation(None));
    println!("Fitted RSS: {}, GCV: {}, terms: {}", model.rss, model.gcv, model.n_terms());
    assert!(model.rss < 5.0, "Degree 2 MARS should fit interaction: RSS {}", model.rss);

    // Points where either factor is <= 7 should predict ~ 5.0
    let test_flat = mars.predict(&[vec![3.0, 3.0], vec![3.0, 10.0], vec![10.0, 3.0]]).unwrap();
    for &p in &test_flat {
        assert!((p - 5.0).abs() < 0.5, "Flat region expected ~5.0, got {}", p);
    }
    // Point at [10.0, 10.0] -> 5.0 + 2.0 * 3.0 * 3.0 = 23.0
    let test_high = mars.predict(&[vec![10.0, 10.0]]).unwrap();
    assert!((test_high[0] - 23.0).abs() < 1.0, "Expected ~23.0, got {}", test_high[0]);
}
