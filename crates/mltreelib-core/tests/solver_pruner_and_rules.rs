use mltreelib_core::solver::{
    fit_sparse_rule_selection, predict_sparse_rules, prune_ensemble_stepwise,
};

#[test]
fn test_ensemble_prune_stepwise() {
    // Ground truth y = 2.0 * x0 + 1.0
    // Create an ensemble of 10 trees:
    // Tree 0: accurate
    // Tree 1: slightly noisy
    // Trees 2..9: garbage / pure noise
    let n = 100;
    let mut preds = vec![vec![0.0f64; 10]; n];
    let mut y = vec![0.0f64; n];

    for i in 0..n {
        let x = i as f64 / 10.0;
        let true_y = 2.0 * x + 1.0;
        y[i] = true_y;

        preds[i][0] = true_y + 0.05 * (i % 3) as f64; // High signal
        preds[i][1] = true_y - 0.05 * (i % 2) as f64; // High signal
        for t in 2..10 {
            preds[i][t] = (i * t) as f64 % 50.0 - 25.0; // Noise
        }
    }

    let result = prune_ensemble_stepwise(&preds, &y, 5, 1e-4).unwrap();

    assert!(!result.selected_indices.is_empty());
    assert!(result.selected_indices.len() <= 5);
    // Tree 0 or Tree 1 must be the top selected trees
    assert!(
        result.selected_indices.contains(&0) || result.selected_indices.contains(&1),
        "Pruner must pick high-signal trees"
    );
    // Noise trees should not dominate weights
    let good_weight: f64 = result
        .selected_indices
        .iter()
        .zip(result.weights.iter())
        .filter(|(&idx, _)| idx == 0 || idx == 1)
        .map(|(_, &w)| w)
        .sum();
    assert!(good_weight > 0.8, "Good trees must account for >80% weight, got {}", good_weight);
    assert!(result.final_loss < 0.05, "Pruned ensemble loss too high: {}", result.final_loss);
}

#[test]
fn test_sparse_rule_selection_l1() {
    // Generate synthetic rule activations
    // N = 200 samples, M = 8 candidate rules
    // True signal: y = 3.0 + 4.0 * Rule_0 - 2.5 * Rule_2
    let n = 200;
    let mut rules = vec![vec![0.0f64; 8]; n];
    let mut y = vec![0.0f64; n];

    for i in 0..n {
        let r0 = if i % 2 == 0 { 1.0 } else { 0.0 };
        let r1 = if i % 3 == 0 { 1.0 } else { 0.0 };
        let r2 = if i % 4 == 0 { 1.0 } else { 0.0 };
        let r3 = if i % 5 == 0 { 1.0 } else { 0.0 };
        for j in 4..8 {
            rules[i][j] = if (i + j) % 3 == 0 { 1.0 } else { 0.0 };
        }
        rules[i][0] = r0;
        rules[i][1] = r1;
        rules[i][2] = r2;
        rules[i][3] = r3;

        y[i] = 3.0 + 4.0 * r0 - 2.5 * r2;
    }

    let result = fit_sparse_rule_selection(&rules, &y, 1.0, 0.01, 300, 1e-5).unwrap();

    assert!(result.selected_indices.contains(&0), "Must select Rule 0");
    assert!(result.selected_indices.contains(&2), "Must select Rule 2");

    // Predict on training data and verify low residual
    let y_hat = predict_sparse_rules(
        &rules,
        result.intercept,
        &result.selected_indices,
        &result.coefficients,
    );
    assert_eq!(y_hat.len(), n);

    let mse: f64 = y_hat.iter().zip(y.iter()).map(|(&yh, &yi)| (yh - yi).powi(2)).sum::<f64>() / n as f64;
    assert!(mse < 0.05, "Sparse rule model MSE too high: {}", mse);
}
