//! Sparse Ensemble Pruning (Caruana Stepwise Selection & 0-1 Subset Knapsack).
//!
//! Compresses an ensemble of M trees (e.g., 200–500 trees) down to K << M trees
//! with non-negative weights by greedily optimizing out-of-bag validation loss.

use std::collections::HashMap;

/// Output of sparse forest pruning.
#[derive(Clone, Debug, PartialEq)]
pub struct PruningResult {
    /// Indices of selected trees in the original ensemble.
    pub selected_indices: Vec<usize>,
    /// Normalized voting weights for the selected trees (sum to 1.0).
    pub weights: Vec<f64>,
    /// Validation loss progression as each tree was added.
    pub loss_history: Vec<f64>,
    /// Final validation loss of the pruned sub-ensemble.
    pub final_loss: f64,
}

/// Caruana greedy forward stepwise selection with replacement.
///
/// Given a matrix of predictions `preds[sample_i][tree_m]` and targets `y[sample_i]`:
/// Selects up to `max_trees` trees that minimize mean squared error.
pub fn prune_ensemble_stepwise(
    preds: &[Vec<f64>], // N rows x M trees
    y: &[f64],          // N rows
    max_trees: usize,   // e.g. 20
    tol: f64,           // e.g. 1e-5
) -> Result<PruningResult, String> {
    if preds.is_empty() || y.is_empty() {
        return Err("preds and y cannot be empty".into());
    }
    let n = preds.len();
    if y.len() != n {
        return Err(format!("mismatched sample counts: preds has {}, y has {}", n, y.len()));
    }
    let m = preds[0].len();
    if m == 0 {
        return Err("ensemble must contain at least 1 tree".into());
    }
    for (i, row) in preds.iter().enumerate() {
        if row.len() != m {
            return Err(format!("row {} has {} tree predictions, expected {}", i, row.len(), m));
        }
    }

    let mut selected_history: Vec<usize> = Vec::new();
    let mut sum_preds = vec![0.0f64; n];
    let mut loss_history = Vec::with_capacity(max_trees);
    let mut prev_loss = f64::INFINITY;

    for step in 1..=max_trees {
        let step_f = step as f64;
        let mut best_tree = 0;
        let mut best_loss = f64::INFINITY;

        for tree_idx in 0..m {
            let mut loss = 0.0f64;
            for i in 0..n {
                let trial_pred = (sum_preds[i] + preds[i][tree_idx]) / step_f;
                let diff = trial_pred - y[i];
                loss += diff * diff;
            }
            loss /= n as f64;

            if loss < best_loss {
                best_loss = loss;
                best_tree = tree_idx;
            }
        }

        if step > 1 && (prev_loss - best_loss) < tol {
            break;
        }

        for i in 0..n {
            sum_preds[i] += preds[i][best_tree];
        }
        selected_history.push(best_tree);
        loss_history.push(best_loss);
        prev_loss = best_loss;
    }

    if selected_history.is_empty() {
        let mut best_tree = 0;
        let mut best_loss = f64::INFINITY;
        for tree_idx in 0..m {
            let loss: f64 = preds
                .iter()
                .zip(y.iter())
                .map(|(p, &yi)| (p[tree_idx] - yi).powi(2))
                .sum::<f64>()
                / n as f64;
            if loss < best_loss {
                best_loss = loss;
                best_tree = tree_idx;
            }
        }
        return Ok(PruningResult {
            selected_indices: vec![best_tree],
            weights: vec![1.0],
            loss_history: vec![best_loss],
            final_loss: best_loss,
        });
    }

    let total_steps = selected_history.len() as f64;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &idx in &selected_history {
        *counts.entry(idx).or_insert(0) += 1;
    }

    let mut unique_pairs: Vec<(usize, f64)> = counts
        .into_iter()
        .map(|(idx, cnt)| (idx, cnt as f64 / total_steps))
        .collect();
    unique_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let selected_indices: Vec<usize> = unique_pairs.iter().map(|&(idx, _)| idx).collect();
    let weights: Vec<f64> = unique_pairs.iter().map(|&(_, w)| w).collect();

    Ok(PruningResult {
        selected_indices,
        weights,
        loss_history,
        final_loss: prev_loss,
    })
}
