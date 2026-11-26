//! Sparse Rule Ensemble Selection (L1 / ElasticNet Coordinate Descent).
//!
//! Fits a sparse linear model over candidate rule activations to select a minimal,
//! non-redundant rule ensemble with exact rule importances.

/// Output of sparse rule ensemble selection.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRuleResult {
    /// Indices of selected rules in the input rule matrix.
    pub selected_indices: Vec<usize>,
    /// Un-standardized linear model coefficients for the selected rules.
    pub coefficients: Vec<f64>,
    /// Intercept term of the linear model.
    pub intercept: f64,
    /// Rule importances: I_m = |w_m| = |c_m| * sqrt(s_m * (1 - s_m)).
    pub importances: Vec<f64>,
    /// Sample supports (frequencies) of the selected rules.
    pub supports: Vec<f64>,
}

/// Fit a sparse rule ensemble selection model via L1/ElasticNet coordinate descent.
///
/// Parameters:
/// - `rule_matrix`: N rows x M rule activation columns (typically values in {0.0, 1.0}).
/// - `y`: N target responses.
/// - `alpha`: ElasticNet mixing parameter (1.0 = pure Lasso, 0.0 = pure Ridge).
/// - `lambda`: Regularization strength penalty.
/// - `max_iter`: Maximum coordinate descent iterations.
/// - `tol`: Convergence tolerance on maximum coordinate change.
pub fn fit_sparse_rule_selection(
    rule_matrix: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<SparseRuleResult, String> {
    if rule_matrix.is_empty() || y.is_empty() {
        return Err("rule_matrix and y cannot be empty".into());
    }
    let n = rule_matrix.len();
    if y.len() != n {
        return Err(format!("mismatched sample counts: rules has {}, y has {}", n, y.len()));
    }
    let m_total = rule_matrix[0].len();
    if m_total == 0 {
        return Err("rule_matrix must have at least 1 column".into());
    }
    for (i, row) in rule_matrix.iter().enumerate() {
        if row.len() != m_total {
            return Err(format!("row {} has {} columns, expected {}", i, row.len(), m_total));
        }
    }

    let n_f = n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n_f;

    // 1. Filter degenerate rules and compute column supports
    let mut valid_indices = Vec::new();
    let mut supports = Vec::new();
    let mut sigmas = Vec::new();

    for j in 0..m_total {
        let count = rule_matrix.iter().filter(|row| row[j] > 0.5).count();
        let s = count as f64 / n_f;
        if (0.005..=0.995).contains(&s) {
            let sigma = (s * (1.0 - s)).sqrt().max(1e-6);
            valid_indices.push(j);
            supports.push(s);
            sigmas.push(sigma);
        }
    }

    let p = valid_indices.len();
    if p == 0 {
        return Ok(SparseRuleResult {
            selected_indices: Vec::new(),
            coefficients: Vec::new(),
            intercept: y_mean,
            importances: Vec::new(),
            supports: Vec::new(),
        });
    }

    // 2. Standardize valid rule features: Z[i][j] = (R[i][j] - s_j) / sigma_j
    let mut z_matrix = vec![vec![0.0f64; p]; n];
    for (col_idx, &orig_j) in valid_indices.iter().enumerate() {
        let s = supports[col_idx];
        let sigma = sigmas[col_idx];
        for i in 0..n {
            z_matrix[i][col_idx] = (rule_matrix[i][orig_j] - s) / sigma;
        }
    }

    // Precompute Z column squared norms / n
    let mut z_norms_sq = vec![0.0f64; p];
    for col_idx in 0..p {
        let sum_sq: f64 = z_matrix.iter().map(|row| row[col_idx] * row[col_idx]).sum();
        z_norms_sq[col_idx] = sum_sq / n_f;
    }

    // 3. Centered targets and coordinate descent state
    let mut residuals: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let mut w = vec![0.0f64; p];

    let l1_penalty = lambda * alpha;
    let l2_denom_term = lambda * (1.0 - alpha);

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        for col_idx in 0..p {
            if z_norms_sq[col_idx] < 1e-12 {
                continue;
            }
            let old_w = w[col_idx];

            let mut dot = 0.0f64;
            for i in 0..n {
                dot += z_matrix[i][col_idx] * residuals[i];
            }
            let rho = (dot / n_f) + old_w * z_norms_sq[col_idx];

            let new_w = if rho > l1_penalty {
                (rho - l1_penalty) / (z_norms_sq[col_idx] + l2_denom_term)
            } else if rho < -l1_penalty {
                (rho + l1_penalty) / (z_norms_sq[col_idx] + l2_denom_term)
            } else {
                0.0
            };

            let delta = new_w - old_w;
            if delta.abs() > 0.0 {
                for i in 0..n {
                    residuals[i] -= delta * z_matrix[i][col_idx];
                }
                max_change = max_change.max(delta.abs());
                w[col_idx] = new_w;
            }
        }

        if max_change < tol {
            break;
        }
    }

    // 4. Recover raw un-standardized coefficients and intercept
    let mut selected_indices = Vec::new();
    let mut coefficients = Vec::new();
    let mut importances = Vec::new();
    let mut sel_supports = Vec::new();
    let mut intercept = y_mean;

    for (col_idx, &orig_j) in valid_indices.iter().enumerate() {
        if w[col_idx].abs() > 1e-8 {
            let c_m = w[col_idx] / sigmas[col_idx];
            let importance = w[col_idx].abs();
            intercept -= c_m * supports[col_idx];

            selected_indices.push(orig_j);
            coefficients.push(c_m);
            importances.push(importance);
            sel_supports.push(supports[col_idx]);
        }
    }

    Ok(SparseRuleResult {
        selected_indices,
        coefficients,
        intercept,
        importances,
        supports: sel_supports,
    })
}

/// Compute predictions using sparse selected rules.
pub fn predict_sparse_rules(
    rule_matrix: &[Vec<f64>],
    intercept: f64,
    selected_indices: &[usize],
    coefficients: &[f64],
) -> Vec<f64> {
    rule_matrix
        .iter()
        .map(|row| {
            let mut pred = intercept;
            for (&idx, &coef) in selected_indices.iter().zip(coefficients.iter()) {
                if idx < row.len() && row[idx] > 0.5 {
                    pred += coef;
                }
            }
            pred
        })
        .collect()
}
