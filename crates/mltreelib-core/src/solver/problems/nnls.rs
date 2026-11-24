//! Non-Negative Least Squares (NNLS) solver using the Lawson-Hanson active-set algorithm
//! with Cholesky decomposition (Spec 27 Phase 5).

use crate::solver::design::DesignMatrix;

/// Result of an NNLS fit.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NnlsFitResult {
    /// Fitted coefficient vector (slopes $\ge 0$; intercept unconstrained unless configured).
    pub coefficients: Vec<f64>,
    /// Indices of variables in the final passive (non-zero positive) set.
    pub active_set: Vec<usize>,
    /// Final penalized squared-error loss.
    pub loss: f64,
    /// Number of active-set outer iterations performed.
    pub iterations: usize,
}

/// Solves a symmetric positive-definite linear system $A x = b$ using Cholesky decomposition ($A = L L^\top$).
/// Returns `None` if matrix is singular or non-positive definite.
pub fn solve_cholesky(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let k = b.len();
    if k == 0 || a.len() != k {
        return None;
    }
    let mut l = vec![vec![0.0; k]; k];

    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in 0..j {
                sum += l[i][p] * l[j][p];
            }
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 1e-14 {
                    // Slight ridge regularization fallback for near-singular subproblem
                    let regularized_diag = diag.max(1e-10);
                    l[i][j] = regularized_diag.sqrt();
                } else {
                    l[i][j] = diag.sqrt();
                }
            } else {
                let denom = l[j][j];
                if denom.abs() <= 1e-14 {
                    return None;
                }
                l[i][j] = (a[i][j] - sum) / denom;
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        let denom = l[i][i];
        if denom.abs() <= 1e-14 {
            return None;
        }
        y[i] = (b[i] - sum) / denom;
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..k {
            sum += l[j][i] * x[j];
        }
        let denom = l[i][i];
        if denom.abs() <= 1e-14 {
            return None;
        }
        x[i] = (y[i] - sum) / denom;
    }

    Some(x)
}

/// Solves Non-Negative Least Squares (NNLS) over a [`DesignMatrix`] via the Lawson-Hanson active-set algorithm:
///
/// $$\min_\beta \frac{1}{2}\sum_{i=1}^n w_i (y_i - \mathbf{x}_i^\top \beta)^2 + \frac{\lambda}{2}\sum_{j \in \text{slopes}} \beta_j^2 \quad \text{s.t.} \quad \beta_j \ge 0 \ (\forall j \in C)$$
///
/// Where $C = \{1, \dots, p-1\}$ if `non_negative_intercept` is false (default, unconstrained intercept),
/// or $C = \{0, \dots, p-1\}$ if `non_negative_intercept` is true.
pub fn solve_nnls<D: DesignMatrix>(
    design: &D,
    l2: f64,
    non_negative_intercept: bool,
    max_iter: usize,
    tol: f64,
) -> Result<NnlsFitResult, String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("empty design matrix for NNLS".into());
    }

    let tol = if tol > 0.0 { tol } else { 1e-8 };
    let max_iter = if max_iter > 0 { max_iter } else { (3 * p).max(30) };

    // 1. Accumulate Gram matrix G = X^T W X + lambda * I and vector c = X^T W y
    let mut g = vec![vec![0.0; p]; p];
    let mut c = vec![0.0; p];
    let mut sum_w_y_sq = 0.0;
    let mut row_buf = vec![0.0; p];

    for i in 0..n {
        let w = design.get_weight(i);
        if w > 0.0 {
            let y_i = design.get_y(i);
            sum_w_y_sq += w * y_i * y_i;
            design.copy_design_row(i, &mut row_buf);
            for j in 0..p {
                let w_xj = w * row_buf[j];
                c[j] += w_xj * y_i;
                for k in 0..=j {
                    let val = w_xj * row_buf[k];
                    g[j][k] += val;
                    if j != k {
                        g[k][j] += val;
                    }
                }
            }
        }
    }

    // Add L2 penalty to slopes (and intercept if non_negative_intercept)
    if l2 > 0.0 {
        let start_j = if non_negative_intercept { 0 } else { 1 };
        for j in start_j..p {
            g[j][j] += l2;
        }
    }

    // Small numerical jitter on diagonal for stability
    for j in 0..p {
        g[j][j] += 1e-12;
    }

    // Passive set P (free to be > 0), Active set Z (constrained to 0)
    let mut p_set: Vec<usize> = Vec::with_capacity(p);
    let mut z_set: Vec<usize> = Vec::with_capacity(p);
    let mut beta = vec![0.0; p];

    // If intercept is unconstrained, start with intercept in P
    if !non_negative_intercept {
        p_set.push(0);
        for j in 1..p {
            z_set.push(j);
        }
        if g[0][0] > 1e-14 {
            beta[0] = c[0] / g[0][0];
        }
    } else {
        for j in 0..p {
            z_set.push(j);
        }
    }

    let mut outer_iter = 0;

    while outer_iter < max_iter {
        outer_iter += 1;

        // Compute dual vector w = c - G * beta
        let mut dual = vec![0.0; p];
        for j in 0..p {
            let mut g_beta_j = 0.0;
            for k in 0..p {
                g_beta_j += g[j][k] * beta[k];
            }
            dual[j] = c[j] - g_beta_j;
        }

        // Check optimality on constrained active variables Z
        let mut max_dual = -f64::INFINITY;
        let mut best_z_idx = None;

        for (idx_in_z, &col) in z_set.iter().enumerate() {
            if dual[col] > max_dual {
                max_dual = dual[col];
                best_z_idx = Some((idx_in_z, col));
            }
        }

        // KKT condition: all dual values in Z are <= tol
        if max_dual <= tol || best_z_idx.is_none() {
            break;
        }

        // Move variable with largest positive dual gradient from Z to P
        let (idx_in_z, col_to_add) = best_z_idx.unwrap();
        z_set.swap_remove(idx_in_z);
        p_set.push(col_to_add);

        // Inner loop: solve unconstrained subproblem on P until all constrained members in P are > 0
        let mut inner_iter = 0;
        let max_inner = 3 * p;

        while inner_iter < max_inner {
            inner_iter += 1;
            let k = p_set.len();
            if k == 0 {
                break;
            }

            // Extract G_P,P and c_P
            let mut sub_g = vec![vec![0.0; k]; k];
            let mut sub_c = vec![0.0; k];
            for (i_sub, &row_col) in p_set.iter().enumerate() {
                sub_c[i_sub] = c[row_col];
                for (j_sub, &col_col) in p_set.iter().enumerate() {
                    sub_g[i_sub][j_sub] = g[row_col][col_col];
                }
            }

            let sub_sol = solve_cholesky(&sub_g, &sub_c).ok_or_else(|| {
                "singular subproblem in Lawson-Hanson NNLS Cholesky solve".to_string()
            })?;

            let mut z = vec![0.0; p];
            for (i_sub, &col) in p_set.iter().enumerate() {
                z[col] = sub_sol[i_sub];
            }

            // Check if all constrained variables in P are strictly positive
            let mut all_positive = true;
            for &col in &p_set {
                let is_constrained = non_negative_intercept || col > 0;
                if is_constrained && z[col] <= 0.0 {
                    all_positive = false;
                    break;
                }
            }

            if all_positive {
                for &col in &p_set {
                    beta[col] = z[col];
                }
                break;
            }

            // Find maximum step alpha in [0, 1] before any constrained variable hits zero
            let mut alpha = 1.0f64;
            for &col in &p_set {
                let is_constrained = non_negative_intercept || col > 0;
                if is_constrained && z[col] <= 0.0 {
                    let denom = beta[col] - z[col];
                    if denom.abs() > 1e-14 {
                        let ratio = beta[col] / denom;
                        if ratio < alpha {
                            alpha = ratio;
                        }
                    }
                }
            }

            let alpha = alpha.clamp(0.0, 1.0);
            for &col in &p_set {
                beta[col] += alpha * (z[col] - beta[col]);
            }

            // Move constrained variables in P with beta[col] <= 1e-14 back to Z
            let mut to_remove = Vec::new();
            for (i_sub, &col) in p_set.iter().enumerate() {
                let is_constrained = non_negative_intercept || col > 0;
                if is_constrained && beta[col] <= 1e-14 {
                    beta[col] = 0.0;
                    to_remove.push(i_sub);
                }
            }

            to_remove.sort_unstable_by(|a, b| b.cmp(a));
            for i_sub in to_remove {
                let col = p_set.swap_remove(i_sub);
                z_set.push(col);
            }
        }
    }

    // Final clean-up: clamp any negative constrained values to zero
    for j in 0..p {
        let is_constrained = non_negative_intercept || j > 0;
        if is_constrained && beta[j] < 0.0 {
            beta[j] = 0.0;
        }
    }

    // Active set: all non-zero coefficients
    let mut active_set = Vec::new();
    for j in 0..p {
        if beta[j].abs() > 1e-12 {
            active_set.push(j);
        }
    }

    // Compute loss = 0.5 * (beta^T G beta - 2 c^T beta + y^T W y)
    let mut g_beta = vec![0.0; p];
    for j in 0..p {
        for k in 0..p {
            g_beta[j] += g[j][k] * beta[k];
        }
    }
    let beta_g_beta: f64 = beta.iter().zip(&g_beta).map(|(b, gb)| b * gb).sum();
    let c_beta: f64 = c.iter().zip(&beta).map(|(ci, b)| ci * b).sum();
    let loss = (0.5 * (beta_g_beta - 2.0 * c_beta + sum_w_y_sq)).max(0.0);

    Ok(NnlsFitResult {
        coefficients: beta,
        active_set,
        loss,
        iterations: outer_iter,
    })
}
