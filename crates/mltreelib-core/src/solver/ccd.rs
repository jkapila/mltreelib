//! Cyclic Coordinate Descent (CCD) Solver for Elastic Net, Lasso, Group Lasso, and Sparse-Group Lasso.
//!
//! Features:
//! - Exact $L_1$ soft-thresholding and $L_2$ Ridge shrinkage.
//! - Group soft-thresholding for Group Lasso and Sparse-Group Lasso (Simon et al. 2013).
//! - Deterministic canonical group ordering and group validation.
//! - Grouped monotone constraints shared across group members.
//! - Internal feature standardization ensuring equitable penalization.
//! - Active-set cycling strategy (glmnet style) for sparse high-dimensional problems.
//! - Monotone constraints (direction -1, 0, +1).
//! - Warm start support from parent nodes or path regularization.

use crate::solver::design::DesignMatrix;
use crate::solver::groups::{canonicalize_groups, validate_groups, GroupKind, GroupSpec};

/// Options controlling Coordinate Descent optimization.
#[derive(Clone, Debug)]
pub struct CcdOptions {
    pub l1: f64,
    pub l2: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub monotone: Vec<i8>,
    pub groups: Option<Vec<GroupSpec>>,
    pub group_l2: f64,
}

impl Default for CcdOptions {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 0.0,
            max_iter: 1000,
            tol: 1e-7,
            monotone: Vec::new(),
            groups: None,
            group_l2: 0.0,
        }
    }
}

/// Fits an ElasticNet model ($L_1 / L_2$) or Group / Sparse-Group Lasso using Cyclic Coordinate Descent with active set.
pub fn solve_ccd<D: DesignMatrix>(
    design: &D,
    options: &CcdOptions,
    warm_start: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    solve_ccd_report(design, options, warm_start).map(|(beta, _)| beta)
}

pub(super) fn solve_ccd_report<D: DesignMatrix>(
    design: &D,
    options: &CcdOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = design.n_rows();
    let p = design.n_features();
    if n == 0 || p == 0 {
        return Err("Empty design matrix".to_string());
    }

    // Check if group structure is specified
    let has_groups = options
        .groups
        .as_ref()
        .map_or(false, |g| !g.is_empty())
        || options.group_l2 > 0.0;

    if has_groups {
        return solve_ccd_grouped_report(design, options, warm_start);
    }

    // Scalar Coordinate Descent Path
    // 1. Compute column standardizations (intercept at index 0 unpenalized)
    let mut col_scales = vec![1.0; p];
    let mut col_norms = vec![0.0; p];
    for j in 0..p {
        let norm_sq = design.col_norm_sq(j, None);
        col_norms[j] = norm_sq;
        if j > 0 && options.l1 > 0.0 {
            let scale = (norm_sq / (n as f64).max(1.0)).sqrt();
            if scale > 1e-12 {
                col_scales[j] = scale;
            }
        }
    }

    // 2. Initialize coefficients & residuals
    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        // Start intercept at weighted mean
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            sum_y += w * design.get_y(i);
            sum_w += w;
        }
        if sum_w > 0.0 {
            beta[0] = sum_y / sum_w;
        }
    }

    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let y_hat = design.eval_row(i, &beta);
        residuals[i] = design.get_y(i) - y_hat;
    }

    // 3. Active set tracking & reusable buffer
    let mut active: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0 || j == 0).collect();
    let mut row_buf = vec![0.0; p];

    // 4. Main Active-Set Coordinate Descent Loop
    for iter in 0..options.max_iter {
        let mut max_active_delta = 0.0f64;

        // Inner loop: cycle over currently active set until convergence
        for _inner in 0..100 {
            let mut inner_delta = 0.0f64;
            for &j in &active {
                let diag = col_norms[j];
                if diag <= 0.0 {
                    continue;
                }
                let old_b = beta[j];
                let dot_res = design.col_dot_residuals(j, &residuals, None);
                let rho = dot_res + diag * old_b;

                let next_b = if j == 0 {
                    // Intercept is unpenalized
                    rho / diag
                } else {
                    let scale = col_scales[j];
                    let l1_pen = options.l1 * scale;
                    let l2_pen = options.l2 * scale * scale;
                    let denom = diag + l2_pen;
                    let raw = rho.signum() * (rho.abs() - l1_pen).max(0.0) / denom;

                    // Monotone constraint
                    let mono = options
                        .monotone
                        .get(j.saturating_sub(1))
                        .copied()
                        .unwrap_or(0);
                    match mono {
                        1 => raw.max(0.0),
                        -1 => raw.min(0.0),
                        _ => raw,
                    }
                };

                let delta = next_b - old_b;
                if delta.abs() > 1e-15 {
                    beta[j] = next_b;
                    // Update residuals in-place using preallocated row_buf
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= delta * row_buf[j];
                    }
                    inner_delta = inner_delta.max(delta.abs());
                }
            }
            max_active_delta = max_active_delta.max(inner_delta);
            if inner_delta < options.tol {
                break;
            }
        }

        // Outer sweep: check all features to see if new features enter the active set
        let mut new_active_added = false;
        for j in 0..p {
            if active.contains(&j) {
                continue;
            }
            let diag = col_norms[j];
            if diag <= 0.0 {
                continue;
            }
            let rho = design.col_dot_residuals(j, &residuals, None);
            let scale = col_scales[j];
            let l1_pen = options.l1 * scale;
            if rho.abs() > l1_pen {
                let l2_pen = options.l2 * scale * scale;
                let denom = diag + l2_pen;
                let raw = rho.signum() * (rho.abs() - l1_pen) / denom;
                let mono = options
                    .monotone
                    .get(j.saturating_sub(1))
                    .copied()
                    .unwrap_or(0);
                let next_b = match mono {
                    1 => raw.max(0.0),
                    -1 => raw.min(0.0),
                    _ => raw,
                };

                if next_b != 0.0 {
                    beta[j] = next_b;
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= next_b * row_buf[j];
                    }
                    active.push(j);
                    new_active_added = true;
                }
            }
        }

        if !new_active_added && max_active_delta < options.tol {
            return Ok((beta, super::Convergence::tolerance(iter + 1)));
        }
    }

    Ok((beta, super::Convergence::limit(options.max_iter)))
}

/// Group Lasso and Sparse-Group Lasso solver using active-set coordinate descent.
fn solve_ccd_grouped_report<D: DesignMatrix>(
    design: &D,
    options: &CcdOptions,
    warm_start: Option<&[f64]>,
) -> Result<(Vec<f64>, super::Convergence), String> {
    let n = design.n_rows();
    let p = design.n_features();

    let mut groups: Vec<GroupSpec> = options
        .groups
        .as_ref()
        .map(|g| {
            g.iter()
                .filter(|item| item.kind == GroupKind::CoefficientGroup)
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    validate_groups(&groups, p, true, Some(n)).map_err(|e| e.to_string())?;
    canonicalize_groups(&mut groups);

    // Compute column standardizations (intercept at index 0 unpenalized)
    let mut col_scales = vec![1.0; p];
    let mut col_norms = vec![0.0; p];
    for j in 0..p {
        let norm_sq = design.col_norm_sq(j, None);
        col_norms[j] = norm_sq;
        if j > 0 && (options.l1 > 0.0 || options.group_l2 > 0.0) {
            let scale = (norm_sq / (n as f64).max(1.0)).sqrt();
            if scale > 1e-12 {
                col_scales[j] = scale;
            }
        }
    }

    // Map column indices to group indices
    let mut col_to_group: Vec<Option<usize>> = vec![None; p];
    for (g_idx, g) in groups.iter().enumerate() {
        for &m in &g.members {
            col_to_group[m] = Some(g_idx);
        }
    }
    let ungrouped_cols: Vec<usize> = (1..p).filter(|&j| col_to_group[j].is_none()).collect();

    // Grouped monotone constraints: if any member has a constraint, the whole group shares it
    let mut effective_monotone = vec![0i8; p];
    for j in 1..p {
        if let Some(&m) = options.monotone.get(j.saturating_sub(1)) {
            effective_monotone[j] = m;
        }
    }
    for g in &groups {
        let mut grp_mono = 0i8;
        for &m in &g.members {
            if effective_monotone[m] != 0 {
                grp_mono = effective_monotone[m];
                break;
            }
        }
        if grp_mono != 0 {
            for &m in &g.members {
                effective_monotone[m] = grp_mono;
            }
        }
    }

    // Precompute group thresholds and Lipschitz constants for proximal block updates
    let mut row_buf = vec![0.0; p];
    let mut group_thresholds = Vec::with_capacity(groups.len());
    let mut group_lipschitz = Vec::with_capacity(groups.len());

    for g in &groups {
        let p_g = g.members.len();
        let w_g = g.effective_weight();
        let avg_scale = g.members.iter().map(|&j| col_scales[j]).sum::<f64>()
            / (p_g as f64).max(1.0);
        group_thresholds.push(options.group_l2 * w_g * avg_scale);

        let ridge = options.l2 * avg_scale * avg_scale;
        if p_g == 1 {
            let j = g.members[0];
            group_lipschitz.push(col_norms[j] + ridge);
        } else {
            // Power iteration for spectral norm of X_g^T X_g
            let mut q = vec![1.0 / (p_g as f64).sqrt(); p_g];
            let mut l_g = 1.0;
            for _ in 0..10 {
                let mut v = vec![0.0; p_g];
                for i in 0..n {
                    design.copy_design_row(i, &mut row_buf);
                    let mut dot = 0.0;
                    for (idx, &m) in g.members.iter().enumerate() {
                        dot += row_buf[m] * q[idx];
                    }
                    let w = design.get_weight(i);
                    for (idx, &m) in g.members.iter().enumerate() {
                        v[idx] += w * row_buf[m] * dot;
                    }
                }
                let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for idx in 0..p_g {
                        q[idx] = v[idx] / norm;
                    }
                    l_g = norm;
                } else {
                    break;
                }
            }
            group_lipschitz.push(l_g.max(1e-10) + ridge);
        }
    }

    // Initialize coefficients & residuals
    let mut beta = vec![0.0; p];
    if let Some(warm) = warm_start {
        let copy_len = p.min(warm.len());
        beta[..copy_len].copy_from_slice(&warm[..copy_len]);
    } else {
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let w = design.get_weight(i);
            sum_y += w * design.get_y(i);
            sum_w += w;
        }
        if sum_w > 0.0 {
            beta[0] = sum_y / sum_w;
        }
    }

    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let y_hat = design.eval_row(i, &beta);
        residuals[i] = design.get_y(i) - y_hat;
    }

    // Active sets: active groups and active ungrouped features
    let mut active_groups: Vec<usize> = (0..groups.len())
        .filter(|&g| groups[g].members.iter().any(|&m| beta[m] != 0.0))
        .collect();
    let mut active_ungrouped: Vec<usize> = ungrouped_cols
        .iter()
        .copied()
        .filter(|&j| beta[j] != 0.0)
        .collect();

    let mut row_buf = vec![0.0; p];

    // Main Coordinate Descent Loop
    for iter in 0..options.max_iter {
        let mut max_active_delta = 0.0f64;

        // Inner loop: cycle over active set
        for _inner in 0..100 {
            let mut inner_delta = 0.0f64;

            // 1. Intercept update (unpenalized)
            let diag0 = col_norms[0];
            if diag0 > 0.0 {
                let old_b0 = beta[0];
                let dot_res0 = design.col_dot_residuals(0, &residuals, None);
                let rho0 = dot_res0 + diag0 * old_b0;
                let next_b0 = rho0 / diag0;
                let delta0 = next_b0 - old_b0;
                if delta0.abs() > 1e-15 {
                    beta[0] = next_b0;
                    for i in 0..n {
                        residuals[i] -= delta0;
                    }
                    inner_delta = inner_delta.max(delta0.abs());
                }
            }

            // 2. Active groups update (Proximal block step with Sparse-Group Lasso thresholding)
            for &g_idx in &active_groups {
                let members = &groups[g_idx].members;
                let l_g = group_lipschitz[g_idx];
                let tau_step = group_thresholds[g_idx] / l_g;

                let mut u_vec = Vec::with_capacity(members.len());
                for &j in members {
                    let dot_res = design.col_dot_residuals(j, &residuals, None);
                    let scale = col_scales[j];
                    let ridge = options.l2 * scale * scale;
                    let u_j = beta[j] + (dot_res - ridge * beta[j]) / l_g;
                    let l1_step = (options.l1 * scale) / l_g;
                    let raw_v = u_j.signum() * (u_j.abs() - l1_step).max(0.0);
                    let mono = effective_monotone[j];
                    let v_j = match mono {
                        1 => raw_v.max(0.0),
                        -1 => raw_v.min(0.0),
                        _ => raw_v,
                    };
                    u_vec.push(v_j);
                }

                let v_norm = u_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let mut next_b = vec![0.0; members.len()];
                if v_norm > tau_step && v_norm > 1e-15 {
                    let factor = if tau_step > 0.0 {
                        1.0 - tau_step / v_norm
                    } else {
                        1.0
                    };
                    for (idx, &v) in u_vec.iter().enumerate() {
                        next_b[idx] = factor * v;
                    }
                }

                for (idx, &j) in members.iter().enumerate() {
                    let new_val = next_b[idx];
                    let old_val = beta[j];
                    let delta = new_val - old_val;
                    if delta.abs() > 1e-15 {
                        beta[j] = new_val;
                        for i in 0..n {
                            design.copy_design_row(i, &mut row_buf);
                            residuals[i] -= delta * row_buf[j];
                        }
                        inner_delta = inner_delta.max(delta.abs());
                    }
                }
            }

            // 3. Active ungrouped features update
            for &j in &active_ungrouped {
                let diag = col_norms[j];
                if diag <= 0.0 {
                    continue;
                }
                let old_b = beta[j];
                let dot_res = design.col_dot_residuals(j, &residuals, None);
                let rho = dot_res + diag * old_b;
                let scale = col_scales[j];
                let l1_pen = options.l1 * scale;
                let l2_pen = options.l2 * scale * scale;
                let denom = diag + l2_pen;
                let raw = rho.signum() * (rho.abs() - l1_pen).max(0.0) / denom;
                let mono = effective_monotone[j];
                let next_b = match mono {
                    1 => raw.max(0.0),
                    -1 => raw.min(0.0),
                    _ => raw,
                };
                let delta = next_b - old_b;
                if delta.abs() > 1e-15 {
                    beta[j] = next_b;
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= delta * row_buf[j];
                    }
                    inner_delta = inner_delta.max(delta.abs());
                }
            }

            // Prune inactive members from active sets
            active_groups.retain(|&g| groups[g].members.iter().any(|&m| beta[m] != 0.0));
            active_ungrouped.retain(|&j| beta[j] != 0.0);

            max_active_delta = max_active_delta.max(inner_delta);
            if inner_delta < options.tol {
                break;
            }
        }

        // Outer sweep: check all inactive groups and features
        let mut new_active_added = false;

        // Inactive groups check
        for g_idx in 0..groups.len() {
            if active_groups.contains(&g_idx) {
                continue;
            }
            let members = &groups[g_idx].members;
            let l_g = group_lipschitz[g_idx];
            let tau_step = group_thresholds[g_idx] / l_g;

            let mut u_vec = Vec::with_capacity(members.len());
            for &j in members {
                let dot_res = design.col_dot_residuals(j, &residuals, None);
                let scale = col_scales[j];
                let u_j = dot_res / l_g;
                let l1_step = (options.l1 * scale) / l_g;
                let raw_v = u_j.signum() * (u_j.abs() - l1_step).max(0.0);
                let mono = effective_monotone[j];
                let v_j = match mono {
                    1 => raw_v.max(0.0),
                    -1 => raw_v.min(0.0),
                    _ => raw_v,
                };
                u_vec.push(v_j);
            }

            let v_norm = u_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if v_norm > tau_step && v_norm > 1e-15 {
                let factor = if tau_step > 0.0 {
                    1.0 - tau_step / v_norm
                } else {
                    1.0
                };
                for (idx, &j) in members.iter().enumerate() {
                    let val = factor * u_vec[idx];
                    beta[j] = val;
                    if val != 0.0 {
                        for i in 0..n {
                            design.copy_design_row(i, &mut row_buf);
                            residuals[i] -= val * row_buf[j];
                        }
                    }
                }
                active_groups.push(g_idx);
                new_active_added = true;
            }
        }

        // Inactive ungrouped features check
        for &j in &ungrouped_cols {
            if active_ungrouped.contains(&j) {
                continue;
            }
            let diag = col_norms[j];
            if diag <= 0.0 {
                continue;
            }
            let rho = design.col_dot_residuals(j, &residuals, None);
            let scale = col_scales[j];
            let l1_pen = options.l1 * scale;
            if rho.abs() > l1_pen {
                let l2_pen = options.l2 * scale * scale;
                let denom = diag + l2_pen;
                let raw = rho.signum() * (rho.abs() - l1_pen) / denom;
                let mono = effective_monotone[j];
                let next_b = match mono {
                    1 => raw.max(0.0),
                    -1 => raw.min(0.0),
                    _ => raw,
                };
                if next_b != 0.0 {
                    beta[j] = next_b;
                    for i in 0..n {
                        design.copy_design_row(i, &mut row_buf);
                        residuals[i] -= next_b * row_buf[j];
                    }
                    active_ungrouped.push(j);
                    new_active_added = true;
                }
            }
        }

        if !new_active_added && max_active_delta < options.tol {
            return Ok((beta, super::Convergence::tolerance(iter + 1)));
        }
    }

    Ok((beta, super::Convergence::limit(options.max_iter)))
}
