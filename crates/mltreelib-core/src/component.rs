//! Consolidated Component Analysis, Matrix Factorization, and Reconstruction Anomaly Scoring.
//!
//! Provides native Rust implementations of:
//! - Classical PCA (centered and standardized) with R-aligned factor rotations
//!   (Varimax, Quartimax, Equamax, Parsimax, Biquartimax, Orthomax, Promax)
//! - Truncated SVD (uncentered dimensionality reduction)
//! - Whitening (ZCA and PCA whitening transforms)
//! - FastICA (independent component analysis with symmetric orthogonalization)
//! - NMF (non-negative matrix factorization via Multiplicative Updates,
//!   Projected Alternating Least Squares (ALS), and Hierarchical ALS (HALS)
//!   with L1/L2 penalties)
//! - Reconstruction-based Anomaly Scoring for PCA and NMF

use std::collections::BTreeMap;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::report::{
    ControllerSection, DiagnosticsSection, LearnerSection, ModelReport,
    ObjectiveSection, OptimizationDirection,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rows_to_matrix(data: &[Vec<f64>]) -> Result<DMatrix<f64>, String> {
    let n = data.len();
    if n == 0 {
        return Err("data must not be empty".into());
    }
    let p = data[0].len();
    if p == 0 {
        return Err("feature dimension must be > 0".into());
    }
    for row in data {
        if row.len() != p {
            return Err(format!("inconsistent row length: expected {}, got {}", p, row.len()));
        }
    }
    let mut mat = DMatrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            let val = data[i][j];
            if !val.is_finite() {
                return Err(format!("non-finite value at row {}, column {}", i, j));
            }
            mat[(i, j)] = val;
        }
    }
    Ok(mat)
}

fn matrix_to_rows(mat: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let (n, p) = (mat.nrows(), mat.ncols());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        for j in 0..p {
            row.push(mat[(i, j)]);
        }
        out.push(row);
    }
    out
}

// ---------------------------------------------------------------------------
// Factor / PCA Rotation Methods (R-aligned)
// ---------------------------------------------------------------------------

/// Rotation methods for PCA loadings (aligned with R's `stats` and `GPArotation`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RotationMethod {
    None,
    Varimax,
    Quartimax,
    Equamax,
    Parsimax,
    Biquartimax,
    Orthomax(f64),
    Promax(f64), // power m (default 4.0 in R)
}

impl RotationMethod {
    pub fn parse(s: &str, m: Option<f64>) -> Result<Self, String> {
        match s.to_lowercase().trim() {
            "none" | "" => Ok(Self::None),
            "varimax" => Ok(Self::Varimax),
            "quartimax" => Ok(Self::Quartimax),
            "equamax" => Ok(Self::Equamax),
            "parsimax" => Ok(Self::Parsimax),
            "biquartimax" => Ok(Self::Biquartimax),
            "promax" => Ok(Self::Promax(m.unwrap_or(4.0))),
            other => {
                if let Some(rest) = other.strip_prefix("orthomax:") {
                    let gamma = rest.parse::<f64>().map_err(|_| "invalid orthomax gamma")?;
                    Ok(Self::Orthomax(gamma))
                } else {
                    Err(format!("unknown rotation method: {}", other))
                }
            }
        }
    }

    pub fn is_oblique(&self) -> bool {
        matches!(self, Self::Promax(_))
    }
}

/// Rotates a loadings matrix L of shape (p, k) using the specified method.
/// Returns (rotated_loadings, rotation_matrix T).
pub fn rotate_loadings(
    loadings: &DMatrix<f64>,
    method: RotationMethod,
    normalize: bool,
    max_iter: usize,
    eps: f64,
) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    let (p, k) = (loadings.nrows(), loadings.ncols());
    if k < 2 || method == RotationMethod::None {
        return Ok((loadings.clone(), DMatrix::identity(k, k)));
    }

    match method {
        RotationMethod::None => Ok((loadings.clone(), DMatrix::identity(k, k))),
        RotationMethod::Varimax
        | RotationMethod::Quartimax
        | RotationMethod::Equamax
        | RotationMethod::Parsimax
        | RotationMethod::Biquartimax
        | RotationMethod::Orthomax(_) => {
            let gamma = match method {
                RotationMethod::Varimax => 1.0,
                RotationMethod::Quartimax => 0.0,
                RotationMethod::Equamax => (k as f64) / 2.0,
                RotationMethod::Parsimax => {
                    let pf = p as f64;
                    let kf = k as f64;
                    (pf * (kf - 1.0)) / (pf + kf - 2.0)
                }
                RotationMethod::Biquartimax => 0.5,
                RotationMethod::Orthomax(g) => g,
                _ => 1.0,
            };
            rotate_orthomax(loadings, gamma, normalize, max_iter, eps)
        }
        RotationMethod::Promax(m) => {
            rotate_promax(loadings, m, normalize, max_iter, eps)
        }
    }
}

/// Generalized Orthomax rotation (Kaiser's algorithm matching R's `stats::varimax`).
fn rotate_orthomax(
    x: &DMatrix<f64>,
    gamma: f64,
    normalize: bool,
    max_iter: usize,
    eps: f64,
) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    let (p, k) = (x.nrows(), x.ncols());
    let mut x_work = x.clone();
    let mut sc = vec![1.0; p];
    if normalize {
        for i in 0..p {
            let row_norm = (0..k).map(|j| x[(i, j)].powi(2)).sum::<f64>().sqrt();
            sc[i] = if row_norm > 1e-12 { row_norm } else { 1.0 };
            for j in 0..k {
                x_work[(i, j)] /= sc[i];
            }
        }
    }

    let mut tt = DMatrix::identity(k, k);
    let mut d = 0.0;
    let pf = p as f64;

    for _ in 0..max_iter {
        let z = &x_work * &tt; // p x k
        // B = t(x_work) %*% (z^3 - gamma * z %*% diag(colSums(z^2)) / p)
        let mut target = DMatrix::zeros(p, k);
        for j in 0..k {
            let col_sum_sq: f64 = (0..p).map(|i| z[(i, j)].powi(2)).sum();
            for i in 0..p {
                let zij = z[(i, j)];
                target[(i, j)] = zij.powi(3) - (gamma * zij * col_sum_sq) / pf;
            }
        }
        let b = x_work.transpose() * &target; // k x k
        let svd = b.svd(true, true);
        let u = svd.u.ok_or_else(|| "SVD failed in rotation".to_string())?;
        let vt = svd.v_t.ok_or_else(|| "SVD failed in rotation".to_string())?;
        tt = &u * &vt;

        let d_past = d;
        d = svd.singular_values.iter().sum::<f64>();
        if d < d_past * (1.0 + eps) {
            break;
        }
    }

    let mut z = &x_work * &tt;
    if normalize {
        for i in 0..p {
            for j in 0..k {
                z[(i, j)] *= sc[i];
            }
        }
    }

    Ok((z, tt))
}

/// Oblique Promax rotation (matching R's `stats::promax`).
fn rotate_promax(
    x: &DMatrix<f64>,
    m: f64,
    normalize: bool,
    max_iter: usize,
    eps: f64,
) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    let (varimax_loadings, varimax_rot) = rotate_orthomax(x, 1.0, normalize, max_iter, eps)?;
    let (p, k) = (x.nrows(), x.ncols());

    // Q = x * |x|^(m - 1)
    let power = m.max(1.0) - 1.0;
    let mut q = DMatrix::zeros(p, k);
    for i in 0..p {
        for j in 0..k {
            let val = varimax_loadings[(i, j)];
            q[(i, j)] = val * val.abs().powf(power);
        }
    }

    // U = (varimax_loadings^T * varimax_loadings)^(-1) * varimax_loadings^T * Q
    let xtx = varimax_loadings.transpose() * &varimax_loadings;
    let xtq = varimax_loadings.transpose() * &q;
    let xtx_inv = xtx.clone().try_inverse().unwrap_or_else(|| xtx.pseudo_inverse(1e-12).unwrap());
    let mut u = &xtx_inv * &xtq; // k x k

    // d = diag(solve(t(U) %*% U)), U = U %*% diag(sqrt(d))
    let utu = u.transpose() * &u;
    let utu_inv = utu.clone().try_inverse().unwrap_or_else(|| utu.pseudo_inverse(1e-12).unwrap());
    for j in 0..k {
        let diag_val = utu_inv[(j, j)].max(1e-12);
        let scale = diag_val.sqrt();
        for i in 0..k {
            u[(i, j)] *= scale;
        }
    }

    let z = &varimax_loadings * &u;
    let rotmat = &varimax_rot * &u;
    Ok((z, rotmat))
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Principal Component Analysis Model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcaModel {
    pub n_components: usize,
    pub standardize: bool,
    pub rotation: RotationMethod,
    pub n_samples: usize,
    pub n_features: usize,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_explained_variance: Vec<f64>,
    pub components: Vec<Vec<f64>>, // shape: (n_components, n_features)
    pub rotation_matrix: Option<Vec<Vec<f64>>>, // shape: (n_components, n_components)
    pub total_variance: f64,
}

impl PcaModel {
    pub fn fit(
        data: &[Vec<f64>],
        n_components: Option<usize>,
        standardize: bool,
        rotation: Option<RotationMethod>,
    ) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 {
            return Err("PCA requires at least 2 samples".into());
        }

        let k = n_components.unwrap_or_else(|| n.min(p)).min(n).min(p);
        if k == 0 {
            return Err("n_components must be > 0".into());
        }
        let rot_method = rotation.unwrap_or(RotationMethod::None);

        // 1. Column means and stds
        let mut mean = vec![0.0; p];
        for j in 0..p {
            let col_sum: f64 = (0..n).map(|i| x[(i, j)]).sum();
            mean[j] = col_sum / (n as f64);
        }

        let mut scale = vec![1.0; p];
        if standardize {
            for j in 0..p {
                let var: f64 = (0..n).map(|i| (x[(i, j)] - mean[j]).powi(2)).sum::<f64>() / ((n - 1) as f64);
                let std = var.sqrt();
                scale[j] = if std > 1e-12 { std } else { 1.0 };
            }
        }

        // 2. Centered & scaled matrix
        let mut xc = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xc[(i, j)] = (x[(i, j)] - mean[j]) / scale[j];
            }
        }

        // 3. SVD: Xc = U S V^T
        let svd = xc.svd(false, true);
        let s = svd.singular_values;
        let vt = svd.v_t.ok_or_else(|| "SVD failed to compute V^T".to_string())?;

        let df = (n - 1) as f64;
        let mut total_var = 0.0;
        let mut exp_var = Vec::with_capacity(k);
        let max_sv = s.len();
        for i in 0..max_sv {
            let v = s[i].powi(2) / df;
            total_var += v;
            if i < k {
                exp_var.push(v);
            }
        }
        if total_var <= 1e-15 {
            total_var = 1.0;
        }

        let mut exp_var_ratio = Vec::with_capacity(k);
        let mut cum_var = Vec::with_capacity(k);
        let mut running = 0.0;
        for &v in &exp_var {
            let ratio = v / total_var;
            running += ratio;
            exp_var_ratio.push(ratio);
            cum_var.push(running.min(1.0));
        }

        // 4. Initial V matrix of shape (p, k)
        let mut v_mat = DMatrix::zeros(p, k);
        for j in 0..p {
            for comp_idx in 0..k {
                v_mat[(j, comp_idx)] = vt[(comp_idx, j)];
            }
        }

        // Apply rotation if requested (on loadings L = V * diag(s/sqrt(df)) or directions V)
        let (final_v_mat, rot_mat_opt) = if rot_method != RotationMethod::None && k >= 2 {
            // Compute loadings
            let mut loadings = DMatrix::zeros(p, k);
            for j in 0..p {
                for comp_idx in 0..k {
                    loadings[(j, comp_idx)] = v_mat[(j, comp_idx)] * (s[comp_idx] / df.sqrt());
                }
            }
            let (rot_loadings, t_mat) = rotate_loadings(&loadings, rot_method, true, 1000, 1e-5)?;
            // Recalculate component directions from rotated loadings
            let mut rot_v = DMatrix::zeros(p, k);
            for comp_idx in 0..k {
                let col_norm = (0..p).map(|j| rot_loadings[(j, comp_idx)].powi(2)).sum::<f64>().sqrt();
                let denom = if col_norm > 1e-12 { col_norm } else { 1.0 };
                for j in 0..p {
                    rot_v[(j, comp_idx)] = rot_loadings[(j, comp_idx)] / denom;
                }
            }
            (rot_v, Some(matrix_to_rows(&t_mat)))
        } else {
            // Sign determinism for unrotated components
            for comp_idx in 0..k {
                let mut max_val = 0.0;
                let mut max_abs = 0.0;
                for j in 0..p {
                    let val = v_mat[(j, comp_idx)];
                    if val.abs() > max_abs {
                        max_abs = val.abs();
                        max_val = val;
                    }
                }
                if max_val < 0.0 {
                    for j in 0..p {
                        v_mat[(j, comp_idx)] = -v_mat[(j, comp_idx)];
                    }
                }
            }
            (v_mat, None)
        };

        let mut components = Vec::with_capacity(k);
        for comp_idx in 0..k {
            let mut comp = vec![0.0; p];
            for j in 0..p {
                comp[j] = final_v_mat[(j, comp_idx)];
            }
            components.push(comp);
        }

        let mut s_vals = Vec::with_capacity(k);
        for i in 0..k {
            s_vals.push(s[i]);
        }

        Ok(Self {
            n_components: k,
            standardize,
            rotation: rot_method,
            n_samples: n,
            n_features: p,
            mean,
            scale,
            singular_values: s_vals,
            explained_variance: exp_var,
            explained_variance_ratio: exp_var_ratio,
            cumulative_explained_variance: cum_var,
            components,
            rotation_matrix: rot_mat_opt,
            total_variance: total_var,
        })
    }

    /// Project input rows into principal component space: Z = X_c * V
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }

        let k = self.n_components;
        let mut z = vec![vec![0.0; k]; n];
        for i in 0..n {
            for comp_idx in 0..k {
                let comp = &self.components[comp_idx];
                let mut sum = 0.0;
                for j in 0..p {
                    let centered = (x[(i, j)] - self.mean[j]) / self.scale[j];
                    sum += centered * comp[j];
                }
                z[i][comp_idx] = sum;
            }
        }
        Ok(z)
    }

    /// Map principal component scores back to the original feature space: X_hat = Z * V^T * scale + mean
    pub fn inverse_transform(&self, scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let n = scores.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = self.n_components;
        let p = self.n_features;
        for row in scores {
            if row.len() != k {
                return Err(format!("score dimension mismatch: expected {}, got {}", k, row.len()));
            }
        }

        let mut x_hat = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                let mut recon_scaled = 0.0;
                for comp_idx in 0..k {
                    recon_scaled += scores[i][comp_idx] * self.components[comp_idx][j];
                }
                x_hat[i][j] = recon_scaled * self.scale[j] + self.mean[j];
            }
        }
        Ok(x_hat)
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let scores = self.transform(data)?;
        self.inverse_transform(&scores)
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let x_hat = self.reconstruct(data)?;
        let mut errors = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..self.n_features {
                let diff = row[j] - x_hat[i][j];
                sum_sq += diff * diff;
            }
            errors.push(sum_sq);
        }
        Ok(errors)
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_components".into(), self.n_components.to_string());
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("n_samples".into(), self.n_samples.to_string());
        struct_map.insert(
            "retained_variance_ratio".into(),
            format!("{:.4}", self.cumulative_explained_variance.last().copied().unwrap_or(0.0)),
        );

        let rot_name = match self.rotation {
            RotationMethod::None => "none",
            RotationMethod::Varimax => "varimax",
            RotationMethod::Quartimax => "quartimax",
            RotationMethod::Equamax => "equamax",
            RotationMethod::Parsimax => "parsimax",
            RotationMethod::Biquartimax => "biquartimax",
            RotationMethod::Orthomax(_) => "orthomax",
            RotationMethod::Promax(_) => "promax",
        };
        struct_map.insert("rotation".into(), rot_name.into());

        let mut cfg_map = BTreeMap::new();
        cfg_map.insert("standardize".into(), self.standardize.to_string());
        cfg_map.insert("n_components".into(), self.n_components.to_string());
        cfg_map.insert("rotation".into(), rot_name.into());

        let mut metrics = BTreeMap::new();
        metrics.insert("total_variance".into(), self.total_variance);
        if let Some(&first_ratio) = self.explained_variance_ratio.first() {
            metrics.insert("first_component_variance_ratio".into(), first_ratio);
        }
        if let Some(&cum_ratio) = self.cumulative_explained_variance.last() {
            metrics.insert("cumulative_variance_ratio".into(), cum_ratio);
        }

        ModelReport::new(
            "Principal Component Analysis",
            ObjectiveSection {
                name: "Maximum Variance Subspace Projection".into(),
                math_form: Some("argmax_V Tr(V^T Cov(X) V) s.t. V^T V = I".into()),
                direction: OptimizationDirection::Reconstruct,
                final_value: Some(1.0 - self.cumulative_explained_variance.last().copied().unwrap_or(1.0)),
                components: vec![
                    ("unexplained_variance_ratio".into(), 1.0 - self.cumulative_explained_variance.last().copied().unwrap_or(1.0)),
                ],
                units: Some("variance ratio".into()),
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: if self.standardize { "Standardized PCA" } else { "Centered PCA" }.into(),
                representation: format!("Orthogonal/Rotated linear basis vectors V^T ({})", rot_name),
                structure: struct_map,
                nominal_complexity: self.n_components * self.n_features,
                effective_complexity: Some(self.n_components as f64),
                capabilities: vec![
                    "transform".into(),
                    "inverse_transform".into(),
                    "reconstruct".into(),
                    "anomaly_score".into(),
                ],
            },
            ControllerSection {
                kind: "pca".into(),
                config: cfg_map,
                seed: None,
                iteration_budget: None,
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(true),
                stop_reason: Some("Exact SVD decomposition completed".into()),
                fit_time_sec: None,
                eval_count: Some(1),
                metrics,
                warnings: Vec::new(),
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Truncated SVD
// ---------------------------------------------------------------------------

/// Truncated SVD Model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TruncatedSvdModel {
    pub n_components: usize,
    pub n_samples: usize,
    pub n_features: usize,
    pub singular_values: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub components: Vec<Vec<f64>>,
    pub total_variance: f64,
}

impl TruncatedSvdModel {
    pub fn fit(data: &[Vec<f64>], n_components: Option<usize>) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 1 {
            return Err("Truncated SVD requires at least 1 sample".into());
        }

        let k = n_components.unwrap_or_else(|| n.min(p)).min(n).min(p);
        if k == 0 {
            return Err("n_components must be > 0".into());
        }

        let svd = x.svd(false, true);
        let s = svd.singular_values;
        let vt = svd.v_t.ok_or_else(|| "SVD failed to compute V^T".to_string())?;

        let df = (n.max(2) - 1) as f64;
        let mut total_var = 0.0;
        let mut exp_var = Vec::with_capacity(k);
        let max_sv = s.len();
        for i in 0..max_sv {
            let v = s[i].powi(2) / df;
            total_var += v;
            if i < k {
                exp_var.push(v);
            }
        }
        if total_var <= 1e-15 {
            total_var = 1.0;
        }

        let mut exp_var_ratio = Vec::with_capacity(k);
        for &v in &exp_var {
            exp_var_ratio.push(v / total_var);
        }

        let mut components = Vec::with_capacity(k);
        for comp_idx in 0..k {
            let mut comp = vec![0.0; p];
            for j in 0..p {
                comp[j] = vt[(comp_idx, j)];
            }
            components.push(comp);
        }

        let mut s_vals = Vec::with_capacity(k);
        for i in 0..k {
            s_vals.push(s[i]);
        }

        Ok(Self {
            n_components: k,
            n_samples: n,
            n_features: p,
            singular_values: s_vals,
            explained_variance: exp_var,
            explained_variance_ratio: exp_var_ratio,
            components,
            total_variance: total_var,
        })
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }
        let k = self.n_components;
        let mut z = vec![vec![0.0; k]; n];
        for i in 0..n {
            for comp_idx in 0..k {
                let comp = &self.components[comp_idx];
                let mut sum = 0.0;
                for j in 0..p {
                    sum += x[(i, j)] * comp[j];
                }
                z[i][comp_idx] = sum;
            }
        }
        Ok(z)
    }

    pub fn inverse_transform(&self, scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let n = scores.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = self.n_components;
        let p = self.n_features;
        for row in scores {
            if row.len() != k {
                return Err(format!("score dimension mismatch: expected {}, got {}", k, row.len()));
            }
        }
        let mut x_hat = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0;
                for comp_idx in 0..k {
                    sum += scores[i][comp_idx] * self.components[comp_idx][j];
                }
                x_hat[i][j] = sum;
            }
        }
        Ok(x_hat)
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let scores = self.transform(data)?;
        self.inverse_transform(&scores)
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let x_hat = self.reconstruct(data)?;
        let mut errors = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..self.n_features {
                let diff = row[j] - x_hat[i][j];
                sum_sq += diff * diff;
            }
            errors.push(sum_sq);
        }
        Ok(errors)
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_components".into(), self.n_components.to_string());
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("n_samples".into(), self.n_samples.to_string());

        let mut cfg_map = BTreeMap::new();
        cfg_map.insert("n_components".into(), self.n_components.to_string());

        let mut metrics = BTreeMap::new();
        metrics.insert("total_variance".into(), self.total_variance);

        ModelReport::new(
            "Truncated SVD",
            ObjectiveSection {
                name: "Uncentered Low-Rank Matrix Factorization".into(),
                math_form: Some("min ||X - U_k Sigma_k V_k^T||_F^2".into()),
                direction: OptimizationDirection::Reconstruct,
                final_value: None,
                components: Vec::new(),
                units: None,
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: "Truncated SVD".into(),
                representation: "Uncentered singular vectors V_k^T".into(),
                structure: struct_map,
                nominal_complexity: self.n_components * self.n_features,
                effective_complexity: Some(self.n_components as f64),
                capabilities: vec![
                    "transform".into(),
                    "inverse_transform".into(),
                    "reconstruct".into(),
                ],
            },
            ControllerSection {
                kind: "svd".into(),
                config: cfg_map,
                seed: None,
                iteration_budget: None,
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(true),
                stop_reason: Some("Exact SVD factorization completed".into()),
                fit_time_sec: None,
                eval_count: Some(1),
                metrics,
                warnings: Vec::new(),
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Whitening
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WhiteningKind {
    Zca,
    Pca,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhiteningModel {
    pub kind: WhiteningKind,
    pub n_features: usize,
    pub n_samples: usize,
    pub mean: Vec<f64>,
    pub whitening_matrix: Vec<Vec<f64>>,
    pub dewhitening_matrix: Vec<Vec<f64>>,
}

impl WhiteningModel {
    pub fn fit(data: &[Vec<f64>], kind: WhiteningKind, eps: Option<f64>) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 {
            return Err("Whitening requires at least 2 samples".into());
        }
        let epsilon = eps.unwrap_or(1e-5).max(1e-12);

        let mut mean = vec![0.0; p];
        for j in 0..p {
            mean[j] = (0..n).map(|i| x[(i, j)]).sum::<f64>() / (n as f64);
        }

        let mut xc = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xc[(i, j)] = x[(i, j)] - mean[j];
            }
        }

        let cov = (&xc.transpose() * &xc) / ((n - 1) as f64);
        let eigen = cov.symmetric_eigen();
        let ev = eigen.eigenvalues;
        let v = eigen.eigenvectors;

        let mut inv_sqrt_d = DMatrix::zeros(p, p);
        let mut sqrt_d = DMatrix::zeros(p, p);
        for j in 0..p {
            let lambda = ev[j].max(0.0) + epsilon;
            inv_sqrt_d[(j, j)] = 1.0 / lambda.sqrt();
            sqrt_d[(j, j)] = lambda.sqrt();
        }

        let w_mat = match kind {
            WhiteningKind::Pca => &inv_sqrt_d * &v.transpose(),
            WhiteningKind::Zca => &v * &inv_sqrt_d * &v.transpose(),
        };

        let de_mat = match kind {
            WhiteningKind::Pca => &v * &sqrt_d,
            WhiteningKind::Zca => &v * &sqrt_d * &v.transpose(),
        };

        Ok(Self {
            kind,
            n_features: p,
            n_samples: n,
            mean,
            whitening_matrix: matrix_to_rows(&w_mat),
            dewhitening_matrix: matrix_to_rows(&de_mat),
        })
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }
        let w = rows_to_matrix(&self.whitening_matrix)?;
        let mut z = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..p {
                    let centered = x[(i, k)] - self.mean[k];
                    sum += centered * w[(j, k)];
                }
                z[i][j] = sum;
            }
        }
        Ok(z)
    }

    pub fn inverse_transform(&self, whitened: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let z = rows_to_matrix(whitened)?;
        let (n, p) = (z.nrows(), z.ncols());
        if p != self.n_features {
            return Err(format!("dimension mismatch: expected {}, got {}", self.n_features, p));
        }
        let de_w = rows_to_matrix(&self.dewhitening_matrix)?;
        let mut x_hat = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..p {
                    sum += z[(i, k)] * de_w[(k, j)];
                }
                x_hat[i][j] = sum + self.mean[j];
            }
        }
        Ok(x_hat)
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("kind".into(), match self.kind { WhiteningKind::Zca => "ZCA", WhiteningKind::Pca => "PCA" }.into());

        ModelReport::new(
            "Whitening Transform",
            ObjectiveSection {
                name: "Covariance Decorrelation and Variance Equalization".into(),
                math_form: Some("Cov(Z) = I".into()),
                direction: OptimizationDirection::Reconstruct,
                final_value: None,
                components: Vec::new(),
                units: None,
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: match self.kind { WhiteningKind::Zca => "ZCA Whitening", WhiteningKind::Pca => "PCA Whitening" }.into(),
                representation: "Linear decorrelation operator W".into(),
                structure: struct_map,
                nominal_complexity: self.n_features * self.n_features,
                effective_complexity: Some(self.n_features as f64),
                capabilities: vec!["transform".into(), "inverse_transform".into()],
            },
            ControllerSection {
                kind: "whitening".into(),
                config: BTreeMap::new(),
                seed: None,
                iteration_budget: None,
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(true),
                stop_reason: Some("Exact covariance eigendecomposition completed".into()),
                fit_time_sec: None,
                eval_count: Some(1),
                metrics: BTreeMap::new(),
                warnings: Vec::new(),
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// FastICA
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastIcaModel {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub n_samples: usize,
    pub n_features: usize,
    pub mean: Vec<f64>,
    pub components: Vec<Vec<f64>>,
    pub mixing_matrix: Vec<Vec<f64>>,
    pub n_iter: usize,
    pub converged: bool,
}

impl FastIcaModel {
    pub fn fit(
        data: &[Vec<f64>],
        n_components: Option<usize>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 {
            return Err("FastICA requires at least 2 samples".into());
        }
        let k = n_components.unwrap_or(p).min(n).min(p);
        let max_iterations = max_iter.unwrap_or(200);
        let tolerance = tol.unwrap_or(1e-4);
        let random_seed = seed.unwrap_or(42);

        let mut mean = vec![0.0; p];
        for j in 0..p {
            mean[j] = (0..n).map(|i| x[(i, j)]).sum::<f64>() / (n as f64);
        }
        let mut xc = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xc[(i, j)] = x[(i, j)] - mean[j];
            }
        }

        let svd = xc.clone().svd(false, true);
        let s = svd.singular_values;
        let vt = svd.v_t.ok_or_else(|| "SVD failed in FastICA".to_string())?;

        let scale = ((n - 1) as f64).sqrt();
        let mut k_mat = DMatrix::zeros(k, p);
        for i in 0..k {
            let inv_s = scale / s[i].max(1e-12);
            for j in 0..p {
                k_mat[(i, j)] = inv_s * vt[(i, j)];
            }
        }
        let xw = &xc * &k_mat.transpose();

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let unif = Uniform::new(-1.0, 1.0);
        let mut w = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                w[(i, j)] = unif.sample(&mut rng);
            }
        }
        w = symmetric_decorrelation(&w)?;

        let mut converged = false;
        let mut n_iter = 0;
        let n_f64 = n as f64;

        for iter in 0..max_iterations {
            n_iter = iter + 1;
            let w_old = w.clone();

            for i in 0..k {
                let w_row = w_old.row(i).transpose();
                let u = &xw * &w_row;
                let mut g_u = DVector::zeros(n);
                let mut sum_g_prime = 0.0;
                for row_idx in 0..n {
                    let val = u[row_idx].tanh();
                    g_u[row_idx] = val;
                    sum_g_prime += 1.0 - val * val;
                }
                let avg_g_prime = sum_g_prime / n_f64;
                let xw_t_g = &xw.transpose() * &g_u;
                let term1 = xw_t_g / n_f64;
                let term2 = &w_row * avg_g_prime;
                let w_new = term1 - term2;
                for j in 0..k {
                    w[(i, j)] = w_new[j];
                }
            }

            w = symmetric_decorrelation(&w)?;

            let mut max_diff: f64 = 0.0;
            for i in 0..k {
                let dot: f64 = (0..k).map(|j| w[(i, j)] * w_old[(i, j)]).sum();
                let diff = (1.0 - dot.abs()).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            if max_diff < tolerance {
                converged = true;
                break;
            }
        }

        let w_total = &w * &k_mat;
        let a_mat = match w_total.clone().pseudo_inverse(1e-12) {
            Ok(inv) => inv,
            Err(_) => w_total.transpose(),
        };

        Ok(Self {
            n_components: k,
            max_iter: max_iterations,
            tol: tolerance,
            seed: random_seed,
            n_samples: n,
            n_features: p,
            mean,
            components: matrix_to_rows(&w_total),
            mixing_matrix: matrix_to_rows(&a_mat),
            n_iter,
            converged,
        })
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }
        let w = rows_to_matrix(&self.components)?;
        let k = self.n_components;
        let mut s = vec![vec![0.0; k]; n];
        for i in 0..n {
            for comp_idx in 0..k {
                let mut sum = 0.0;
                for j in 0..p {
                    let centered = x[(i, j)] - self.mean[j];
                    sum += centered * w[(comp_idx, j)];
                }
                s[i][comp_idx] = sum;
            }
        }
        Ok(s)
    }

    pub fn inverse_transform(&self, sources: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let s = rows_to_matrix(sources)?;
        let (n, k) = (s.nrows(), s.ncols());
        if k != self.n_components {
            return Err(format!("source dimension mismatch: expected {}, got {}", self.n_components, k));
        }
        let a = rows_to_matrix(&self.mixing_matrix)?;
        let p = self.n_features;
        let mut x_hat = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0;
                for comp_idx in 0..k {
                    sum += s[(i, comp_idx)] * a[(j, comp_idx)];
                }
                x_hat[i][j] = sum + self.mean[j];
            }
        }
        Ok(x_hat)
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let sources = self.transform(data)?;
        self.inverse_transform(&sources)
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let x_hat = self.reconstruct(data)?;
        let mut errors = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..self.n_features {
                let diff = row[j] - x_hat[i][j];
                sum_sq += diff * diff;
            }
            errors.push(sum_sq);
        }
        Ok(errors)
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_components".into(), self.n_components.to_string());
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("iterations".into(), self.n_iter.to_string());

        let mut cfg = BTreeMap::new();
        cfg.insert("max_iter".into(), self.max_iter.to_string());
        cfg.insert("tol".into(), format!("{:e}", self.tol));

        ModelReport::new(
            "FastICA (Independent Component Analysis)",
            ObjectiveSection {
                name: "Non-Gaussianity Maximization (Negentropy Contrast)".into(),
                math_form: Some("max sum_i E[G(w_i^T X_w)] s.t. W W^T = I".into()),
                direction: OptimizationDirection::Maximize,
                final_value: None,
                components: Vec::new(),
                units: None,
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: "FastICA".into(),
                representation: "Unmixing matrix W and mixing matrix A".into(),
                structure: struct_map,
                nominal_complexity: self.n_components * self.n_features,
                effective_complexity: Some(self.n_components as f64),
                capabilities: vec!["transform".into(), "inverse_transform".into(), "reconstruct".into()],
            },
            ControllerSection {
                kind: "ica".into(),
                config: cfg,
                seed: Some(self.seed),
                iteration_budget: Some(self.max_iter),
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(self.converged),
                stop_reason: Some(if self.converged { "Convergence tolerance achieved".into() } else { "Maximum iterations reached".into() }),
                fit_time_sec: None,
                eval_count: Some(self.n_iter),
                metrics: BTreeMap::new(),
                warnings: if self.converged {
                    Vec::new()
                } else {
                    vec!["FastICA failed to converge within max_iter iterations".into()]
                },
            },
            None,
        )
    }
}

fn symmetric_decorrelation(w: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let k = w.nrows();
    let wwt = w * w.transpose();
    let eigen = wwt.symmetric_eigen();
    let mut inv_sqrt_d = DMatrix::zeros(k, k);
    for i in 0..k {
        let val = eigen.eigenvalues[i].max(1e-12);
        inv_sqrt_d[(i, i)] = 1.0 / val.sqrt();
    }
    let factor = &eigen.eigenvectors * &inv_sqrt_d * &eigen.eigenvectors.transpose();
    Ok(&factor * w)
}

// ---------------------------------------------------------------------------
// NMF (Nonnegative Matrix Factorization)
// ---------------------------------------------------------------------------

/// NMF Solver Algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NmfSolver {
    Mu,   // Multiplicative Updates (Lee & Seung 2001)
    Als,  // Projected Alternating Least Squares (Paatero 1999)
    Hals, // Hierarchical Alternating Least Squares (Cichocki et al. 2007, 2009)
}

impl NmfSolver {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().trim() {
            "mu" | "multiplicative" => Ok(Self::Mu),
            "als" | "pals" | "alternating_least_squares" => Ok(Self::Als),
            "hals" | "fast_hals" | "coordinate_descent" | "cd" => Ok(Self::Hals),
            other => Err(format!("unknown NMF solver: {}", other)),
        }
    }
}

/// Non-negative Matrix Factorization Model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NmfModel {
    pub n_components: usize,
    pub solver: NmfSolver,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub l1_reg: f64,
    pub l2_reg: f64,
    pub n_samples: usize,
    pub n_features: usize,
    pub components: Vec<Vec<f64>>, // H matrix: (n_components, n_features)
    pub reconstruction_error: f64,
    pub n_iter: usize,
    pub converged: bool,
}

impl NmfModel {
    pub fn fit(
        data: &[Vec<f64>],
        n_components: Option<usize>,
        solver: Option<NmfSolver>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
        l1_reg: Option<f64>,
        l2_reg: Option<f64>,
    ) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 1 {
            return Err("NMF requires at least 1 sample".into());
        }
        for i in 0..n {
            for j in 0..p {
                if x[(i, j)] < 0.0 {
                    return Err(format!("NMF requires non-negative inputs: negative value at ({}, {})", i, j));
                }
            }
        }

        let k = n_components.unwrap_or_else(|| n.min(p)).min(n).min(p);
        let nmf_solver = solver.unwrap_or(NmfSolver::Hals);
        let max_iterations = max_iter.unwrap_or(200);
        let tolerance = tol.unwrap_or(1e-4);
        let random_seed = seed.unwrap_or(42);
        let lambda1 = l1_reg.unwrap_or(0.0).max(0.0);
        let lambda2 = l2_reg.unwrap_or(0.0).max(0.0);

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let unif = Uniform::new(0.1, 1.0);

        let mut w = DMatrix::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                w[(i, j)] = unif.sample(&mut rng);
            }
        }

        let mut h = DMatrix::zeros(k, p);
        for i in 0..k {
            for j in 0..p {
                h[(i, j)] = unif.sample(&mut rng);
            }
        }

        let eps = 1e-12;
        let mut prev_error = f64::MAX;
        let mut converged = false;
        let mut n_iter = 0;
        let mut final_error = 0.0;

        for iter in 0..max_iterations {
            n_iter = iter + 1;

            match nmf_solver {
                NmfSolver::Mu => {
                    let wt = w.transpose();
                    let wt_x = &wt * &x;
                    let wt_w_h = &(&wt * &w) * &h;
                    for i in 0..k {
                        for j in 0..p {
                            h[(i, j)] = (h[(i, j)] * wt_x[(i, j)] / (wt_w_h[(i, j)] + lambda1 + lambda2 * h[(i, j)] + eps)).max(0.0);
                        }
                    }

                    let ht = h.transpose();
                    let x_ht = &x * &ht;
                    let w_h_ht = &w * &(&h * &ht);
                    for i in 0..n {
                        for j in 0..k {
                            w[(i, j)] = (w[(i, j)] * x_ht[(i, j)] / (w_h_ht[(i, j)] + lambda1 + lambda2 * w[(i, j)] + eps)).max(0.0);
                        }
                    }
                }
                NmfSolver::Als => {
                    let wt = w.transpose();
                    let mut wt_w = &wt * &w;
                    for i in 0..k {
                        wt_w[(i, i)] += lambda2 + eps;
                    }
                    let wt_x = &wt * &x;
                    let h_solve = wt_w.clone().cholesky()
                        .map(|chol| chol.solve(&wt_x))
                        .unwrap_or_else(|| wt_w.pseudo_inverse(1e-12).unwrap() * &wt_x);
                    for i in 0..k {
                        for j in 0..p {
                            let mut val = h_solve[(i, j)];
                            if lambda1 > 0.0 {
                                val -= lambda1;
                            }
                            h[(i, j)] = val.max(0.0);
                        }
                    }

                    let ht = h.transpose();
                    let mut h_ht = &h * &ht;
                    for i in 0..k {
                        h_ht[(i, i)] += lambda2 + eps;
                    }
                    let x_ht = &x * &ht;
                    let wt_solve = h_ht.clone().cholesky()
                        .map(|chol| chol.solve(&x_ht.transpose()))
                        .unwrap_or_else(|| h_ht.pseudo_inverse(1e-12).unwrap() * &x_ht.transpose());
                    let w_solve = wt_solve.transpose();
                    for i in 0..n {
                        for j in 0..k {
                            let mut val = w_solve[(i, j)];
                            if lambda1 > 0.0 {
                                val -= lambda1;
                            }
                            w[(i, j)] = val.max(0.0);
                        }
                    }

                    for j in 0..k {
                        let norm = (0..n).map(|i| w[(i, j)].powi(2)).sum::<f64>().sqrt();
                        if norm > 1e-12 {
                            for i in 0..n {
                                w[(i, j)] /= norm;
                            }
                            for col in 0..p {
                                h[(j, col)] *= norm;
                            }
                        }
                    }
                }
                NmfSolver::Hals => {
                    let wt = w.transpose();
                    let a = &wt * &w;
                    let b = &wt * &x;

                    for j in 0..k {
                        let denom = a[(j, j)] + lambda2 + eps;
                        for l in 0..p {
                            let mut pred_j = 0.0;
                            for m in 0..k {
                                pred_j += a[(j, m)] * h[(m, l)];
                            }
                            let diff = b[(j, l)] - pred_j - lambda1;
                            let new_val = h[(j, l)] + diff / denom;
                            h[(j, l)] = new_val.max(0.0);
                        }
                    }

                    let ht = h.transpose();
                    let c = &h * &ht;
                    let d = &x * &ht;

                    for j in 0..k {
                        let denom = c[(j, j)] + lambda2 + eps;
                        for i in 0..n {
                            let mut pred_j = 0.0;
                            for m in 0..k {
                                pred_j += w[(i, m)] * c[(m, j)];
                            }
                            let diff = d[(i, j)] - pred_j - lambda1;
                            let new_val = w[(i, j)] + diff / denom;
                            w[(i, j)] = new_val.max(0.0);
                        }

                        let norm = (0..n).map(|i| w[(i, j)].powi(2)).sum::<f64>().sqrt();
                        if norm > 1e-12 {
                            for i in 0..n {
                                w[(i, j)] /= norm;
                            }
                            for col in 0..p {
                                h[(j, col)] *= norm;
                            }
                        }
                    }
                }
            }

            let x_approx = &w * &h;
            let diff = &x - &x_approx;
            let err = diff.norm();
            final_error = err;

            if (prev_error - err).abs() < tolerance {
                converged = true;
                break;
            }
            prev_error = err;
        }

        Ok(Self {
            n_components: k,
            solver: nmf_solver,
            max_iter: max_iterations,
            tol: tolerance,
            seed: random_seed,
            l1_reg: lambda1,
            l2_reg: lambda2,
            n_samples: n,
            n_features: p,
            components: matrix_to_rows(&h),
            reconstruction_error: final_error,
            n_iter,
            converged,
        })
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }
        for i in 0..n {
            for j in 0..p {
                if x[(i, j)] < 0.0 {
                    return Err("NMF requires non-negative inputs".into());
                }
            }
        }

        let k = self.n_components;
        let h = rows_to_matrix(&self.components)?;
        let ht = h.transpose();
        let h_ht = &h * &ht;

        let mut w = DMatrix::from_element(n, k, 0.5);
        let x_ht = &x * &ht;
        let eps = 1e-12;

        for _ in 0..40 {
            let w_h_ht = &w * &h_ht;
            for i in 0..n {
                for j in 0..k {
                    w[(i, j)] = (w[(i, j)] * x_ht[(i, j)] / (w_h_ht[(i, j)] + eps)).max(0.0);
                }
            }
        }

        Ok(matrix_to_rows(&w))
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let w_rows = self.transform(data)?;
        let w = rows_to_matrix(&w_rows)?;
        let h = rows_to_matrix(&self.components)?;
        let x_hat = &w * &h;
        Ok(matrix_to_rows(&x_hat))
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let x_hat = self.reconstruct(data)?;
        let mut errors = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..self.n_features {
                let diff = row[j] - x_hat[i][j];
                sum_sq += diff * diff;
            }
            errors.push(sum_sq);
        }
        Ok(errors)
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_components".into(), self.n_components.to_string());
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("iterations".into(), self.n_iter.to_string());
        struct_map.insert("solver".into(), match self.solver {
            NmfSolver::Mu => "Multiplicative Updates (MU)".into(),
            NmfSolver::Als => "Projected Alternating Least Squares (ALS)".into(),
            NmfSolver::Hals => "Hierarchical Alternating Least Squares (HALS)".into(),
        });

        let mut metrics = BTreeMap::new();
        metrics.insert("frobenius_reconstruction_error".into(), self.reconstruction_error);

        ModelReport::new(
            "Non-negative Matrix Factorization",
            ObjectiveSection {
                name: "Additive Non-negative Factorization (Frobenius Loss)".into(),
                math_form: Some("min_{W,H >= 0} ||X - WH||_F^2 + lambda1(|W|_1+|H|_1) + lambda2/2(|W|_F^2+|H|_F^2)".into()),
                direction: OptimizationDirection::Reconstruct,
                final_value: Some(self.reconstruction_error),
                components: vec![("frobenius_error".into(), self.reconstruction_error)],
                units: None,
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: "NMF".into(),
                representation: "Non-negative additive basis factors H".into(),
                structure: struct_map,
                nominal_complexity: self.n_components * self.n_features,
                effective_complexity: Some(self.n_components as f64),
                capabilities: vec!["transform".into(), "reconstruct".into(), "anomaly_score".into()],
            },
            ControllerSection {
                kind: "nmf".into(),
                config: BTreeMap::new(),
                seed: Some(self.seed),
                iteration_budget: Some(self.max_iter),
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(self.converged),
                stop_reason: Some(if self.converged { "Convergence tolerance achieved".into() } else { "Maximum iterations reached".into() }),
                fit_time_sec: None,
                eval_count: Some(self.n_iter),
                metrics,
                warnings: if self.converged {
                    Vec::new()
                } else {
                    vec!["NMF did not converge within max_iter".into()]
                },
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Factor Analysis (PAF & Minres)
// ---------------------------------------------------------------------------

/// Exploratory Factor Analysis Model (Principal Axis Factoring / Minres).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactorAnalysisModel {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub rotation: RotationMethod,
    pub method: String,
    pub n_samples: usize,
    pub n_features: usize,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub components: Vec<Vec<f64>>, // shape: (n_components, n_features) - factor loadings L^T
    pub uniquenesses: Vec<f64>,     // specific variances psi_i (shape: n_features)
    pub communalities: Vec<f64>,    // h_i^2 = 1 - psi_i (shape: n_features)
    pub rotation_matrix: Option<Vec<Vec<f64>>>,
    pub factor_correlations: Option<Vec<Vec<f64>>>,
    pub n_iter: usize,
    pub converged: bool,
}

impl FactorAnalysisModel {
    pub fn fit(
        data: &[Vec<f64>],
        n_components: Option<usize>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        rotation: Option<RotationMethod>,
        method: &str,
    ) -> Result<Self, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 {
            return Err("FactorAnalysis requires at least 2 samples".into());
        }
        let k = n_components.unwrap_or(2).min(p).max(1);
        let max_iterations = max_iter.unwrap_or(100);
        let tolerance = tol.unwrap_or(1e-4);
        let rot_method = rotation.unwrap_or(RotationMethod::None);

        // 1. Center and standardize columns
        let mut mean = vec![0.0; p];
        for j in 0..p {
            let s: f64 = (0..n).map(|i| x[(i, j)]).sum();
            mean[j] = s / (n as f64);
        }
        let mut scale = vec![1.0; p];
        for j in 0..p {
            let var: f64 = (0..n).map(|i| (x[(i, j)] - mean[j]).powi(2)).sum::<f64>() / ((n - 1) as f64);
            let std = var.sqrt();
            scale[j] = if std > 1e-12 { std } else { 1.0 };
        }

        let mut xc = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xc[(i, j)] = (x[(i, j)] - mean[j]) / scale[j];
            }
        }

        // 2. Sample correlation matrix R = (1 / (n - 1)) Xc^T Xc
        let r_mat = (xc.transpose() * &xc) / ((n - 1) as f64);

        // 3. Initial communality estimates h_i^2:
        let mut h2 = vec![0.5; p];
        if let Some(r_inv) = r_mat.clone().try_inverse() {
            for j in 0..p {
                let smc = 1.0 - 1.0 / r_inv[(j, j)].max(1e-6);
                h2[j] = smc.clamp(0.01, 0.99);
            }
        } else {
            for j in 0..p {
                let max_r = (0..p)
                    .filter(|&l| l != j)
                    .map(|l| r_mat[(j, l)].abs())
                    .fold(0.0, f64::max);
                h2[j] = max_r.clamp(0.01, 0.99);
            }
        }

        // 4. Principal Axis Factoring (iterative communality update)
        let mut n_iter = 0;
        let mut converged = false;
        let mut unrotated_loadings = DMatrix::zeros(p, k);

        for it in 0..max_iterations {
            n_iter = it + 1;
            let mut r_star = r_mat.clone();
            for j in 0..p {
                r_star[(j, j)] = h2[j];
            }

            let eigen = r_star.symmetric_eigen();
            let mut pairs: Vec<(usize, f64)> = eigen.eigenvalues
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut new_loadings = DMatrix::zeros(p, k);
            for (comp_idx, &(orig_idx, val)) in pairs.iter().take(k).enumerate() {
                let pos_val = val.max(0.0);
                let sqrt_val = pos_val.sqrt();
                for j in 0..p {
                    new_loadings[(j, comp_idx)] = eigen.eigenvectors[(j, orig_idx)] * sqrt_val;
                }
            }

            for comp_idx in 0..k {
                let mut max_abs = 0.0;
                let mut sign = 1.0;
                for j in 0..p {
                    let v = new_loadings[(j, comp_idx)];
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                        sign = if v >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
                if sign < 0.0 {
                    for j in 0..p {
                        new_loadings[(j, comp_idx)] = -new_loadings[(j, comp_idx)];
                    }
                }
            }

            let mut max_diff: f64 = 0.0;
            let mut new_h2 = vec![0.0; p];
            for j in 0..p {
                let sum_sq: f64 = (0..k).map(|c| new_loadings[(j, c)].powi(2)).sum();
                let clamped = sum_sq.clamp(0.001, 0.999);
                let diff = (clamped - h2[j]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                new_h2[j] = clamped;
            }

            h2 = new_h2;
            unrotated_loadings = new_loadings;

            if max_diff < tolerance {
                converged = true;
                break;
            }
        }

        let uniquenesses: Vec<f64> = h2.iter().map(|&h| (1.0 - h).max(1e-4)).collect();
        let communalities = h2.clone();

        // 5. Factor Rotation
        let (final_loadings, rot_mat_opt, factor_corrs_opt) = if rot_method != RotationMethod::None && k >= 2 {
            let (rot_loadings, t_mat) = rotate_loadings(&unrotated_loadings, rot_method, true, 1000, 1e-5)?;
            let rot_mat_rows = matrix_to_rows(&t_mat);
            let factor_corrs = if rot_method.is_oblique() {
                let tt = t_mat.transpose() * &t_mat;
                let phi = tt.try_inverse().unwrap_or_else(|| DMatrix::identity(k, k));
                Some(matrix_to_rows(&phi))
            } else {
                Some(matrix_to_rows(&DMatrix::identity(k, k)))
            };
            (rot_loadings, Some(rot_mat_rows), factor_corrs)
        } else {
            (unrotated_loadings, None, None)
        };

        let mut components = Vec::with_capacity(k);
        for comp_idx in 0..k {
            let mut comp = vec![0.0; p];
            for j in 0..p {
                comp[j] = final_loadings[(j, comp_idx)];
            }
            components.push(comp);
        }

        Ok(Self {
            n_components: k,
            max_iter: max_iterations,
            tol: tolerance,
            rotation: rot_method,
            method: method.to_string(),
            n_samples: n,
            n_features: p,
            mean,
            scale,
            components,
            uniquenesses,
            communalities,
            rotation_matrix: rot_mat_opt,
            factor_correlations: factor_corrs_opt,
            n_iter,
            converged,
        })
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let x = rows_to_matrix(data)?;
        let n = x.nrows();
        let p = x.ncols();
        if p != self.n_features {
            return Err(format!("feature dimension mismatch: expected {}, got {}", self.n_features, p));
        }

        let mut xc = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xc[(i, j)] = (x[(i, j)] - self.mean[j]) / self.scale[j];
            }
        }

        let k = self.n_components;
        let mut l = DMatrix::zeros(p, k);
        for comp_idx in 0..k {
            for j in 0..p {
                l[(j, comp_idx)] = self.components[comp_idx][j];
            }
        }

        let mut psi_inv_l = DMatrix::zeros(p, k);
        for j in 0..p {
            let w = 1.0 / self.uniquenesses[j].max(1e-4);
            for c in 0..k {
                psi_inv_l[(j, c)] = l[(j, c)] * w;
            }
        }
        let lt_psi_inv_l = l.transpose() * &psi_inv_l;
        let inv_term = lt_psi_inv_l.clone().try_inverse()
            .unwrap_or_else(|| lt_psi_inv_l.pseudo_inverse(1e-12).unwrap());
        let score_weights = psi_inv_l * inv_term;

        let scores = &xc * &score_weights;
        Ok(matrix_to_rows(&scores))
    }

    pub fn inverse_transform(&self, scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let f = rows_to_matrix(scores)?;
        let n = f.nrows();
        let k = f.ncols();
        if k != self.n_components {
            return Err(format!("score dimension mismatch: expected {}, got {}", self.n_components, k));
        }
        let p = self.n_features;
        let mut l = DMatrix::zeros(p, k);
        for comp_idx in 0..k {
            for j in 0..p {
                l[(j, comp_idx)] = self.components[comp_idx][j];
            }
        }
        let xc_hat = &f * l.transpose();
        let mut x_hat = DMatrix::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x_hat[(i, j)] = xc_hat[(i, j)] * self.scale[j] + self.mean[j];
            }
        }
        Ok(matrix_to_rows(&x_hat))
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let scores = self.transform(data)?;
        self.inverse_transform(&scores)
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let x_hat = self.reconstruct(data)?;
        let mut errors = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let mut sum_sq = 0.0;
            for j in 0..self.n_features {
                let diff = row[j] - x_hat[i][j];
                sum_sq += diff * diff;
            }
            errors.push(sum_sq);
        }
        Ok(errors)
    }

    pub fn schmid_leiman(&self) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        let k = self.n_components;
        let p = self.n_features;
        let phi_mat = match &self.factor_correlations {
            Some(phi) => rows_to_matrix(phi)?,
            None => DMatrix::identity(k, k),
        };

        let eigen = phi_mat.symmetric_eigen();
        let mut pairs: Vec<(usize, f64)> = eigen.eigenvalues
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let g_idx = pairs[0].0;
        let g_val = pairs[0].1.max(0.0);
        let sqrt_g = g_val.sqrt();

        let mut gamma = vec![0.0; k];
        for c in 0..k {
            gamma[c] = eigen.eigenvectors[(c, g_idx)] * sqrt_g;
        }

        let mut l = DMatrix::zeros(p, k);
        for comp_idx in 0..k {
            for j in 0..p {
                l[(j, comp_idx)] = self.components[comp_idx][j];
            }
        }

        let mut l_g = vec![0.0; p];
        for j in 0..p {
            l_g[j] = (0..k).map(|c| l[(j, c)] * gamma[c]).sum();
        }

        let mut u_group = vec![0.0; k];
        for c in 0..k {
            u_group[c] = (1.0 - gamma[c].powi(2)).max(0.0).sqrt();
        }

        let mut bifactor = Vec::with_capacity(p);
        for j in 0..p {
            let mut row = Vec::with_capacity(k + 1);
            row.push(l_g[j]);
            for c in 0..k {
                row.push(l[(j, c)] * u_group[c]);
            }
            bifactor.push(row);
        }

        let total_g_var: f64 = l_g.iter().map(|&v| v.powi(2)).sum();
        let total_group_var: f64 = (0..p).map(|j| {
            (0..k).map(|c| (l[(j, c)] * u_group[c]).powi(2)).sum::<f64>()
        }).sum();
        let total_communal = total_g_var + total_group_var;
        let g_share = if total_communal > 1e-12 { total_g_var / total_communal } else { 0.0 };

        Ok((bifactor, vec![g_share, 1.0 - g_share]))
    }

    pub fn report(&self) -> ModelReport {
        let mut struct_map = BTreeMap::new();
        struct_map.insert("n_components".into(), self.n_components.to_string());
        struct_map.insert("n_features".into(), self.n_features.to_string());
        struct_map.insert("method".into(), match self.method.as_str() {
            "paf" => "Principal Axis Factoring (PAF)".into(),
            other => other.to_string(),
        });
        struct_map.insert("rotation".into(), match self.rotation {
            RotationMethod::None => "None".into(),
            RotationMethod::Varimax => "Varimax (stats::varimax)".into(),
            RotationMethod::Quartimax => "Quartimax".into(),
            RotationMethod::Equamax => "Equamax".into(),
            RotationMethod::Parsimax => "Parsimax".into(),
            RotationMethod::Biquartimax => "Biquartimax".into(),
            RotationMethod::Orthomax(g) => format!("Orthomax(gamma={:.2})", g),
            RotationMethod::Promax(m) => format!("Promax(power={:.1})", m),
        });

        let avg_communal: f64 = self.communalities.iter().sum::<f64>() / (self.n_features as f64);
        let avg_uniqueness: f64 = self.uniquenesses.iter().sum::<f64>() / (self.n_features as f64);

        let mut metrics = BTreeMap::new();
        metrics.insert("mean_communality".into(), avg_communal);
        metrics.insert("mean_uniqueness".into(), avg_uniqueness);
        metrics.insert("iterations".into(), self.n_iter as f64);

        ModelReport::new(
            "Exploratory Factor Analysis",
            ObjectiveSection {
                name: "Common Factor Decomposition (Sigma = Lambda*Lambda^T + Psi)".into(),
                math_form: Some("R* = R - diag(1 - h^2); R* = V * Lambda * V^T".into()),
                direction: OptimizationDirection::Reconstruct,
                final_value: Some(avg_communal),
                components: vec![
                    ("mean_communality".into(), avg_communal),
                    ("mean_uniqueness".into(), avg_uniqueness),
                ],
                units: None,
            },
            LearnerSection {
                family: "Component Analysis".into(),
                kind: "FactorAnalysis".into(),
                representation: "Common factor loadings Lambda + Uniquenesses Psi".into(),
                structure: struct_map,
                nominal_complexity: self.n_components * self.n_features + self.n_features,
                effective_complexity: Some(self.n_components as f64),
                capabilities: vec!["transform".into(), "inverse_transform".into(), "schmid_leiman".into()],
            },
            ControllerSection {
                kind: "fa".into(),
                config: BTreeMap::new(),
                seed: None,
                iteration_budget: Some(self.max_iter),
                evaluation_budget: None,
                missing_policy: "error_on_missing".into(),
                fallback_reason: None,
            },
            DiagnosticsSection {
                converged: Some(self.converged),
                stop_reason: Some(if self.converged { "Communality convergence achieved".into() } else { "Maximum iterations reached".into() }),
                fit_time_sec: None,
                eval_count: Some(self.n_iter),
                metrics,
                warnings: if self.converged {
                    Vec::new()
                } else {
                    vec!["Factor analysis communality iteration did not converge within max_iter".into()]
                },
            },
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Unified ComponentModel Facade
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "model")]
pub enum ComponentModelCore {
    Pca(PcaModel),
    Svd(TruncatedSvdModel),
    Whitening(WhiteningModel),
    Ica(FastIcaModel),
    Nmf(NmfModel),
    Fa(FactorAnalysisModel),
}

impl ComponentModelCore {
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Pca(m) => m.transform(data),
            Self::Svd(m) => m.transform(data),
            Self::Whitening(m) => m.transform(data),
            Self::Ica(m) => m.transform(data),
            Self::Nmf(m) => m.transform(data),
            Self::Fa(m) => m.transform(data),
        }
    }

    pub fn inverse_transform(&self, scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Pca(m) => m.inverse_transform(scores),
            Self::Svd(m) => m.inverse_transform(scores),
            Self::Whitening(m) => m.inverse_transform(scores),
            Self::Ica(m) => m.inverse_transform(scores),
            Self::Nmf(_) => Err("NMF does not support direct inverse_transform from arbitrary coordinates; use reconstruct".into()),
            Self::Fa(m) => m.inverse_transform(scores),
        }
    }

    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Pca(m) => m.reconstruct(data),
            Self::Svd(m) => m.reconstruct(data),
            Self::Whitening(m) => {
                let z = m.transform(data)?;
                m.inverse_transform(&z)
            }
            Self::Ica(m) => m.reconstruct(data),
            Self::Nmf(m) => m.reconstruct(data),
            Self::Fa(m) => m.reconstruct(data),
        }
    }

    pub fn reconstruction_errors(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        match self {
            Self::Pca(m) => m.reconstruction_errors(data),
            Self::Svd(m) => m.reconstruction_errors(data),
            Self::Whitening(_) => {
                let rec = self.reconstruct(data)?;
                let mut errs = Vec::with_capacity(data.len());
                for (i, row) in data.iter().enumerate() {
                    let mut s = 0.0;
                    for j in 0..row.len() {
                        let d = row[j] - rec[i][j];
                        s += d * d;
                    }
                    errs.push(s);
                }
                Ok(errs)
            }
            Self::Ica(m) => m.reconstruction_errors(data),
            Self::Nmf(m) => m.reconstruction_errors(data),
            Self::Fa(m) => m.reconstruction_errors(data),
        }
    }

    pub fn report(&self) -> ModelReport {
        match self {
            Self::Pca(m) => m.report(),
            Self::Svd(m) => m.report(),
            Self::Whitening(m) => m.report(),
            Self::Ica(m) => m.report(),
            Self::Nmf(m) => m.report(),
            Self::Fa(m) => m.report(),
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
