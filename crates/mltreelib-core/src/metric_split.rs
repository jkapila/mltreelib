//! Metric and Kernel Splitting Criteria (Energy Distance & MMD).
//!
//! Splitting for multi-output, high-dimensional vector targets Y in R^D,
//! manifolds, and complex probability distributions without parametric assumptions:
//!
//! 1. Energy Distance (Székely & Rizzo 2004, 2013):
//!    E(L, R) = 2 E||Y_L - Y_R|| - E||Y_L - Y_L'|| - E||Y_R - Y_R'||
//!    Maximizes dispersion reduction across multi-target metric spaces.
//!
//! 2. Maximum Mean Discrepancy (MMD / Gretton et al. 2012):
//!    MMD^2(L, R) = E[k(Y_L, Y_L')] + E[k(Y_R, Y_R')] - 2 E[k(Y_L, Y_R)]
//!    With characteristic kernels (RBF), distinguishes any difference in
//!    distribution (including identical means with orthogonal covariances).

use std::collections::HashMap;

use crate::criteria::{GainFn, KernelKind};
use crate::model_tree::{Grower, NodeProfile, Split};
use crate::split::ColType;

/// Evaluates scalar kernel function between two vectors.
pub fn eval_kernel(v1: &[f64], v2: &[f64], kernel: KernelKind, bandwidth: f64) -> f64 {
    let bw = if bandwidth > 0.0 { bandwidth } else { 1.0 };
    match kernel {
        KernelKind::Rbf => {
            let mut sum_sq = 0.0;
            for (a, b) in v1.iter().zip(v2.iter()) {
                let d = a - b;
                sum_sq += d * d;
            }
            (-sum_sq / (2.0 * bw * bw)).exp()
        }
        KernelKind::Linear => {
            let mut dot = 0.0;
            for (a, b) in v1.iter().zip(v2.iter()) {
                dot += a * b;
            }
            dot
        }
        KernelKind::Polynomial { degree, c } => {
            let mut dot = 0.0;
            for (a, b) in v1.iter().zip(v2.iter()) {
                dot += a * b;
            }
            (dot + c).powi(degree as i32)
        }
    }
}

/// Evaluates Euclidean distance between two vectors.
pub fn eval_euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for (a, b) in v1.iter().zip(v2.iter()) {
        let d = a - b;
        sum_sq += d * d;
    }
    sum_sq.sqrt()
}

/// Median heuristic for RBF bandwidth estimation (Gretton et al. 2012).
pub fn estimate_median_bandwidth(targets: &[&[f64]]) -> f64 {
    let n = targets.len();
    if n < 2 {
        return 1.0;
    }
    let mut distances = Vec::new();
    let step = (n / 50).max(1);
    for i in (0..n).step_by(step) {
        for j in (i + 1..n).step_by(step) {
            distances.push(eval_euclidean_distance(targets[i], targets[j]));
            if distances.len() > 1000 {
                break;
            }
        }
        if distances.len() > 1000 {
            break;
        }
    }
    if distances.is_empty() {
        return 1.0;
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = distances[distances.len() / 2];
    if median > 1e-6 { median } else { 1.0 }
}

/// Computes Random Fourier Features (Rahimi & Recht 2007) for RBF kernel:
/// phi(y) in R^{2R} where phi_2k = cos(w_k^T y) / sqrt(R), phi_{2k+1} = sin(w_k^T y) / sqrt(R).
pub fn compute_rff_embeddings(
    targets: &[&[f64]],
    num_components: usize,
    bandwidth: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let d = if targets.is_empty() { 0 } else { targets[0].len() };
    if d == 0 {
        return Vec::new();
    }
    let r = num_components.max(8);
    let bw = if bandwidth > 0.0 { bandwidth } else { 1.0 };
    let std_dev = 1.0 / bw;

    let mut w_mat = Vec::with_capacity(r);
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for _ in 0..r {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            row.push(z0 * std_dev);
        }
        w_mat.push(row);
    }

    let norm_factor = (1.0 / (r as f64)).sqrt();
    let mut embeddings = Vec::with_capacity(targets.len());
    for target in targets {
        let mut phi = Vec::with_capacity(2 * r);
        for row in &w_mat {
            let mut dot = 0.0;
            for (w_val, y_val) in row.iter().zip(target.iter()) {
                dot += w_val * y_val;
            }
            phi.push(dot.cos() * norm_factor);
            phi.push(dot.sin() * norm_factor);
        }
        embeddings.push(phi);
    }
    embeddings
}

impl<'a> Grower<'a> {
    /// Fast metric splitting (Energy Distance / MMD) on multi-dimensional vector targets Y in R^D.
    pub(crate) fn metric_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }

        let target_vecs = self.target_vectors?;
        let m_rows = rows.len();

        let cand_cols: Vec<usize> = self
            .active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .filter(|&j| {
                self.params
                    .split_cols
                    .as_ref()
                    .map(|sc| sc.contains(&j))
                    .unwrap_or(true)
            })
            .collect();

        if cand_cols.is_empty() {
            return None;
        }

        let eff_bw = match self.params.gain {
            GainFn::Mmd { kernel: _, bandwidth } => {
                if bandwidth <= 0.0 {
                    let sample_targets: Vec<&[f64]> = rows.iter().map(|&r| target_vecs[r].as_slice()).collect();
                    estimate_median_bandwidth(&sample_targets)
                } else {
                    bandwidth
                }
            }
            _ => 1.0,
        };

        if m_rows >= 1000 && matches!(self.params.gain, GainFn::Mmd { kernel: KernelKind::Rbf, .. }) {
            return self.rff_mmd_split(rows, path, prof, &cand_cols, eff_bw);
        }

        // 1. Precompute pairwise affinity matrix A[i][j] for rows in this node
        // Energy Distance uses A_ij = -||y_i - y_j||, MMD uses A_ij = k(y_i, y_j)
        let mut a_mat = vec![vec![0.0; m_rows]; m_rows];
        for i in 0..m_rows {
            let r_i = rows[i];
            let v_i = &target_vecs[r_i];
            for j in i..m_rows {
                let r_j = rows[j];
                let v_j = &target_vecs[r_j];
                let val = match self.params.gain {
                    GainFn::EnergyDistance => -eval_euclidean_distance(v_i, v_j),
                    GainFn::Mmd { kernel, .. } => eval_kernel(v_i, v_j, kernel, eff_bw),
                    _ => 0.0,
                };
                a_mat[i][j] = val;
                a_mat[j][i] = val;
            }
        }

        let w_node: Vec<f64> = rows.iter().map(|&r| self.w[r]).collect();
        let mut s_total = 0.0;
        let mut w_total = 0.0;
        for i in 0..m_rows {
            let wi = w_node[i];
            w_total += wi;
            for j in 0..m_rows {
                s_total += wi * w_node[j] * a_mat[i][j];
            }
        }

        if w_total <= 0.0 {
            return None;
        }

        let min_leaf = self.min_child(m_rows);
        let mut best_gain = self.params.min_gain;
        let mut best_split: Option<Split> = None;
        let mut best_left_levels: Option<Vec<f64>> = None;

        for &col_idx in &cand_cols {
            prof.cols += 1;
            match self.col_type(col_idx) {
                ColType::Float | ColType::Integer => {
                    let mut vals: Vec<(f64, usize)> = Vec::with_capacity(m_rows);
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(x) = self.cols[col_idx][r] {
                            vals.push((x, idx));
                        }
                    }
                    if vals.len() < 2 * min_leaf {
                        continue;
                    }
                    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    let mut s_ll = 0.0;
                    let mut w_l = 0.0;
                    let mut v_l = vec![0.0; m_rows];

                    for m in 0..(vals.len() - 1) {
                        let (x_curr, local_i) = vals[m];
                        let wt = w_node[local_i];
                        w_l += wt;
                        s_ll += 2.0 * wt * v_l[local_i] + wt * wt * a_mat[local_i][local_i];
                        for j in 0..m_rows {
                            v_l[j] += wt * a_mat[local_i][j];
                        }

                        let n_l = m + 1;
                        let n_r = vals.len() - n_l;
                        let x_next = vals[m + 1].0;

                        if x_curr < x_next && n_l >= min_leaf && n_r >= min_leaf {
                            let w_r = w_total - w_l;
                            if w_l > 0.0 && w_r > 0.0 {
                                let mut sum_vl_w = 0.0;
                                for j in 0..m_rows {
                                    sum_vl_w += v_l[j] * w_node[j];
                                }
                                let s_lr = sum_vl_w - s_ll;
                                let s_rr = s_total - s_ll - 2.0 * s_lr;

                                let div = (s_ll / (w_l * w_l)) + (s_rr / (w_r * w_r)) - (2.0 * s_lr / (w_l * w_r));
                                let mut gain = (w_l * w_r / (w_l + w_r)) * div;
                                if self.params.imbalance_penalty > 0.0 {
                                    gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                                }

                                if gain > best_gain {
                                    best_gain = gain;
                                    best_split = Some(Split {
                                        col: col_idx,
                                        threshold: (x_curr + x_next) / 2.0,
                                        missing_left: true,
                                        gain,
                                        cat: None,
                                        sur: None,
                                        oblique: None,
                                    });
                                    best_left_levels = None;
                                }
                            }
                        }
                    }
                }
                ColType::Categorical => {
                    let mut cat_map: HashMap<u64, Vec<usize>> = HashMap::new();
                    let mut cat_orig: HashMap<u64, f64> = HashMap::new();
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(v) = self.cols[col_idx][r] {
                            let key = v.to_bits();
                            cat_map.entry(key).or_default().push(idx);
                            cat_orig.insert(key, v);
                        }
                    }
                    if cat_map.len() < 2 {
                        continue;
                    }

                    let categories: Vec<u64> = cat_map.keys().copied().collect();
                    let n_cats = categories.len();
                    let mut b_mat = vec![vec![0.0; n_cats]; n_cats];
                    let mut w_cat = vec![0.0; n_cats];
                    let mut n_cat = vec![0; n_cats];

                    for (c1, &k1) in categories.iter().enumerate() {
                        let list1 = &cat_map[&k1];
                        n_cat[c1] = list1.len();
                        for &i in list1 {
                            w_cat[c1] += w_node[i];
                        }
                        for (c2, &k2) in categories.iter().enumerate() {
                            let list2 = &cat_map[&k2];
                            let mut b_sum = 0.0;
                            for &i in list1 {
                                for &j in list2 {
                                    b_sum += w_node[i] * w_node[j] * a_mat[i][j];
                                }
                            }
                            b_mat[c1][c2] = b_sum;
                        }
                    }

                    for c in 0..n_cats {
                        let n_l = n_cat[c];
                        let n_r = m_rows - n_l;
                        if n_l >= min_leaf && n_r >= min_leaf {
                            let w_l = w_cat[c];
                            let w_r = w_total - w_l;
                            if w_l > 0.0 && w_r > 0.0 {
                                let s_ll = b_mat[c][c];
                                let mut s_lr = 0.0;
                                for other in 0..n_cats {
                                    if other != c {
                                        s_lr += b_mat[c][other];
                                    }
                                }
                                let s_rr = s_total - s_ll - 2.0 * s_lr;

                                let div = (s_ll / (w_l * w_l)) + (s_rr / (w_r * w_r)) - (2.0 * s_lr / (w_l * w_r));
                                let mut gain = (w_l * w_r / (w_l + w_r)) * div;
                                if self.params.imbalance_penalty > 0.0 {
                                    gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                                }

                                if gain > best_gain {
                                    best_gain = gain;
                                    best_split = Some(Split {
                                        col: col_idx,
                                        threshold: 0.0,
                                        missing_left: true,
                                        gain,
                                        cat: None,
                                        sur: None,
                                        oblique: None,
                                    });
                                    best_left_levels = Some(vec![cat_orig[&categories[c]]]);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut sp = best_split?;
        if let Some(left_levels) = best_left_levels {
            sp.cat = Some(self.cat_parts.len());
            self.cat_parts.push(left_levels);
        }

        let (lv, rv) = self.partition_rows(rows, &sp);
        if lv.len() < min_leaf || rv.len() < min_leaf {
            return None;
        }

        Some((sp, lv, rv))
    }

    /// Fast Random Fourier Features (RFF) approximation for large-sample MMD (Gretton et al. 2012, Rahimi & Recht 2007).
    /// Bypasses O(N^2) pairwise kernel matrix allocation with O(N * R) running vector projection scans.
    pub(crate) fn rff_mmd_split(
        &mut self,
        rows: &[usize],
        _path: &[usize],
        prof: &mut NodeProfile,
        cand_cols: &[usize],
        eff_bw: f64,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        let target_vecs = self.target_vectors?;
        let m_rows = rows.len();
        let sample_targets: Vec<&[f64]> = rows.iter().map(|&r| target_vecs[r].as_slice()).collect();
        let rff_phi = compute_rff_embeddings(&sample_targets, 64, eff_bw, self.params.seed);
        let d = if rff_phi.is_empty() { 0 } else { rff_phi[0].len() };
        if d == 0 {
            return None;
        }

        let mut s_total = vec![0.0; d];
        let mut w_total = 0.0;
        for i in 0..m_rows {
            let r_i = rows[i];
            let wt = self.w[r_i];
            w_total += wt;
            for k in 0..d {
                s_total[k] += wt * rff_phi[i][k];
            }
        }
        if w_total <= 0.0 {
            return None;
        }

        let min_leaf = self.min_child(m_rows);
        let mut best_gain = self.params.min_gain;
        let mut best_split: Option<Split> = None;
        let mut best_left_levels: Option<Vec<f64>> = None;

        for &col_idx in cand_cols {
            prof.cols += 1;
            match self.col_type(col_idx) {
                ColType::Float | ColType::Integer => {
                    let mut vals: Vec<(f64, usize)> = Vec::with_capacity(m_rows);
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(x) = self.cols[col_idx][r] {
                            vals.push((x, idx));
                        }
                    }
                    if vals.len() < 2 * min_leaf {
                        continue;
                    }
                    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    let mut w_l = 0.0;
                    let mut s_l = vec![0.0; d];

                    for m in 0..(vals.len() - 1) {
                        let (x_curr, local_i) = vals[m];
                        let r_i = rows[local_i];
                        let wt = self.w[r_i];
                        w_l += wt;
                        for k in 0..d {
                            s_l[k] += wt * rff_phi[local_i][k];
                        }

                        let n_l = m + 1;
                        let n_r = vals.len() - n_l;
                        let x_next = vals[m + 1].0;

                        if x_curr < x_next && n_l >= min_leaf && n_r >= min_leaf {
                            let w_r = w_total - w_l;
                            if w_l > 0.0 && w_r > 0.0 {
                                let mut diff_norm_sq = 0.0;
                                for k in 0..d {
                                    let diff = (s_l[k] / w_l) - ((s_total[k] - s_l[k]) / w_r);
                                    diff_norm_sq += diff * diff;
                                }
                                let mut gain = (w_l * w_r / (w_l + w_r)) * diff_norm_sq;
                                if self.params.imbalance_penalty > 0.0 {
                                    gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                                }

                                if gain > best_gain {
                                    best_gain = gain;
                                    best_split = Some(Split {
                                        col: col_idx,
                                        threshold: (x_curr + x_next) / 2.0,
                                        missing_left: true,
                                        gain,
                                        cat: None,
                                        sur: None,
                                        oblique: None,
                                    });
                                    best_left_levels = None;
                                }
                            }
                        }
                    }
                }
                ColType::Categorical => {
                    let mut cat_map: HashMap<u64, Vec<usize>> = HashMap::new();
                    let mut cat_orig: HashMap<u64, f64> = HashMap::new();
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(x) = self.cols[col_idx][r] {
                            let bits = x.to_bits();
                            cat_map.entry(bits).or_default().push(idx);
                            cat_orig.insert(bits, x);
                        }
                    }
                    if cat_map.len() < 2 {
                        continue;
                    }
                    let categories: Vec<u64> = cat_map.keys().copied().collect();
                    for c in 0..categories.len() {
                        let left_indices = &cat_map[&categories[c]];
                        let n_l = left_indices.len();
                        let n_r = m_rows - n_l;
                        if n_l < min_leaf || n_r < min_leaf {
                            continue;
                        }
                        let mut w_l = 0.0;
                        let mut s_l = vec![0.0; d];
                        for &idx in left_indices {
                            let r_i = rows[idx];
                            let wt = self.w[r_i];
                            w_l += wt;
                            for k in 0..d {
                                s_l[k] += wt * rff_phi[idx][k];
                            }
                        }
                        let w_r = w_total - w_l;
                        if w_l > 0.0 && w_r > 0.0 {
                            let mut diff_norm_sq = 0.0;
                            for k in 0..d {
                                let diff = (s_l[k] / w_l) - ((s_total[k] - s_l[k]) / w_r);
                                diff_norm_sq += diff * diff;
                            }
                            let mut gain = (w_l * w_r / (w_l + w_r)) * diff_norm_sq;
                            if self.params.imbalance_penalty > 0.0 {
                                gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                            }
                            if gain > best_gain {
                                best_gain = gain;
                                best_split = Some(Split {
                                    col: col_idx,
                                    threshold: 0.0,
                                    missing_left: true,
                                    gain,
                                    cat: None,
                                    sur: None,
                                    oblique: None,
                                });
                                best_left_levels = Some(vec![cat_orig[&categories[c]]]);
                            }
                        }
                    }
                }
            }
        }

        let mut sp = best_split?;
        if let Some(left_levels) = best_left_levels {
            let cat_idx = self.cat_parts.len();
            self.cat_parts.push(left_levels);
            sp.cat = Some(cat_idx);
        }

        let (lv, rv) = self.partition_rows(rows, &sp);
        if lv.len() < min_leaf || rv.len() < min_leaf {
            return None;
        }

        Some((sp, lv, rv))
    }

    /// Fast Wasserstein-2 Optimal Transport splitting on probability distributions / quantile vectors Y in R^D.
    /// Evaluates Fréchet barycenter dispersion reduction:
    /// Gain_W2(L, R) = (W_L * W_R / (W_L + W_R)) * (1 / D) * ||barycenter_L - barycenter_R||_2^2
    pub(crate) fn wasserstein_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }

        let target_vecs = self.target_vectors?;
        let m_rows = rows.len();
        if m_rows == 0 {
            return None;
        }
        let d = target_vecs[rows[0]].len();
        if d == 0 {
            return None;
        }

        let cand_cols: Vec<usize> = self
            .active
            .iter()
            .copied()
            .filter(|&j| self.interact_ok(path, j))
            .filter(|&j| {
                self.params
                    .split_cols
                    .as_ref()
                    .map(|sc| sc.contains(&j))
                    .unwrap_or(true)
            })
            .collect();

        if cand_cols.is_empty() {
            return None;
        }

        let mut w_total = 0.0;
        let mut s_total = vec![0.0; d];
        let mut s_sq_total = 0.0;
        let mut w_total_prot = 0.0;
        let prot_col_opt = self.params.protected_col;

        for &r in rows {
            let wt = self.w[r];
            w_total += wt;
            let v = &target_vecs[r];
            for k in 0..d {
                let vk = v[k];
                s_total[k] += wt * vk;
                s_sq_total += wt * vk * vk;
            }
            if let Some(pc) = prot_col_opt {
                if self.cols[pc][r].unwrap_or(0.0) >= 0.5 {
                    w_total_prot += wt;
                }
            }
        }

        if w_total <= 0.0 {
            return None;
        }

        let s_norm_sq: f64 = s_total.iter().map(|&x| x * x).sum();
        let parent_dispersion = (s_sq_total - s_norm_sq / w_total) / (d as f64);
        if parent_dispersion <= 1e-12 {
            return None;
        }

        let min_leaf = self.min_child(m_rows);
        let mut best_gain = self.params.min_gain;
        let mut best_split: Option<Split> = None;
        let mut best_left_levels: Option<Vec<f64>> = None;

        for &col_idx in &cand_cols {
            prof.cols += 1;
            match self.col_type(col_idx) {
                ColType::Float | ColType::Integer => {
                    let mut vals: Vec<(f64, usize)> = Vec::with_capacity(m_rows);
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(x) = self.cols[col_idx][r] {
                            vals.push((x, idx));
                        }
                    }
                    if vals.len() < 2 * min_leaf {
                        continue;
                    }
                    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    let mut w_l = 0.0;
                    let mut s_l = vec![0.0; d];
                    let mut w_l_prot = 0.0;

                    for m in 0..(vals.len() - 1) {
                        let (x_curr, local_i) = vals[m];
                        let r_i = rows[local_i];
                        let wt = self.w[r_i];
                        let v = &target_vecs[r_i];
                        w_l += wt;
                        for k in 0..d {
                            s_l[k] += wt * v[k];
                        }
                        if let Some(pc) = prot_col_opt {
                            if self.cols[pc][r_i].unwrap_or(0.0) >= 0.5 {
                                w_l_prot += wt;
                            }
                        }

                        let n_l = m + 1;
                        let n_r = vals.len() - n_l;
                        let x_next = vals[m + 1].0;

                        if x_curr < x_next && n_l >= min_leaf && n_r >= min_leaf {
                            let w_r = w_total - w_l;
                            if w_l > 0.0 && w_r > 0.0 {
                                let mut diff_norm_sq = 0.0;
                                for k in 0..d {
                                    let diff = (s_l[k] / w_l) - ((s_total[k] - s_l[k]) / w_r);
                                    diff_norm_sq += diff * diff;
                                }
                                let mut gain = (w_l * w_r / (w_l + w_r)) * (diff_norm_sq / (d as f64));
                                if self.params.imbalance_penalty > 0.0 {
                                    gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                                }
                                if self.params.fairness_penalty > 0.0 && prot_col_opt.is_some() {
                                    let p_l = w_l_prot / w_l;
                                    let p_r = (w_total_prot - w_l_prot) / w_r;
                                    let disparity = (p_l - p_r).abs();
                                    gain -= self.params.fairness_penalty * parent_dispersion * disparity;
                                }

                                if gain > best_gain {
                                    best_gain = gain;
                                    best_split = Some(Split {
                                        col: col_idx,
                                        threshold: (x_curr + x_next) / 2.0,
                                        missing_left: true,
                                        gain,
                                        cat: None,
                                        sur: None,
                                        oblique: None,
                                    });
                                    best_left_levels = None;
                                }
                            }
                        }
                    }
                }
                ColType::Categorical => {
                    let mut cat_map: HashMap<u64, Vec<usize>> = HashMap::new();
                    let mut cat_orig: HashMap<u64, f64> = HashMap::new();
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(v) = self.cols[col_idx][r] {
                            let key = v.to_bits();
                            cat_map.entry(key).or_default().push(idx);
                            cat_orig.insert(key, v);
                        }
                    }
                    if cat_map.len() < 2 {
                        continue;
                    }

                    // For each category, compute w_c, s_c, and w_c_prot
                    let categories: Vec<u64> = cat_map.keys().copied().collect();
                    let mut cat_stats: Vec<(u64, usize, f64, Vec<f64>, f64)> = Vec::new();
                    let u_dir: Vec<f64> = s_total.iter().map(|&x| x / w_total).collect();

                    for &k in &categories {
                        let list = &cat_map[&k];
                        let mut w_c = 0.0;
                        let mut s_c = vec![0.0; d];
                        let mut w_c_prot = 0.0;
                        for &loc_i in list {
                            let r = rows[loc_i];
                            let wt = self.w[r];
                            w_c += wt;
                            let v = &target_vecs[r];
                            for kk in 0..d {
                                s_c[kk] += wt * v[kk];
                            }
                            if let Some(pc) = prot_col_opt {
                                if self.cols[pc][r].unwrap_or(0.0) >= 0.5 {
                                    w_c_prot += wt;
                                }
                            }
                        }
                        cat_stats.push((k, list.len(), w_c, s_c, w_c_prot));
                    }

                    // Sort categories by 1D projection score: (s_c . u_dir) / w_c
                    cat_stats.sort_by(|a, b| {
                        let score_a = if a.2 > 0.0 { a.3.iter().zip(&u_dir).map(|(x, y)| x * y).sum::<f64>() / a.2 } else { 0.0 };
                        let score_b = if b.2 > 0.0 { b.3.iter().zip(&u_dir).map(|(x, y)| x * y).sum::<f64>() / b.2 } else { 0.0 };
                        score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let mut w_l = 0.0;
                    let mut s_l = vec![0.0; d];
                    let mut w_l_prot = 0.0;
                    let mut n_l = 0;
                    let mut left_levels = Vec::new();

                    for c in 0..(cat_stats.len() - 1) {
                        let (k, cnt, wt, ref s_c, wt_prot) = cat_stats[c];
                        n_l += cnt;
                        let n_r = m_rows - n_l;
                        w_l += wt;
                        w_l_prot += wt_prot;
                        for kk in 0..d {
                            s_l[kk] += s_c[kk];
                        }
                        left_levels.push(cat_orig[&k]);

                        if n_l >= min_leaf && n_r >= min_leaf {
                            let w_r = w_total - w_l;
                            if w_l > 0.0 && w_r > 0.0 {
                                let mut diff_norm_sq = 0.0;
                                for kk in 0..d {
                                    let diff = (s_l[kk] / w_l) - ((s_total[kk] - s_l[kk]) / w_r);
                                    diff_norm_sq += diff * diff;
                                }
                                let mut gain = (w_l * w_r / (w_l + w_r)) * (diff_norm_sq / (d as f64));
                                if self.params.imbalance_penalty > 0.0 {
                                    gain -= self.params.imbalance_penalty * (1.0 / n_l as f64 + 1.0 / n_r as f64);
                                }
                                if self.params.fairness_penalty > 0.0 && prot_col_opt.is_some() {
                                    let p_l = w_l_prot / w_l;
                                    let p_r = (w_total_prot - w_l_prot) / w_r;
                                    let disparity = (p_l - p_r).abs();
                                    gain -= self.params.fairness_penalty * parent_dispersion * disparity;
                                }

                                if gain > best_gain {
                                    best_gain = gain;
                                    best_split = Some(Split {
                                        col: col_idx,
                                        threshold: 0.0,
                                        missing_left: true,
                                        gain,
                                        cat: None,
                                        sur: None,
                                        oblique: None,
                                    });
                                    best_left_levels = Some(left_levels.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut sp = best_split?;
        if let Some(left_levels) = best_left_levels {
            sp.cat = Some(self.cat_parts.len());
            self.cat_parts.push(left_levels);
        }

        let (lv, rv) = self.partition_rows(rows, &sp);
        if lv.len() < min_leaf || rv.len() < min_leaf {
            return None;
        }

        Some((sp, lv, rv))
    }
}
