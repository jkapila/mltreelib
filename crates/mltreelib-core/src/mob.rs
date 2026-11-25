//! Model-Based Recursive Partitioning (MOB / Zeileis, Hothorn & Hornik 2008).
//!
//! Fits the parametric leaf model once at the parent node, evaluates individual
//! score estimating functions (derivatives of the loss with respect to parameters),
//! and tests for parameter instability along candidate covariates using cumulative
//! empirical fluctuation processes (supLM for numeric variables, Chi-square for
//! categorical variables).
//!
//! Only covariates showing statistically significant instability (Bonferroni-adjusted
//! p < alpha) are partitioned, selecting the cutpoint that maximizes the fluctuation peak.
//! Bypasses candidate-level model refitting while ensuring unbiased variable selection.

use crate::gate::GateNote;
use crate::model_tree::{Grower, NodeProfile, Split};
use crate::split::ColType;
use crate::stats::{chi2_sf, suplm_pvalue};

/// Invert a symmetric positive semi-definite matrix with adaptive ridge regularization.
/// Uses Gauss-Jordan elimination with partial pivoting.
pub fn invert_symmetric_psd(mat: &[f64], k: usize) -> Option<Vec<f64>> {
    if k == 0 {
        return Some(Vec::new());
    }
    if k == 1 {
        let v = mat[0];
        let reg_v = if v.abs() < 1e-12 {
            1e-6
        } else {
            v
        };
        return Some(vec![1.0 / reg_v]);
    }

    let mut trace = 0.0;
    for i in 0..k {
        trace += mat[i * k + i];
    }
    let ridge = 1e-7 * (trace / k as f64).max(1.0);

    // Augmented matrix [A | I] of size k x 2k
    let n_cols = 2 * k;
    let mut a = vec![0.0; k * n_cols];
    for i in 0..k {
        for j in 0..k {
            let mut val = mat[i * k + j];
            if i == j {
                val += ridge;
            }
            a[i * n_cols + j] = val;
        }
        a[i * n_cols + k + i] = 1.0;
    }

    for i in 0..k {
        // Pivot selection
        let mut pivot_row = i;
        let mut max_val = a[i * n_cols + i].abs();
        for r in (i + 1)..k {
            let v = a[r * n_cols + i].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-14 {
            // Near-singular even with ridge; fallback pseudo-inverse
            return None;
        }
        if pivot_row != i {
            for col in 0..n_cols {
                a.swap(i * n_cols + col, pivot_row * n_cols + col);
            }
        }

        let pivot = a[i * n_cols + i];
        for col in 0..n_cols {
            a[i * n_cols + col] /= pivot;
        }

        for r in 0..k {
            if r != i {
                let factor = a[r * n_cols + i];
                if factor.abs() > 1e-15 {
                    for col in 0..n_cols {
                        let sub = factor * a[i * n_cols + col];
                        a[r * n_cols + col] -= sub;
                    }
                }
            }
        }
    }

    let mut inv = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            inv[i * k + j] = a[i * n_cols + k + j];
        }
    }
    Some(inv)
}

/// Compute individual score estimating function vectors psi_r in R^k for all rows at this node.
/// For linear models: psi_r = w_r * e_r * [1, x_r,1, ..., x_r,p]
/// For mean models: psi_r = [w_r * e_r]
pub fn compute_node_scores(
    cols: &[Vec<Option<f64>>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    leaf: &crate::node_stats::LeafModel,
) -> (Vec<Vec<f64>>, usize) {
    let n = rows.len();
    if let Some(ref lm) = leaf.linear {
        if !lm.cols.is_empty() {
            let k = 1 + lm.cols.len();
            let mut scores = Vec::with_capacity(n);
            for &r in rows {
                let wt = w[r];
                let y_val = y[r];
                let pred = leaf.predict_columns(cols, r);
                let e = y_val - pred;
                let mut psi = Vec::with_capacity(k);
                psi.push(wt * e); // intercept score
                for &c in &lm.cols {
                    let x_val = cols[c][r].unwrap_or(0.0);
                    psi.push(wt * e * x_val);
                }
                scores.push(psi);
            }
            return (scores, k);
        }
    }

    // Mean / constant leaf fallback
    let center = leaf.center.unwrap_or_else(|| {
        let (mut swy, mut sw) = (0.0, 0.0);
        for &r in rows {
            let wt = w[r];
            sw += wt;
            swy += wt * y[r];
        }
        if sw > 0.0 {
            swy / sw
        } else {
            0.0
        }
    });

    let mut scores = Vec::with_capacity(n);
    for &r in rows {
        let wt = w[r];
        let e = y[r] - center;
        scores.push(vec![wt * e]);
    }
    (scores, 1)
}

/// Empirical fluctuation test along a continuous/numeric candidate variable.
/// Computes supLM statistic over candidate thresholds in [min_leaf, n - min_leaf].
pub fn test_numeric_instability(
    values: &[(f64, usize)],
    scores: &[Vec<f64>],
    v_inv: &[f64],
    k: usize,
    min_leaf: usize,
    eps: f64,
) -> Option<(f64, f64, f64)> {
    let n = values.len();
    if n < 2 * min_leaf || n < 10 {
        return None;
    }

    // S_m is the cumulative sum of score vectors up to observation m
    let mut s_m = vec![0.0; k];
    let mut best_lm = 0.0;
    let mut best_threshold = values[0].0;
    let mut found_valid = false;

    let n_f = n as f64;
    let min_m = min_leaf.max((eps * n_f).floor() as usize).max(1);
    let max_m = (n - min_leaf).min(((1.0 - eps) * n_f).ceil() as usize);

    for m in 1..n {
        let (_, row_idx) = values[m - 1];
        let psi = &scores[row_idx];
        for d in 0..k {
            s_m[d] += psi[d];
        }

        let curr_val = values[m - 1].0;
        let next_val = values[m].0;

        if m >= min_m && m <= max_m && curr_val < next_val {
            // Quadratic form: Q_m = S_m^T V_inv S_m
            let mut q_m = 0.0;
            for i in 0..k {
                let mut v_row = 0.0;
                for j in 0..k {
                    v_row += v_inv[i * k + j] * s_m[j];
                }
                q_m += s_m[i] * v_row;
            }

            let m_f = m as f64;
            let denom = (m_f * (n_f - m_f)).max(1.0);
            let lm = (n_f / denom) * q_m;

            if lm > best_lm || !found_valid {
                best_lm = lm;
                best_threshold = (curr_val + next_val) / 2.0;
                found_valid = true;
            }
        }
    }

    if !found_valid {
        return None;
    }

    let p = suplm_pvalue(best_lm, k, eps);
    Some((best_lm, best_threshold, p))
}

/// Empirical fluctuation test along a categorical candidate variable.
/// Computes Chi-square statistic across discrete levels.
pub fn test_categorical_instability(
    col: &[Option<f64>],
    rows: &[usize],
    scores: &[Vec<f64>],
    v_inv: &[f64],
    k: usize,
    min_leaf: usize,
) -> Option<(f64, Vec<f64>, f64)> {
    use std::collections::BTreeMap;

    // Group rows by category level
    let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut level_map: BTreeMap<u64, f64> = BTreeMap::new();

    for (idx, &r) in rows.iter().enumerate() {
        if let Some(v) = col[r] {
            let key = v.to_bits();
            groups.entry(key).or_default().push(idx);
            level_map.insert(key, v);
        }
    }

    if groups.len() < 2 {
        return None;
    }

    let c = groups.len();
    let df = (k * (c - 1)).max(1);
    let mut chi2 = 0.0;
    let mut level_projections: Vec<(f64, f64, usize)> = Vec::with_capacity(c);

    for (&key, member_indices) in &groups {
        let n_c = member_indices.len();
        if n_c == 0 {
            continue;
        }
        let n_c_f = n_c as f64;
        let mut t_c = vec![0.0; k];
        for &idx in member_indices {
            let psi = &scores[idx];
            for d in 0..k {
                t_c[d] += psi[d];
            }
        }

        let mut q_c = 0.0;
        let mut proj = 0.0;
        for i in 0..k {
            let mut v_row = 0.0;
            for j in 0..k {
                v_row += v_inv[i * k + j] * t_c[j];
            }
            q_c += t_c[i] * v_row;
            proj += v_row;
        }
        chi2 += q_c / n_c_f;
        let orig_val = level_map[&key];
        level_projections.push((proj / n_c_f, orig_val, n_c));
    }

    let p = chi2_sf(chi2, df as f64);

    // Partition levels based on sorted 1D score projection
    level_projections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut left_levels = Vec::new();
    let mut cum_n = 0;
    let total_n = rows.len();
    for (_proj, val, count) in &level_projections {
        cum_n += count;
        left_levels.push(*val);
        if cum_n >= min_leaf && (total_n - cum_n) >= min_leaf {
            break;
        }
    }
    if left_levels.is_empty() || left_levels.len() == level_projections.len() {
        if let Some((_, first_val, _)) = level_projections.first() {
            left_levels = vec![*first_val];
        }
    }

    Some((chi2, left_levels, p))
}

impl<'a> Grower<'a> {
    /// Model-based recursive partitioning (MOB / Zeileis et al. 2008):
    /// Fits parent leaf model once, evaluates individual score estimating functions,
    /// tests parameter instability across candidate covariates, and splits only if
    /// instability is statistically significant (Bonferroni p < alpha).
    pub(crate) fn mob_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }

        // 1. Fit parent leaf model once
        let leaf = self.make_leaf_for_scan(rows, path);

        // 2. Compute individual score estimating function vectors
        let (scores, k) = compute_node_scores(self.cols, self.y, self.w, rows, &leaf);

        // 3. Compute score covariance matrix V = (1/n) * sum_i psi_i psi_i^T
        let n_f = rows.len() as f64;
        let mut v_mat = vec![0.0; k * k];
        for psi in &scores {
            for i in 0..k {
                for j in 0..k {
                    v_mat[i * k + j] += psi[i] * psi[j];
                }
            }
        }
        for elem in &mut v_mat {
            *elem /= n_f;
        }

        // 4. Invert V matrix
        let v_inv = match invert_symmetric_psd(&v_mat, k) {
            Some(inv) => inv,
            None => return None,
        };

        // 5. Candidate partitioning variables
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

        let p_covariates = cand_cols.len() as f64;
        let min_leaf = self.min_child(rows.len());
        let eps = 0.10f64;

        enum CandidateSplit {
            Numeric { threshold: f64 },
            Categorical { left_levels: Vec<f64> },
        }

        let mut test_results: Vec<(usize, f64, f64, f64, CandidateSplit)> = Vec::new();
        let mut var_p_values: Vec<(usize, f64, f64)> = Vec::new();

        for &j in &cand_cols {
            prof.cols += 1;
            match self.params.dtypes.get(j).copied().unwrap_or(ColType::Float) {
                ColType::Float | ColType::Integer => {
                    let mut vals: Vec<(f64, usize)> = Vec::with_capacity(rows.len());
                    for (idx, &r) in rows.iter().enumerate() {
                        if let Some(v) = self.cols[j][r] {
                            vals.push((v, idx));
                        }
                    }
                    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    if let Some((stat, th, raw_p)) = test_numeric_instability(
                        &vals, &scores, &v_inv, k, min_leaf, eps,
                    ) {
                        let adj_p = if self.params.bonferroni {
                            (raw_p * p_covariates).min(1.0)
                        } else {
                            raw_p
                        };
                        var_p_values.push((j, stat, adj_p));
                        test_results.push((j, stat, raw_p, adj_p, CandidateSplit::Numeric { threshold: th }));
                    }
                }
                ColType::Categorical => {
                    if let Some((stat, left_levels, raw_p)) = test_categorical_instability(
                        &self.cols[j], rows, &scores, &v_inv, k, min_leaf,
                    ) {
                        let adj_p = if self.params.bonferroni {
                            (raw_p * p_covariates).min(1.0)
                        } else {
                            raw_p
                        };
                        var_p_values.push((j, stat, adj_p));
                        test_results.push((j, stat, raw_p, adj_p, CandidateSplit::Categorical { left_levels }));
                    }
                }
            }
        }

        if test_results.is_empty() {
            return None;
        }

        // Sort candidates by raw p-value ascending, then statistic descending
        test_results.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let (best_col, best_stat, _best_raw_p, best_adj_p, best_cand) = &test_results[0];
        let alpha = self.params.split_alpha;
        let passed = *best_adj_p < alpha;

        prof.gate = Some(GateNote {
            f: *best_stat,
            p: *best_adj_p,
            alpha,
            passed,
            test_name: "mob",
            var_p_values,
        });

        if !passed {
            return None;
        }

        let mut sp = Split {
            col: *best_col,
            threshold: 0.0,
            missing_left: true,
            gain: *best_stat,
            cat: None,
            sur: None,
            oblique: None,
        };

        match best_cand {
            CandidateSplit::Numeric { threshold } => {
                sp.threshold = *threshold;
            }
            CandidateSplit::Categorical { left_levels } => {
                sp.cat = Some(self.cat_parts.len());
                self.cat_parts.push(left_levels.clone());
            }
        }

        let (lv, rv) = self.partition_rows(rows, &sp);
        if lv.len() < min_leaf || rv.len() < min_leaf {
            return None;
        }

        Some((sp, lv, rv))
    }
}
