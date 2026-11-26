//! Conditional Inference Trees (CTree / Hothorn, Hornik & Zeileis 2006).
//!
//! Unbiased recursive partitioning using permutation hypothesis testing:
//! 1. Test of Independence: For each candidate covariate X_j, test the
//!    null hypothesis H_0^j: D(Y | X_j) = D(Y) against alternatives using
//!    linear permutation statistics T_j.
//! 2. Variable Selection: Standardize T_j under the conditional permutation
//!    distribution. Compute asymptotic p-values with Bonferroni correction.
//!    Select variable j* = argmin_j p_j. If min_j p_adj_j >= split_alpha, stop (terminal leaf).
//! 3. Splitpoint Selection: For chosen variable j*, choose threshold s*
//!    maximizing the two-sample linear rank statistic between children.
//!
//! This completely eliminates selection bias toward variables with many
//! split points or high category cardinality (Loh's cardinality trap).

use std::collections::BTreeMap;

use crate::gate::GateNote;
use crate::model_tree::{Grower, NodeProfile, Split};
use crate::split::ColType;
use crate::stats::chi2_sf;

/// Computes the standardized permutation linear rank statistic and asymptotic p-value
/// for a numeric covariate X_j against target Y under the conditional null hypothesis.
pub fn compute_ctree_test_numeric(
    col: &[Option<f64>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    y_mean: f64,
    vy: f64,
) -> Option<(f64, f64, f64)> {
    if vy <= 1e-12 {
        return None;
    }

    let mut w_col = 0.0;
    let mut sum_wx = 0.0;
    let mut count = 0;

    for &r in rows {
        if let Some(x) = col[r] {
            let wt = w[r];
            w_col += wt;
            sum_wx += wt * x;
            count += 1;
        }
    }

    if count < 2 || w_col <= 1.0 {
        return None;
    }

    let x_mean = sum_wx / w_col;
    let mut vx = 0.0;
    let mut cov_xy = 0.0;

    for &r in rows {
        if let Some(x) = col[r] {
            let wt = w[r];
            let dx = x - x_mean;
            vx += wt * dx * dx;
            cov_xy += wt * dx * (y[r] - y_mean);
        }
    }

    if vx <= 1e-12 {
        return None;
    }

    // Standardized statistic: z = cov_xy / sqrt(vy * vx)
    // Quadratic form: c = z^2 ~ Chi2(1)
    let c = (cov_xy * cov_xy) / (vy * vx);
    let p = chi2_sf(c, 1.0);
    Some((c, 1.0, p))
}

/// Computes the standardized permutation linear rank statistic and asymptotic p-value
/// for a categorical covariate X_j (with C levels) against target Y under the conditional null hypothesis.
pub fn compute_ctree_test_categorical(
    col: &[Option<f64>],
    y: &[f64],
    w: &[f64],
    rows: &[usize],
    y_mean: f64,
    vy: f64,
) -> Option<(f64, f64, f64)> {
    if vy <= 1e-12 {
        return None;
    }

    // Group rows by category level
    let mut level_w: BTreeMap<u64, f64> = BTreeMap::new();
    let mut level_sy: BTreeMap<u64, f64> = BTreeMap::new();

    for &r in rows {
        if let Some(v) = col[r] {
            let key = v.to_bits();
            let wt = w[r];
            *level_w.entry(key).or_insert(0.0) += wt;
            *level_sy.entry(key).or_insert(0.0) += wt * y[r];
        }
    }

    let c_levels = level_w.len();
    if c_levels < 2 {
        return None;
    }

    // Between-group sum of squares (SSB)
    let mut ssb = 0.0;
    for (key, &wk) in &level_w {
        if wk > 0.0 {
            let mean_k = level_sy[key] / wk;
            let dy = mean_k - y_mean;
            ssb += wk * dy * dy;
        }
    }

    // Quadratic form: c = SSB / vy ~ Chi2(C - 1)
    let df = (c_levels - 1) as f64;
    let c = ssb / vy;
    let p = chi2_sf(c, df);
    Some((c, df, p))
}

impl<'a> Grower<'a> {
    /// Conditional Inference Trees (CTree / Hothorn, Hornik & Zeileis 2006):
    /// 1. Evaluates independence test p-values for all candidate covariates against target Y.
    /// 2. Applies Bonferroni adjustment and selects j* = argmin_j p_j.
    ///    If min_j p_adj_j >= split_alpha, stops partitioning (node remains leaf).
    /// 3. Chooses optimal split point s* on selected variable j* maximizing
    ///    the two-sample linear rank statistic between children.
    pub(crate) fn ctree_split(
        &mut self,
        rows: &[usize],
        path: &[usize],
        prof: &mut NodeProfile,
    ) -> Option<(Split, Vec<usize>, Vec<usize>)> {
        if rows.len() < self.min_split() {
            return None;
        }

        // 1. Compute node target statistics
        let mut w_total = 0.0;
        let mut sum_y = 0.0;
        for &r in rows {
            let wt = self.w[r];
            w_total += wt;
            sum_y += wt * self.y[r];
        }

        if w_total <= 1.0 {
            return None;
        }

        let y_mean = sum_y / w_total;
        let mut sst = 0.0;
        for &r in rows {
            let wt = self.w[r];
            let dy = self.y[r] - y_mean;
            sst += wt * dy * dy;
        }

        if sst <= 1e-12 {
            return None;
        }

        let vy = sst / (w_total - 1.0);

        // 2. Filter candidate columns
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

        let mut test_results: Vec<(usize, f64, f64, f64)> = Vec::new();
        let mut var_p_values: Vec<(usize, f64, f64)> = Vec::new();

        // 3. Step 1: Variable Selection via Conditional Independence Tests
        for &j in &cand_cols {
            prof.cols += 1;
            let dtype = self.params.dtypes.get(j).copied().unwrap_or(ColType::Float);
            let test_res = match dtype {
                ColType::Float | ColType::Integer => {
                    compute_ctree_test_numeric(&self.cols[j], self.y, self.w, rows, y_mean, vy)
                }
                ColType::Categorical => {
                    compute_ctree_test_categorical(&self.cols[j], self.y, self.w, rows, y_mean, vy)
                }
            };

            if let Some((stat, _df, raw_p)) = test_res {
                let adj_p = if self.params.bonferroni {
                    (raw_p * p_covariates).min(1.0)
                } else {
                    raw_p
                };
                var_p_values.push((j, stat, adj_p));
                test_results.push((j, stat, raw_p, adj_p));
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

        let (best_col, best_stat, _best_raw_p, best_adj_p) = test_results[0];
        let alpha = self.params.split_alpha;
        let passed = best_adj_p < alpha;

        prof.gate = Some(GateNote {
            f: best_stat,
            p: best_adj_p,
            alpha,
            passed,
            test_name: "ctree",
            var_p_values,
        });

        if !passed {
            return None;
        }

        // 4. Step 2: Split Point Selection on Chosen Variable best_col
        let dtype = self.params.dtypes.get(best_col).copied().unwrap_or(ColType::Float);
        let mut sp = Split {
            col: best_col,
            threshold: 0.0,
            missing_left: true,
            gain: best_stat,
            cat: None,
            sur: None,
            oblique: None,
        };

        match dtype {
            ColType::Float | ColType::Integer => {
                let mut vals: Vec<(f64, f64, f64)> = Vec::with_capacity(rows.len());
                for &r in rows {
                    if let Some(x) = self.cols[best_col][r] {
                        vals.push((x, self.y[r], self.w[r]));
                    }
                }
                vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let n = vals.len();
                if n < 2 * min_leaf {
                    return None;
                }

                let mut w_l = 0.0;
                let mut sy_l = 0.0;
                let mut best_split_stat = 0.0;
                let mut best_threshold = vals[0].0;
                let mut found = false;

                for m in 0..(n - 1) {
                    let (x_curr, y_curr, w_curr) = vals[m];
                    w_l += w_curr;
                    sy_l += w_curr * y_curr;
                    let n_l = m + 1;
                    let n_r = n - n_l;

                    let x_next = vals[m + 1].0;
                    if x_curr < x_next && n_l >= min_leaf && n_r >= min_leaf {
                        let w_r = w_total - w_l;
                        let sy_r = sum_y - sy_l;

                        let mean_l = sy_l / w_l;
                        let mean_r = sy_r / w_r;
                        let dy = mean_l - mean_r;
                        let ssb = (w_l * w_r / w_total) * dy * dy;
                        let stat = ssb / vy;

                        if stat > best_split_stat || !found {
                            best_split_stat = stat;
                            best_threshold = (x_curr + x_next) / 2.0;
                            found = true;
                        }
                    }
                }

                if !found {
                    return None;
                }

                sp.threshold = best_threshold;
                sp.gain = best_split_stat;
            }
            ColType::Categorical => {
                let mut groups: BTreeMap<u64, (f64, f64, usize, f64)> = BTreeMap::new();
                for &r in rows {
                    if let Some(v) = self.cols[best_col][r] {
                        let key = v.to_bits();
                        let entry = groups.entry(key).or_insert((0.0, 0.0, 0, v));
                        entry.0 += self.w[r];
                        entry.1 += self.w[r] * self.y[r];
                        entry.2 += 1;
                    }
                }

                let mut level_stats: Vec<(f64, f64, usize, f64)> = groups
                    .into_values()
                    .map(|(wk, syk, count, orig_val)| {
                        let mean_k = if wk > 0.0 { syk / wk } else { 0.0 };
                        (mean_k, wk, count, orig_val)
                    })
                    .collect();

                if level_stats.len() < 2 {
                    return None;
                }

                // Sort category levels by mean response ascending (Fisher's optimal partition)
                level_stats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let total_n = rows.len();
                let mut cum_w = 0.0;
                let mut cum_sy = 0.0;
                let mut cum_n = 0;
                let mut best_split_stat = 0.0;
                let mut best_cut = 0;
                let mut found = false;

                for k in 0..(level_stats.len() - 1) {
                    let (mean_k, wk, count, _) = level_stats[k];
                    cum_w += wk;
                    cum_sy += wk * mean_k;
                    cum_n += count;
                    let rem_n = total_n - cum_n;

                    if cum_n >= min_leaf && rem_n >= min_leaf {
                        let rem_w = w_total - cum_w;
                        let rem_sy = sum_y - cum_sy;

                        let mean_l = cum_sy / cum_w;
                        let mean_r = rem_sy / rem_w;
                        let dy = mean_l - mean_r;
                        let ssb = (cum_w * rem_w / w_total) * dy * dy;
                        let stat = ssb / vy;

                        if stat > best_split_stat || !found {
                            best_split_stat = stat;
                            best_cut = k;
                            found = true;
                        }
                    }
                }

                if !found {
                    // Fallback to single first level
                    best_cut = 0;
                }

                let left_levels: Vec<f64> = level_stats[0..=best_cut]
                    .iter()
                    .map(|&(_, _, _, val)| val)
                    .collect();

                sp.cat = Some(self.cat_parts.len());
                self.cat_parts.push(left_levels);
                sp.gain = best_split_stat;
            }
        }

        let (lv, rv) = self.partition_rows(rows, &sp);
        if lv.len() < min_leaf || rv.len() < min_leaf {
            return None;
        }

        Some((sp, lv, rv))
    }
}
