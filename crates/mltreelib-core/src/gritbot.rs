//! GritBot: Quinlan's contextual anomaly and outlier engine.
//!
//! Reimplementation of RuleQuest GritBot (Quinlan 2010):
//! Finds surprising values conditioned on subgroups defined by rules
//! (attribute conjunctions). A case is anomalous if its value is implausible
//! within its subgroup, even if normal in the global distribution.

use std::cmp::Ordering;
use crate::split::ColType;

/// Operator defining a condition in a subgroup rule.
#[derive(Clone, Debug, PartialEq)]
pub enum ConditionOp {
    /// Numeric <= threshold
    Le(f64),
    /// Numeric > threshold
    Gt(f64),
    /// Numeric low < x <= high
    Between(f64, f64),
    /// Discrete / Categorical equality
    Eq(f64),
}

/// A single predicate on an attribute in a contextual subgroup rule.
#[derive(Clone, Debug, PartialEq)]
pub struct GritCondition {
    pub col: usize,
    pub col_name: String,
    pub op: ConditionOp,
    pub actual_val: f64,
}

impl GritCondition {
    /// Test whether a feature vector matches this condition.
    pub fn matches(&self, row: &[f64]) -> bool {
        if self.col >= row.len() {
            return false;
        }
        let v = row[self.col];
        if v.is_nan() {
            return false;
        }
        match self.op {
            ConditionOp::Le(thresh) => v <= thresh,
            ConditionOp::Gt(thresh) => v > thresh,
            ConditionOp::Between(lo, hi) => v > lo && v <= hi,
            ConditionOp::Eq(target) => (v - target).abs() < 1e-6,
        }
    }

    /// Human-readable condition representation.
    pub fn display_str(&self) -> String {
        match self.op {
            ConditionOp::Le(thresh) => format!("{} <= {:.4}", self.col_name, thresh),
            ConditionOp::Gt(thresh) => format!("{} > {:.4}", self.col_name, thresh),
            ConditionOp::Between(lo, hi) => format!("{} > {:.4} and <= {:.4}", self.col_name, lo, hi),
            ConditionOp::Eq(target) => format!("{} == {:.0}", self.col_name, target),
        }
    }
}

/// A detected contextual anomaly record.
#[derive(Clone, Debug)]
pub struct GritAnomaly {
    /// 0-indexed case / row index in the dataset.
    pub case_id: usize,
    /// Column index of the anomalous attribute.
    pub attribute: usize,
    /// Column name of the anomalous attribute.
    pub attribute_name: String,
    /// Observed value for this case.
    pub actual_value: f64,
    /// Expected value (mean) within the subgroup context.
    pub expected_value: f64,
    /// Standard deviation within the subgroup context.
    pub std_dev: f64,
    /// Standardized deviation z = |actual - expected| / std_dev.
    pub z_score: f64,
    /// Estimated chance significance p-value (smaller = more surprising).
    pub significance: f64,
    /// Size of the subgroup context (N cases).
    pub subgroup_size: usize,
    /// Boundary limit of normal cases (e.g. 99% <= limit).
    pub normal_limit: f64,
    /// Fraction of subgroup cases obeying the normal limit.
    pub normal_fraction: f64,
    /// Conjunction of conditions defining the context. Empty = global outlier.
    pub conditions: Vec<GritCondition>,
    /// Global standardized deviation in the unconditioned dataset.
    pub global_z: f64,
    /// Contextual elevation ratio: z / (global_z + 0.1).
    /// High ratio (>= 1.5) identifies pure contextual anomalies invisible to global detectors.
    pub context_ratio: f64,
    /// Density surprise score: -ln(p_value) inspired by Density Estimation Trees (DET).
    pub density_score: f64,
    /// Indicator whether this is a pure contextual anomaly (amplified by subgroup rule).
    pub is_pure_contextual: bool,
    /// Formatted Quinlan-style diagnostic line.
    pub summary: String,
}

/// GritBot contextual anomaly engine configuration.
#[derive(Clone, Debug)]
pub struct GritBot {
    /// Minimum cases in a subgroup (default: max(35, 0.005 * N)).
    pub min_items: Option<usize>,
    /// Minimum Z-score to consider a candidate anomaly (default: 3.0).
    pub max_norm: f64,
    /// Significance p-value cutoff (default: 0.05).
    pub sig_threshold: f64,
    /// Maximum rule depth / conditions per subgroup (default: 3).
    pub max_depth: usize,
    /// Optional subset of columns to check for anomalies (None = all non-constant).
    pub check_cols: Option<Vec<usize>>,
    /// Minimum standard deviation to guard against divide-by-zero (default: 1e-5).
    pub eps: f64,
}

impl Default for GritBot {
    fn default() -> Self {
        Self {
            min_items: None,
            max_norm: 3.0,
            sig_threshold: 0.05,
            max_depth: 3,
            check_cols: None,
            eps: 1e-5,
        }
    }
}

/// Internal subgroup representation during partition search.
struct Subgroup {
    indices: Vec<usize>,
    conditions: Vec<GritCondition>,
}

impl GritBot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sig_threshold(mut self, alpha: f64) -> Self {
        self.sig_threshold = alpha;
        self
    }

    pub fn with_max_norm(mut self, z: f64) -> Self {
        self.max_norm = z;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_min_items(mut self, min_items: usize) -> Self {
        self.min_items = Some(min_items);
        self
    }

    pub fn with_check_cols(mut self, cols: Vec<usize>) -> Self {
        self.check_cols = Some(cols);
        self
    }

    /// Scan a dataset for contextual anomalies across attributes.
    pub fn scan(
        &self,
        data: &[Vec<f64>],
        col_names: &[String],
        dtypes: &[ColType],
    ) -> Vec<GritAnomaly> {
        let n_rows = data.len();
        if n_rows < 10 {
            return Vec::new();
        }
        let n_cols = if n_rows > 0 { data[0].len() } else { 0 };
        let min_items = self.min_items.unwrap_or_else(|| {
            let dyn_min = (0.005 * n_rows as f64).round() as usize;
            std::cmp::max(10, std::cmp::max(dyn_min, 35.min(n_rows / 4)))
        });

        let mut all_anomalies = Vec::new();

        let target_cols: Vec<usize> = match &self.check_cols {
            Some(cols) => cols.clone(),
            None => (0..n_cols).collect(),
        };

        for &target_col in &target_cols {
            if target_col >= n_cols {
                continue;
            }
            let att_name = col_names.get(target_col).cloned().unwrap_or_else(|| format!("col_{}", target_col));

            // Gather valid (non-NA) row indices for this target column
            let valid_indices: Vec<usize> = (0..n_rows)
                .filter(|&i| !data[i][target_col].is_nan())
                .collect();

            if valid_indices.len() < min_items {
                continue;
            }

            let g_sum: f64 = valid_indices.iter().map(|&i| data[i][target_col]).sum();
            let g_sum_sq: f64 = valid_indices.iter().map(|&i| data[i][target_col] * data[i][target_col]).sum();
            let global_mean = g_sum / valid_indices.len() as f64;
            let global_sd = sample_sd(valid_indices.len(), g_sum, g_sum_sq).max(self.eps);

            // 1. Check global outliers (no conditions)
            self.check_subgroup(
                &valid_indices,
                &[],
                target_col,
                &att_name,
                global_mean,
                global_sd,
                data,
                n_rows,
                &mut all_anomalies,
            );

            // 2. Discover conditional subgroups by recursive partitioning
            let subgroups = self.discover_subgroups(
                &valid_indices,
                target_col,
                data,
                col_names,
                dtypes,
                min_items,
            );

            // 3. Inspect each subgroup for contextual outliers
            for sg in subgroups {
                self.check_subgroup(
                    &sg.indices,
                    &sg.conditions,
                    target_col,
                    &att_name,
                    global_mean,
                    global_sd,
                    data,
                    n_rows,
                    &mut all_anomalies,
                );
            }
        }

        // Sort anomalies by significance ascending (most surprising first)
        all_anomalies.sort_by(|a, b| {
            a.significance
                .partial_cmp(&b.significance)
                .unwrap_or(Ordering::Equal)
        });

        all_anomalies
    }

    /// Check a subgroup of rows for tail anomalies on target_col.
    fn check_subgroup(
        &self,
        indices: &[usize],
        conditions: &[GritCondition],
        target_col: usize,
        att_name: &str,
        global_mean: f64,
        global_sd: f64,
        data: &[Vec<f64>],
        total_rows: usize,
        results: &mut Vec<GritAnomaly>,
    ) {
        let n = indices.len();
        if n < 5 {
            return;
        }

        let mut vals: Vec<(usize, f64)> = indices
            .iter()
            .map(|&idx| (idx, data[idx][target_col]))
            .collect();

        // Sort by value
        vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Trimmed / robust mean and std dev (trim 5% on each tail to resist existing outliers)
        let trim_k = (0.05 * n as f64).floor() as usize;
        let core_slice = if n > 10 && trim_k > 0 {
            &vals[trim_k..(n - trim_k)]
        } else {
            &vals[..]
        };

        let core_n = core_slice.len() as f64;
        let mean = core_slice.iter().map(|(_, v)| *v).sum::<f64>() / core_n;
        let var = core_slice.iter().map(|(_, v)| (v - mean) * (v - mean)).sum::<f64>() / (core_n - 1.0).max(1.0);
        let sd = var.sqrt().max(self.eps);

        // Max allowable outlier tail count (at most 5% of subgroup or 10 cases)
        let max_anoms = std::cmp::max(1, (0.05 * n as f64).ceil() as usize).min(10);

        // Check high tail
        for offset in 0..max_anoms {
            let (idx, val) = vals[n - 1 - offset];
            let z = (val - mean) / sd;
            if z >= self.max_norm {
                let p_val = normal_survival(z);
                // Bonferroni adjustment factor for candidate evaluations
                let adj_p = (p_val * (total_rows as f64 / n as f64)).min(1.0);
                if adj_p <= self.sig_threshold {
                    // Normal limit: the preceding normal item or percentile
                    let normal_lim_idx = n - 1 - offset - 1;
                    let (norm_lim, norm_frac) = if normal_lim_idx < n {
                        (vals[normal_lim_idx].1, (normal_lim_idx + 1) as f64 / n as f64)
                    } else {
                        (mean + 2.0 * sd, 0.95)
                    };

                    let mut conds_with_actual = Vec::new();
                    for c in conditions {
                        let actual_val = data[idx][c.col];
                        let mut c_copy = c.clone();
                        c_copy.actual_val = actual_val;
                        conds_with_actual.push(c_copy);
                    }

                    let summary = format_diagnostic(
                        idx,
                        att_name,
                        val,
                        n,
                        mean,
                        norm_frac * 100.0,
                        "<=",
                        norm_lim,
                        adj_p,
                        &conds_with_actual,
                    );

                    let global_z = ((val - global_mean).abs() / global_sd).max(0.0);
                    let context_ratio = z / (global_z + 0.1);
                    let density_score = -adj_p.max(1e-15).ln();
                    let is_pure_contextual = !conditions.is_empty() && context_ratio > 1.05;

                    results.push(GritAnomaly {
                        case_id: idx,
                        attribute: target_col,
                        attribute_name: att_name.to_string(),
                        actual_value: val,
                        expected_value: mean,
                        std_dev: sd,
                        z_score: z,
                        significance: adj_p,
                        subgroup_size: n,
                        normal_limit: norm_lim,
                        normal_fraction: norm_frac,
                        conditions: conds_with_actual,
                        global_z,
                        context_ratio,
                        density_score,
                        is_pure_contextual,
                        summary,
                    });
                }
            }
        }

        // Check low tail
        for offset in 0..max_anoms {
            let (idx, val) = vals[offset];
            let z = (mean - val) / sd;
            if z >= self.max_norm {
                let p_val = normal_survival(z);
                let adj_p = (p_val * (total_rows as f64 / n as f64)).min(1.0);
                if adj_p <= self.sig_threshold {
                    let normal_lim_idx = offset + 1;
                    let (norm_lim, norm_frac) = if normal_lim_idx < n {
                        (vals[normal_lim_idx].1, (n - normal_lim_idx) as f64 / n as f64)
                    } else {
                        (mean - 2.0 * sd, 0.95)
                    };

                    let mut conds_with_actual = Vec::new();
                    for c in conditions {
                        let actual_val = data[idx][c.col];
                        let mut c_copy = c.clone();
                        c_copy.actual_val = actual_val;
                        conds_with_actual.push(c_copy);
                    }

                    let summary = format_diagnostic(
                        idx,
                        att_name,
                        val,
                        n,
                        mean,
                        norm_frac * 100.0,
                        ">=",
                        norm_lim,
                        adj_p,
                        &conds_with_actual,
                    );

                    let global_z = ((val - global_mean).abs() / global_sd).max(0.0);
                    let context_ratio = z / (global_z + 0.1);
                    let density_score = -adj_p.max(1e-15).ln();
                    let is_pure_contextual = !conditions.is_empty() && context_ratio > 1.05;

                    results.push(GritAnomaly {
                        case_id: idx,
                        attribute: target_col,
                        attribute_name: att_name.to_string(),
                        actual_value: val,
                        expected_value: mean,
                        std_dev: sd,
                        z_score: z,
                        significance: adj_p,
                        subgroup_size: n,
                        normal_limit: norm_lim,
                        normal_fraction: norm_frac,
                        conditions: conds_with_actual,
                        global_z,
                        context_ratio,
                        density_score,
                        is_pure_contextual,
                        summary,
                    });
                }
            }
        }
    }

    /// Recursively discover homogeneous contextual subgroups using variance reduction splits.
    fn discover_subgroups(
        &self,
        indices: &[usize],
        target_col: usize,
        data: &[Vec<f64>],
        col_names: &[String],
        dtypes: &[ColType],
        min_items: usize,
    ) -> Vec<Subgroup> {
        let mut leaf_subgroups = Vec::new();
        let mut queue: Vec<(Vec<usize>, Vec<GritCondition>, usize)> = vec![(
            indices.to_vec(),
            Vec::new(),
            0,
        )];

        while let Some((sub_indices, conds, depth)) = queue.pop() {
            if depth >= self.max_depth || sub_indices.len() < 2 * min_items {
                if !conds.is_empty() && sub_indices.len() >= min_items {
                    leaf_subgroups.push(Subgroup {
                        indices: sub_indices,
                        conditions: conds,
                    });
                }
                continue;
            }

            // Find best split on any non-target feature that maximizes standard deviation reduction (SDR)
            if let Some((best_col, best_thresh, left_indices, right_indices)) =
                self.find_best_sdr_split(&sub_indices, target_col, data, dtypes, min_items)
            {
                let col_name = col_names.get(best_col).cloned().unwrap_or_else(|| format!("col_{}", best_col));

                // Left branch: <= threshold
                let mut left_conds = conds.clone();
                left_conds.push(GritCondition {
                    col: best_col,
                    col_name: col_name.clone(),
                    op: ConditionOp::Le(best_thresh),
                    actual_val: 0.0,
                });
                queue.push((left_indices, left_conds, depth + 1));

                // Right branch: > threshold
                let mut right_conds = conds;
                right_conds.push(GritCondition {
                    col: best_col,
                    col_name,
                    op: ConditionOp::Gt(best_thresh),
                    actual_val: 0.0,
                });
                queue.push((right_indices, right_conds, depth + 1));
            } else if !conds.is_empty() && sub_indices.len() >= min_items {
                leaf_subgroups.push(Subgroup {
                    indices: sub_indices,
                    conditions: conds,
                });
            }
        }

        leaf_subgroups
    }

    /// Find the split on feature X_j that maximizes standard deviation reduction on target Y.
    fn find_best_sdr_split(
        &self,
        indices: &[usize],
        target_col: usize,
        data: &[Vec<f64>],
        _dtypes: &[ColType],
        min_items: usize,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let n = indices.len();
        let n_cols = data[0].len();

        let parent_sum: f64 = indices.iter().map(|&i| data[i][target_col]).sum();
        let parent_sum_sq: f64 = indices.iter().map(|&i| data[i][target_col] * data[i][target_col]).sum();
        let parent_sd = sample_sd(n, parent_sum, parent_sum_sq);

        if parent_sd < self.eps {
            return None;
        }

        let mut best_gain = 0.0f64;
        let mut best_split = None;

        for col in 0..n_cols {
            if col == target_col {
                continue;
            }

            // Collect non-NA values of feature col
            let mut pairs: Vec<(usize, f64, f64)> = indices
                .iter()
                .filter_map(|&idx| {
                    let feat_val = data[idx][col];
                    if feat_val.is_nan() {
                        None
                    } else {
                        Some((idx, feat_val, data[idx][target_col]))
                    }
                })
                .collect();

            if pairs.len() < 2 * min_items {
                continue;
            }

            // Sort by feature value
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // Cumulative sweep
            let mut left_sum = 0.0f64;
            let mut left_sum_sq = 0.0f64;
            let mut right_sum = parent_sum;
            let mut right_sum_sq = parent_sum_sq;

            for i in 0..(pairs.len() - 1) {
                let (_, x_val, y_val) = pairs[i];
                left_sum += y_val;
                left_sum_sq += y_val * y_val;
                right_sum -= y_val;
                right_sum_sq -= y_val * y_val;

                let n_left = i + 1;
                let n_right = pairs.len() - n_left;

                if n_left < min_items || n_right < min_items {
                    continue;
                }

                // Check for value boundary
                let next_x_val = pairs[i + 1].1;
                if (next_x_val - x_val).abs() < 1e-9 {
                    continue;
                }

                let sd_left = sample_sd(n_left, left_sum, left_sum_sq);
                let sd_right = sample_sd(n_right, right_sum, right_sum_sq);

                let expected_sd = (n_left as f64 * sd_left + n_right as f64 * sd_right) / (pairs.len() as f64);
                let gain = parent_sd - expected_sd;

                if gain > best_gain {
                    best_gain = gain;
                    let thresh = 0.5 * (x_val + next_x_val);
                    best_split = Some((col, thresh));
                }
            }
        }

        // If a good split was found (gain > 5% relative SD reduction), partition indices
        if best_gain > 0.05 * parent_sd {
            if let Some((b_col, b_thresh)) = best_split {
                let mut left_idxs = Vec::new();
                let mut right_idxs = Vec::new();
                for &idx in indices {
                    let v = data[idx][b_col];
                    if v.is_nan() || v <= b_thresh {
                        left_idxs.push(idx);
                    } else {
                        right_idxs.push(idx);
                    }
                }
                return Some((b_col, b_thresh, left_idxs, right_idxs));
            }
        }

        None
    }
}

/// Compute sample standard deviation from N, Sum, SumSq.
#[inline]
fn sample_sd(n: usize, sum: f64, sum_sq: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    let var = (sum_sq - (sum * sum) / n_f) / (n_f - 1.0);
    if var > 0.0 {
        var.sqrt()
    } else {
        0.0
    }
}

/// Approximate Gaussian standard normal upper survival function Q(z) = P(Z >= z).
/// Accurate complementary error function approximation (Abramowitz & Stegun).
fn normal_survival(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0 - normal_survival(-z);
    }
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    (phi * poly).clamp(1e-15, 1.0)
}

/// Quinlan-style diagnostic formatter.
fn format_diagnostic(
    case_id: usize,
    att_name: &str,
    val: f64,
    n_cases: usize,
    mean: f64,
    pct: f64,
    op: &str,
    limit: f64,
    sig: f64,
    conditions: &[GritCondition],
) -> String {
    let mut out = format!(
        "case {}: [{:.4}] {} = {:.2} ({} cases, mean {:.2}, {:.1}% {} {:.2})",
        case_id + 1,
        sig,
        att_name,
        val,
        n_cases,
        mean,
        pct,
        op,
        limit
    );
    if !conditions.is_empty() {
        out.push_str(" where ");
        let cond_strs: Vec<String> = conditions
            .iter()
            .map(|c| format!("{} [{:.2}]", c.display_str(), c.actual_val))
            .collect();
        out.push_str(&cond_strs.join(" and "));
    }
    out
}
