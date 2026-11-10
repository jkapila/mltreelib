//! Predictive Clustering Tree (PCT / CLUS) - Blockeel et al. (1998, 2002).
//!
//! Re-conceptualizes a decision tree as top-down divisive hierarchical clustering.
//! Every node represents a cluster of instances described by descriptive attributes X,
//! partitioned by maximizing multi-target variance reduction over target attributes Y.
//!
//! Supports:
//! - Multi-target continuous clustering.
//! - Unsupervised clustering (when targets Y are omitted, defaults to numeric features of X).
//! - Fast O(1) incremental running sums for split evaluations.
//! - Transparent decision rules for every extracted cluster.

use std::cmp::Ordering;
use crate::split::ColType;

/// A node / cluster in the Predictive Clustering Tree.
#[derive(Clone, Debug)]
pub struct PctNode {
    pub is_leaf: bool,
    pub depth: usize,
    pub cluster_id: usize,
    pub size: usize,
    pub total_variance: f64,
    /// Centroid / prototype of the cluster in target space.
    pub prototype: Vec<f64>,
    /// Split feature index in descriptive attributes X.
    pub split_col: usize,
    pub is_categorical: bool,
    pub threshold: f64,
    pub left_categories: Vec<f64>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    /// Descriptive rule context from root to this node.
    pub rule_desc: String,
}

/// Predictive Clustering Tree (CLUS) model.
#[derive(Clone, Debug)]
pub struct PredictiveClusteringTree {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_variance_reduction: f64,
    pub target_weights: Option<Vec<f64>>,
    pub nodes: Vec<PctNode>,
    pub dtypes: Vec<ColType>,
    pub col_names: Vec<String>,
    pub target_names: Vec<String>,
    pub n_clusters: usize,
}

impl Default for PredictiveClusteringTree {
    fn default() -> Self {
        Self {
            max_depth: 6,
            min_samples_split: 10,
            min_samples_leaf: 5,
            min_variance_reduction: 1e-4,
            target_weights: None,
            nodes: Vec::new(),
            dtypes: Vec::new(),
            col_names: Vec::new(),
            target_names: Vec::new(),
            n_clusters: 0,
        }
    }
}

impl PredictiveClusteringTree {
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            max_depth,
            min_samples_split: min_samples_split.max(2),
            min_samples_leaf: min_samples_leaf.max(1),
            ..Default::default()
        }
    }

    /// Fit the Predictive Clustering Tree on descriptive features `x` and targets `y`.
    ///
    /// If `y` is empty or omitted, defaults to unsupervised clustering using the numeric
    /// columns of `x`.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: Option<&[Vec<f64>]>,
        col_names: Option<&[String]>,
        dtypes: Option<&[ColType]>,
        target_names: Option<&[String]>,
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let d = x[0].len();
        self.nodes.clear();
        self.n_clusters = 0;

        self.dtypes = match dtypes {
            Some(dt) => dt.to_vec(),
            None => vec![ColType::Float; d],
        };

        self.col_names = match col_names {
            Some(cn) => cn.to_vec(),
            None => (0..d).map(|i| format!("x_{}", i)).collect(),
        };

        // Determine clustering targets Y: explicit Y or unsupervised numeric X
        let targets: Vec<Vec<f64>> = match y {
            Some(targets_matrix) if !targets_matrix.is_empty() => targets_matrix.to_vec(),
            _ => {
                // Unsupervised mode: extract all numeric columns from x
                let num_cols: Vec<usize> = (0..d)
                    .filter(|&j| self.dtypes.get(j) != Some(&ColType::Categorical))
                    .collect();
                if num_cols.is_empty() {
                    // Fallback to all columns if all were marked categorical
                    x.iter().map(|r| r.clone()).collect()
                } else {
                    x.iter()
                        .map(|r| num_cols.iter().map(|&j| r[j]).collect())
                        .collect()
                }
            }
        };

        let t_dim = targets[0].len();
        self.target_names = match target_names {
            Some(tn) => tn.to_vec(),
            None => (0..t_dim).map(|i| format!("target_{}", i)).collect(),
        };

        // 1. Calculate global target variances for normalization: sigma_t^2
        let mut global_var = vec![0.0f64; t_dim];
        for t in 0..t_dim {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut valid = 0.0f64;
            for r in &targets {
                let v = r[t];
                if !v.is_nan() {
                    sum += v;
                    sum_sq += v * v;
                    valid += 1.0;
                }
            }
            if valid > 1.0 {
                let mean = sum / valid;
                let var = (sum_sq / valid) - (mean * mean);
                global_var[t] = var.max(1e-8);
            } else {
                global_var[t] = 1.0;
            }
        }

        let weights: Vec<f64> = match &self.target_weights {
            Some(w) if w.len() == t_dim => w.clone(),
            _ => vec![1.0; t_dim],
        };

        // 2. Initialize root node and grow tree recursively
        let root_prototype = Self::compute_centroid(&targets, &(0..n).collect::<Vec<_>>());
        let root_var = Self::compute_weighted_variance(
            &targets,
            &(0..n).collect::<Vec<_>>(),
            &global_var,
            &weights,
        );

        let root_idx = self.nodes.len();
        self.nodes.push(PctNode {
            is_leaf: true,
            depth: 0,
            cluster_id: 0,
            size: n,
            total_variance: root_var,
            prototype: root_prototype,
            split_col: 0,
            is_categorical: false,
            threshold: 0.0,
            left_categories: Vec::new(),
            left_child: None,
            right_child: None,
            rule_desc: "root".to_string(),
        });

        let all_indices: Vec<usize> = (0..n).collect();
        let mut cluster_counter = 0;
        self.grow_recursive(
            root_idx,
            &all_indices,
            0,
            x,
            &targets,
            &global_var,
            &weights,
            &mut cluster_counter,
        );
        self.n_clusters = cluster_counter;
    }

    fn grow_recursive(
        &mut self,
        node_idx: usize,
        indices: &[usize],
        depth: usize,
        x: &[Vec<f64>],
        targets: &[Vec<f64>],
        global_var: &[f64],
        weights: &[f64],
        cluster_counter: &mut usize,
    ) {
        let n = indices.len();
        if depth >= self.max_depth || n < self.min_samples_split {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].cluster_id = *cluster_counter;
            *cluster_counter += 1;
            return;
        }

        let parent_var = self.nodes[node_idx].total_variance;
        if parent_var <= 1e-9 {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].cluster_id = *cluster_counter;
            *cluster_counter += 1;
            return;
        }

        let d = x[0].len();
        let mut best_reduction = self.min_variance_reduction;
        let mut best_split_col = 0;
        let mut best_is_cat = false;
        let mut best_thresh = 0.0;
        let mut best_left_cats = Vec::new();
        let mut best_left_idxs = Vec::new();
        let mut best_right_idxs = Vec::new();

        // Evaluate candidate splits across descriptive attributes X
        for j in 0..d {
            let is_cat = self.dtypes.get(j) == Some(&ColType::Categorical);

            if is_cat {
                // Categorical feature: group by categories
                let mut levels: Vec<f64> = indices
                    .iter()
                    .map(|&i| x[i][j])
                    .filter(|v| !v.is_nan())
                    .collect();
                levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                levels.dedup();

                if levels.len() <= 1 {
                    continue;
                }

                // Evaluate one-vs-rest category partition
                for &cat in &levels {
                    let mut left = Vec::new();
                    let mut right = Vec::new();
                    for &i in indices {
                        let v = x[i][j];
                        if v.is_nan() || (v - cat).abs() < 1e-6 {
                            left.push(i);
                        } else {
                            right.push(i);
                        }
                    }

                    if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
                        continue;
                    }

                    let var_l = Self::compute_weighted_variance(targets, &left, global_var, weights);
                    let var_r = Self::compute_weighted_variance(targets, &right, global_var, weights);
                    let weighted_child_var = (left.len() as f64 / n as f64) * var_l
                        + (right.len() as f64 / n as f64) * var_r;
                    let reduction = parent_var - weighted_child_var;

                    if reduction > best_reduction {
                        best_reduction = reduction;
                        best_split_col = j;
                        best_is_cat = true;
                        best_thresh = cat;
                        best_left_cats = vec![cat];
                        best_left_idxs = left;
                        best_right_idxs = right;
                    }
                }
            } else {
                // Numeric feature: sort and evaluate splits
                let mut vals: Vec<(usize, f64)> = indices
                    .iter()
                    .map(|&i| (i, x[i][j]))
                    .filter(|(_, v)| !v.is_nan())
                    .collect();

                if vals.len() < 2 * self.min_samples_leaf {
                    continue;
                }

                vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                // Incremental running sums for O(1) variance evaluation
                let t_dim = targets[0].len();
                let mut left_sum = vec![0.0f64; t_dim];
                let mut left_sum_sq = vec![0.0f64; t_dim];
                let mut total_sum = vec![0.0f64; t_dim];
                let mut total_sum_sq = vec![0.0f64; t_dim];

                for &(idx, _) in &vals {
                    for t in 0..t_dim {
                        let y_val = targets[idx][t];
                        total_sum[t] += y_val;
                        total_sum_sq[t] += y_val * y_val;
                    }
                }

                for i in 0..(vals.len() - 1) {
                    let (idx, v_curr) = vals[i];
                    for t in 0..t_dim {
                        let y_val = targets[idx][t];
                        left_sum[t] += y_val;
                        left_sum_sq[t] += y_val * y_val;
                    }

                    let n_l = i + 1;
                    let n_r = vals.len() - n_l;
                    if n_l < self.min_samples_leaf || n_r < self.min_samples_leaf {
                        continue;
                    }

                    let v_next = vals[i + 1].1;
                    if (v_next - v_curr).abs() < 1e-9 {
                        continue;
                    }

                    let mut var_l = 0.0f64;
                    let mut var_r = 0.0f64;
                    for t in 0..t_dim {
                        let mean_l = left_sum[t] / (n_l as f64);
                        let v_l = ((left_sum_sq[t] / (n_l as f64)) - (mean_l * mean_l)).max(0.0);

                        let right_sum_t = total_sum[t] - left_sum[t];
                        let right_sum_sq_t = total_sum_sq[t] - left_sum_sq[t];
                        let mean_r = right_sum_t / (n_r as f64);
                        let v_r = ((right_sum_sq_t / (n_r as f64)) - (mean_r * mean_r)).max(0.0);

                        var_l += weights[t] * (v_l / global_var[t]);
                        var_r += weights[t] * (v_r / global_var[t]);
                    }

                    let weighted_child_var =
                        (n_l as f64 / n as f64) * var_l + (n_r as f64 / n as f64) * var_r;
                    let reduction = parent_var - weighted_child_var;

                    if reduction > best_reduction {
                        best_reduction = reduction;
                        best_split_col = j;
                        best_is_cat = false;
                        best_thresh = 0.5 * (v_curr + v_next);
                    }
                }
            }
        }

        // Check if acceptable split was found
        if best_reduction <= self.min_variance_reduction {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].cluster_id = *cluster_counter;
            *cluster_counter += 1;
            return;
        }

        // If continuous was chosen, partition all rows
        if !best_is_cat {
            best_left_idxs.clear();
            best_right_idxs.clear();
            for &i in indices {
                let v = x[i][best_split_col];
                if v.is_nan() || v <= best_thresh {
                    best_left_idxs.push(i);
                } else {
                    best_right_idxs.push(i);
                }
            }
        }

        if best_left_idxs.len() < self.min_samples_leaf || best_right_idxs.len() < self.min_samples_leaf {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].cluster_id = *cluster_counter;
            *cluster_counter += 1;
            return;
        }

        // Create children
        let left_proto = Self::compute_centroid(targets, &best_left_idxs);
        let left_var = Self::compute_weighted_variance(targets, &best_left_idxs, global_var, weights);
        let right_proto = Self::compute_centroid(targets, &best_right_idxs);
        let right_var = Self::compute_weighted_variance(targets, &best_right_idxs, global_var, weights);

        let col_name = self.col_names.get(best_split_col).cloned().unwrap_or_else(|| format!("x_{}", best_split_col));
        let left_rule = if best_is_cat {
            format!("{} == {:.0}", col_name, best_thresh)
        } else {
            format!("{} <= {:.4}", col_name, best_thresh)
        };
        let right_rule = if best_is_cat {
            format!("{} != {:.0}", col_name, best_thresh)
        } else {
            format!("{} > {:.4}", col_name, best_thresh)
        };

        let left_idx = self.nodes.len();
        self.nodes.push(PctNode {
            is_leaf: true,
            depth: depth + 1,
            cluster_id: 0,
            size: best_left_idxs.len(),
            total_variance: left_var,
            prototype: left_proto,
            split_col: 0,
            is_categorical: false,
            threshold: 0.0,
            left_categories: Vec::new(),
            left_child: None,
            right_child: None,
            rule_desc: left_rule,
        });

        let right_idx = self.nodes.len();
        self.nodes.push(PctNode {
            is_leaf: true,
            depth: depth + 1,
            cluster_id: 0,
            size: best_right_idxs.len(),
            total_variance: right_var,
            prototype: right_proto,
            split_col: 0,
            is_categorical: false,
            threshold: 0.0,
            left_categories: Vec::new(),
            left_child: None,
            right_child: None,
            rule_desc: right_rule,
        });

        self.nodes[node_idx].is_leaf = false;
        self.nodes[node_idx].split_col = best_split_col;
        self.nodes[node_idx].is_categorical = best_is_cat;
        self.nodes[node_idx].threshold = best_thresh;
        self.nodes[node_idx].left_categories = best_left_cats;
        self.nodes[node_idx].left_child = Some(left_idx);
        self.nodes[node_idx].right_child = Some(right_idx);

        self.grow_recursive(
            left_idx,
            &best_left_idxs,
            depth + 1,
            x,
            targets,
            global_var,
            weights,
            cluster_counter,
        );
        self.grow_recursive(
            right_idx,
            &best_right_idxs,
            depth + 1,
            x,
            targets,
            global_var,
            weights,
            cluster_counter,
        );
    }

    /// Compute cluster centroid across target dimensions.
    fn compute_centroid(targets: &[Vec<f64>], indices: &[usize]) -> Vec<f64> {
        if indices.is_empty() {
            return Vec::new();
        }
        let t_dim = targets[0].len();
        let mut sums = vec![0.0f64; t_dim];
        let mut counts = vec![0.0f64; t_dim];

        for &i in indices {
            for t in 0..t_dim {
                let v = targets[i][t];
                if !v.is_nan() {
                    sums[t] += v;
                    counts[t] += 1.0;
                }
            }
        }

        sums.iter()
            .zip(counts.iter())
            .map(|(&s, &c)| if c > 0.0 { s / c } else { 0.0 })
            .collect()
    }

    /// Compute normalized multi-target variance for an instance subset.
    fn compute_weighted_variance(
        targets: &[Vec<f64>],
        indices: &[usize],
        global_var: &[f64],
        weights: &[f64],
    ) -> f64 {
        let n = indices.len();
        if n <= 1 {
            return 0.0;
        }
        let t_dim = targets[0].len();
        let centroid = Self::compute_centroid(targets, indices);
        let mut total_var = 0.0f64;

        for t in 0..t_dim {
            let mean = centroid[t];
            let mut sum_sq_diff = 0.0f64;
            let mut valid = 0.0f64;
            for &i in indices {
                let v = targets[i][t];
                if !v.is_nan() {
                    let diff = v - mean;
                    sum_sq_diff += diff * diff;
                    valid += 1.0;
                }
            }
            if valid > 1.0 {
                let var_t = sum_sq_diff / valid;
                total_var += weights[t] * (var_t / global_var[t]);
            }
        }

        total_var
    }

    /// Route a descriptive feature vector to its assigned leaf cluster ID.
    pub fn predict_cluster(&self, row: &[f64]) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut curr = 0;
        while !self.nodes[curr].is_leaf {
            let col = self.nodes[curr].split_col;
            let val = row.get(col).copied().unwrap_or(f64::NAN);
            let go_left = if self.nodes[curr].is_categorical {
                if val.is_nan() {
                    true
                } else {
                    self.nodes[curr].left_categories.iter().any(|cat| (val - cat).abs() < 1e-6)
                }
            } else {
                if val.is_nan() {
                    true
                } else {
                    val <= self.nodes[curr].threshold
                }
            };

            curr = if go_left {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
        }

        self.nodes[curr].cluster_id
    }

    /// Predict the target prototype vector (centroid) for a descriptive sample.
    pub fn predict_prototype(&self, row: &[f64]) -> Vec<f64> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut curr = 0;
        while !self.nodes[curr].is_leaf {
            let col = self.nodes[curr].split_col;
            let val = row.get(col).copied().unwrap_or(f64::NAN);
            let go_left = if self.nodes[curr].is_categorical {
                if val.is_nan() {
                    true
                } else {
                    self.nodes[curr].left_categories.iter().any(|cat| (val - cat).abs() < 1e-6)
                }
            } else {
                if val.is_nan() {
                    true
                } else {
                    val <= self.nodes[curr].threshold
                }
            };

            curr = if go_left {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
        }

        self.nodes[curr].prototype.clone()
    }

    /// Extract human-interpretable rules for all leaf clusters.
    pub fn cluster_rules(&self) -> Vec<(usize, String, Vec<f64>, usize)> {
        let mut rules = Vec::new();
        self.collect_rules(0, Vec::new(), &mut rules);
        rules
    }

    fn collect_rules(
        &self,
        node_idx: usize,
        current_conditions: Vec<String>,
        out: &mut Vec<(usize, String, Vec<f64>, usize)>,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let node = &self.nodes[node_idx];
        if node.is_leaf {
            let rule = if current_conditions.is_empty() {
                "ALL".to_string()
            } else {
                current_conditions.join(" AND ")
            };
            out.push((node.cluster_id, rule, node.prototype.clone(), node.size));
            return;
        }

        let col_name = self.col_names.get(node.split_col).cloned().unwrap_or_else(|| format!("x_{}", node.split_col));
        let (left_cond, right_cond) = if node.is_categorical {
            (
                format!("{} == {:.0}", col_name, node.threshold),
                format!("{} != {:.0}", col_name, node.threshold),
            )
        } else {
            (
                format!("{} <= {:.4}", col_name, node.threshold),
                format!("{} > {:.4}", col_name, node.threshold),
            )
        };

        if let Some(left_child) = node.left_child {
            let mut left_conds = current_conditions.clone();
            left_conds.push(left_cond);
            self.collect_rules(left_child, left_conds, out);
        }

        if let Some(right_child) = node.right_child {
            let mut right_conds = current_conditions;
            right_conds.push(right_cond);
            self.collect_rules(right_child, right_conds, out);
        }
    }
}
