//! Density Estimation Tree (DET) - Ram & Gray (KDD 2011, JMLR 2018).
//!
//! Non-parametric density estimation via recursive partitioning minimizing
//! Integrated Squared Error (ISE):
//!
//!   R(f) = ∫ f²(x) dx - 2 ∫ f(x) dP(x) = - Σ_leaves (N_l² / (N² * V_l))
//!
//! The split criterion directly maximizes ISE reduction:
//!   ΔR = (N_L² / V_L) + (N_R² / V_R) - (N_P² / V_P)
//!
//! Handles both continuous attributes and discrete categorical features.
//! Anomaly score is defined as the negative log-density: -ln(f̂(x)).

use std::cmp::Ordering;
use crate::split::ColType;

/// Split specification in a DET node.
#[derive(Clone, Debug)]
pub enum DetSplit {
    /// Continuous threshold split: x[col] <= threshold goes Left, > threshold goes Right.
    Continuous {
        col: usize,
        threshold: f64,
        lower: f64,
        upper: f64,
    },
    /// Categorical partition: if x[col] in left_categories goes Left, else Right.
    Categorical {
        col: usize,
        left_categories: Vec<f64>,
        total_categories: usize,
    },
}

/// A node in the Density Estimation Tree.
#[derive(Clone, Debug)]
pub struct DetNode {
    pub is_leaf: bool,
    pub split: Option<DetSplit>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    /// Number of training samples that reached this node.
    pub n_samples: usize,
    /// Relative volume of this hyper-rectangle relative to the root (v ∈ (0, 1]).
    pub rel_volume: f64,
    /// Estimated probability density f̂(x) inside this leaf.
    pub density: f64,
}

/// Density Estimation Tree estimator.
#[derive(Clone, Debug)]
pub struct DensityEstimationTree {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_leaf_samples: usize,
    pub min_density_gain: f64,
    pub alpha: f64,
    pub max_leaves: Option<usize>,
    pub nodes: Vec<DetNode>,
    /// Total training samples N.
    pub n_train: usize,
    /// Total volume of root bounding box.
    pub root_volume: f64,
    /// Bounding box of continuous features [min_j, max_j].
    pub bounds: Vec<(f64, f64)>,
    /// Cardinalities of categorical features.
    pub cat_cardinalities: Vec<usize>,
    pub dtypes: Vec<ColType>,
    pub col_names: Vec<String>,
}

impl Default for DensityEstimationTree {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 10,
            min_leaf_samples: 5,
            min_density_gain: 0.0,
            alpha: 0.0,
            max_leaves: None,
            nodes: Vec::new(),
            n_train: 0,
            root_volume: 1.0,
            bounds: Vec::new(),
            cat_cardinalities: Vec::new(),
            dtypes: Vec::new(),
            col_names: Vec::new(),
        }
    }
}

impl DensityEstimationTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_min_samples_split(mut self, min_split: usize) -> Self {
        self.min_samples_split = min_split;
        self
    }

    pub fn with_min_leaf_samples(mut self, min_leaf: usize) -> Self {
        self.min_leaf_samples = min_leaf;
        self
    }

    /// Fit the Density Estimation Tree on training data.
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        col_names: &[String],
        dtypes: &[ColType],
    ) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();
        self.n_train = n;
        self.dtypes = dtypes.to_vec();
        self.col_names = col_names.to_vec();
        self.nodes.clear();

        // 1. Compute global bounds for continuous columns and cardinalities for categoricals
        let mut bounds = Vec::with_capacity(d);
        let mut cat_cardinalities = Vec::with_capacity(d);
        let mut root_vol = 1.0f64;

        for j in 0..d {
            let is_cat = dtypes.get(j) == Some(&ColType::Categorical);
            if is_cat {
                let mut levels: Vec<f64> = data.iter().map(|r| r[j]).filter(|v| !v.is_nan()).collect();
                levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                levels.dedup();
                let card = levels.len().max(1);
                cat_cardinalities.push(card);
                bounds.push((0.0, card as f64));
            } else {
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for row in data {
                    let v = row[j];
                    if !v.is_nan() {
                        if v < min_val { min_val = v; }
                        if v > max_val { max_val = v; }
                    }
                }
                if min_val.is_infinite() || (max_val - min_val).abs() < 1e-9 {
                    min_val = 0.0;
                    max_val = 1.0;
                }
                // Expand bounds slightly to prevent zero volume at boundaries
                let span = max_val - min_val;
                let pad = 0.001 * span.max(1e-4);
                let lower = min_val - pad;
                let upper = max_val + pad;
                root_vol *= upper - lower;
                bounds.push((lower, upper));
                cat_cardinalities.push(0);
            }
        }

        self.bounds = bounds.clone();
        self.cat_cardinalities = cat_cardinalities;
        self.root_volume = if root_vol > 0.0 && root_vol.is_finite() { root_vol } else { 1.0 };

        // 2. Recursive greedy growth maximizing ISE gain
        let all_indices: Vec<usize> = (0..n).collect();
        let current_bounds = bounds;

        let root_idx = self.nodes.len();
        self.nodes.push(DetNode {
            is_leaf: true,
            split: None,
            left_child: None,
            right_child: None,
            n_samples: n,
            rel_volume: 1.0,
            density: 1.0 / (self.root_volume),
        });

        self.grow_node(root_idx, &all_indices, &current_bounds, 1.0, 0, data);
    }

    fn grow_node(
        &mut self,
        node_idx: usize,
        indices: &[usize],
        cur_bounds: &[(f64, f64)],
        rel_vol: f64,
        depth: usize,
        data: &[Vec<f64>],
    ) {
        let n_node = indices.len();

        if depth >= self.max_depth || n_node < self.min_samples_split {
            // Leaf node: compute density f̂ = N_l / (N * V_l)
            let actual_vol = (rel_vol * self.root_volume).max(1e-15);
            let dens = (n_node as f64) / ((self.n_train as f64) * actual_vol);
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].density = dens;
            return;
        }

        // Find split maximizing ISE gain ΔR = (N_L² / V_L) + (N_R² / V_R) - (N_P² / V_P)
        let parent_term = ((n_node * n_node) as f64) / rel_vol.max(1e-15);
        let mut best_gain = self.min_density_gain + self.alpha;
        let mut best_split = None;
        let mut best_left_idxs = Vec::new();
        let mut best_right_idxs = Vec::new();
        let mut best_left_rel_vol = rel_vol;
        let mut best_right_rel_vol = rel_vol;
        let mut best_left_bounds = cur_bounds.to_vec();
        let mut best_right_bounds = cur_bounds.to_vec();

        let d = data[0].len();

        for j in 0..d {
            let is_cat = self.dtypes.get(j) == Some(&ColType::Categorical);

            if is_cat {
                // Categorical scan: find subset partition
                let mut levels: Vec<f64> = indices
                    .iter()
                    .map(|&idx| data[idx][j])
                    .filter(|v| !v.is_nan())
                    .collect();
                levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                levels.dedup();

                if levels.len() <= 1 {
                    continue;
                }

                let total_cat = self.cat_cardinalities[j].max(levels.len());
                let cur_cat_count = cur_bounds[j].1 - cur_bounds[j].0;

                // Evaluate one-vs-rest binary partitions on categorical levels
                for &lvl in &levels {
                    let mut left = Vec::new();
                    let mut right = Vec::new();

                    for &idx in indices {
                        let v = data[idx][j];
                        if (v - lvl).abs() < 1e-6 {
                            left.push(idx);
                        } else {
                            right.push(idx);
                        }
                    }

                    if left.len() < self.min_leaf_samples || right.len() < self.min_leaf_samples {
                        continue;
                    }

                    // Volume split proportional to category counts
                    let frac_left = (1.0 / cur_cat_count).clamp(0.01, 0.99);
                    let frac_right = 1.0 - frac_left;

                    let v_l = rel_vol * frac_left;
                    let v_r = rel_vol * frac_right;

                    let gain = (((left.len() * left.len()) as f64) / v_l)
                        + (((right.len() * right.len()) as f64) / v_r)
                        - parent_term;

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some(DetSplit::Categorical {
                            col: j,
                            left_categories: vec![lvl],
                            total_categories: total_cat,
                        });
                        best_left_idxs = left;
                        best_right_idxs = right;
                        best_left_rel_vol = v_l;
                        best_right_rel_vol = v_r;

                        let mut b_left = cur_bounds.to_vec();
                        let mut b_right = cur_bounds.to_vec();
                        b_left[j] = (0.0, 1.0);
                        b_right[j] = (0.0, (cur_cat_count - 1.0).max(1.0));
                        best_left_bounds = b_left;
                        best_right_bounds = b_right;
                    }
                }
            } else {
                // Continuous scan: scan thresholds
                let mut vals: Vec<(usize, f64)> = indices
                    .iter()
                    .map(|&idx| (idx, data[idx][j]))
                    .filter(|(_, v)| !v.is_nan())
                    .collect();

                if vals.len() < 2 * self.min_leaf_samples {
                    continue;
                }

                vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                let (col_lo, col_hi) = cur_bounds[j];
                let col_width = (col_hi - col_lo).max(1e-12);

                for i in (self.min_leaf_samples - 1)..(vals.len() - self.min_leaf_samples) {
                    let v_curr = vals[i].1;
                    let v_next = vals[i + 1].1;

                    if (v_next - v_curr).abs() < 1e-9 {
                        continue;
                    }

                    let thresh = 0.5 * (v_curr + v_next);
                    let frac_left = ((thresh - col_lo) / col_width).clamp(1e-4, 1.0 - 1e-4);
                    let frac_right = 1.0 - frac_left;

                    let v_l = rel_vol * frac_left;
                    let v_r = rel_vol * frac_right;

                    let n_l = i + 1;
                    let n_r = vals.len() - n_l;

                    let gain = (((n_l * n_l) as f64) / v_l)
                        + (((n_r * n_r) as f64) / v_r)
                        - parent_term;

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some(DetSplit::Continuous {
                            col: j,
                            threshold: thresh,
                            lower: col_lo,
                            upper: col_hi,
                        });
                        best_left_idxs = vals[..=i].iter().map(|&(idx, _)| idx).collect();
                        best_right_idxs = vals[(i + 1)..].iter().map(|&(idx, _)| idx).collect();
                        best_left_rel_vol = v_l;
                        best_right_rel_vol = v_r;

                        let mut b_left = cur_bounds.to_vec();
                        let mut b_right = cur_bounds.to_vec();
                        b_left[j] = (col_lo, thresh);
                        b_right[j] = (thresh, col_hi);
                        best_left_bounds = b_left;
                        best_right_bounds = b_right;
                    }
                }
            }
        }

        if best_gain > 0.0 && best_split.is_some() {
            let left_node_idx = self.nodes.len();
            self.nodes.push(DetNode {
                is_leaf: true,
                split: None,
                left_child: None,
                right_child: None,
                n_samples: best_left_idxs.len(),
                rel_volume: best_left_rel_vol,
                density: 0.0,
            });

            let right_node_idx = self.nodes.len();
            self.nodes.push(DetNode {
                is_leaf: true,
                split: None,
                left_child: None,
                right_child: None,
                n_samples: best_right_idxs.len(),
                rel_volume: best_right_rel_vol,
                density: 0.0,
            });

            self.nodes[node_idx].is_leaf = false;
            self.nodes[node_idx].split = best_split;
            self.nodes[node_idx].left_child = Some(left_node_idx);
            self.nodes[node_idx].right_child = Some(right_node_idx);

            self.grow_node(
                left_node_idx,
                &best_left_idxs,
                &best_left_bounds,
                best_left_rel_vol,
                depth + 1,
                data,
            );
            self.grow_node(
                right_node_idx,
                &best_right_idxs,
                &best_right_bounds,
                best_right_rel_vol,
                depth + 1,
                data,
            );
        } else {
            // No split improved ISE: terminate as leaf
            let actual_vol = (rel_vol * self.root_volume).max(1e-15);
            let dens = (n_node as f64) / ((self.n_train as f64) * actual_vol);
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].density = dens;
        }
    }

    /// Predict estimated probability density f̂(x) for a sample.
    pub fn predict_density(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        // Support check: outside bounding support, density is 0.0
        for (j, &(lo, hi)) in self.bounds.iter().enumerate() {
            if j < row.len() && !row[j].is_nan() {
                if self.dtypes.get(j) != Some(&ColType::Categorical) {
                    if row[j] < lo || row[j] > hi {
                        return 0.0;
                    }
                }
            }
        }

        let mut curr = 0;
        while !self.nodes[curr].is_leaf {
            let split = self.nodes[curr].split.as_ref().unwrap();
            let go_left = match split {
                DetSplit::Continuous { col, threshold, .. } => {
                    let v = row[*col];
                    if v.is_nan() {
                        true
                    } else {
                        v <= *threshold
                    }
                }
                DetSplit::Categorical { col, left_categories, .. } => {
                    let v = row[*col];
                    if v.is_nan() {
                        true
                    } else {
                        left_categories.iter().any(|cat| (v - cat).abs() < 1e-6)
                    }
                }
            };

            curr = if go_left {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
        }

        self.nodes[curr].density
    }

    /// Predict negative log-density anomaly score: -ln(f̂(x)).
    /// Higher score indicates rarer / lower density region (higher anomaly likelihood).
    pub fn predict_anomaly_score(&self, row: &[f64]) -> f64 {
        let density = self.predict_density(row);
        let safe_density = density.max(1e-15);
        -safe_density.ln()
    }
}
