//! CLTree: Clustering through Decision Tree Construction - Bing Liu et al. (KDD 2000).
//!
//! Converts unsupervised clustering into a decision tree problem by contrasting
//! real data (Class 1) against uniformly distributed virtual background non-data (Class 0).
//!
//! Non-data points are computed analytically from bounding box volumes without
//! materializing synthetic points in memory:
//!   U(child) = U(parent) * (Volume(child) / Volume(parent))
//!
//! Clusters are dense leaves with density D / (D + U) > threshold.
//! Sparse/empty leaves isolate noise and background space.

use std::cmp::Ordering;

/// A node in the Clustering Tree.
#[derive(Clone, Debug)]
pub struct ClTreeNode {
    pub is_leaf: bool,
    pub is_cluster: bool,
    pub cluster_id: Option<usize>,
    pub depth: usize,
    pub n_data: usize,
    pub n_virtual: f64,
    pub density: f64,
    pub bounds: Vec<(f64, f64)>,
    pub split_dim: usize,
    pub split_val: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
}

/// Clustering Tree (CLTree) model.
#[derive(Clone, Debug)]
pub struct ClTree {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_density: f64,
    pub min_gain: f64,
    pub nodes: Vec<ClTreeNode>,
    pub bounds: Vec<(f64, f64)>,
    pub col_names: Vec<String>,
    pub n_clusters: usize,
}

impl Default for ClTree {
    fn default() -> Self {
        Self {
            max_depth: 6,
            min_samples_leaf: 5,
            min_density: 0.60, // >= 60% data ratio = dense cluster
            min_gain: 1e-4,
            nodes: Vec::new(),
            bounds: Vec::new(),
            col_names: Vec::new(),
            n_clusters: 0,
        }
    }
}

impl ClTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize, min_density: f64) -> Self {
        Self {
            max_depth,
            min_samples_leaf: min_samples_leaf.max(2),
            min_density: min_density.clamp(0.5, 0.99),
            ..Default::default()
        }
    }

    /// Fit the Clustering Tree on continuous data.
    pub fn fit(&mut self, data: &[Vec<f64>], col_names: Option<&[String]>) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();
        self.nodes.clear();
        self.n_clusters = 0;

        self.col_names = match col_names {
            Some(cn) => cn.to_vec(),
            None => (0..d).map(|i| format!("x_{}", i)).collect(),
        };

        // Compute bounding box
        let mut bounds = Vec::with_capacity(d);
        for j in 0..d {
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
            let span = max_val - min_val;
            // Pad bounds slightly to enclose boundary points
            let pad = 0.05 * span.max(1e-4);
            bounds.push((min_val - pad, max_val + pad));
        }

        self.bounds = bounds.clone();

        // Total virtual non-data points equals total real data points
        let total_virtual = n as f64;
        let root_density = (n as f64) / ((n as f64) + total_virtual);

        let root_idx = self.nodes.len();
        self.nodes.push(ClTreeNode {
            is_leaf: true,
            is_cluster: false,
            cluster_id: None,
            depth: 0,
            n_data: n,
            n_virtual: total_virtual,
            density: root_density,
            bounds: bounds.clone(),
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
        });

        let all_indices: Vec<usize> = (0..n).collect();
        let mut cluster_counter = 0;

        self.grow_node(
            root_idx,
            &all_indices,
            &bounds,
            0,
            data,
            &mut cluster_counter,
        );

        self.n_clusters = cluster_counter;
    }

    fn grow_node(
        &mut self,
        node_idx: usize,
        indices: &[usize],
        cur_bounds: &[(f64, f64)],
        depth: usize,
        data: &[Vec<f64>],
        cluster_counter: &mut usize,
    ) {
        let n_data = indices.len();
        let n_virtual = self.nodes[node_idx].n_virtual;
        let density = (n_data as f64) / ((n_data as f64) + n_virtual + 1e-9);
        self.nodes[node_idx].density = density;

        if depth >= self.max_depth || n_data < 2 * self.min_samples_leaf {
            self.nodes[node_idx].is_leaf = true;
            if density >= self.min_density && n_data >= self.min_samples_leaf {
                self.nodes[node_idx].is_cluster = true;
                self.nodes[node_idx].cluster_id = Some(*cluster_counter);
                *cluster_counter += 1;
            }
            return;
        }

        let d = data[0].len();
        let parent_entropy = Self::binary_entropy(n_data as f64, n_virtual);
        let mut best_gain = self.min_gain;
        let mut best_dim = 0;
        let mut best_val = 0.0;
        let mut best_left_idxs = Vec::new();
        let mut best_right_idxs = Vec::new();
        let mut best_u_l = 0.0f64;
        let mut best_u_r = 0.0f64;

        for j in 0..d {
            let (lo, hi) = cur_bounds[j];
            let width = (hi - lo).max(1e-12);

            let mut vals: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, data[i][j]))
                .filter(|(_, v)| !v.is_nan())
                .collect();

            if vals.len() < 2 * self.min_samples_leaf {
                continue;
            }

            vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            for i in (self.min_samples_leaf - 1)..(vals.len() - self.min_samples_leaf) {
                let v_curr = vals[i].1;
                let v_next = vals[i + 1].1;
                if (v_next - v_curr).abs() < 1e-9 {
                    continue;
                }

                let split_thresh = 0.5 * (v_curr + v_next);
                let frac_l = ((split_thresh - lo) / width).clamp(1e-4, 1.0 - 1e-4);
                let frac_r = 1.0 - frac_l;

                let u_l = n_virtual * frac_l;
                let u_r = n_virtual * frac_r;

                let d_l = (i + 1) as f64;
                let d_r = (vals.len() - (i + 1)) as f64;

                let total_l = d_l + u_l;
                let total_r = d_r + u_r;
                let total = total_l + total_r;

                let ent_l = Self::binary_entropy(d_l, u_l);
                let ent_r = Self::binary_entropy(d_r, u_r);
                let child_entropy = (total_l / total) * ent_l + (total_r / total) * ent_r;
                let gain = parent_entropy - child_entropy;

                if gain > best_gain {
                    best_gain = gain;
                    best_dim = j;
                    best_val = split_thresh;
                    best_u_l = u_l;
                    best_u_r = u_r;
                }
            }
        }

        if best_gain <= self.min_gain {
            self.nodes[node_idx].is_leaf = true;
            if density >= self.min_density && n_data >= self.min_samples_leaf {
                self.nodes[node_idx].is_cluster = true;
                self.nodes[node_idx].cluster_id = Some(*cluster_counter);
                *cluster_counter += 1;
            }
            return;
        }

        // Partition indices
        for &i in indices {
            let v = data[i][best_dim];
            if v.is_nan() || v <= best_val {
                best_left_idxs.push(i);
            } else {
                best_right_idxs.push(i);
            }
        }

        // Subdivide bounds
        let mut best_left_bounds = cur_bounds.to_vec();
        best_left_bounds[best_dim].1 = best_val;

        let mut best_right_bounds = cur_bounds.to_vec();
        best_right_bounds[best_dim].0 = best_val;

        let left_idx = self.nodes.len();
        self.nodes.push(ClTreeNode {
            is_leaf: true,
            is_cluster: false,
            cluster_id: None,
            depth: depth + 1,
            n_data: best_left_idxs.len(),
            n_virtual: best_u_l,
            density: (best_left_idxs.len() as f64) / (best_left_idxs.len() as f64 + best_u_l + 1e-9),
            bounds: best_left_bounds.clone(),
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
        });

        let right_idx = self.nodes.len();
        self.nodes.push(ClTreeNode {
            is_leaf: true,
            is_cluster: false,
            cluster_id: None,
            depth: depth + 1,
            n_data: best_right_idxs.len(),
            n_virtual: best_u_r,
            density: (best_right_idxs.len() as f64) / (best_right_idxs.len() as f64 + best_u_r + 1e-9),
            bounds: best_right_bounds.clone(),
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
        });

        self.nodes[node_idx].is_leaf = false;
        self.nodes[node_idx].split_dim = best_dim;
        self.nodes[node_idx].split_val = best_val;
        self.nodes[node_idx].left_child = Some(left_idx);
        self.nodes[node_idx].right_child = Some(right_idx);

        self.grow_node(
            left_idx,
            &best_left_idxs,
            &best_left_bounds,
            depth + 1,
            data,
            cluster_counter,
        );
        self.grow_node(
            right_idx,
            &best_right_idxs,
            &best_right_bounds,
            depth + 1,
            data,
            cluster_counter,
        );
    }

    /// Binary entropy H(p) = -p log2(p) - (1-p) log2(1-p).
    fn binary_entropy(data_count: f64, virtual_count: f64) -> f64 {
        let total = data_count + virtual_count;
        if total <= 1e-12 {
            return 0.0;
        }
        let p_data = (data_count / total).clamp(1e-12, 1.0 - 1e-12);
        let p_virt = 1.0 - p_data;
        -(p_data * p_data.log2() + p_virt * p_virt.log2())
    }

    /// Predict cluster assignment for a sample:
    /// Returns Some(cluster_id) if the sample lands in a dense cluster,
    /// or None if the sample lands in sparse background / noise.
    pub fn predict_cluster(&self, row: &[f64]) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }

        // Support check: if strictly outside initial bounds, it's noise
        for (j, &(lo, hi)) in self.bounds.iter().enumerate() {
            if j < row.len() && !row[j].is_nan() {
                if row[j] < lo || row[j] > hi {
                    return None;
                }
            }
        }

        let mut curr = 0;
        while !self.nodes[curr].is_leaf {
            let dim = self.nodes[curr].split_dim;
            let val = self.nodes[curr].split_val;
            let v = row.get(dim).copied().unwrap_or(0.0);

            curr = if v <= val {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
        }

        let leaf = &self.nodes[curr];
        if leaf.is_cluster {
            leaf.cluster_id
        } else {
            None
        }
    }

    /// Returns the bounding hyper-rectangles and instance sizes for all clusters.
    pub fn cluster_bounding_boxes(&self) -> Vec<(usize, Vec<(f64, f64)>, usize, f64)> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if node.is_leaf && node.is_cluster {
                if let Some(cid) = node.cluster_id {
                    out.push((cid, node.bounds.clone(), node.n_data, node.density));
                }
            }
        }
        out
    }
}
