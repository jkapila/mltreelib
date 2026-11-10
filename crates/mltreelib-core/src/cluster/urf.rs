//! Unsupervised Random Forest & Tree Proximity (Breiman 2001, Shi & Horvath 2006).
//!
//! Converts unsupervised clustering into a tree classification task against synthetic
//! marginal null data. Calculates leaf co-occurrence proximity matrix:
//!   P(i, j) = (1 / M) * sum_{m=1}^M I(leaf_m(i) == leaf_m(j))
//!
//! Provides:
//! - Pairwise tree proximity matrix P in [0, 1].
//! - Pairwise distance matrix D = sqrt(1 - P).
//! - Hierarchical agglomerative clustering on tree proximity distances.
//! - Tree proximity outlier scores.

use std::cmp::Ordering;

/// A node in an individual Unsupervised Proximity Tree.
#[derive(Clone, Debug)]
pub struct ProximityTreeNode {
    pub is_leaf: bool,
    pub split_dim: usize,
    pub split_val: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub leaf_id: usize,
}

/// An individual decision tree trained to separate real data from synthetic marginal noise.
#[derive(Clone, Debug)]
pub struct ProximityTree {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub nodes: Vec<ProximityTreeNode>,
    pub n_leaves: usize,
}

impl ProximityTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize) -> Self {
        Self {
            max_depth,
            min_samples_leaf: min_samples_leaf.max(1),
            nodes: Vec::new(),
            n_leaves: 0,
        }
    }

    /// Fit tree on combined real (Class 1) and synthetic (Class 0) data.
    pub fn fit(&mut self, data: &[Vec<f64>], labels: &[u8], rng: &mut u64) {
        self.nodes.clear();
        self.n_leaves = 0;
        let n = data.len();
        if n == 0 {
            return;
        }

        let mut leaf_counter = 0;
        let all_indices: Vec<usize> = (0..n).collect();

        self.grow_node(0, &all_indices, 0, data, labels, rng, &mut leaf_counter);
        self.n_leaves = leaf_counter;
    }

    fn grow_node(
        &mut self,
        _node_idx: usize,
        indices: &[usize],
        depth: usize,
        data: &[Vec<f64>],
        labels: &[u8],
        rng: &mut u64,
        leaf_counter: &mut usize,
    ) {
        let n = indices.len();
        let current_node = self.nodes.len();
        self.nodes.push(ProximityTreeNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            leaf_id: 0,
        });

        // Purity check
        let n_pos = indices.iter().filter(|&&i| labels[i] == 1).count();
        let n_neg = n - n_pos;

        if depth >= self.max_depth || n < 2 * self.min_samples_leaf || n_pos == 0 || n_neg == 0 {
            self.nodes[current_node].is_leaf = true;
            self.nodes[current_node].leaf_id = *leaf_counter;
            *leaf_counter += 1;
            return;
        }

        let d = data[0].len();
        let p_parent = (n_pos as f64) / (n as f64);
        let parent_gini = 1.0 - (p_parent * p_parent + (1.0 - p_parent) * (1.0 - p_parent));

        // Subsample features: mt = sqrt(d)
        let m_try = ((d as f64).sqrt().ceil() as usize).max(1).min(d);
        let mut best_gain = 1e-5;
        let mut best_dim = 0;
        let mut best_val = 0.0;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        // Sample m_try random feature dimensions
        for _ in 0..m_try {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let dim = (*rng as usize) % d;

            let mut vals: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, data[i][dim]))
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

                let thresh = 0.5 * (v_curr + v_next);

                let mut left_pos = 0;
                for k in 0..=i {
                    if labels[vals[k].0] == 1 {
                        left_pos += 1;
                    }
                }
                let left_tot = i + 1;
                let right_pos = n_pos - left_pos;
                let right_tot = n - left_tot;

                let p_l = (left_pos as f64) / (left_tot as f64);
                let gini_l = 1.0 - (p_l * p_l + (1.0 - p_l) * (1.0 - p_l));

                let p_r = (right_pos as f64) / (right_tot as f64);
                let gini_r = 1.0 - (p_r * p_r + (1.0 - p_r) * (1.0 - p_r));

                let weighted_child_gini =
                    (left_tot as f64 / n as f64) * gini_l + (right_tot as f64 / n as f64) * gini_r;
                let gain = parent_gini - weighted_child_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_dim = dim;
                    best_val = thresh;
                }
            }
        }

        if best_gain <= 1e-5 {
            self.nodes[current_node].is_leaf = true;
            self.nodes[current_node].leaf_id = *leaf_counter;
            *leaf_counter += 1;
            return;
        }

        // Partition
        for &i in indices {
            let v = data[i][best_dim];
            if v.is_nan() || v <= best_val {
                best_left.push(i);
            } else {
                best_right.push(i);
            }
        }

        if best_left.len() < self.min_samples_leaf || best_right.len() < self.min_samples_leaf {
            self.nodes[current_node].is_leaf = true;
            self.nodes[current_node].leaf_id = *leaf_counter;
            *leaf_counter += 1;
            return;
        }

        let left_child_idx = self.nodes.len();
        self.grow_node(
            left_child_idx,
            &best_left,
            depth + 1,
            data,
            labels,
            rng,
            leaf_counter,
        );

        let right_child_idx = self.nodes.len();
        self.grow_node(
            right_child_idx,
            &best_right,
            depth + 1,
            data,
            labels,
            rng,
            leaf_counter,
        );

        self.nodes[current_node].is_leaf = false;
        self.nodes[current_node].split_dim = best_dim;
        self.nodes[current_node].split_val = best_val;
        self.nodes[current_node].left_child = Some(left_child_idx);
        self.nodes[current_node].right_child = Some(right_child_idx);
    }

    /// Retrieve the leaf ID where a sample lands.
    pub fn apply(&self, row: &[f64]) -> usize {
        if self.nodes.is_empty() {
            return 0;
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
        self.nodes[curr].leaf_id
    }
}

/// Unsupervised Random Forest & Proximity Model.
#[derive(Clone, Debug)]
pub struct UnsupervisedProximityForest {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub seed: u64,
    pub trees: Vec<ProximityTree>,
    /// Pairwise proximity matrix for training instances: [N x N].
    pub proximity_matrix: Vec<Vec<f64>>,
    pub cluster_labels: Vec<usize>,
    pub outlier_scores: Vec<f64>,
    pub n_clusters: usize,
}

impl Default for UnsupervisedProximityForest {
    fn default() -> Self {
        Self {
            n_trees: 50,
            max_depth: 6,
            min_samples_leaf: 4,
            seed: 42,
            trees: Vec::new(),
            proximity_matrix: Vec::new(),
            cluster_labels: Vec::new(),
            outlier_scores: Vec::new(),
            n_clusters: 0,
        }
    }
}

impl UnsupervisedProximityForest {
    pub fn new(n_trees: usize, max_depth: usize, min_samples_leaf: usize, seed: u64) -> Self {
        Self {
            n_trees: n_trees.max(1),
            max_depth,
            min_samples_leaf: min_samples_leaf.max(1),
            seed,
            ..Default::default()
        }
    }

    /// Fit the Unsupervised Random Forest on real data instances.
    pub fn fit(&mut self, data: &[Vec<f64>], n_clusters: usize) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();
        let mut rng = self.seed;

        // 1. Generate Synthetic Null Data by independent marginal scrambling
        let mut synth_data = Vec::with_capacity(n);
        for _ in 0..n {
            synth_data.push(vec![0.0f64; d]);
        }

        for j in 0..d {
            let mut col_vals: Vec<f64> = data.iter().map(|r| r[j]).collect();
            // Permute column independently
            for i in (1..n).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let swap_idx = (rng as usize) % (i + 1);
                col_vals.swap(i, swap_idx);
            }
            for i in 0..n {
                synth_data[i][j] = col_vals[i];
            }
        }

        // Combine: Real (Class 1) + Synthetic (Class 0)
        let mut combined_data = Vec::with_capacity(2 * n);
        let mut combined_labels = Vec::with_capacity(2 * n);

        for row in data {
            combined_data.push(row.clone());
            combined_labels.push(1u8);
        }
        for row in synth_data {
            combined_data.push(row);
            combined_labels.push(0u8);
        }

        // 2. Grow Ensemble of Proximity Trees
        self.trees.clear();
        for _ in 0..self.n_trees {
            let mut tree = ProximityTree::new(self.max_depth, self.min_samples_leaf);
            tree.fit(&combined_data, &combined_labels, &mut rng);
            self.trees.push(tree);
        }

        // 3. Compute NxN Proximity Matrix on Real Data
        let mut prox = vec![vec![0.0f64; n]; n];

        // Precompute leaf indices for each point across all trees: [n x n_trees]
        let mut leaf_assignments = vec![vec![0usize; self.n_trees]; n];
        for (t_idx, tree) in self.trees.iter().enumerate() {
            for i in 0..n {
                leaf_assignments[i][t_idx] = tree.apply(&data[i]);
            }
        }

        for i in 0..n {
            prox[i][i] = 1.0;
            for j in (i + 1)..n {
                let mut matches = 0;
                for t in 0..self.n_trees {
                    if leaf_assignments[i][t] == leaf_assignments[j][t] {
                        matches += 1;
                    }
                }
                let p = (matches as f64) / (self.n_trees as f64);
                prox[i][j] = p;
                prox[j][i] = p;
            }
        }

        self.proximity_matrix = prox.clone();

        // 4. Calculate Outlier Scores from Proximity
        let mut outlier_scores = vec![0.0f64; n];
        for i in 0..n {
            let sum_sq: f64 = prox[i].iter().map(|&p| p * p).sum();
            outlier_scores[i] = if sum_sq > 1e-12 {
                (n as f64) / sum_sq
            } else {
                n as f64
            };
        }

        // Normalize outlier scores to [0, 1]
        let min_out = outlier_scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max_out = outlier_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_out - min_out).max(1e-6);
        for s in &mut outlier_scores {
            *s = (*s - min_out) / range;
        }
        self.outlier_scores = outlier_scores;

        // 5. Cluster via Average-Linkage Agglomeration on Proximity Distances
        let target_k = n_clusters.clamp(1, n);
        self.n_clusters = target_k;
        self.cluster_labels = Self::hierarchical_cluster(&prox, target_k);
    }

    /// Compute pairwise distance matrix D = sqrt(1 - P).
    pub fn pairwise_distances(&self) -> Vec<Vec<f64>> {
        let n = self.proximity_matrix.len();
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = (1.0 - self.proximity_matrix[i][j]).max(0.0).sqrt();
            }
        }
        dist
    }

    /// Agglomerative hierarchical clustering on distance matrix D = sqrt(1 - P) via Lance-Williams average linkage.
    fn hierarchical_cluster(prox: &[Vec<f64>], target_k: usize) -> Vec<usize> {
        let n = prox.len();
        if n == 0 {
            return Vec::new();
        }
        if target_k >= n {
            return (0..n).collect();
        }

        // Convert proximity to initial distance
        let mut cluster_dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                cluster_dist[i][j] = (1.0 - prox[i][j]).max(0.0).sqrt();
            }
        }

        // Active clusters tracking
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut sizes: Vec<usize> = vec![1; n];

        while clusters.len() > target_k {
            let k_curr = clusters.len();
            let mut best_dist = f64::INFINITY;
            let mut best_pair = (0, 1);

            for c1 in 0..k_curr {
                for c2 in (c1 + 1)..k_curr {
                    let d = cluster_dist[c1][c2];
                    if d < best_dist {
                        best_dist = d;
                        best_pair = (c1, c2);
                    }
                }
            }

            let (c1, c2) = best_pair;
            let s1 = sizes[c1] as f64;
            let s2 = sizes[c2] as f64;
            let s_new = s1 + s2;

            // Lance-Williams update for average linkage:
            // D(c1+c2, k) = (s1 * D(c1, k) + s2 * D(c2, k)) / (s1 + s2)
            for k in 0..k_curr {
                if k != c1 && k != c2 {
                    let d_new = (s1 * cluster_dist[c1][k] + s2 * cluster_dist[c2][k]) / s_new;
                    cluster_dist[c1][k] = d_new;
                    cluster_dist[k][c1] = d_new;
                }
            }
            cluster_dist[c1][c1] = 0.0;
            sizes[c1] += sizes[c2];

            // Merge membership
            let mut c2_members = std::mem::take(&mut clusters[c2]);
            clusters[c1].append(&mut c2_members);

            // Swap-remove c2 to keep contiguous index
            let last_idx = k_curr - 1;
            if c2 != last_idx {
                clusters.swap(c2, last_idx);
                sizes.swap(c2, last_idx);
                for row in 0..k_curr {
                    cluster_dist[row].swap(c2, last_idx);
                }
                cluster_dist.swap(c2, last_idx);
            }
            clusters.pop();
            sizes.pop();
            for row in &mut cluster_dist {
                row.pop();
            }
            cluster_dist.pop();
        }

        let mut labels = vec![0usize; n];
        for (cluster_id, members) in clusters.iter().enumerate() {
            for &pt in members {
                labels[pt] = cluster_id;
            }
        }

        labels
    }
}
