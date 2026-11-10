//! HDBSCAN Condensed Tree (Campello, Moulavi, Sander 2013).
//!
//! Hierarchical Density-Based Spatial Clustering of Applications with Noise.
//! Core tree representations:
//! 1. Mutual Reachability Graph & Minimum Spanning Tree (MST).
//! 2. Full single-linkage cluster dendrogram.
//! 3. Condensed Tree: Tracks cluster birth, point shedding (fallout lambda),
//!    and true cluster splits (where both children >= min_cluster_size).
//! 4. Cluster Extraction via Excess of Mass (Stability).
//! 5. GLOSH outlier scoring: 1 - (lambda_p / lambda_max).

use std::cmp::Ordering;

/// An edge in the Minimum Spanning Tree or Dendrogram.
#[derive(Clone, Copy, Debug)]
pub struct MstEdge {
    pub u: usize,
    pub v: usize,
    pub weight: f64,
}

/// A node in the HDBSCAN Condensed Tree.
#[derive(Clone, Debug)]
pub struct CondensedNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub lambda_birth: f64,
    pub lambda_death: f64,
    pub size: usize,
    pub children: Vec<usize>,
    /// (point_index, lambda_fallout) for points that shed from this cluster.
    pub points: Vec<(usize, f64)>,
    /// Excess of mass (stability): sum_{p} (lambda_p - lambda_birth).
    pub stability: f64,
    pub selected: bool,
}

/// HDBSCAN Condensed Tree Clustering Model.
#[derive(Clone, Debug)]
pub struct HdbscanTree {
    pub min_cluster_size: usize,
    pub min_samples: usize,
    pub condensed_nodes: Vec<CondensedNode>,
    /// Cluster label for each training point: Some(cluster_id) or None (-1 for noise).
    pub labels: Vec<Option<usize>>,
    /// GLOSH outlier score in [0, 1] for each training point.
    pub outlier_scores: Vec<f64>,
    pub n_clusters: usize,
    pub train_data: Vec<Vec<f64>>,
    pub core_distances: Vec<f64>,
}

impl Default for HdbscanTree {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: 5,
            condensed_nodes: Vec::new(),
            labels: Vec::new(),
            outlier_scores: Vec::new(),
            n_clusters: 0,
            train_data: Vec::new(),
            core_distances: Vec::new(),
        }
    }
}

impl HdbscanTree {
    pub fn new(min_cluster_size: usize, min_samples: usize) -> Self {
        Self {
            min_cluster_size: min_cluster_size.max(2),
            min_samples: min_samples.max(1),
            ..Default::default()
        }
    }

    /// Fit HDBSCAN on numeric feature vectors.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        self.condensed_nodes.clear();
        self.labels = vec![None; n];
        self.outlier_scores = vec![1.0; n];
        self.n_clusters = 0;
        self.train_data = data.to_vec();

        if n < self.min_cluster_size {
            return;
        }

        // 1. Compute core distances: distance to k-th nearest neighbor (k = min_samples)
        let k = self.min_samples.min(n - 1);
        let mut core_dists = vec![0.0f64; n];
        for i in 0..n {
            let mut dists = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i != j {
                    dists.push(Self::euclidean_dist(&data[i], &data[j]));
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            core_dists[i] = dists[k - 1].max(1e-12);
        }
        self.core_distances = core_dists.clone();

        // 2. Build Minimum Spanning Tree (MST) on Mutual Reachability Distance via Prim's algorithm
        let mut in_tree = vec![false; n];
        let mut min_edge = vec![f64::INFINITY; n];
        let mut parent = vec![0usize; n];
        min_edge[0] = 0.0;

        let mut mst_edges = Vec::with_capacity(n - 1);

        for _ in 0..n {
            let mut u = None;
            let mut best_val = f64::INFINITY;
            for i in 0..n {
                if !in_tree[i] && min_edge[i] < best_val {
                    best_val = min_edge[i];
                    u = Some(i);
                }
            }

            let u = match u {
                Some(idx) => idx,
                None => break,
            };

            in_tree[u] = true;
            if u != 0 {
                mst_edges.push(MstEdge {
                    u: parent[u],
                    v: u,
                    weight: best_val,
                });
            }

            for v in 0..n {
                if !in_tree[v] {
                    let d_raw = Self::euclidean_dist(&data[u], &data[v]);
                    let mreach = core_dists[u].max(core_dists[v]).max(d_raw);
                    if mreach < min_edge[v] {
                        min_edge[v] = mreach;
                        parent[v] = u;
                    }
                }
            }
        }

        // Sort MST edges by weight ascending
        mst_edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));

        // 3. Build Full Single-Linkage Dendrogram using Union-Find
        let mut uf = UnionFind::new(n);
        let mut dendro_nodes = Vec::with_capacity(2 * n);
        for i in 0..n {
            dendro_nodes.push(DendroNode {
                size: 1,
                members: vec![i],
                left: None,
                right: None,
                weight: 0.0,
            });
        }

        for edge in mst_edges {
            let root_u = uf.find(edge.u);
            let root_v = uf.find(edge.v);
            if root_u != root_v {
                let new_node_idx = dendro_nodes.len();
                let mut members = dendro_nodes[root_u].members.clone();
                members.extend_from_slice(&dendro_nodes[root_v].members);
                let size = members.len();

                dendro_nodes.push(DendroNode {
                    size,
                    members,
                    left: Some(root_u),
                    right: Some(root_v),
                    weight: edge.weight,
                });

                uf.union_to(root_u, root_v, new_node_idx);
            }
        }

        let dendro_root = dendro_nodes.len() - 1;

        // 4. Build Condensed Tree from Dendrogram
        self.condensed_nodes.push(CondensedNode {
            node_id: 0,
            parent_id: None,
            lambda_birth: 0.0,
            lambda_death: f64::INFINITY,
            size: n,
            children: Vec::new(),
            points: Vec::new(),
            stability: 0.0,
            selected: false,
        });

        self.condense_recursive(dendro_root, 0, 0.0, &dendro_nodes);

        // 5. Compute Cluster Stabilities (Excess of Mass)
        for node in &mut self.condensed_nodes {
            let birth = node.lambda_birth;
            let mut stab = 0.0f64;
            for &(_, lambda_p) in &node.points {
                stab += (lambda_p - birth).max(0.0);
            }
            node.stability = stab;
        }

        // 6. Select Optimal Clusters Bottom-Up
        self.select_clusters();

        // 7. Assign Labels and Outlier Scores
        self.assign_labels_and_outliers(n);
    }

    fn condense_recursive(
        &mut self,
        dendro_idx: usize,
        current_cluster: usize,
        _current_lambda: f64,
        dendro: &[DendroNode],
    ) {
        let node = &dendro[dendro_idx];
        if node.left.is_none() || node.right.is_none() {
            return;
        }

        let left_idx = node.left.unwrap();
        let right_idx = node.right.unwrap();
        let left_size = dendro[left_idx].size;
        let right_size = dendro[right_idx].size;
        let lambda = if node.weight > 1e-12 {
            1.0 / node.weight
        } else {
            1e12
        };

        if left_size >= self.min_cluster_size && right_size >= self.min_cluster_size {
            // True cluster split: spawn 2 new condensed clusters
            self.condensed_nodes[current_cluster].lambda_death = lambda;

            let left_c_id = self.condensed_nodes.len();
            self.condensed_nodes.push(CondensedNode {
                node_id: left_c_id,
                parent_id: Some(current_cluster),
                lambda_birth: lambda,
                lambda_death: f64::INFINITY,
                size: left_size,
                children: Vec::new(),
                points: Vec::new(),
                stability: 0.0,
                selected: false,
            });

            let right_c_id = self.condensed_nodes.len();
            self.condensed_nodes.push(CondensedNode {
                node_id: right_c_id,
                parent_id: Some(current_cluster),
                lambda_birth: lambda,
                lambda_death: f64::INFINITY,
                size: right_size,
                children: Vec::new(),
                points: Vec::new(),
                stability: 0.0,
                selected: false,
            });

            self.condensed_nodes[current_cluster].children.push(left_c_id);
            self.condensed_nodes[current_cluster].children.push(right_c_id);

            self.condense_recursive(left_idx, left_c_id, lambda, dendro);
            self.condense_recursive(right_idx, right_c_id, lambda, dendro);
        } else if left_size >= self.min_cluster_size {
            // Right child dissolves: shed right points as noise
            for &pt in &dendro[right_idx].members {
                self.condensed_nodes[current_cluster].points.push((pt, lambda));
            }
            self.condense_recursive(left_idx, current_cluster, lambda, dendro);
        } else if right_size >= self.min_cluster_size {
            // Left child dissolves: shed left points as noise
            for &pt in &dendro[left_idx].members {
                self.condensed_nodes[current_cluster].points.push((pt, lambda));
            }
            self.condense_recursive(right_idx, current_cluster, lambda, dendro);
        } else {
            // Both children dissolve
            for &pt in &dendro[left_idx].members {
                self.condensed_nodes[current_cluster].points.push((pt, lambda));
            }
            for &pt in &dendro[right_idx].members {
                self.condensed_nodes[current_cluster].points.push((pt, lambda));
            }
            self.condensed_nodes[current_cluster].lambda_death = lambda;
        }
    }

    fn select_clusters(&mut self) {
        let n_nodes = self.condensed_nodes.len();
        let mut subtree_stability = vec![0.0f64; n_nodes];

        // Bottom-up traversal
        for i in (0..n_nodes).rev() {
            let child_stab_sum: f64 = self.condensed_nodes[i]
                .children
                .iter()
                .map(|&c| subtree_stability[c])
                .sum();

            if self.condensed_nodes[i].stability >= child_stab_sum && !self.condensed_nodes[i].children.is_empty() {
                subtree_stability[i] = self.condensed_nodes[i].stability;
                self.condensed_nodes[i].selected = true;
                // Unselect descendants
                Self::unselect_subtree(&mut self.condensed_nodes, i);
            } else if self.condensed_nodes[i].children.is_empty() {
                subtree_stability[i] = self.condensed_nodes[i].stability;
                self.condensed_nodes[i].selected = true;
            } else {
                subtree_stability[i] = child_stab_sum;
                self.condensed_nodes[i].selected = false;
            }
        }

        // If root is selected, but has selected children, keep children
        let selected_count = self.condensed_nodes.iter().filter(|n| n.selected).count();
        if selected_count == 1 && self.condensed_nodes[0].selected && !self.condensed_nodes[0].children.is_empty() {
            self.condensed_nodes[0].selected = false;
            let root_children = self.condensed_nodes[0].children.clone();
            for child in root_children {
                self.condensed_nodes[child].selected = true;
            }
        }
    }

    fn unselect_subtree(nodes: &mut [CondensedNode], root: usize) {
        let children = nodes[root].children.clone();
        for child in children {
            nodes[child].selected = false;
            Self::unselect_subtree(nodes, child);
        }
    }

    fn assign_labels_and_outliers(&mut self, n: usize) {
        let mut cluster_id_map = std::collections::HashMap::new();
        let mut label_counter = 0;

        for (idx, node) in self.condensed_nodes.iter().enumerate() {
            if node.selected {
                cluster_id_map.insert(idx, label_counter);
                label_counter += 1;
            }
        }

        self.n_clusters = label_counter;

        // Collect point lambdas per cluster
        let mut point_lambdas = vec![0.0f64; n];
        let mut point_clusters = vec![None; n];

        for (node_idx, node) in self.condensed_nodes.iter().enumerate() {
            let target_cluster = if node.selected {
                cluster_id_map.get(&node_idx).copied()
            } else {
                // Find nearest selected ancestor
                Self::find_selected_ancestor(&self.condensed_nodes, node_idx)
                    .and_then(|anc| cluster_id_map.get(&anc).copied())
            };

            for &(pt, lam) in &node.points {
                if target_cluster.is_some() && point_clusters[pt].is_none() {
                    point_clusters[pt] = target_cluster;
                    point_lambdas[pt] = lam;
                }
            }
        }

        self.labels = point_clusters;

        // GLOSH outlier scores: 1 - (lambda_p / lambda_max)
        let mut max_lambda_per_cluster = vec![1e-6f64; self.n_clusters];
        for i in 0..n {
            if let Some(c) = self.labels[i] {
                if point_lambdas[i] > max_lambda_per_cluster[c] {
                    max_lambda_per_cluster[c] = point_lambdas[i];
                }
            }
        }

        for i in 0..n {
            if let Some(c) = self.labels[i] {
                let max_lam = max_lambda_per_cluster[c];
                let lam_p = point_lambdas[i];
                self.outlier_scores[i] = (1.0 - (lam_p / max_lam)).clamp(0.0, 1.0);
            } else {
                self.outlier_scores[i] = 1.0; // Noise points have max outlier score
            }
        }
    }

    fn find_selected_ancestor(nodes: &[CondensedNode], start: usize) -> Option<usize> {
        let mut curr = nodes[start].parent_id;
        while let Some(idx) = curr {
            if nodes[idx].selected {
                return Some(idx);
            }
            curr = nodes[idx].parent_id;
        }
        None
    }

    /// Predict cluster for an out-of-sample query point.
    pub fn predict(&self, query: &[f64]) -> Option<usize> {
        if self.train_data.is_empty() || self.n_clusters == 0 {
            return None;
        }

        let mut best_dist = f64::INFINITY;
        let mut best_pt = 0;

        for (i, train_pt) in self.train_data.iter().enumerate() {
            let d = Self::euclidean_dist(query, train_pt);
            if d < best_dist {
                best_dist = d;
                best_pt = i;
            }
        }

        self.labels[best_pt]
    }

    fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

struct DendroNode {
    size: usize,
    members: Vec<usize>,
    left: Option<usize>,
    right: Option<usize>,
    weight: f64,
}

struct UnionFind {
    parent: Vec<usize>,
    node_map: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            node_map: (0..n).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        let mut curr = i;
        while curr != root {
            let nxt = self.parent[curr];
            self.parent[curr] = root;
            curr = nxt;
        }
        self.node_map[root]
    }

    fn union_to(&mut self, _u_node: usize, _v_node: usize, new_node: usize) {
        let root_u = self.find_raw(self.parent_of_node(_u_node));
        let root_v = self.find_raw(self.parent_of_node(_v_node));
        if root_u != root_v {
            self.parent[root_v] = root_u;
            self.node_map[root_u] = new_node;
        }
    }

    fn find_raw(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        root
    }

    fn parent_of_node(&self, node_idx: usize) -> usize {
        self.node_map.iter().position(|&n| n == node_idx).unwrap_or(0)
    }
}
