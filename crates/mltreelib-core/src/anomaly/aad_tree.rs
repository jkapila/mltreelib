//! Active Anomaly Discovery Tree (AAD Base Tree) - Das et al. (KDD 2016, JAIR 2020).
//!
//! Exposes tree paths as node indicator embeddings z(x) with adjustable node weights w.
//! Anomaly score is computed as linear combination: Score(x) = w^T * z(x).
//! Supports active human feedback (true anomaly / false positive) to dynamically update
//! node weights without restructuring the tree.

/// A node in the Active Anomaly Discovery Tree.
#[derive(Clone, Debug)]
pub struct AadNode {
    pub is_leaf: bool,
    pub split_dim: usize,
    pub split_val: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub depth: usize,
    pub size: usize,
}

/// Active Anomaly Discovery single base tree.
#[derive(Clone, Debug)]
pub struct ActiveAnomalyTree {
    pub max_depth: usize,
    pub seed: u64,
    pub nodes: Vec<AadNode>,
    /// Active node weights w_v.
    pub weights: Vec<f64>,
    pub n_samples: usize,
}

impl Default for ActiveAnomalyTree {
    fn default() -> Self {
        Self {
            max_depth: 8,
            seed: 42,
            nodes: Vec::new(),
            weights: Vec::new(),
            n_samples: 0,
        }
    }
}

impl ActiveAnomalyTree {
    pub fn new(max_depth: usize, seed: u64) -> Self {
        Self {
            max_depth,
            seed,
            nodes: Vec::new(),
            weights: Vec::new(),
            n_samples: 0,
        }
    }

    /// Fit the tree structure on data.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        self.nodes.clear();
        self.weights.clear();
        self.n_samples = n;

        let mut rng = self.seed.wrapping_add(1);
        let indices: Vec<usize> = (0..n).collect();

        let root_idx = self.nodes.len();
        self.nodes.push(AadNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: 0,
            size: n,
        });

        self.grow_recursive(root_idx, &indices, 0, data, &mut rng);

        // Initialize uniform node weights w_v = 1.0 / sqrt(V)
        let v = self.nodes.len().max(1);
        let init_w = 1.0 / (v as f64).sqrt();
        self.weights = vec![init_w; v];
    }

    fn grow_recursive(
        &mut self,
        node_idx: usize,
        indices: &[usize],
        depth: usize,
        data: &[Vec<f64>],
        rng: &mut u64,
    ) {
        let n = indices.len();
        if depth >= self.max_depth || n <= 1 {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            return;
        }

        let d = data[0].len();
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let dim = (*rng as usize) % d;

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &idx in indices {
            let v = data[idx][dim];
            if !v.is_nan() {
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }
        }

        if min_val.is_infinite() || (max_val - min_val).abs() < 1e-9 {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            return;
        }

        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let cut_u = ((*rng >> 11) as f64) / ((1u64 << 53) as f64);
        let split_val = min_val + cut_u * (max_val - min_val);

        let mut left_idxs = Vec::new();
        let mut right_idxs = Vec::new();
        for &idx in indices {
            let v = data[idx][dim];
            if v.is_nan() || v <= split_val {
                left_idxs.push(idx);
            } else {
                right_idxs.push(idx);
            }
        }

        if left_idxs.is_empty() || right_idxs.is_empty() {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            return;
        }

        let left_child = self.nodes.len();
        self.nodes.push(AadNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: depth + 1,
            size: left_idxs.len(),
        });

        let right_child = self.nodes.len();
        self.nodes.push(AadNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: depth + 1,
            size: right_idxs.len(),
        });

        self.nodes[node_idx].is_leaf = false;
        self.nodes[node_idx].split_dim = dim;
        self.nodes[node_idx].split_val = split_val;
        self.nodes[node_idx].left_child = Some(left_child);
        self.nodes[node_idx].right_child = Some(right_child);

        self.grow_recursive(left_child, &left_idxs, depth + 1, data, rng);
        self.grow_recursive(right_child, &right_idxs, depth + 1, data, rng);
    }

    /// Extract the sparse node path vector for a sample: list of (node_idx, activation).
    pub fn node_representation(&self, row: &[f64]) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut path = Vec::new();
        let mut curr = 0;

        let act = 1.0 / ((self.nodes[curr].depth + 1) as f64).sqrt();
        path.push((curr, act));

        while !self.nodes[curr].is_leaf {
            let dim = self.nodes[curr].split_dim;
            let val = self.nodes[curr].split_val;
            let v = row.get(dim).copied().unwrap_or(0.0);

            curr = if v <= val {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
            let act = 1.0 / ((self.nodes[curr].depth + 1) as f64).sqrt();
            path.push((curr, act));
        }

        path
    }

    /// Compute weighted anomaly score: Score(x) = w^T * z(x).
    pub fn score(&self, row: &[f64]) -> f64 {
        let rep = self.node_representation(row);
        let mut score = 0.0f64;
        for (node_idx, act) in rep {
            if node_idx < self.weights.len() {
                score += self.weights[node_idx] * act;
            }
        }
        score
    }

    /// Incorporate user feedback for a reviewed instance:
    ///   is_anomaly = true  => reinforce separating nodes (+eta * z(x))
    ///   is_anomaly = false => penalize false positive nodes (-eta * z(x))
    pub fn update_feedback(&mut self, row: &[f64], is_anomaly: bool, eta: f64) {
        let rep = self.node_representation(row);
        for (node_idx, act) in rep {
            if node_idx < self.weights.len() {
                if is_anomaly {
                    self.weights[node_idx] += eta * act;
                } else {
                    self.weights[node_idx] = (self.weights[node_idx] - eta * act).max(0.0);
                }
            }
        }

        // L2 normalize weights
        let norm_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        if norm_sq > 1e-12 {
            let norm = norm_sq.sqrt();
            for w in &mut self.weights {
                *w /= norm;
            }
        }
    }
}
