//! Half-Space Tree (HS-Tree Base Tree) - Tan, Ting, & Liu (2011).
//!
//! Fast, lightweight streaming anomaly detector. Builds a fixed, balanced
//! space-partitioning tree over domain bounding boxes.
//! Anomaly score is computed from the node mass profile: anomalies land in
//! sparse sub-trees that capture very few points.

/// A node in the Half-Space Tree.
#[derive(Clone, Debug)]
pub struct HsNode {
    pub is_leaf: bool,
    pub split_dim: usize,
    pub split_val: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub depth: usize,
    /// Reference mass: number of training samples passing through this node.
    pub mass: usize,
}

/// Half-Space Tree single base tree.
#[derive(Clone, Debug)]
pub struct HalfSpaceTree {
    pub max_depth: usize,
    pub seed: u64,
    pub nodes: Vec<HsNode>,
    pub bounds: Vec<(f64, f64)>,
    pub n_samples: usize,
}

impl Default for HalfSpaceTree {
    fn default() -> Self {
        Self {
            max_depth: 8,
            seed: 42,
            nodes: Vec::new(),
            bounds: Vec::new(),
            n_samples: 0,
        }
    }
}

impl HalfSpaceTree {
    pub fn new(max_depth: usize, seed: u64) -> Self {
        Self {
            max_depth,
            seed,
            nodes: Vec::new(),
            bounds: Vec::new(),
            n_samples: 0,
        }
    }

    /// Build the fixed balanced half-space tree topology over bounding box bounds.
    pub fn build_topology(&mut self, bounds: &[(f64, f64)]) {
        self.bounds = bounds.to_vec();
        self.nodes.clear();
        self.n_samples = 0;

        let mut rng_state = self.seed.wrapping_add(1);
        let root_idx = self.nodes.len();
        self.nodes.push(HsNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: 0,
            mass: 0,
        });

        let mut cur_bounds = bounds.to_vec();
        self.build_recursive(root_idx, 0, &mut cur_bounds, &mut rng_state);
    }

    fn build_recursive(
        &mut self,
        node_idx: usize,
        depth: usize,
        cur_bounds: &mut [(f64, f64)],
        rng: &mut u64,
    ) {
        if depth >= self.max_depth {
            self.nodes[node_idx].is_leaf = true;
            return;
        }

        let d = cur_bounds.len();
        if d == 0 {
            return;
        }

        // Pick split dimension pseudo-randomly
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let q = (*rng as usize) % d;

        let (lo, hi) = cur_bounds[q];
        let mid = 0.5 * (lo + hi);

        let left_idx = self.nodes.len();
        self.nodes.push(HsNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: depth + 1,
            mass: 0,
        });

        let right_idx = self.nodes.len();
        self.nodes.push(HsNode {
            is_leaf: true,
            split_dim: 0,
            split_val: 0.0,
            left_child: None,
            right_child: None,
            depth: depth + 1,
            mass: 0,
        });

        self.nodes[node_idx].is_leaf = false;
        self.nodes[node_idx].split_dim = q;
        self.nodes[node_idx].split_val = mid;
        self.nodes[node_idx].left_child = Some(left_idx);
        self.nodes[node_idx].right_child = Some(right_idx);

        // Recurse left
        let orig_hi = cur_bounds[q].1;
        cur_bounds[q].1 = mid;
        self.build_recursive(left_idx, depth + 1, cur_bounds, rng);

        // Recurse right
        cur_bounds[q].1 = orig_hi;
        let orig_lo = cur_bounds[q].0;
        cur_bounds[q].0 = mid;
        self.build_recursive(right_idx, depth + 1, cur_bounds, rng);
        cur_bounds[q].0 = orig_lo;
    }

    /// Insert a sample vector to update node masses.
    pub fn insert(&mut self, row: &[f64]) {
        if self.nodes.is_empty() {
            return;
        }
        self.n_samples += 1;
        let mut curr = 0;
        self.nodes[curr].mass += 1;

        while !self.nodes[curr].is_leaf {
            let dim = self.nodes[curr].split_dim;
            let val = self.nodes[curr].split_val;
            let v = row.get(dim).copied().unwrap_or(0.0);

            curr = if v <= val {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
            self.nodes[curr].mass += 1;
        }
    }

    /// Fit on a batch of samples: builds topology and updates masses.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();

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
            // Pad bounds by 20% to allow streaming points to fall within range
            let pad = 0.2 * span.max(1e-4);
            bounds.push((min_val - pad, max_val + pad));
        }

        self.build_topology(&bounds);
        for row in data {
            self.insert(row);
        }
    }

    /// Compute raw mass score: S(x) = Σ_{u ∈ path} r_u * 2^(depth(u)).
    /// Dense normal points produce high scores; sparse anomalies produce low scores.
    pub fn mass_score(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut curr = 0;
        let mut score = 0.0f64;

        score += (self.nodes[curr].mass as f64) * 2.0f64.powi(self.nodes[curr].depth as i32);

        while !self.nodes[curr].is_leaf {
            let dim = self.nodes[curr].split_dim;
            let val = self.nodes[curr].split_val;
            let v = row.get(dim).copied().unwrap_or(0.0);

            curr = if v <= val {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
            score += (self.nodes[curr].mass as f64) * 2.0f64.powi(self.nodes[curr].depth as i32);
        }

        score
    }

    /// Normalized anomaly score in [0, 1]. Higher score = more anomalous.
    pub fn anomaly_score(&self, row: &[f64]) -> f64 {
        if self.n_samples == 0 || self.nodes.is_empty() {
            return 0.5;
        }
        let raw = self.mass_score(row);
        let max_possible = (self.n_samples as f64) * 2.0f64.powi((self.max_depth + 1) as i32);
        (1.0 - (raw / max_possible)).clamp(0.0, 1.0)
    }
}
