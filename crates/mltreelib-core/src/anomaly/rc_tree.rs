//! Random Cut Tree (RCT Base Tree) - Guha et al. (ICML 2016).
//!
//! Base tree of Robust Random Cut Forest (RRCF).
//! Cuts are made with probability proportional to feature dimension span.
//! Anomaly score is computed via Collusive Displacement (CoDisp):
//! measures how much deleting a sample or small cluster shrinks the tree depth
//! of remaining points. Robust to duplicates and masking.

/// A node in the Random Cut Tree.
#[derive(Clone, Debug)]
pub struct RcNode {
    pub is_leaf: bool,
    /// Bounding box: [min_j, max_j] for each dimension.
    pub bbox: Vec<(f64, f64)>,
    /// Number of points in this subtree.
    pub size: usize,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub parent: Option<usize>,
    /// Split dimension and cut point for internal nodes.
    pub split_dim: usize,
    pub split_val: f64,
    /// Point index for leaf nodes.
    pub point_idx: Option<usize>,
}

/// Random Cut Tree estimator (single base tree).
#[derive(Clone, Debug)]
pub struct RandomCutTree {
    pub seed: u64,
    pub nodes: Vec<RcNode>,
    pub root: Option<usize>,
    pub n_points: usize,
}

impl Default for RandomCutTree {
    fn default() -> Self {
        Self {
            seed: 42,
            nodes: Vec::new(),
            root: None,
            n_points: 0,
        }
    }
}

impl RandomCutTree {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            nodes: Vec::new(),
            root: None,
            n_points: 0,
        }
    }

    /// Fit the Random Cut Tree on data.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        self.nodes.clear();
        self.n_points = n;

        let mut rng = self.seed.wrapping_add(1);
        let indices: Vec<usize> = (0..n).collect();
        self.root = Some(self.build_recursive(&indices, data, &mut rng));
    }

    fn build_recursive(&mut self, indices: &[usize], data: &[Vec<f64>], rng: &mut u64) -> usize {
        let d = data[0].len();
        let mut bbox = Vec::with_capacity(d);

        for j in 0..d {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &idx in indices {
                let v = data[idx][j];
                if !v.is_nan() {
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
            }
            if min_val.is_infinite() {
                min_val = 0.0;
                max_val = 0.0;
            }
            bbox.push((min_val, max_val));
        }

        let n = indices.len();
        if n <= 1 {
            let node_idx = self.nodes.len();
            self.nodes.push(RcNode {
                is_leaf: true,
                bbox,
                size: n,
                left_child: None,
                right_child: None,
                parent: None,
                split_dim: 0,
                split_val: 0.0,
                point_idx: indices.first().copied(),
            });
            return node_idx;
        }

        // Sum of bounding box ranges
        let mut spans = Vec::with_capacity(d);
        let mut total_span = 0.0f64;
        for (lo, hi) in &bbox {
            let span = (hi - lo).max(0.0);
            spans.push(span);
            total_span += span;
        }

        // If all points are identical, create a leaf
        if total_span < 1e-12 {
            let node_idx = self.nodes.len();
            self.nodes.push(RcNode {
                is_leaf: true,
                bbox,
                size: n,
                left_child: None,
                right_child: None,
                parent: None,
                split_dim: 0,
                split_val: 0.0,
                point_idx: indices.first().copied(),
            });
            return node_idx;
        }

        // Choose dimension with probability proportional to span
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rand_u = ((*rng >> 11) as f64) / ((1u64 << 53) as f64);
        let mut target = rand_u * total_span;
        let mut split_dim = 0;
        for (j, &s) in spans.iter().enumerate() {
            if target <= s || j == d - 1 {
                split_dim = j;
                break;
            }
            target -= s;
        }

        // Choose cut value uniformly at random along selected dimension
        let (lo, hi) = bbox[split_dim];
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let cut_u = ((*rng >> 11) as f64) / ((1u64 << 53) as f64);
        let split_val = lo + cut_u * (hi - lo);

        let mut left_idxs = Vec::new();
        let mut right_idxs = Vec::new();
        for &idx in indices {
            let v = data[idx][split_dim];
            if v.is_nan() || v <= split_val {
                left_idxs.push(idx);
            } else {
                right_idxs.push(idx);
            }
        }

        // Guard against all points falling to one side
        if left_idxs.is_empty() || right_idxs.is_empty() {
            let half = indices.len() / 2;
            left_idxs = indices[..half].to_vec();
            right_idxs = indices[half..].to_vec();
        }

        let curr_idx = self.nodes.len();
        self.nodes.push(RcNode {
            is_leaf: false,
            bbox,
            size: n,
            left_child: None,
            right_child: None,
            parent: None,
            split_dim,
            split_val,
            point_idx: None,
        });

        let left_child = self.build_recursive(&left_idxs, data, rng);
        let right_child = self.build_recursive(&right_idxs, data, rng);

        self.nodes[curr_idx].left_child = Some(left_child);
        self.nodes[curr_idx].right_child = Some(right_child);
        self.nodes[left_child].parent = Some(curr_idx);
        self.nodes[right_child].parent = Some(curr_idx);

        curr_idx
    }

    /// Compute Collusive Displacement (CoDisp) anomaly score for a point.
    /// Traverses the tree path for row, checking:
    ///   CoDisp(row, T) = max_{u ∈ path} (|S_u| - |S_v|) / |S_v|
    /// Higher CoDisp indicates an anomalous outlier or isolated tight cluster.
    pub fn codisp(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() || self.root.is_none() {
            return 0.0;
        }

        // Find leaf node for this row
        let mut curr = self.root.unwrap();
        let mut path = Vec::new();

        while !self.nodes[curr].is_leaf {
            path.push(curr);
            let dim = self.nodes[curr].split_dim;
            let val = self.nodes[curr].split_val;
            let v = row.get(dim).copied().unwrap_or(0.0);

            curr = if v <= val {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
        }
        path.push(curr);

        // Compute max displacement over ancestors
        let mut max_codisp = 0.0f64;

        for i in (0..(path.len() - 1)).rev() {
            let parent_idx = path[i];
            let child_idx = path[i + 1];

            let s_u = self.nodes[parent_idx].size as f64;
            let s_v = (self.nodes[child_idx].size as f64).max(1.0);

            let disp = (s_u - s_v) / s_v;
            if disp > max_codisp {
                max_codisp = disp;
            }
        }

        max_codisp
    }

    /// Compute normalized anomaly score in [0, 1].
    pub fn anomaly_score(&self, row: &[f64]) -> f64 {
        if self.n_points <= 1 {
            return 0.5;
        }
        let raw = self.codisp(row);
        let max_possible = (self.n_points - 1) as f64;
        (raw / max_possible.max(1.0)).clamp(0.0, 1.0)
    }
}
