//! Extended Isolation Tree (EIF Base Tree) with First-Class Categorical Support.
//!
//! Based on Hariri, Kind, & Brunner (IEEE TKDE 2019):
//! Extends standard isolation trees by cutting along random oblique hyperplanes
//! rather than axis-aligned cuts, completely eliminating coordinate banding artifacts.
//!
//! Categorical extension:
//! Supports both random subset partitioning and random level embedding projections,
//! allowing categorical dimensions to participate natively in oblique hyperplanes.

use std::collections::HashMap;
use crate::split::ColType;

/// A split in an Extended Isolation Tree.
#[derive(Clone, Debug)]
pub enum IsolationSplit {
    /// Oblique hyperplane cut:
    ///   Σ_{j ∈ Cont} (n_j * x_j) + Σ_{c ∈ Cat} (cat_weights[c][x_c]) <= threshold
    Oblique {
        cont_cols: Vec<usize>,
        cont_weights: Vec<f64>,
        cat_cols: Vec<usize>,
        cat_weights: Vec<HashMap<u64, f64>>,
        threshold: f64,
        missing_left: bool,
    },
    /// Pure categorical subset cut: if x[col] ∈ left_levels, go Left; else go Right.
    CategoricalSubset {
        col: usize,
        left_levels: Vec<u64>,
        missing_left: bool,
    },
}

impl IsolationSplit {
    /// Test whether a feature vector satisfies the split (goes left).
    pub fn goes_left(&self, row: &[f64]) -> bool {
        match self {
            IsolationSplit::Oblique {
                cont_cols,
                cont_weights,
                cat_cols,
                cat_weights,
                threshold,
                missing_left,
            } => {
                let mut proj = 0.0f64;
                for (&col, &w) in cont_cols.iter().zip(cont_weights.iter()) {
                    let v = row[col];
                    if v.is_nan() {
                        return *missing_left;
                    }
                    proj += w * v;
                }
                for (&col, weight_map) in cat_cols.iter().zip(cat_weights.iter()) {
                    let v = row[col];
                    if v.is_nan() {
                        return *missing_left;
                    }
                    let key = v.round() as u64;
                    if let Some(&w) = weight_map.get(&key) {
                        proj += w;
                    }
                }
                proj <= *threshold
            }
            IsolationSplit::CategoricalSubset {
                col,
                left_levels,
                missing_left,
            } => {
                let v = row[*col];
                if v.is_nan() {
                    return *missing_left;
                }
                let key = v.round() as u64;
                left_levels.contains(&key)
            }
        }
    }
}

/// A node in the Extended Isolation Tree.
#[derive(Clone, Debug)]
pub struct IsolationNode {
    pub is_leaf: bool,
    pub split: Option<IsolationSplit>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    /// Number of training samples that terminated at this node.
    pub size: usize,
    /// Tree depth of this node.
    pub depth: usize,
}

/// Extended Isolation Tree estimator (single base tree).
#[derive(Clone, Debug)]
pub struct ExtendedIsolationTree {
    /// Maximum tree height (default: ceil(log2(max(n, 2)))).
    pub max_depth: Option<usize>,
    /// Extension level: 0 = axis-aligned, None = fully oblique (all features).
    pub extension_level: Option<usize>,
    /// RNG seed for deterministic splits.
    pub seed: u64,
    pub nodes: Vec<IsolationNode>,
    /// Total training instances N.
    pub n_train: usize,
    pub dtypes: Vec<ColType>,
    pub col_names: Vec<String>,
}

impl Default for ExtendedIsolationTree {
    fn default() -> Self {
        Self {
            max_depth: None,
            extension_level: None,
            seed: 42,
            nodes: Vec::new(),
            n_train: 0,
            dtypes: Vec::new(),
            col_names: Vec::new(),
        }
    }
}

impl ExtendedIsolationTree {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn with_extension_level(mut self, level: usize) -> Self {
        self.extension_level = Some(level);
        self
    }

    /// Fit the Extended Isolation Tree on training data.
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
        self.n_train = n;
        self.dtypes = dtypes.to_vec();
        self.col_names = col_names.to_vec();
        self.nodes.clear();

        let max_d = self.max_depth.unwrap_or_else(|| {
            (n as f64).log2().ceil() as usize
        });

        let mut rng = LcgRng::new(self.seed);
        let all_indices: Vec<usize> = (0..n).collect();

        let root_idx = self.nodes.len();
        self.nodes.push(IsolationNode {
            is_leaf: true,
            split: None,
            left_child: None,
            right_child: None,
            size: n,
            depth: 0,
        });

        self.grow_tree(root_idx, &all_indices, 0, max_d, data, &mut rng);
    }

    fn grow_tree(
        &mut self,
        node_idx: usize,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        data: &[Vec<f64>],
        rng: &mut LcgRng,
    ) {
        let n = indices.len();

        if depth >= max_depth || n <= 1 {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            self.nodes[node_idx].depth = depth;
            return;
        }

        let d = data[0].len();
        let cont_cols: Vec<usize> = (0..d).filter(|&j| self.dtypes.get(j) != Some(&ColType::Categorical)).collect();
        let cat_cols: Vec<usize> = (0..d).filter(|&j| self.dtypes.get(j) == Some(&ColType::Categorical)).collect();

        // Check if all rows are identical across features
        let mut has_variation = false;
        if n > 1 {
            let first_idx = indices[0];
            for &idx in &indices[1..] {
                for col in 0..d {
                    if (data[idx][col] - data[first_idx][col]).abs() > 1e-9 {
                        has_variation = true;
                        break;
                    }
                }
                if has_variation {
                    break;
                }
            }
        }

        if !has_variation {
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            self.nodes[node_idx].depth = depth;
            return;
        }

        // Generate split
        let mut left_idxs = Vec::new();
        let mut right_idxs = Vec::new();
        let mut chosen_split = None;

        // Try up to 10 random candidates to find one that partitions the data
        for _ in 0..10 {
            // Decide whether to perform pure categorical split or oblique hyperplane
            let do_cat_split = !cat_cols.is_empty() && (cont_cols.is_empty() || rng.next_f64() < 0.3);

            if do_cat_split {
                // Select a random categorical column with >= 2 distinct values
                let cat_col = cat_cols[rng.next_usize(cat_cols.len())];
                let mut levels: Vec<u64> = indices
                    .iter()
                    .map(|&i| data[i][cat_col])
                    .filter(|v| !v.is_nan())
                    .map(|v| v.round() as u64)
                    .collect();
                levels.sort();
                levels.dedup();

                if levels.len() > 1 {
                    // Random non-empty proper subset of levels for Left
                    let mut left_levels = Vec::new();
                    for &lvl in &levels {
                        if rng.next_f64() < 0.5 {
                            left_levels.push(lvl);
                        }
                    }
                    if left_levels.is_empty() {
                        left_levels.push(levels[0]);
                    } else if left_levels.len() == levels.len() {
                        left_levels.pop();
                    }

                    let split = IsolationSplit::CategoricalSubset {
                        col: cat_col,
                        left_levels,
                        missing_left: rng.next_f64() < 0.5,
                    };

                    let mut l = Vec::new();
                    let mut r = Vec::new();
                    for &idx in indices {
                        if split.goes_left(&data[idx]) {
                            l.push(idx);
                        } else {
                            r.push(idx);
                        }
                    }
                    if !l.is_empty() && !r.is_empty() {
                        left_idxs = l;
                        right_idxs = r;
                        chosen_split = Some(split);
                        break;
                    }
                }
            }

            // Oblique hyperplane split
            let mut active_cont = Vec::new();
            let mut cont_weights = Vec::new();

            // Select coordinates based on extension_level
            let ext_level = self.extension_level.unwrap_or(cont_cols.len()).max(1);
            let n_coords = ext_level.min(cont_cols.len());

            let mut shuffled_cont = cont_cols.clone();
            rng.shuffle(&mut shuffled_cont);
            for &c in shuffled_cont.iter().take(n_coords) {
                active_cont.push(c);
                cont_weights.push(rng.next_gaussian());
            }

            // Assign random level weights for any included categorical features
            let mut active_cats = Vec::new();
            let mut cat_weight_maps = Vec::new();
            if !cat_cols.is_empty() {
                let cat_sample = cat_cols[rng.next_usize(cat_cols.len())];
                let mut levels: Vec<u64> = indices
                    .iter()
                    .map(|&i| data[i][cat_sample])
                    .filter(|v| !v.is_nan())
                    .map(|v| v.round() as u64)
                    .collect();
                levels.sort();
                levels.dedup();

                let mut w_map = HashMap::new();
                for &lvl in &levels {
                    w_map.insert(lvl, rng.next_gaussian());
                }
                active_cats.push(cat_sample);
                cat_weight_maps.push(w_map);
            }

            // Calculate projections
            let mut projs: Vec<f64> = Vec::with_capacity(n);
            let mut min_p = f64::INFINITY;
            let mut max_p = f64::NEG_INFINITY;

            for &idx in indices {
                let mut p = 0.0f64;
                for (&c, &w) in active_cont.iter().zip(cont_weights.iter()) {
                    let val = data[idx][c];
                    if !val.is_nan() {
                        p += w * val;
                    }
                }
                for (&c, w_map) in active_cats.iter().zip(cat_weight_maps.iter()) {
                    let val = data[idx][c];
                    if !val.is_nan() {
                        if let Some(&w) = w_map.get(&(val.round() as u64)) {
                            p += w;
                        }
                    }
                }
                projs.push(p);
                if p < min_p { min_p = p; }
                if p > max_p { max_p = p; }
            }

            if (max_p - min_p) > 1e-9 {
                let threshold = min_p + rng.next_f64() * (max_p - min_p);
                let split = IsolationSplit::Oblique {
                    cont_cols: active_cont,
                    cont_weights,
                    cat_cols: active_cats,
                    cat_weights: cat_weight_maps,
                    threshold,
                    missing_left: rng.next_f64() < 0.5,
                };

                let mut l = Vec::new();
                let mut r = Vec::new();
                for (i, &p) in projs.iter().enumerate() {
                    if p <= threshold {
                        l.push(indices[i]);
                    } else {
                        r.push(indices[i]);
                    }
                }

                if !l.is_empty() && !r.is_empty() {
                    left_idxs = l;
                    right_idxs = r;
                    chosen_split = Some(split);
                    break;
                }
            }
        }

        if let Some(split) = chosen_split {
            let left_node_idx = self.nodes.len();
            self.nodes.push(IsolationNode {
                is_leaf: true,
                split: None,
                left_child: None,
                right_child: None,
                size: left_idxs.len(),
                depth: depth + 1,
            });

            let right_node_idx = self.nodes.len();
            self.nodes.push(IsolationNode {
                is_leaf: true,
                split: None,
                left_child: None,
                right_child: None,
                size: right_idxs.len(),
                depth: depth + 1,
            });

            self.nodes[node_idx].is_leaf = false;
            self.nodes[node_idx].split = Some(split);
            self.nodes[node_idx].left_child = Some(left_node_idx);
            self.nodes[node_idx].right_child = Some(right_node_idx);

            self.grow_tree(left_node_idx, &left_idxs, depth + 1, max_depth, data, rng);
            self.grow_tree(right_node_idx, &right_idxs, depth + 1, max_depth, data, rng);
        } else {
            // Cannot split further
            self.nodes[node_idx].is_leaf = true;
            self.nodes[node_idx].size = n;
            self.nodes[node_idx].depth = depth;
        }
    }

    /// Compute path length h(x) to isolate sample x in this tree.
    pub fn path_length(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut curr = 0;
        let mut depth = 0.0f64;

        while !self.nodes[curr].is_leaf {
            let split = self.nodes[curr].split.as_ref().unwrap();
            let go_left = split.goes_left(row);
            curr = if go_left {
                self.nodes[curr].left_child.unwrap()
            } else {
                self.nodes[curr].right_child.unwrap()
            };
            depth += 1.0;
        }

        // Add expected path length adjustment for terminal leaf size
        let leaf_size = self.nodes[curr].size;
        depth + expected_path_length(leaf_size)
    }

    /// Compute anomaly score s(x, n) = 2^(-h(x) / c(n)).
    /// Score near 1.0 indicates anomaly (early isolation); near 0.5 indicates normal sample.
    pub fn anomaly_score(&self, row: &[f64]) -> f64 {
        let h = self.path_length(row);
        let c = expected_path_length(self.n_train);
        if c <= 0.0 {
            0.5
        } else {
            2.0f64.powf(-h / c)
        }
    }
}

/// Average path length c(n) of an unsuccessful search in a BST / Isolation Tree:
///   c(n) = 2 * (ln(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
#[inline]
pub fn expected_path_length(n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else if n == 2 {
        1.0
    } else {
        let n_f = n as f64;
        let euler_gamma = 0.5772156649015328606;
        2.0 * ((n_f - 1.0).ln() + euler_gamma) - (2.0 * (n_f - 1.0) / n_f)
    }
}

/// Simple fast linear congruential generator for reproducible random projections.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }

    /// Box-Muller transform for standard Gaussian variable.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}
