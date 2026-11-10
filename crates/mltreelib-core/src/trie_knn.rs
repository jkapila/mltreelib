//! TrieKNN: Non-binary categorical prefix trie with local numerical K-Nearest Neighbors.
//!
//! Concept by Jitin Kapila (2018 / 2025):
//! Sidesteps distance distortion and dimensional explosion on mixed tabular data by:
//! 1. Routing categorical combinations through a non-binary prefix Trie.
//! 2. Fitting localized KNN models on numerical features at nodes/leaves.
//! 3. Hierarchical back-off: if a leaf has fewer than K samples or encounters unseen
//!    categorical branches, it seamlessly climbs to parent prefix nodes.

use std::collections::{BTreeMap, HashMap};

/// Task type for TrieKNN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrieKnnTask {
    Classification,
    Regression,
}

/// A node in the non-binary categorical Trie.
#[derive(Clone, Debug)]
pub struct TrieKnnNode {
    pub depth: usize,
    pub parent: Option<usize>,
    pub children: HashMap<u64, usize>,
    pub sample_indices: Vec<usize>,
    pub count: usize,
    /// Class counts for classification.
    pub class_counts: BTreeMap<i64, usize>,
    /// Running sum of target for regression.
    pub target_sum: f64,
}

impl TrieKnnNode {
    pub fn new(depth: usize, parent: Option<usize>) -> Self {
        Self {
            depth,
            parent,
            children: HashMap::new(),
            sample_indices: Vec::new(),
            count: 0,
            class_counts: BTreeMap::new(),
            target_sum: 0.0,
        }
    }
}

/// Distance weighting mode for KNN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnnWeights {
    Uniform,
    Distance,
}

/// Distance metric used for leaf numerical KNN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnMetric {
    Euclidean,
    Manhattan,
}

/// TrieKNN model for mixed-type tabular classification and regression.
#[derive(Clone, Debug)]
pub struct TrieKnn {
    pub task: TrieKnnTask,
    pub k: usize,
    pub weights: KnnWeights,
    pub metric: KnnMetric,
    pub leaf_min_size: Option<usize>,
    pub cat_cols: Vec<usize>,
    pub num_cols: Vec<usize>,
    pub nodes: Vec<TrieKnnNode>,
    /// Sorted list of unique class labels for classification.
    pub classes: Vec<i64>,
    /// Stored numeric feature matrix: [N, d_num].
    pub num_data: Vec<Vec<f64>>,
    /// Stored targets: class indices or continuous values.
    pub targets: Vec<f64>,
}

impl TrieKnn {
    pub fn new(
        task: TrieKnnTask,
        k: usize,
        weights: KnnWeights,
        cat_cols: Vec<usize>,
        num_cols: Vec<usize>,
    ) -> Self {
        Self {
            task,
            k: k.max(1),
            weights,
            metric: KnnMetric::Euclidean,
            leaf_min_size: None,
            cat_cols,
            num_cols,
            nodes: vec![TrieKnnNode::new(0, None)],
            classes: Vec::new(),
            num_data: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Fit the TrieKNN model on mixed tabular data.
    ///
    /// - `data`: full feature rows (numeric values and encoded categorical IDs).
    /// - `targets`: target values (integer classes or continuous floats).
    pub fn fit(&mut self, data: &[Vec<f64>], targets: &[f64]) {
        let n = data.len();
        self.num_data.clear();
        self.targets.clear();
        self.nodes = vec![TrieKnnNode::new(0, None)];

        if n == 0 {
            return;
        }

        self.targets = targets.to_vec();

        // If classification, identify unique classes
        if self.task == TrieKnnTask::Classification {
            let mut cls_set = std::collections::BTreeSet::new();
            for &y in targets {
                cls_set.insert(y.round() as i64);
            }
            self.classes = cls_set.into_iter().collect();
        }

        // Cache numerical features
        self.num_data.reserve(n);
        for row in data {
            let num_row: Vec<f64> = self.num_cols.iter().map(|&j| row[j]).collect();
            self.num_data.push(num_row);
        }

        // Insert each sample into the Trie
        for i in 0..n {
            let mut curr = 0;
            self.nodes[curr].count += 1;
            self.nodes[curr].sample_indices.push(i);
            self.nodes[curr].target_sum += targets[i];

            if self.task == TrieKnnTask::Classification {
                let c = targets[i].round() as i64;
                *self.nodes[curr].class_counts.entry(c).or_insert(0) += 1;
            }

            for &cat_col in &self.cat_cols {
                let cat_val = if data[i][cat_col].is_nan() {
                    u64::MAX // missing categorical code
                } else {
                    data[i][cat_col].round() as u64
                };

                let next_idx = if let Some(&child) = self.nodes[curr].children.get(&cat_val) {
                    child
                } else {
                    let new_idx = self.nodes.len();
                    self.nodes.push(TrieKnnNode::new(self.nodes[curr].depth + 1, Some(curr)));
                    self.nodes[curr].children.insert(cat_val, new_idx);
                    new_idx
                };

                curr = next_idx;
                self.nodes[curr].count += 1;
                self.nodes[curr].sample_indices.push(i);
                self.nodes[curr].target_sum += targets[i];

                if self.task == TrieKnnTask::Classification {
                    let c = targets[i].round() as i64;
                    *self.nodes[curr].class_counts.entry(c).or_insert(0) += 1;
                }
            }
        }
    }

    /// Route a query row down the categorical prefix trie, returning the deepest matching node.
    pub fn route(&self, row: &[f64]) -> usize {
        let mut curr = 0;
        for &cat_col in &self.cat_cols {
            let cat_val = if row.get(cat_col).copied().unwrap_or(f64::NAN).is_nan() {
                u64::MAX
            } else {
                row[cat_col].round() as u64
            };

            if let Some(&child) = self.nodes[curr].children.get(&cat_val) {
                curr = child;
            } else {
                // Unseen categorical branch: stop at current prefix node
                break;
            }
        }
        curr
    }

    /// Gather candidate sample indices for KNN with hierarchical parent back-off.
    /// If the matched leaf has fewer than K samples, climbs up parent nodes
    /// until at least K unique samples are collected (or reaches the root).
    pub fn gather_candidate_indices(&self, start_node: usize) -> Vec<usize> {
        let mut gathered = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut curr = Some(start_node);
        let min_target = self.leaf_min_size.unwrap_or(self.k);

        while let Some(idx) = curr {
            for &sample_idx in &self.nodes[idx].sample_indices {
                if seen.insert(sample_idx) {
                    gathered.push(sample_idx);
                }
            }
            if gathered.len() >= min_target {
                break;
            }
            curr = self.nodes[idx].parent;
        }

        gathered
    }

    /// Compute distance between two numeric vectors according to KnnMetric.
    pub fn compute_dist(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.metric {
            KnnMetric::Euclidean => {
                let mut d2 = 0.0f64;
                for i in 0..a.len() {
                    let diff = a[i] - b[i];
                    d2 += diff * diff;
                }
                d2.sqrt()
            }
            KnnMetric::Manhattan => {
                let mut sum = 0.0f64;
                for i in 0..a.len() {
                    sum += (a[i] - b[i]).abs();
                }
                sum
            }
        }
    }

    /// Predict probabilities for classification query.
    pub fn predict_proba(&self, row: &[f64]) -> Vec<f64> {
        if self.classes.is_empty() {
            return Vec::new();
        }

        let query_num: Vec<f64> = self.num_cols.iter().map(|&j| row[j]).collect();
        let leaf = self.route(row);
        let candidates = self.gather_candidate_indices(leaf);

        if candidates.is_empty() {
            // Prior fallback: root class counts
            let total = self.nodes[0].count.max(1) as f64;
            return self
                .classes
                .iter()
                .map(|c| {
                    let cnt = self.nodes[0].class_counts.get(c).copied().unwrap_or(0) as f64;
                    cnt / total
                })
                .collect();
        }

        // Compute distances on numerical features
        let mut dists: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&idx| {
                let d = self.compute_dist(&query_num, &self.num_data[idx]);
                (idx, d)
            })
            .collect();

        // Sort by distance ascending
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = dists.iter().take(self.k);

        // Compute weighted votes aligned to classes
        let mut votes = vec![0.0f64; self.classes.len()];
        let mut total_weight = 0.0f64;

        for &(idx, dist) in top_k {
            let w = match self.weights {
                KnnWeights::Uniform => 1.0,
                KnnWeights::Distance => 1.0 / (dist + 1e-5),
            };
            let c = self.targets[idx].round() as i64;
            if let Some(pos) = self.classes.iter().position(|&cls| cls == c) {
                votes[pos] += w;
                total_weight += w;
            }
        }

        if total_weight > 0.0 {
            for v in &mut votes {
                *v /= total_weight;
            }
        } else {
            // Uniform fallback
            let uniform = 1.0 / (self.classes.len() as f64);
            for v in &mut votes {
                *v = uniform;
            }
        }

        votes
    }

    /// Predict single class label (argmax) or regression continuous value.
    pub fn predict(&self, row: &[f64]) -> f64 {
        match self.task {
            TrieKnnTask::Classification => {
                let probs = self.predict_proba(row);
                let mut best_prob = -1.0;
                let mut best_cls = self.classes.first().copied().unwrap_or(0);
                for (i, &p) in probs.iter().enumerate() {
                    if p > best_prob {
                        best_prob = p;
                        best_cls = self.classes[i];
                    }
                }
                best_cls as f64
            }
            TrieKnnTask::Regression => {
                let query_num: Vec<f64> = self.num_cols.iter().map(|&j| row[j]).collect();
                let leaf = self.route(row);
                let candidates = self.gather_candidate_indices(leaf);

                if candidates.is_empty() {
                    return self.nodes[0].target_sum / (self.nodes[0].count.max(1) as f64);
                }

                let mut dists: Vec<(usize, f64)> = candidates
                    .iter()
                    .map(|&idx| {
                        let d = self.compute_dist(&query_num, &self.num_data[idx]);
                        (idx, d)
                    })
                    .collect();

                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_k = dists.iter().take(self.k);

                let mut weighted_sum = 0.0f64;
                let mut total_weight = 0.0f64;

                for &(idx, dist) in top_k {
                    let w = match self.weights {
                        KnnWeights::Uniform => 1.0,
                        KnnWeights::Distance => 1.0 / (dist + 1e-5),
                    };
                    weighted_sum += w * self.targets[idx];
                    total_weight += w;
                }

                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    self.nodes[0].target_sum / (self.nodes[0].count.max(1) as f64)
                }
            }
        }
    }
}
