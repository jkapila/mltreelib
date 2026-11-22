//! Optimal Decision Trees (Exact Global Search & Shallow OCT Solver).
//!
//! Solves for the provably globally optimal decision tree of shallow depth (depth 1 or 2)
//! minimizing 0-1 misclassification loss or regression mean squared error,
//! with complexity penalty alpha per split.

use serde::{Deserialize, Serialize};

/// Node in an optimal decision tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimalTreeNode {
    pub is_leaf: bool,
    pub feature: usize,
    pub threshold: f64,
    pub value: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub n_samples: usize,
    pub impurity: f64,
}

/// Result of an exact shallow optimal decision tree optimization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimalTreeResult {
    pub task: String,
    pub max_depth: usize,
    pub n_splits: usize,
    pub total_loss: f64,
    pub nodes: Vec<OptimalTreeNode>,
}

impl OptimalTreeResult {
    /// Predicts a single sample across the optimal tree.
    pub fn predict_row(&self, x: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut curr = 0;
        while !self.nodes[curr].is_leaf {
            let feat = self.nodes[curr].feature;
            let thresh = self.nodes[curr].threshold;
            let val = x.get(feat).copied().unwrap_or(0.0);
            if val <= thresh {
                match self.nodes[curr].left_child {
                    Some(left) => curr = left,
                    None => break,
                }
            } else {
                match self.nodes[curr].right_child {
                    Some(right) => curr = right,
                    None => break,
                }
            }
        }
        self.nodes[curr].value
    }

    /// Predicts multiple samples.
    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| self.predict_row(row)).collect()
    }

    /// Serializes optimal tree structure to JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
}

/// Solves for the globally optimal decision tree of depth <= 2.
///
/// Parameters:
/// - `x`: N rows x P features matrix
/// - `y`: N target values (class labels in {0, 1, ...} for classification or continuous for regression)
/// - `task`: "classification" or "regression"
/// - `max_depth`: 1 (stump) or 2 (up to 3 splits, 4 leaves)
/// - `min_samples_leaf`: minimum samples required in each leaf
/// - `alpha`: complexity penalty added for each internal split
pub fn fit_optimal_tree(
    x: &[Vec<f64>],
    y: &[f64],
    task: &str,
    max_depth: usize,
    min_samples_leaf: usize,
    alpha: f64,
) -> Result<OptimalTreeResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("x and y cannot be empty".into());
    }
    let n = x.len();
    if y.len() != n {
        return Err(format!("mismatched sample count: x has {n}, y has {}", y.len()));
    }
    let p = x[0].len();
    if p == 0 {
        return Err("x must have at least 1 feature column".into());
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(format!("row {i} has {} columns, expected {p}", row.len()));
        }
    }
    if max_depth != 1 && max_depth != 2 {
        return Err(format!("max_depth must be 1 or 2 for exact OCT solver, got {max_depth}"));
    }
    let min_leaf = min_samples_leaf.max(1);
    let is_classification = match task {
        "classification" => true,
        "regression" => false,
        other => return Err(format!("unknown task {other:?}, expected 'classification' or 'regression'")),
    };

    let all_indices: Vec<usize> = (0..n).collect();

    // Leaf loss of a single partition
    let (root_leaf_val, root_leaf_loss) = compute_leaf_stats(y, &all_indices, is_classification);

    // If max_depth == 1 (Decision Stump):
    if max_depth == 1 {
        let best_split = find_best_split(x, y, &all_indices, is_classification, min_leaf);
        if let Some((feat, thresh, loss_split, left_idx, right_idx)) = best_split {
            if loss_split + alpha < root_leaf_loss {
                let (left_val, left_loss) = compute_leaf_stats(y, &left_idx, is_classification);
                let (right_val, right_loss) = compute_leaf_stats(y, &right_idx, is_classification);
                let nodes = vec![
                    OptimalTreeNode {
                        is_leaf: false,
                        feature: feat,
                        threshold: thresh,
                        value: root_leaf_val,
                        left_child: Some(1),
                        right_child: Some(2),
                        n_samples: n,
                        impurity: root_leaf_loss,
                    },
                    OptimalTreeNode {
                        is_leaf: true,
                        feature: 0,
                        threshold: 0.0,
                        value: left_val,
                        left_child: None,
                        right_child: None,
                        n_samples: left_idx.len(),
                        impurity: left_loss,
                    },
                    OptimalTreeNode {
                        is_leaf: true,
                        feature: 0,
                        threshold: 0.0,
                        value: right_val,
                        left_child: None,
                        right_child: None,
                        n_samples: right_idx.len(),
                        impurity: right_loss,
                    },
                ];
                return Ok(OptimalTreeResult {
                    task: task.to_string(),
                    max_depth: 1,
                    n_splits: 1,
                    total_loss: loss_split + alpha,
                    nodes,
                });
            }
        }
        // No split improved upon root
        let nodes = vec![OptimalTreeNode {
            is_leaf: true,
            feature: 0,
            threshold: 0.0,
            value: root_leaf_val,
            left_child: None,
            right_child: None,
            n_samples: n,
            impurity: root_leaf_loss,
        }];
        return Ok(OptimalTreeResult {
            task: task.to_string(),
            max_depth: 1,
            n_splits: 0,
            total_loss: root_leaf_loss,
            nodes,
        });
    }

    // Depth 2:
    // Global search over root split (feat0, thresh0).
    // For each candidate root split, partition into L and R.
    // By dynamic programming / Bellman optimality:
    // optimal left sub-split depends only on L, and optimal right sub-split depends only on R!
    let mut best_global_loss = root_leaf_loss;
    let mut best_tree_nodes: Option<Vec<OptimalTreeNode>> = None;
    let mut best_splits_count = 0;

    // Scan all candidate root splits
    for f in 0..p {
        let candidate_threshs = get_candidate_thresholds(x, f, &all_indices);
        for &thresh in &candidate_threshs {
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &idx in &all_indices {
                if x[idx][f] <= thresh {
                    left_idx.push(idx);
                } else {
                    right_idx.push(idx);
                }
            }
            if left_idx.len() < min_leaf || right_idx.len() < min_leaf {
                continue;
            }

            // Evaluate left branch: keep as leaf OR split further
            let (l_val, l_loss) = compute_leaf_stats(y, &left_idx, is_classification);
            let l_split = if left_idx.len() >= 2 * min_leaf {
                find_best_split(x, y, &left_idx, is_classification, min_leaf)
            } else {
                None
            };
            let (best_l_loss, l_is_split, l_split_opt) = match l_split {
                Some((lf, lt, l_spl_loss, l_l_idx, l_r_idx)) if l_spl_loss + alpha < l_loss => {
                    (l_spl_loss + alpha, true, Some((lf, lt, l_spl_loss, l_l_idx, l_r_idx)))
                }
                _ => (l_loss, false, None),
            };

            // Evaluate right branch: keep as leaf OR split further
            let (r_val, r_loss) = compute_leaf_stats(y, &right_idx, is_classification);
            let r_split = if right_idx.len() >= 2 * min_leaf {
                find_best_split(x, y, &right_idx, is_classification, min_leaf)
            } else {
                None
            };
            let (best_r_loss, r_is_split, r_split_opt) = match r_split {
                Some((rf, rt, r_spl_loss, r_l_idx, r_r_idx)) if r_spl_loss + alpha < r_loss => {
                    (r_spl_loss + alpha, true, Some((rf, rt, r_spl_loss, r_l_idx, r_r_idx)))
                }
                _ => (r_loss, false, None),
            };

            let total_tree_loss = best_l_loss + best_r_loss + alpha; // alpha for root split
            if total_tree_loss < best_global_loss {
                best_global_loss = total_tree_loss;

                // Build node list for this tree
                let mut nodes = Vec::new();
                let mut splits_count = 1;

                // Node 0: Root
                nodes.push(OptimalTreeNode {
                    is_leaf: false,
                    feature: f,
                    threshold: thresh,
                    value: root_leaf_val,
                    left_child: Some(1),
                    right_child: None, // set below
                    n_samples: n,
                    impurity: root_leaf_loss,
                });

                // Node 1: Left child
                if l_is_split {
                    splits_count += 1;
                    let (lf, lt, _, ll_idx, lr_idx) = l_split_opt.unwrap();
                    let (ll_val, ll_loss) = compute_leaf_stats(y, &ll_idx, is_classification);
                    let (lr_val, lr_loss) = compute_leaf_stats(y, &lr_idx, is_classification);

                    let _left_node_idx = 1;
                    let ll_node_idx = 2;
                    let lr_node_idx = 3;

                    nodes.push(OptimalTreeNode {
                        is_leaf: false,
                        feature: lf,
                        threshold: lt,
                        value: l_val,
                        left_child: Some(ll_node_idx),
                        right_child: Some(lr_node_idx),
                        n_samples: left_idx.len(),
                        impurity: l_loss,
                    });
                    nodes.push(OptimalTreeNode {
                        is_leaf: true,
                        feature: 0,
                        threshold: 0.0,
                        value: ll_val,
                        left_child: None,
                        right_child: None,
                        n_samples: ll_idx.len(),
                        impurity: ll_loss,
                    });
                    nodes.push(OptimalTreeNode {
                        is_leaf: true,
                        feature: 0,
                        threshold: 0.0,
                        value: lr_val,
                        left_child: None,
                        right_child: None,
                        n_samples: lr_idx.len(),
                        impurity: lr_loss,
                    });
                    let right_node_idx = nodes.len();
                    nodes[0].right_child = Some(right_node_idx);

                    if r_is_split {
                        splits_count += 1;
                        let (rf, rt, _, rl_idx, rr_idx) = r_split_opt.unwrap();
                        let (rl_val, rl_loss) = compute_leaf_stats(y, &rl_idx, is_classification);
                        let (rr_val, rr_loss) = compute_leaf_stats(y, &rr_idx, is_classification);

                        let rl_node_idx = right_node_idx + 1;
                        let rr_node_idx = right_node_idx + 2;

                        nodes.push(OptimalTreeNode {
                            is_leaf: false,
                            feature: rf,
                            threshold: rt,
                            value: r_val,
                            left_child: Some(rl_node_idx),
                            right_child: Some(rr_node_idx),
                            n_samples: right_idx.len(),
                            impurity: r_loss,
                        });
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: rl_val,
                            left_child: None,
                            right_child: None,
                            n_samples: rl_idx.len(),
                            impurity: rl_loss,
                        });
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: rr_val,
                            left_child: None,
                            right_child: None,
                            n_samples: rr_idx.len(),
                            impurity: rr_loss,
                        });
                    } else {
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: r_val,
                            left_child: None,
                            right_child: None,
                            n_samples: right_idx.len(),
                            impurity: r_loss,
                        });
                    }
                } else {
                    nodes.push(OptimalTreeNode {
                        is_leaf: true,
                        feature: 0,
                        threshold: 0.0,
                        value: l_val,
                        left_child: None,
                        right_child: None,
                        n_samples: left_idx.len(),
                        impurity: l_loss,
                    });
                    let right_node_idx = 2;
                    nodes[0].right_child = Some(right_node_idx);

                    if r_is_split {
                        splits_count += 1;
                        let (rf, rt, _, rl_idx, rr_idx) = r_split_opt.unwrap();
                        let (rl_val, rl_loss) = compute_leaf_stats(y, &rl_idx, is_classification);
                        let (rr_val, rr_loss) = compute_leaf_stats(y, &rr_idx, is_classification);

                        let rl_node_idx = 3;
                        let rr_node_idx = 4;

                        nodes.push(OptimalTreeNode {
                            is_leaf: false,
                            feature: rf,
                            threshold: rt,
                            value: r_val,
                            left_child: Some(rl_node_idx),
                            right_child: Some(rr_node_idx),
                            n_samples: right_idx.len(),
                            impurity: r_loss,
                        });
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: rl_val,
                            left_child: None,
                            right_child: None,
                            n_samples: rl_idx.len(),
                            impurity: rl_loss,
                        });
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: rr_val,
                            left_child: None,
                            right_child: None,
                            n_samples: rr_idx.len(),
                            impurity: rr_loss,
                        });
                    } else {
                        nodes.push(OptimalTreeNode {
                            is_leaf: true,
                            feature: 0,
                            threshold: 0.0,
                            value: r_val,
                            left_child: None,
                            right_child: None,
                            n_samples: right_idx.len(),
                            impurity: r_loss,
                        });
                    }
                }

                best_tree_nodes = Some(nodes);
                best_splits_count = splits_count;
            }
        }
    }

    if let Some(nodes) = best_tree_nodes {
        Ok(OptimalTreeResult {
            task: task.to_string(),
            max_depth: 2,
            n_splits: best_splits_count,
            total_loss: best_global_loss,
            nodes,
        })
    } else {
        // No split improved upon root
        let nodes = vec![OptimalTreeNode {
            is_leaf: true,
            feature: 0,
            threshold: 0.0,
            value: root_leaf_val,
            left_child: None,
            right_child: None,
            n_samples: n,
            impurity: root_leaf_loss,
        }];
        Ok(OptimalTreeResult {
            task: task.to_string(),
            max_depth: 2,
            n_splits: 0,
            total_loss: root_leaf_loss,
            nodes,
        })
    }
}

/// Computes candidate thresholds along a feature column for given subset of sample indices.
fn get_candidate_thresholds(x: &[Vec<f64>], feat: usize, indices: &[usize]) -> Vec<f64> {
    let mut vals: Vec<f64> = indices.iter().map(|&i| x[i][feat]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    if vals.len() <= 1 {
        return Vec::new();
    }
    let mut threshs = Vec::with_capacity(vals.len() - 1);
    for i in 0..vals.len() - 1 {
        threshs.push(0.5 * (vals[i] + vals[i + 1]));
    }
    threshs
}

/// Finds the single best split for a partition.
fn find_best_split(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    is_classification: bool,
    min_leaf: usize,
) -> Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> {
    let p = x[0].len();
    let mut best_loss = f64::INFINITY;
    let mut best_split = None;

    for f in 0..p {
        let threshs = get_candidate_thresholds(x, f, indices);
        for &thresh in &threshs {
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &idx in indices {
                if x[idx][f] <= thresh {
                    left_idx.push(idx);
                } else {
                    right_idx.push(idx);
                }
            }
            if left_idx.len() < min_leaf || right_idx.len() < min_leaf {
                continue;
            }

            let (_, l_loss) = compute_leaf_stats(y, &left_idx, is_classification);
            let (_, r_loss) = compute_leaf_stats(y, &right_idx, is_classification);
            let split_loss = l_loss + r_loss;

            if split_loss < best_loss {
                best_loss = split_loss;
                best_split = Some((f, thresh, split_loss, left_idx, right_idx));
            }
        }
    }
    best_split
}

/// Computes the optimal constant leaf prediction and total loss for a sample subset.
fn compute_leaf_stats(y: &[f64], indices: &[usize], is_classification: bool) -> (f64, f64) {
    if indices.is_empty() {
        return (0.0, 0.0);
    }
    let count = indices.len() as f64;
    if is_classification {
        // Find majority class in indices
        use std::collections::HashMap;
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        for &i in indices {
            let label = y[i].round() as i64;
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let mut best_label = 0;
        let mut max_count = 0;
        for (&label, &cnt) in &class_counts {
            if cnt > max_count {
                max_count = cnt;
                best_label = label;
            }
        }
        let misclassifications = (indices.len() - max_count) as f64;
        (best_label as f64, misclassifications)
    } else {
        // Regression: mean minimizes SSE
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let mean = sum / count;
        let sse: f64 = indices.iter().map(|&i| {
            let d = y[i] - mean;
            d * d
        }).sum();
        (mean, sse)
    }
}
