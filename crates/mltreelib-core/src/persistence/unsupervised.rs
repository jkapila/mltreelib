use std::collections::BTreeSet;

use crate::anomaly::eif_tree::{ExtendedIsolationTree, IsolationSplit};
use crate::anomaly::{
    ActiveAnomalyTree, ContextAnomalyTree, DensityEstimationTree, HalfSpaceTree, RandomCutTree,
};
use crate::cluster::{ClTree, HdbscanTree, PredictiveClusteringTree, UnsupervisedProximityForest};
use crate::persistence::{decode_snapshot, encode_snapshot, PersistenceError};

const EIF_FORMAT: &str = "mltreelib.eif";
const PCT_FORMAT: &str = "mltreelib.pct";
const HST_FORMAT: &str = "mltreelib.hst";
const RCT_FORMAT: &str = "mltreelib.rct";
const AAD_FORMAT: &str = "mltreelib.aad";
const CAT_FORMAT: &str = "mltreelib.cat";
const DET_FORMAT: &str = "mltreelib.det";
const CLTREE_FORMAT: &str = "mltreelib.cltree";
const HDBSCAN_FORMAT: &str = "mltreelib.hdbscan";
const URF_FORMAT: &str = "mltreelib.urf";

fn validate_tree(
    n: usize,
    mut children: impl FnMut(usize) -> Result<Option<[usize; 2]>, PersistenceError>,
) -> Result<(), PersistenceError> {
    let invalid = || PersistenceError::Invalid("invalid unsupervised tree topology");
    let mut seen = vec![false; n];
    let mut pending = vec![(0, 0)];
    let mut visited = 0;
    while let Some((index, depth)) = pending.pop() {
        if depth > 512 || index >= n || std::mem::replace(&mut seen[index], true) {
            return Err(invalid());
        }
        visited += 1;
        if let Some([left, right]) = children(index)? {
            pending.extend([(left, depth + 1), (right, depth + 1)]);
        }
    }
    if visited != n {
        return Err(invalid());
    }
    Ok(())
}

impl ExtendedIsolationTree {
    /// Serialize fitted EIF state as versioned JSON.
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        self.check_snapshot()?;
        encode_snapshot(EIF_FORMAT, self)
    }

    /// Restore fitted EIF state from versioned JSON.
    pub fn from_json(data: &str) -> Result<Self, PersistenceError> {
        let model: Self = decode_snapshot(EIF_FORMAT, data)?;
        model.check_snapshot()?;
        Ok(model)
    }

    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        let width = self.dtypes.len();
        if self.n_train == 0 || self.col_names.len() != width {
            return Err(PersistenceError::Invalid(
                "expected a fitted isolation tree",
            ));
        }
        validate_tree(self.nodes.len(), |index| {
            let node = &self.nodes[index];
            match (node.is_leaf, &node.split, node.left_child, node.right_child) {
                (true, None, None, None) => Ok(None),
                (false, Some(split), Some(left), Some(right)) => {
                    let valid = match split {
                        IsolationSplit::Oblique {
                            cont_cols,
                            cont_weights,
                            cat_cols,
                            cat_weights,
                            ..
                        } => {
                            cont_cols.len() == cont_weights.len()
                                && cat_cols.len() == cat_weights.len()
                                && cont_cols.iter().chain(cat_cols).all(|&col| col < width)
                        }
                        IsolationSplit::CategoricalSubset { col, .. } => *col < width,
                    };
                    if !valid {
                        return Err(PersistenceError::Invalid("invalid isolation split"));
                    }
                    Ok(Some([left, right]))
                }
                _ => Err(PersistenceError::Invalid("invalid isolation tree node")),
            }
        })
    }
}

impl PredictiveClusteringTree {
    /// Serialize fitted PCT state as versioned JSON.
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        self.check_snapshot()?;
        encode_snapshot(PCT_FORMAT, self)
    }

    /// Restore fitted PCT state from versioned JSON.
    pub fn from_json(data: &str) -> Result<Self, PersistenceError> {
        let model: Self = decode_snapshot(PCT_FORMAT, data)?;
        model.check_snapshot()?;
        Ok(model)
    }

    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        let width = self.dtypes.len();
        if self.n_clusters == 0 || self.col_names.len() != width {
            return Err(PersistenceError::Invalid(
                "expected a fitted predictive clustering tree",
            ));
        }
        let mut leaves = BTreeSet::new();
        validate_tree(self.nodes.len(), |index| {
            let node = &self.nodes[index];
            if node.prototype.len() != self.nodes[0].prototype.len() {
                return Err(PersistenceError::Invalid("invalid cluster prototype"));
            }
            match (node.is_leaf, node.left_child, node.right_child) {
                (true, None, None)
                    if node.cluster_id < self.n_clusters && leaves.insert(node.cluster_id) =>
                {
                    Ok(None)
                }
                (false, Some(left), Some(right)) if node.split_col < width => {
                    Ok(Some([left, right]))
                }
                _ => Err(PersistenceError::Invalid(
                    "invalid predictive clustering node",
                )),
            }
        })?;
        if leaves.len() != self.n_clusters {
            return Err(PersistenceError::Invalid(
                "invalid predictive clustering count",
            ));
        }
        Ok(())
    }
}

macro_rules! simple_snapshot {
    ($ty:ty, $format:expr, $check:ident) => {
        impl $ty {
            pub fn to_json(&self) -> Result<String, PersistenceError> {
                self.$check()?;
                encode_snapshot($format, self)
            }
            pub fn from_json(data: &str) -> Result<Self, PersistenceError> {
                let model: Self = decode_snapshot($format, data)?;
                model.$check()?;
                Ok(model)
            }
        }
    };
}

fn check_nodes<T>(
    nodes: &[T],
    children: impl Fn(usize, &T) -> Result<Option<[usize; 2]>, PersistenceError>,
) -> Result<(), PersistenceError> {
    if nodes.is_empty() {
        return Err(PersistenceError::Invalid(
            "expected a fitted unsupervised tree",
        ));
    }
    validate_tree(nodes.len(), |i| children(i, &nodes[i]))
}

impl HalfSpaceTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.n_samples == 0 || self.bounds.is_empty() {
            return Err(PersistenceError::Invalid(
                "expected a fitted half-space tree",
            ));
        }
        check_nodes(&self.nodes, |_, n| {
            match (n.is_leaf, n.left_child, n.right_child) {
                (true, None, None) => Ok(None),
                (false, Some(l), Some(r)) if n.split_dim < self.bounds.len() => Ok(Some([l, r])),
                _ => Err(PersistenceError::Invalid("invalid half-space tree node")),
            }
        })
    }
}
simple_snapshot!(HalfSpaceTree, HST_FORMAT, check_snapshot);

impl RandomCutTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.n_points == 0 || self.root != Some(0) {
            return Err(PersistenceError::Invalid(
                "expected a fitted random cut tree",
            ));
        }
        if self.nodes.first().is_none_or(|node| node.parent.is_some()) {
            return Err(PersistenceError::Invalid("invalid random cut root"));
        }
        check_nodes(&self.nodes, |index, n| {
            match (n.is_leaf, n.left_child, n.right_child) {
                (true, None, None)
                    if n.parent.is_none() || n.parent.is_some_and(|p| p < self.nodes.len()) =>
                {
                    Ok(None)
                }
                (false, Some(l), Some(r))
                    if n.split_dim < n.bbox.len()
                        && self.nodes.get(l).is_some_and(|node| node.parent == Some(index))
                        && self.nodes.get(r).is_some_and(|node| node.parent == Some(index)) =>
                {
                    Ok(Some([l, r]))
                }
                _ => Err(PersistenceError::Invalid("invalid random cut tree node")),
            }
        })
    }
}
simple_snapshot!(RandomCutTree, RCT_FORMAT, check_snapshot);

impl ActiveAnomalyTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.n_samples == 0 || self.weights.len() != self.nodes.len() {
            return Err(PersistenceError::Invalid(
                "expected a fitted active anomaly tree",
            ));
        }
        check_nodes(&self.nodes, |_, n| {
            match (n.is_leaf, n.left_child, n.right_child) {
                (true, None, None) => Ok(None),
                (false, Some(l), Some(r)) => Ok(Some([l, r])),
                _ => Err(PersistenceError::Invalid(
                    "invalid active anomaly tree node",
                )),
            }
        })
    }
}
simple_snapshot!(ActiveAnomalyTree, AAD_FORMAT, check_snapshot);

impl ContextAnomalyTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.nodes.is_empty() || self.alphabet_size < 2 {
            return Err(PersistenceError::Invalid("invalid context anomaly tree"));
        }
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![0];
        while let Some(index) = stack.pop() {
            if index >= self.nodes.len() || reachable[index] {
                return Err(PersistenceError::Invalid("invalid context tree topology"));
            }
            reachable[index] = true;
            let node = &self.nodes[index];
            if node.symbol_counts.values().try_fold(0usize, |total, &count| total.checked_add(count))
                != Some(node.total_count)
            {
                return Err(PersistenceError::Invalid("invalid context tree counts"));
            }
            for child in node.children.values() {
                if *child >= self.nodes.len() {
                    return Err(PersistenceError::Invalid("invalid context tree child"));
                }
                stack.push(*child);
            }
        }
        if reachable.iter().any(|&seen| !seen) {
            return Err(PersistenceError::Invalid("invalid context tree topology"));
        }
        Ok(())
    }
}
simple_snapshot!(ContextAnomalyTree, CAT_FORMAT, check_snapshot);

impl DensityEstimationTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.n_train == 0 || self.col_names.len() != self.dtypes.len() {
            return Err(PersistenceError::Invalid("expected a fitted density tree"));
        }
        if self.bounds.len() != self.dtypes.len() || self.cat_cardinalities.len() != self.dtypes.len() {
            return Err(PersistenceError::Invalid("invalid density tree dimensions"));
        }
        check_nodes(&self.nodes, |_, n| {
            match (n.is_leaf, &n.split, n.left_child, n.right_child) {
                (true, None, None, None) => Ok(None),
                (false, Some(split), Some(l), Some(r)) => {
                    let col = match split {
                        crate::anomaly::det::DetSplit::Continuous { col, .. }
                        | crate::anomaly::det::DetSplit::Categorical { col, .. } => *col,
                    };
                    if col >= self.dtypes.len() {
                        return Err(PersistenceError::Invalid(
                            "invalid density tree split column",
                        ));
                    }
                    Ok(Some([l, r]))
                }
                _ => Err(PersistenceError::Invalid("invalid density tree node")),
            }
        })
    }
}
simple_snapshot!(DensityEstimationTree, DET_FORMAT, check_snapshot);

impl ClTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        if self.nodes.is_empty() || self.bounds.is_empty() {
            return Err(PersistenceError::Invalid("expected a fitted CLTree"));
        }
        check_nodes(&self.nodes, |_, n| {
            match (n.is_leaf, n.left_child, n.right_child) {
                (true, None, None) => Ok(None),
                (false, Some(l), Some(r)) if n.split_dim < self.bounds.len() => Ok(Some([l, r])),
                _ => Err(PersistenceError::Invalid("invalid CLTree node")),
            }
        })
    }
}
simple_snapshot!(ClTree, CLTREE_FORMAT, check_snapshot);

impl HdbscanTree {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        let n = self.train_data.len();
        if n == 0
            || self.train_data[0].is_empty()
            || self.train_data.iter().skip(1).any(|row| row.len() != self.train_data[0].len())
        {
            return Err(PersistenceError::Invalid("invalid HDBSCAN training dimensions"));
        }
        if self.labels.len() != n || self.outlier_scores.len() != n {
            return Err(PersistenceError::Invalid("invalid HDBSCAN fitted sizes"));
        }
        if !self.core_distances.is_empty() && self.core_distances.len() != n {
            return Err(PersistenceError::Invalid("invalid HDBSCAN core distances"));
        }
        if self
            .labels
            .iter()
            .flatten()
            .any(|&label| label >= self.n_clusters)
        {
            return Err(PersistenceError::Invalid("invalid HDBSCAN cluster label"));
        }
        if self.condensed_nodes.is_empty() {
            if self.n_clusters != 0 {
                return Err(PersistenceError::Invalid("invalid HDBSCAN cluster count"));
            }
            return Ok(());
        }
        if self.condensed_nodes[0].node_id != 0 {
            return Err(PersistenceError::Invalid("invalid HDBSCAN condensed root"));
        }
        let count = self.condensed_nodes.len();
        validate_tree(count, |index| {
            let node = &self.condensed_nodes[index];
            if node.node_id != index
                || (index == 0 && node.parent_id.is_some())
                || (index != 0 && node.parent_id.is_none())
                || node.parent_id.is_some_and(|p| p >= count)
                || node.children.iter().any(|&child| child >= count)
                || node.points.iter().any(|&(point, _)| point >= n)
            {
                return Err(PersistenceError::Invalid(
                    "invalid HDBSCAN condensed reference",
                ));
            }
            Ok(if node.children.is_empty() {
                None
            } else {
                if node.children.len() != 2 {
                    return Err(PersistenceError::Invalid(
                        "invalid HDBSCAN condensed children",
                    ));
                }
                if node
                    .children
                    .iter()
                    .any(|&child| self.condensed_nodes[child].parent_id != Some(index))
                {
                    return Err(PersistenceError::Invalid(
                        "invalid HDBSCAN parent reference",
                    ));
                }
                Some([node.children[0], node.children[1]])
            })
        })
    }
}
simple_snapshot!(HdbscanTree, HDBSCAN_FORMAT, check_snapshot);

impl UnsupervisedProximityForest {
    fn check_snapshot(&self) -> Result<(), PersistenceError> {
        let n = self.cluster_labels.len();
        if self.trees.is_empty()
            || self.n_clusters == 0
            || self.outlier_scores.len() != n
            || self.proximity_matrix.len() != n
            || self.proximity_matrix.iter().any(|row| row.len() != n)
            || self.training_encodings.len() != n
            || self
                .training_encodings
                .iter()
                .any(|row| row.len() != self.trees.len())
        {
            return Err(PersistenceError::Invalid("invalid URF fitted state"));
        }
        for tree in &self.trees {
            check_nodes(&tree.nodes, |_, node| {
                match (node.is_leaf, node.left_child, node.right_child) {
                    (true, None, None) if node.leaf_id < tree.n_leaves => Ok(None),
                    (false, Some(left), Some(right)) => Ok(Some([left, right])),
                    _ => Err(PersistenceError::Invalid("invalid URF tree node")),
                }
            })?;
        }
        if self.training_encodings.iter().any(|row| {
            row.iter()
                .enumerate()
                .any(|(i, &leaf)| leaf >= self.trees[i].n_leaves)
        }) {
            return Err(PersistenceError::Invalid("invalid URF leaf encoding"));
        }
        if self
            .cluster_labels
            .iter()
            .any(|&label| label >= self.n_clusters)
        {
            return Err(PersistenceError::Invalid("invalid URF cluster label"));
        }
        Ok(())
    }
}
simple_snapshot!(UnsupervisedProximityForest, URF_FORMAT, check_snapshot);
