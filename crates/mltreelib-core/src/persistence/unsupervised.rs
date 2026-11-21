use std::collections::BTreeSet;

use crate::anomaly::eif_tree::{ExtendedIsolationTree, IsolationSplit};
use crate::cluster::PredictiveClusteringTree;
use crate::persistence::{decode_snapshot, encode_snapshot, PersistenceError};

const EIF_FORMAT: &str = "mltreelib.eif";
const PCT_FORMAT: &str = "mltreelib.pct";

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
