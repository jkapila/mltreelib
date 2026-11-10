//! Tree-based clustering algorithms for mltreelib.
//!
//! Provides:
//! - `PredictiveClusteringTree` (PCT / CLUS): Multi-target variance reduction clustering tree.
//! - `ClTree` (CLTree): Clustering through Decision Tree Construction (Liu et al. KDD 2000).
//! - `HdbscanTree` (HDBSCAN): Density-based hierarchy, condensed tree & GLOSH outlier scores.
//! - `UnsupervisedProximityForest` (URF): Unsupervised Random Forest proximity & clustering.

pub mod cltree;
pub mod hdbscan_tree;
pub mod pct;
pub mod urf;

pub use cltree::{ClTree, ClTreeNode};
pub use hdbscan_tree::{ClusterSelectionMethod, CondensedNode, HdbscanTree, Metric as HdbscanMetric};
pub use pct::{PctNode, PredictiveClusteringTree};
pub use urf::{ProximityTree, UnsupervisedProximityForest, UrfLinkage};

