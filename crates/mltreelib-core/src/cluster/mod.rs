//! Tree-based clustering algorithms for mltreelib.
//!
//! Provides:
//! - `PredictiveClusteringTree` (PCT / CLUS): Multi-target variance reduction clustering tree.
//! - `ClTree` (CLTree): Clustering through Decision Tree Construction (Liu et al. KDD 2000).

pub mod cltree;
pub mod pct;

pub use cltree::{ClTree, ClTreeNode};
pub use pct::{PctNode, PredictiveClusteringTree};
