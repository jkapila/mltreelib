//! Tree-based clustering algorithms for mltreelib.
//!
//! Provides:
//! - `PredictiveClusteringTree` (PCT / CLUS): Multi-target variance reduction clustering tree.

pub mod pct;

pub use pct::{PctNode, PredictiveClusteringTree};
