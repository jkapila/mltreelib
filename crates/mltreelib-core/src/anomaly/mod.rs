//! Anomaly and outlier detection module for mltreelib.
//!
//! Provides three tree-based anomaly paradigms:
//! 1. `GritBot`: Quinlan's contextual anomaly and subgroup quality auditor.
//! 2. `DensityEstimationTree`: Ram & Gray's Integrated Squared Error density tree.
//! 3. `ExtendedIsolationTree`: Hariri et al.'s oblique isolation tree with first-class categorical support.

pub mod det;
pub mod eif_tree;
pub mod gritbot;

pub use det::DensityEstimationTree;
pub use eif_tree::ExtendedIsolationTree;
pub use gritbot::{ConditionOp, GritAnomaly, GritBot, GritCondition};
