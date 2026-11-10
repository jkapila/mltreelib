//! Anomaly and outlier detection module for mltreelib.
//!
//! Provides seven tree-based anomaly paradigms:
//! 1. `GritBot`: Quinlan's contextual anomaly and subgroup quality auditor.
//! 2. `DensityEstimationTree`: Ram & Gray's Integrated Squared Error density tree.
//! 3. `ExtendedIsolationTree`: Hariri et al.'s oblique isolation tree with first-class categorical support.
//! 4. `HalfSpaceTree`: Mass-based streaming anomaly tree with fixed balanced topology.
//! 5. `RandomCutTree`: Robust random cut tree with Collusive Displacement (CoDisp).
//! 6. `ActiveAnomalyTree`: Active anomaly discovery tree with human-in-the-loop weight updates.
//! 7. `ContextAnomalyTree`: Context Tree Weighting / Probabilistic Suffix Tree for sequential and categorical surprise.

pub mod aad_tree;
pub mod context_tree;
pub mod det;
pub mod eif_tree;
pub mod gritbot;
pub mod hs_tree;
pub mod rc_tree;

pub use aad_tree::ActiveAnomalyTree;
pub use context_tree::ContextAnomalyTree;
pub use det::DensityEstimationTree;
pub use eif_tree::ExtendedIsolationTree;
pub use gritbot::{ConditionOp, GritAnomaly, GritBot, GritCondition};
pub use hs_tree::HalfSpaceTree;
pub use rc_tree::RandomCutTree;
