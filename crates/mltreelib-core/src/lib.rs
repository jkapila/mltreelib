//! mltreelib-core: dataset component (data layer + adjustable benchmark
//! generator) and, next, the tree learners. Pure Rust; the Python and R
//! bindings stay thin shells around this crate.

pub mod anomaly;
pub mod builder;
pub mod calibration;
pub mod cluster;
pub mod criteria;
pub mod ctree;
pub mod encode;
pub mod family;
pub mod frame;
pub mod gate;
pub mod generator;
pub mod gritbot;
pub mod leaf;
pub mod metrics;
pub mod metric_split;
pub mod mob;
pub mod model_tree;
pub mod node_stats;
pub mod nn;
pub mod prune;
pub mod split;
pub mod stats;
pub mod surrogate;
pub mod text_print;
pub mod preselect;
pub mod index;
pub mod schema;
pub mod timefeat;
pub mod transform;
pub mod trie_knn;
pub mod mars;
pub mod rules;
pub mod knn;
pub mod solver;
pub mod fast_interaction;
pub mod report;
pub mod component;
mod persistence;

pub use solver::generic;
pub use fast_interaction::{fast_interaction_screen, FastInteraction};
pub use knn::{DistanceMetric, KnnVariant, LeafKnnModel};
pub use rules::{ConditionOp, Rule, RuleCondition, RuleSet};
pub use report::ModelReport;
pub use component::{ComponentModelCore, FactorAnalysisModel, FastIcaModel, NmfModel, NmfSolver, PcaModel, RotationMethod, TruncatedSvdModel, WhiteningKind, WhiteningModel};
pub use persistence::PersistenceError;
