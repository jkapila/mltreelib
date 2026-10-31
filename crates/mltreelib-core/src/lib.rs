//! mltreelib-core: dataset component (data layer + adjustable benchmark
//! generator) and, next, the tree learners. Pure Rust; the Python and R
//! bindings stay thin shells around this crate.

pub mod builder;
pub mod criteria;
pub mod encode;
pub mod family;
pub mod frame;
pub mod gate;
pub mod generator;
pub mod leaf;
pub mod metrics;
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
