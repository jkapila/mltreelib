//! mltreelib-core: dataset component (data layer + adjustable benchmark
//! generator) and, next, the tree learners. Pure Rust; the Python and R
//! bindings stay thin shells around this crate.

pub mod builder;
pub mod frame;
pub mod generator;
pub mod index;
pub mod schema;
pub mod transform;
