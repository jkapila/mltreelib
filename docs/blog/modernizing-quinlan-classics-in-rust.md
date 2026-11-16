# Why We Rebuilt Quinlan's Cubist and C5.0 in Rust

**Author:** Jitin Kapila & mltreelib Team  
**Category:** Core Architecture & Systems ML  

---

## The Forgotten Masterpieces of Machine Learning

Long before gradient boosting and tabular foundation models dominated benchmarks, J. Ross Quinlan introduced a series of machine learning innovations that remain unmatched in rule interpretability and compact leaf modeling:
- **C4.5 / C5.0**: Multi-way discrete branching, rule-set extraction with pruning, and information gain ratio.
- **Cubist**: M5-style model trees combining logical rules with multivariate linear models at each terminal leaf, backed by composite committee boosting and sample neighbor adjustments.
- **GritBot**: Contextual anomaly auditing that uses tree partitions to detect local condition-specific outliers.

In R, packages like `Cubist` and `C50` developed cult followings among chemometricians and domain scientists. However, in Python, these algorithms remained trapped in 1990s C extensions—plagued by segmentation faults, messy memory leaks, and lack of modern DataFrame/Arrow integration.

---

## Enter Rust: Safety, SIMD, and Zero-Copy Frames

`mltreelib` rewrites Quinlan's algorithms from scratch in modern, memory-safe Rust (`crates/mltreelib-core`):

1. **Zero-Copy Arrow Buffering**: Ingests PyArrow and Polars tables directly into contiguous column memory buffers.
2. **Streaming Split Scans**: Implements $O(N)$ numerical linear scans and SIMD aggregations without intermediate heap allocations.
3. **Pure Python Frontends**: Exposed via thin PyO3 bindings with a clean, modern API (`mltreelib._core`).
4. **First-Class Missingness**: Implements Quinlan's original fractional missing value weight distributions without fragile post-hoc imputers.

The result is a production-grade implementation ready for modern tabular machine learning pipelines.
