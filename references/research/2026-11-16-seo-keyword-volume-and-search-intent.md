# SEO Search Volume & Keyword Intent Analysis for mltreelib

**Date:** 2026-11-16  
**Source:** Live Google US search data via `treg` (`serpstat` provider live database)  
**Scope:** Decision trees, rule-based models, CART, C4.5/C5.0, Cubist, tabular machine learning, Rust ML  

---

## 1. Executive Summary & Strategy

`mltreelib` occupies a distinct technical niche:
1. **Tree and rule-based classics in Rust** (CART, C4.5, C5.0, Cubist, GritBot).
2. **First-class missing data semantics** (`None` + `NaN` = missing; no silent panics; explicit imputation).
3. **Structured feature engineering** (target encoding, `GroupedScaler`, time transforms).
4. **Research challenge to tabular foundation models** (TabArena benchmarks).

The SEO search landscape shows high-volume demand for general decision tree implementations (`decision tree`: 18.1k, `decision tree python`: 210, `decision tree classifier`: 590), while high-intent, low-difficulty technical terms (`cart modeling`: 210, `cart algorithm`: 170, `rust machine learning`: 170, `c4.5 algorithm`: 40, `cubist model`: 20) have virtually zero competition (difficulty < 10) and can be dominated immediately.

---

## 2. Real Monthly Search Volumes (Google US)

### Core Seed & Category Keywords

| Keyword | Monthly Volume | Difficulty (0–100) | CPC (USD) | SERP Features | Target Placement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `decision tree` | 18,100 | 31 | $1.30 | AI Overview, Video, PAA | Docs Overview / General intro |
| `decision tree classifier` | 590 | 24 | $1.84 | AI Overview, PAA, Video | Python API reference |
| `decision tree regression` | 390 | 8 | $0.00 | AI Overview, Video | Python API reference |
| `target encoding` | 390 | 3 | $0.00 | AI Overview, PAA | Transforms & Encoding guide |
| `tree based models` | 260 | 10 | $0.00 | Featured Snippet, PAA | Conceptual guides / Benchmarks |
| `decision trees in python` | 260 | 5 | $0.00 | AI Overview, PAA | Quickstart tutorial |
| `decision tree python` | 210 | 11 | $1.48 | AI Overview, Video | README title / Docs landing page |
| `cart modeling` | 210 | 9 | $0.00 | AI Overview, Video | Technical deep-dive article |
| `rust machine learning` | 170 | 6 | $1.88 | Related searches | GitHub tags, Crates.io metadata |
| `cart algorithm` | 170 | 8 | $0.00 | AI Overview, Video | Algorithm reference page |
| `cart decision trees` | 140 | 16 | $0.00 | AI Overview, Video | Algorithm reference page |
| `python decision tree` | 110 | 11 | $0.00 | AI Overview | Package description & metadata |
| `cart decision tree` | 110 | 16 | $0.00 | AI Overview, PAA | Algorithm comparison page |
| `cart analysis` | 110 | 5 | $0.00 | AI Overview | Domain modeling / Statistical notes |
| `cart machine learning` | 90 | 6 | $0.00 | AI Overview, Video | Modern vs classic CART comparison |

---

### High-Intent Long-Tail & Niche Keywords

| Keyword | Monthly Volume | Difficulty | Search Intent & Relevance |
| :--- | :--- | :--- | :--- |
| `code decision tree` | 50 | 15 | Implementation lookup; target with code snippets |
| `rule based machine learning` | 40 | 3 | Core Cubist / Rule-tree positioning |
| `c4.5 algorithm` | 40 | 3 | Pure algorithmic search; mltreelib C4.5 port |
| `cart ml` | 40 | 6 | Abbreviated machine learning query |
| `c4.5 decision tree` | 30 | 3 | Explaining gain ratio vs Gini |
| `decision tree code in python` | 30 | 8 | Code-first onboarding |
| `cart algorithm decision tree` | 30 | 5 | Technical explanation |
| `cart statistics` | 30 | 3 | Academic / statistical practitioner query |
| `tabular machine learning` | 20 | Low | Research thesis against tabular foundation models |
| `tabular data machine learning` | 20 | 3 | Tabular benchmark / TabArena positioning |
| `cubist model` | 20 | 1 | Cubist port; high monopoly potential |
| `cart in python` / `cart python` | 20 | 2 | Primary search for users needing CART outside sklearn |
| `decision trees python code` | 20 | 8 | Code example target |
| `python classification tree` | 20 | 7 | Alternative phrase for Scikit-learn switchers |
| `implementation of decision tree in python` | 20 | 4 | Architectural guide |
| `decision tree missing values` | 10 | 1 | Highlights mltreelib first-class missing data policy |
| `cubist algorithm` / `cubist machine learning` | 10 | 1 | High intent for Quinlan's Cubist model |
| `decision tree from scratch python` | 10 | 1 | Rust engine + clear Python architecture |
| `target encoding python` | 10 | 2 | Ingest / preprocessing feature docs |
| `fast decision tree` | 10 | 1 | Rust performance vs Python speed benchmarks |
| `c50 decision tree` | 10 | 1 | C5.0 tree search |

---

## 3. Recommended Keyword Mapping by Surface

### A. Repository & Landing Page (`README.md`, Crates.io, PyPI)
- **Primary Keywords**: `decision tree python`, `rust machine learning`, `fast decision tree`, `cart algorithm`.
- **Meta Description / Tagline**:
  > *"mltreelib: Fast, deterministic decision tree and rule-based machine learning in Rust with thin Python bindings. Implements CART, C4.5, C5.0, Cubist, and first-class missing data handling."*

### B. Documentation Architecture (`docs/`)
1. **Getting Started (`docs/getting-started/`)**:
   - `decision tree python`, `decision trees in python`, `decision tree code in python`.
2. **Algorithm Guides (`docs/algorithms/`)**:
   - `cart in python`, `cart algorithm`, `c4.5 decision tree`, `c50 decision tree`, `cubist model`, `rule based machine learning`.
3. **Data Handling & Preprocessing (`docs/transforms/`)**:
   - `decision tree missing values`, `target encoding python`, `tabular data preprocessing`.
4. **Benchmarks & Performance (`docs/benchmarks/`)**:
   - `fast decision tree`, `tabular machine learning`, `tabular data machine learning`, `tree based models`.

### C. Research & Positioning Notes (`references/research/`)
- Target long-term research citations:
  - *Tabular Machine Learning: Tree Ensembles vs Tabular Foundation Models*.
  - *First-Class Missing Data in Tree Models: A Deterministic Benchmark*.
  - *Recovering Quinlan's Classics: Modernizing C4.5 and Cubist in Rust*.
