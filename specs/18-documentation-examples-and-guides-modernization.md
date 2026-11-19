# Spec 18: Documentation, Examples, and Guides Modernization

## Problem Statement

Users and researchers adopting `mltreelib` encounter a severe disconnect between the library's high-performance Rust core and the quality, coherence, and grounding of its documentation, examples, and guides:

1. **Scattered and Fragmented Examples**: The example library has sprawled into 24 fragmented pages and over 40 scripts. Users must jump between multiple disjointed pages to understand basic workflows because tree controls, hyperparameter knobs, 0D leaf stats, and model inspection are artificially split across separate stubs.
2. **Ungrounded Toy and Synthetic Formulations**: The Quickstart and introductory tutorials rely on arbitrary mathematical toy loops and synthetic formulas with text placeholders (`"DOT output omitted"`). Users never see how the model performs on real tabular distributions or why piecewise linear leaves outperform standard CART step functions.
3. **Absence of Domain Rationale and Quantitative Interpretation**: Examples dump raw numbers and text trees without providing domain rationale. Users are left to wonder why a method was built, when to choose it over gradient boosted trees, and how to interpret leaf parameters, breakdown thresholds, or quantile safety corridors.
4. **Lack of High-Impact Visual Assets and Decision Boundaries**: Key capabilities lack publication-grade visualizations. There are no scikit-learn style 2D decision boundary maps, piecewise linear response surfaces, survival Kaplan-Meier curves, or calibration panels. The existing visual assets are inconsistent in style and palette.
5. **Friction and External Download Barriers in Guides**: Certain guides force users to execute manual shell commands downloading external archives, failing in offline, hermetic, or sandboxed environments where legacy decompression tools are unavailable.
6. **Missing Architectural and Decision Flowcharts**: Complex mechanisms like surrogate split routing, two-control tree architecture, honest sample splitting, and production serving contracts are explained in dense text walls without clear structural diagrams.

## Solution

Modernize the entire documentation, guide, and example suite by enforcing the **Gold Standard** established in the distributional and hierarchical leaf modules (`distributional-leaves.md`, `stump_distributional_leaves.py`, and `stump_leaf_retrain.py`):

1. **Ground Every Workflow in Real Cited Benchmark Data**: Eliminate toy formulas from entry points. Wire tutorials to standard benchmarks (Concrete Compressive Strength, Diamonds, Scania Truck APS Failure, Give Me Some Credit, Wine Quality, FLCHAIN, and Hillstrom).
2. **Rationalize and Consolidate 24 Scattered Examples into 8 Cohesive Suites**:
   - *Suite 1: Core Tree Architecture, Two-Control Structure & Growth Knobs*
   - *Suite 2: Structural Constraints & Honest Inference*
   - *Suite 3: Native Missing Data & Surrogate Routing*
   - *Suite 4: Distributional & Censored Leaf Families*
   - *Suite 5: Dynamic Leaf Retraining & Hierarchical Adaptation*
   - *Suite 6: Statistical, Metric & Oblique Splitting*
   - *Suite 7: Time Series Feature Engineering & Additive Forecasting*
   - *Suite 8: Unsupervised Trees: Clustering & Anomaly Triage*
3. **Elevate Quickstart & Landing Pages**: Provide an immediate visual hero comparison showing continuous piecewise linear regression outperforming CART staircase step functions, alongside an inference pipeline diagram.
4. **Upgrade Modeling Pathways with 2D Visual Boundaries and Response Surfaces**: Generate scikit-learn style 2D decision contours, regression surfaces, and multi-cohort survival curves. Eliminate external download scripts in favor of built-in offline datasets.
5. **Embed High-Resolution Empirical Cards and Mermaid Flowcharts Across Blog Posts**: Supplement narrative deep dives with quantitative empirical cards and architectural DAGs.
6. **Strict Visual Palette and Zero-Dependency Guarantee**: Lock all visual assets to the project palette (Teal `#00897b`, Deep Orange `#f4511e`, Cerulean Blue `#0288d1`). Guarantee that the core library remains zero-dependency and imports in under 5 milliseconds.

## User Stories

1. As a new user landing on the Quickstart, I want to see a real dataset and a side-by-side plot comparing a model tree against a standard decision tree, so that I immediately understand why piecewise linear leaves beat step functions.
2. As a machine learning practitioner, I want the Quickstart to include a visual flowchart of the inference pipeline, so that I understand how input feature vectors route to local linear regressions.
3. As a quantitative researcher, I want the example catalog consolidated into 8 domain-grounded suites, so that I can explore cohesive, comprehensive workflows without navigating through dozens of fragmented 1-page stubs.
4. As an ML engineer, I want the Core Tree Architecture suite to clearly explain the two-control decoupling between tree topology and leaf solvers, so that I can configure tree growth and leaf estimation independently.
5. As an applied econometrician, I want the Structural Constraints suite to demonstrate monotone split constraints and local leaf slope restrictions on real financial data, so that I can guarantee economic theory invariants in production.
6. As a causal inference specialist, I want the Honest Inference suite to provide a Mermaid diagram and code demonstrating sample splitting into structure and estimation partitions, so that I can estimate unbiased treatment effects without data snooping.
7. As an industrial IoT engineer, I want the Native Missing Data suite to evaluate Scania truck sensor failure telemetry without imputation, so that I can see how learned missing directions and surrogate splits route intermittent hardware dropouts.
8. As a risk validator, I want the Credit Classification guide to show a 2D probability contour map and adverse action explanation paths, so that I can satisfy FCRA and ECOA regulatory transparency audits.
9. As an actuarial data scientist, I want the Distributional Leaves suite to compare Tweedie, Tobit, and Gamma models against real zero-inflated claim data, so that I can accurately estimate pure premiums and handle censoring without OLS attenuation.
10. As a production deployment lead, I want the Serving Contract guide to provide a Mermaid sequence diagram detailing schema validation, sentinel normalization, and threshold application, so that I can build reliable inference services.
11. As a biostatistician, I want the Survival Tree guide to display Kaplan-Meier curves and Nelson-Aalen cumulative hazard corridors across distinct tree leaves, so that I can analyze patient time-to-event outcomes with statistical rigor.
12. As a marketing analytics lead, I want the Uplift Modeling guide to include Qini curves and cumulative gain metrics comparing causal tree targeting against random treatment, so that I can optimize marketing intervention budgets.
13. As an exploratory data analyst, I want the Unsupervised Trees suite to visualize Predictive Clustering Trees using 2D PCA projections with interpretable rectangular bounds, so that I can understand how multi-target trees cluster feature space.
14. As a fraud detection analyst, I want the Anomaly Detection guide to run on offline benchmark datasets without manual curl and uncompress steps, so that I can run reproducible evaluations in air-gapped environments.
15. As a security and platform engineer, I want the Anomaly guide to compare Extended Isolation Forests against standard Isolation Forests using 2D anomaly density contours, so that I can see how oblique cuts eliminate axis-aligned ghost anomalies.
16. As a supply chain planner, I want the Time Series suite to demonstrate entity-grouped calendar day lags and additive trend decomposition in pure Rust, so that I can forecast store demand without data leakage.
17. As an embedded systems developer, I want importing the root package to load zero optional dependencies, so that scoring startup latency and memory overhead remain minimal.
18. As a technical reader, I want consistent visual aesthetics across all documentation charts using the project's teal, deep orange, and cerulean palette, so that the documentation feels cohesive and polished.
19. As a library maintainer, I want the documentation build to complete with zero warnings, so that no broken links or orphaned pages reach publication.
20. As an open-source contributor, I want every consolidated suite to include empirical interpretation and clear rationale, so that extending the library follows an established, high-quality standard.

## Implementation Decisions

### Architectural Decisions
- **Consolidated Example Taxonomy**: The 24 individual example pages are consolidated into 8 comprehensive suites. Existing URLs in navigation and documentation are repointed to maintain link integrity while eliminating fragmented stubs.
- **Visual Design Standard**: All generated plots must adhere to the project palette:
  - Primary Brand / Fitted Curve: Teal (`#00897b`)
  - Accent / CART Baseline / Error: Deep Orange (`#f4511e`)
  - Secondary Reference / Alternative: Cerulean Blue (`#0288d1`)
  - Neutral Grid / Confidence Shading: Blue Grey (`#cfd8dc` / `#eceff1`)
  - Structural Borders & Text: Dark Slate (`#263238`)
  - Multi-panel grids rendered at 300 DPI with clear annotations and legends.
- **Zero-Dependency Core Isolation**:
  - The core package must have zero required runtime dependencies.
  - Optional extras (`dataset`, `plotting`, `tuning`) must remain strictly isolated.
  - Root package initialization utilizes lazy loading so that importing base estimators never pulls heavy packages into memory.
  - All example and benchmark scripts must guard optional plotting or data imports with helpful error messages directing users to install the appropriate extras.
- **Self-Contained Offline Benchmark Ingestion**:
  - All guides and examples must consume data via dataset loaders or local benchmark files.
  - External bash scripts utilizing manual curl or decompression tools are completely removed.

### Interface & Schema Standards for Examples
- Every consolidated suite must be structured with:
  1. **Domain Context & Rationale**: The business or scientific problem, failure modes of standard algorithms, and why tree structures are suitable.
  2. **Architectural / Flow Diagram**: A Mermaid diagram illustrating the data lifecycle, split search, or decision logic.
  3. **Verified Benchmark Implementation**: Fully runnable code using real benchmark datasets.
  4. **Empirical Output & Quantitative Interpretation**: A clear breakdown of holdout metrics, error reductions, parameter interpretations, and edge-case behavior.
  5. **Scikit-Learn Style Multi-Panel Visual Card**: A high-resolution figure illustrating the empirical results.

## Testing Decisions

- **Documentation Build Integrity**: Every change must pass the documentation generator build with zero warnings (`uv run --no-sync zensical build`).
- **Behavioral Test Coverage**: Python integration tests and Rust unit tests must verify external behavior and API contracts without pinning incidental internal implementation details.
- **Zero-Import Leakage Verification**: Unit tests must confirm that importing the root module does not import heavy optional dependencies (`pyarrow`, `matplotlib`, `pandas`, `optuna`).
- **Script Executability**: All example scripts must be executable from the repository root via `uv run python examples/<script>.py` and terminate with return code 0.

## Out of Scope

- Introducing new third-party machine learning dependencies into the core Rust or Python engines.
- Modifying the underlying Rust tree splitting algorithms or PyO3 FFI layout.
- Rewriting the legacy reference oracle codebase in `legacy/`.

## Further Notes

- All changes must remain unstaged and uncommitted in the working tree until explicit user confirmation is received.
- Existing tests (`uv run --no-sync pytest tests/ -q`) and core Rust tests (`cargo test --quiet`) must remain 100% green throughout the modernization process.
