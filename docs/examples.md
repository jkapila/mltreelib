---
title: Examples
description: Runnable example scripts for mltreelib, one per topic.
---

# Examples

Short, runnable scripts. Each one finishes quickly and prints what to look for. For explanation, use the [Guides](guides/index.md) (do a task) or the [Concepts](concepts/core-tree-architecture.md) (how it works).

Run any script from the repository root:

```bash
uv run python examples/quickstart.py
```

| Script | What it shows |
| :--- | :--- |
| [`anomaly_comparison.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/anomaly_comparison.py) | Comprehensive Comparison of Anomaly & Outlier Detection in mltreelib |
| [`anomaly_shuttle_evaluation.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/anomaly_shuttle_evaluation.py) | UCI Statlog Shuttle anomaly proxy, using the provided train/test split |
| [`anomaly_suite_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/anomaly_suite_showcase.py) | Comprehensive Showcase of mltreelib Base Anomaly Trees on Real Benchmark Data |
| [`anomaly_workflow.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/anomaly_workflow.py) | Synthetic anomaly review workflow through AnomalyTree |
| [`causal_uplift.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/causal_uplift.py) | Causal Uplift Modeling Case Study: Honest Causal Trees vs Adaptive Trees and T-Learners |
| [`churn_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/churn_demo.py) | End-to-end churn walkthrough on the bundled Bank Customer Churn dataset |
| [`clustering_comparison.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/clustering_comparison.py) | Comprehensive Comparison of Tree-Based Clustering Methods in mltreelib |
| [`clustering_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/clustering_showcase.py) | Comprehensive Showcase of mltreelib Tree-Based Clustering & TrieKNN |
| [`clustering_workflow.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/clustering_workflow.py) | A reproducible clustering workflow using the unified ClusterTree API |
| [`components_and_rotations.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/components_and_rotations.py) | Component Analysis, R-Aligned Factor Rotations, and Matrix Factorization Showcase |
| [`credit_scoring.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/credit_scoring.py) | Credit Scoring Case Study: Glass-Box Decision Trees under Asymmetric Misclassification Costs |
| [`cubist_comparison.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/cubist_comparison.py) | ModelTree vs Cubist (RuleQuest demo, Release 2.10) on the boston data |
| [`data_layer_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/data_layer_demo.py) | mltreelib data & feature layer demo |
| [`data_metrics_pathway.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/data_metrics_pathway.py) | Native mixed inputs and held-out evaluation with visual diagnostics |
| [`distributional_split_criteria.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/distributional_split_criteria.py) | Demonstrate the 5 new splitting criteria on standard benchmark datasets |
| [`encoding_demo.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/encoding_demo.py) | mltreelib categorical-encoding demo |
| [`explainer_and_curves_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/explainer_and_curves_showcase.py) | Model explanation & evaluation surface showcase on real data |
| [`gritbot_anomaly_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/gritbot_anomaly_showcase.py) | GritBot contextual anomaly showcase on Bank_Customer_Churn dataset |
| [`hierarchical_leaf_retrain.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hierarchical_leaf_retrain.py) | Hierarchical Macro-Tree to Micro-Leaf Retraining & Entity Clustering |
| [`honesty_weights.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/honesty_weights.py) | Honest tree estimation and sample weights in ModelTree |
| [`hpo_global_defaults.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hpo_global_defaults.py) | Global HPO Bake-Off — Find optimal global default hyperparameters for ModelTree |
| [`hpo_optuna.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hpo_optuna.py) | Hyperparameter search over the FULL ModelTree parameter surface with Optuna |
| [`hpo_optuna_classification.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hpo_optuna_classification.py) | Full-Surface Hyperparameter Optimization for Binary & Multiclass Classification |
| [`hpo_optuna_defaults.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hpo_optuna_defaults.py) | Comprehensive HPO Campaign to evaluate and establish empirical global defaults for ModelTree |
| [`hpo_shapes.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/hpo_shapes.py) | HPO by data shape — one parameter at a time, raw data, real TabArena |
| [`inspection.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/inspection.py) | Inspection: metrics reports and every structural view of a fitted tree |
| [`knn_leaf_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/knn_leaf_showcase.py) | Non-parametric KNN and Double-Weighted KNN (DW-KNN) leaf models showcase |
| [`knob_tour.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/knob_tour.py) | Key-knob tour — every growth knob with real, deterministic output |
| [`leaf_family.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/leaf_family.py) | Comprehensive Tour of Distributional Leaf Families in mltreelib |
| [`metrics_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/metrics_showcase.py) | Every Metrics surface and every plot, in one runnable showcase |
| [`missing_routing.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/missing_routing.py) | Missing data routing, learned split directions, and surrogate splits |
| [`model_building.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/model_building.py) | Model building & predictions: every parameter and nuance the tree handles |
| [`modeling_pathways.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/modeling_pathways.py) | Reproducible, synthetic examples for the practical modeling guides |
| [`optimize_generic.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/optimize_generic.py) | Tune ModelTree hyperparameters on real data with the generic optimizer |
| [`pca_rotations_educational.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/pca_rotations_educational.py) | Educational Deep-Dive: Factor Rotations in Principal Component Analysis |
| [`plot_sklearn_comparisons.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/plot_sklearn_comparisons.py) | Generate scikit-learn style benchmark comparison figures using matplotlib |
| [`practical_stump_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/practical_stump_showcase.py) | Explainable Decision Stumps & Frozen-Topology Retraining on Real Data |
| [`predictive_maintenance.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/predictive_maintenance.py) | Predictive Maintenance Case Study: Zero-Imputation Decision Trees on Scania Truck Telemetry |
| [`real_data_workflow.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_data_workflow.py) | Real-data model and review example using TabArena Bank Customer Churn |
| [`real_survival_uplift.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_survival_uplift.py) | Evaluate survival and uplift on real datasets with held-out observations |
| [`real_tree_studies.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_tree_studies.py) | Reproduce the held-out case studies in the practical guides and blog |
| [`real_unsupervised_workflow.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_unsupervised_workflow.py) | Benchmark unsupervised facades on fixed TabArena-v0.1 partitions |
| [`regression_real_and_synthetic.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/regression_real_and_synthetic.py) | Compare regression configurations on synthetic dirty data, then fit real data |
| [`rules_summary_knn_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/rules_summary_knn_showcase.py) | Showcase of C5.0 Rule Extraction, R/statsmodels Leaf Summaries, and Cubist k-NN Adjustment |
| [`solvers_overview.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/solvers_overview.py) | Unified Solver Architecture Example in mltreelib |
| [`stump_distributional_leaves.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/stump_distributional_leaves.py) | Showcase of Distributional & Survival Leaves in mltreelib |
| [`survival_causal_survival_honest.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/survival_causal_survival_honest.py) | Causal Survival Trees with Honest Sample Splitting (Spec 19 Research Track) |
| [`survival_competing_risks_pbc.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/survival_competing_risks_pbc.py) | Survival and Competing Risks Modeling: Aalen-Johansen CIF & Gray's Test |
| [`survival_flchain_mortality.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/survival_flchain_mortality.py) | FLCHAIN Mortality Follow-up: Clinical Survival Modeling with Native Missing Values |
| [`survival_uplift_evaluation.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/survival_uplift_evaluation.py) | Held-out synthetic survival and randomized uplift evaluations |
| [`timeseries_forecasting.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/timeseries_forecasting.py) | End-to-end time series feature engineering and forecasting with ModelTree |
| [`two_controls_and_0d_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/two_controls_and_0d_showcase.py) | Two-control architecture, 0D leaf formulations, dual monotonicity, weights, and interaction constraints |
| [`two_stage_survival.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/two_stage_survival.py) | Two-Stage Hierarchical Survival Trees: Segregate First, Model Later |
| [`unified_task_facades.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/unified_task_facades.py) | Unified Task Facades Showcase: AnomalyModel, ClusterModel, RuleModel, ComponentModel |
| [`visualization.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/visualization.py) | Visualization for real work: graphviz tree diagrams + diagnostic plots |
| [`quickstart.py`](https://github.com/jkapila/mltreelib/blob/main/examples/quickstart.py) | mltreelib Quickstart: Fit a ModelTree on real benchmark data in seconds |
