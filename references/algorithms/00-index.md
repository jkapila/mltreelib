# Algorithm pseudocode set — master index

**Date:** 2026-10-28. Every algorithm in the study set, with core pseudocode.
Sources are local: `references/sources/extracted/` (unpacked tarballs/zips),
`references/repos/` (cloned projects), or the linked pages. Learn the family
files; this index is the map.

| # | Algorithm | Family | Source | File |
|---|---|---|---|---|
| 1 | CART | single tree | legacy oracle + literature | 01-single-trees.md |
| 2 | C4.5 | single tree | `extracted/see5-public/` (sibling) + literature | 01-single-trees.md |
| 3 | C5.0 | single tree/rules | `extracted/see5-public/`, `references/docs/see5-unix.md` | 01-single-trees.md |
| 4 | OneR | single rule | mlxtend.classifier | 01-single-trees.md |
| 5 | CHAID | single tree | literature (Kass 1980) | 01-single-trees.md |
| 6 | ctree (conditional inference) | single tree | `extracted/party_1.3-23/` | 01-single-trees.md |
| 7 | MOB (model-based partitioning) | tree + model | `extracted/partykit_1.3-1/`, `repos/modeltrees/` | 01-single-trees.md |
| 8 | MARS / `earth` | basis expansion | `extracted/earth_5.3-6/` | 01-single-trees.md |
| 9 | semtree | tree + SEM leaves | `extracted/semtree_0.9.23/` | 01-single-trees.md |
| 10 | glmertree | tree + GLMM leaves | `extracted/glmertree_0.2-7/` | 01-single-trees.md |
| 11 | M5 / M5P model tree | model tree | literature (Quinlan/Kramer) | 02-model-trees.md |
| 12 | Cubist | model tree | `extracted/cubist-public/`, `references/docs/cubist-unix.md` | 02-model-trees.md |
| 13 | Linear Tree | model tree | `repos/linear-tree/` | 02-model-trees.md |
| 14 | TRUST | model tree | `repos/trust-free/` | 02-model-trees.md |
| 15 | LightGBM `linear_tree` | model tree in boosting | LightGBM docs | 02-model-trees.md |
| 16 | AdaBoost (SAMME / SAMME.R) | boosting | literature + mlxtend lineage | 03-boosting.md |
| 17 | Gradient Boosting (GBRT) | boosting | Friedman 2001 | 03-boosting.md |
| 18 | XGBoost | boosting | literature | 03-boosting.md |
| 19 | LightGBM | boosting | literature | 03-boosting.md |
| 20 | CatBoost | boosting | CatBoost docs/paper | 03-boosting.md |
| 21 | mboost (glmboost/gamboost/blackboost) | boosting | `extracted/mboost_2.9-14/` | 03-boosting.md |
| 22 | NGBoost | boosting | `repos/ngboost/` | 03-boosting.md |
| 23 | Quantile GBT (pinball) | boosting | literature + crates | 03-boosting.md |
| 24 | LinearBoost | boosting | `repos/linearboost-classifier/` | 03-boosting.md |
| 25 | SEFRBoost | boosting | `repos/linearboost-classifier/` | 03-boosting.md |
| 26 | Bagging | ensemble | literature + mlxtend.ensemble | 04-ensembles.md |
| 27 | Random Forest | ensemble | `extracted/randomForest_4.7-1.2/` | 04-ensembles.md |
| 28 | Extra Trees | ensemble | literature (Geurts 2006) | 04-ensembles.md |
| 29 | wsrf (weighted subspace RF) | ensemble | `extracted/wsrf_1.7.32/` | 04-ensembles.md |
| 30 | RLT (reinforcement learning trees) | ensemble | `extracted/RLT_3.2.6/` | 04-ensembles.md |
| 31 | grf / Local Linear Forests | ensemble | grf-labs (R) | 04-ensembles.md |
| 32 | Linear Forest | ensemble | `repos/linear-tree/` | 04-ensembles.md |
| 33 | Voting / Blending / Stacking / StackingCV | meta-ensemble | mlxtend | 04-ensembles.md |
| 34 | BART (+ BayesTree, bartMachine) | Bayesian | `extracted/BART_2.9.10/`, `BayesTree_0.3-1.5/`, `bartMachine_1.4.2/` | 05-bayesian-quantile.md |
| 35 | Quantile Regression Forests | quantile | `extracted/quantregForest_1.4-0/` | 05-bayesian-quantile.md |
| 36 | EBM (GA2M) | interpretable | interpret.ml | 06-interpretable.md |
| 37 | GritBot | interpretable rules | `extracted/GritBot/`, `references/docs/gritbot-unix.md` | 06-interpretable.md |
| 38 | Adaline / Perceptron / Logistic / Softmax / MLP | mlxtend classics | mlxtend.classifier | 07-mlxtend.md |
| 39 | mlxtend LinearRegression (+ Stacking regressors) | mlxtend classics | mlxtend.regressor | 07-mlxtend.md |
| 40 | Rust implementations: forust, perpetual, gbdt, bosk, smartcore, linfa, miniboosts, sooboost, biosphere, sklears-tree, treeboost, affinitree, gbt-quantile, rpart | Rust | crates + `repos/` | 08-rust-implementations.md |
| 41 | TabArena field: ChimeraBoost, CTBoost, xRFM, PrismBoost, APLR (+ PerpetualBooster, Elo snapshot) | benchmark field | `data/tabarena-leaderboard.csv` + repos | 09-tabarena-field.md |
| 42 | CTW, Bayesian Context Trees, Parsimonious Context Trees; applied: news rec (Garcin, RecSys 2013) | context trees | Willems 1995 / Kontoyiannis / Eggeling / Garcin | 10-context-trees-tries.md |
| 43 | Trie family (trie, radix, suffix tree, TST; TrieKNN, context-tree-as-trie) | data structure | `repos/` + literature | 10-context-trees-tries.md |
| 44 | TDM, TDR (deep tree retrieval); CHAMELEON session hybrid | retrieval trees | Zhu KDD'18 / Liu'24 / Moreira'19 | 11-sequential-rec-trees.md |
| 45 | Dynamic time-decay CF, TDTL (trust-region), forgetting-curve rec | temporal decay | Chen'20 / Heryawan'25 / Jin CIKM'23 | 11-sequential-rec-trees.md |

Legend for the entry template: **Source · Idea · Pseudocode · Knobs · Note**.
Pseudocode is the algorithm's core loop in plain steps — faithful in
structure, simplified where a paper's proof machinery does not change what a
reader must understand.
