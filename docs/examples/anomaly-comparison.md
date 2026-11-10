# Tree-based anomaly detection suite comparison

Comprehensive benchmark comparing all 9 tree-based anomaly and density models in `mltreelib` on the `credit-g` dataset (1000 observations, 9 mixed attributes).

Script: [`examples/anomaly_comparison.py`](https://github.com/jkapila/mltreelib/blob/main/examples/anomaly_comparison.py)
(`uv run python examples/anomaly_comparison.py`).

## Evaluated models

| Method | Paradigm | Scoring Principle | Time / Space Complexity |
| :--- | :--- | :--- | :--- |
| **Extended Isolation Tree (EIF)** | Oblique Hyperplane Cuts | Path depth under random oblique cuts | $O(N \log N)$ |
| **Density Estimation Tree (DET)** | Integrated Squared Error | Density estimate $\hat{f}(x)$ & negative log-density | $O(N \log N)$ |
| **Half-Space Tree (HST)** | Streaming Mass Estimation | Node mass profile across perturbed half-spaces | $O(\text{depth})$ update |
| **Random Cut Tree (RCT)** | Collusive Displacement | CoDisp with dimension selection $\propto$ bbox | $O(\text{sample\_size})$ |
| **Active Anomaly Tree (AAD)** | Human-in-the-Loop Feedback | Gradient adaptation on confirmed outliers | $O(\text{depth})$ update |
| **Context Anomaly Tree (CAT)** | Sequential Context Model | Context transition surprise $-\log_2 P_{KT}$ | $O(\text{sequence\_len})$ |
| **GritBot** | Subgroup Rule Discrepancy | Contextual deviation from local regression | Rule search |
| **HDBSCAN (GLOSH)** | Density Clustering Hierarchy | Relative lambda death within cluster hierarchy | $O(N^2)$ |
| **URF (Proximity)** | Synthetic Marginal Affinity | Proximity co-occurrence isolation | $O(N^2)$ |

## Inter-model agreement (Spearman rank correlation)

Comparing normalized $[0, 1]$ anomaly scores across all models on `credit-g`:

| Method | EIF | DET | HST | RCT | AAD | CAT | GritBot | HDBSCAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EIF (Oblique)** | 1.00 | 0.42 | 0.23 | 0.10 | -0.00 | 0.09 | 0.21 | 0.37 |
| **DET (Density)** | 0.42 | 1.00 | 0.44 | 0.20 | -0.05 | 0.13 | 0.33 | 0.47 |
| **HST (Mass)** | 0.23 | 0.44 | 1.00 | 0.35 | 0.23 | 0.07 | 0.22 | 0.45 |
| **RCT (CoDisp)** | 0.10 | 0.20 | 0.35 | 1.00 | 0.06 | 0.04 | 0.13 | 0.13 |
| **AAD (Feedback)** | -0.00 | -0.05 | 0.23 | 0.06 | 1.00 | 0.03 | 0.14 | 0.53 |
| **CAT (Surprise)** | 0.09 | 0.13 | 0.07 | 0.04 | 0.03 | 1.00 | 0.06 | 0.11 |
| **GritBot** | 0.21 | 0.33 | 0.22 | 0.13 | 0.14 | 0.06 | 1.00 | 0.37 |
| **HDBSCAN (GLOSH)**| 0.37 | 0.47 | 0.45 | 0.13 | 0.53 | 0.11 | 0.37 | 1.00 |

## Consensus anomalies

The top multi-model consensus anomaly detected across models:

- **Observation #987** (Mean Score: 0.683):
  - *Attributes*: Duration 24 mo, Credit Amount $6,615, Age 75 yrs, Checking `< 0 DM`, Critical account history.
  - *Scores*: EIF=0.18, DET=0.78, HST=0.99, RCT=1.00, AAD=1.00.
  - *Reason*: Advanced age (75) paired with critical credit history and high loan amount relative to savings.

## Architectural selection guide

1. **EIF**: Best for mixed tabular data with diagonal/correlated features where standard isolation trees suffer from axis ghost masking.
2. **DET**: Best for exact density estimation with theoretical ISE optimality.
3. **HST**: Best for streaming data pipelines requiring strict $O(\text{depth})$ update budgets.
4. **RCT**: Best for streaming datasets with dense clusters of anomalies (collusion resistance).
5. **AAD**: Best for security or audit desks where analysts review flagged anomalies in real time.
6. **CAT**: Best for event streams and user clickstream / session transitions.
7. **GritBot**: Best for conditional rule violations in domain data.
8. **HDBSCAN**: Best when anomalies should be defined relative to cluster densities.
