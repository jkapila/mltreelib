# Categorical encoders deep-dive

Comprehensive tour of all 13 categorical encoders in `mltreelib` evaluated on real churn data (5,000 rows, 4 categorical columns).

Script: [`examples/encoding_demo.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/encoding_demo.py>) (`uv run --with pandas python examples/encoding_demo.py`).

```
==============================================================================
1. DATASET: churn — target 'CustomerChurned' -> 1.0 = 'Yes'
==============================================================================
rows x cols     : 5000 x 20
categorical feats: ['state', 'area_code', 'international_plan', 'voice_mail_plan']
numeric feats    : ['account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes'] …
base rate       : 0.1414 (707 of 5000)
primary feature : 'state' with 51 levels

==============================================================================
2. UNSUPERVISED ENCODERS (task: both)
==============================================================================
OneHot     : 51 indicator columns; row 0 -> {'3': 1.0, '23': 0.0, '36': 0.0}
Hashing    : 16 buckets; row 0 bucket = ['hash_2']
BaseN      : 6 binary digits for 51 levels
Count      : row 0 level occurs 89 times; unseen -> 0
Sum        : 50 contrast columns; row 0 -> ['1', '0', '0', '0', '0'] …

==============================================================================
3. SUPERVISED ENCODERS (task: both) on state
==============================================================================
TargetEncoder (smoothing=10) churn estimate, lowest levels : 45=0.05792, 11=0.06681, 29=0.07565, 39=0.07719, 0=0.07822
TargetEncoder churn estimate, highest levels: 4=0.2486, 31=0.2411, 47=0.2353, 43=0.2176, 26=0.2056
LeaveOneOut fit_transform (leakage-aware) rows 0..4: ['0.1023', '0.1613', '0.1798', '0.1593', '0.05882']
CatBoost ordered TS      rows 0..4           : ['0.07624', '0.15', '0.1685', '0.1883', '0.06456']
ordered TS values for level '3': 89 distinct per-row values (plain TargetEncoder gives exactly 1 — that difference is the leakage guard)
M-estimate (m=20) row 0: 0.1085 (shrunk harder toward 0.1414)

==============================================================================
4. CLASSIFICATION FAMILY: class-space vectors
==============================================================================
Tfidf('state', row 0, L2-normalized): ['0.9608', '0.2772']
Tfidf missing row -> None
WoE for churn, protective (negative) states: 45=-1.036, 11=-0.8691, 29=-0.7271, 39=-0.7073, 0=-0.6819
WoE for churn, risky (positive) states     : 4=0.7904, 31=0.6705, 47=0.6473, 43=0.5308, 26=0.4675
Embedding (dim=8) churn accuracy on train: 0.859 (base rate 0.141)
Embedding for '3' -> 8-d vector

==============================================================================
5. REGRESSION FAMILY (task: regression) on account_length
==============================================================================
Quantile (median, smoothing=2) by 'area_code', row 0: 99.5
Embedding regression on 'account_length', rows 0..4: ['100.3', '100.3', '100.3', '100.3', '100.3']
```

## Leakage Protection in Supervised Encoders

Directly mapping high-cardinality levels to their target mean leads to target leakage on train data. `mltreelib` provides three built-in mitigations: - **`CatBoostEncoder`**: Ordered target statistics using seeded random row permutations. For level `'3'` (which appears 89 times), `CatBoostEncoder` assigns 89 distinct incremental target statistic values on train, completely guarding against memorization. - **`LeaveOneOutEncoder`**: Subtracts the target of the current row: $(S - y\_i) / (n - 1)$. - **`MEstimatorEncoder`**: Shrinks small-count categories toward the global prior: $(S + m \\cdot p) / (n + m)$.

## Weight of Evidence &amp; Neural Embeddings

- **`WoEEncoder`**: Computes $\\ln(\\text{distribution of non-events} / \\text{distribution of events})$ for credit risk and churn scorecards.
- **`EmbeddingEncoder`**: Trains dense entity embeddings directly in pure Rust via stochastic gradient descent with an SGD optimizer and cosine decay schedule.
