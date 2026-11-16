# Uplift Modeling from Scratch: Maximizing ROI with Honest Causal Trees

**Author:** Jitin Kapila & mltreelib Team  
**Category:** Commercial Growth & Decision Optimization  

---

## The Response Modeling Trap

Most marketing, churn retention, and discount campaigns rely on standard **response modeling**:
- Train a classifier to predict $P(Y=1 \mid X)$ (e.g. will the customer renew their subscription or click an ad?).
- Target the top decile of predicted probabilities with discounts or incentives.

This wastes budget. Standard response models target **Sure Things** (customers who would have renewed anyway) and **Lost Causes** (customers who will never convert, discount or not). Worst of all, they often contact **Sleeping Dogs**—customers who cancel *because* you reminded them of their recurring charge.

---

## True Incremental Uplift (CATE)

Uplift modeling estimates the **Conditional Average Treatment Effect (CATE)**:

$$\tau(x) = \mathbb{E}[Y \mid X=x, W=1] - \mathbb{E}[Y \mid X=x, W=0]$$

Where $W \in \{0, 1\}$ represents receiving the intervention (discount, email, outreach).

`mltreelib` implements **Honest Causal Trees** (Athey & Imbens / Wager):

```python
from mltreelib import ModelTree

# Causal tree targeting treatment effect divergence
causal_tree = ModelTree(
    gain="causal",
    max_depth=3,
    min_rows=30,
    prune="none",
    honesty=True,             # Sample splitting to prevent spurious subgroup discovery
    honesty_fraction=0.5,
    seed=42,
)

# Fit over tuple target: (outcomes, treatments)
causal_tree.fit(X_train, (y_train, w_train))

# Predict individual uplift (tau / CATE)
uplift_scores = causal_tree.predict(X_test, kind="uplift")
```

---

## Data Analysis & Persona Structure

In our customer discount benchmark ($N=1,000$ campaign members), customers fall into four distinct behavioral regimes:

| Persona | Feature Profile | Control Rate ($W=0$) | Treated Rate ($W=1$) | True CATE ($\tau$) | Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Persuadables** | $X_0 > 0.0$ (High loyalty / tenure) | 15% | 75% | **+60%** | **Target with Discount** |
| **Sleeping Dogs** | $X_0 \le 0.0, X_1 > 0.0$ (High renewal risk) | 55% | 10% | **-45%** | **Do Not Disturb** |
| **Sure Things** | $X_0 \le 0.0, X_1 \le 0.0$ | 25% | 25% | **0%** | **Save Discount Budget** |
| **Lost Causes** | $X_0 \le 0.0, X_1 \le 0.0$ | 25% | 25% | **0%** | **Do Not Target** |

---

## Fitted Causal Tree Structure

Below is the trained causal tree diagram rendered via `causal_tree.to_dot()` and Graphviz:

![Honest Causal Tree Structure](../assets/blog_causal_tree.svg)

Notice how the tree immediately splits on feature $X_0 \approx 0.0$. Customers with $X_0 > 0.0$ are grouped into the left leaves where $\hat{\tau} \approx +0.60$. Customers with $X_0 \le 0.0$ branch right, where the tree isolates the Sleeping Dogs on $X_1 > 0.0$ with $\hat{\tau} \approx -0.45$.

---

## Model Evaluation & Uplift Metrics

Targeting based on honest uplift scores completely reverses campaign economics:

![Uplift Metrics and Qini Curve](../assets/blog_causal_metrics.png)

1. **Cumulative Uplift (Qini Curve)**: Targeting the top 40% of the customer base ordered by predicted CATE captures all incremental conversions. Standard response modeling would mistakenly target the high-baseline Sleeping Dogs, driving negative net ROI.
2. **Quintile Uplift Profile**: The top quintile captures over $+55\%$ empirical treatment uplift, while the bottom quintile exhibits $-43\%$ negative uplift—validating that the model separates customers who convert because of outreach from those who churn because of it.
