# Uplift Modeling from Scratch: Maximizing ROI with Honest Causal Trees

**Author:** Jitin Kapila &amp; mltreelib Team  
**Category:** Commercial Growth &amp; Decision Optimization

---

## The Response Modeling Trap

Most marketing, churn retention, and discount campaigns rely on standard **response modeling**: - Train a classifier to predict $P(Y=1 \\mid X)$ (e.g. will the customer renew their subscription or click an ad?). - Target the top decile of predicted probabilities with discounts or incentives.

This wastes budget. Standard response models target **Sure Things** (customers who would have renewed anyway) and **Lost Causes** (customers who will never convert, discount or not). Worst of all, they often contact **Sleeping Dogs**—customers who cancel *because* you reminded them of their recurring charge.

---

## True Incremental Uplift (CATE)

Uplift modeling estimates the **Conditional Average Treatment Effect (CATE)**:

$$\\tau(x) = \\mathbb{E}\[Y \\mid X=x, W=1\] - \\mathbb{E}\[Y \\mid X=x, W=0\]$$

Where $W \\in {0, 1}$ represents receiving the intervention (discount, email, outreach).

`mltreelib` implements **Honest Causal Trees** (Athey &amp; Imbens / Wager):

```
import mltreelib as mt

# Causal tree targeting treatment effect divergence
causal_tree = mt.ModelTree(
    criterion="causal_divergence",
    honesty=True,             # Sample splitting to prevent spurious subgroup discovery
    honesty_ratio=0.5,
    min_leaf_samples=50
)

# Fit over observational or A/B test data
causal_tree.fit(
    frame,
    target="conversion_value",
    treatment_col="received_discount"
)

# Predict individual uplift (tau)
uplift_scores = causal_tree.predict_uplift(test_frame)
```

By splitting nodes to maximize the difference in treatment effects rather than mean outcome variance, `mltreelib` isolates the true **Persuadables**—yielding maximum ROI per marketing dollar spent.
