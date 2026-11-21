# Core ML Leaf-Family Gap Analysis for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Major classical and modern machine-learning families not yet represented as terminal leaves, assessed through the package's divide-and-conquer architecture.  
**Decision informed:** Which additional model families deserve future specifications after association mining, discriminant analysis, and Bayesian classifiers.  
**Confidence:** High for package fit. Medium for newer methods and exact implementation effort.

## 1. Executive Summary

`mltreelib` currently supports a strong but incomplete leaf portfolio:

- Linear and generalized linear models.
- Mean, median, quantile, and Huber location leaves.
- MARS spline leaves.
- Binary probability, odds, and log-odds leaves.
- Multinomial softmax.
- KNN leaves.
- Survival, AFT, cure, and spline survival leaves.
- Heteroscedastic and Tobit-style distributional leaves.
- Planned or recently added causal/uplift, copula, and other specialized families.

The largest gaps in core ML are not exotic algorithms. They are families that are:

1. Interpretable.
2. Naturally local.
3. Useful on moderate-size leaf datasets.
4. Complementary to existing linear and nonparametric leaves.

### Recommended additions after discriminant analysis and Bayesian leaves

1. **Ordinal cumulative logit/probit leaf.** Highest value and lowest implementation cost.
2. **Generalized additive model leaf.** High value; partially covered by MARS.
3. **Mixture and latent-class leaf.** High potential for mixed populations, reusing EM.
4. **Prototype or Gower k-medoids leaf.** Useful for mixed-type local models.
5. **Kernel or SVM leaf.** Medium value, poor scaling fit.
6. **Gaussian-process leaf.** Valuable uncertainty, poor per-leaf scaling fit.
7. **Matrix, tensor, and structured-response leaves.** High research value, poor current package fit.

Association rule mining should not be a leaf. It belongs in a separate rules module.

## 2. Ordinal Regression Leaves

Ordinal targets carry order: customer tier, disease severity, satisfaction score, income band.

### 2.1 Cumulative link model

For thresholds \(k=1,\dots,K-1\):

\[
\operatorname{logit}P(Y\le k\mid x)=\theta_k-x^\top\beta
\]

A probit variant replaces the logit with a normal CDF.

Strengths:

- Uses ordering information.
- Predicts a full class probability vector.
- Coefficients remain interpretable.
- Fits through existing IRLS or L-BFGS infrastructure.
- Natural extension of softmax and binomial leaves.

Weaknesses:

- Proportional odds assumption can be restrictive.
- Threshold parameters and coefficients need coordinated fitting.
- Small leaves can have missing thresholds.

Source: McCullagh, “Regression for Ordered Categorical Responses,” *JRSS-B* 42(1), 1980.

### Package decision

**Tier 1.** Implement cumulative logit first. Add probit after shared link infrastructure proves stable. Optional non-proportional odds extension is research-grade.

## 3. Generalized Additive Model Leaves

GAMs fit smooth univariate functions:

\[
g(E[Y\mid x])=\beta_0+\sum_j f_j(x_j)
\]

They are interpretable through shape functions and capture nonlinear effects without dense interactions.

MARS already provides piecewise-linear basis expansion and interaction support. A GAM leaf should therefore be a refinement, not a duplicate:

- Natural cubic or cyclic splines per feature.
- Backfitting for numeric predictors.
- Residual or shrinkage smoothing per term.
- Optional pairwise interactions controlled explicitly.

Source: Hastie and Tibshirani, *Generalized Additive Models*.

### Package decision

**Tier 1, but reuse MARS plumbing.** Expose a GAM leaf only if it offers smoother, lower-variance shape functions than MARS. Otherwise improve MARS and avoid another leaf name.

## 4. Mixture and Latent-Class Leaves

A leaf may contain several latent regimes. A mixture model fits:

\[
p(y\mid x)=\sum_{m=1}^M \pi_m(x)p_m(y\mid x)
\]

or, for classification, class-conditional Gaussian mixtures.

Strengths:

- Captures heterogeneous leaf populations.
- Gives soft membership and component probabilities.
- Naturally supports anomaly scores through mixture likelihood.
- Useful for grouped, mixed, and latent-structure data.

Weakness:

- EM can converge to local optima.
- Small leaves provide too little data for many components.
- Component labels require deterministic ordering for reproducibility.

Current package already has mixture-cure survival logic. That EM machinery should be generalized rather than duplicated.

### Package decision

**Tier 1 research implementation.** Start with one-dimensional mixtures or low-dimensional Gaussian mixtures. Add latent-class categorical models later.

## 5. Prototype and Gower Leaves

Prototype classifiers assign a row to the nearest representative or medoid.

- Euclidean or Mahalanobis prototypes overlap KNN.
- Gower prototypes support mixed numeric and categorical data.
- Learned prototypes can become sparse or rule-like.

Strength:

- Very cheap prediction.
- Interpretable representative rows.
- Natural mixed-type support using Gower distance.

Weakness:

- Medoid fitting on large data is expensive.
- Overlaps KNN and planned Gower clustering.

### Package decision

**Tier 2.** Add a Gower prototype leaf only when mixed-type classification evidence justifies it. Keep clustering and leaf code on one shared distance layer.

## 6. Support Vector and Kernel Leaves

SVM leaves offer strong local classification and regression, especially in moderate dimensions.

Strengths:

- Maximum-margin behavior.
- Kernel flexibility.
- Well-understood regularization.

Weaknesses:

- Kernel matrix is \(O(n^2)\) memory per leaf.
- Sequential SMO optimization is difficult.
- Probability calibration needs extra work.
- Repeated fits across tree nodes are expensive.

### Better placement: oblique splits

Kernel or margin-based models may be more useful as **split rules** than leaves. A tree split can use an oblique hyperplane, kernelized criterion, or SVM-derived separator, while leaves remain cheap probabilistic models.

### Package decision

- **Leaf:** defer.
- **Split-side:** research track.
- **Linear SVM leaf:** possible but redundant with regularized linear and discriminant leaves.

## 7. Gaussian Process Leaves

GP leaves offer smooth nonlinear regression and calibrated uncertainty.

Strengths:

- Principled posterior mean and variance.
- Works naturally with categorical kernels and structured inputs.
- Small leaf datasets can be tractable.

Weaknesses:

- Exact GP is \(O(n^3)\) time and \(O(n^2)\) memory.
- Kernel design and hyperparameter tuning add complexity.
- Repeated fits inside trees are costly.

### Package decision

**Tier 3.** Consider sparse GP or inducing-point approximation only after a concrete need such as heteroscedastic local uncertainty. Existing Gaussian NLL leaf partially covers simpler variance needs.

## 8. Matrix, Tensor, and Structured Leaves

Matrix-variate, tensor, functional, panel, and structured-response models preserve input geometry that flattening destroys.

Strengths:

- Can model rich structure unavailable to flat leaves.
- Aligns with `mltreelib`'s research direction around grouped and hierarchical data.

Weakness:

- Current frame and leaf contracts are row-by-feature oriented.
- Each family needs distinct input, fit, prediction, and serialization semantics.

### Package decision

**Research track, not general leaf.** Structured responses are better handled by dedicated model paths or new input abstractions. The panel, VAR/VECM, and matrix-functional research notes already document candidates.

## 9. Ranking

| Rank | Family | Package value | Implementation effort | Leaf fit | Decision |
|---:|---|---:|---:|---:|---|
| 1 | Ordinal cumulative logit/probit | Very high | Low–medium | Excellent | Add next |
| 2 | Discriminant analysis | Very high | Medium | Excellent | Add next |
| 3 | Categorical/Gaussian/Mixed NB | Very high | Low–medium | Excellent | Add next |
| 4 | GAM leaf | High | Medium | Excellent | Add after MARS refinement |
| 5 | Mixture/latent-class leaf | High | Medium–high | Good | Generalize EM |
| 6 | Prototype/Gower leaf | Medium | Low–medium | Good | Add with evidence |
| 7 | AODE/TAN/KDB leaf | Medium–high | Medium–high | Good | After basic NB |
| 8 | Kernel/SVM leaf | Medium | High | Poor | Prefer split-side |
| 9 | GP leaf | Medium | High | Poor | Defer |
| 10 | Matrix/tensor leaf | Niche now | Very high | Poor | Dedicated path |

## 10. Architecture Implications

A new leaf should require more than another enum variant. It should expose:

- A fitted state specific to the model family.
- Class or response distribution semantics.
- Probability, interval, and distribution queries where supported.
- Missing-data policy.
- Feature-type support.
- Serialization round-trip.
- Fallback behavior for insufficient leaf data.
- Diagnostics explaining selected structure and regularization.

The existing `LeafKind` and `LeafControl` should remain stable. New fitted models belong in node or leaf statistics, not in a giant flat control struct.

## 11. Evaluation Standard

Every new leaf family must compare against:

- Constant class or mean leaf.
- Existing linear/softmax leaf.
- KNN leaf.
- Best boosted-tree baseline on the same task.
- A global version of its own model, to verify local-leaf benefit.

Report:

- Predictive performance.
- Calibration and distribution quality.
- Fit and prediction time.
- Memory.
- Failure and fallback frequency.
- Explanation size and stability.

## 12. Risks

1. **Leaf zoo.** Every new family increases interface and persistence complexity.
2. **Duplicate capability.** GAM versus MARS, prototype versus KNN, NB versus LDA.
3. **Small-leaf overfitting.** Many probabilistic and kernel models need larger leaves.
4. **Fallback invisibility.** Simple fallback can hide an unusable requested model.
5. **Serialization growth.** Each fitted leaf state needs versioned persistence.
6. **Split-leaf confusion.** Some algorithms belong as split criteria rather than leaves.
7. **Research-method hype.** Recent preprints need independent reproduction.

## 13. Primary Sources

1. McCullagh, P. (1980). “Regression for Ordered Categorical Responses.” *JRSS-B* 42(1). [DOI](https://doi.org/10.2307/2984772).
2. Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*. Wiley.
3. Schwarz, G. (1978). “Estimating the Dimension of a Model.” *Annals of Statistics*. [DOI](https://doi.org/10.1214/aos/1176344276).
4. Fraley, C. et al. (2003). “Model-based Clustering, Discriminant Analysis, and Density Estimation.” *JASA*. [DOI](https://doi.org/10.1198/016214503000037).
5. Cortes, C. and Vapnik, V. (1995). “Support-Vector Networks.” *Machine Learning*. [DOI](https://doi.org/10.1007/BF00994018).
6. Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
7. Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 14 “Neural Nets, Kernels, and Regularization.”
