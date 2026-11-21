# Discriminant Analysis as an `mltreelib` Leaf Family

**Date:** 2026-11-21  
**Scope:** LDA, QDA, regularized discriminant analysis, sparse discriminant analysis, robust and semiparametric variants, and exponential-family discriminant analysis as terminal classification leaves.  
**Decision informed:** Which discriminant estimators should enter the package, in what order, and how they should integrate with tree leaves.  
**Confidence:** High for classical estimators and package fit. Medium for recent methods, especially 2026 preprints.

## 1. Executive Summary

Discriminant analysis is an excellent leaf-family candidate. It predicts a class from a generative class-conditional model, returns calibrated probabilities, exposes class means and covariance structure, and fits naturally inside a divide-and-conquer tree.

The package currently lacks discriminant leaves. `LeafKind` supports scalar leaves, linear models, MARS, KNN, binary probability leaves, and a multinomial softmax path. Discriminant analysis would add a class-conditional generative model that complements discriminative logistic or softmax leaves.

### Recommended ladder

1. **Shrinkage LDA** with pooled covariance. Default discriminant leaf.
2. **Diagonal LDA** fallback for very small leaves or numerical failure.
3. **Regularized Discriminant Analysis (RDA)** interpolating pooled and class-specific covariance.
4. **Sparse LDA** when `p >> n` or users request feature selection.
5. **Sparse QDA / Community Bayes** as research work for explicit interactions and sparse feature communities.
6. **Exponential Family Discriminant Analysis (EFDA)** for mixed and non-Gaussian response-generating features.

Full unregularized QDA must not be the default. Within small tree leaves, per-class covariance matrices are frequently singular.

## 2. Classical LDA

For class \(k\), assume:

\[
X \mid Y=k \sim \mathcal{N}(\mu_k,\Sigma)
\]

with shared covariance \(\Sigma\). The discriminant score is:

\[
\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\frac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k
\]

and prediction uses the largest score.

Strengths:

- Closed-form fit.
- Low training and prediction cost.
- Strong with enough rows relative to features.
- Coefficients and class centroids remain inspectable.
- Natural probabilistic output.

Weaknesses:

- Shared-covariance Gaussian assumption.
- Singular or ill-conditioned covariance when leaf rows are few.
- Poor behavior under strong class-specific covariance or heavy tails.

## 3. Classical QDA

QDA allows a separate covariance \(\Sigma_k\) for each class:

\[
\delta_k(x)=-\frac12\log|\Sigma_k|
-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)
+\log\pi_k
\]

Strengths:

- Captures class-specific variance and interaction structure.
- Still generative and interpretable through class centroids and covariance matrices.

Weaknesses:

- Requires substantially more observations per class than LDA.
- Singular covariance is common in small leaves.
- Raw QDA overfits leaf noise.
- Parameter count grows as \(O(KP^2)\).

### Package decision

Implement QDA only through shrinkage or regularization. An unregularized `LeafKind::Qda` should not exist.

## 4. Regularized Discriminant Analysis

Friedman's RDA interpolates between LDA and QDA using two parameters:

- \(\lambda\): weight of class-specific covariance.
- \(\gamma\): shrinkage of pooled covariance toward a scaled identity matrix.

Conceptually:

\[
\widehat\Sigma_k(\lambda,\gamma)
=(1-\lambda)S_k+\lambda S_{\text{pooled}}
\]

followed by shrinkage of the result toward a scaled identity matrix.

This gives one estimator spanning:

- \(\lambda=0\): diagonal or pooled LDA-like behavior.
- \(\lambda=1\): class-specific covariance.
- \(\gamma\): covariance stabilization when rows are scarce.

RDA is the most important bridge between LDA and QDA. It should expose both parameters and choose them inside each leaf through a documented internal validation rule, or accept user-specified fixed values.

Source: Friedman, “Regularized Discriminant Analysis,” *JASA* 84(405), 1989.

## 5. Covariance Stabilization

### 5.1 Ledoit-Wolf shrinkage

Ledoit-Wolf estimates an optimal shrinkage intensity between a sample covariance matrix and a scaled identity target.

This is a strong default because it adapts to leaf sample size and feature dimension without a manual grid.

Source: Ledoit and Wolf, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,” *Journal of Multivariate Analysis* 88, 2004. [Paper](http://www.ledoit.net/Well-conditioned2004.pdf).

### 5.2 Ridge and diagonal fallbacks

Ridge covariance:

\[
\widehat\Sigma=(1-\gamma)S+\gamma I
\]

Diagonal LDA discards off-diagonal covariance entirely. It is less accurate but remains stable when a leaf has too few rows for a full covariance estimate.

### 5.3 Banding and thresholding

Bickel-Levina banding uses feature order to restrict covariance neighborhoods. Thresholding produces sparse covariance graphs. These fit structured feature orders and grouped data, but do not fit arbitrary columns without an order.

Source: Bickel and Levina, “Regularized Estimation of Large Covariance Matrices,” *Annals of Statistics* 36(1), 2008. [Paper](https://www.stat.berkeley.edu/~bickel/BL2008-banding.pdf).

## 6. Sparse Discriminant Analysis

Sparse discriminant analysis penalizes discriminant directions or precision terms so that irrelevant features receive zero coefficients.

### 6.1 Penalized optimal scoring

Hastie, Buja, and Tibshirani's Penalized Discriminant Analysis penalizes changes in discriminant functions along curves. Clemmensen et al.'s Sparse Discriminant Analysis uses elastic-net penalties on discriminant components.

Sources:

- Hastie, Buja, and Tibshirani, “Penalized Discriminant Analysis,” *Annals of Statistics* 23(1), 1995. [Paper](https://hastie.su.domains/public/Papers/pda.pdf).
- Clemmensen, Hastie, Witten, and Ersbøll, “Sparse Discriminant Analysis,” *Technometrics* 53(4), 2011. [Paper](http://hastie.su.domains/public/Papers/sda_resubm_daniela-final.pdf).

### 6.2 Direct sparse LDA

DSDA and related methods directly optimize a sparse discriminant objective. They are designed for \(p \gg n\).

Sources:

- Mai and Zou, “Direct Sparse LDA,” *Biometrika* 99(1), 2012.
- Cai and Liu, “A Direct Estimation Approach to Sparse LDA,” *JASA* 106(496), 2011.

### 6.3 Package fit

The existing `LeafControl.l1` and `LeafControl.l2` naturally map to sparse discriminant penalties. `Ccd` can solve penalized least-squares formulations.

However, direct sparse LDA needs a dedicated objective. It should be a later estimator, not conflated with ridge LDA.

## 7. Sparse QDA and Community Bayes

Le and Hastie's *Sparse Quadratic Discriminant Analysis and Community Bayes* creates a continuous family between QDA and naive Bayes.

### 7.1 Sparse QDA

A group-lasso penalty is applied across class-specific precision matrices. The same feature-pair sparsity pattern is encouraged across classes.

- \(\lambda=0\): QDA.
- \(\lambda\rightarrow\infty\): naive Bayes.
- Intermediate values retain selected interactions while shrinking others.

### 7.2 Community Bayes

Conditional-independent feature communities are estimated from class-specific rank correlations. A separate classifier is fit inside each community. Community posteriors combine analytically.

This is highly aligned with `mltreelib` because:

- It reduces variance in small leaves.
- It exposes feature communities.
- It handles interactions explicitly.
- It generalizes beyond Gaussian data.

Source: Le and Hastie, “Sparse Quadratic Discriminant Analysis and Community Bayes.” [arXiv:1407.4543](https://arxiv.org/abs/1407.4543).

### Package decision

Research-grade estimator after LDA, RDA, and sparse LDA work.

## 8. Robust Discriminant Analysis

Class means and sample covariances are sensitive to outliers. Robust DA replaces them with robust location and scatter estimators such as minimum covariance determinant (MCD).

Strength:

- Better behavior on contaminated leaf rows.

Weakness:

- MCD estimation is expensive, often \(O(n^2)\) or worse.
- Repeated robust estimation inside every tree node may dominate runtime.
- Tree splits already isolate unusual regions.

Package decision:

- Optional robust covariance mode later.
- Not a default leaf.

Recent review: [arXiv:2408.15701](https://arxiv.org/abs/2408.15701).

## 9. Semiparametric and Copula Discriminant Analysis

Gaussian-copula discriminant analysis estimates monotone marginal transformations and a latent correlation structure. It can support mixed-type or non-Gaussian features better than raw LDA.

Recent examples:

- Sparse semiparametric discriminant analysis for high-dimensional zero-inflated data, *JMLR* 26, 2025. [Paper](https://jmlr.org/papers/volume26/24-0046/24-0046.pdf).
- Transfer learning through regularized random-effects LDA, *JMLR* 27, 2026. [Paper](https://jmlr.org/papers/volume27/25-0031/25-0031.pdf).

Package decision:

- Research phase after core discriminant leaves.
- Copula transforms inside every leaf would add substantial fit complexity.

## 10. Exponential Family Discriminant Analysis

Lakkapragada's 2026 EFDA paper generalizes LDA to class-conditionals in a common exponential family:

\[
f(x\mid k)=h(x)\exp(\eta_k^\top T(x)-A(\eta_k))
\]

The log-odds become linear in the sufficient statistic \(T(x)\), not necessarily in \(x\). Under conditional independence, EFDA becomes naive Bayes. With multivariate Gaussian class-conditionals, EFDA recovers LDA.

The paper derives closed-form estimators for Normal, Laplace, Exponential, Gamma, Weibull, Poisson, Bernoulli, and Negative Binomial components. It reports materially better calibration than misspecified Gaussian and logistic models on non-Gaussian synthetic data.

Source: [arXiv:2603.20655](https://arxiv.org/abs/2603.20655).

### Strengths

- Naturally mixed-type: Poisson counts, Gamma or Weibull positive features, and Bernoulli indicators.
- Closed-form univariate components.
- Interpretable log-odds in sufficient-statistic space.
- Complements existing GLM response families.

### Risks

- Very recent preprint.
- Claims need independent reproduction.
- Multivariate dependence remains difficult.
- Known-shape assumptions may be unrealistic.

Package decision:

- Add as experimental leaf after classical LDA.
- Begin with one-dimensional product EFDA for mixed numeric and binary leaves.
- Never present it as peer-reviewed settled methodology.

## 11. Relationship to Other Leaf Families

| Leaf | Objective | Discriminant alternative |
|---|---|---|
| Linear/logistic | Discriminative conditional likelihood | Generative class-conditional model |
| Softmax | Multinomial discriminative likelihood | Multinomial LDA/QDA |
| MARS | Flexible additive regression | FDA on nonlinear basis |
| KNN | Local nonparametric regression/classification | Prototype or discriminant leaf |
| Gaussian GLM | Conditional mean, homoscedastic | LDA/QDA model class priors too |

Discriminant leaves should be a distinct `LeafKind`, not hidden behind `Linear`. Their state includes class priors, means, covariance representation, regularization parameters, and feature scaling.

## 12. Integration with Tree Growth

### 12.1 Leaf fitting

The leaf receives class labels and active predictor columns. It fits the discriminant model on rows reaching the leaf.

### 12.2 Split compatibility

Current split criteria use response deviance. Discriminant leaves fit independently after partition selection, like linear, MARS, and KNN leaves.

A future model-based split criterion could use discriminant likelihood, but that is separate work.

### 12.3 Class balance

Discriminant leaves need class priors. Small leaves may contain one class only. Such leaves must fall back to the constant class probability rather than attempting covariance estimation.

### 12.4 Missing data

Discriminant estimation needs complete-case or explicit missing-data handling. The package's existing leaf NA policies can apply, but covariance estimation under imputation needs care. Missing values must not silently become zero.

### 12.5 Prediction

The leaf should return:

- Class probabilities.
- Predicted class.
- Discriminant scores.
- Optional projected class features for visualization.

## 13. Implementation Ladder

### Step 1: Shrinkage LDA

- Multiclass support.
- Pooled covariance.
- Ledoit-Wolf shrinkage.
- Cholesky solve through existing `faer` dependency.
- Diagonal fallback when estimation is impossible.
- Constant-class fallback.

### Step 2: RDA

- Class-covariance weight \(\lambda\).
- Identity-shrinkage weight \(\gamma\).
- Fixed user values or internal validation.
- Report selected regularization.

### Step 3: Sparse LDA

- Reuse `LeafControl.l1` and `LeafControl.l2`.
- Penalized least-squares or optimal-scoring formulation.
- Report selected features and zero coefficients.

### Step 4: Research discriminants

- Sparse QDA and Community Bayes.
- Robust covariance.
- Copula discriminant.
- EFDA.

## 14. Evaluation Protocol

Compare discriminant leaves against softmax/logistic, Gaussian NB, KNN, and boosted-tree baselines on:

- Multiclass classification accuracy, macro F1, log loss, and Brier score.
- Calibration error and reliability plots.
- Performance by leaf size and feature-to-row ratio.
- Missingness rates.
- Class imbalance.
- Correlated and interaction-heavy features.
- Categorical and mixed numerical predictors where supported.
- Runtime, allocation count, and memory.

Use synthetic data with known class means and covariances to verify parameter recovery. Use public tabular datasets for predictive comparison.

## 15. Risks

1. **Singular covariance.** Small leaves make this the dominant numerical risk.
2. **Hidden shrinkage defaults.** Results become opaque if \(\gamma\) and \(\lambda\) are not reported.
3. **False linear interpretation.** LDA coefficients are valid under shared-covariance Gaussian assumptions.
4. **QDA overfitting.** Unregularized class covariance must never be default.
5. **Recent-method overreach.** EFDA and semiparametric variants need reproduction.
6. **Class absence.** One-class leaves require explicit fallback.
7. **Missing-value ambiguity.** Complete-case and imputed covariance need separate documentation.
8. **Duplicate leaf roles.** MARS plus FDA and KNN plus prototypes can become redundant without distinct use cases.

## 16. Primary Sources

1. Fisher, R. A. (1936). “The Use of Multiple Measurements in Taxonomic Problems.” *Annals of Eugenics*.
2. Friedman, J. H. (1989). “Regularized Discriminant Analysis.” *JASA*. [DOI](https://doi.org/10.1080/01621459.1989.10478752).
3. Hastie, T., Buja, A., and Tibshirani, R. (1995). “Penalized Discriminant Analysis.” [Paper](https://hastie.su.domains/public/Papers/pda.pdf).
4. Hastie, T., Tibshirani, R., and Buja, A. (1994). “Flexible Discriminant Analysis by Optimal Scoring.” [DOI](https://doi.org/10.1080/01621459.1994.10476866).
5. Ledoit, O. and Wolf, M. (2004). “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.” [Paper](http://www.ledoit.net/Well-conditioned2004.pdf).
6. Clemmensen, S. et al. (2011). “Sparse Discriminant Analysis.” [Paper](http://hastie.su.domains/public/Papers/sda_resubm_daniela-final.pdf).
7. Le, Y. and Hastie, T. (2016). “Sparse Quadratic Discriminant Analysis and Community Bayes.” [arXiv:1407.4543](https://arxiv.org/abs/1407.4543).
8. Ghojogh, B. and Crowley, M. (2019). “Linear and Quadratic Discriminant Analysis: Tutorial.” [arXiv:1906.02590](https://arxiv.org/abs/1906.02590).
9. Lakkapragada, A. (2026). “Exponential Family Discriminant Analysis.” [arXiv:2603.20655](https://arxiv.org/abs/2603.20655).
10. Chung, H. C., Ni, Y., and Gaynanova, I. (2025). “Sparse Semiparametric Discriminant Analysis for High-dimensional Zero-inflated Data.” [JMLR](https://jmlr.org/papers/v26/24-0046.html).
