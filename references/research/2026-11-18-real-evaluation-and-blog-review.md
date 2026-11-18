# Real evaluation and blog review

**Date:** 2026-11-18  
**Scope:** Real survival and randomized uplift examples, current blog claims, existing keyword research, and primary search guidance. The examples are exploratory single-split checks, not deployment validations.

## Dataset provenance and outcomes

- **FLCHAIN:** The R `survival` package describes a stratified sample of 7,874 subjects, with follow-up in days and death as the event ([package manual](https://cran.r-project.org/web/packages/survival/survival.pdf)). The example uses the [Rdatasets CSV](https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv), SHA-256 `a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563`. It excludes cause-of-death `chapter` from predictors. A 5,500/2,374 random split reports test IPCW Brier at 365, 1,095 and 1,825 days. Training-estimated marginal censoring weights rely on a censoring assumption. The 1,825-day tree and global-baseline scores are 0.0884 and 0.1026. Missing creatinine is passed to the tree directly.
- **Hillstrom:** The [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/hillstrom) describes 64,000 customers randomized to men's email, women's email and no email, with outcomes over two weeks. The original download failed HTTPS certificate verification locally. The [public mirror](https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv) used here has SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` and matches the documented 64,000 rows. We did not independently establish byte equivalence with the original. Data stays outside Git.
- The Hillstrom example compares women's email to no email using website visits, not purchases or profit. The 25,615/8,539/8,539 split selects treat-all on validation. On the test split, the selected policy and treat-all both estimate 0.04517 incremental visits per eligible customer, approximate interval [0.03108, 0.05927]. This run provides no evidence that the learned targeting policy beats treat-all. It assumes zero cost and uses a historical randomized trial.

## Blog findings and decisions

The original six articles contained claims that exceeded the cited or checked evidence: unconditional legal and monotonicity claims, synthetic examples described as industrial or customer benchmarks, universal model superiority, an unsupported zero-copy assertion, and claims that squared error assumes Gaussian outcomes or that Tobit eliminates bias. The updated articles use practical questions, identify synthetic data, point to reproducible examples, and preserve limits. The blog index and homepage summaries should describe the final articles, not the abandoned claims.

The repository's [November 16 keyword research](2026-11-16-seo-keyword-volume-and-search-intent.md) supplies dated US search-intent hypotheses such as `decision tree regression`, `decision tree missing values`, `cubist model` and `uplift model`. Its volumes are historical, not current performance or ranking guarantees. Use terms naturally where the article actually answers the question. The documentation pathways retain detailed reference material; articles explain the decision, evidence, limitations and next step rather than repeating the entire manual.

[Google's people-first guidance](https://developers.google.com/search/docs/fundamentals/creating-helpful-content) calls for useful, original and reliable material. Its [guidance for AI search](https://developers.google.com/search/blog/2025/05/succeeding-in-ai-search) recommends accessible, useful content and structured data that matches visible content. Its [generative content guidance](https://developers.google.com/search/docs/fundamentals/using-gen-ai-content) stresses accuracy and review. Question-led headings improve navigation when they answer genuine reader needs; none of these sources guarantees ranking, citations, or preferential treatment by an LLM.

## Remaining checks

- Real-world censoring can depend on features. Assess conditional censoring and horizon-specific calibration before deployment.
- Run repeated splits and compare uplift policies with uncertainty and real campaign cost. The current policy did not improve on treat-all.
- Audit remaining generated charts, benchmark claims, documentation snippets and category-label explanations. Keep a single visualization catalog and stable links.
- Treat indexed keyword volumes as dated estimates. Measure actual search and reader behavior before expanding articles.

## Real data first revision

By task, the primary studies now use diamonds (regression), Give Me Some Credit
(classification), Wine Quality (clustering), Shuttle (anomaly review), FLCHAIN
(survival) and Hillstrom (uplift). The [dataset setup](../../docs/guides/real-data.md)
records source links and local paths. Bank churn remains a secondary interface
comparison. Synthetic cases test ground truth and controlled missingness. They
are separate from evidence about observed data.

[Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) provides
measured chemistry for a clustering case with interpretable profiles. The
refit diagnostic compares co-assigned pairs on common review data. The same
training scale stays fixed, so this measures partition stability with that scale.
[APS Failure](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
provides genuinely missing sensors and an observed component label. That label
supports diagnostic triage. It gives no measured time to failure.

The [keyword intent note](2026-11-16-seo-keyword-volume-and-search-intent.md)
informed the blog headings for decision tree regression, missing values, model
trees and credit classification. Each article answers a concrete question,
shows the fit and independent evaluation, and interprets its own figure. Search
volume remains a dated estimate. Publication is separate from documentation
build and evaluation.
