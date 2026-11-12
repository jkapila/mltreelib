# Microprediction positioning patterns for mltreelib

**Date:** 2026-11-12  
**Question:** What do the user's Microprediction links actually show about linking open-source packages, public writing, agent skills, and commercial credibility?

## Observed first-party pattern

1. **One identity connects many technical assets.** The [Microprediction GitHub profile](https://github.com/microprediction) links a home site, a repository directory, a book, a blog, and specific packages. The [repository directory](https://repos.microprediction.org/) groups algorithm sites and research/writing sites under one name and links back to the GitHub organization and main home. The [home repository](https://github.com/microprediction/home) collects papers, essays, presentations, audio/video, packages, and book material. This is a publication and navigation system around the work, not merely a package README.
2. **Each useful package gets its own concrete door.** [HumpDay](https://github.com/microprediction/humpday) foregrounds a narrow job (derivative-free optimizer selection), links documentation and live demos, and carries examples, benchmarks, a changelog, citation guidance, contribution guidance, and a site. [Optimizer Elo Ratings](https://github.com/microprediction/optimizer-elo-ratings) is a distinct, public comparison artifact with HTML/JSON ratings and a Colab notebook; the package and comparison reinforce each other. These pages do not prove every method is best, but they give readers something to try and inspect.
3. **The writing connects methods to applied domains.** The [home repository](https://github.com/microprediction/home) and [repository directory](https://repos.microprediction.org/) connect packages to portfolio construction, time series, inventory/pricing, mechanisms, and other concrete problems. Papers, blogs, demos, and implementations can lead a reader from a business question to the author's method and code. This is an inference about the navigation design, not a measured acquisition funnel.
4. **An agent skill is used as product distribution.** HumpDay includes a [SKILL.md](https://github.com/microprediction/humpday/blob/main/SKILL.md) that tells a coding agent when to recommend the package during Python code review and includes an install command, replacement example, and limits. It is explicitly promotional. For mltreelib, a skill that diagnoses a real tabular workflow and compares honestly would likely earn more trust than one that merely recommends mltreelib whenever trees appear. This latter judgment is an inference.
5. **Funding is visible; consulting conversion is not established from these pages.** The [repository directory](https://repos.microprediction.org/) asks employers to nominate the projects for open-source funding and links a funding file. The [GitHub profile](https://github.com/microprediction) displays a Sponsor entry and author identity. The three requested pages did not provide a verified consulting-services offer, price, conversion rate, or evidence that these assets generate consulting clients. Do not describe a proven consulting funnel from them.

## Adaptation to the current mltreelib phase

The useful model is **one public theme, multiple proof artifacts**. The theme could be: *transparent decisions from structured tabular data, with honest validation and inspectable uncertainty*. The forest/ensemble phase should deepen that theme through a shared reporting contract, rather than adding isolated models and examples.

### What the forest and reporting layer should make reusable

The base-tree capability should remain the engine. A forest should add a measured improvement in prediction or stability while preserving a common way to inspect any fitted model. Before expanding the algorithm menu, define the **observable report contract** across one tree and an ensemble. It can be assembled from the existing `Metrics`, group reports, leaf summaries, importance, and explanation methods; it need not be a new public class until a second model proves the shared shape.

At minimum, a saved report should identify the package/model version, task and target, data and split provenance (including group/time columns), sample and missing counts, held-out metrics and baseline, calibration or intervals where meaningful, subgroup performance, model size/runtime, and warnings or unsupported explanation methods. For an ensemble, add tree count, variation across members, and global/local explanation method with its assumptions. A single-tree path breakdown is not automatically an explanation of a forest prediction; the report must say which prediction it explains and whether attributions add up to that prediction. Make this output machine-readable (JSON or a stable dict) and render a concise human summary from the same facts, so an agent cannot silently invent the numbers.

Business use cases also need two distinct sets of controls. **Model controls** govern fitting (sampling, tree count, leaf form, monotonicity, weighting, and runtime). **Decision controls** govern what to do with predictions (contact capacity, false-positive cost, intervention cost, alert threshold, and segment policy). Keep the latter in the report/evaluation workflow rather than baking a single business action into the estimator. This lets a customer reuse one fitted forest under several operational policies and compare them on held-out data.

| Pattern to adapt | Concrete mltreelib artifact | Why it fits now |
|---|---|---|
| A narrow, useful package door | A “business decision trees” landing page with three entry paths: predict, explain, audit | Readers can choose a job before facing the full algorithm menu. |
| Public, inspectable comparison | A reproducible comparison site showing per-dataset accuracy, runtime, completion rate, and model complexity for single trees and ensembles | A forest can be presented as a measured accuracy/explanation tradeoff, including losses, rather than a new feature claim. |
| Applied writing linked to executable code | Three business case studies with checked-in synthetic or redistributable data, a runnable script, report JSON, and human narrative | Demonstrates deployment shape without exposing client data. |
| Reusable agent workflow | One task-specific skill that produces a leakage-aware fit/compare/explain report from a dataset and explicitly records split, model version, and caveats | Makes an agent a repeatable operator over a stable package contract. |
| Clear human contact | One maintained site page that states available consulting help, target problems, expected inputs/deliverables, and a contact path, if the author wants consulting leads | Makes the commercial offer explicit; the Microprediction pages alone do not verify this step. |

## Business cases and article angles to test

- **Churn and retention:** compare a transparent single tree with an ensemble; report calibrated risk, top subgroups, intervention eligibility, and the accuracy/explanation tradeoff. Article: “When a forest improves churn prediction but weakens the decision story.”
- **Warranty or quality failures:** grouped by product line and plant; show how group-aware validation changes apparent performance, then surface segments and missingness in a report. Article: “Why a random split overstated our warranty model.”
- **Inventory/demand anomalies:** time-aware split, seasonal context, and contextual anomaly explanation. Article: “An anomaly score needs its context before an operations team can act.”
- **Credit or risk triage:** demonstrate monotonic constraints, missingness handling, and report provenance on synthetic data; avoid implying regulatory approval or causal explanations. Article: “From a risk score to an auditable reason code: what the model can and cannot say.”

Each case should show a **decision**, a baseline, a held-out result, an explanation/summarization artifact, and a failure mode. A notebook alone is weaker than the runnable script and stored report it narrates. Do not assert these capabilities already exist uniformly; the [modularity/coverage audit](2026-11-12-modularity-coverage-audit.md) documents gaps in docs and public examples.

A compact publication sequence would be: (1) one conceptual essay on why group/time structure changes a business decision; (2) the warranty case as a reproducible validation lesson; (3) churn as the single-tree-versus-forest tradeoff with intervention capacity; (4) contextual anomaly reporting; (5) a transparent benchmark update showing losses as well as wins. Each post should point to the same runnable example and versioned report artifact as the docs. This gives a reader a path from problem to evidence to package, and gives a prospective consulting client a concrete example of the work being offered.

## Limits

`https://home.microprediction.org/` was inaccessible through the available web tool on 2026-11-12, so the accessible [first-party home repository](https://github.com/microprediction/home) was used as a substitute. No traffic, conversion, downloads, consulting revenue, or user-demand claims were verified. The recommendations are inferences from observed first-party packaging and mltreelib's current phase, not evidence of business results.
