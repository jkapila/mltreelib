# Writing specs

A spec records a change before it is built: what problem, what solution, what is decided, how it is tested, and what is out of scope. Specs live in `specs/NN-slug.md`, numbered in order, and are indexed in `specs/README.md`.

## Structure (in this order)

1. **Title and status line.** `# Spec NN: Title`, then `Status:` using one of Proposed, Active, Partially landed, Completed, Superseded (name the superseding spec).
2. **Scope boundary.** One paragraph on what this spec owns and what neighbouring specs own.
3. **Problem Statement.** From the user's view, with evidence (counts, examples). Findings from audits are written as verified or unverified.
4. **Solution.** From the user's view.
5. **User Stories.** A long numbered list, `As a <actor>, I want <capability>, so that <benefit>`.
6. **Implementation Decisions.** Modules, interfaces, schemas, contracts, rules (missing values, dependencies, determinism). No file paths and no code, except a snippet from a prototype that encodes a decision more precisely than prose.
7. **Testing Decisions.** What a good test is here (behaviour, not wiring), the highest seam, which modules are tested, prior art.
8. **Out of Scope.**
9. **Further Notes.**

## Rules

- **Verify before you write.** A claim that came from an automated run or a search is marked unverified until reproduced. Record outcomes under a heading such as "Audit Findings (verified)".
- **Decisions are recorded where they are made.** When the maintainer decides something, add it to the spec in the same change that acts on it ("Decisions Recorded").
- **Keep the index true.** Add a row to `specs/README.md` when a spec is created and update its status when it ships. Note overlaps and cross-reference neighbouring specs.
- **Specs describe the destination; issues describe the work.** Open issues for the units of work and reference the spec.
- Shared principles every spec respects: no scikit-learn and a shrinking dependency set; missing values are never imputed or recoded outside the model; results reproducible to about 12 significant digits on the same platform; one set of defaults on every surface.
