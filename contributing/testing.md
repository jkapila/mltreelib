# Writing tests

## Principles

- **Test behaviour, not wiring.** Assert what a user can observe (outputs, errors, saved files), not how the code is built. No test pins incidental implementation details.
- **Highest seam first.** Prefer the public API over internals; add a lower-level test only when the public one cannot isolate the behaviour.
- **Rust core is tested in Rust first** (`crates/mltreelib-core/tests/`); the Python layer gets pytest coverage of the user-visible API (`tests/`).

## What every new capability needs

1. A normal case with a **hand-checkable expected value**.
2. **Missing values:** `None` and NaN are kept, counted, never imputed, dropped or recoded by data helpers; the input object is unchanged (compare with `copy.deepcopy` and a NaN-safe equality).
3. **Empty and all-missing input** returns a sensible result rather than raising, unless raising is the documented behaviour.
4. **Determinism** wherever randomness is used: same seed, same output, across processes (a subprocess with different `PYTHONHASHSEED` values is a good check).
5. **Error cases** with the documented exception and a message that names the problem.
6. **Parity tests** for defaults across surfaces (Rust core, constructor, controls).

## Numerics

- Results are reproducible to about 12 significant digits on the same platform and build. Compare floats with a relative tolerance of 1e-12 (`math.isclose(a, b, rel_tol=1e-12, abs_tol=...)`), never with string equality. Do not claim bit-identical output across platforms, BLAS builds or thread counts.
- Timing numbers never appear in assertions or saved models.

## Running

```sh
cargo test                      # Rust core
uv run pytest tests -q          # Python behaviour tests
uv run ruff check <files>       # lint (must be clean)
```

Tests that need an extra package use `pytest.importorskip` and run with `uv run --with <package> pytest <file>` (for example the CVXPY reference solvers in `tests/cvx_reference.py`); the package itself never depends on them. Network tests are gated (for example `MLTREELIB_NETWORK_TESTS=1`) and skipped by default. Run the Rust and Python suites before committing; run the docs gates in [definition-of-done.md](definition-of-done.md) when docs or examples changed.
