# Testing Guide

## Purpose

The test setup separates quick correctness checks from expensive benchmark runs and tutorial/notebook validation.

The goals are:

- fast PR feedback for changed code
- reproducible baselines for selected regression tests
- scheduled heavy runs (large and benchmark)
- smoke checks to keep tutorials and notebooks up-to-date

## Folder Structure

The `tests` directory is organized by intent first, then by package/backend:

- `tests/unit/`
  - small, focused unit tests
  - no heavy runtime expectations
  - split into `zoomy_core` and `zoomy_jax`

- `tests/regression/`
  - behavior/regression checks on small canonical cases
  - includes tutorial import smoke checks
  - split into `zoomy_core` and `zoomy_jax`

- `tests/benchmarks/`
  - expensive performance-oriented tests
  - disabled by default in local/PR runs

- `tests/scripts/`
  - script-style test scenarios (recent SWE v2 benchmark/check scripts)
  - imported by pytest smoke/regression tests

- `tests/results/baselines/`
  - tiny reference artifacts used for regression comparisons
  - created automatically if missing or when `ZOOMY_CREATE_BASELINES=1`

- `tests/common/`
  - shared test helpers (baseline storage, utilities)

- `tests/notebooks/`
  - notebook smoke list and notebook testing support files

- `tests/old/`
  - archived legacy tests kept for historical reference
  - excluded from active pytest discovery

- `tests/reporting/`
  - test-report generation scripts (HTML/JUnit)
  - notebook validation and jupytext compile checks

## Markers

Important pytest markers:

- `small`: quick tests for PR/local iteration
- `tutorial`: tutorial smoke checks (orthogonal intent marker)
- `jax`, `numpy`: backend-specific grouping
- `core`, `amrex`, `petsc`, `firedrake`: runtime/container-specific grouping
- `large`, `benchmark`: expensive tests (scheduled/manual)

Default local fast run:

```bash
pytest tests -m "small or tutorial"
```

Recommended stack-selective runs:

```bash
pytest tests -m "small and core"
pytest tests -m "small and jax"
pytest tests -m "small and amrex"
pytest tests -m "small and petsc"
pytest tests -m "small and firedrake"
```

Tutorial checks should always combine `tutorial` with a runtime marker:

```bash
pytest tests -m "small and tutorial and core"
# or:
pytest tests -m "small and tutorial and jax"
```

## Baseline Workflow

Some regression tests compare against compact baseline files in `tests/results/baselines`.

- If baseline exists: test compares current output against baseline.
- If baseline missing: baseline is created.
- To refresh baselines intentionally:

```bash
ZOOMY_CREATE_BASELINES=1 pytest tests -m small
```

## CI Workflows

### Smart test workflow

`.github/workflows/tests-report.yml`

- path-aware test selection on PRs per runtime group:
  - tutorial tests run inside their runtime group job (no dedicated tutorial runtime lane)
  - `core` / `jax`
  - `amrex`
  - `petsc`
  - `firedrake`
- scheduled and manual **large / benchmark** runs: one job per stack (same backends as small), merged into
  **`test-reports-large-bundle`**
- manual run with optional large test toggle
- HTML + JUnit per stack job; follow-up jobs merge stack artifacts into **`test-reports-small-bundle`** and
  **`test-reports-large-bundle`** so docs can download two artifacts (small vs large, each with per-stack folders).
- **Render Webpage** downloads the latest completed bundles of those names before building the book, and can also
  run after Smart Tests via `workflow_run`.
- runtime jobs can opt into dedicated dependency sets via:
  - `tests/requirements/amrex.txt`
  - `tests/requirements/petsc.txt`
  - `tests/requirements/firedrake.txt`
  (if a file is absent, CI skips that stack-specific install step)

### Notebook workflow

`.github/workflows/notebooks.yml`

- PR: validate changed notebooks + jupytext temporary conversion compile check
- schedule/manual: validate all notebooks
- optional smoke execution using `tests/notebooks/smoke_notebooks.txt`

## Notebook Policy

- Source of truth remains `.ipynb` (for docs publishing).
- No paired `.py` notebook files are committed.
- `jupytext` is used only transiently in checks:
  - convert notebook content to temporary Python text
  - run compile/syntax check
  - discard temporary files

This keeps notebook docs authoritative while still improving maintainability and CI diagnostics.
