# Testing Guide

```{warning}
**Work in progress.** The folder layout and workflow descriptions below have
drifted from the tree. The parts that are current are the **marker model** and
the **CI tiering**, both summarised on [CI Test Reports](ci-reports.md).
```

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

### Containers workflow

`.github/workflows/build-containers.yml` (workflow name **Containers**)

- Builds and pushes stack images to GHCR (`zoomy_core`, `zoomy_jax`, `zoomy_firedrake`, dev bases, placeholders, …)
  when Dockerfiles, `install/*.yml`, or selected library `pyproject.toml` files change.

### Smart test workflow

`.github/workflows/tests-report.yml` (workflow name **Smart Tests**)

- **Runtime**: each stack job logs into GHCR, pulls the matching image, bind-mounts the repo at `/workspace`, runs
  `pip install -e` for the relevant `library/*` packages **inside** the container, then
  `tests/reporting/generate_test_report.py` (so CI exercises the checked-out tree against the image’s solver stack).
- path-aware test selection on PRs per runtime group:
  - tutorial tests run inside their runtime group job (no dedicated tutorial runtime lane)
  - `core` / `jax`
  - `amrex`
  - `dmplex` / `fenicsx` (split paths; `dmplex` and `firedrake` share one container)
  - `firedrake`
  - PRs that only touch **containers**, **install/**, or the **Containers** workflow hit the **`infra`** path and run
    all stack jobs.
- **After images**: when **Containers** finishes successfully, **Smart Tests** is triggered again via `workflow_run`,
  checking out the **same commit** (`workflow_run.head_sha`) so pulls of `:latest` match the images just pushed.
  Pushes that match the Smart Tests path filters can still start a run in parallel; the `workflow_run` pass is the
  one guaranteed to see fresh GHCR tags for container-only changes.
- scheduled and manual **large / benchmark** runs: one job per stack (same backends as small), merged into
  **`test-reports-large-bundle`**
- manual run with optional large test toggle
- HTML + JUnit per stack job; follow-up jobs merge stack artifacts into **`test-reports-small-bundle`** and
  **`test-reports-large-bundle`** so docs can download two artifacts (small vs large, each with per-stack folders).
- **Render Webpage** downloads the latest completed bundles of those names before building the book, and can also
  run after Smart Tests via `workflow_run` (so the usual chain is **Containers → Smart Tests → Render Webpage** when
  all succeed).
- Optional extra dependency pins for local or legacy setups live under `tests/requirements/*.txt`; the Smart Tests
  workflow does not use those files on the GitHub runner (dependencies come from the container images plus editable
  installs of the repo).

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
