# CI Test Reports

## Summary

```{include} ../_generated/test_report_summary.md
```

---

## Detailed HTML reports (per stack)

Embedded **pytest-html** reports (wider layout: scroll horizontally on narrow viewports).

```{eval-rst}
.. raw:: html

   <style>
   .pytest-report-scroll { margin:1rem 0; overflow-x:auto; width:100%;
     -webkit-overflow-scrolling:touch; border:1px solid #e0e0e0; border-radius:6px; background:#fff; }
   .pytest-report-scroll iframe { display:block; border:0; height:960px;
     width:min(100%, 1600px); min-width:min(100%, 920px); }
   </style>
```

### Zoomy Core

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-core.html"
     title="Pytest HTML — Core small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-core.html"
     title="Pytest HTML — Core large"></iframe>
   </div>
```

### Zoomy JAX

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-jax.html"
     title="Pytest HTML — JAX small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-jax.html"
     title="Pytest HTML — JAX large"></iframe>
   </div>
```

### AMReX

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-amrex.html"
     title="Pytest HTML — AMReX small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-amrex.html"
     title="Pytest HTML — AMReX large"></iframe>
   </div>
```

### DMPlex

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-dmplex.html"
     title="Pytest HTML — DMPlex small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-dmplex.html"
     title="Pytest HTML — DMPlex large"></iframe>
   </div>
```

### FEniCSx

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-fenicsx.html"
     title="Pytest HTML — FEniCSx small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-fenicsx.html"
     title="Pytest HTML — FEniCSx large"></iframe>
   </div>
```

### Firedrake

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-firedrake.html"
     title="Pytest HTML — Firedrake small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-firedrake.html"
     title="Pytest HTML — Firedrake large"></iframe>
   </div>
```

Missing reports show a short placeholder inside the iframe.

---

## How to use this page

### Local book build with real reports

From the **repository root**:

```bash
python tests/reporting/generate_test_report.py … \
  --output-dir artifacts/test-reports/small/core   # or large/<stack>/…
python docs/scripts/generate_ci_test_report.py
jupyter-book build docs/book
```

**Files created or refreshed by `docs/scripts/generate_ci_test_report.py`:**

| Output | Role |
|--------|------|
| `docs/_generated/test_report_summary.md` | Summary tables for this page (`{include}`) |
| `docs/book/_static/pytest-report-small-<stack>.html` | Embedded small-suite HTML per stack |
| `docs/book/_static/pytest-report-large-<stack>.html` | Embedded large-suite HTML per stack |
| `docs/book/tutorials/ipynb/**` | Mirror of `tutorials/**/*.ipynb` for MyST pages |

Stub HTML is written when no matching `report.html` exists under `artifacts/`.

### Adding or changing tests

- Mark tests with the right **pytest markers** (`small`, `jax`, `core`, `firedrake`, …) so the correct CI job collects them.
- After CI runs, use the **Summary** table and the matching **pytest-html** section on this page.

### Docs-only changes

- Pushing documentation alone still rebuilds the site; **Render Webpage** re-downloads the latest Smart Tests artifacts from **`main`** when it runs.

---

## Architecture

- **Smart Tests** (`.github/workflows/tests-report.yml`) runs pytest inside **GHCR** images (`zoomy_core`, `zoomy_jax`, `zoomy_firedrake`, placeholders, …), bind-mounts the repo, and `pip install -e`’s the relevant `library/*` trees so reports match the commit under test.
- **Artifacts**: per-stack HTML/JUnit are merged into **`test-reports-small-bundle`** and **`test-reports-large-bundle`** (small vs large/benchmark jobs).
- **Render Webpage** downloads those bundles, runs `docs/scripts/generate_ci_test_report.py`, then builds the book (including this page).
- **Containers** must publish images before pulls make sense; a successful **Containers** run triggers another **Smart Tests** pass via `workflow_run` so `:latest` images match the same SHA.
- **Submodules**: Smart Tests uses `actions/checkout` with **`submodules: recursive`** so `library/*` exists on the runner.
