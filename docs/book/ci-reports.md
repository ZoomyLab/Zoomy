# CI Test Reports

This page embeds **pytest-html** output per solver backend. The script `docs/scripts/generate_ci_test_report.py`
(run before `jupyter-book build`) copies the newest `report.html` for each stack from:

- `artifacts/test-reports/small/<stack>/…` — artifact **`test-reports-small-bundle`** (PR / push path-triggered jobs,
  or a follow-up **Smart Tests** run after **Containers** completes)
- `artifacts/test-reports/large/<stack>/…` — artifact **`test-reports-large-bundle`** (weekly schedule or manual Smart Tests with large tests)

**Smart Tests** runs pytest inside prebuilt **GHCR** images (`zoomy_core`, `zoomy_jax`, `zoomy_firedrake`, placeholder
stacks, …), with the repository bind-mounted and `library/*` installed editable in the container so reports reflect
the commit under test.

**You do not need to manually run Smart Tests and then Render Webpage in sequence:** when Smart Tests finishes on
`main` or `master`, **Render Webpage** is also triggered (`workflow_run`) and downloads the **latest** bundles from
the branch. If you change only documentation, a normal docs push still builds the site (it re-downloads those latest
artifacts). You can still run either workflow by hand from the Actions tab when you want.

When you change Dockerfiles or conda/pip install specs, **Containers** must publish images before **Smart Tests** can
meaningfully use new system dependencies; the repo wires a **`workflow_run`** on successful **Containers** so **Smart
Tests** re-runs on the same SHA after images land. **Render Webpage** still follows **Smart Tests** via its own
`workflow_run` trigger.

DMPlex vs FEniCSx: tests under `tests/**/zoomy_dmplex/` get the `dmplex` marker; tests under `tests/**/zoomy_fenicsx/`
get `fenicsx` (see `tests/conftest.py`). CI runs DMPlex and Firedrake in the same **zoomy_firedrake** container; FEniCSx
uses a separate placeholder image.

### First-time setup (GitHub Actions)

The monorepo uses **git submodules** under `library/` (see `.gitmodules`). Smart Tests runs `actions/checkout` with
`submodules: recursive` so `library/zoomy_core/pyproject.toml` and siblings exist on the runner. If submodule checkout
fails (e.g. private submodules on a fork PR), fix access or use a machine user / deploy key.

1. **Merge** workflow changes to `main` (or `master`) so **Smart Tests** includes the bundle upload jobs.
2. Ensure **Containers** has published the GHCR images your stacks need (at least once after changing Dockerfiles or
   `install/*.yml`; otherwise `docker pull` in Smart Tests will fail).
3. Run **Smart Tests** once successfully on that branch (push a commit under the workflow path filters, open a PR, or use **Run workflow**).  
   - For **large** iframes too: run Smart Tests manually with **“Also run large/benchmark test suite”** checked, or wait for the weekly schedule.
4. Confirm the run published artifacts **`test-reports-small-bundle`** and **`test-reports-large-bundle`** (Actions run → **Artifacts**).  
   - If tests **fail**, you should still get HTML/JUnit when pytest wrote reports (`--ignore-pytest-exit-code` + upload on `always()`).
5. Trigger **Render Webpage** (it may already have run via `workflow_run` from step 3). Check the site **CI Test Reports** page.

Locally: run `python tests/reporting/generate_test_report.py …` into `artifacts/test-reports/small/<stack>/` or `large/…`, then `python docs/scripts/generate_ci_test_report.py` and `jupyter-book build docs/book`.

---

## Zoomy Core

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-core.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — Core small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-core.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — Core large"></iframe>
   </div>
```

## Zoomy JAX

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-jax.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — JAX small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-jax.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — JAX large"></iframe>
   </div>
```

## AMReX

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-amrex.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — AMReX small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-amrex.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — AMReX large"></iframe>
   </div>
```

## DMPlex

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-dmplex.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — DMPlex small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-dmplex.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — DMPlex large"></iframe>
   </div>
```

## FEniCSx

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-fenicsx.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — FEniCSx small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-fenicsx.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — FEniCSx large"></iframe>
   </div>
```

## Firedrake

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-firedrake.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — Firedrake small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-firedrake.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — Firedrake large"></iframe>
   </div>
```

Missing reports show a short placeholder inside the iframe.

## JUnit summary

```{include} ../_generated/test_report_summary.md
```
