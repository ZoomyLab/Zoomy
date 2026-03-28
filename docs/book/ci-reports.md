# CI Test Reports

This page embeds **pytest-html** output per solver backend. The script `docs/scripts/generate_ci_test_report.py`
(run before `jupyter-book build`) copies the newest `report.html` for each stack from:

- `artifacts/test-reports/small/<stack>/…` — artifact **`test-reports-small-bundle`** (PR / push path-triggered jobs)
- `artifacts/test-reports/large/<stack>/…` — artifact **`test-reports-large-bundle`** (weekly schedule or manual Smart Tests with large tests)

**You do not need to manually run Smart Tests and then Render Webpage in sequence:** when Smart Tests finishes on
`main` or `master`, **Render Webpage** is also triggered (`workflow_run`) and downloads the **latest** bundles from
the branch. If you change only documentation, a normal docs push still builds the site (it re-downloads those latest
artifacts). You can still run either workflow by hand from the Actions tab when you want.

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

## PETSc / DMPlex / FEniCSx

### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-small-petsc.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — PETSc small"></iframe>
   </div>
```

### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-frame" style="margin:1rem 0;">
   <iframe src="_static/pytest-report-large-petsc.html" width="100%" height="900"
     style="border:1px solid #e0e0e0;border-radius:6px;background:#fff;"
     title="Pytest HTML — PETSc large"></iframe>
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
