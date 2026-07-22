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
     title="Pytest HTML — Zoomy Core small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-core.html"
     title="Pytest HTML — Zoomy Core large"></iframe>
   </div>
```

### Zoomy JAX

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-jax.html"
     title="Pytest HTML — Zoomy JAX small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-jax.html"
     title="Pytest HTML — Zoomy JAX large"></iframe>
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

### OpenFOAM

#### Small

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-small-foam.html"
     title="Pytest HTML — OpenFOAM small"></iframe>
   </div>
```

#### Large

```{eval-rst}
.. raw:: html

   <div class="pytest-report-scroll">
   <iframe src="_static/pytest-report-large-foam.html"
     title="Pytest HTML — OpenFOAM large"></iframe>
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
| `docs/book/tutorials/ipynb/**` | Mirror of the published `tutorials/{swe,sme,amrex}` notebooks |

Stub HTML is written when no matching `report.html` exists under `artifacts/`.

### Adding or changing tests

Markers are mostly **automatic** — the root `conftest.py` applies them, so there
are no hand-maintained lists:

- A test under `library/zoomy_<stack>/tests` gets the `<stack>` marker from its
  path (`foam` maps to `openfoam`).
- A test with no size marker becomes `small`. Anything taking more than
  **5 minutes individually** must be marked `large` explicitly.
- `library/zoomy_<stack>/tests` is only collected when `import zoomy_<stack>`
  resolves, so each container automatically runs exactly the suites it can.

So in practice: put the test in the right directory, and mark it `large` if it
is slow. After CI runs, read the **Summary** table and the matching
**pytest-html** section above.

### Docs-only changes

Pushing documentation alone still rebuilds the site; **Render Webpage**
re-downloads the latest Smart Tests artifacts from **`main`** when it runs.

---

## Architecture

**Scope.** CI tests four stacks: **zoomy_core**, **zoomy_jax**, **zoomy_amrex**
and **zoomy_foam**. Each runs in its own GHCR image, with the checked-out tree
`pip install -e`'d over it so reports match the commit under test.

**Two tiers.**

| Tier | Runs on | Markers | Gates? |
|---|---|---|---|
| small | every push and PR | `small and <stack>` | **yes** — a failing test fails the build |
| large | weekly (Sun 03:00 UTC) + manual | `(large or benchmark or regression) and <stack>` | yes |

**Chain.** `Containers` publishes the images → `Smart Tests`
(`.github/workflows/tests-report.yml`) runs both tiers and merges per-stack
HTML/JUnit into `test-reports-small-bundle` and `test-reports-large-bundle` →
`Render Webpage` downloads those bundles, runs
`docs/scripts/generate_ci_test_report.py`, and builds this page. A successful
`Containers` run re-triggers `Smart Tests` on the same SHA so `:latest` images
match the tree.

`Smart Tests` checks out with `submodules: recursive` — `library/*` are
submodules and `pip install -e` fails without them.
