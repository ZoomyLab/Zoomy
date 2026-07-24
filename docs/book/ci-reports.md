# CI Test Reports

Every push runs the **small** suite as a gate; the **large / regression** suite
runs weekly. The results below cover three groups:

- **Zoomy** — the superproject's own `tests/`, run per backend.
- **Library** — each `library/zoomy_*` submodule's own `tests/`.
- **Notebooks** — the published tutorial notebooks, executed end to end.

## Summary

```{include} ../_generated/test_report_summary.md
```

A ⚠️ marks a suite with failures. A dash means that suite produced no report in
the latest run (a stub is shown in its section below).

---

## Detailed HTML reports

Embedded **pytest-html** reports. Scroll horizontally on narrow viewports.

```{eval-rst}
.. raw:: html

   <style>
   .pytest-report-scroll { margin:1rem 0; overflow-x:auto; width:100%;
     -webkit-overflow-scrolling:touch; border:1px solid #e0e0e0; border-radius:6px; background:#fff; }
   .pytest-report-scroll iframe { display:block; border:0; height:960px;
     width:min(100%, 1600px); min-width:min(100%, 920px); }
   </style>
```

```{include} ../_generated/test_report_embeds.md
```

---

## How to use this page

### Local build with real reports

From the **repository root**:

```bash
# one suite (repeat per group/unit you care about)
python tests/reporting/generate_test_report.py \
  --markers "small and core" --pytest-paths "tests" \
  --output-dir artifacts/test-reports/small/zoomy/core
# then assemble + build
python docs/scripts/generate_ci_test_report.py
jupyter-book build docs/book
```

`docs/scripts/generate_ci_test_report.py` reads
`artifacts/test-reports/<tier>/<group>/<unit>/**` and writes the summary
(`docs/_generated/test_report_summary.md`), the embeds
(`docs/_generated/test_report_embeds.md`), the per-unit
`docs/book/_static/pytest-report-<tier>-<group>-<unit>.html`, and mirrors the
published tutorials into `docs/book/tutorials/ipynb/`. A stub is written wherever
a run produced no `report.html`.

### Adding or changing tests

Markers are mostly **automatic** — the root `conftest.py` applies them:

- A test under `library/zoomy_<stack>/tests` (or `tests/**/zoomy_<stack>/`) gets
  the `<stack>` marker from its path (`foam` → `openfoam`).
- A test with no size marker becomes `small`; anything taking more than
  **5 minutes individually** must be marked `large` explicitly.
- `library/zoomy_<stack>/tests` is collected only when `import zoomy_<stack>`
  resolves, so each container runs exactly the suites it can.

A **tutorial notebook** joins the Notebooks group by being added to
`tests/notebooks/test_tutorials.py::PUBLISHED` (path + the backend marker it
needs) and to `PUBLISHED_TUTORIALS` in the docs generator — it is then executed
as a `@pytest.mark.notebook` test and mirrored onto the site.

### Docs-only changes

Pushing documentation alone still rebuilds the site; **Render Webpage**
re-downloads the latest Smart Tests artifacts from **`main`** when it runs.

---

## Architecture

**Scope.** CI runs three groups across four backend containers (GHCR images with
the checked-out tree `pip install -e`'d over them, so reports match the commit):

| Group | What runs | Where |
|---|---|---|
| Zoomy | superproject `tests/`, filtered per backend | core / jax / amrex / foam images |
| Library | each `library/zoomy_*/tests/` | its own backend image (`prepost`/`server` in the core image) |
| Notebooks | `tests/notebooks/` (`@pytest.mark.notebook`) | core (+ amrex) images |

**Two tiers.**

| Tier | Runs on | Markers | Gates? |
|---|---|---|---|
| small | every push and PR | `small and not regression and …` | **yes** — a failing test fails the build |
| large | weekly (Sun 03:00 UTC) + manual | `(large or benchmark or regression) and …` | yes |

**Chain.** `Containers` publishes the images → `Smart Tests`
(`.github/workflows/tests-report.yml`) runs both tiers and merges the per-unit
HTML/JUnit into `test-reports-small-bundle` and `test-reports-large-bundle` →
`Render Webpage` downloads those bundles, runs
`docs/scripts/generate_ci_test_report.py`, and builds this page. `Smart Tests`
checks out with `submodules: recursive` — `library/*` are submodules and
`pip install -e` fails without them.
