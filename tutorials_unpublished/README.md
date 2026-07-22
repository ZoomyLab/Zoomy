# Unpublished tutorials

Notebooks kept for reference but **not built into the documentation site**.
Moved here 2026-07-22 so that `tutorials/` contains exactly the published set
and nothing can reach the site without being executed first.

Why they are not published: most were written against APIs that have since been
renamed (`zoomy_core.model.models.shallow_moments`,
`zoomy_core.model.models.system_model`), several depend on the external SWASHES
binary, and the two AMReX ones hardcode an absolute path from a developer's
machine. They are stale, not wrong-in-principle — several are worth reviving.

## Structure

Mirrors the old `tutorials/` layout exactly, so a path in an old comment or
commit message maps by prefix:

    tutorials/sme/simple.ipynb  ->  tutorials_unpublished/sme/simple.ipynb

## Reviving one

The bar is the same as for any published notebook:

1. It executes top to bottom with no errors, on `zoomy_core` alone where
   possible — no external binary, no data submodule, no machine-specific path.
2. It asserts something real (a conservation law, a convergence rate, an
   analytical solution), not just "it ran".
3. Add it to **both** `PUBLISHED_TUTORIALS` in
   `docs/scripts/generate_ci_test_report.py` and
   `tests/notebooks/smoke_notebooks.txt` — these two lists must agree, and the
   second is what CI executes weekly.
4. Add it to `docs/book/_toc.yml`.

See `tutorials/swe/advanced_numpy.ipynb` for the current house style.
