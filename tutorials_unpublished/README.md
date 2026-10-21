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

## `legacy/`

`legacy/{swe,sme,amrex}/` hold the three notebooks the tutorials page used to
publish (`swe/simple_numpy`, `sme/moments_2d`, `amrex/minimal`), moved out
2026-10-21 so that the page starts clean for the new set.  They import the
symbolic solvers (`zoomy_core.symsolver`); `swe` and `sme` executed on that
day, `amrex` needs the AMReX container.  Restore one with `git mv` back to
`tutorials/<area>/`, plus the metadata below.

## Reviving one

The bar is the same as for any published notebook:

1. It executes top to bottom with no errors, on `zoomy_core` alone where
   possible (otherwise tag `requires:` accordingly) — no external binary, no
   data submodule, no machine-specific path.
2. It asserts something real (a conservation law, a convergence rate, an
   analytical solution), not just "it ran".
3. Move it to `tutorials/` and give it `metadata.zoomy` (`title`, `summary`,
   `tags` with exactly one `type:`; see the vocabulary on the tutorials page
   or `docs/scripts/generate_tutorial_index.py`).  Nothing else: the index,
   the card, the book page and the test (`tests/tutorials`) follow from it.
