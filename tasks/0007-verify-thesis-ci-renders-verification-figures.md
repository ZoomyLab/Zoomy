# 0007 — Confirm thesis CI renders the new Verification section + figures

**What:** verify the built thesis site/PDF renders the
`chapters/35_validation.md` → "Convergence of the moment hierarchy" section with
its two embedded figures.

**Where:** `thesis/chapters/35_validation.md` (the new subsection) references
`../cases/hoern/bbsm13_nobackflow_figure.png` and `../cases/hoern/bbsm13_recirc_figure.png`;
thesis CI builds on push (commits `ee96ab8`, `716c348` on `main`).

**How:** check the CI-built page / PDF for the section + both figures. If the
cross-dir `../cases/hoern/` relative path does not resolve in the MyST build, fix
it (either correct the path or copy the figures into `chapters/figures/` and
re-point) — a one-line edit, no solver re-run.

**Why:** the figures are referenced across directories; MyST relative-path
resolution from `chapters/` into `cases/` may need adjusting for the build.

**Learned:** other chapters use cross-dir refs successfully (e.g. `90_functions.md`
uses `../gui/previews/`), so `../cases/hoern/` should resolve — but it was not
build-verified in this thread. Figure embed convention is `:::{figure} <relpath>`
with `:label:` / `:width:` (see `10_intro.md`).
