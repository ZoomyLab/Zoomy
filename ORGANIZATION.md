# Zoomy — Organization

Live coordination state for the **steward** agents. Rules: [STEWARD.md](STEWARD.md).
Each steward owns ONE path below and edits only it; a named commit is the "done"
signal. Start a steward with `/steward <path>`.

## Ownership

Most-specific path wins. Set `status`/`last seen` when you adopt a row; **remove
your row when you no longer need the path**. `unclaimed` = free to adopt.

| agent      | owns (path)                                          | branch  | status    | last seen |
|------------|------------------------------------------------------|---------|-----------|-----------|
| root       | `/` — superrepo: submodule pointers, top-level dirs  | develop | unclaimed | —         |
| core       | `library/zoomy_core`                                 | main    | unclaimed | —         |
| jax        | `library/zoomy_jax`                                  | main    | unclaimed | —         |
| foam       | `library/zoomy_foam`                                 | main    | unclaimed | —         |
| firedrake  | `library/zoomy_firedrake`                            | main    | unclaimed | —         |
| fenicsx    | `library/zoomy_fenicsx`                              | main    | unclaimed | —         |
| amrex      | `library/zoomy_amrex`                                | main    | unclaimed | —         |
| dmplex     | `library/zoomy_dmplex`                               | main    | unclaimed | —         |
| gui        | `library/zoomy_gui`                                  | main    | unclaimed | —         |
| thesis     | `thesis` (except more-specific sub-paths below)      | main    | active    | 2026-06-19 |

Add a finer row when a sub-path needs its own steward, e.g.
`| thesis-applications | thesis/chapters/40_applications | main | … |`.

## Tasks

The backlog. **Detail lives in [`tasks/`](tasks/)** (one file per task: what ·
where · how · why · learned); this index only maps each open task to the
responsible agent(s) (by folder ownership) and links to its file. When you take
new ownership, claim any task in your path here (STEWARD §1.5). Finishing a task
= delete its `tasks/` file **and** its row below, in the same commit.

| task | description                                  | responsible        | detail |
|------|----------------------------------------------|--------------------|--------|
| 0001 | remove stray root `memory.md`                | root               | [tasks/0001-remove-stray-memory-md.md](tasks/0001-remove-stray-memory-md.md) |
| 0002 | reconcile dirty submodules + untracked tutorials | core, jax, firedrake, amrex, dmplex, root | [tasks/0002-reconcile-dirty-submodules-and-tutorials.md](tasks/0002-reconcile-dirty-submodules-and-tutorials.md) |
| 0004 | document CLI/GUI card contract in the docs (optional) | root | [tasks/0004-document-cli-gui-card-contract.md](tasks/0004-document-cli-gui-card-contract.md) |
| 0009 | VAM Chorin-split predictor drops the non-x conservative flux (dim≥3 mass leak / over-fill) | core | [tasks/0009-vam-split-predictor-nd-flux-misclassification.md](tasks/0009-vam-split-predictor-nd-flux-misclassification.md) |
| 0010 | re-assess JAX VAM pressure-BC change (`56eff9a`) after the split fix | jax | [tasks/0010-assess-vam-pressure-bc-jax-after-split-fix.md](tasks/0010-assess-vam-pressure-bc-jax-after-split-fix.md) |
| 0011 | re-run VAM steffler bend + point-data BC interpolation for plots (blocked on 0009) | core, jax | [tasks/0011-rerun-vam-steffler-secondary-circulation-and-pointdata-bc.md](tasks/0011-rerun-vam-steffler-secondary-circulation-and-pointdata-bc.md) |

## Requests

Append-only. One block per request; a named commit closes it.
Format: `### REQ-<n>  <from> → <to>  ·  <open|resolved>`

<!-- TEMPLATE — copy below, delete this comment once real requests exist:
### REQ-01  jax → core  ·  open
- Problem: JAX FVM needs the flux Jacobian; `Model._dF` is private.
- End-state: public `model.flux_jacobian(q)`, importable from `zoomy_core`.
- Test: `pytest library/zoomy_core/tests/test_flux_jacobian.py` green.
- core: taking it. (19:20)
- core: done — zoomy_core@`a1b2c3d` "expose flux_jacobian"; test green. (19:40)
- jax: pulled, verified, integrated. → resolved (19:45)
-->
