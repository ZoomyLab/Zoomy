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
| thesis     | `thesis` (except more-specific sub-paths below)      | main    | unclaimed | —         |

Add a finer row when a sub-path needs its own steward, e.g.
`| thesis-applications | thesis/chapters/40_applications | main | … |`.

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
