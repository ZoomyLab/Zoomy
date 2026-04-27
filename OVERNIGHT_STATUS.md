# Overnight session — status note

Three background agents are running in parallel, each in its own git worktree
on a feature branch. They commit + push to their own branch. Expect 3 fresh
branches in `origin` tomorrow.

## Tasks delegated

| # | Branch | Goal | Time budget |
|---|--------|------|-------------|
| 1 | `numerical-regularization-comparison` | Numerically simulate SME L=2 with full SME, our minimum-entry regularisation, K&T-style drop-feedback, and Newtonian τ_xx. Compare time evolution, blowup, max wave speeds. | ~4 h |
| 2 | `generic-runtime-bridge` | Generic bridge: `PDESystem` (transparent derivation) → `RuntimeModel` (existing FVM solver). Verify VAM(1,2) and multi-layer SWE work via the bridge. | ~5–6 h |
| 3 | `dae-solver-direct-vam` | New `DAESolver(IMEXSolver)` that integrates VAM directly without Chorin pressure splitting. Stress + source remain implicit; the new piece is constraint-aware Newton/GMRES. | ~5–6 h |

## What's already on `migrate-tutorials-transparent-physical-z`

The current branch has:

- All hyperbolicity-analysis scripts in `tutorials/sme/sme_l2_*.py` and `tutorials/sme/sme_l3_*.py`.
- The notes file `tutorials/sme/sme_l2_regularization_findings.md`.
- The notebook `notebooks/006_sme_hyperbolicity.{py,ipynb}` covering route A vs route B, block structure, the single-entry minimum regularisation, the analytic eigenstructure at ū_2 = 0, viscous τ_xx, L=3 mode pattern, and the eigenvalues-vs-proof discussion.
- `tutorials/vam/vam_l1_physical_z.py` (the transparent VAM L=1 derivation).
- Submodule `library/zoomy_core` bumped to `c7a47ce` (mechanical Galerkin projection + symbolic primitives).

PR for this branch: https://github.com/ZoomyLab/Zoomy/pull/3

## Expected morning state

When the agents finish:

- Three new branches on `origin`.
- Each branch has its own commits, notebook(s), and tests.
- Each agent will leave a STATUS.md or a final commit message describing what worked, what didn't, and recommended next steps.
- If an agent hit a hard blocker, it will stop early and report — preferring honesty over fake success.

If anything didn't get done, the relevant tasks in the issue tracker should
still be marked `in_progress` (not `completed`). Inspect each agent's final
report (last commit message + STATUS.md if present).

## Constraints all agents are respecting

- TDD: small standalone test before/after each `library/zoomy_core` change.
- No `Co-Authored-By: Claude` trailer in commits (per `CLAUDE.md`).
- Don't force-push, don't skip hooks.
- Generality preferred over hard-coded logic.
- No breaking of existing solver paths.

— left running 2026-04-27 evening
