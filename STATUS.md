# dev/precice-chorin (wt-chorincpl) — status

Worktrees: superrepo + zoomy_core, zoomy_jax (changed), zoomy_foam, zoomy_amrex (unchanged, for the
C16 runs) on `dev/precice-chorin` from `integ/20261014`; zoomy_gui detached at integ (QR suite).
Not pushed, not merged.  Details and measurements: ANALYSIS.md section 56.

## preCICE + Chorin: overlapping Schwarz inside an implicit coupling window

**Design (as built).**  `ChorinTemplate.precice_march`: each participant's mesh reaches W cell
rings across the interface (the overlap, `zoomy_core.mesh.overlap.overlap_layers`: W rings from
the boundary tag `couple['tag']`); one step per window of an implicit preCICE scheme.  Each
coupling iteration runs the whole step from the window start (the template keeps the window start:
the checkpoint), the overlap holding the peer's state at the window start (`couple.read`, relative
time 0): predictor and aux run over the overlap too (redundant; the outer rings are invalid, W must
cover the own cells' stencils: W >= 2 for VAM(1) O1), the pressure solve keeps the overlap's
pressure at the peer's last iterate (`couple.read_iterate`, the window end: identity rows, a
Dirichlet subdomain solve, `pressure(dirichlet=)`), the participant writes all rows of its mirror
cells; `couple.repeat` after `advance` restarts the window.  At the fixed point every own-cell
equation reads the peer's values: the single-domain step to the coupling / pressure tolerances.
Exchanged data: the n_state rows of the mirror cells (as the hyperbolic full-state exchange, on W
rings), no stage data.

## Milestones

1. JAX, VAM(1) | VAM(1) 1-D — done: gate 54 (`gates/gate54_chorin_precice.py`), C10's wave 30 | 30
   cells, dt = window 2^-7, t 0.25.  max|coupled - single| 6.9e-16 (standard projection), 4.7e-16
   (increment), |mass coupled - single| 0; serial-implicit + IQN-ILS on P: 6-7 iterations per
   window (no acceleration: 19-23, independent of W: the subdomain solve covers the own cells only).
   W = 1: 1.4e-3 (refused now), W = 2, 3, 6: round-off.
2. NumPy — see below.  foam / AMReX: see below.
3. Matrix C16 — see below.

## Verification

- C16 (all pairs, serial + mpi2, scratch store): identical to table.md (jax_jax 3.61e-15, jax_foam
  4.94e-15, foam_foam 5.16e-15, amrex_amrex 3.55e-15, amrex_jax 3.33e-15, mass 0).
- zoomy_core 397 passed (default tier); QR 27 passed, 3 skipped; zoomy_jax
  test_lifted_shared_face 3 passed.
