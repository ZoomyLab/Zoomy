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

## Milestones (all reached)

1. JAX, VAM(1) | VAM(1) 1-D — gate 54 (`gates/gate54_chorin_precice.py`), C10's wave 30 | 30
   cells, dt = window 2^-7, t 0.25.  max|coupled - single| 6.9e-16 (standard projection), 4.7e-16
   (increment), |mass coupled - single| 0; serial-implicit + IQN-ILS on P: 6-7 iterations per
   window (no acceleration: 19-23, independent of W: the subdomain solve covers the own cells only).
   W = 1: 1.4e-3 (refused now), W = 2, 3, 6: round-off.  core 3791df0, jax b3e2375; superrepo 23cf829d.
2. NumPy 6.3e-16 (core 397d74e; test `tests/symsolver/test_overlap_schwarz.py`: the pair in
   process, two threads, in-memory parallel-implicit coupling, = single < 1e-12); foam 9.8e-16 and
   AMReX 2.8e-16 (mass 1.8e-16 each; core a4e2304: rt_common.H effects, foam 5a21240, amrex
   1c266fa: overlap blocks + setup; the AMReX run binds the preCICE config's directory).
3. Matrix C16: rows vam_jax, vam_jax_increment, vam_numpy, vam_foam, vam_amrex (gate 54, serial):
   0 FAIL; the hyperbolic pairs unchanged.

## Added after the milestones

- SSP-RK projection stages (time_order 2 / 3): each stage's pressure exchanged as further rows,
  per-stage Dirichlet data; `ChorinTemplate.overlap_rings()` = 2 per stage + 1 (measured exact in
  every case; one less is off by 1e-13 .. 5e-8 in some), enforced by every driver; order 2
  refused (rings not established).  Gate 54 `GATE54_TIME_ORDER=2`, W 5: JAX 6.1e-16, NumPy
  6.0e-16, foam 8.7e-16, AMReX 3.1e-16.
- 2-D (C10's quads, ny 2, `GATE54_DIM=2`), W 3: JAX 8.5e-16, NumPy 1.3e-15, foam 1.35e-15, AMReX
  4.0e-16.
- foam build cache: locked build + atomic rename (two participants of one binary raced).
- C16 rows vam_jax_t2, vam_jax_2d, vam_foam_2d, vam_amrex_2d: 0 FAIL.

## Remaining (not built)

- IMEX-SSP2's projected stages (three projections per step) and reconstruction order 2.
- Several MPI ranks per participant (refused in every backend): the overlap cells may sit on
  several ranks (JAX: `_coupling` ordering + the distributed linsolve's mask; foam / AMReX: the
  vertex lists per rank).
- 3-D and unstructured overlaps over preCICE untested (the rings and the AMReX slabs are general).
- A true overlap (solve the inner overlap rings too) to cut the iterations without acceleration.
- MOOD and an implicit diffusion stay refused (a global decision / a second global solve).

## Verification

- C16 (all pairs, serial + mpi2, scratch store `matrix/work/store_chorincpl`): the hyperbolic pairs
  identical to table.md (jax_jax 3.61e-15, jax_foam 4.94e-15, foam_foam 5.16e-15, amrex_amrex
  3.55e-15, amrex_jax 3.33e-15, mass 0), rerun after the C++ header change; VAM rows as above.
- Final state (core 49fe88a, jax c55086a, foam dbd041a, amrex da325be): zoomy_core default tier
  399 passed (297 + 102, run in two parts), QR 27 passed / 3 skipped, zoomy_jax
  test_lifted_shared_face 3 passed; C16 0 FAIL (all vam_* rows rerun on this round's code).
- Gates' work dirs: set `GATE_WORK` / `ZOOMY_GATE_WORK` to a private directory when another
  session runs gates 10 / 26 / 38 (their default `/tmp/...` paths are shared: one C16 amrex_amrex
  run timed out on a collision here).
