# SME | VOF live coupling on the symbolic solvers (dev/smevof-live, merged 2026-10-14)

Goal: GUI sessions `coupling-live-0`, `coupling-triple-live-0` run LIVE on the symbolic
solvers (SME arm: JAX symsolver in the JAX image; VOF: OpenFOAM 13 incompressibleVoF +
`swePreciceCoupling` FO in the OpenFOAM image; preCICE across the two containers), skip
removed in `tests/backend_e2e`; old-vs-new metrics; then the thesis cases
`sme_vof_coupling`, `sme_coupling_jax` migrated (thesis worktree `~/git/wt-thsmevof`, branch
`mig/sme-vof`).

Worktrees: `~/git/wt-smevof` (superrepo + submodule worktrees zoomy_core, zoomy_jax,
zoomy_foam, zoomy_gui, zoomy_amrex, all on `dev/smevof-live`), `~/git/wt-thsmevof`.

## Facts established
- The old live sessions ran `thesis/notebooks/coupling/cases/sme_vof{,_triple}/run.py` in place
  (zoomyFoam_L1w + FO), H_R 0.10, T 4. At H_R 0.04 (Fr* 0.752, the only configuration in use)
  the old FO profile (statebc, ledger 20) loses the tank (dt collapse) at t 0.38 (Aug FO) /
  1.63 (current FO). The production profile (starstate, characteristic q*, no ledger,
  executeAtStart no; sme_vof_overshoot) runs the pair; the triple needs fluxonly +
  imposeVertical no (starstate: dt collapse / FPE at the outflow seam, t ~1.25, old solver).
- Past t ~2.5 the Fr* 0.752 run is chaotic: the OLD solver amplifies a 1e-9 dam-height
  perturbation ~1e6-fold by t 3.6 and crashes (pair at 3.66). => sessions run to t_end 2.5.
- Old zoomyFoam vs symbolic JAX, uncoupled 1-D arm, fixed dt: steps 1-2 identical, then q_1 at
  fronts differs ~1e-3 rel (the symbolic step = the core kernels evaluated by hand; the old
  foam differs from them; cause not found). This is the floor of all old-vs-new numbers.
- zoomyFoam's CFL step: maxCo 0.9 with spaceDimension 2 (its default) on the 1-D arm =>
  the symbolic arm runs cfl 0.45 (CflWaveSpeed.cells).
- The w column (derivative aux at the coupled cell): old = Gauss gradient with the ghost as the
  face value (stale aux of the step start); new = LSQ with the Coupled BC's extrapolation
  (closure class Neumann). Inert under starstate; under fluxonly + imposeVertical yes it moves
  the VOF (=> imposeVertical no for the triple). Open: wire the coupling ghost into the
  derivative aux (closure class of Coupled faces).

## Done
- zoomy_core 719172c, 36925a8: `CoupledFace` (shared | characteristic), frozen depth row,
  vocabulary blocks characteristic_ghost / ghost_flux / replace_row, effects
  couple.window_complete / read_round_trip; `systemmodel.column.ColumnMap`.
- zoomy_jax 5d5dbff, c24ef07: ColumnPreciceAdapter, the blocks, checkpoint queries (multi),
  `symsolver.coupling.column_participant`.
- zoomy_foam 5b955ae: `zoomy_foam.vof` (tank case, preCICE config, FO build cached by hash).
- zoomy_server: `peers.py` (ZOOMY_PEERS, ZOOMY_COUPLE_DIR, Peer, compose).
- zoomy_gui 0745880, 3d20fa5: sessions repacked (`pack_coupling_live_sessions.py`, cards
  `tools/symsolver/coupling_live_*.py`), old-vs-new reference runs + figures
  (`tools/symsolver/figures/coupling_live_*`).
- tests/backend_e2e: shared couple dir + peers registry, PEERS / NO_HOST_RUN, skip removed.
  `-k live`: 2 passed (pair 120 s, triple 76 s), GHCR images 2026-10-14, dev mode.
- Old vs new (live runs, t <= 2.5): pair interface h rel L2 8.6e-5; triple arm Swe 3.9e-4,
  arm Swe2 5.4e-3 (old's own 1e-9 floor 3.5e-3). C16 jax_jax shared path unchanged (3.6e-15).

- zoomy_core 81beb82, zoomy_jax 751eb3f: Coupled faces are ghost rows of the derivative fits; a
  characteristic coupling writes its column with the step's aux, derivatives seeing the
  coupling's ghost (as zoomyFoam). zoomy_foam c93b53a: FO build moved in whole (concurrency).
  e2e `-k live` rerun on these: 2 passed, metrics unchanged.
- thesis (wt-thsmevof, mig/sme-vof): 83b79774 sme_coupling_jax (symbolic JAX over preCICE; new vs
  old 4.6e-7 h, cause: Gelfand wave speed vs eig, eig gives 5e-16); de8e198e sme_vof_coupling
  migration (Fr* 0.752 SME(0): new = old to 6e-11; existing statebc config: 2.5e-3, the w column
  under cellLimited Gauss vs LSQ).

## Open
- Old zoomyFoam vs the symbolic kernels: q_1 rows differ ~1e-3 at fronts (SME(1) only; SME(0)
  agrees to output precision). Not located (flux, fluctuation, source and assembly read the same).
- Fr* 0.752 past t ~2.5 is chaotic for the old solver too; the triple needs fluxonly at the
  outflow seam.
- w column: zoomyFoam limits its Gauss gradient (cellLimited); the symbolic fit does not.
