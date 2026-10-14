# dev/numpy-mpi (wt-npmpi) -- status

MPI for the NumPy backend (mpi4py; PETSc linear solves): ANALYSIS 54.  Merged integ/20261014
(superrepo d127e36c, core f1708af, jax 2815437): the solver-combos `implicit_solve` (IMEX-SSP2 +
implicit diffusion) is distributed too (owned rows, halo off for the FD Jacobian, parallel LU).

Stores: ~/git/wt-npmpi-matrix/store_integ_base (numpy serial on integ's own libraries, the
baseline), store_integ (the merged branch: numpy serial + mpi2 / mpi4, JAX for the compared
cases).  Previous store (pre-merge): ~/git/wt-npmpi-matrix/store.

Verified after the merge: numpy serial 288 runs bitwise = integ baseline; numpy mpi2 / mpi4
(C1 C5 C8 C13 C14 C21 C26, C35 / C37 mpi2) + JAX serial (those cases, C32, C37) + JAX C37 mpi2:
partial table 342 PASS / 0 FAIL (~/git/wt-npmpi-matrix/table_integ_partial.md; table.md kept
as integ's: the store holds only these backends / cases).  core 457 passed + the 2 known
failures (all tiers); QR 27 passed, 3 skipped.  ANALYSIS 54 (renumbered).

---

# dev/solver-combos (wt-features) — status

Worktrees: superrepo + zoomy_core, zoomy_jax, zoomy_foam, zoomy_amrex on `dev/solver-combos`
(not pushed, not merged).  Details and measurements: ANALYSIS.md section 49.

## 1. IMEX-SSP2 + implicit diffusion — done

- core f5b690e, jax 162e785, foam c6bb3b0, amrex a3dec89; superrepo 873fd401 (gate 50, C37).
- `implicit_stage` (residual.py): X - Y - A_ii dt (D(X) + S(X)) = 0 by one Newton over the mesh
  (`implicit_solve`), D the two-point part (`diffusion_tpfa`), the rest of the flux
  (`diffusion_correction`) explicit; Chorin's projected stages take it too.
- gate 50 (JAX, NumPy): 0 FAIL; matrix C37 (heat, advection-diffusion vs the matrix-form scheme,
  shear decay, VAM Stress.full Chorin; foam / AMReX = JAX <= 1.9e-14; mpi2 = serial): all PASS.

## 2. AMR + diffusion (AMReX) — done

- core 129e32e, amrex c49e2ef; superrepo 0c165125 (gate 51, C17 `diff_*`).
- Diffusive face fluxes refluxed (stage log entries; Crank-Nicolson by share of dt in `dlog`);
  the levels' linear solves get their mesh; AMR `dt_of` honours a constant dt.
- gate 51 / C17: covering level = fine single level (bitwise / 6e-16), patch mass 2e-16 with
  reflux (3e-4 without), L1(AMR - fine) / L1(coarse - fine) 0.37 (explicit), 0.59 (implicit:
  O(dt) halo lag, 0.38 at dt / 8).

## Verification

- Matrix (scratch store = main's + C17, C26, C28, C31-C34, C36 rerun, C37 new; no GPU): touched
  cases 478 rows, 0 FAIL except C17 `dry` (NaN, pre-existing: main's libraries give the identical
  NaN; main's store had no `dry` run).  table.md not committed (mixed-provenance store).
- zoomy_core 362 passed; QR 27 passed, 3 skipped.

## 3. preCICE + Chorin — not implemented (multi-day): plan

**Where it is refused.**  `zoomy_core/symsolver/strategies/chorin.py`: `ChorinTemplate.precice_march`
raises `NotImplementedError("Chorin with preCICE coupling")`; every backend prints the march from
the template (JAX `SymbolicChorinSolverJax(couple=...)`, foam / AMReX `CxxChorinBackend` with
`couple`), so all refuse there.  No matrix case covers it (C16: SME only).

**Why the hyperbolic coupling does not carry over.**  `HyperbolicTemplate.precice_march` exchanges
the O1 states next to the coupled faces once per window; both sides solve the same Riemann problem
(`interface_flux`), so at O1 with one step per window the coupled run is the single domain to
round-off (C16).  A Chorin step also needs (a) the predictor's derivative aux (LSQ stencils of
radius R reach R cells across the interface: VAM's h_x, b_x, ...), (b) one elliptic pressure solve
coupling every cell of both domains (LSQ second-derivative stencils across the interface), (c) the
corrector's derivative aux.  No single exchange of interface states reproduces (b).

**Design: overlapping Schwarz inside a preCICE implicit coupling window.**
1. Overlap: each participant's mesh gets R_o = R + 1 layers of the peer's cells beyond the coupled
   face (a new boundary role, filled like a distributed halo: `zoomy_core.mesh.distributed` HALO
   cells, where the other "rank" is reached over preCICE).  Coupling mesh = the overlap cell
   centres (read) and the own mirror layers (write).  One window = one Chorin step (constant dt,
   MOOD off: the stage sequence must be data-independent).
2. Window loop = preCICE implicit coupling (`serial-implicit`, IQN-ILS acceleration, relative
   convergence 1e-12 on the exchanged data), with checkpoints: on
   `requires_writing_checkpoint` save (Q, Qaux; with `Projection.increment` the lagged pressure is
   in Q), on `requires_reading_checkpoint` restore.  Iteration k: the whole step (predictor,
   pressure, correction) with every overlap read from iteration k - 1's data, the pressure solve
   local with the peer's pressure as Dirichlet data in the overlap (Schwarz subdomain solve); at
   the fixed point every stencil reads the peer's converged values: the single-domain step to the
   coupling tolerance.
3. Stage data: the states the template halos in a step (predictor input, predictor output, the
   pressure iterate after the solve, the corrected state; IMEX-SSP2 / time_order 2: per stage),
   one preCICE data field per (stage, row); the template numbers the halo points of the step
   (`overlap(X, k, mesh)` block: fill stage k's overlap from the read data; `overlap_states(X, k,
   mesh)`: the own layers to write).  Krylov iterations inside `linsolve` never exchange (the
   outer Schwarz iteration does).
4. Template: `ChorinTemplate.precice_march` = a While over windows, inside a While over coupling
   iterations; new effects `couple.checkpoint()` -> (write?, read?) and `couple.converged()`;
   `couple.read/write` take the stage-indexed data.  JAX: `PreciceAdapter` (+ checkpoint flags,
   stage data), the overlap fill in `runtime.py` next to `_coupling`; foam: the overlap as a
   cellZone set from data (OpenFOAM has no ghost cells on a patch); AMReX: ghost cells beyond the
   coupled side filled from data (`couple_setup`, `Interface`).
5. Test (gate 52, matrix C16 `vam_vam`): C10's wave, VAM(1), O1, 60 cells split 30 | 30, constant
   dt = window: coupled vs single domain <= 1e-10 (coupling tol 1e-12, pressure rtol 1e-10), mass
   drift <= 1e-12, coupling iterations per window reported; JAX first, then foam / AMReX, then
   mpi2 participants.
6. Risks: Schwarz convergence of the non-hydrostatic pressure (a Helmholtz-type operator with the
   depth as length scale; overlap R + 1 cells; IQN needed); the volume of stage data; JAX Chorin
   setup ~130 s per participant (the coupled gate needs both alive in one window budget).

Estimate: 3-4 days (template + JAX + gate: 2; foam / AMReX overlap: 1-2).
