# Branch status notes of the 2026-10-14 integration (dev/solver-combos, dev/numpy-mpi, dev/oblique-walls)

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

---

# dev/oblique-walls (wt-oblique) — status

Merged integ/20261014 (not pushed, not merged back).  ANALYSIS 57, gate 57, matrix C38.
- Oblique walls (closure in the walls' normal frames): core 8cb9ff2 4f2d7a4, jax 43c1db9, foam d215a44.
- foam O2 = the template's gradient / limiters (the shore form of ANALYSIS 50 in them): core
  9f739cf, foam a37c9c8; merge: core bc90274, jax a8eb5d4, foam d5cbe75, amrex 514d9e0.
- Checks after the merge: core 414 pass; gates 39/42/44/57 OK; QR 28 / 2 skipped; matrix
  C2 C3 C5 C6 C12 C19 C26 C32 C37 C38 (jax, foam; serial, mpi2) 0 FAIL; pit lake foam serial /
  mpi2 <= 1.1e-14.

# dev/solver-combos (wt-features) — status

---

# dev/amrex-amr-gpu (wt-amrexfix) — status

Worktrees: superrepo + zoomy_amrex, zoomy_core on `dev/amrex-amr-gpu` (from integ/20261014);
zoomy_jax, zoomy_foam detached at integ's commits (unchanged).  Not pushed, not merged.
Details: ANALYSIS.md section 55.  Matrix store: main's store copied, the touched cases rerun
(`docs/dev/symbolic_solver/matrix/store`, untracked).  Image: `containers/zoomy_amrex/zoomy_amrex.sif`
of this worktree = the published image + libnvjitlink-12-6 (untracked; the recipes are fixed).

## 1. C17 `dry` — done (core 8f56c4c, amrex 55a6b3a)
- The reflux made h < 0: the fine level's stage-2 halo held the coarse state after its whole step.
- Composite stages: all levels advance stage by stage, reflux + average-down per stage; MOOD on
  the composite.  dry: mass 0, h_min 0.  Lie's interface lag gone as a by-product (0.59 -> 0.38).
## 2. C17 long-run hang — done (core 63cdf15, amrex a100cf2)
- h refluxed, q not -> garbage velocity in thin cells -> dt collapse (base 64: 9556 steps vs 386).
- amr_keep: a refluxed cell keeps its velocity.  New C17 row `dry_long` (0.94 of the fine steps).
## 3. GPU — done (amrex 9c133a5; superrepo: image recipes)
- 3 boxes / 2 GPUs: no-sync march raced AMReX's unfused per-box streams -> garbage pressure rows,
  NaN FlexGMRES.  One GPU stream.  CUDA_VISIBLE_DEVICES passed into the image.  Image lacked
  libnvJitLink (no hypre-CUDA link).  hypre GPU smoother = CPU's: 38 -> 20 iterations, but
  5-10x slower per solve (open: Chebyshev 23 its at 0.8x).
## 4. AMR implicit time lag — Lie fixed by (1); IMEX-SSP2 analysed, plan in ANALYSIS 55d

---

# dev/jax-chorin-perf — STATUS

Task: symbolic JAX Chorin ML-VAM(2) on GPU ~1 s/step vs legacy 0.1 s/step (thesis
escalante_vam_bump/run_mlvam2_jax_jit.py, RIEMANN=hr ORDER=2 TORDER=2, dt 1e-3, 60 cells).
Also: cells-CFL eig on GPU slow.

Worktrees: superrepo ~/git/wt-jaxperf (branch dev/jax-chorin-perf from main a2ba0f57);
library/zoomy_jax (468fd7f), library/zoomy_core (9308483) on dev/jax-chorin-perf.
Bench scripts: scratchpad bench/ (bench.py, case copy).

## Reproduce (GPU 0, L40S, taskset 32-39)
- case copy T_END=3 (3000 steps): 671 s total = build 20 + setup 115 + compile 270-290 + steps.
- warm step 0.088 s/step (t 0..3, flat); with on_write every 100 steps the same.
  The "~1 s/step" of step4 does not reproduce: per step = legacy (0.1); total wall is
  dominated by one-time setup + XLA compile (not persistent-cached: host callbacks).
- one compile of the march per run (jax_log_compiles), no per-step recompiles.

## Profile (nsys, 50 warm steps)
- cuSOLVER Xgeev (jnp.linalg.eigvals on GPU) for the cells CFL check (face_wavespeed on
  eigenvalues_cfl, 2 x 61 14x14 matrices per step, computed every step, used only by the
  every-10-steps check at constant dt): 17000 kernel launches, 2300 D2H copies, 3700
  syncs per step, ~70 ms of the 88 ms step.
- AmgX (FGMRES + aggregation AMG, 2 solves/step): ~1800 launches, 450 copies, 640 syncs.

## Done
- [x] zoomy_jax 828a818: device eig (LAPACK dgeev path, vmapped XLA); face wave speeds of both
  sides in one batch.  0.088 -> 0.021 s/step; case t=3 bitwise identical.
- [x] zoomy_core ae456d1 LinearSolver.dense; zoomy_jax 9539c38 dense LU pressure solve (auto on a
  GPU, serial, <= pressure_dense_max=1024 unknowns).  0.021 -> 0.0082 s/step; case t=3 vs AmgX:
  final rel L2 3.6e-12 (frames max 5.9e-11).
- [x] zoomy_jax ccd96b8: eig as one Triton kernel (Pallas) on CUDA.  0.0082 -> 0.0046 s/step;
  case bitwise = previous.  d3e7134 test under vmap.
- [x] legacy run_jit_steps today (base libs): 0.21 s/step.
- [x] ANALYSIS.md section 59 (superrepo, uncommitted until the matrix numbers are in).

## Checks (done)
- case t=3 frames: eig changes bitwise; dense vs AmgX final rel L2 3.6e-12 (frames 5.9e-11).
- matrix (store ~/git/wt-jaxperf-matrix): C8 C10 C31 jax serial + C8 gpu1: 12 PASS 0 FAIL;
  C8 serial = main bitwise; gpu1 vs main gpu1 2.9e-12; gpu1 dense vs serial 1.3e-11.
- zoomy_core 375 pass; zoomy_jax 26 failures = main's 26 (same ids); QR 27 pass / 3 skip
  (library/zoomy_gui, meshes, zoomy_amrex: detached worktrees at the recorded commits).

## Open
- compile 270-300 s per run, never persistent-cached (host callbacks in the march).
- setup 115-140 s (sympy, Riemann numerics build).
- pressure LU 1.1 ms/step ignores the banded structure; cells speeds computed every step at
  constant dt (template, integ's area).
