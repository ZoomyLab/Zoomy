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

---

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

---

# dev/jax-compile — status at wrap-up (2026-10-21)

Full write-up: docs/dev/symbolic_solver/ANALYSIS.md section 65.  Bench: docs/dev/symbolic_solver/cases/jax_compile_time.py.

## Measured (jax 0.9.2, GPU 1 L40S, taskset 24-31)
- Escalante ML-VAM(2) march (GPU): trace 12.5 s, lower 3.6 s, XLA compile 300-303 s; 113.7k jaxpr eqns,
  75.8k StableHLO, 163k optimised HLO.  CPU: compile 33.6 s (191k HLO).  C8 vam GPU 51.8 s (27.9k HLO), CPU 6.8 s; C1 CPU 2.7 s.
- XLA GPU phases (Escalante): HLO passes 241 s, of them priority-fusion 220 s (serial;
  --xla_gpu_force_compilation_parallelism=8: 215 s, no gain), IR emission 24 s, PTX 8 s, ptxas 9 s.
- Expected for the size: ~1.8 ms / HLO instr on GPU; legacy run_jit_steps same case: 228 s, 88k HLO, priority-fusion 148 s.
- Size: 55 % lambdified sympy kernels inlined per call site (edge source 26.9k x6 calls, face kernel 22.4k x6,
  pressure slot ops 17.9k).  Constants not the issue (1991 hoisted args, 155 kB; folding off).
- Callbacks: every Python callback kind (debug.callback ordered/unordered, io_callback, pure_callback) blocks jax's
  cache write.  In the march: check.cfl, check.march (always), io.write, io.save/load_state, accept gates, pyamg/hypre
  pure_callback, preCICE.  Key identical across processes; setup's small programs do hit.
- Setup 117-140 s = sympy subs in Numerics._model_eval (99 s; 155 calls, 56 distinct).

## Changes (committed)
1. zoomy_core riemann_solvers._substitute: xreplace where provably the same tree as subs (guards); 609 calls checked == subs.
   Escalante setup 120 -> 20.7 s; C8 31 -> 18 s.
2. zoomy_jax symsolver/compile_cache.py + hoisted_jit: executables with host callbacks cached in
   jax_compilation_cache_dir (zoomy-cb-<jax key>), loaded via deserialize_executable(host_callbacks=).
   Escalante 2nd run: compile 303 -> 12.5 s (load), run 323 -> 24 s; bitwise vs compiling run; on_write frames same.
   Test: library/zoomy_jax/tests/symsolver/test_compile_cache.py (2 pass).
## Acceptance done
- Matrix C1/C8/C10 jax serial bitwise vs main (cache written and loaded); gpu1 C1 1.4e-13, C8 1.4e-11 (GPU round-off;
  two GPU compiles of the same program differ 3.8e-15; base-vs-base gpu1 not rerun).
## NOT done
- zoomy_core pytest, zoomy_jax full suite (vs main failures), QR suite.
- Program-size reduction (template: one kernel batch for O2 + MOOD fallback; stages as a loop) — proposal only.
- Caching is active only with JAX_COMPILATION_CACHE_DIR set; no default dir chosen (user decision).
- Possible merge conflict: dev/wavespeed-frame edits riemann_solvers.py near _model_eval.

---

# dev/wavespeed-frame (2026-10-21)

- Item A: default wave speed = closed form, else Gelfand bound of A_n in the face normal frame,
  every consumer; `SpectralRadius.eig` opt-in kept (JAX device eigensolver kept). ANALYSIS 61.
- Item B: shore rule already role-based; test added, no code change. ANALYSIS 62.
- Not done: full matrix (main runs it after merge). Partial store comparison in ANALYSIS 61.
- Beyond scope (main may drop): Perron bound / SME cone / WAVE_SPEED removed; `bound_wave_speed`
  op for no-gate transport systems; `symbolic_spectrum()` re-gates; C38 adaptive rows; gate 49
  checks the cells slot; legacy JAX refs re-blessed.
