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
