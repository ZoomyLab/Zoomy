# GPU AMG for the AMReX pressure solve: AmgX vs hypre-CUDA (measured 2026-10-12)

Matrices: the JAX-assembled VAM(1) pressure operator of `matrix/large.py` (`vam_large(nx)`,
first pressure solve), 2 unknowns per cell, 18 nnz/row. L40S GPU; (F)GMRES(30) with one AMG
V-cycle as preconditioner; tolerance 1e-8 (relative residual, recomputed independently).
Scripts and raw results: session scratchpad `amg/` (capture.py, hdriver.c, amgx_bench.py, table.txt).

## Recommendation

**hypre built with CUDA: BoomerAMG (PMIS, ext+i interpolation, l1-Jacobi) preconditioning
FlexGMRES on `S = D^-1 A`**, `D` the 2x2 cell block (the block rt_amrex.H already inverts for
its block-Jacobi preconditioner). One library and one IJ code path for CPU and GPU, native
AMReX integration (`Src/Extern/HYPRE`: `HypreSolver<MSS>`, `HypreIJIface`), active upstream.

The 2x2 per-cell scaling is mandatory: scalar AMG stalls on the raw `A` (hypre 0.62, AmgX
classical 0.61); after the scaling both libraries converge in 18-30 iterations.

Implemented on AMReX (2026-10-13, `rt_amrex.H` `hypre_solve`, `linear_solver="hypre"`, the
Chorin default): the matrix of the printed operator by colouring on the AMReX grid, the 2x2
scaling, FlexGMRES + BoomerAMG (CPU HMIS + hybrid symmetric GS: 19 iterations at 1M cells; GPU
PMIS + l1-Jacobi: 38), to the template's tolerance 1e-10 on the true residual.  hypre 3.2.0 is
built into `~/.cache/zoomy` by `containers/zoomy_amrex/build_hypre_cuda.sh` (CPU and CUDA; the
images have no hypre, and AMReX 25.02's `USE_HYPRE` does not compile against hypre 3.x CUDA).
Results: `matrix/README.md`.

Implemented on foam (2026-10-13, `rt.H` `hypre_solve`, `linear_solver="hypre"`, the Chorin
default; `"schur-gamg"` keeps the old path): the same matrix, scaling and CPU settings, rows
owned per rank in OpenFOAM's global cell numbering (`globalIndex`).  The colouring is the
driver's: a greedy distance-2 colouring of the global LSQ pattern (`constant/zoomyColours`;
9 colours on quads), so probes agree across processor boundaries.  symFoam links the same
static CPU hypre 3.2.0 (the OF13 image has the AMReX image's Ubuntu 24.04, gcc 13.3 and
OpenMPI 4.1.6), and initializes MPI itself in serial runs.

## Measured (setup s / solve s / iterations)

| cells | AmgX, JAX's config (raw A) | AmgX classical PMIS (S) | hypre-CUDA PMIS + l1-Jacobi (S) | hypre 2.28 CPU, 8 ranks (S) | pyamg, 1 core (S) |
|---|---|---|---|---|---|
| 64k | 0.004 / 0.051 / 49 | 0.016 / 0.019 / 26 | 0.011 / 0.026 / 29 | 0.044 / 0.057 / 16 | 0.55 / 0.49 / 8 |
| 256k | 0.006 / 0.138 / 71 | 0.040 / 0.031 / 22 | 0.022 / 0.048 / 30 | 0.189 / 0.224 / 16 | 1.46 / 2.21 / 9 |
| 1.02M | 0.027 / 0.687 / 109 | 0.140 / 0.098 / 24 | **0.080 / 0.132 / 30** | 0.700 / 1.093 / 16 | 6.41 / 9.39 / 9 |

- Iterations stay flat with hypre and with classical AmgX on `S`; JAX's current AmgX config
  (block aggregation on the raw `A`) grows 49 -> 109.
- Side result for JAX: the same scaling + classical PMIS cuts its per-solve cost at 1M cells
  from ~0.71 s to ~0.24 s (matrix only, not measured inside a step).
- Multi-GPU: no gain at <= 1M cells (hypre 2/4 GPUs 0.20 / 0.29 s; container OpenMPI not
  CUDA-aware). AmgX via jaxamg MPI: first call 34-58 s at 64k, > 9 min at 1M.

## Criteria

| | hypre (CUDA) | AmgX |
|---|---|---|
| CPU | same library (CPU hypre 2.28 already in both containers) | GPU only |
| AMReX | `HypreSolver<MSS>` (arbitrary stencil, device fill), `HypreIJIface` | no integration |
| JAX | CPU, serial or MPI: `pressure_linear_solver="hypre"` (`auto` on CPU ranks; ctypes on the shared `build_hypre_cuda.sh cpu-env` build, MPICH), 18 / 19 / 20 iterations on 1 / 2 / 4 ranks at 65k cells (ANALYSIS 46) | GPU: `"amgx"` (jaxamg) |
| build | v3.2.0 in `zoomy_amrex_gpu.sif`, autotools, ~5 min | needs cmake (not in container) |
| non-symmetric `A` | with the 2x2 scaling + FlexGMRES | block aggregation on raw `A`, iterations grow |
| license / activity | Apache-2.0 or MIT; releases 2025-09, 2026-01, 2026-08 | BSD-3; releases 2023-10, 2025-12 |

hypre build (inside the GPU container):
`./configure --with-cuda --with-cuda-home=/usr/local/cuda --with-gpu-arch=89 --without-umpire
--enable-device-malloc-async CC=mpicc CXX=mpicxx && make -j32 install`
(hypre 3.x enables Umpire for GPU builds by default; `agg_interp_type=6` segfaults in 3.2.0.)

Caveats: one matrix (the first solve); hierarchy reuse across steps not measured; AmgX setup
times are a lower bound (cached hierarchy).
