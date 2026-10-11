# AMReX templated backend (single level, 2-D Cartesian; CPU / GPU / MPI)

Commits (feature/amrex-templates, not pushed)
  zoomy_amrex 3912830 runtime rt_amrex.H, app symAmrex.cpp, device kernels, driver
              8c309c8 ZoomyEig: MATLAB exceptional shift only at iteration 30; nvcc-legal lambdas
              35f4d19 initial state synced before the pinned buffer is freed
  zoomy_core  36831f8 symsolver/program.py: the C++ host program (run/march/initial_aux/configure)
Printer: CxxPrinter unchanged; foam and amrex print the same text (program.py).
Templates (hyperbolic/chorin/vocabulary/reconstruction): unchanged.

Block -> what implements it
  cells/pointwise/faces/face_aux/boundary_*  ParallelFor over MultiArray4 (all boxes, one launch);
                                             kernels = device functors (symkernels.py, AmrexCore printer)
  gradient/derivative_aux                    zoomy mesh's own lsq_gradQ*scale per cell class (proxy grid)
  minmod/venkatakrishnan/barth_jespersen     limit() on 4 face neighbours + 2 Wb - W ghosts
  divergence/edge_source                     per cell over its 4 faces (owner/other as JAX)
  halo                                       MultiFab::FillBoundary; ghost validity tracked (ngv)
  reduce_min/max/any                         ParReduce + ParallelDescriptor (NaN propagates)
  linsolve                                   amrex::GMRES, right precond = exact cell blocks probed by
                                             colouring ((R+1)^2 colours), inverted per cell
                                             (chorin_common.H ptype 3 idea; MLPoisson not used: Dirichlet-
                                             only, heuristic scale, indefinite 2x2 VAM block)
  root                                       per-cell FD Newton (foam rt.H scheme), device solve
  while_loop/cond                            host
  io_write                                   WriteSingleLevelPlotfile + raw dump
  eigenvalues (in kernels)                   ZoomyEig.H device QR (fixed)
Reused unchanged: UserFunctions.H, ZoomyEig.H (1 fix), build.py GNUMAKEFILE, AmrexCore printer.

Lines
  new templated path: rt_amrex.H 1409, symAmrex.cpp 144, symkernels.py 178, symsolver.py 354,
                      program.py 71 (core)                                        = 2156
  hand-written single-level path it can replace (not deleted): make_rhs.H, ZoomyAmr.{cpp,H}
  (single-level part), main.cpp, imex_solver.H, chorin_main.cpp, chorin_common.H, init_solution.cpp,
  write_plotfiles.cpp, plotfile_utils.*, constants.H, to_amrex.py, run_case.py, solvers.py,
  _constants.py, build.py                                                         = 4562
  ZoomyEig.H fix: +21 / -15

Gates: table.txt (gate27_amrex.py table), figure amrex_vs_jax.png (gate27_figure.py).
  Hyperbolic: AMReX vs JAX at round-off; MPI 2/4 ranks and GPU bitwise = AMReX serial.
  Dry dam break 1.1e-9: the eigenvalue cluster at wet/dry faces (foam vs JAX: 1.5e-9).
  VAM: 4.9e-11 = elliptic tolerance (1e-10); MPI/GPU vs serial 1e-14 (GMRES dot order).
  Mass: closed domains exact (0); VAM (inflow) AMReX vs JAX 2e-13 rel.
  IMEX (root): SWE bump O1 + Manning n=0.05 vs JAX 7.0e-13 (scratch test, not in gate).
Regression with zoomy_core worktree: gates 1, 7, 9 (0:1, 2d:2, dry:2), 12 unchanged
  (regress_gates_1_7_9_12.txt; gate 1's dry 4.8e-1 vs the OLD solver is the known
  template/old difference, ANALYSIS 11).

Bug found in existing code: ZoomyEig.H ran MATLAB's ad hoc shift on EVERY QR sweep (the
iter == 30 guard was lost in the port): NOCONV -> +inf wave speed -> dt = 0 on clustered
spectra (214/3000 random clustered 4x4; wet/dry SWE face; VAM 10x10). Fixed: 0/9000 fail.

Open
  - diffusion / implicit_diffusion blocks not implemented (foam rt.H lacks them too).
  - preCICE blocks skipped (as instructed); 1-D and 3-D builds not supported (2-D only).
  - Every block allocates its output (value semantics); no in-place lowering, no fused kernels.
    Timings are tiny meshes only (march: o2 40^2 CPU 0.44 s, 1 GPU 0.09 s; VAM 60x6 GPU slower
    than CPU, launch-bound).
  - Driver re-prints the kernels every run (VAM ~100 s of sympy) before the binary-cache lookup.
  - Gate number: named gate27_amrex.py (gate23 is taken by another agent's foam gate).
  - foam's SymbolicHyperbolicSolverFoam._print duplicates program.py; switch it over (foam owner).


# Old hand-written path vs templated path (deletion evidence, 418e29a)

Catchment (526 x 555 @ 10 m, 1 h, 30 mm/h rain, O1 PositiveHLL, CFL 0.5):

| run | old: steps / march s / ms/step | new: steps / march s / ms/step | water - rain |
|---|---|---|---|
| CPU 8 ranks | 6835 / 54.9 / 8.04 | 7415 / 70.5 / 9.50 | 0 both |
| 1 GPU | 6835 / 5.18 / 0.76 | 7415 / 8.94 / 1.21 | 0 both |
| 4 GPUs | 6835 / 2.92 / 0.43 | 7415 / 3.46 / 0.47 | 0 both |

- New h_max 4.156 = JAX (4.155); old 2.837: the old WB reconstruction / bed
  source is a different scheme (mean |dh| 1.1 cm, max 3.7 m in a pit).
- New is 1.18x (CPU 8) to 1.6x (1 GPU) slower per step, +8.5 % steps (face
  wave speeds).  Profile: unfused per-block elementwise kernels, not the flux
  (the new face kernel is cheaper than the old fused RHS); load imbalance
  ~1/3 of CPU time on both.
- Old-only capabilities: AMR (non-conservative: no reflux), explicit
  diffusion, aux BC kernel, 1-D, VTK / 3-D extruded output, raster / gmsh IC
  loaders, max_step / dtmin, global JFNK source, MLPoisson / block-MG Chorin
  preconditioners.
- Not deleted (criterion: at least as capable and as fast).
