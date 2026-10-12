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

## Speed after lazy expressions (be19146, +277 / -129 lines, runtime only)

| config, best box layout | old | new before | new after |
|---|---|---|---|
| CPU 8 ranks | 42.0 s | 53.4 s | 47.6 s (+13 %; per step +5 %) |
| 1 GPU | 4.76 s | 8.94 s | 5.66 s (+19 %; per step +10 %) |
| 4 GPUs | 2.92 s | 3.46 s | 2.73 s (-7 %) |

Water = rain in every run.  The rest: +8.5 % steps (face wave speeds, as
JAX) and on 1 GPU the dt reduction and three unfused cells() kernels.

## Capability round 4 (40b82ed, 3370001, ee56d68)

    1. Pressure solver (template-driven: wiring + gate)            gate27b_amrex.py
       - pressure_solve="newton" wired through the driver (40b82ed, +9 lines); time_order 2 needed nothing.
       | case | CPU 1 vs JAX | MPI 2 vs CPU 1 | GPU 1 vs JAX |
       | VAM(1) Chorin, time_order 2     | 5.5e-11 | 2.6e-15 | 5.5e-11 |
       | VAM(1) Chorin, Newton pressure  | 4.5e-12 | 2.1e-15 | 4.5e-12 |
       (elliptic tolerance 1e-10; 2-D bump channel of gate 21, t = 0.05)
    
    2. IMEX (per-cell Newton root)                                  gate27b_amrex.py
       | SWE bump O1 + Manning n = 0.05, IMEX | 7.0e-13 | 0.0 | 7.3e-13 (GPU vs CPU 6.9e-13) |
       (FD Jacobian on AMReX, jvp on JAX: agreement = the Newton tolerance; 0 lines)
    
    3. preCICE (couple.*): NOT implemented -- cost above the 150-line bar, and a container change:
       - The AMReX containers have no preCICE (no headers, no lib; the OF-built
         ~/of_build/precice-3.4.0 lacks boost 1.83 runtime libs there).  Needs preCICE 3.4
         (the Python peers' version) in zoomy_amrex.def and zoomy_amrex_gpu.def + rebuilds.
       - Code: rt_amrex.H ~130 (Coupling: participant, mesh, data names, the coupled side's
         faces per rank; coupled_faces mask; interface_states gather -> host; interface_flux
         kernel over the coupled faces with the peer states uploaded; couple_init / max_dt /
         read / write / advance as foam's rt.H), symAmrex.cpp ~40 (one participant per rank,
         vertices = this rank's face centres on the coupled side, as symFoam.C:100), driver ~40
         (couple dict -> inputs, -lprecice), gate ~60 (gate-3 dam break AMReX<->AMReX,
         AMReX<->JAX, serial and 2+2).  Total ~210 + 60 lines + 2 container rebuilds.
    
    4. 1-D: done with 6. (an AMREX_SPACEDIM = 1 build of the same, now dimension-generic,
       runtime): cheaper than an n x 1 2-D grid (that needed ~60 special cases for the inactive
       y faces / sides / LSQ); 0 extra lines once 6. is generic.
    
    5. AMR: plan only -- see below.
    
    6. 3-D (and 1-D): rt_amrex.H made dimension-generic (IntVect cells, D = AMREX_SPACEDIM face
       directions, LSQ classes per axis, periodic directions), symAmrex.cpp generic grid / IO,
       driver: grids, sides, periodic flags, LSQ tables (create_1d/2d/3d proxies) and build trees
       per dimension.  Templates unchanged.  (3370001 + periodic fix)
       Periodic sides follow JAX: the seam is a boundary in the LSQ stencils (boundary class,
       ghost sample = the partner cell), the faces see the partner cell (FillBoundary images).
       Results: table_3d.txt (gate3d_amrex.py), all vs JAX at round-off:
       Sod x/y/z O1 3.6-4.0e-16, O2 0.75-1.2e-15; wave periodic O1/O2 4.1e-16, 16^3 5.4e-16,
       EOC 2.16 (= JAX); wall box 6.0e-16, mass/energy drift 0; Sod 1-D 1.2e-15; SME(1) 1-D
       3.4e-11 (ANALYSIS 23 round-off amplification). MPI 2 = serial (0); GPU = CPU on sodx1/2,
       wave2, wall2, sod1d2, sme1d2 (other cases share the binary path; not run on GPU).
       Regression 2-D (table_27.txt, core b2717f6): SWE O1 4.4e-16, O2 8.1e-16, dry 1.1e-9,
       VAM 8.1e-11; MPI 2 0/0/0/6.8e-15; GPU = JAX numbers.
       Final gate27b: table_27b.txt (core b2717f6, GPU vs CPU 3.0e-15 / 1.8e-15 / 6.9e-13).
    

## AMR plan (not implemented)

    AMR plan (no implementation)
     What AMReX offers: AmrCore (level hierarchy, regrid, MakeNewLevelFromScratch/FromCoarse,
     RemakeLevel, ClearLevel, ErrorEst tagging), FillPatchSingleLevel / FillPatchTwoLevels
     (ghosts from the same level + space-time interpolation from the coarse one, with a
     PhysBCFunct), FillCoarsePatch (new fine boxes), average_down, FluxRegister (CrseInit /
     FineAdd / Reflux: the coarse-fine flux mismatch put back into the coarse cells), and either
     subcycling (r time steps per coarse step, time-interpolated ghosts) or one dt for all
     levels.  The old hand-written path uses AmrCore with subcycling and NO reflux
     (ZoomyAmr.H:85 TODO): not conservative at coarse-fine faces.
     Templates: nothing new.  The runtime level loop calls a printed single-level STEP instead
     of the printed march: program.py prints, besides ``run``, a ``step`` procedure =
     HyperbolicTemplate.step(...) with a given dt (and ``dt_of`` = HyperbolicTemplate.dt / face_dt on
     the cells' speeds) -- printing existing template functions, not changing them.
     Runtime: (a) zs::Mesh per level (geometry, BoxArray, LSQ classes for that dx); (b) halo =
     the level's FillPatch (FillBoundary on level 0, FillPatchTwoLevels on finer levels, time
     interpolation between the coarse old/new states at the RK stage time); (c) divergence
     records, when a FluxRegister is attached, the coarse-side face contribution (Wm on owned
     faces, Wp on the others: for an NCP scheme the coarse cell's own side, not a symmetric
     flux) into the register (coarse: CrseInit, fine: FineAdd x area x dt / dt_coarse);
     (d) after the fine level's step: Reflux into the coarse level, average_down, full_aux.
     Driver: per-level LSQ tables, AMR settings (max_level, ref_ratio, regrid_int, a tagging
     criterion on a named state row or its gradient), multi-level plotfiles.
     Conservation: reflux of the coarse-side contributions -> mass to round-off for closed
     domains (the conservative part of Wm/Wp is antisymmetric); interpolation at C/F on the
     model's reconstruction variables W (eta = b + h for SWE) so a lake at rest stays at rest.
     Time stepping: start with one dt for all levels (min over levels; no time interpolation
     inside a step except the RK stage), subcycling as a second step.
     Lines: AmrCore driver ~250 (level data, the five AmrCore overrides, FillPatch wrappers,
     the level loop with reflux / average_down / regrid, multi-level output), rt_amrex.H ~60
     (per-level halo, divergence -> FluxRegister hook), program.py ~30 (step, dt_of), driver
     ~60, gate ~80.  ~400 lines; subcycling +~100.
     Tests: (1) 2 levels, fine level covering the domain == fine single level (round-off);
     (2) a static fine patch vs fine single level: L1 difference -> 0 under refinement, and
     the observed order of a smooth Euler / SWE wave crossing the patch (O2: ~2);
     (3) mass (and energy for Euler) to round-off with reflux in a closed box, a measurable
     drift without it (shows the register works); (4) lake at rest over a bump crossing the
     C/F interface: |u| ~ 1e-15; (5) dynamic regrid on the dry dam break front: mass to
     round-off, h >= 0; (6) MPI 2/4 ranks = serial, GPU = CPU, as gates 27 / 3d.

## Round 5: AMR (one dt for all levels) and preCICE (3e95abb; core a82305d)

AMR runs the printed template step on AmrCore: per-level mesh + LSQ table,
finer levels' halos interpolated from the coarser level in the
reconstruction variables W, FluxRegister reflux of the conservation-law
rows (refluxing SWE momentum broke the lake at rest: its NCP part has no
conservative reading), average_down, regrid, multi-level output.  Templates
unchanged (program.amr_procedures prints existing pieces).  ~470 code
lines (+ comments: 595).  Gate 37:

| test | result |
|---|---|
| fine level covering the domain = fine single level | 0.0 (AMReX), 6e-16 / 3e-15 vs JAX |
| static patch, Euler wave O2 | composite L1 EOC 2.39 / 2.22; AMR - fine 8.9e-3 -> 3.8e-4 |
| closed box, Euler O2 + MOOD | mass 0, energy 2.2e-16 (without reflux 5.9e-4 / 1.4e-3) |
| lake at rest across C/F | abs(u) 1.1e-15 / 1.3e-15 |
| dry dam break, regrid every 2 steps | mass 2.9e-15, h >= 0 |
| MPI 2, GPU | = serial / = CPU (0.0) |

Subcycling: not done.  preCICE (gate 38): AMReX<->AMReX 1+1 and 2+2,
AMReX<->JAX vs single domain 3-4e-15, mass exact; images rebuilt with
preCICE 3.4.0 (containers/zoomy_amrex/*.def).  No GPU coupling run.

Correction: the dry dam break's 1.1e-9 vs JAX was not an eigenvalue
cluster: the driver took the domain from rounded cell centres (3e-10
off).  Fixed: 3e-15.
