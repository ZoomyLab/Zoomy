# AMReX backend

`zoomy_amrex` runs the solver templates on [AMReX](https://amrex-codes.github.io/), for
block-structured adaptive mesh refinement (AMR) on CPU and GPU.  The driver
`zoomy_amrex.symsolver` prints the template as C++, builds it in the AMReX apptainer image,
runs it on the mesh's uniform Cartesian grid and reads the result back.

**Highlights**
- The same arithmetic as JAX: AMReX agrees with it to round-off on the shared cases of the
  [test matrix](../developer/tests.md).
- Adaptive mesh refinement (`amr=dict(...)`), with refluxing, on one `dt` for all levels.
- MPI (`ranks=N`) and GPU (`gpu=True`, one GPU per rank).
- preCICE coupling (`couple=dict(...)`).

## Solvers

| Class | Template | Call |
| --- | --- | --- |
| `SymbolicHyperbolicSolverAmrex` | `HyperbolicTemplate` | `time, Q = solver.run(mesh, nsm, Q0, workdir)` |
| `SymbolicChorinSolverAmrex` | `ChorinTemplate` | `time, Q = solver.run(mesh, (pred, press, corr), Q0, workdir)` |

`Q0` is the initial state `(n_state, n_cells)` in the mesh's cell order, `workdir` the case
directory (recreated).  `run` returns the final state in the same order; the snapshots of
`dt_write` stay in `solver.snapshots`.  The mesh must be a full uniform Cartesian grid, 1-D to 3-D
(anything else raises `NotImplementedError`); the LSQ stencils are the zoomy mesh's own.

## Settings

The settings shared by all backends are the tables of [Solvers and options](../guide/solvers.md).
AMReX adds:

| option | default | meaning |
|---|---|---|
| `ranks` | `1` | MPI ranks (with `gpu=True`: GPUs) |
| `gpu` | `False` | the CUDA build; one GPU per rank |
| `max_grid_size` | `None` | AMReX box size; `None`: about 8 boxes per rank on CPU, 1 per GPU (Chorin: 4), 32 with AMR |
| `amr` | `None` | refinement, see below |
| `linear_solver` | `LinearSolver.gmres` (Chorin: `LinearSolver.hypre`) | `LinearSolver.hypre` or `LinearSolver.gmres`; anything else raises `ValueError` |
| `tiny_profile`, `build_only` | `False` | AMReX TinyProfiler build; only build and cache the binary |
| `couple`, `subcycling` | `None`, `False` | preCICE participant, see the [OpenFOAM page](openfoam.md) |

Environment: `ZOOMY_AMREX_CONTAINERS` (the folder of `zoomy_amrex.sif` and
`zoomy_amrex_gpu.sif`), `ZOOMY_AMREX_SYMCACHE` (the binary cache, by a hash of the sources and the
image), `ZOOMY_AMREX_BUILD_CPUS` / `ZOOMY_AMREX_RUN_CPUS` (the `taskset` lists of the build and
the run).  Inside the AMReX image the driver builds and runs natively.

## End-to-end example: shallow water dam break

[AMReX: from symbolic model to generated C++](../tutorials/ipynb/amrex/minimal.ipynb) builds and runs
a 1-D dam break.

## Adaptive mesh refinement

`amr` is a dict: `max_level`, `ratio` (2), `regrid_int` (0: the initial grids only), the criterion
`tag=(state name, 'value' | 'gradient', threshold)` or a fixed `region=(lo, hi)`, `reflux` (True),
`blocking_factor` (8), `n_error_buf` (2), `grid_eff` (0.7).  The mesh given to `run` is level 0, and
its cell counts must be divisible by `blocking_factor`.  `Q` comes back on level 0 (the composite
averaged down); the states of every level are in `solver.levels` and `solver.log` has the cells per
level.  Restarts and checkpoints are not available with AMR or preCICE.

[AMReX: adaptive mesh refinement](../tutorials/ipynb/amrex/amr.ipynb) runs a dam break with one level.

## Authoring checklist

- Give `run` a uniform Cartesian mesh (`LSQMesh.create_1d`, `create_2d`, ...); boundary conditions
  are declared by the mesh's boundary tags, as on every backend.
- Choose `max_grid_size` for the rank count when you tune performance; the default aims at about
  eight boxes per rank on CPU.
- Pin the build and the run with `ZOOMY_AMREX_BUILD_CPUS` and `ZOOMY_AMREX_RUN_CPUS` on a shared
  machine.

**Install**: AMReX is independent of the Conda/Mamba Zoomy environment; the supported path is the
`zoomy_amrex` [container](../containers.md).

**Repository**: [`library/zoomy_amrex`](https://github.com/ZoomyLab/zoomy-amrex)  
**API reference**: {doc}`zoomy_amrex <../api/_autodoc2/zoomy_amrex/zoomy_amrex>`.
