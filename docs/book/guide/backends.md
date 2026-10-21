# Backend comparison

Every backend runs the same templates on the same NSM; the
[test matrix](../developer/tests.md) holds them to the same checks.

| | NumPy | JAX | foam | AMReX |
|---|---|---|---|---|
| package | `zoomy_core` (`symsolver.numpy_backend`) | `zoomy_jax` | `zoomy_foam` | `zoomy_amrex` |
| language | Python, NumPy | Python, XLA-compiled | C++ (OpenFOAM 13) | C++ (AMReX) |
| runs in | the Python process | the Python process (the `zoomy_jax` image: CPU, or GPU with `--nv`) | apptainer image | apptainer image (CPU, or CUDA with `--nv`) |
| meshes | as JAX | 1-D, quads, triangles, 3-D (unstructured) | 1-D, quads, triangles, 3-D (unstructured) | Cartesian 1-/2-/3-D |
| MPI | `comm=MPI.COMM_WORLD` under `mpiexec` (mpi4py; PETSc linear solves) | `comm=MPI.COMM_WORLD` under `mpiexec` | `nprocs=N` (decomposePar) | `ranks=N` |
| GPU | no | `JAX_PLATFORMS=cuda`; one GPU per rank via `python -m zoomy_jax.gpurank` | no | `gpu=True` (one GPU per rank) |
| extras | no JAX, no compiler | AD (`jax.jvp`, `jax.grad`), preCICE | preCICE | AMR (`amr=dict(...)`), preCICE |
| call | `setup(mesh, nsm)`, then `run(Q, Qaux)` | `setup(mesh, nsm)`, then `run(Q, Qaux)` | `run(mesh, nsm, Q0, workdir)` | `run(mesh, nsm, Q0, workdir)` |
| classes | `SymbolicHyperbolicSolverNumpy`, `SymbolicChorinSolverNumpy` | `SymbolicHyperbolicSolverJax`, `SymbolicChorinSolverJax` | `SymbolicHyperbolicSolverFoam`, `SymbolicChorinSolverFoam` | `SymbolicHyperbolicSolverAmrex`, `SymbolicChorinSolverAmrex` |

The options of [Solvers and options](solvers.md) are the same on all four; this page
lists what differs.  One page per backend: [NumPy](../backends/numpy.md),
[JAX](../backends/jax.md), [AMReX](../backends/amrex.md), [OpenFOAM](../backends/openfoam.md).

## NumPy

The default backend: no JAX, no compiler.  It prints the template as Python source
and runs it against `zoomy_core.symsolver.numpy_backend.runtime`.  `comm=` (mpi4py)
runs it distributed on JAX's partition; the implicit and pressure solves go through PETSc
(petsc4py) when installed, else a distributed GMRES.  Chorin solves the pressure with PyAMG when it is
installed (`pressure_linear_solver=LinearSolver.auto`), else with matrix-free GMRES.

## JAX

`import zoomy_jax` switches JAX to double precision (`ZOOMY_JAX_ENABLE_X64=0` keeps float32);
a script that imports `jax` first may set `jax.config.update("jax_enable_x64", True)` itself, as
the examples do.  A distributed run partitions the mesh (`zoomy_core.mesh.distributed`), each
rank keeps its cells plus halo layers; `solver.partition.gather(Q)` collects the state on rank 0.
Bind GPUs at launch: `mpiexec -n 4 python -m zoomy_jax.gpurank run.py` (one GPU per rank,
round robin over the node's GPUs or over the ones `CUDA_VISIBLE_DEVICES` lists).  The
`zoomy_jax` image carries the CUDA plugin: it runs on the GPU under `apptainer --nv`, on the
CPU without it.

## foam

The driver prints `Solver.H` and `Kernels.H`, builds them with `wmake` in the
OpenFOAM 13 image (`ZOOMY_OF_SIF`, default `~/of_build/zoomy_openfoam.sif`)
and caches the binary by a hash of the sources (`ZOOMY_FOAM_BINCACHE`; default
`library/zoomy_foam/.bincache` when writable, else `~/.cache/zoomy/foam_bincache`).
`workdir` must be an absolute path.  Options: `nprocs`, `decomposition`
(`Decomposition.scotch`, `Decomposition.simple`; `zoomy_foam.symsolver`).  Second-order
gradients use OpenFOAM's least-squares stencils, so foam agrees with JAX to the
discretization error at second order and to round-off at first order.

## AMReX

The driver prints the same `Solver.H` as foam plus device kernels, builds them
with GNU make in the AMReX image (`zoomy_amrex.sif` in `ZOOMY_AMREX_CONTAINERS`;
`gpu=True`: the CUDA build with the image's nvcc, run under `apptainer --nv`) and caches
builds in `ZOOMY_AMREX_SYMCACHE`.  `ZOOMY_AMREX_BUILD_CPUS` / `ZOOMY_AMREX_RUN_CPUS`
pin the build and the run.  Options: `ranks`, `gpu`, `max_grid_size`, `amr`
(`max_level`, `ratio`, `regrid_int`, `tag` or `region`, `reflux`),
`tiny_profile`, `build_only`.  The grid is the mesh's uniform Cartesian grid;
the LSQ stencils are the zoomy mesh's own.

## Other packages

`zoomy_core` still contains the earlier NumPy solvers (`zoomy_core.fvm`), and
`zoomy_jax.fvm` the earlier JAX solvers (`HyperbolicSolver`, `ChorinSplitVAMSolverJax`); the
symbolic solvers superseded them; the remaining pages that mention them are under
[Legacy](../legacy.md).
Firedrake, FEniCSx, DMPlex and DUNE backends do not print the symbolic templates.

API: {doc}`zoomy_jax <../api/_autodoc2/zoomy_jax/zoomy_jax>`,
{doc}`zoomy_foam <../api/_autodoc2/zoomy_foam/zoomy_foam>`,
{doc}`zoomy_amrex <../api/_autodoc2/zoomy_amrex/zoomy_amrex>`.
