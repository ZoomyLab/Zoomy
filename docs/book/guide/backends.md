# Backends

Every backend runs the same templates on the same NSM; the
[test matrix](../testing/symsolver.md) holds them to the same checks.

| | JAX | foam | AMReX |
|---|---|---|---|
| package | `zoomy_jax` | `zoomy_foam` | `zoomy_amrex` |
| language | Python, XLA-compiled | C++ (OpenFOAM 13) | C++ (AMReX) |
| runs in | the Python process (the `zoomy_jax` image: CPU, or GPU with `--nv`) | apptainer image | apptainer image (CPU, or CUDA with `--nv`) |
| meshes | 1-D, quads, triangles, 3-D (unstructured) | 1-D, quads, triangles, 3-D (unstructured) | Cartesian 1-/2-/3-D |
| MPI | `comm=MPI.COMM_WORLD` under `mpiexec` | `nprocs=N` (decomposePar) | `ranks=N` |
| GPU | `JAX_PLATFORMS=cuda`; one GPU per rank via `python -m zoomy_jax.gpurank` | no | `gpu=True` (one GPU per rank) |
| extras | AD (`jax.jvp`, `jax.grad`), preCICE | preCICE | AMR (`amr=dict(...)`), preCICE |
| call | `setup(mesh, nsm)`, then `run(Q, Qaux)` | `run(mesh, nsm, Q0, workdir)` | `run(mesh, nsm, Q0, workdir)` |

## JAX

Double precision needs `jax.config.update("jax_enable_x64", True)` (or
`JAX_ENABLE_X64=1`); without it JAX runs in float32.  A distributed run
partitions the mesh (`zoomy_core.mesh.distributed`), each rank keeps its cells
plus halo layers; `solver.partition.gather(Q)` collects the state on rank 0.
Bind GPUs at launch: `mpiexec -n 4 python -m zoomy_jax.gpurank run.py` (one GPU per rank,
round robin over the node's GPUs or over the ones `CUDA_VISIBLE_DEVICES` lists).  The
`zoomy_jax` image carries the CUDA plugin: it runs on the GPU under `apptainer --nv`, on the
CPU without it.

## foam

The driver prints `Solver.H` and `Kernels.H`, builds them with `wmake` in the
OpenFOAM 13 image (`ZOOMY_OF_SIF`, default `~/of_build/zoomy_openfoam.sif`)
and caches the binary by a hash of the sources (`library/zoomy_foam/.bincache`).
`workdir` must be an absolute path.  Options: `nprocs`, `decomposition`
(`"scotch"`, `"simple"`).  Second-order gradients use OpenFOAM's
least-squares stencils, so foam agrees with JAX to the discretization error at
second order and to round-off at first order.

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
`zoomy_jax.fvm` the earlier JAX solvers; they are documented under
[Legacy](../legacy.md).  Firedrake, FEniCSx, DMPlex and DUNE backends do not
print the symbolic templates.

API: {doc}`zoomy_jax <../api/_autodoc2/zoomy_jax/zoomy_jax>`,
{doc}`zoomy_foam <../api/_autodoc2/zoomy_foam/zoomy_foam>`,
{doc}`zoomy_amrex <../api/_autodoc2/zoomy_amrex/zoomy_amrex>`.
