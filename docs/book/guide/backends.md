# Solver backends

The same `NumericalSystemModel` runs on every backend. Switching is a change of the solver class, not of the model.

| backend | GPU | MPI | AD | preCICE coupling | use it for |
|---|---|---|---|---|---|
| NumPy | no | yes | no | Chorin solver | first runs, the browser, nothing to install but `zoomy-core` |
| JAX | yes | yes | yes | yes | speed, GPUs, gradients |
| AMReX | yes | yes | no | yes | adaptive mesh refinement on Cartesian grids |
| OpenFOAM | no | yes | no | yes | unstructured meshes, coupling to a resolved 3-D free surface |

Each backend has a hyperbolic solver for balance laws and a Chorin solver for pressure-split models
([Solvers and options](solvers.md)): `SymbolicHyperbolicSolver<Backend>` and
`SymbolicChorinSolver<Backend>`, with `<Backend>` one of `Numpy`, `Jax`, `Amrex`, `Foam`.

Code printing: NumPy and JAX run the template printed as Python, AMReX and OpenFOAM as C++, see
{doc}`Code printer: AMReX <../tutorials/ipynb/components/printer_amrex>` and
[Templates and backends](../developer/templates.md).

## NumPy

`zoomy_core.symsolver.numpy_backend`, part of `zoomy-core` and the solver of the in-browser GUI.
`setup(mesh, nsm)` returns the initial state, `run(Q, Qaux)` marches it.
Tutorial: {doc}`Shallow water in 1-D <../tutorials/ipynb/swe/simple_numpy>`.

## JAX

`zoomy_jax.symsolver.solver`, compiled with XLA, same calls as NumPy.
It runs on a GPU when JAX has its CUDA build, as in the `zoomy_jax` container started with `--nv`.
A distributed run passes `comm=MPI.COMM_WORLD` under `mpiexec`, one GPU per rank through
`mpiexec -n 4 python -m zoomy_jax.gpurank run.py`.
`ad_mode` selects forward or reverse differentiation through the time loop.
Tutorials: {doc}`forward-mode AD <../tutorials/ipynb/ad/forward_ad>`, {doc}`reverse-mode AD <../tutorials/ipynb/ad/reverse_ad>`.

## AMReX

`zoomy_amrex.symsolver`, in the AMReX container. One call `run(mesh, nsm, Q0, workdir)` generates, builds and runs the solver.
Uniform Cartesian meshes, `ranks=N` for MPI, `gpu=True` for one GPU per rank, `amr=dict(...)` for refinement.
Tutorial: {doc}`AMReX: from symbolic model to generated C++ <../tutorials/ipynb/amrex/minimal>`.

## OpenFOAM

`zoomy_foam.symsolver`, in the OpenFOAM container. Same call as AMReX.
Structured and unstructured meshes, `nprocs=N` for MPI, `couple=dict(...)` for preCICE.
`zoomy_foam.vof` couples a depth-averaged model to a resolved 3-D free surface.

## Connect a backend

NumPy runs inside the GUI. Every other backend is a `zoomy-server` that the GUI submits cases to:

```bash
pip install zoomy-server zoomy-jax
zoomy-server --adapter jax            # also numpy
apptainer run zoomy_amrex.sif 8080    # or the OpenFOAM or JAX image
```

Then open the [GUI](gui.md), choose "Connect backend" and enter `http://localhost:8080`.
Installation and images: [Install](../installation.md), [Containers](../containers.md).

API: {doc}`zoomy_jax <../api/_autodoc2/zoomy_jax/zoomy_jax>`,
{doc}`zoomy_foam <../api/_autodoc2/zoomy_foam/zoomy_foam>`,
{doc}`zoomy_amrex <../api/_autodoc2/zoomy_amrex/zoomy_amrex>`.
