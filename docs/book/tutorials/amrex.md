# AMReX

[Minimal walkthrough](ipynb/amrex/minimal.ipynb) — the same shallow-water model
as the [NumPy tutorial](swe.md), with the solver template printed as C++,
built and run on AMReX instead of a NumPy solve.

**You need the AMReX container.** The page does not stop at the generated code:
it builds it, runs the case and plots the result. The notebook runs on your
machine with `zoomy_amrex` installed from a clone; the driver builds and runs
the C++ inside the AMReX Apptainer image:

```bash
git submodule update --init library/zoomy_core library/zoomy_amrex
pip install -e library/zoomy_core -e library/zoomy_amrex -e library/zoomy_plotting
apptainer pull containers/zoomy_amrex/zoomy_amrex.sif oras://ghcr.io/zoomylab/zoomy_amrex_sif:latest
```

See [Installation](../installation.md#prebuilt-containers) for every container
and [Backends](../guide/backends.md) for the AMReX settings (MPI ranks, GPU,
AMR). The maintained end-to-end runs are the cross-backend
[test matrix](../testing/symsolver.md).
