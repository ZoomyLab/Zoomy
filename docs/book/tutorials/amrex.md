# AMReX

[Minimal walkthrough](ipynb/amrex/minimal.ipynb) — the same shallow-water model
as the [NumPy tutorial](swe.md), taken to generated C++ instead of a NumPy
solve.

**You need the AMReX container** to compile and run. The codegen itself runs
anywhere `zoomy_core` is installed:

```bash
apptainer pull zoomy_amrex.sif oras://ghcr.io/zoomylab/zoomy_amrex_sif:latest
apptainer run --bind $PWD:/workspace zoomy_amrex.sif jupyter
```

See [Installation](../installation.md#prebuilt-containers) for every container
and the three run modes (solver API, Jupyter, shell).

The maintained end-to-end cases are the AMReX test suite,
`library/zoomy_amrex/tests/` — wet and dry dam breaks, 2-D, a lake-at-rest
well-balancing check, NumPy parity, and a two-rank parallel run.
