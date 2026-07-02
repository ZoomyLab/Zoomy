# zoomy_telemac — TELEMAC-MASCARET reference container

Apptainer image providing **TELEMAC-MASCARET v9.1.1**, used as an external
reference hydrodynamics solver to benchmark the Zoomy backends against.

> **Scope.** This folder owns the **container only** (the `.def` build). Running
> simulations, converting results, and post-evaluation (the Malpasset comparison)
> live with the case that needs them — `thesis/cases/malpasset_jax` (see task
> 0016 / REQ-08 for the run + SELAFIN→Zoomy conversion + gauge evaluation).

## GPU vs. MPI — does TELEMAC scale on a GPU?

**No GPU backend.** Mainline TELEMAC (up to v9.1.1) is **CPU + MPI only**.
Parallelism is pure **MPI domain decomposition**: `PARTEL` partitions the mesh
(using METIS), each subdomain is solved by an MPI rank under `mpiexec`, and
`GRETEL` stitches the results back together — scale with `--ncsize N`. GPU
acceleration of TELEMAC exists only as unmerged research ports (OpenACC/OpenMP
of the TOMAWAC wave module on OpenPOWER); it does **not** cover TELEMAC-2D. So
for timing comparisons TELEMAC uses CPU ranks, not the L40S GPUs. This box has
96 physical cores.

## What is built

* Modules: the full `all` target (telemac2d/3d, tomawac, artemis, gaia, …) +
  the `partel`/`gretel` executables.
* Toolchain: Ubuntu 24.04, gfortran-13, OpenMPI, system METIS.
* Build: CMake (default `all` target), **Release**, MPI + METIS only.
* I/O: **SELAFIN (`.slf`) only**. MED/HDF5/MUMPS/AED2/GOTM are intentionally not
  built (the Malpasset verification case and the Zoomy SWE benchmarks are
  SELAFIN). To add `.med` output or the MUMPS direct solver, flip the CMake
  flags in `zoomy_telemac.def`.

Two non-obvious build fixes are baked into the `.def` (both documented inline):
the build must be the default `all` target (a module subset skips linking the
executables and the `build_commands.json` the run driver needs), and TELEMAC
v9.1.1's `generate_build_commands.py` mis-parses Ubuntu's OpenMPI link line, so
the user-fortran link in `build_commands.json` is patched post-build.

## Build

```bash
apptainer build --fakeroot zoomy_telemac.sif zoomy_telemac.def
apptainer test zoomy_telemac.sif        # toolchain + build artifacts + driver checks
```

## Verify it runs (container smoke)

The image ships the bundled `examples/telemac2d/malpasset` case (same geometry as
`data/malpasset/geo_malpasset-small.slf`). The SIF is read-only, so copy the case
to a writable workdir first:

```bash
WORK=$(mktemp -d)
WORK=$WORK apptainer exec zoomy_telemac.sif bash -c \
  'cp -r "$HOMETEL/examples/telemac2d/malpasset/." "$WORK"/'
cd "$WORK"
apptainer exec /path/to/zoomy_telemac.sif telemac2d.py t2d_malpasset-char.cas --ncsize=8
```

Expect `CORRECT END OF RUN` and a `r2d_malpasset-char.slf` result. The full
event (8000 steps) is ~27 s wall at 8 ranks; t=2000 s ~19 s. For the actual
benchmark sweep, the SELAFIN→Zoomy conversion, and the gauge comparison, see the
`malpasset_jax` case (task 0016).
