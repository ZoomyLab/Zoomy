# zoomy_telemac — TELEMAC-MASCARET reference container

Apptainer image providing **TELEMAC-MASCARET v9.1.1**, used as an external
reference hydrodynamics solver to benchmark the Zoomy backends against (the
Malpasset dam-break is the shared validation case).

## GPU vs. MPI — does TELEMAC scale on a GPU?

**No GPU backend.** Mainline TELEMAC (up to v9.1.1) is **CPU + MPI only**.
Parallelism is pure **MPI domain decomposition**: the mesh is partitioned by
`PARTEL` (using METIS), each subdomain is solved by an MPI rank under `mpiexec`,
and the results are stitched back together by `GRETEL`.

GPU acceleration of TELEMAC exists only as **research ports** — OpenACC/OpenMP
offloading of the **TOMAWAC** spectral-wave module on OpenPOWER (Bahbah et al.),
reporting ~5–7× on a single subroutine set. That work was never merged into the
distribution and does **not** cover TELEMAC-2D. So for our timing comparison,
TELEMAC scales by adding CPU ranks (`--ncsize N`), not by using the L40S GPUs.
This box has 96 physical cores, so a strong-scaling sweep up to ~96 ranks is
meaningful.

## What is built

* Modules: `telemac2d`, `partel`, `gretel` (+ their CMake dependencies: `bief`,
  `hermes`, `parallel`, `special`, `damocles`).
* Toolchain: Ubuntu 24.04, gfortran-13, OpenMPI, system METIS.
* Build: CMake via `build_telemac.py`, **Release**, `--deps mpi metis`.
* I/O: **SELAFIN (`.slf`) only**. MED/HDF5/MUMPS/AED2/GOTM are intentionally not
  built — the Malpasset verification case (and the Zoomy SWE benchmarks) are
  SELAFIN. To add `.med` output or the MUMPS direct solver, rebuild with extra
  `--deps` / `--modules` (see `zoomy_telemac.def`).

## Build

```bash
apptainer build --fakeroot zoomy_telemac.sif zoomy_telemac.def
```

## Run the Malpasset verification case

The image ships the bundled `examples/telemac2d/malpasset` case (same geometry
as `data/malpasset/geo_malpasset-small.slf`). The SIF is read-only, so a case is
always copied to a writable workdir first — the helper scripts do this:

```bash
# single run (8 MPI ranks, characteristics scheme)
./run_malpasset.sh -n 8 -c t2d_malpasset-char.cas

# strong-scaling sweep -> bench_malpasset_results.csv
./bench_malpasset.sh -r "1 2 4 8 16 32 64"
```

Available steering files (different advection schemes): `t2d_malpasset-char`
(characteristics), `-nerd`, `-eria`, `-hllc`, `-kin1`, `-lips`, `-prim`,
`-fine` (refined mesh). The `-fine` case is the one to use for a heavier
scaling study.

## Manual run (without the helpers)

```bash
WORK=$(mktemp -d)
apptainer exec zoomy_telemac.sif bash -c \
  'cp -r "$HOMETEL/examples/telemac2d/malpasset/." "'$WORK'"/'
cd "$WORK"
apptainer exec /path/to/zoomy_telemac.sif telemac2d.py t2d_malpasset-char.cas --ncsize=8
```

Outputs land in `$WORK` (a SELAFIN results file `r2d_malpasset*.slf`), readable
with any SELAFIN reader (TELEMAC's `scripts/python3` ships `run_telfile.py`).

## Timing & strong scaling

Small mesh (26 000 triangles, 13 541 nodes), characteristics scheme, on the
96-core dual-Xeon box. Wall clock includes PARTEL partitioning + the per-run
user-fortran JIT compile + GRETEL merge (a fixed few-second overhead), so the
pure solver scaling is better than the wall numbers below.

| ncsize | full event t=4000 s (8000 steps) | t=2000 s (4000 steps) |
|-------:|---------------------------------:|----------------------:|
|   1    | 73.4 s | — |
|   2    | 46.6 s | — |
|   4    | 37.9 s | — |
|   8    | 27.0 s | **19.2 s** |
|  16    | 22.7 s | — |

1→16 ranks ≈ 3.2× on this small mesh — it is communication-bound past ~8 ranks.
Use `t2d_malpasset-fine.cas` (refined mesh) for a meaningful many-core study.

## Comparing with the Zoomy backends

The thesis Malpasset cases (`thesis/cases/malpasset_jax`) read cell-centred Zoomy
HDF5 stores via `zoomy_plotting.read_hdf5` (`fields/iteration_*/Q = [b,h,hu,hv]`,
plus a `mesh/` group). `selafin_to_zoomy.py` converts a TELEMAC SELAFIN result
into exactly that store, reusing a reference Zoomy store's mesh so the cell
ordering matches the jax runs **cell-for-cell** (both meshes come from the same
`geo_malpasset-small.slf` — node order is bit-identical). The P1 node values are
projected to P0 cells by triangle averaging.

```bash
# run a case, then convert its result into a Zoomy store
./run_malpasset.sh -n 8 -c t2d_malpasset-char.cas      # -> $WORK/r2d_malpasset-char.slf
./telemac_to_zoomy.sh "$WORK/r2d_malpasset-char.slf" output/telemac
#   -> output/telemac/{telemac.h5, settings.h5, telemac.h5.ckpt.json}
```

The resulting `telemac.h5` is a drop-in for `malpasset_jax/output/telemac/` so
`compare_telemac.py` plots real TELEMAC vs the SME/SWE runs. Validated: the
char-run gauge max-depths track the observed police-survey values closely
(P6 40.0 m vs 40.3 obs, P8 23.0 vs 24.0, P14 4.2 vs 5.4).
