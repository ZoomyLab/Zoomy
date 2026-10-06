# zoomy_dune — DUNE-FEM (UFL / DG) container

DUNE-FEM 2.12 with ALUGrid, dune-fem-dg and polygongrid, for the UFL-written
DG backend. Sister image to `zoomy_firedrake`: both write the weak form in
plain UFL, so a form derived for one is readable on the other.

## Build

Unprivileged, no root and no docker needed:

```bash
apptainer build --fakeroot zoomy_dune.sif containers/zoomy_dune/zoomy_dune.def
```

~11 min; all eight DUNE modules compile from source. The `%test` section gates
the build on importing exactly what the demos need, so a broken image cannot
be produced.

## Run the Euler tutorial

The upstream dune-fem Euler demo
(<https://www.dune-project.org/sphinx/content/sphinx/dune-fem/euler_nb.html>),
copied verbatim into `examples/euler.py` apart from running headless:

```bash
mkdir -p run && cp containers/zoomy_dune/examples/{euler.py,triangle.dgf} run/
cd run
apptainer exec --bind "$PWD:/workspace" --pwd /workspace \
    ../zoomy_dune.sif python3 euler.py
```

Writes `euler_grid_simplex.png`, `euler_evolution_*.png`, `euler_diagonal.png`.
Takes ~8 min the first time (JIT), seconds afterwards — the cache persists in
`.dune-py/`.

Expected output, which matches the tutorial including its deliberate failure:

```
dune.fem.threading: use=4 max=4
0.100  0.200  0.300            # DG order 1, MinMod limiter
8.09e+307  SOLUTION NOT VALID  # DG order 1, NO limiter -- the demo's own point
0.102  0.200  0.302            # FV order 0, no reconstruction
0.100  0.201  0.301            # FV order 0, MinMod reconstruction
```

## Threading

`dune.fem.threading.max` is `omp_get_max_threads()`, so it is **capped by
`OMP_NUM_THREADS`** — setting `dune.fem.threading.use = 8` under
`OMP_NUM_THREADS=1` silently gives you one thread, and an empty value counts
as 1. The image defaults it to 4; raise it per run:

```bash
apptainer exec --env OMP_NUM_THREADS=16 zoomy_dune.sif python3 case.py
DUNE_THREADS=16 ...   # examples/euler.py reads this
```

## dune-py JIT cache

DUNE compiles each algorithm into a module cache on first use. The image is
read-only, so the cache goes in the bound workspace: `DUNE_PY_DIR` defaults to
`$PWD/.dune-py`. Budget ~165 MB and a few minutes for a fresh case; delete it
to force a rebuild.

## Two ways to write a scheme

| | `dune.femdg.femDGOperator` | `dune.fem.operator.molGalerkin` |
|---|---|---|
| you supply | flux functions `F_c`, `maxWaveSpeed` | the **weak form in UFL** |
| limiter | built in (MinMod, `physical`, `jump`) | yours (we use MOOD) |
| custom Riemann solver, Audusse WB, NCP | not a hook | yours to write |
| portable to Firedrake | no | yes |

`examples/euler.py` uses the first and exists to prove the image. Zoomy cases
use the second — see the upstream `demos/discontinuousgalerkin.py` for the
pattern (UFL residual → `molGalerkin` → your own explicit RK loop).
