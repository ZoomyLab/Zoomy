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

## Adaptive refinement

`HyperbolicSolver` marks on **η = b + h**, never on `h` alone: over varying
bathymetry a lake at rest has constant η but strongly varying `h`, so an
`h`-gradient refines the whole reservoir. Three regimes give a target level —
any water at all (`wet_level`), the wet/dry front (`front_level`), a steep
surface relative to the local depth (`shock_level`) — and `buffer_cells` rings
of dilation keep the band ahead of the wave between adaptations. Hysteresis
sits between `refine_tolerance` and `coarsen_tolerance`.

**`max_level` defaults to 0, which means no AMR.** A case that never sets it
runs on a uniform mesh; the solver now logs `uniform mesh` so this is visible.

The per-cell level is written as the cell field `level`, so refinement is
directly visible in ParaView. Do not infer it from cell size — the base mesh
is strongly graded and cell size mostly shows that grading, not the AMR.

## Running a case against the repo rather than the image

The image ships an installed `zoomy_dune`, which **shadows the repo copy**.
Editing `library/zoomy_dune/` changes nothing until the image is rebuilt,
unless the repo is put ahead of it:

```bash
apptainer exec --bind "$ZOOMY" --pwd "$PWD" \
    --env PYTHONPATH="$ZOOMY/library/zoomy_dune" \
    --env OMP_NUM_THREADS=8 \
    "$ZOOMY/containers/zoomy_dune/zoomy_dune.sif" python3 run.py
```

Without the `--env PYTHONPATH`, the installed copy wins silently.

## A model with no boundary conditions is not boundary-free

`BoundaryConditions([])` lowers to a kernel that returns `Q` itself, so every
exterior facet becomes a **free in/outflow against an infinite reservoir**. On
a closed basin that is a mass source, and it is invisible in every short test:
while `u = 0` the transmissive ghost and a reflective wall's ghost are *the
same state*, so a lake at rest, a well-balancing check and a 10 s dam break all
pass. The leak only switches on once water moves at the edge.

On Malpasset (1027 of 1080 exterior facets wet) it bled 2.5e5 m^3/s and
tripled the water volume by t = 600 s, growing inward from the wet boundary
band. Attach the wall the other backends use:

```python
from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall
model = MalpassetSWE(boundary_conditions=BoundaryConditions([
    Wall(tag="wall", momentum_field_indices=[[2, 3]],
         permeability=0.0, wall_slip=1.0)]))
```

Every run now logs its treatment — `boundary: 1 condition(s) [wall]` or
`boundary: TRANSMISSIVE`. **Read that line.**
`thesis/cases/malpasset_dune/probe_boundary.py` measures the leak directly in
one residual evaluation: for a conservative scheme the interior telescopes, so
`d/dt ∫h == -(flux through ∂Ω)`, which must be zero on a closed basin.

## What the dune solvers do not discretise

* `diffusion_matrix_explicit` — no viscous path in either solver (`F_v` is
  unset on the femDG model). The explicit jax solver drops it too; the
  Firedrake reference and amrex do not. Set `nu=0` so it is not declared and
  then ignored.
* `HyperbolicSolver` has no volume flux term, so it is DG0-only and now
  refuses `dg_degree > 0`. `FemDGSolver` supplies the volume term via `F_c`.
