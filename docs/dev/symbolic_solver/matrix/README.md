# Cross-backend test matrix (symbolic solvers)

The acceptance test of every backend that prints the symbolic solver
templates (JAX, foam, AMReX, and any future one).  Backends no longer agree
to round-off (foam uses OpenFOAM's own stencils at second order); every
backend must pass the same physics checks instead: convergence, exact
solutions, conservation, symmetry, parallel = serial, GPU = CPU.

## Coverage: capability x backend -> case

| capability | JAX | foam | AMReX |
|---|---|---|---|
| SWE 2-D quads, O1 / O2: smooth convergence | C1 | C1 | C1 |
| shock, O2 + MOOD | C2 | C2 | C2 |
| well-balanced (lake at rest) | C3 | C3 | C3 |
| BCs: wall, Dirichlet, InflowOutflow, Extrapolation, FromModel | C4 | C4 | C4 |
| wet/dry, positivity | C5 | C5 | C5 |
| conservation (closed domains) | C6 | C6 | C6 |
| MPI = serial (hyperbolic) | C7 | C7 | C7 |
| pressure (VAM(1) Chorin): linear, Newton, time_order 2 | C8 | C8 | C8 |
| pressure on triangles | C8 | C8 | n/a (Cartesian) |
| GPU = CPU | C9 | n/a | C9 |
| pressure convergence (Chorin, Richardson) | C10 | C10 | C10 |
| 1-D | C11 | C11 | C11 |
| triangles | C12 | C12 | n/a (Cartesian) |
| periodic, serial and distributed | C13 | C13 | C13 |
| IMEX (implicit source) | C14 | C14 | C14 |
| implicit diffusion (IMEX), distributed | C26 | C26 | C26 |
| AD: jvp vs FD, reverse vs forward (hyperbolic, IMEX, Chorin; 3-D Euler) | C15, C22 | n/a | n/a |
| preCICE: coupled = single domain | C16 (jax-jax, jax-foam) | C16 (foam-foam, jax-foam) | C16 (amrex-amrex, amrex-jax) |
| AMR (reflux, regrid, one dt; gate 37) | n/a | n/a | C17 |
| compressible Euler 3-D: Sod, smooth wave, walls, MPI | C18-C21 | C18-C21 | C18-C21 |
| Euler 3-D cross-backend (O1 round-off, O2 converging) | C23 | C23 | C23 |
| compressible Euler 2-D | C24 | C24 | C24 |
| large tier (scaling, catchment, I/O; `--tier large`) | L1-L5 | L1, L2, L2P, L4 | L1, L2, L2P, L3, L4 |
| multi-node / Slurm | untestable here | untestable here | untestable here |

```bash
source env.sh                                   # PYTHONPATH = the main checkout's library/
python matrix.py run BACKEND CASE [CONFIG]      # -> store/BACKEND/CONFIG/CASE/KEY.npz
python matrix.py table                          # store -> table.md (PASS / FAIL / n/a / unsupported)
python matrix.py run --tier large L1 jax mpi2   # the large tier (never part of the default run)
python deliverable.py                           # store -> matrix.png
```

BACKEND `jax foam amrex`; CASE `C1`..`C26`; CONFIG `serial` (default)
`mpi2 mpi4 gpu1 gpu2` (`gpu2` = gpu-mpi: 2 ranks, a GPU each).  A run in the
store is skipped (`--force` reruns), so a command that times out resumes.
A run that raises `NotImplementedError` is stored as **unsupported**, any
other exception as a **FAIL**; a backend without the config or the case is
**n/a**.  Stored per run: `t, Q, Q0, xc, vol, names, wall, lib` (the library
commits it ran on); custom runs (C15, C16, C22): their metrics.  Files:
`cases.py` (every case, defined once), `backends.py` (adapters),
`custom.py` (AD, preCICE), `matrix.py` (runner, checks, table), `large.py`
(large tier), `deliverable.py` (figure), `table.md` (the last table).

## Cases and pass criteria

Every case is built once (`cases.py`: mesh, model, IC, BCs, `t_end`, CFL,
MOOD, constant `dt`, solver options) by backend-neutral code (`prepare`: LSQ
mesh, `Q0`, spectral radius `eig`) and handed unchanged to each backend.
Models: SME(0), VAM(1) and the compressible Euler SystemModel
(`zoomy_core.systemmodel.euler`).  Sizes: 20-100 cells per side, <= ~100
steps; most runs take 5-30 s, the Chorin runs ~2.5 min (JAX: the setup of
the pressure template, ~130 s at any mesh size; foam/AMReX: printing).

| case | setup | checks (threshold, source) |
|---|---|---|
| C1 smooth | `common.bump_2d`, flow 0, walls, t 2.5, O1 / O2, nx 20/40/80 | L1(h) vs JAX nx 320 (8 ranks); order 40->80: O1 >= 0.6 (observed 0.71, pre-asymptotic), O2 >= 1.8 (theory); AMReX = JAX (round-off); foam-JAX difference shrinks (order >= 0.5, observed) |
| C2 shock | bump, flow (0.3, 0.1) into walls, O2 + MOOD, t 0.5, nx 40/80 | order 40->80 vs JAX nx 320 >= 0.6 (spec); foam L1 vs JAX <= 1.5e-3 (observed) |
| C3 well-balanced | lake at rest over b = 0.2 exp(-r^2/2), nx 24, dt 0.01 x 100 | max abs(u) <= 1e-12 (round-off) |
| C4 boundaries | channel 25 x 1 (ny 2), smooth bump, frictionless, t 3 from the exact state, nx 25/50/100. `sub`: q by `Dirichlet`, h by `Dirichlet`; `trans`: q by `InflowOutflow`, outflow `Extrapolation`; sides `Wall` | L1(h) vs Bernoulli, order 50->100: O1 >= 0.45 / 0.55 (observed, pre-asymptotic), O2 >= 1.8 (theory); max abs(q_y) <= 1e-12 (1-D flow); max abs(q/q_in - 1) (information: the open-domain mass balance) |
| C5 wet/dry | `common.dry_dambreak_2d`, nx 48, O2 eta + MOOD, CFL 0.45, t 0.02 | h_min >= 0; foam L1 vs JAX <= 3.5e-2 (observed); AMReX max rel <= 1e-9 (observed 6e-10, wet/dry eigenvalue cluster) |
| C6 conservation | every run of the closed cases (C1-C3, C5, C11-C14, C20, C23) on every config | rel. mass change <= 1e-12 |
| C7 parallel | C1 O2 nx 40, C5, C13, C14, C21 on 2 and 4 ranks | max rel vs the same backend serial <= 1e-12 |
| C8 pressure | VAM(1) Chorin, gate-21 channel (60 x 6), t 0.05: `vam` (linear), `vam_newton`, `vam_t2` (time_order 2), `vam_tri` (triangles) | foam L1 vs JAX <= 3e-2 (observed); AMReX max rel vs JAX <= 1e-9; Newton vs linear <= 1e-8; ranks / GPU vs serial <= 1e-9 (10 x the pressure rtol 1e-10, over the steps) |
| C9 GPU | C1, C5, C8, C14 on gpu1 / gpu-mpi | max rel vs the same backend CPU serial (round-off; C8 pressure tolerance) |
| C10 Chorin convergence | VAM(1), smooth wave (a 0.01, sigma 0.3) over the bump, no inflow, t 0.2, nx 30/60/120; space O1 (gate 7's config) and O2 minmod, time_order 1 / 2 | Richardson order log2(abs(h30-h60)/abs(h60-h120)): >= 0.8 (theory 1), space O2 + time 2 >= 1.5 (observed) |
| C11 1-D | smooth wave (walls) nx 40/80/160 vs JAX nx 1280; C4 `sub` in 1-D | as C1 / C4 (O1 >= 0.9, O2 >= 1.8: theory) |
| C12 triangles | C1 on triangles nx 10/20/40 (vs the C1 reference, sampled); C4 `sub` on triangles | O1 >= 0.5, O2 >= 1.4 (observed, coarse); C4 as above |
| C13 periodic | bump in a uniform flow 0.3, periodic in x, walls in y, t 1, at x0 = 2.5 and 7.5 | the two runs equal up to the shift by L/2 (round-off); foam O1 = JAX (round-off) |
| C14 IMEX | bump in flow + Manning n 0.5 (implicit source), t 0.3, nx 20/40/80 vs JAX nx 160 | order >= 0.5 (O1, observed) / 0.8 (O2, Lie splitting); mass; parallel; GPU |
| C15 AD (JAX) | hyperbolic and IMEX bump 12^2, Chorin C10 wave 30 x 3; dJ/dg | abs(FD - jvp) <= 1e-5 (rel); abs(rev - jvp) <= 1e-10; mpi2 / gpu1 jvp vs serial <= 1e-12 |
| C16 preCICE | gates 28 (jax-jax), 10 (jax-foam), 26 (foam-foam): 2 x 100 cells, O1, window 1e-3 | coupled vs single domain and mass drift <= 1e-14 (gate 26) |
| C17 AMR (AMReX) | gate 37: covering fine level, static patch (Euler wave), box with / without reflux, lake, regrid on the dry front | covering = fine single level, mass / energy with reflux, lake abs(u) <= 1e-12; wave order >= 1.5 |
| C18 Euler Sod 3-D | n x 2 x 2 along x, y, z, HLL, minmod, t 0.2, O1 / O2 + MOOD, n 50/100 | L1(rho) vs exact (Toro ch. 4) at n 100: O1 <= 4.2e-2, O2 <= 1e-2; order >= 0.45 / 0.8 (observed; discontinuous); x / y / z profiles and u_perp <= 1e-12 |
| C19 Euler wave 3-D | periodic [0,1]^3, u (1, .5, .25), n 8/16/24, t 0.1, O1 / O2 none / O2 minmod | order 16->24: O1 >= 0.75 (observed), O2 >= 1.8 (theory); u and p constant (round-off) |
| C20 Euler walls 3-D | 16^3, centred pulse, t 0.6, O1 / O2 + MOOD | mass, energy, mirror / permutation symmetry <= 1e-12 |
| C21 Euler MPI 3-D | pulse 12^3 (centre 0.4) O1, O2 + MOOD; Sod along y 60 x 4 x 4 | 2 / 4 ranks = serial (round-off) |
| C22 Euler AD 3-D | pulse 10^3, dJ/dgamma; O1 dt 5e-3, O1 adaptive, O2 + MOOD dt 5e-3 | abs(FD - jvp) <= 1e-6 (observed 4.4e-8); abs(rev - jvp) <= 1e-12 |
| C23 Euler cross-backend | acoustic pulse (0.01, 0.04) in the C20 box, t 0.3, n 8/16/24 | O1 = JAX (round-off); O2: L1 abs(backend - JAX) order 16->24 >= 1.5 (observed) |
| C24 Euler 2-D | Sod along x, y (n x 2); periodic wave [0,1]^2, u (1, .5), n 16/32/64 | as C18 / C19; wave O2 none >= 1.8, minmod >= 1.7 (observed), O1 >= 0.85 |
| C26 implicit diffusion | gate 31: SME(0) + NewtonianInPlane (dense, state-dependent diffusion_matrix), nu 1e-3, 20^2, IMEX (Crank-Nicolson Newton), t 0.05 | ranks / GPU vs serial <= 1e-12; AMReX vs JAX <= 1e-8 (Newton / GMRES tolerances) |

Sources: *theory* (order, positivity, symmetry), *tolerance* (round-off
1e-12, the pressure solve), *spec* (the task's number), *observed* (about 2x
the worst value seen, or ~0.8 x an observed pre-asymptotic order).  Observed
thresholds are regression bounds for the next backend, not proofs.  Orders
are raw log(e_coarse / e_fine) / log(refinement).

## Large tier (`python matrix.py run --tier large ...`, `large.py`)

Never in the default run; each command <= 5 min.  L1 weak scaling (C1 bump
O2, 200^2 cells per rank, 1/2/4 ranks, 20 constant steps: ms/step); L2
strong scaling (1600^2 = 2.56M cells); L2P pressure strong scaling (VAM(1)
channel 3200 x 320 = 1M cells; JAX pyamg on CPU ranks, AmgX on GPUs; foam
GAMG; AMReX); L3 catchment (JAX: the 14.8M-cell 1 m mesh on 4 GPUs, a
time window; AMReX: its 10 m grid driver); L4 setup / compile / memory vs
mesh size; L5 parallel mesh + field write of the 14.8M-cell mesh.
Results: the "Large tier" section of `table.md`.

## Adding a backend

Write one class in `backends.py` and add it to `BACKENDS` (JAX: 42 lines,
foam: 23, AMReX: 20):

```python
class MyBackend:
    configs = ("serial", "mpi2", "mpi4")     # what it supports
    def launch(self, config):                # argv prefix to re-run matrix.py
        return None                          # under (mpiexec, an env); None: in-process
    def run(self, r, mesh, systems, Q0, config, work):
        """March cases.Run r from Q0 (rows = state, columns = mesh cells);
        solver settings: _solver_kw(r, cls) (t_end, cfl, mood, dt, imex,
        time_order, pressure_solve).  systems: the NSM, or (pred, press,
        corr) if r.chorin.  Return (t, Q) in the mesh's cell order; None on
        non-root MPI ranks.  Raise NotImplementedError for what the backend
        does not support."""
```

Then `python matrix.py run mybackend CASE` for every case and `python
matrix.py table`: accepted when every row passes.  The checks know nothing
about the backend; a backend-specific threshold is a defect of the backend
or of the case, to be argued here.

## Environment

`env.sh`: the main checkout's libraries on PYTHONPATH, `PYTHONNOUSERSITE=1`,
`JAX_PLATFORMS=cpu`, x64, AMReX build and run on cores 0-7, an own AMReX
build cache (`work/amrex_cache`).  Every command: `timeout 295 taskset -c
0-7`.  JAX `mpiN` re-launches under `mpiexec -n N` (mpi4jax), `gpu1` in the
`zoomy-cuda` env, `gpu2` under `mpiexec -n 2 python -m zoomy_jax.gpurank`;
foam `mpiN` = `decomposePar` + `-parallel` (OF13 container); AMReX `mpiN`,
`gpuN` = its container builds.  The fine references use JAX `mpi8` (C7:
parallel = serial).  The preCICE gates write their run directories in
`gates/`.

Not part of the matrix: 8 model goldens in zoomy_core (m01-m04, m06-m08,
s01) fail on the base checkout too (pre-existing).  The diffusion_matrix
sign fix (cfc19a85) changes no matrix case except C26: every other model
here has nu = 0.

## Results (table.md, 2026-10-12)

Libraries: jax b59227c, amrex a9ef6c2, foam 3c51d7b (superrepo 9dd4420f + 42df7e8e).
Every run in the store ran on these heads (older runs were deleted by
library commit).  Wrapped up before the end of the full run:

- **Run:** JAX C1-C5, C8, C10-C16, C18-C24, C26 (serial, mpi2/4, gpu1, part of gpu-mpi);
  AMReX C1-C5, C8, C10, C11, C13, C14, C16, C17 (serial), C18-C24, C26 (serial, mpi2/4).
- **Not run on these heads:** foam (every case; its earlier numbers are in git
  history of table.md only), AMReX C17 mpi2 / gpu1, AMReX gpu configs, JAX C22 gpu,
  the large tier (`large.py`, never executed in full).
- **Failures (164 PASS, 8 FAIL):** AMReX vs JAX not at round-off on open
  boundaries (C4 1.7e-3, C11 4.5e-6) and periodic O1/O2 vs JAX (C13 3.9e-3;
  both are shift-invariant to 0); JAX gpu1 Chorin AD is NaN (C15, eig spectral
  radius on the GPU).
- C10 (VAM(1) Chorin, Richardson): JAX and AMReX ~2.0 for every
  space/time combination at nx 30/60/120 (the error is spatial; time_order 1
  does not limit it here).
- Runtime: a Chorin run costs ~2.5 min whatever the mesh (JAX: ~130 s setup
  of the pressure template; foam / AMReX: printing), so C8 / C10 / C15 dominate.
