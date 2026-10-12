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
| IMEX time order: Lie vs IMEX-SSP2(2,2,2), hyperbolic and Chorin | C31 | C31 (SSP2 = JAX) | C31 (SSP2 = JAX) |
| AD: jvp vs FD, reverse vs forward (hyperbolic, IMEX, Chorin; 3-D Euler) | C15, C22 | n/a | n/a |
| preCICE: coupled = single domain | C16 (jax-jax, jax-foam) | C16 (foam-foam, jax-foam) | C16 (amrex-amrex, amrex-jax) |
| AMR (reflux, regrid, one dt; gate 37) | n/a | n/a | C17 |
| compressible Euler 3-D: Sod, smooth wave, walls, MPI | C18-C21 | C18-C21 | C18-C21 |
| Euler 3-D cross-backend (O1 round-off, O2 converging) | C23 | C23 | C23 |
| compressible Euler 2-D | C24 | C24 | C24 |
| SSP-RK time order 1 / 2 / 3 (gate 40) | C27 | C27 | C27 |
| diffusion blocks at boundaries: orders (gate 41, JAX); foam / AMReX = JAX (gate 43) | C28 | C28 | C28 (1-D, quads; triangles n/a: Cartesian) |
| LSQ derivatives with the boundary closure (gate 44) | C29 | n/a (JAX-runtime kernels) | n/a (JAX-runtime kernels) |
| LSQ polynomial reproduction (gate 39) | C30 | n/a (JAX-runtime kernels) | n/a (JAX-runtime kernels) |
| large tier (scaling, catchment, I/O; `--tier large`) | L1-L5 | L1, L2, L2P, L4 | L1, L2, L2P, L3, L4 |
| multi-node / Slurm | untestable here | untestable here | untestable here |

```bash
source env.sh                                   # PYTHONPATH = the main checkout's library/
python matrix.py run BACKEND CASE [CONFIG]      # -> store/BACKEND/CONFIG/CASE/KEY.npz
python matrix.py table                          # store -> table.md (PASS / FAIL / n/a / unsupported)
python matrix.py run --tier large L1 jax mpi2   # the large tier (never part of the default run)
python deliverable.py                           # store -> matrix.png
```

BACKEND `jax foam amrex`; CASE `C1`..`C31`; CONFIG `serial` (default)
`mpi2 mpi4 gpu1 gpu2` (`gpu2` = gpu-mpi: 2 ranks, a GPU each).  A run in the
store is skipped (`--force` reruns), so a command that times out resumes.
A run that raises `NotImplementedError` is stored as **unsupported**, any
other exception as a **FAIL**; a backend without the config or the case is
**n/a**.  Stored per run: `t, Q, Q0, xc, vol, names, wall, lib` (the library
commits it ran on); custom runs (C15-C17, C22, C27-C30): their metrics.  Files:
`cases.py` (every case, defined once), `backends.py` (adapters),
`custom.py` (AD, preCICE, AMR, the wrapped gates 39-44), `matrix.py` (runner, checks, table), `large.py`
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
| C12 triangles | C1 on triangles nx 10/20/40 (vs the C1 reference, sampled); C4 `sub` on triangles refined in x and y (ny 1/2/4; at a fixed ny foam O2 loses its order on the stretched wall triangles) | O1 >= 0.5, O2 >= 1.4 (observed, coarse); C4 as above |
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
| C27 time order | gate 40 `forced_swe` (SME(0), 1-D walls, first-order space, dh/dt = a cos(w t + 0.3) h), constant dt, time_order 1 / 2 / 3. `ode_t*`: 8 cells at rest, T 1, dt T/16, T/32, T/64, error vs the exact h(T); `wave_t*`: 100 cells, T 0.04, dt 2e-3 / 2^k, k = 0, 1, 2, error vs k = 4. JAX runs the gate's own `run`; foam / AMReX the same NSM and mesh through their drivers (`time_order` of `backend_cxx`). `ode_control` (JAX): RK3 with the last stage at the wrong time | order of the last pair >= k - 0.2 (theory k; observed 1.01 / 1.92 / 3.00 ODE); control order <= 2.5 (gate 40; observed 0.99) |
| C28 diffusion BCs | heat equation u_t = div(nu grad u) on [0,1]^d (gate 41's SystemModel), cases dir0 / neu / dirL / per / steady, explicit and implicit (IMEX), on 1-D, quads, triangles. `orders_*` (JAX): L_inf at nx 16 / 32 / 64 (gate 41); `{kind}_{path}_{case}` (foam, AMReX): nx 16 against JAX, the same Q0 (gate 43; AMReX has no triangles) | L_inf order 32->64 >= 1.8 (theory 2); steady max abs(u - L) <= 1e-12; foam / AMReX vs JAX: explicit <= 1e-12, implicit <= 1e-9 (gate 43's tolerances: round-off; Newton 1e-10, GMRES 1e-13) |
| C29 LSQ boundary closure | gate 44 (JAX, quads, default set): `Lambda` (Dirichlet) and `Flux` (Neumann) BCs on SME(0), h = f(x, y), nx 12 / 24 / 48, fit degree p = 2, 3; max error of the derivatives of order k = 1..p over the boundary and interior cells, with the closure | order 24->48 >= p + 1 - k - 0.2 for every k, boundary and interior (the gate's criterion); the ghost-as-sample errors are in the notes |
| C30 LSQ polynomials | gate 39 (subprocess): seeded random polynomials of degree p = 1..3 on 1-D, quads, jittered quads, triangles, hexahedra; interior, boundary with exact mirror ghost, Dirichlet / Neumann / zero-gradient ghosts through the closure | number of failing checks (PASS + FAIL = 72) = 0, each exact to 1e-9 |
| C31 IMEX time order | constant dt (`cfl_check_every=0`), fixed mesh; schemes `lie1` / `lie2` (Lie: SSP-RK1 / RK2, then the implicit source), `ssp2` (IMEX-SSP2(2,2,2), Pareschi & Russo 2005: the implicit source in every stage). `swe_*`: C14 (bump in flow, Manning n 0.5) at space O1, nx 40, t 0.3, dt 4e-3 / 2^k, k = 0..3, L1(h) vs dt / 32 of the same scheme. `vam_*`: C10's wave + Manning n 0.5 (VAM(1) Chorin, space O1, nx 60, t 0.2), dt 8e-3 / 2^k, Richardson; `t1` / `t2` = time_order 1 / 2 with the Lie source per projection stage, `ssp2` = projected IMEX stages (ChorinTemplate.step); `pred_ssp2_*`: the VAM predictor alone (no projection). foam / AMReX: `swe_ssp2_k0`; AMReX: `vam_ssp2_k0` | SWE order dt/4->dt/8: lie1 >= 0.8, ssp2 >= 1.8 (theory), lie2 <= 1.4 (theory 1: the Lie splitting caps SSP-RK2); VAM: t1 >= 0.8, predictor ssp2 >= 1.8 (theory), t2 >= 0.5, ssp2 >= 0.7 (observed: the approximate projection, below); vs JAX: SWE round-off, VAM 1e-9 (pressure) |

Sources: *theory* (order, positivity, symmetry), *tolerance* (round-off
1e-12, the pressure solve), *spec* (the task's number), *observed* (about 2x
the worst value seen, or ~0.8 x an observed pre-asymptotic order).  Observed
thresholds are regression bounds for the next backend, not proofs.  Orders
are raw log(e_coarse / e_fine) / log(refinement).

## Large tier (`python matrix.py run --tier large ...`, `large.py`)

Never in the default run; each command <= 5 min.  L1 weak scaling (C1 bump
O2, 200^2 cells per rank, 1/2/4 ranks, 20 constant steps: ms/step); L2
strong scaling (1600^2 = 2.56M cells, 5 steps); L2P pressure strong scaling
(VAM(1) channel 3200 x 320 = 1M cells; JAX pyamg on CPU ranks, AmgX on GPUs;
foam Schur-GAMG; AMReX GMRES + cell-block Jacobi); L3 catchment (JAX: the
14.8M-cell 1 m mesh on 4 GPUs, a 5 s window, reads the mesh L5 writes: run
L5 first; AMReX: `gates/catchment_amrex.py`, the 10 m raster with nodata
cells as a bed-1e4 wall, against JAX on the same mesh); L4 setup / compile /
memory vs mesh size; L5 parallel mesh + field write of the 14.8M-cell mesh.
Each backend uses its default wave speed (JAX: eig on the CPU, power
iteration on a GPU: JAX's eig is a host callback there).  A timed-out AMReX
run can log a SIGBUS: `timeout` kills apptainer, which unmounts the image
under the still-running ranks.
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

Libraries: core 6afd79d, jax 8a163d3, foam c6ed617, amrex eb9be98.
The whole store rerun from empty on these heads (the previous one is kept
as `store_prev_20261012`); every backend on every config it has (serial,
mpi2/4/8, gpu1, gpu2).  **459 PASS, 0 FAIL** (ANALYSIS 42-44).

- Fixed during the rerun: foam C16 (`build` without a topology), AMReX C17
  off cpu1 (shared work directory), foam C10 n120 (Schur-GAMG divergence ->
  Jacobi fallback), C12 `sub` refined in x and y, foam = JAX at round-off
  counted as a pass, AMReX = JAX tolerances from JAX's measured round-off
  sensitivity (`*_pert` runs: C4 trans O2 1.3e-11, C11 sub O2 2.0e-5,
  bounded limiter chattering), C15 Chorin AD with power-iteration wave
  speeds (the eig tangent is ~1e15 at a defective eigenvalue: NaN on gpu1).
- Large tier (first full run; ms/step at 1/2/4 ranks or GPUs):
  - L1 weak, 200^2 per rank: JAX CPU 380/423/389, GPU 2.9/3.4/3.6; AMReX CPU
    845/898/809, GPU 12.9/14.7/13.5; foam 1387/1380/1380.
  - L2 strong, 2.56M cells: JAX CPU 18681/10946/5909 (85 / 79 %), GPU
    253/127/63 (100 %); AMReX CPU 36260/19712/10938 (92 / 83 %), GPU
    433/221/112 (98 / 97 %); foam mpi4 17962 (serial, mpi2 > 5 min).
  - L2P pressure, 1M cells: only JAX on GPUs (AmgX) fits, 1410/1338/1092
    (53 / 32 %); AMReX (GMRES + cell-block Jacobi, no multigrid), foam
    (Schur-GAMG) and JAX pyamg on CPUs do not fit 5 min.
  - L3: JAX 14.8M cells on 4 GPUs, 5 s in 6.7 s (~29 ms/step); AMReX 10 m
    (292k cells) = JAX on the same mesh to 6.5e-15, water = rain to 1.5e-15,
    cpu1 221, cpu4 62, gpu1 1.4, gpu4 0.6 ms/step.
  - L4: memory linear in cells (JAX 15.8 GB, AMReX 5.9 GB at 2.56M); L5
    parallel write 14.8M cells 55 / 35 / 27 s.
  - Fixed: the tier forced eig wave speeds, a host callback on a JAX GPU
    (L1/L2 GPU > 5 min, now 2.9 ms/step); L2 at 5 steps; the AMReX catchment
    driver (never committed) written.  Open: CPU O2 cost per cell (AMReX 24,
    foam 35 us/cell/step vs 0.44 for AMReX O1 on the catchment); pressure
    preconditioners of AMReX and foam at 1M cells.
- C27-C30 (wrapped gates 40, 41, 43, 44, 39; added after the full run, JAX / foam / AMReX
  on the same heads): 130 rows PASS, 0 FAIL, 5 n/a.  C27 orders 1.01 / 1.92 / 3.00
  (ODE) and 1.26 / 2.08 / 3.02 (wave) on all three backends (the foam / AMReX errors
  equal JAX's to round-off); C28 L_inf orders 2.00 (JAX), foam / AMReX vs JAX <= 5e-15
  explicit, <= 3e-10 implicit; C29 boundary / interior orders >= p+1-k-0.2; C30
  PASS=72 FAIL=0.  Runtime: C27 20-40 s per run (C++ drivers: printing and build per dt),
  C28 foam ~25 s per case, AMReX ~16 s, C29 ~25 s per (p, BC).
- C10 (VAM(1) Chorin, Richardson): JAX and AMReX ~2.0 for every
  space/time combination at nx 30/60/120 (the error is spatial; time_order 1
  does not limit it here).
- C31 (IMEX time order; added 2026-10-12, JAX / foam / AMReX on dev/imex): SWE + Manning
  orders 1.06 / 1.10 / 1.22 for Lie with RK1 and with RK2 alike (the splitting; the last
  pair is lifted by the dt/32 reference), 2.01 / 2.02 / 2.08 for IMEX-SSP2; foam and
  AMReX = JAX to 2.4e-14. VAM(1): the predictor alone with IMEX-SSP2 2.02 / 2.01, but
  with the projection t1 1.01 / 0.98, t2 0.81 / 0.67, ssp2 1.15 / 0.91 (L1(h);
  max-norm ~0.5, largest at the closed left end); AMReX = JAX to 7e-12. The cap is
  the projection, not the scheme: the VAM projection is approximate, P(P(U)) - P(U)
  is 2.5 / 0.74 / 0.23 % of P(U) - U at nx 30 / 60 / 120 (the discrete constraint is
  not exactly met by the corrected state), so every projection-stage scheme keeps an
  O(dt) error at a fixed mesh. Without the stage pressure in the implicit tableau
  the max error is ~3x larger (L1(h) 1.2-1.6x), the orders the same. Manning n 0.5
  is not stiff at these dt (rate x dt < 3e-3); n 0 gives the same VAM orders.
- Runtime: a Chorin run costs ~2.5 min whatever the mesh (JAX: ~130 s setup
  of the pressure template; foam / AMReX: printing), so C8 / C10 / C15 dominate.
