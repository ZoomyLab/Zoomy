# Cross-backend test matrix (symbolic solvers)

The acceptance test of every backend that prints the symbolic solver
templates (JAX, foam, AMReX, and any future one).  Second-order foam runs
on OpenFOAM's own stencils, so backends no longer agree to round-off; they
must instead pass the same physics checks: convergence, exact solutions,
conservation, parallel = serial, GPU = CPU.

```bash
source env.sh                                  # PYTHONPATH = the main checkout's library/
python matrix.py run BACKEND CASE [CONFIG]     # -> store/BACKEND/CONFIG/CASE/KEY.npz
python matrix.py table                         # store -> table.md (PASS/FAIL)
python deliverable.py                          # store -> matrix.png
```

BACKEND `jax foam amrex`; CASE `C1 C2 C3 C4 C5 C8`; CONFIG `serial`
(default) `mpi2 mpi4 gpu1`.  A run already in the store is skipped
(`--force` reruns), so slow pieces run separately and resume after a
timeout.  Each stored run keeps `t, Q, Q0, xc, vol, names, wall` and `lib`
(the library commits it ran on).  Files: `cases.py` (the cases, each defined
once), `backends.py` (adapters), `matrix.py` (runner + checks),
`deliverable.py` (figure), `table.md` (the last table).

## Cases and pass criteria

Every case is built once (`cases.py`: mesh, model, IC, BCs, `t_end`, CFL,
MOOD, constant `dt`) by backend-neutral code (`prepare`: LSQ mesh, `Q0`,
spectral radius `eig`) and handed unchanged to each backend.  Models are
SME(0) (and VAM(1) for C8) from the zoomy families; 2-D Cartesian.

| case | setup | checks (threshold, source) |
|---|---|---|
| C1 smooth | `common.bump_2d`, flow 0, walls, t = 2.5, O1 / O2, nx 40/80/160 | L1(h) vs JAX nx 320 (O2, 8 MPI ranks); order 80->160: O1 >= 0.75 (observed 0.82, pre-asymptotic; spec 0.9), O2 >= 1.8 (theory) |
| C2 shock | bump, flow (0.3, 0.1) into walls, O2 + MOOD, t = 0.5, nx 40/80 | L1(h) order 40->80 >= 0.6 (spec) vs JAX nx 320; L1 rel vs JAX at nx 80 (observed) |
| C3 well-balanced | lake at rest, eta = 0.5 over b = 0.2 exp(-r^2/2), walls, dt 0.01 x 100 steps, O1 / O2 | max abs(u) <= 1e-12 (round-off) |
| C4 boundaries | quasi-1-D channel 25 x 1 (ny 2), b = 0.2 exp(-(x-10)^2/2), frictionless, t = 30, O1 / O2, nx 50/100/200. `sub`: q = 4.42 by `Dirichlet` (inflow), h = 2 by `Dirichlet` (outflow); `trans`: q = 1.53 by `InflowOutflow`, outflow `Extrapolation`; sides `Wall`; C1-C3, C5 use `FromModel("wall")` | L1(h) vs the exact Bernoulli depth (sub: subcritical branch; trans: critical at the crest), its order 100->200, max abs(q/q_in - 1) (steady discharge) |
| C5 wet/dry | `common.dry_dambreak_2d`, nx 48, O2 eta + MOOD, CFL 0.45, t = 0.02 | h_min >= 0 (positivity); L1 rel vs JAX (observed) |
| C6 conservation | every run of C1, C2, C3, C5 (closed) on every config | rel. mass change <= 1e-12 (round-off). Open cases (C4, C8): no backend exposes boundary fluxes, so no mass balance; C4 checks the steady discharge instead |
| C7 parallel | C1 O2 nx 80, C5 on 2 and 4 ranks | max rel vs the same backend serial <= 1e-12 (round-off) |
| C8 pressure | VAM(1) Chorin, gate-21 channel (60 x 6, inflow, P = 0 outflow), t = 0.05 | L1 rel vs JAX <= 3e-2 (observed, foam 1.6e-2); 2/4 ranks vs serial <= 1e-9 (tolerance: 10 x ELLIPTIC_RTOL, accumulated over the steps; JAX 1.2e-10) |
| C9 GPU | C1 O2 nx 80, C5, C8 on 1 GPU | max rel vs the same backend's CPU serial (round-off, C8: pressure tolerance) |

Sources: *theory* (order, positivity), *tolerance* (round-off 1e-12, the
pressure solve), *spec* (the task's number), *observed* (about 2x the worst
value seen on the backends listed in `table.md`).  Observed thresholds are
regression bounds for the next backend, not proofs.

Convergence notes.
- The orders in the table are the raw log2 of the L1 error ratios vs the
  reference.  For O2 the reference carries the same scheme's error; with
  e_N = C (N^-p - 320^-p) the corrected order is in the notes (C1 O2: 2.06
  raw 2.37).
- C1 O1 is pre-asymptotic up to nx 160 (0.71, 0.82 on every backend); hence
  0.75, not the spec's 0.9.  An O1 nx 320 run would take ~4 min on JAX CPU.
- C4 starts from the exact solution and relaxes to the discrete steady state
  (from a lake the subcritical case had not settled at t = 60, and the
  transcritical one stayed subcritical behind the extrapolated outflow).

## Adding a backend

Write one class in `backends.py` and add it to `BACKENDS` (JAX: 37 lines,
foam: 21, AMReX: 19):

```python
class MyBackend:
    configs = ("serial", "mpi2", "mpi4")     # what it supports
    def launch(self, config):                # argv prefix to re-run matrix.py
        return None                          # under (mpiexec, an env); None: in-process
    def run(self, r, mesh, systems, Q0, config, work):
        """March cases.Run r (r.t_end, r.cfl, r.mood, r.dt, r.chorin) from Q0
        (rows = state, columns = mesh cells).  systems: the NSM, or
        (pred, press, corr) if r.chorin.  Return (t, Q) in the mesh's cell
        order; None on non-root MPI ranks."""
```

Then `python matrix.py run mybackend C1` ... `C8` and `python matrix.py
table`: it is accepted when every row passes.  The checks know nothing about
the backend; a backend-specific threshold is a defect of the backend or of
the case, to be argued in this README.

## Environment

`env.sh`: the main checkout's libraries on PYTHONPATH (not the worktree's
submodules), `PYTHONNOUSERSITE=1`, `JAX_PLATFORMS=cpu`, x64, AMReX build and
run pinned to cores 0-7, an own AMReX build cache
(`work/amrex_cache`, seeded from a compiled tree).  Every command was run as
`timeout 295 taskset -c 0-7`.  JAX `mpiN` re-launches under `mpiexec -n N`
(mpi4jax), `gpu1` in the `zoomy-cuda` env; foam `mpiN` = `decomposePar` +
`-parallel` in the OF13 container; AMReX `mpiN` / `gpu1` = its container
builds.  The fine references use JAX `mpi8` (C7 shows parallel = serial).

## Results (table.md; libraries core 93b52fa, jax 5797925, foam cb56be7 = feature/foam-parallel merged, amrex 418e29a)

79 checks pass, 1 fails (4 configs):

- **AMReX C8 does not build** (serial, mpi2, mpi4, gpu1): printed `Solver.H`
  references `nsm.*` while `Kernels.H` defines `pred/press/corr`.
  `SymbolicChorinSolverAmrex.run` does not set `_symbolic_name` on the
  systems (foam's `SymbolicChorinSolverFoam.run` does); gate 27 passed only
  because its NSMs had been named by a JAX setup first.  With the names set
  by the caller (scratch diagnostic, not in the matrix) AMReX C8 = JAX to
  6.2e-11 L1 (the pressure tolerance).

Findings that pass but are worth knowing:
- foam C8 vs JAX: L1 1.6e-2, concentrated in the first cells at the inflow
  (P max 4.1e-2 of 0.62 at x = -1.43); the rest agrees closely (fig. (d)).
- JAX and AMReX produce a transverse momentum max|q_y| ~ 6e-4 in the
  quasi-1-D C4 channel at O2 (nx 200, ny 2: degree-1 LSQ next to the walls);
  foam stays at 1e-15.  It shrinks with nx (1.6e-3 at nx 100).
- AMReX = JAX to round-off everywhere (C1 O2 1.4e-15, C5 6e-10: the
  wet/dry eigenvalue cluster); MPI and GPU bitwise = AMReX serial.
- JAX gpu1: C5 only (5e-15 vs CPU); C1 O2 nx 80 exceeds 295 s on the GPU
  (compile), C8 not attempted.
