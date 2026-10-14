# Cross-backend test matrix (symbolic solvers)

The acceptance test of every backend that prints the symbolic solver
templates (JAX, foam, AMReX, and any future one).  Backends no longer agree
to round-off (foam uses OpenFOAM's own stencils at second order); every
backend must pass the same physics checks instead: convergence, exact
solutions, conservation, symmetry, parallel = serial, GPU = CPU.

**NumPy** (`zoomy_core.symsolver.numpy_backend`, the default install: no JAX, no
compiler) prints the same templates as JAX and runs serially.  It runs every
serial JAX run except the fine JAX references the checks read, the `*_pert`
sensitivity runs, the gate-40 control, AD (C15, C22), preCICE (C16) and AMR
(C17); gate 39 (C30) checks the fit numpy takes (`GATE39_BACKEND=numpy`: core
`lsq_derivatives` on the topology); its own order / conservation
checks are JAX's, plus one row per case `numpy = JAX (max rel, worst run)`:
round-off (1e-12, or 10 x JAX's `*_pert` sensitivity), or the case's
implicit-solve tolerance (C8, C10, C31 VAM: 1e-9; C14, C26, C31 SWE, C32: 1e-8;
C33 imex: 1e-10).  `Root` uses a central-difference cell Jacobian, implicit
diffusion a finite-difference Newton-GMRES (JAX: exact AD).

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
| in-plane viscosity with viscous walls vs an exact solution (no-slip / free-slip shear decay) | C32 | C32 | C32 |
| BC channels: `face=` / `ghost=` set apart (`Wall` face, `Dirichlet` ghost: a viscous no-slip wall) | C32 | C32 | C32 |
| vertical viscosity of the moment model (SME(2)) vs the exact `expm`; time order of the IMEX / SSP-RK source | C33 | C33 | C33 |
| in-plane viscous operator vs the exact div(h tau'), tau' = tau - tau_zz I, mixed derivatives (gate 45) | C34 | C34 (= JAX) | C34 (= JAX) |
| restart (`dt_checkpoint`, `restart_from`): restarted = continuous, bitwise, serial and distributed | C35 | C35 | C35 |
| the full Newtonian stress at level 1 (`Stress.full`: complete tau_ez, Marche (3.11) pressure) runs and conserves mass | C36 | n/a | n/a |
| large tier (scaling, catchment, I/O; `--tier large`) | L1-L5 | L1, L2, L2P, L4 | L1, L2, L2P, L3, L4 |
| multi-node / Slurm | untestable here | untestable here | untestable here |

```bash
source env.sh                                   # PYTHONPATH = the main checkout's library/
python matrix.py run BACKEND CASE [CONFIG]      # -> store/BACKEND/CONFIG/CASE/KEY.npz
python matrix.py table                          # store -> table.md (PASS / FAIL / n/a / unsupported)
python matrix.py run --tier large L1 jax mpi2   # the large tier (never part of the default run)
python deliverable.py                           # store -> matrix.png
```

BACKEND `jax foam amrex numpy`; CASE `C1`..`C36`; CONFIG `serial` (default)
`mpi2 mpi4 gpu1 gpu2` (`gpu2` = gpu-mpi: 2 ranks, a GPU each).  A run in the
store is skipped (`--force` reruns), so a command that times out resumes.
A run that raises `NotImplementedError` is stored as **unsupported**, any
other exception as a **FAIL**; a backend without the config or the case is
**n/a**.  Stored per run: `t, Q, Q0, xc, vol, names, wall, lib` (the library
commits it ran on); custom runs (C15-C17, C22, C27-C30, C34, C35): their metrics.  Files:
`cases.py` (every case, defined once; jax-free, like the gate setups it uses, so a case
is built in the NumPy / foam / AMReX images too), `backends.py` (adapters),
`custom.py` (AD, preCICE, AMR, restart, the wrapped gates 39-45), `matrix.py` (runner, checks, table), `large.py`
(large tier), `deliverable.py` (figure), `table.md` (the last table).

## Cases and pass criteria

Every case is built once (`cases.py`: mesh, model, IC, BCs, `t_end`, CFL,
MOOD, constant `dt`, solver options) by backend-neutral code (`prepare`: LSQ
mesh, `Q0`; the wave speed is the solvers' default, ANALYSIS 48) and handed unchanged to each backend.
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
| C6 conservation | every run of the closed cases (C1-C3, C5, C11-C14, C20, C23, C32, C33, C36) on every config | rel. mass change <= 1e-12 |
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
| C26 implicit diffusion | gate 31: SME(0) + Newtonian, `stress=Stress.full` (dense, state-dependent diffusion_matrix), nu 1e-3, 20^2, IMEX (Crank-Nicolson Newton), t 0.05 | ranks / GPU vs serial <= 1e-12; AMReX vs JAX <= 1e-8 (Newton / GMRES tolerances) |
| C27 time order | gate 40 `forced_swe` (SME(0), 1-D walls, first-order space, dh/dt = a cos(w t + 0.3) h), constant dt, time_order 1 / 2 / 3. `ode_t*`: 8 cells at rest, T 1, dt T/16, T/32, T/64, error vs the exact h(T); `wave_t*`: 100 cells, T 0.04, dt 2e-3 / 2^k, k = 0, 1, 2, error vs k = 4. JAX runs the gate's own `run`; foam / AMReX the same NSM and mesh through their drivers (`time_order` of `backend_cxx`). `ode_control` (JAX): RK3 with the last stage at the wrong time | order of the last pair >= k - 0.2 (theory k; observed 1.01 / 1.92 / 3.00 ODE); control order <= 2.5 (gate 40; observed 0.99) |
| C28 diffusion BCs | heat equation u_t = div(nu grad u) on [0,1]^d (gate 41's SystemModel), cases dir0 / neu / dirL / per / steady, explicit and implicit (IMEX), on 1-D, quads, triangles. `orders_*` (JAX): L_inf at nx 16 / 32 / 64 (gate 41); `{kind}_{path}_{case}` (JAX, foam, AMReX, numpy): nx 16, the same Q0, each backend's run stores its final field and the table compares it with JAX's (gate 43; AMReX has no triangles) | L_inf order 32->64 >= 1.8 (theory 2); steady max abs(u - L) <= 1e-12; foam / AMReX vs JAX: explicit <= 1e-12, implicit <= 1e-9 (gate 43's tolerances: round-off; Newton 1e-10, GMRES 1e-13) |
| C29 LSQ boundary closure | gate 44 (JAX, quads, default set): `Lambda` (Dirichlet) and `Flux` (Neumann) BCs on SME(0), h = f(x, y), nx 12 / 24 / 48, fit degree p = 2, 3; max error of the derivatives of order k = 1..p over the boundary and interior cells, with the closure | order 24->48 >= p + 1 - k - 0.2 for every k, boundary and interior (the gate's criterion); the ghost-as-sample errors are in the notes |
| C30 LSQ polynomials | gate 39 (subprocess): seeded random polynomials of degree p = 1..3 on 1-D, quads, jittered quads, triangles, hexahedra; interior, boundary with exact mirror ghost, Dirichlet / Neumann / zero-gradient ghosts through the closure | number of failing checks (PASS + FAIL = 72) = 0, each exact to 1e-9 |
| C31 IMEX time order | constant dt (`cfl_check_every=0`), fixed mesh; schemes `lie1` / `lie2` (Lie: SSP-RK1 / RK2, then the implicit source), `ssp2` (IMEX-SSP2(2,2,2), Pareschi & Russo 2005: the implicit source in every stage). `swe_*`: C14 (bump in flow, Manning n 0.5) at space O1, nx 40, t 0.3, dt 4e-3 / 2^k, k = 0..3, L1(h) vs dt / 32 of the same scheme. `vam_*`: C10's wave + Manning n 0.5 (VAM(1) Chorin, space O1, nx 60, t 0.2), dt 8e-3 / 2^k, Richardson; `t1` / `t2` = time_order 1 / 2 with the Lie source per projection stage, `ssp2` = projected IMEX stages (ChorinTemplate.step); `pred_ssp2_*`: the VAM predictor alone (no projection). foam / AMReX: `swe_ssp2_k0`, `vam_ssp2_k0` | SWE order dt/4->dt/8: lie1 >= 0.8, ssp2 >= 1.8 (theory), lie2 <= 1.4 (theory 1: the Lie splitting caps SSP-RK2); VAM: t1 >= 0.8, t2 >= 0.8 (theory 1: the Lie source per stage), predictor ssp2 >= 1.8, ssp2 >= 1.8 (theory 2, with the increment projection); vs JAX: SWE round-off, VAM 1e-9 (pressure). The VAM projection is approximate by construction (ANALYSIS 45: the pressure rows are the constraint with the corrector substituted before the discretization, a compact operator L_a, not the composition constraint∘correction), which left an O(dt) error on a fixed mesh (ssp2 0.91, t2 0.67); the pressure-increment projection (`projection=Projection.increment`, opt-in, used here; the default is `standard`, ANALYSIS 45e / 45g) lets the defect act on the increment only |
| C32 viscous shear decay | SME(0) + `Newtonian`, `stress=Stress.full` (`diffusion_matrix` A[q_x, q_x, y, y] = nu, A[q_x, h, y, y] = -nu q_x / h, asserted in `cases.shear_decay`), nu 0.05, g 0.01, h = 1, v = 0, u = u(y); 8 x ny cells (dx = dy), periodic in x, walls at y = 0, 1; O2 without limiter, IMEX (Crank-Nicolson diffusion, Newton 1e-12), dt = dy/16, t 0.5. Then every x-derivative vanishes and the model is exactly u_t = nu u_yy. `noslip`: `Dirichlet(on="momentum", value=0)` (ghost -q), u0 = sin(pi y), exact sin(pi y) e^(-nu pi^2 t); `free`: `Wall` (tangential momentum kept: zero normal gradient), u0 = cos(pi y), exact cos(pi y) e^(-nu pi^2 t); ny 20 / 40 / 80 (JAX), ny 20 (foam, AMReX). Controls (JAX, ny 20): `ctrl` = the sin profile on `Wall`s, `inviscid` = nu 0 on the no-slip walls | L_inf(q_x) vs exact: order 40->80 >= 1.8 (theory 2; observed 1.87 / 1.94 for 20->40 / 40->80, both variants); ny 20 <= 7e-4 (observed 3.3e-4, JAX, foam, AMReX); h - 1, q_y, x-variation <= 1e-12 (theory: h stays 1, v stays 0, u = u(y)); AMReX max rel vs JAX <= 1e-8 (tolerance as C26; observed 3e-16); foam L_inf abs(foam - JAX) <= 0.5 x L_inf(JAX - exact) (observed 0.02 / 0.21); `ctrl` L_inf >= 0.1 (observed 0.42: the walls act); `inviscid` max abs(u - u0) <= 4e-4 (observed 1.7e-4, a regression bound: the numerical wall friction of the Dirichlet wall's Riemann state) |
| C32 split wall | as C32, walls `Wall(tag, ghost=Dirichlet(on="momentum", value=0))`: the `face` channel (the Riemann state) reflects the normal momentum and keeps the tangential, the `ghost` channel (gradients, limiter, diffusion, LSQ closure) is -q.  `noslip_split`: u0 = sin(pi y), ny 20 / 40 / 80 (JAX, numpy), ny 20 (foam, AMReX); `inviscid_split` (JAX, numpy): nu 0 | `noslip_split` as `noslip`: order 40->80 >= 1.8 (observed 1.94, 1.87 for 20->40), ny 20 L_inf <= 7e-4 (observed 3.34e-4 on all four), h - 1, q_y, x-variation <= 1e-12, AMReX max rel vs JAX <= 1e-8 (observed 4.4e-16), foam ratio <= 0.5 (observed 0.015); `inviscid_split`: wall friction = rel. change of sum(q_x vol) <= 1e-12 (round-off; observed 0; the Dirichlet wall's `inviscid`: 4.6e-5, information), max abs(u - u0) <= 8e-4 (observed 4.1e-4, a regression bound: the interior O2 Rusanov dissipation has no exit through the reflective wall and piles up in the wall cells) |
| C33 vertical viscosity | SME(2) + `Newtonian` + `NavierSlip` + `StressFree`, nu 0.05, lambda_s 0.5; 1-D, periodic (8 cells, dx 1), flat bed, uniform h = 1, q = (0.5, 0.2, -0.1): the flux divergence vanishes, only the source acts, and it is linear in q at fixed h, so q' = J q with J = d(source_q)/d(q_0..q_2) at h = 1 by sympy differentiation of the SystemModel's `source` (`cases.moment_jacobian`; linearity asserted, `hinv` = 1/h of the NSM substituted); exact q(T) = expm(T J) q0 (eigenvalues -6.68, -1.31, -0.10), T 1. Constant dt = 0.05 / 2^k, k = 0, 1, 2. `imex`: explicit flux step then an implicit-Euler source (the viscous / slip source is in the implicit slot `source`; order 1); `rk2`: explicit SSP-RK2, the source in the explicit step (order 2, the reconstruction order) | order dt/2 -> dt/4 vs expm: imex >= 0.8 (theory 1, observed 0.995), rk2 >= 1.8 (theory 2, observed 2.03); rel error at dt 0.0125 <= 1e-3 / 7e-6 (observed 5.1e-4 / 3.3e-6); q(T) = the integrator's own scheme (I - dt J)^-N resp. (1 + dt J + (dt J)^2 / 2)^N: rk2 <= 1e-12 (round-off), imex <= 1e-10 (Newton 1e-12; observed 3e-12 foam / AMReX, 4e-15 JAX); uniform state kept and h = 1 <= 1e-12; foam / AMReX = JAX: rk2 <= 1e-12, imex <= 1e-10 (observed 1e-16 / 1.3e-12) |
| C34 in-plane viscous operator | gate 45: SME(0) + `Newtonian`, `stress=Stress.full`, smooth periodic h, q_x, q_y on [0,1]^2 varying in x and y, O2 without limiter, one implicit-Lie step dt 1e-7 with nu 1 and nu 0, L = (Q_1 - Q_0) / dt on the momentum rows (the discrete viscous operator); nx 16 / 32 / 64 (JAX), nx 16 (foam, AMReX, against JAX's L) | max rel vs the exact div(h tau'), tau' = nu (grad u + grad u^T) + 2 nu (div u) I (the in-plane stress minus the viscous normal stress tau_zz = -2 nu div u, Marche 2007 eq. (5.19); = the SystemModel operator, `diffusion_matrix` with its mixed-index entries A[i, j, x, y] plus the nu-part of the source, symbolically, ANALYSIS 45e / 45f): order 32->64 >= 1.8 (theory 2; observed 1.99 / 1.98 for q_x / q_y), nx 64 <= 6e-3 (observed 2.3e-3 / 3.9e-3); foam / AMReX max rel vs JAX <= 2e-7 (observed 7e-8 foam: its own O2 stencils in the explicit stage, an O(dt) effect in L; 6e-10 AMReX: round-off / dt). The two-point fluxes alone (before 2026-10-13) converged to the diagonal-index part and stayed 16 % / 38 % off the full operator; before ANALYSIS 45e the SystemModel operator was h div(tau), before 45f div(h tau) |
| C35 restart | C1 `o2_n40` (adaptive dt) and C8 `vam` with `projection=Projection.increment` (the lagged pressure in Q's pressure rows): `full` 0 -> T with `dt_checkpoint` T/2 (the march lands there), `half` 0 -> T/2, `restart` from `half`'s restart state -> T; `typo`: `restart_from` misspelt; c1 serial and mpi2, c8 serial | max abs(restart - full) = 0 (spec: bitwise); `typo` raises FileNotFoundError naming `restart_from` before anything is built (spec) |
| C36 full stress, level 1 | SME(1) + `Newtonian`, `stress=Stress.full` (ANALYSIS 45h), periodic [0,1]^2, 16^2, smooth h, q_x_0, q_x_1, q_y_0, q_y_1, flat bed, lambda_s 0, nu 1e-2, O2 minmod, IMEX (Crank-Nicolson diffusion, Newton 1e-12), dt 1e-3, t 0.02 (JAX; NumPy) | the run completes; mass (C6) <= 1e-12; numpy = JAX <= 1e-8 (Newton / GMRES tolerances) |

Sources: *theory* (order, positivity, symmetry), *tolerance* (round-off
1e-12, the pressure solve), *spec* (the task's number), *observed* (about 2x
the worst value seen, or ~0.8 x an observed pre-asymptotic order).  Observed
thresholds are regression bounds for the next backend, not proofs.  Orders
are raw log(e_coarse / e_fine) / log(refinement).

## Large tier (`python matrix.py run --tier large ...`, `large.py`)

Never in the default run; each command <= 5 min.  L1 weak scaling (C1 bump
O2, 200^2 cells per rank, 1/2/4 ranks, 20 constant steps: ms/step); L2
strong scaling (1600^2 = 2.56M cells, 5 steps); L2P pressure strong scaling
(VAM(1) channel 3200 x 320 = 1M cells; JAX `auto`: pyamg serial, hypre on CPU ranks, AmgX on GPUs;
foam and AMReX hypre BoomerAMG + FlexGMRES); L3 catchment (JAX: the
14.8M-cell 1 m mesh on 4 GPUs, a 5 s window, reads the mesh L5 writes: run
L5 first; AMReX: `gates/catchment_amrex.py`, the 10 m raster with nodata
cells as a bed-1e4 wall, against JAX on the same mesh); L4 setup / compile /
memory vs mesh size; L5 parallel mesh + field write of the 14.8M-cell mesh.
Each backend uses the default wave speed (the Gelfand bound,
ANALYSIS 48, on every backend and device).  A timed-out AMReX
run can log a SIGBUS: `timeout` kills apptainer, which unmounts the image
under the still-running ranks.
Results: the "Large tier" section of `table.md`.

## Adding a backend

Write one class in `backends.py` and add it to `BACKENDS` (JAX: 42 lines, NumPy: 25,
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

## Results (2026-10-13, dev/restart: restart files)

`time_start`, `restart_from`, `dt_checkpoint`, `checkpoint` on every backend (solvers guide,
"Restart").  The march state is the template's loop carry `(time, Q, Qaux)`: the lagged pressure
of `Projection.increment` is Q's pressure rows, the `cfl_check_every` cadence is
`floor(time / dt + 1/2)` and the write / checkpoint counters are now derived from the time at the
loop start (equal to the carried ones; `time_start` != 0 lands on the right write instants).
- C35 (this store, the committed code): max abs(restart - full) = 0 on NumPy, JAX, foam, AMReX for
  c1 (C1 O2, adaptive dt) and c8 (C8 VAM, increment projection), serial; c1 mpi2 on JAX, foam,
  AMReX: 0.  A misspelt `restart_from` raises FileNotFoundError naming it (with "did you mean"),
  before anything is built.  Control: c8 restarted with the pressure rows of the NumPy state zeroed
  ends 4.5e-2 off (the lagged pressure is carried and used).
- Beyond C35: JAX serial restarted from the mpi2 state 5.7e-16 off the mpi2 run (one file, any rank
  count); AMReX `max_grid_size` 8 from a 16-box state (redistributed) 0; foam / AMReX restarted
  into the case the state was written in (staged) 0; `time_start` 1 -> 2.5 vs 0 -> 1.5 <= 2.2e-16
  on all four.
- Unchanged: C1 `o2_n40`, C8 `vam`, C31 `swe_ssp2_k0`, `vam_ssp2_k0` on all four backends equal the
  base commits bit for bit; `matrix.py table` on this store (C1, C8, C31, C35 on every config but
  GPU, the C1 mpi8 reference): C1 13, C6 10, C7 12, C8 12, C31 19, C35 22 rows PASS, none FAIL
  (`table.md` not regenerated: the store holds these cases only).

## Results (2026-10-13, dev/bc-channels: the two BC channels)

Every `BoundaryCondition` takes `face=` / `ghost=`: another BC that supplies the Riemann state
(`boundary_conditions`) or the stencil ghost (`boundary_ghost_state`) on the slots of its `on`;
the rest keep the BC's own value, and the own ghost is derived from the own face value.  The LSQ
closure classifies a row from the ghost kernel, so an overridden ghost takes the override's kind
(C32 mesh at degree 2: `Wall` q_x Neumann, split wall q_x Dirichlet, as `Dirichlet`).
- Defaults unchanged: every kernel of C1, C4, C28 (gate 43 setups) and C32 (57 systems; sympy
  `srepr`, foam and AMReX C++ text) and the closure classes are identical before and after.
- C32 `noslip_split` (scratch store, serial): L_inf 3.34e-4 / 9.12e-5 / 2.38e-5 at ny 20 / 40 / 80
  (orders 1.87, 1.94; `noslip` 3.33e-4 / 9.12e-5 / 2.38e-5) on JAX and numpy (numpy = JAX
  1.6e-12); ny 20: AMReX = JAX 4.4e-16, foam 3.34e-4 (ratio 0.015).
- nu 0: the split wall carries no momentum through the walls (rel. change of sum(q_x vol): JAX 0,
  numpy 2.2e-16; ny 40: 1e-15); the Dirichlet wall 4.6e-5 (ny 40: 6.2e-6).  max abs(u - u0) is
  not round-off on either: 4.1e-4 / 8.8e-5 (split) and 1.7e-4 / 3.3e-5 (Dirichlet) at ny 20 / 40,
  second order.  Its source is the interior O2 Rusanov dissipation (the face jumps of the
  reconstruction, ~ s dy^3 d4u/dy4): -8.0e-5 at the centre in both runs; it moves momentum
  towards the walls, where the reflective wall lets none out, so the wall cells gain (+4.1e-4
  split, +1.7e-4 Dirichlet, whose wall flux takes part of it away).

## Results (2026-10-13, dev/incremental-projection: pressure-increment projection)

`ChorinTemplate` projects with `projection=Projection.increment` when asked (opt-in; the default is `standard`, ANALYSIS 45g): correct with
the lagged pressure, solve `press` for the increment, store the sum (Almgren, Bell, Crutchfield
2000, version (1)); IMEX-SSP2 lags each stage by its tableau coefficient.  foam `linsolve`
evaluates op(x) as |x| op(x / |x|) (a small right-hand side).  Scratch store, serial: C31 VAM ssp2
1.96 (was 0.91), t2 1.12 (Lie-capped, was 0.67), t1 1.03; C10 unchanged (1.91 / 2.00 / 2.01, all
backends); C8 foam / AMReX vs JAX <= 2.3e-10; C8 / C10 / C31 29 rows PASS, 0 FAIL.  Not rerun:
mpiN / GPU configs.  Triangles: the lagged iteration does not contract on 368 of 1440 modes of
C8's triangle mesh (ANALYSIS 45e); C8 `vam_tri` (2 steps) passes.

## Results (2026-10-13, dev/foam-hypre: AMG pressure solve on foam)

foam `linsolve` with `linear_solver=LinearSolver.hypre` (the Chorin default; `LinearSolver.schur_gamg` keeps the old
FGMRES + Schur-complement GAMG, `"gmres"` FGMRES alone): AMReX's solve on OpenFOAM's mesh.  The
printed operator's matrix on the LSQ stencil by colouring (the driver's greedy distance-2
colouring of the global LSQ pattern, `constant/zoomyColours`: 9 colours on quads, 22 on the C8
triangles; one op application per colour and row), left-scaled by the inverse 2x2 cell blocks,
rows owned per rank in OpenFOAM's global cell numbering, hypre 3.2.0 FlexGMRES + one BoomerAMG
V-cycle (HMIS, ext+i, hybrid symmetric GS) to the true residual 1e-10.  symFoam links the CPU
build of `containers/zoomy_amrex/build_hypre_cuda.sh` (same toolchain and OpenMPI as the OF13
image).  Scratch store, against the JAX store:
- C8 `vam` / `vam_newton` / `vam_t2` / `vam_tri`: L1 rel vs JAX 2.5e-10 / 1.3e-10 / 4.7e-11 /
  6.9e-12 (max rel <= 2.8e-10; old path 6.1e-10 / 1.4e-10 / 8.7e-11 / 3.0e-11); mpi2 and mpi4
  (all four) vs serial <= 3.1e-11; new vs old foam <= 8.0e-10.
- C10 (9 runs): Richardson orders 1.91 / 2.00 / 2.01 as before; space O1 = JAX 1.6e-12 (O2: foam's
  own stencils); mpi2 / mpi4 vs serial <= 7.3e-13.  Iterations per solve at n120 (3 variants):
  233 with the old Schur-GAMG (103-125 ms per solve) -> 16.3-16.9 (4.4 ms: assembly 2.1, AMG
  setup 0.6, Krylov 1.7).
- C31 VAM (standard projection: the branch predates the increment projection), all 12 runs on foam (the case now runs `vam_ssp2_k0` on foam): vs JAX <= 2.9e-11,
  orders t1 / t2 / ssp2 0.98 / 0.67 / 0.91 (= JAX); `vam_ssp2_k0` mpi2 / mpi4 vs serial <= 3.4e-13.
- L2P (1M cells, 5 steps; ms/step, iterations per solve, ms per solve assembly / AMG setup /
  Krylov, the whole command): serial 33840, 19, 12235 / 2810 / 8690, 5:38; mpi2 15251, 19.8,
  5222 / 1435 / 4059, 3:58; mpi4 8661, 19.4, 2886 / 882 / 2347, 3:08.  Before: > 10 min on every
  config.  `large.py` stores the C++ backends' linsolve summary with each run.

## Results (2026-10-13, dev/hypre: AMG pressure solve on AMReX)

AMReX `linsolve` with `linear_solver=LinearSolver.hypre` (the Chorin default; `LinearSolver.gmres` keeps the old
amrex::GMRES + cell-block Jacobi): the printed operator's matrix on the (2R+1)^D cell stencil by
colouring (18 op applications in 2-D), left-scaled by the inverse 2x2 cell blocks, hypre
FlexGMRES + one BoomerAMG V-cycle (CPU HMIS / hybrid symmetric GS, GPU PMIS / l1-Jacobi, ext+i)
to the true residual 1e-10 (`amg_comparison.md`).  Scratch store, against the JAX store:
- C8 `vam` / `vam_newton` / `vam_t2`: max rel vs JAX 2.8e-10 / 1.0e-10 / 8.8e-11 (<= 1e-9);
  mpi2 (all three) and gpu1 (`vam`) vs serial <= 1.3e-11.  C10 (9 runs) vs JAX <= 1.6e-12,
  mpi2 vs serial <= 8.2e-13; C31 `vam_ssp2_k0` 7.5e-12, mpi2 2.1e-13.
- L2P (1M cells, 5 steps; ms/step, iterations per solve, ms per solve assembly / AMG setup /
  Krylov): serial 24396, 19, 7400 / 3111 / 8361; mpi4 8180, 20, 2581 / 1055 / 2644 (6 boxes on
  4 ranks); gpu1 512, 38, 72 / 148 / 251; gpu4 570, 38 (2 boxes: 2 GPUs busy; MPI not
  CUDA-aware).  Before: > 10 min on every config.
- 64k cells (`vam_large(800)`), old path vs hypre: CPU serial 892 iterations, 174 s per solve
  (cores shared with other runs) vs 19, 1.04 s (assembly 0.51, setup 0.17, solve 0.36); gpu1 893,
  13.9 s vs 36, 0.165 s (0.003 / 0.048 / 0.114).

## Results (2026-10-13, dev/tpfa: the diffusion face gradient)

The `diffusion` block now applies the full flux `n.A.grad_f` (ANALYSIS 45): `grad_f` = the
mean of the two cells' LSQ gradients plus the two-point difference as its part along the
cell-to-cell vector, so the mixed-index entries A[i, j, x, y] act; `implicit_diffusion` keeps
its Crank-Nicolson matrix (the two-point part) and takes the correction explicitly at the start
state.  Rerun into a scratch store (C26, C28, C32, C33, C34; serial, all three backends), against
the 2026-10-12 store:
- C33 (1-D: no correction), C28 1-D: bit-identical on every backend.
- C28 quads: bit-identical (metrics: errors and the vs-JAX differences) on every backend.
- C32: AMReX bit-identical; JAX and foam within 1-2 ulp (<= 4.4e-16; nu 0 control identical): their
  unstructured LSQ tables give an x-derivative of the x-uniform field at round-off, not 0, and
  A has mixed entries; AMReX's Cartesian tables give exact zeros.
- C28 triangles: within round-off (foam: error 8.147044629183342e-4 vs 8.147044629183316e-4, vs JAX
  6e-16): gate 41's triangles are orthogonal (max 1 - |n.e| = 2.2e-16), so the correction is
  round-off there too; JAX `orders_tri_*` errors equal to 4 digits (orders unchanged).
- C26 (the field varies in x and y, A has mixed entries): max |Q_new - Q_old| 2.3e-4 (JAX, AMReX),
  2.5e-4 (foam), 2.1e-4 of max |Q|; AMReX = JAX 1.1e-13 (<= 1e-8), foam vs JAX L1 rel 1.72e-3
  (unchanged: foam's own O2 stencils).
- C34 (new, gate 45): max rel vs the full SystemModel operator q_x 3.05e-2 / 7.69e-3 / 1.94e-3,
  q_y 4.62e-2 / 1.21e-2 / 3.07e-3 at nx 16 / 32 / 64 (orders 1.98 / 1.98); foam and AMReX
  give the same errors (L vs JAX 7e-8 foam, 6e-10 AMReX).

## Results (table.md, 2026-10-14, dev/wavespeed-bound)

The Gelfand upper bound as the wave speed of models without a closed-form spectrum, the faces rule
with MOOD repairs at their own step and the cells' speeds under the order-2 edge source (ANALYSIS
48).  The whole store rerun from empty, 962 runs on core 082acb3, jax edc4c35, foam c4310bc, amrex
a177451: **871 rows, 801 PASS, 0 FAIL** (926 passing cells; the 2026-10-13 store 925, + AMReX
mpi2 C17 lake).  Changes against the 2026-10-13 store (power x 1.06 wave speeds), all inside
their thresholds:

- C11 smooth O2 order 80->160 1.81 -> 1.65 (JAX, NumPy, AMReX), foam 1.78 -> 1.63 (>= 1.5; the
  exact spectrum gives 1.69: power x 1.06 under-estimated the speed, i.e. the dissipation); O1
  0.96 -> 1.02; C14 O1 order 0.59 -> 0.63.
- C8 (VAM(1) Chorin): foam / AMReX vs JAX, Newton vs linear, parallel and GPU vs serial 1e-10 ->
  1e-11; C11 AMReX `sub` O2 vs JAX 1.8e-7 -> 3.8e-6 (tolerance 2e-5, JAX's sensitivity); C15
  Chorin FD - jvp 6.4e-9 -> 1.7e-8 (<= 1e-5).
- C17 `lake` and `dry` do not return in a long-lived matrix process after the Euler `box` cases (also
  on main bb5f636, whose store has neither `dry` nor mpi2 `lake`); `lake` ran in a fresh process
  (serial, mpi2, gpu1: |u| 1.3e-15, = cpu1).  `dry`'s mass metric sums the NaN-masked AMR levels
  (NaN on main and on this branch): left out, as before.

## Results (table.md, 2026-10-12, night rerun)

Power-iteration wave speeds on every backend and device (`zoomy_core.symsolver.WAVE_SPEED`;
foam / AMReX gained `spectral_radius`).  The whole store rerun from empty (the eig store is kept
as `store_prev_eig_20261012`); C31 (IMEX-SSP2 time order), C32 (viscous shear decay, exact),
C33 (SME(2) vertical viscosity, expm) added.  Libraries: core 09d1584, jax 5fd4252, foam
e17026f, amrex ce3f72a.  **538 check rows PASS, 0 FAIL** (651 passing cells over the configs; the docs page counts cells).  After the diffusion face-gradient fix (C34, ANALYSIS 45; C26 changed by ~2e-4, C28 / C32 / C33 unchanged to round-off): **544 rows PASS, 0 FAIL**.  After the in-plane viscosity derivation fix (`package_viscous` evaluated d(C)/dx, ANALYSIS 45; C26 moved by 5e-5): still 544 PASS, 0 FAIL.  With the NumPy backend (serial; AD, preCICE, AMR, large tier excluded) and the in-plane traction fix (C26 / C34 rerun on every backend and config): **759 rows PASS, 0 FAIL**.  The table now judges every cell by its own
threshold: the old table hid an errored C16 jax-foam run behind an "unsupported" sibling (459
PASS, 1 FAIL there); it passes now (4.3e-15).  Open: the VAM projection is not idempotent
(P(P(U)) != P(U), 0.2-2.5 % of the correction), which caps every Chorin time order near 1 (C31).

## Results (table.md, 2026-10-12, eig store)

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
- Large tier (rerun 2026-10-13, power wave speeds everywhere, 10-min limit per run; ms/step at
  1/2/4 ranks or GPUs):
  - L1 weak, 200^2 per rank: JAX CPU 212/236/242, GPU 2.9/3.6/3.4; AMReX CPU 229/264/251, GPU
    1.76/2.06/1.90; foam 384/419/399 (eig before: AMReX CPU 845, GPU 12.9; foam 1387; JAX 380).
  - L2 strong, 2.56M cells (5 steps): JAX CPU 7521/4155/2574 (91 / 73 %), GPU 253/127/64 (99 %);
    AMReX CPU 9492/5301/3201 (90 / 74 %), GPU 62/33/17 (93 / 90 %); foam 19280/9536/4910 (101 / 98 %).
  - L2P pressure, 1M cells: JAX GPU (AmgX) 1417/1383/1135, JAX serial (pyamg) 30020; > 10 min:
    JAX mpi2/mpi4, AMReX (all; GMRES + cell-block Jacobi, no multigrid), foam (all; Schur-GAMG).
  - L3: JAX 14.8M cells on 4 GPUs, 5 s in 6.7 s; AMReX 10 m (292k cells) = JAX to 6.5e-15, water =
    rain to 1.5e-15, cpu1 / cpu4 / gpu1 / gpu4 5.1 / 1.4 / 0.03 / 0.01 s for 5 s.
  - L4: setup / memory linear in cells; L5 parallel write of 14.8M cells 55 / 37 / 25 s.
  - Open: AMReX / foam pressure preconditioners (`amg_comparison.md`: hypre-CUDA BoomerAMG on the
    2x2-scaled operator), JAX's MPI pressure path, foam's implicit diffusion (659 us/cell/step vs
    JAX 9.5, `tests/results/symsolver`).
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
  O(dt) error, and not only at a fixed mesh: the ssp2 dt/4 -> dt/8 difference is
  1.53e-7 / 1.55e-7 / 1.19e-7 at nx 30 / 60 / 120 (ANALYSIS 45: approximate by
  construction; the exact composition reaches 1.98 / 1.95 on quads, not adopted). Without the stage pressure in the implicit tableau
  the max error is ~3x larger (L1(h) 1.2-1.6x), the orders the same. Manning n 0.5
  is not stiff at these dt (rate x dt < 3e-3); n 0 gives the same VAM orders.
- Runtime: a Chorin run costs ~2.5 min whatever the mesh (JAX: ~130 s setup
  of the pressure template; foam / AMReX: printing), so C8 / C10 / C15 dominate.
- C32-C33 (viscosity with exact solutions, added after the full run; libraries core 44bc573,
  jax f39b67f, foam e17026f, amrex ce3f72a; their 66 rows are merged into table.md from a
  separate store): all PASS, 0 FAIL.  C32 L_inf(q_x) at ny 20 / 40 / 80, no-slip 3.33e-4 /
  9.12e-5 / 2.38e-5, free-slip 3.32e-4 / 9.11e-5 / 2.38e-5 (orders 1.87, 1.94 for both);
  at ny 20 foam 3.34e-4 / 3.33e-4, AMReX = JAX to 3e-16; control (sin profile on free-slip
  walls) 0.42.  C33 time orders imex 0.99 / 0.995 (implicit Euler), rk2 2.07 / 2.03; foam and
  AMReX = JAX to 1e-16 (rk2) and 1.3e-12 (imex).  Runtime: C32 JAX ~40 s per run (compile),
  foam ~60 s, AMReX ~20 s (8 JAX runs, which need a second command: the 270 s budget; 2 foam, 2 AMReX);
  C33 6-18 s per run.
- C32 build notes.  (i) the full stress's (`Stress.full`) `diffusion_matrix` is nu for the shear
  (A[q_x, q_x, y, y]; 2 nu is the xx entry) and -nu q_x / h on h, so with h = 1, v = 0, u(y)
  the model is exactly u_t = nu u_yy (asserted in `cases.shear_decay`).  (ii) No custom BC is
  needed: free-slip = `Wall`, no-slip = `Dirichlet(on="momentum", value=0)` (ghost -q).  But
  that BC also sets the Riemann outer state to zero momentum, so Rusanov damps the wall
  cells: the nu = 0 control (`inviscid`) changes u by 1.7e-4 (g 0.01; 2.5e-3 at g 9.81).
  The split wall `Wall(ghost=Dirichlet(on="momentum", value=0))` (BC channels, `noslip_split`)
  keeps the reflecting wall as the Riemann state and the odd ghost for the diffusion block.  (iii) g = 0.01: the exact solution does not depend on
  it, but at g 9.81 the Rusanov error (~ sqrt(g h) dy^3) partly cancels the O(dy^2) error of
  the viscous flux and the measured order is erratic (L_inf 1.56e-3 / 1.47e-4 / 6.1e-6 /
  2.3e-6 at ny 20 / 40 / 80 / 160: 3.4, 4.6, 1.4); first order in space at g 9.81 gives 0.26
  at ny 20 (numerical viscosity 0.5 s dy = 0.078 > nu).  (iv) `SystemModel.from_model` warns that 2
  equation rows (the momenta) carry terms "classified into NO slot ... silently dropped"
  for `Stress.full` (as for C26); the second-derivative terms listed there are -2x the
  diffusion block's (2 nu q_xx + nu q_x,yy + nu q_y,xy), and the exact decay rate shows the block is applied once with the right
  sign, so the warning looks like the extractor's residual bookkeeping.
- C33 build notes.  The viscous / slip source of SME is in the implicit slot `source`, so
  `imex` is Lie splitting with an implicit-Euler source (order 1) and `rk2` takes the same
  source in the explicit SSP-RK2 step (the reconstruction order is the default time order).
  The NSM source uses the aux `hinv` = 1/h; `cases.moment_jacobian` substitutes it.
- Jax-free case builders (2026-10-13).  The builders of C8, C10, C26, C28, C31 (gates 7, 21,
  31, 41, 43) imported JAX, so the NumPy / foam / AMReX images could not set them up.  The
  predictor numerics are core's `predictor_nsm` (= gate 7's `pred_nsm`), the gates import JAX
  inside their JAX runs only, `gates/common.py` enables x64 when JAX is present.  C28's
  per-case runs (`{kind}_{path}_{case}`) now run on every backend alone, JAX included, and
  store the final field `u`; the table compares it with JAX's (was: each foam / AMReX / numpy
  run also ran JAX).  A store from before needs `matrix.py run B C28 --force` on jax, numpy,
  foam, amrex.  Rerun into a scratch store (C8, C10, C26, C28, C31, C35 jax + numpy, C15 jax,
  C28 foam + AMReX; serial): every run bit-identical to the previous store (C35's misspelt
  path note aside: its timing and work path), the table identical, 0 FAIL.
