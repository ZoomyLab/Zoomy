**651 PASS, 0 FAIL** (37 information rows, 12 n/a, 1 unsupported).  Last run 2026-10-12; library commits: zoomy_amrex ce3f72a, zoomy_core 09d1584/44bc573, zoomy_foam e17026f, zoomy_jax 5fd4252/f39b67f; rendered from `matrix_latest.json` of 2026-10-12.

A cell counts the checks of a case on a backend over every config (serial, MPI, GPU).  Case setups and pass criteria: [matrix README](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/matrix/README.md#cases-and-pass-criteria); every row: [table.md](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/matrix/table.md).  `-`: no check of its own on that backend (JAX is the reference, or the runs are counted under C6 / C7 / C9).

| case | setup | jax | foam | amrex |
|---|---|---|---|---|
| C1 | smooth: bump at rest, walls, t = 2.5 | 2 PASS | 4 PASS | 4 PASS |
| C2 | shock: bump in flow (0.3, 0.1), walls, O2 + MOOD | 1 PASS | 2 PASS | 2 PASS |
| C3 | lake at rest over a bed bump, 100 steps | 2 PASS | 2 PASS | 3 PASS |
| C4 | channel over a bump vs Bernoulli (inflow/outflow/wall BCs) | 8 PASS | 12 PASS | 12 PASS |
| C5 | radial dry dam break, O2 eta + MOOD | 5 PASS | 4 PASS | 6 PASS |
| C6 |  | 25 PASS | 20 PASS | 24 PASS |
| C7 |  | 18 PASS | 18 PASS | 18 PASS |
| C8 | VAM(1) Chorin channel (gate 21): linear / Newton / time_order 2 / triangles | 1 PASS | 5 PASS | 4 PASS |
| C9 |  | 8 PASS | - | 8 PASS |
| C10 | VAM(1) Chorin convergence: smooth wave over the bump | 3 PASS | 3 PASS | 3 PASS |
| C11 | 1-D: smooth wave (walls), channel vs Bernoulli (inflow/outflow) | 4 PASS | 8 PASS | 8 PASS |
| C12 | triangles: C1 bump and C4 channel | 4 PASS | 8 PASS | n/a |
| C13 | periodic in x: shifted bump in a uniform flow | 6 PASS | 7 PASS | 7 PASS |
| C14 | IMEX: SWE + Manning friction (implicit source), bump in flow, t = 0.3 | 2 PASS | 4 PASS | 4 PASS |
| C15 | AD (JAX): jvp vs central FD, reverse vs forward, dJ/dg | 20 PASS | n/a | n/a |
| C16 | preCICE: coupled (2 x 100 cells, window 1e-3, O1) vs single domain | 6 PASS | 4 PASS | 8 PASS |
| C17 | AMR (AMReX, gate 37): covering, static patch, reflux, lake, regrid | n/a | n/a | 19 PASS |
| C18 | Euler 3-D Sod along x, y, z (n x 2 x 2), HLL, t = 0.2 | 8 PASS | 8 PASS | 8 PASS |
| C19 | Euler 3-D smooth periodic wave, [0,1]^3, t = 0.1 | 6 PASS | 6 PASS | 6 PASS |
| C20 | Euler 3-D walled box 16^3, centred pulse, t = 0.6 | 4 PASS | 4 PASS | 4 PASS |
| C21 | Euler 3-D parallel: pulse 12^3 (centre 0.4), Sod along y 60 x 4 x 4 | - | - | - |
| C22 | Euler 3-D AD (JAX): pulse 10^3, dJ/dgamma, t = 0.1 | 6 PASS | n/a | n/a |
| C23 | Euler 3-D cross-backend: acoustic pulse (amp 0.01, width 0.04), t = 0.3 | - | 4 PASS | 4 PASS |
| C24 | Euler 2-D: Sod along x, y (n x 2); periodic wave in [0,1]^2 | 12 PASS | 12 PASS | 12 PASS |
| C26 | implicit diffusion (IMEX, NewtonianInPlane, nu 1e-3; gate 31), 20^2, t 0.05 | - | info only | 1 PASS |
| C27 | SSP-RK time order 1 / 2 / 3 (gate 40): forced SWE, exact ODE and a smooth wave | 7 PASS | 6 PASS | 6 PASS |
| C28 | diffusion blocks at the boundaries (gates 41, 43): heat equation, dir0 / neu / dirL / per | 30 PASS | 36 PASS | 24 PASS |
| C29 | LSQ derivatives with the boundary closure (gate 44): quads, p 2 / 3, dir / neu | 20 PASS | n/a | n/a |
| C30 | LSQ polynomial reproduction (gate 39): 1-D, quads, jittered quads, triangles, hexahedra | 1 PASS | n/a | n/a |
| C31 | IMEX time order (Lie vs IMEX-SSP2(2,2,2)): SWE + Manning (C14), VAM(1) wave + Manning | 7 PASS | 1 PASS | 2 PASS |
| C32 | viscous shear decay between walls (SWE + NewtonianInPlane): u_t = nu u_yy, exact; no-slip and free-slip walls | 12 PASS | 10 PASS | 10 PASS |
| C33 | vertical viscosity of SME(2) (Newtonian + NavierSlip): uniform state, q(t) = expm(t J) q0 | 8 PASS | 10 PASS | 10 PASS |

Failures:

- none
