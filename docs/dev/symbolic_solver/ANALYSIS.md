# Symbolic solvers — analysis and module classification

Branch `feature/solver-template`. Prototype: `prototype/` (runnable, `run_all.py`).

## 0. "Plain Python" — withdrawn

- Meant: algorithms executed as Python on numpy/JAX. No C++. Wrong goal.
- Now: Python **builds** a sympy DAG once; printers emit Python or C++.
- Identical to models today: Python authors sympy, printers emit backends.

## 1. Ground truth read (line by line)

| code | lines |
|---|---|
| zoomy_core, all 185 files | 76.6k |
| jax `solver_jax.py` | 2368 |
| jax `solver_imex_jax.py` + `implicit_ad.py` | 358 |
| jax `solver_chorin_vam_jax.py` | 1105 |
| jax `jax_runtime.py` + `userfunctions.py` | 1349 |
| jax `precice_solver.py` (legacy) | 1287 |
| jax `march_emitted.py` | 337 |
| foam `zoomyFoam.C` | 678 |
| foam `numerics.H` + `numerics_o2.H` | 903 |
| foam `imex_kernel.H` + `UserFunctions.H` + `init.H` | 776 |
| foam `precice/PreciceManager.H` | 1142 |
| foam `chorin_app/chorinFoam.C` | 343 |

Not re-read: `zoomy_core/solver/template/` (my October IR, 6.0k, to delete).

## 2. The algorithms, minimal form (from the ground truth)

Semi-discrete operator, order k ∈ {1,2}:

```
L_k(Q)_i = −1/|V_i| Σ_f |f| ( F*(Q_f⁻,Q_f⁺,n_f) + D∓(Q_f⁻,Q_f⁺,n_f) ) + S(Q_i) − NC_i
Q_f^±   = R⁻¹( W_c + ∇W_c·r_f ),  W = R(Q)        (k = 2; k = 1: cell values)
Q⁺_bnd  = BC_b(t, x, Q⁻, aux⁻, p, n)               (ghost, Piecewise by branch)
```

Hyperbolic step (Shu–Osher SSP-RK2) + MOOD:

```
Δt   = min( min_f C·2r_f/(d(2k+1)λ_f), T−t, Δt_cap, Δt_couple )
Q¹   = Q⁰ + Δt L(Q⁰)
Q    = ½Q⁰ + ½(Q¹ + Δt L(Q¹))
B    = { i : h_i < 0  ∨  ¬finite(Q_i) }
Q_i ← Q⁰_i + Δt L₁(Q⁰)_i,  i ∈ B
Q   ← U(Q)                                          (update_variables)
```

IMEX — two different schemes in the ground truth:

```
JAX  (Lie):  Q* = RK(Q);  Q** = CN-diffusion(Q*);  solve Y − Q** − Δt S(Y) = 0
foam (ARK):  solve Y_i − r_i − Δt a^I_ii S(Y_i) = 0,
             r_i = Qⁿ + Δt Σ_{j<i} (a^E_ij K^E_j + a^I_ij K^I_j)
```

Chorin (one stage S):

```
Q̃ = Q + Δt L_pred(Q)
solve  A x = −R_P(Q̃, 0),   A x := R_P(Q̃, x) − R_P(Q̃, 0)
Q ← U_corr(Q̃, P)
JAX:  matvec = jvp(R_P), GMRES, order 2 = ½(Q + S(S(Q)))
foam: BiCGStab on R(x) − R(0), order 1
```

preCICE window (foam `zoomyFoam.C` + `PreciceManager.H`):

```
while ongoing:
  [ck ← (Q,t)]                       if writeCheckpoint
  Δt ← min(Δt, Δt_pc)
  peer ← P₃ᴰ(read(Δt))  → ghost | hi-fi face term (D∓ frozen per window)
  step
  write(I₃ᴰ(Q_patch));  Δt_pc ← advance(Δt)
  [(Q,t) ← ck]                       if readCheckpoint
```

- Ghost modes (fullstate, characteristic, froude): kernels plus `eigensystem`.
- JAX: hook seam plus `interlevel_face_contribution`; `precice_solver.py` is legacy.

AD (JAX only):

- Forward: `lax.while_loop`.
- Reverse: equinox checkpointed while.
- Root: `custom_root`. LinSolve: `custom_linear_solve`.
- Effects off in reverse.

## 3. Building blocks (the whole vocabulary)

| class | blocks |
|---|---|
| K kernels (exist) | flux, D±, BC ghost, source, aux, U, R/R⁻¹, λ, I₃ᴰ/P₃ᴰ, R_P |
| M mesh ops | ghost, reconstruct, face_map, div, cell_map, grad, halo, reduce_{min,max,any}, select, comp/assemble |
| C control | while_loop, cond, root, linsolve, guard |
| E effects | write, log, couple_{read,write,advance,checkpoint,ongoing} |
| static | Python `if` / `for` at build time: order, tableau, mood, coupled |

Per-backend bodies (what the implementer writes):

| block | JAX | C++ (foam) |
|---|---|---|
| while_loop | `lax.while_loop` / eqx checkpointed | `while` |
| cond | `lax.cond` | `p ? a() : b()` |
| root | Newton in `custom_root` | per-cell Newton (`imex_kernel.H`) |
| linsolve | GMRES/CG in `custom_linear_solve` | BiCGStab (`imex_kernel.H`) |
| guard | `debug.callback` raise | `FatalError` |
| face_map + div | gather, vmap, scatter-add | face loop, owner/neighbour |
| grad | LSQ (`compute_derivative`) | `fvc::grad` (`compute_derivative`) |
| halo / reduce | SPMD / `jnp.min` | `swap_coupled_halos` / `returnReduce` |
| couple_* | `io_callback` | `PreciceManager` calls |

## 4. Module classification

| # | module | status |
|---|---|---|
| 1 | Kernels: Model→SM→NSM + per-backend printers | exists, reuse |
| 2 | Vocabulary + 2 printers (Python, C++), scoped CSE | new, ~200 lines, core |
| 3 | Templates: L, rk, mood, imex_lie, imex_ark, chorin, march, coupled march | new, ~300 lines, core |
| 4 | Backend runtime: M + C + E bodies | per backend, **extracted** from existing code |
| 5 | Backend driver: setup, print, jit/compile, run | per backend, slimmed |
| 6 | preCICE adapter: API calls only | foam exists; JAX from legacy plumbing |

- One printer per language family, not per backend.
- Backend semantics live in module 4.
- numpy and JAX share the printed source; only the module table differs.
- foam and amrex share the printed C++; only the runtime header differs.

## 5. Reuse vs. not

| piece | reuse? | why |
|---|---|---|
| sympy expressions, `Function` (undefined) | yes | the opaque-block pattern of `conditional` |
| `Lambda`, `Tuple` | yes | closures, functional carries |
| `PythonCodePrinter`, `CXX17CodePrinter` | yes | subclassed, ~30 lines each |
| `sympy.cse` | kernels only | it would hoist out of Lambda / `cond` branches |
| `sympy.codegen.ast` (For/While/Assign) | no | imperative, typed declarations, no Python `For`, no lax semantics |
| `lambdify` | kernels only | the solver needs nested closures |
| JaxRuntime, UserFunctions, foam `numerics.H` | yes | become module-4 bodies |
| `solver/constants.py`, `arg_slots.py` | yes | used by foam, amrex, chorin, generic_c |

## 6. Why the earlier IRs grew, and why this does not

| cause | earlier | now |
|---|---|---|
| expression layer | own IR | sympy |
| statements | own Seq/Assign/Hook | none: DAG + scoped CSE |
| C++ out-params | kinds, slots, type checker | functional + `auto`, no types |
| runtime branching | build-time only (MOOD fell back to reference) | `cond` block |
| scope of emitted code | ~30 shims, "no scheme logic" | the algorithm itself |
| extras | show/report, plugins | none |

## 7. Prototype evidence (`prototype/run_all.py`)

Setup:
- 1-D SWE dam break, Rusanov, minmod O2, SSP-RK2.
- MOOD.
- Implicit friction via `root`, implicit viscosity via `linsolve`.

| check | result |
|---|---|
| steps, symbolic JAX / hand JAX / C++ | 160 / 160 / 160 |
| max \|symbolic − hand JAX\| | 1.8e-15 |
| max \|symbolic − C++\| | 3.9e-14 |
| MOOD branch exercised | Δ = 0.11 vs off |
| jvp vs FD (g, cf, nu), best ε | 7e-10, 1e-9, 3e-10 |
| reverse (checkpointed) vs jvp | ≤ 5e-13 |
| preCICE JAX ↔ C++ halves vs monolith | 2.2e-14, mass drift 0 |
| preCICE JAX ↔ JAX halves vs monolith | 3.3e-15, mass drift 0 |
| hyperbolic solver / hand-written lines | 35 / 27 |
| preCICE variant, extra lines | 11 |
| core: `symsolve.py` + `vocabulary.py` | 232 + 28 |
| runtime: JAX / C++ / C++ preCICE | 120 / 137 / 35 |

- Solvers: `prototype/solvers.py`. Blocks: `prototype/vocabulary.py`.
- Printed outputs: `gen_hyperbolic.py`, `gen_precice.py`, `gen_hyperbolic.cpp`, `gen_precice_right.cpp`.
- preCICE run: order 1, Δt = window, 50 windows, real libpreCICE 3.4.0.
- FD at large ε crosses minmod kinks, so ε needs a sweep.

## 8. Risks and open decisions

1. sympy reorders sums: no bitwise parity with the old solvers (observed ~1e-15).
2. C++ value semantics create temporaries; amrex may need in-place lowering. Measure this.
3. In the prototype `root` is elementwise. Real cases need per-cell n×n and global Newton–Krylov.
4. foam storage is `List<volScalarField*>`; the runtime wraps it as one `Field`.
5. MPI stays inside the mesh ops; the templates are unaffected.
6. Ground truth has 2 IMEX schemes and 2 Chorin variants; each becomes a few template lines.

## 9. Next steps (proposal)

1. Core: vocabulary, printers, hyperbolic template. JAX runtime extracted from `solver_jax.py`. Gate vs `solver_jax` on SWE/SME.
2. IMEX (Lie + ARK) and Chorin templates. JAX gates.
3. foam runtime from `numerics.H` / `imex_kernel.H`. Gate vs zoomyFoam.
4. preCICE effects plus adapters. jax↔foam gate.
5. Delete the old code (§10).

## 10. Deletion list (after approval)

- `zoomy_core/solver/template/` (6.0k) and its `tests/template_gates`.
- `zoomy_core/solver/{ir,march,external}.py` and `transformation/procedure_{python,c}.py`.
  - Cut the `constants.py:232` import.
- `zoomy_jax/fvm/{solver2/,march_emitted.py,precice_solver.py}`.
- `docs/dev/solver_template/` (10.6k, including the prototypes).
- Importers to update:
  - `tests/goldens/goldenlib.py`
  - `scripts/verify.py`
  - `tests/fvm/test_dae_aux_sizing.py`
  - jax `tests/cases.py`
  - jax `tests/runtime/test_solver2_swashes.py`
- Keep from the 1a commits:
  - `geometry_xp` batched LSQ
  - JaxRuntime admissible slot + batched ghost kernels
  - diffusion classification

## 11. Reconstruction as template blocks (2026-10-09)

`zoomy_core/symsolver/discretization/reconstruction.py`; blocks in `vocabulary.py`, bodies in
`zoomy_jax/symsolver/runtime.py` and `zoomy_foam/symsolver/rt.H`.

```
W, Wb   = pointwise(W(Q), Q, Qaux, p), pointwise(W(Q), Qb, aux(Qb), p)
O1      : (WL, WR) = cell_traces(W, Wb)
O2      : G = gradient(W, Wb);  phi = <limiter>(rows, W, Wb, G)  [ties: put/take/minimum]
          phi = select(demote, 0, phi)                           [MOOD]
          (WL, WR) = extrapolate(W, Wb, G, phi)
QL, QR  = pointwise(Q(W), WL, AL, p), pointwise(Q(W), WR, AR, p)
```

NSM `ReconstructionSpec` (the user interface):

| field | default | meaning |
|---|---|---|
| `variables` | `"model"` | the model's W(Q) and inverse; `"conservative"`: Q |
| `limiter` | venkatakrishnan | every row; `"none"`: unlimited |
| `limiters` | `{}` | per-row override by state name, e.g. `{"b": "none"}` |
| `ties` | `{}` | `{row: lead}`: phi_row <= phi_lead (old eta path: b, q -> h) |

- Order 1 is constant in W (Q(W(Q)) at the faces), as the old eta path.
- No face clamp `eta_f >= b_f` (the old JAX eta path had one).
- Round-off below zero of the model's declared depth role is set to 0
  after every RK stage: h in [-tol, 0) -> 0, tol = `nsm.depth_round_off`
  (absolute, default 1e-12; NSM regularization, the field from the model).

Parity:

| gate | result |
|---|---|
| SWE bump O1/O2, SME(1) O2 vs `solver_jax` | 2e-16, 1.7e-13, 9.9e-13 |
| one reconstruction, dry state, vs `EmittedWLSQMUSCLJAX` | 0 (bitwise) |
| foam vs JAX: conservative O2, 2-D O2, model-W O2 | 7.2e-13, 4.5e-16, 6.6e-13 |
| foam vs JAX: dry eta + MOOD | 1.5e-9 (3.6e-2 before the round-off cleanup) |

Dry dam break, why the old solver and the template differ:

- Both use MOOD with the strict test h < 0.
- Without cleanup MOOD fired in EVERY step; flagged h in [-1e-155, -1e-18].
- Pure round-off: O1 HR keeps h >= 0 exactly (56 steps, min h = 0).
- Repairs still flagged -> whole-step O1 in ~14 of 22 steps.
- With cleanup: 0 MOOD triggers in 26 steps, O2 throughout, mass exact.
- Runtime 2.41 s -> 1.31 s (old solver 2.37 s).

Audusse et al. 2004 (Zotero PTMTF2TS): Thm 2.1 / (2.34) positivity at O1;
Sec. 3, Fig. 1: of (z, h+z), (z, h), (h, h+z) only (h, h+z) keeps the lake
at rest and h >= 0 at a dry shore; Thm 3.1.  Our SME map reconstructs
(z, h+z).  Probe with (h + z, h, u): same MOOD counts (flat bed).

## 12. CFL wave speed: cells vs faces

`wave_speed=CflWaveSpeed.cells` (default): eigenvalues of the cell states of each face
(own eigensolve).  `"faces"`: `max(|s_L|, |s_R|)` of the Riemann solver's own
`wave_speed_bounds` at the states its dissipation uses, from the first RK
stage (`numerical_face_sides_speed`, rows F + D-, -F + D+, s).

| case (CPU, one process, min of 3) | cells/faces per step | steps |
|---|---|---|
| SME(1) 1-D O2 | 1.10x | 80 / 80 |
| SWE bump O2 80^2 | 1.33x | 21 / 21 |
| dry dam break eta + MOOD | 1.28x | 26 / 41 (0.81x total) |
| IMEX SME(1) 1-D | 1.14x | 40 / 40 |
| Chorin, const dt, check every step | 1.53x | same |
| Chorin, adaptive | 1.16x | 296 / 297 |
| Malpasset O1 (HLL, analytic CFL) | 0.69x | 31 / 31 |
| Malpasset IMEX | 0.92x | 13 / 13 |

- faces: free when the flux already needs a numerical eigensolve.
- faces: larger speeds at wet/dry fronts (reconstructed states) -> more steps.
- HLL: speed = max(-min(s_L,0), max(s_R,0)), the flux's own clamped pair.
- JAX: speed returned as one row (not broadcast over n rows).
- After both fixes (pinned, 16 cores): Malpasset 1.12x, IMEX 1.09x,
  SWE 1.34x, SME 1-D 1.01x, Chorin 1.45-1.57x; dry 0.81x (41 vs 26 steps:
  the cell CFL zeroes dry-cell speeds; the face speed is Audusse (2.34)).
- Default: "faces" (2026-10-09); parity gates vs the old solvers: "cells".

## 13. CFL check cost (Chorin bump, const dt)

| config | ms/step |
|---|---|
| never | 1.99 |
| every step, cells | 2.96 |
| every step, faces | 2.05 |
| every 10th, cells | 2.38 |
| every 10th, faces | 2.02 |

- Profile (256 steps): every-step cells adds eig ops 0.51 ms/step of 1.0.
- Rest: assembly of the 8x8 matrices and gathers of both cell states.
- Same configuration: 2.0 - 5.5 ms/step between runs on gaia (96 cores,
  60-cell problem); pinned to 4 cores (`taskset -c 0-3`) 2.44 - 2.52.
- Cause: XLA's CPU thread pool spreading tiny per-step work over many
  threads.  Small cases: benchmark pinned, compare within one process.

## 14. AmgX pressure solves (optional, GPU)

See `AMG.md`.  `gates/gate14_amgx.py`, Chorin VAM(1) bump, L40S:

| cells (unknowns) | GMRES | AmgX block-AMG | max diff |
|---|---|---|---|
| 60 (120), t = 0.5 | 1.71 s | 1.46 s | 3.1e-15 |
| 600 (1200), t = 0.05 | 29.3 s | 14.5 s | 1.4e-10 |

- Scalar AMG: stalls (rel. residual 0.17) on the indefinite 2x2 block.
- Dense LU / plain FGMRES through AmgX: 3e-15 / 6e-12 -> assembly correct.
- Block AMG (block_dim 2, aggregation + block-Jacobi): converges.

2-D VAM(1) (`gates/gate15_vam2d_amgx.py`, t = 0.02, one solver per process):

| mesh (unknowns) | GMRES | AmgX | max diff |
|---|---|---|---|
| 16^2 (512) | 0.48 s | 0.43 s | 2.0e-11 |
| 32^2 (2048) | 2.13 s | 1.68 s | 3.8e-11 |
| 64^2 (8192) | 16.8 s | 16.5 s | 1.9e-10 |

- Pattern: one LSQ ring (exact; the two-ring superset gave 163 colours).
- Cost: assembling by colouring = one operator evaluation per colour, every
  solve; that eats the AMG gain.  Next: build the AMG hierarchy once and
  reuse it as a frozen preconditioner for the matrix-free JAX GMRES.
- GMRES then AmgX in ONE process hangs (JAX/AmgX interplay, not chased).
- Default `pressure_linear_solver=LinearSolver.auto`: amgx on a GPU with jaxamg.

CPU: PyAMG (smoothed aggregation, block 2x2) preconditioning GMRES on the
host, the matrix assembled in JAX by colouring (one vmapped evaluation per
colour, one-ring pattern), `custom_linear_solve` for AD.  2-D VAM(1),
t = 0.02, 16 cores:

| mesh | GMRES | PyAMG + GMRES | max diff |
|---|---|---|---|
| 16^2 | 0.30 s | 0.04 s | 3.8e-12 |
| 32^2 | 2.54 s | 0.08 s | 3.6e-11 |
| 64^2 | 12.1 s | 0.70 s | 2.8e-11 |

- GPU: AmgX 17.1 s, PyAMG-on-host 16.7 s, GMRES 16.8 s at 64^2 -- the GPU
  time is outside the linear solve (likely the VAM numerical eigensolve,
  CPU-only in JAX); not profiled yet.
- `pressure_linear_solver=LinearSolver.auto`: amgx on a GPU, pyamg on the CPU, when
  installed; else jax.scipy GMRES.  Chorin gates 7/8/12/13 pass on pyamg
  (vs old 8.7e-13, AD rev-jvp 4.8e-12, foam 1.0e-12).

## 15. Reconstruction pair (h, eta)

`ReconstructionSpec.bed = BedReconstruction.derived` (default): with the model's
free-surface map (roles bed, depth; W = (b, b + h, u)) the template
reconstructs h in the bed's slot and sets b_f = eta_f - h_f (Audusse et al.
2004, Sec. 3); "reconstructed" keeps (b, eta).  Rows limited independently
(no ties).  Lake at rest around a dry island (probe_lake_at_rest.py),
PositiveNonconservativeRusanov: O1 / O2 both pairs max|q| <= 1.1e-15.
With the NSM default flux NonconservativeRusanov (no hydrostatic
reconstruction) the same lake goes NaN at O1 already, old solver too.

## 16. Model tags -> numerics defaults

`zoomy_core/systemmodel/tags.py`: `ModelTag` str-enum (`ModelTag.free_surface` == `"free_surface"`).  A
model states `tags`, a user calls `sm.declare_tags("...")`, free-surface roles
imply `"free_surface"`; stored in `sm.history`, inherited by Chorin splits.
`NumericalSystemModel.from_system_model(tag_rules=...)` / `register_tag_rule`
map tags to defaults for what the caller left unset (square systems only):
`"free_surface" -> riemann = PositiveNonconservativeRusanov`.

## 17. CPU scaling, 2-D flow over the bump (bench_vam2d_bump.py)

Escalante bump extruded to 3 m x 0.3 m, VAM(1) Chorin, CFL 0.9, PyAMG
(smoothed aggregation on the block-diagonally scaled pressure matrix) +
GMRES, all 96 cores:

| mesh | cells | steps / sim-s | s / step | wall s / sim-s |
|---|---|---|---|---|
| 60x6 | 360 | ~100 | 0.06 | 6 |
| 120x12 | 1440 | ~210 | 0.15 | 30 |
| 240x24 | 5760 | ~400 | 0.67 | 270 |
| 480x48 | 23040 | ~800 | 3.9 | 3100 |

- Block scaling: V-cycles/solve 66 -> 14 (240), 480 converges (16).
- At 480: pressure solve 49 of 63 s (PyAMG host GMRES).
- 1 minute: T = 1 s -> ~2,200 cells; T = 5 s -> ~650 cells.

## 18. AMG reuse, tolerance, CPU vs GPU (240x24, 20 steps)

| pyamg option | time | setups | V-cycles/step |
|---|---|---|---|
| rebuild every solve | 9.16 s | 20 | 14.2 |
| rebuild every 10 (default) | 7.87 s | 2 | 19.2 |
| rebuild every 50 | 7.42 s | 0-1 | 18.6 |
| tol 1e-8 / 1e-6 | 9.05 / 8.57 s | 20 | 12.7 / 10.3 |

- Options: `pressure_amg_rebuild` (default 10), `pressure_amg_growth` (2),
  `pressure_tol` (default 1e-10).  Results agree to <= 5e-8.
- CPU 8 cores (taskset 0-7): 240x24 0.38 s/step, 480x48 1.43 s/step
  (96 cores: 0.67 / 3.9 s/step).
- GPU (L40S): 120x12 4.0 s/step; 240x24 > 5 min.  Cause, measured:
  `jnp.linalg.eigvals` of 40,000 10x10 matrices 18.4 s on the GPU vs 1.5 s
  on 8 CPU cores -- the VAM flux takes one per face side per stage.

## 19. Spectral radius by power iteration (GPU)

`SystemModel.spectral_radius_method = SpectralRadius.eig | .power | .bound`; "power":
opaque `spectral_radius(k, safety, *A)` = safety * sqrt(rho(A^2)), k power
steps on A^2 from x = 1 (A^2: the +-c pairs of a hyperbolic spectrum).
JAX (fori_loop, any device) and numpy; foam: not yet.

Accuracy on 2880 VAM(1) 2-D A_n (bump, t = 0.1), estimate / exact rho:

| method | min | median | max |
|---|---|---|---|
| Gershgorin (bound k=1) | 1.27 | 3.48 | 3.85 |
| Perron bound k=8 | 1.05 | 1.09 | 1.31 |
| power A^2, k=10 | 0.95 | 1.00 | 1.005 |

Default safety 1.06 (>= rho everywhere measured, <= +6.5 %).  vs "eig" at
120x12, t = 0.25: h 0.7 %, q 2 %, P 8 % (dissipation +6 %, dt).

| bump channel | CPU 8 cores (power) | GPU L40S (power + AmgX) |
|---|---|---|
| 240x24 | 283 ms/step | 54 ms/step |
| 480x48 | 1101 ms/step | 92 ms/step |
| 960x96 | - | 196 ms/step (compile 135 s) |

- eig on the GPU: 40,000 10x10 eigvals 18.4 s (CPU 1.5 s) -> unusable.
- GPU compile grows with the mesh (geometry folded as constants).

## 20. Device defaults, compile time

- `spectral_radius="auto"` (JAX driver): eig on the CPU, power iteration on
  a GPU, safety 1.1 (`spectral_power_safety`), 10 steps; an NSM that sets
  `spectral_radius_method` keeps it.
- Geometry passed as an input (`fold_geometry=False` default): CPU step
  unchanged (379 vs 384 ms, 240x24); AD unchanged (gates 2/5/6/8).
- GPU compile, 960x96: 135 s -> 128 s (geometry input) -> 45 s with XLA
  constant folding off (set on `import zoomy_jax.symsolver`; opt-out
  `ZOOMY_XLA_CONSTANT_FOLDING=1`); step time unchanged (197 ms).
- Pattern / colouring: cells coloured (k x k blocks: column d*k+q takes
  k*colour(d)+q), speculative parallel greedy over the uncoloured cells;
  CSR pattern without a sort.  Trace time flat in the mesh (section 26).

## 21. Distributed hyperbolic solver (MPI ranks, mpi4jax)

Design (`zoomy_jax/symsolver/parallel.py`, guide: PARALLEL.md):

- One MPI rank per device; each runs the serial runtime on its local mesh.
- Local mesh: owned cells, then 2 halo layers (vertex rings: the LSQ stencil
  and limiter of every cell across an owned face).  Built by
  `BaseMesh.from_arrays` (factored out of `from_msh`); cut faces tagged
  `halo` (no BC: the runtime copies the inner state).
- Template: one block `halo(X, mesh)` (serial: identity, JAX and foam), two
  lines -- after each RK stage (`explicit_step`) and after the derivative
  aux (`full_aux`).  No other template change.
- Reductions (`reduce_min/max/any`): every face and cell counted on one
  rank; global by mpi4jax.  min/max: `custom_jvp`, tangent = mean over the
  attaining entries (as `jnp.min`), summed over ranks by `alltoall`.
- AD: alltoall jvp/transpose registered (linear; transpose = alltoall).
  mpi4jax transposes allreduce(SUM) as the identity -- right only for a
  cotangent equal on all ranks; the per-rank shares of dt are not (reverse
  error 5e-2..3e0 before, 1e-14 after).
- Output: snapshots gathered to rank 0 (host, mpi4py), written on the
  global mesh; `on_write` sees each rank's local state.
- Setup checks the local LSQ stencils of owned + first-layer cells against
  the global mesh (raises if the halo is too thin).
- Not yet: Chorin (pressure solve), IMEX, preCICE, periodic meshes.
  (1-D: enabled with the stencil fix of section 23.)

Gates (CPU, mpiexec, vs serial):

| gate | 2 ranks | 3 ranks | 4 ranks |
|---|---|---|---|
| 16: SWE bump O1 | 2.1e-16 | 3.1e-16 | 2.9e-16 |
| 16: SWE bump O2 + MOOD | 5.8e-16 | 7.1e-16 | 7.1e-16 |
| 16: dry dam break eta + MOOD | 2.4e-15 | 4.3e-15 | 3.3e-15 |
| 16: SWE bump O2 + MOOD, triangles | 1.5e-15 | 1.1e-15 | 1.7e-15 |
| 17: AD O1 adaptive: dJ/dg fwd / rev | 2.8e-15 / 8.0e-15 | | |
| 17: AD O1 adaptive: dJ/dX.v fwd / rev | 3.7e-12 / 8.4e-13 | | |
| 17: AD O2 + MOOD adaptive: dJ/dg fwd / rev | | 1.9e-16 / 1.3e-14 | |
| 18: HDF5 output (6 snapshots) | | 1.3e-15 | |

GPU (L40S, one rank per GPU bound automatically, MPICH not CUDA-aware:
exchange staged through host memory; gate 16 on 4 GPUs <= 1.7e-15, gate 17
on 2 GPUs: rev 8.9e-15 / mesh 1.9e-12; exchange switched off: 4.46 vs
4.18 ms/step at 160k, i.e. no measurable exchange cost),
SWE bump O2 + MOOD, constant dt (bench_parallel.py):

| cells | GPUs | ms/step | speedup | setup | compile |
|---|---|---|---|---|---|
| 40k | 1 | 5.07 | | 32 s | 22 s |
| 40k | 2 | 4.58 | 1.11x | 37 s | 19 s |
| 160k | 1 | 12.34 | | 71 s | 29 s |
| 160k | 4 | 4.18 | 2.95x | 74 s | 18 s |

- Setup is serial Python on every rank (global mesh, geometry, LSQ
  stencils): 45 s for the 160k mesh alone; extrapolated ~50 min at 10M.
  Next bottleneck, before any larger run.
- 1-D aside (pre-existing): `from_msh` 1-D meshes get 3-point asymmetric
  LSQ stencils (e.g. cell 100 -> [99, 101, 98]; `create_1d`: [99, 101]):
  ghost rows widen the legacy path's global max.

GPU transport (PARALLEL.md table): CUDA-aware OpenMPI 5 + UCX works
(gate 16 on 4 GPUs <= 2.8e-15) once `opal_cuda_support`, `pml=ucx` and
`UCX_MEMTYPE_CACHE=n` are set; exchange 52-74 us vs 78 us through host
memory (63 KiB); step time unchanged on this PCIe node (6.72 vs 6.73 ms).
Matched rerun of the 160k bench (400 steps): 7.00 ms MPICH, 6.73 ms OpenMPI
staged -- the 4.18 ms above was 200 steps (cause of the gap not isolated).
Container `containers/zoomy_jax_mpi` (OpenMPI 5 + UCX CUDA-aware, mpi4jax
CUDA, JAX 0.9.2): `apptainer test` OK; gate 16 on 4 GPUs <= 1.3e-15; gate 17
(2 GPUs) rev 4.0e-15, mesh 5.8e-13; 160k cells 7.06 ms/step (7.20 staged).

## 22. Large meshes: vectorised setup, mesh files, parallel I/O

Setup (serial, CPU), per stage:

| stage | 160k before | 160k after | 1M after |
|---|---|---|---|
| `FVMMesh.from_base` (geometry) | 25.3 s | 0.1 s | 1.2 s |
| `LSQMesh.from_fvm` (stencils, matrices) | 6.3 s | 0.6 s | 6.0 s |
| `BaseMesh.create_2d` | 1.5 s | 0.3 s | 2.5 s |
| local mesh of rank 0 (4 ranks) | 14.3 s | 0.3 s | 1.9 s |

- One vectorised geometry: `FVMMesh` / `LSQMesh` call `geometry_xp`
  (the solver's AD geometry) for line/triangle/quad; 3-D keeps the loops.
- Vectorised: face topology (`np.unique` on face keys), cell and face
  neighbours, the >= 2-D vertex-ring stencil (one sparse product; loops
  only for ghosts and too-small rings), face subvolumes.
- Gate 19: every builder equal to the replaced loops (zoomy_core 6b20967)
  on quads, jittered quads, triangles, 1-D, gmsh, 3-D: integers exact,
  floats <= 1e-14; LSQ degree 1 and 2.

Distributed meshes (`zoomy_core.mesh.distributed`, backend-neutral):

- Global numbering: cells along a Hilbert curve (2-D; Morton 3-D),
  vertices by first appearance; rank r owns a contiguous block.
- Mesh file (`write_mesh`): the output `/mesh` group + `cell_face_tags`
  (format 3); output files are again inputs.
- `distribute`: vertex directory (block of vertex ids per rank); layer 1 =
  cells around the owned cells' vertices, layer 2 = around layer 1's new
  vertices; each rank reads only its cells, vertices, tags (slices of
  consecutive ids).  No rank holds the whole mesh.
- Output: mesh file + h5py with MPI -> every rank writes its block (mesh
  copied blockwise from the input); otherwise gathered on rank 0.
- `setup(path_or_mesh, model)`; serial runs read a mesh file too.

Compile time: the runtime's mesh index arrays were baked into the program
(~800 B/cell: 51 MB at 62k cells, 198 MB at 250k).  `hoisted_jit` traces
once and passes every captured constant as an argument: compile no longer
grows with the mesh (CPU, trace + compile: 8.7 s at 62k, ~8 s at 250k).

Gates: 16 (now Hilbert partition; + mesh-file case 9.4e-16, 3 ranks),
18 (in-memory 1.6e-15; mesh file gathered 1.3e-15; per-rank blocks with
h5py-MPI 1.1e-15), 17 (AD) unchanged.

GPU, 4 x L40S, SWE O2 + MOOD from a mesh file (bench_large.py):

| cells | per GPU | setup | compile | ms/step |
|---|---|---|---|---|
| 1M | 250k | 14 s | 4 s (cache) | 20-25 |
| 4M | 1M | 26 s | 6 s (cache) | 101.5 |
| 16M | 4M | 82 s | 19 s (cold) | 414.5 |

- Before: 1M setup 41 s; 4M setup 81 s, compile 64 s; 16M compile > 210 s.
- Writing the 16M mesh file (serial, once) took minutes; per-rank setup
  is now ~20 s per million local cells.

## 23. 1-D gmsh stencils (fix)

- Bug: the 1-D (legacy) LSQ stencil width was the max over ALL rows; a
  `from_msh` ghost row copies its inner cell's neighbours, its expansion
  keeps the inner cell itself -> width 3 -> every interior stencil padded
  one-sidedly (cell 100: [99, 101, 98]; `create_1d`: [99, 101]).
- Fix: width over the inner cells (ghost rows cut to it), in both copies
  of the 1-D loop (`geometry_xp`, `lsq_reconstruction`); the >= 2-D vertex
  ring is untouched.
- Gate 19: only "from_msh line" changes (width 3 -> 2); its inner stencils
  and LSQ matrices now equal `create_1d`'s exactly.
- Distributed 1-D enabled (the width is now local): SME(1) dam break O2,
  2 and 4 ranks: 0 until the front reaches the cut, 4e-15 at t = 0.02,
  1.7e-11 at t = 0.1; O1 7.8e-16.  Round-off amplification of the O2 run
  itself: serial, one cell perturbed by 1e-15 -> 6.2e-12 at t = 0.1 (O1
  6.7e-16).
- `tests/unit/zoomy_core/test_mesh_snapshot.py`: 29 failures before and
  after (its `.msh` inputs are not in the repo; `compute_derivatives` now
  requires `u_boundary_face`).

## 24. Parallel mesh writing; a 14.8M-cell catchment

`write_mesh_parallel(comm, path, type, X, v0, cv, c0, faces=, face_tags=,
cell_data=)`: every rank passes its blocks of vertices, cells and boundary
elements; no rank holds the mesh.  Distributed sample sort along the curve,
vertices renumbered by first appearance (vertex directory), boundary faces
and tags by face keys hashed to their smallest vertex, one block per rank
written with MPI-IO (gathered without h5py-MPI).  `cell_data` (e.g. the
bed) travels with the cells; `FileSource.cell_data` reads it per rank.

- Gate 20: the parallel file equals the serial `write_mesh` file, every
  dataset (quads, jittered quads, triangles, 1-D, hexahedra; 3 ranks; with
  and without h5py-MPI).

SERGHEI LowerTriangle catchment (thesis/cases/river/serghei; RainSWE: rain
30 mm/h, Manning 0.03, first order, PositiveHLL, point-implicit friction,
CFL 0.5, dt_max 5 s, walls; `bench_catchment.py`):

| step | result |
|---|---|
| 10 m LowRes, 148k cells, 1 h, 1 GPU | 5-7 s; water 445017.0 m3 = r_o A t; h_max 4.155 m (old solver 4.1561) |
| HighRes DEM 0.5 m -> 1 m (block mean) | 5604 x 5264, 14.77M catchment cells |
| mesh file, 4 ranks in parallel | 24-30 s, 1.77 GB (b + mapped state as cell data) |
| start state | the 10 m 1 h state mapped by depth + velocity (443082 m3) |
| setup, 4 GPUs (3.69M owned/rank) | 66-77 s |
| 10 s simulated | 13.3 s wall; water change 1230.553 m3 = rain 1230.553 m3 |
| CFL dt | 0.022 s -> 0.021 s: ~470 steps, ~28 ms/step (7.6 ns/cell-step) |
| output, 3 snapshots (MPI-IO blocks) | +3.8 s; 3.07 GB |

- 1 h of the 1 m catchment ~ 1.3 h wall on 4 L40S; the SERGHEI 72 h run
  ~ 4 days.  Same case at 10 m on 1 GPU: 1 h in 5-7 s.
- dt from the CFL estimate of the actual state at start and end (the
  template's loop has no step counter).

Container `zoomy_jax_mpi` (rebuilt): the 14.8M-cell catchment, 10 s with 3
snapshots on 4 GPUs: 17.5 s (env 17.1 s), water change = rain exactly;
setup 90 s.  Two container fixes: the GPU MPI settings (UCX) only when
`--nv` binds the driver (without a GPU, UCX cannot start); the halo
exchange through host memory by default -- GPU -> GPU stalled at this size
(halo messages up to 350 kB per peer, 13.6k-20.5k halo cells per rank).

## 25. Distributed Chorin (pressure solves)

Template (`strategies/chorin.py`): `halo` on the pressure iterate inside R(P) (its
derivative aux is a stencil), after the pressure solve and after the
correction (+ after the predictor's explicit diffusion).  Serial: identity
(foam: a template returning by value, also for the scalar 0 of R(0)).

Runtime, distributed `linsolve`:

| backend | solve | preconditioner |
|---|---|---|
| gmres | FGMRES, global inner products (one allreduce per CGS2 pass), Givens early exit, restart <= 60 | none |
| pyamg | same FGMRES | block Jacobi: per-rank SA V-cycle (host, lagged, block-diagonal scaling) + coarse correction (one unknown per rank and row, Nicolaides) |
| amgx | ONE global AmgX solve, AmgX MPI mode via JAX-AMG (FGMRES + AMG as serial) | AmgX's own, global |

- amgx: each rank assembles its owned rows by colouring the operator with
  the halo off (`parallel.no_halo`), halo columns numbered by their owners;
  MPI plans from a scipy placeholder at trace time (`cache_mpi_metadata`).
  Needs AmgX built with MPI and JAX-AMG with `JAXAMG_ENABLE_MPI=1` (AMG.md).
- GPUs: `python -m zoomy_jax.gpurank` binds `CUDA_VISIBLE_DEVICES` before
  CUDA loads; binding inside the run left AmgX / mpi4jax on GPU 0 (OOM).

Gates (vs serial; agreement = the elliptic tolerance, not round-off):

| gate | result |
|---|---|
| 21, VAM(1) 2-D bump, gmres, 2/3 ranks | 8.9e-12 / 3.8e-10 |
| 21, pyamg, 2/3/4 ranks | 1.0e-10 / 1.3e-10 / 1.3e-10 |
| 21, amgx (MPI mode), 2/4 GPUs | 4.1e-10 / 2.0e-10 |
| 22, Chorin AD (1-D VAM, 3 ranks): J, dJ/dg fwd/rev, dJ/dX.v fwd/rev | 1.8e-13, 5.3e-12/6.0e-12, 9.6e-11/2.4e-11 |
| 7, 8, 12, 13, 9 (serial, foam) | unchanged |

Iterations per solve (VAM(1) bump, 960 x 96): serial AmgX ~128; 4 ranks
AmgX MPI ~90; FGMRES + block Jacobi (1 AmgX V-cycle per rank) ~220 (~260
without the coarse correction); `reuse_setup` (lagged hierarchy) does not
converge (500).

GPU timing (L40S, PCIe, MPICH -> exchanges through host memory):

| cells | 1 GPU | 4 GPUs (AmgX MPI) |
|---|---|---|
| 92k | 208 ms/step | 828 ms/step |
| 369k | 689 ms/step | 1057 ms/step |

- The distributed AMG exchanges halos on every level of every V-cycle (~90
  per solve): latency-bound at these sizes on this node.  The gap closes
  with size (4x -> 1.5x); not measured beyond 369k here.

## 26. Compile time flat in the mesh; distributed preCICE

- The 1-GPU AmgX growth was the TRACE, not XLA (hoisted_jit already keeps
  constants out): the unknown-level Jones-Plassmann colouring inside
  `assembled` (90 rounds over P^T P, 10M entries at 100k cells).  Cell-level
  colouring (above): trace 69 -> 5.8 s at 400k cells; XLA 20-30 s at every
  size.  1 GPU, compile + 1 step: 0.73M cells 150 -> 43 s, 1.47M timeout ->
  49 s.  Same pattern bitwise; 40 vs 36 colours (VAM(1)); ms/step unchanged.
- Open: 2.95M cells on 1 GPU: OOM in the colour-probe vmap (33 GiB);
  196k cells: the first AmgX solve misses the 1e-6 accept gate (old and new).
- preCICE in distributed runs: one `Participant(name, cfg, rank, size)` per
  rank; the runtime registers each rank's owned coupling faces.  Every preCICE
  effect's token passes an MPI reduction before and after its callback, so the
  call stays in the rank's ordered MPI chain (without: a rank with no coupling
  vertices ran `advance` before its halo exchange -> deadlock).  pyprecice 3.4
  passes preCICE a size-1 communicator for an mpi4py one: one `mpiexec` per
  participant (COMM_WORLD), other communicators refused.
- Gate 28 (`gate28_jax_couple_parallel.py`, `participant_mpi.py`): 1-D and
  2-D, O1/O2, windows and subcycling, up to 3 + 3 ranks vs 1 + 1:
  <= 1.4e-14.  Gate 3 unchanged (3.4e-15 O1, 1.8e-2 O2 coupled vs single).

## 27. foam on OpenFOAM's own components (parallel, pressure)

- Mesh: blockMesh (1-D, quads); unstructured 2-D through gmshToFoam (one
  extruded layer).  Distribution: decomposePar (scotch), `-parallel`; the
  halo is the processor patches (`correctBoundaryConditions`).
- Stencils: `fvc::grad` (leastSquares), `cellLimited` limiters, compact
  second derivatives (Gauss integral of the face gradient).  Reductions:
  `returnReduce` with an explicit NaN check.  Host program: zoomy_core
  `symsolver/backend_cxx.py`, shared with AMReX.
- Pressure: FGMRES on the full system (global dots); right preconditioner =
  the probed compact operator, algebraic rows eliminated per cell, GAMG on
  the elliptic rows' Schur complement (block Jacobi across ranks).  Same
  solution (gamg vs none at tol 1e-12: <= 7e-13); iterations 1-D 7614 ->
  1725, 2-D 973 -> 200.
- foam no longer matches JAX to round-off at second order (different
  stencils).  Smooth bump (flow 0, t 2.5): max |foam - JAX| 1.0e-3 / 1.7e-4
  / 3.4e-5 at nx 40/80/160 (order ~2.5).  Wall-shock cases: L1 (1.8e-3);
  their max does not converge between any two schemes (JAX vs JAX at another
  CFL neither).  Gates 9/12/13 switched to these criteria; 1-D and O1 stay
  at round-off.  Parallel = serial: hyperbolic <= 3.5e-15, Chorin <= 1e-10.
- preCICE with `-parallel`: one participant per rank; gate 26 (2+2 ranks)
  5.2e-15 vs single domain.  Gates 23-26.

## 28. Boundary stencils of the LSQ derivatives

Found by the debugger on the gate-21 channel (y-invariant, Extrapolation
walls), where JAX grew a spurious q_y ~ 3e-3 that foam did not.

- Cause 1: a degree-2 fit needs 5 samples (no constant), the ring check asked
  for 6 and did not count the boundary ghosts.  Every wall cell (5 ring
  cells + 1 ghost) and corner (3 + 2) was extended one-sidedly into the
  domain (12 cells): d/dy of a y-invariant b(x) 1.06e-3 at the walls, and
  d2h/dx2 at the inflow corners 0.4 off the row above.  Fix: count ghosts,
  extend below 5 only (`geometry_xp._lsq_stencils`).
- Cause 2: the ghost sample sat at x_c + 2(x_f - x_c), the point reflection,
  not the mirror image x_c + 2((x_f - x_c).n) n where the ghost centres and
  the BC kernels' ghost values (`d_normal`) live.  Same on Cartesian cells;
  on triangles the boundary derivatives did not converge (first O(1), second
  diverging).  Fix: mirror offset (`geometry_xp._lsq_matrices`).

| check | before | after |
|---|---|---|
| channel q_y at t = 0.05 (gate-21 case) | 3.1e-3 | 1.9e-14 |
| d/dy b(x), wall rows, NX 60/120/240 | 1.1e-3 / 1.4e-4 / 1.8e-5 | <= 1.7e-14 |
| quads, Neumann field, wall rows: d/dy, d2/dy2 (NX 240) | 9.6e-5, 1.5e-2 | 5.8e-5, 7.3e-3 (same orders 3, 2) |
| quads, corners, d2/dx2 (NX 80) | 4.8e-2 | 7.4e-3 (order 2) |
| triangles, boundary cells: d/dx order, d2/dx2 order | 0.0, -1.0 | 2.0, 1.0 |

One-sided limits stay: at a Dirichlet (odd ghost) wall, d2/dn2 is first
order (0.98) before and after.  Gates: 1, 5, 7, 16 (4 ranks), 21 (3 ranks;
q_y now at round-off, so its relative metric got a floor: abs 2.8e-13), 22
unchanged or better; 19 marks the stencil change as intended.  The old
solvers share the mesh, so they get the same stencils (gate 1 unchanged).
Literature (Zotero): the half-distance boundary sample mechanism, Syrakos
et al. 2017 (Phys. Fluids, GU9F3I5R) eq. 13; constrained LSQ with BC rows as
the ghost-free alternative (Koellermeier thesis VCXBJ6RN p. 137).

Addendum (degree 1, the MUSCL gradient): the vertex ring of a wall cell
(three cells inside, one ghost outside) has no curvature term to absorb the
imbalance, so a field constant along the wall got a wall-normal gradient
6.0e-2 / 3.1e-2 / 1.6e-2 (NX 60/120/240): the q_y ~ 5.6e-4 of JAX and AMReX
in the matrix's channel case C4 (foam 1e-15).  Boundary cells of degree-1
stencils now use face neighbours + ghosts (mirror-symmetric; OpenFOAM's
choice): leak 0; wall gradients on quads first -> third order (Neumann d/dn
1.2e-1 -> 1.5e-3 at NX 60); triangles' boundary errors 2-3x smaller (first
order, as their interior).  foam vs JAX: wall-shock bump L1 1.8e-3 ->
6.5e-4; smooth bump nx 80 max 1.7e-4 -> 1.3e-4.  Gates 1, 9, 16, 19 OK;
gate 5's O2 FD check stays loose (limiter kinks: |FD-jvp| 2.1e-5 on 1.5e-5;
rev = jvp to 1.2e-11).

## 29. Printed constants, loop-invariant code, distributed IMEX

- Floats print as the exact double (`repr`); sympy's 15 digits made a
  constant-dt run end 4e-20 early and take one more 4e-20 step whose
  pressure operator is ~1e-13: AmgX ran 500 iterations and failed the
  accept gate (196k cells: 641 -> 209 ms/step).  The old GPU Chorin scaling
  numbers (ANALYSIS 25) include that step: re-measure.
- The printer placed each statement in the innermost scope of its uses, so
  the IMEX source's frozen derivative aux (with its halo exchange) landed
  inside the Newton residual: computed per iteration, and on 3+ ranks
  (different Newton counts) a deadlock (gdb: ranks in different
  collectives; MPI4JAX_DEBUG: one extra alltoall).  Now a statement leaves
  every function with parameters it does not depend on (loop bodies, Root
  residuals, LinSolve operators; Cond branches keep theirs).  Gates 1, 2, 6,
  7, 8, 9, 12, 13, 16, 27 unchanged.
- IMEX in distributed runs: gate 29, 2/3/4 ranks vs serial <= 4.5e-13
  (1-D, Newton tolerance) and 4.2e-16 (2-D).  Implicit diffusion stays
  serial (its dense operator): refused in distributed runs.

## 30. 3-D and the compressible Euler equations (a SystemModel written directly)

- `zoomy_core/systemmodel/euler.py`: `compressible_euler(dimension, gamma,
  ...)`, state (rho, rho u_d, E), built with the SystemModel constructor,
  BCs, roles and admissibility (rho, p for MOOD): no new API.  Wall now reads
  the declared momentum role (it derived the rows from the SWE layout).
- 3-D: geometry_xp handles tetrahedra / hexahedra; the boundary stencils of
  ANALYSIS 28 apply in 3-D.  Templates unchanged.  Two periodic-seam fixes
  in the JAX runtime: the seam takes the partner face's trace (O2 was first
  order there) and the limiter ghost bounds go to the face owner (matrix
  C13 O2 shift 7.8e-3 -> 1.6e-15).
- JAX 3-D (<= 24^3): Sod along x/y/z identical (1e-13) and = 1-D (1.6e-13);
  smooth diagonal wave O1 0.85, O2 2.23 / 2.53; wall box mass / energy
  1e-15 and mirror symmetry 1.8e-15; 4 ranks = serial 5.3e-16; AD d/dgamma
  FD 4e-8, rev = jvp 7e-15.  2-D Euler: Sod x = y, wave O2 2.5.
- foam 3-D (blockMesh hex): Sod = JAX (1.7e-14, same stencils); acoustic
  pulse foam-JAX converging (order 2.1); 2/4 ranks = serial.  AMReX 3-D:
  not done (~250-350 lines, an AMREX_SPACEDIM=3 build).
- Gates gate3d_a..f (+ euler_common.py, gate3d_figure.py).

## 31. Distributed periodic BCs and implicit diffusion; the diffusion sign

- Periodic in distributed runs: the owned cells' periodic partners (with
  their stencils) join halo layer 1, layer 1's partners layer 2;
  `Partition.resolve_periodic` points the piece's seam faces at the halo
  copies, the halo exchange keeps them current (in-memory meshes; a mesh
  file refuses).  Gate 32: 1-D O1/O2, 2-D x+y periodic O2+MOOD, 2/3/4 ranks
  vs serial <= 7.9e-15, mass drift <= 2.2e-16.
- Implicit diffusion: one runtime path for serial and distributed runs --
  Crank-Nicolson residual, Newton (global norm: every rank takes the same
  number of updates), each update a `linsolve` (distributed: `_dlinsolve`).
  Gate 31: vs the legacy dense operator 9.3e-17, 3 ranks vs serial
  7.1e-15; AD rev = jvp to 7.6e-14.
- AD on one GPU = CPU (gates 2, 8) when both use the same spectral radius.
- The diffusion SIGN: the contract (and the stability analysis, the MMS,
  every runtime block, the hand-written Malpasset tensor) is RHS
  +div(A : grad Q) with A > 0 for diffusion, but the extractor put the
  RESIDUAL coefficient into A: every extracted viscosity (NewtonianInPlane,
  sigma-3D) was anti-diffusive in every solver (A[q, q] = -2 nu).  Fixed in
  `system_extract._route_diffusion`; goldens m07 / m14 / m16 updated (only
  their diffusion lines).
- 8 model goldens (m01-m04, m06-m08, s01) were stale since 48ceef8 ("SME:
  numerical eigenvalues by default"; found by git bisect): their only
  difference was the dropped analytic `eigenvalues` block.  Re-blessed;
  the golden suites pass (17/17).

## 32. foam: periodic, IMEX in parallel, Chorin on triangles (8534d22)

- Periodic -> OpenFOAM cyclic patches (blockMesh 1-D/2-D/3-D, gmshToFoam),
  processorCyclic across ranks; seam halo centres = periodic images.
  Gate 35: O1 = JAX to 1.6e-15 (1-D/2-D/3-D/tri), O2 converging, 2/4 ranks
  = serial <= 3.7e-15.
- Correction to section 27: foam's DERIVATIVE AUX (the Chorin pressure
  operator's first/second derivatives) is now zoomy_core's LSQ fit on
  zoomy_core's stencil (written by the driver, gathered by a
  distributionMap), not fvc::grad + the compact Gauss second derivative:
  the latter is inconsistent on triangles (|d2/dx2| error 1.84 at every h;
  Chorin on triangles vs JAX L1 0.24).  Now: triangles L1 2.8e-6 (gate 36),
  quads gate 12 / 24 at the pressure tolerance (2.8e-12 / 4.6e-11).  The
  reconstruction for the fluxes stays OpenFOAM's (leastSquares,
  cellLimited).  inradius as zoomy_core (O1 triangles 2.6e-3 -> 4e-16).
- IMEX with -parallel: gate 11 reworked, 2/4 ranks = serial 3.5e-12 (Newton
  tolerance).
- preCICE + Chorin: needs a template change (couple the predictor ~60
  lines; implicit pressure coupling with checkpoints ~100 per backend):
  not done.
- Open: a JAX periodic shift-invariance check (foam's gate 35) shows 8.3e-6
  / 5.9e-5 / 8.5e-4 (1-D / 2-D / tri) where foam has 1e-15.

## 33. Periodic seams wrap the LSQ stencils (core 20712a9, jax b59227c, foam 3c51d7b)

JAX treated a periodic seam as a wall: the seam cell took the boundary
stencil (face neighbours + a ghost; 1-D even extended one-sidedly), the
interior the vertex ring -> the result moved with the seam (shift by half a
period: 1-D 1.2e-4, 2-D 2.2e-3; foam 1e-15).  Now `build_topology`
identifies seam vertices across the period: seam cells get the wrapped
interior ring, offsets to the nearest periodic image, no ghost sample;
distributed halos ring across seams.  JAX and foam take their stencils
from this topology (foam's own ring code removed).  Shift invariance: 1-D
2.2e-15, 2-D 6.2e-16 (my rerun), triangles 2.1e-15, 3-D 6.8e-16; gate 32
(3 ranks) <= 1.2e-14; gate 35 1-D O2 foam vs JAX 3.7e-15.  AMReX: its
periodic path now takes the interior stencil on periodic axes, the
cross-seam samples being the partner cells in the halo (a9ef6c2; the class
also had to take a true interior proxy cell): AMReX vs JAX 5e-16 - 2e-15
(1-/2-/3-D, SWE and Euler; cpu1, gpu1; cpu2 = cpu1), half-period shift
<= 1.7e-15, AMR periodic waves EOC 2.5 / 2.3.  (3-D periodic wave O2
cpu1 vs JAX: 5.5e-16, rerun.)

## 34. Boundary ghosts: BC position and distance

A ghost sits at the mirror image of the cell centre, but the runtimes
evaluated the BC kernel at the FACE CENTRE with the distance |x_f - x_c|.
On orthogonal cells the two coincide; on triangles a position-dependent
Dirichlet value (inflow profiles, data) put an O(h) error into the ghost:
boundary first derivatives did not converge (8e-2 at nx 12/24/48) and
second derivatives diverged (1.5e-1 -> 6e-1).  Now ghosts
(boundary_states / boundary_ghosts) take the foot of the perpendicular
and the normal distance (dn = (x_f - x_c).n, which Neumann gradients
need); face traces (boundary_traces) stay at the face centre.  With the
foot point: first derivatives order ~1, second derivatives bounded
(2.5e-2; a single odd ghost cannot do better at a Dirichlet wall).
JAX and foam; AMReX is Cartesian (no change).  Gates 1, 16, 23 tri2
unchanged (constant BC data).

## 35. SSP-RK3; the Chorin step is first order in time

`SSP_RK[3]` (Shu-Osher; stage time = the stage input's time) and a
`time_order` option (default: the reconstruction order) on the templates,
JAX and C++ drivers.  Gate 40: ODE with exact solution 1.01 / 1.92 / 3.00
(rerun), forced SWE wave on a fixed mesh 1.28 / 2.07 / 3.00, 2 ranks =
serial 4e-16; a wrong last-stage time drops RK3 to 0.99 (the gate catches
it).  Default output byte-identical.  Open: the Chorin projection stage is
itself first order (Lie-split sources, pressure at the stage end), so its
temporal order stays ~1.2 for time_order 1, 2, 3; a second-order
projection is needed before the pressure solver can claim O2 in time.

## 36. Diffusion fluxes through boundary faces; two BC kernels

The diffusion blocks summed interior faces only: every wall acted as
homogeneous Neumann (no wall shear, no Dirichlet diffusion).  Now
`diffusion(k, bc, Q, Qaux, p, time, mesh)` / `implicit_diffusion(k, bc, Q,
Qaux, p, time, dt, mesh)` add n.A.n (X_ghost - X)/(2 d_n) per boundary face
with the D5 GHOST kernel (`nsm.call.boundary_ghost_state`, now in the NSM
call table), the CN iterate's ghosts at t+dt.  Gate 41 (1-D, quads,
triangles; explicit and CN; Dirichlet, Neumann, position-dependent
Dirichlet, periodic): order 2.00 everywhere (before: no convergence, errors
0.26-1.5), linear steady state exact, 2 ranks = serial 4.8e-15.

Found on the way: the templates' `boundary_states` uses the OTHER kernel,
`boundary_conditions`, whose value is a FACE value for Dirichlet-type BCs
(Dirichlet q = 0.11197 at the gate-21 inflow) but a ghost for reflecting /
extrapolating / periodic BCs.  The ghost consumers disagree with it:
gradient and limiter bounds treat it as a ghost (Dirichlet slope halved),
the JAX derivative aux lifts 2Qb - Q (right for Dirichlet, wrong for walls
and periodic partners).  Being unified on the D5 ghost kernel (section 38).

## 37. LSQ stencils chosen by rank, not by sample count

Gate 39 (polynomial reproduction, degree 1-3, 1-D / quads / jittered quads
/ triangles / 3-D hex; interior, exact-ghost boundary, BC-built ghosts)
found 4 rank-deficient stencil classes left by the ANALYSIS-28 count rule:
3-D hex degree 3 interior (the 3x3x3 ring has x^3 = x: rank 16 of 19),
triangle corner neighbours degree 2, triangle boundary cells degree 3,
jittered-quad corners degree 2.  Now a ring is kept iff every coefficient
the fit reads has relative singular value >= 1e-3 (ghosts count only for
their pure-normal rows, REQ-174b); otherwise the next face-neighbour layer
is added, nearest equidistant group first, <= 2 layers; 1-D the same.
Gate 39: 57 PASS / 0 FAIL / 15 XFAIL (BC-built Dirichlet ghosts at degree
>= 2 and Neumann at degree 3: the boundary closure's job).  Gates 1, 16,
32 identical; wall d/dy check unchanged; setup 160k quads deg 1/2/3
0.56/1.21/5.9 s -> 0.67/1.44/4.8 s.  1-D degree 2 is now the compact 3-point
fit (was 5).  Unit test tests/template_gates/test_gate_1a_core_lsq_unisolvent.py.

## 39. Cleanup: template classes, one C++ backend, one C++ printer base

- Templates: `strategies/` (HyperbolicTemplate -> ChorinTemplate) and
  `discretization/` (reconstruction, residual, time); 46 snapshots
  byte-identical.
- `backend_cxx.py` (CxxBackend, CxxChorinBackend; foam / AMReX drivers as
  mixins): program.py gone, drivers 1135 -> 1074 lines; 554 generated files
  byte-identical.
- `transformation/cxx_base.py` (CxxBase, on CXX17): the solver lane's
  CxxPrinter and the kernel lane's GenericCppBase (Cpp, Foam, Amrex,
  GLSL/JS out-param printers) derive from it; call syntax and n-ary Min/Max
  shared, Pow / Abs / Float stay lane policy; 1088 generated files
  byte-identical (both lanes, legacy generators, GLSL/JS, a 172-expression
  battery).  Net +36 lines: the shared surface is small.
- Runtime duplication (measured, not changed yet): rt.H / rt_amrex.H are
  ~10% / 6% backend-neutral, ~35% similar (different containers), ~56%
  specific.  A ~150-line shared header (preCICE sequence, configure()
  settings, writer / CFL check, NaN-safe min/max, Newton skeleton,
  elementwise ops) removes ~270 duplicates: planned for drift control.

## 38. One ghost semantics (the D5 kernel) for every ghost consumer

The gradient, limiter bounds, the Chorin pressure operator's derivative aux
and the diffusion blocks all take `nsm.call.boundary_ghost_state` now
(reconstruction.py, chorin.py; built only where used); the runtimes' own
`2Qb - Q` lift is gone (runtime.py, rt.H, rt_amrex.H; foam / AMReX print
the ghost kernel).  The Riemann outer state keeps the face kernel at the
face centre (`boundary_traces`: a ghost there would add 2 d g for
Neumann).  Gate 42 (real BC objects through the runtime blocks: 1-D /
quads / triangles x Dirichlet const and position-dependent, Neumann,
Extrapolation, Wall, Periodic): 18/18 exact to 1.9e-15 (old kernels 6e-2 -
5e-1).  Gates 1, 7, 9, 12, 16, 32, 41 identical; 21 / 27 at the pressure
tolerance.  Changed by design: Dirichlet + O2 (matrix C4/C11 sub O2: JAX
moves 2.5e-4 -> 5.1e-6 at n 25 -> 100, order unchanged).  NOT the cause of
the AMReX-vs-JAX C4 mismatch: that one is at O1 too, only on non-square
cells -- AMReX's CFL size d|V|/|dV| (0.333) vs zoomy_core's inradius
(0.25).  Open: full_aux's derivative aux ignores BCs (zero gradient);
FromData builds its own 2v - Q.

## 40. C++ runtimes: AMReX CFL size, diffusion blocks, rt_common.H

- AMReX's face_inradius was d|V|/|dV| (0.333 on non-square cells), zoomy
  core / JAX / foam use min |(x_f - x_c).n| (0.25): different dt.  Now min
  dx/2.  Matrix C4 AMReX vs JAX (forced rerun): O1 1.2e-15 / 8.3e-16, O2
  sub 7.3e-14, O2 trans 1.7e-12 (JAX vs JAX with Q0(1+eps) gives 1.3e-11:
  sensitivity, the bound must follow it).  C11 (4.9e-6) is the same
  sensitivity: JAX vs perturbed JAX already differs by 4.9e-6.
- diffusion / implicit_diffusion in foam (rt.H) and AMReX (rt_amrex.H), as
  JAX (ANALYSIS 36).  Gate 43: 90 checks, explicit at round-off (<= 4e-15),
  implicit at the solver tolerance (<= 2.5e-10), 2 ranks = serial.  AMR +
  diffusion refuses (diffusive fluxes are not refluxed yet).
- `zoomy_core/symsolver/rt_common.H` (278 lines): preCICE sequence,
  configure() settings, writer / CFL check, NaN-safe min / max, Newton
  skeleton, elementwise functors.  rt.H -157, rt_amrex.H -181 (net -60).
  Gates 11, 12, 23, 26, 27, 34, 37, 38 bitwise identical; CUDA build OK.

## 41. Constrained LSQ boundary closure (zoomy_core + JAX)

Each state row is classed per boundary face from the D5 ghost kernel's
Jacobian (`numerics/boundary_closure.py`): Dirichlet (ghost = -Q + 2v),
Neumann (ghost = Q + 2 d g) or ghost (coupled rows: oblique walls).
Dirichlet / Neumann rows enter fits of degree >= 2 as EXACT constraints at
the foot point (KKT, precomputed in geometry_xp; the runtime inputs, the
ghosts, are unchanged); degree-1 fits keep the ghost sample (measured
better there).  Used by the JAX gradient, derivative_aux and full_aux
(which now sees BC ghosts at all; it assumed zero gradient).  Gate 39: 72
PASS / 0 FAIL (was 57 + 15 XFAIL), group (c) through the real closure;
gate 42 18/18; gate 44 (boundary orders, quads / triangles, p 2/3,
Dirichlet / Neumann): order p+1-k reached (ghost sample: d2 ~0.1).  Gates
1, 7, 12, 16, 32, 41 identical; gate 21 changes at the Dirichlet-q inflow
only (P_0 0.29 -> 0.064 at nx 60, toward the refined runs); C10 orders
1.92 / 2.00 / 2.01.  Open: oblique-wall rows (frame rotation), foam /
AMReX follow (CXX_NOTES), setup cost of the classification (gate 39 27 s
-> 171 s; gate 44 > 5 min).

## 42. Closure in foam / AMReX; conditioning of the decoupled wall fits

- foam: the derivative aux reads zoomy_core's tables (base `lsq_gradQ`,
  one `lsq_closure` table per active class) instead of its own Eigen pinv;
  slots matched to foam's boundary faces by face centre.  Gate 24 serial
  3.2e-3 -> 7.6e-11, gate 36 (triangles) 6.0e-2 -> 1.9e-10, gates 12 / 13 /
  35 / 9 unchanged.  `gradient` (O2) stays OpenFOAM's `fv_gradient`.
- AMReX: the proxy topology carries the closure kinds; per-class sample
  lists, class per row in `gradient` / `aux_derivatives`.  Gate 27 vam
  1.5e-2 -> 1.5e-10 (cpu1 = cpu2), O2 unchanged; a degree-2 O2 probe
  6.7e-4 -> 9.8e-16.  Kernels.H now carries mesh-dependent class ids: a
  precompile without `classes` has no closure (degree 2 recompiles).
- Setup cost (section 41) was the gates, not the library: JAX op-by-op
  geometry (gate 39: 171 -> 30 s) and a re-derivation per call (gate 44:
  -> 87 s, quads by default; `tri` / `all` behind an argument).
- Conditioning (bisect of gate 1a's geometry AD test to 93b52fa): a
  one-deep wall ring of degree 2 tells y from y**2 only through jitter
  (relative singular value 1e-4 .. 1e-6); `_rows_ok` counted that as
  information (cut at 1e-7) and the pinv inverted it, leaking the
  near-null vector into the tangential rows: wall operators 30-400x an
  interior cell's (|A| h^deg).  Now directions below `_LSQ_COND` are null
  in the check (such a wall cell is extended) and cut in the bint / bone
  pinvs on radius-scaled offsets (`_pinv_cut`: pinv(V P V), the exact
  derivative of the cut map).  Jittered quads: wall max 14-116 -> 1-22
  (interior 0.3-0.45).  Gate 1a 30/30 (AD = FD again), 39 72/72, 41
  orders 2.0, 42 18/18, 44 OK, 43 PASS as far as run.

## 43. Test matrix rerun (all backends, serial / MPI / GPU): 327 PASS, 1 FAIL

From an empty store on the current heads (the old one kept as
`store_prev_20261012`).  Fixed on the way:
- foam C16 (preCICE foam-foam): `FoamDriver.build(nsm, dim)` without a
  topology again (precompile; the closure classes come with `run`).
- AMReX C17 off cpu1 (mpi2, gpu1): first measured; the runs share one work
  directory, so the cpu1 comparison finds its reference.  All pass.
- foam C10 n120: the Schur-GAMG preconditioner diverged at the first solve
  (FGMRES ended at a residual of 1e107; no preconditioner: = JAX 1.3e-12).
  A diverged GAMG cycle now falls back to Jacobi on the Schur diagonal:
  foam = JAX 1.4e-12, but ~260 GMRES iterations per solve (n60: ~50).
  Open: a Schur approximation / smoother that stays convergent.
- C12 `sub` on triangles refined in x only (ny 2) stretched the triangles:
  foam O2 lost its order (L1 3.0e-3 / 7.9e-4 / 1.6e-3; OpenFOAM's
  leastSquares at the walls).  Refined in x and y (ny 1/2/4) foam is
  second order (orders 2.0, 2.8; JAX 2.1, 2.4); the case now does that.
- foam-JAX "difference order" at round-off was noise: foam = JAX at the
  finest mesh (L1 <= 1e-11) passes (C4 trans O2: 2.5e-6 at n25, where
  OpenFOAM's cellLimited acts, then 6e-16 / 2e-13).
- AMReX = JAX tolerances from measured sensitivity: JAX with Q0 perturbed
  at 1e-14 (`*_pert` runs); tolerance 10x that.  C4 trans O2 3.7e-12 (JAX
  sensitivity 1.3e-11); C11 sub O2 4.5e-6 (JAX sensitivity 2.0e-5: that
  1-D steady run amplifies round-off 1e9-fold -- limiter switching; open
  whether that is benign).
Remaining FAIL: C15 Chorin AD on gpu1 (NaN; CPU and mpi2 pass).

## 44. C15 GPU NaN: the eig tangent at a defective eigenvalue; C11 sensitivity

- C15 Chorin jvp on gpu1 was NaN (J finite, = CPU) from t ~0.015 on.  Not
  GMRES (its outputs stayed finite).  `eigvals`' tangent at a defective
  dominant eigenvalue is ~1e15 (unit test: a 2x2 Jordan block, CPU and GPU);
  the CPU's exact tie lets max|lambda| average the +-1e15 pair to a finite
  value, the GPU splits the tie and propagates 1e15 into dt -> NaN.  Power
  iteration (the GPU default) is smooth: gpu1 finite.  C15 Chorin now runs
  `power` on every config: serial / mpi2 fd_rel 6.4e-9, rev = fwd.  Open:
  AD through `eig` wave speeds is ill-posed at defective points on any
  device (a cluster-mean tangent from Schur vectors would fix it).
- C11 `sub` O2 (1-D): the 1e-14 sensitivity is bounded, not growing:
  8e-8 at t 0.5, 1e-6 .. 2e-5 to t 6, error vs Bernoulli steady 5.6e-4 --
  limiter chattering in the near-steady state; the sensitivity tolerance
  (section 43) is the right check there.

## 45. Night 2026-10-12: power wave speeds, IMEX-SSP2, viscosity tests, two viscous defects

- **Wave speeds.** Profiling (AMReX TinyProfiler + perf, matched O2 bump 200^2): 76-87 % of the CPU
  time was the per-face numerical 4x4 eigensolve (SME(0) from the derivation has no closed-form
  spectrum).  The symbolic solvers now default to power iteration on every backend and device
  (`zoomy_core.symsolver.WAVE_SPEED`; foam / AMReX gained `spectral_radius`).  us/cell/step eig ->
  power -> closed form: AMReX 27.2 -> 6.7 -> 3.3, foam 28.9 -> 10.8 -> 7.9, JAX (8 cores) 10.0 ->
  6.0 -> 1.8.  Matrix rerun: 538 rows PASS, 0 FAIL.
- **IMEX-SSP2(2,2,2)** (Pareschi & Russo 2005) as `imex=Imex.ssp2`: SWE + Manning time order 2.01-2.08
  (Lie 1.06-1.22), foam / AMReX = JAX to 2e-14 (C31).  VAM(1): every scheme caps near 1 because the
  discrete projection is not idempotent (P(P(U)) != P(U), 2.5 / 0.74 / 0.23 % of the correction at
  nx 30 / 60 / 120); the predictor alone is order 2.02.  Why: an approximate projection by
  construction (last bullet).
- **Viscosity** (C32 shear decay between no-slip / free-slip walls, exact; C33 SME(2) vertical
  viscosity vs expm): pass.  But both see only the diagonal-index diffusion entries.  Gate 45
  (smooth periodic h, q_x, q_y varying in x and y; one step nu = 1 minus nu = 0) finds:
  1. **Numerics:** the discrete in-plane operator equals the SystemModel operator with every
     mixed-index entry `A[i, j, x, y]` dropped (second-order convergence to THAT: 4.95e-3 ->
     1.24e-3 (q_x), 7.72e-3 -> 1.97e-3 (q_y)); against the full SystemModel operator the error
     stays 16 % / 38 %.  The TPFA diffusion block has no tangential face gradient.  All backends
     (C26: foam / AMReX = JAX).  Fix: face gradient = mean of the cell LSQ gradients + normal
     correction, flux n.A.grad_f; mixed part explicit (deferred correction) in the implicit solve.
  2. **Derivation / extraction (open):** the SystemModel operator (diffusion + nu-part of the
     source, aux -> exact derivatives) matches neither div(nu h (grad u + grad u^T)) (5.6 % / 11 %)
     nor h div(tau) (2.8 % / 7.5 %); `from_model` warns that the in-plane terms of both momentum
     rows are "classified into NO slot".  To be checked against the paper equations before any fix.
- **No-slip wall:** a `Dirichlet` momentum wall also zeroes the Riemann outer state, a small
  numerical wall friction (C32 `inviscid`: 1.7e-4 at g = 0.01, 2.5e-3 at g = 9.81).
- **Fix of 1 (2026-10-13, branch dev/tpfa).**  The `diffusion` block of every runtime (JAX
  `runtime.py`, foam `rt.H`, AMReX `rt_amrex.H`) now applies the full face flux `n.A.grad_f`:
  `grad_f` = the mean of the two cells' LSQ gradients (the derivative-aux fit, boundary ghosts as
  samples, no closure) plus `((Q_b - Q_a)/|d| - mean.e) e`, `e` the unit cell-to-cell vector; on
  boundary faces and periodic seams `e = n`, i.e. the two-point normal part as before and the
  tangential part from the cell gradients.  Written as two-point part + correction; the
  correction is zero on a Cartesian face when A has no mixed entries (exactly: `n` and `e` are
  axis vectors), and in 1-D it is not computed.  `implicit_diffusion` keeps its Crank-Nicolson
  Newton solve on the two-point part only and adds `dt` times the correction at the start state
  (deferred correction; Eymard, Gallouet, Herbin 2000).  On non-orthogonal faces (triangles) the
  correction also carries the non-orthogonal part `(n.A.e - n.A.n / max(|n.e|, 0.1)) (Q_b - Q_a)/|d|`.
  Gate 45 (now against the full SystemModel operator, nx 16 / 32 / 64, max rel): q_x 3.05e-2 /
  7.69e-3 / 1.94e-3 (orders 1.99 / 1.98), q_y 4.62e-2 / 1.21e-2 / 3.07e-3 (1.93 / 1.98) on JAX,
  foam and AMReX alike (L vs JAX's: foam 6.9e-8 / 1.1e-8 / 1.6e-9, its own O2
  stencils in the explicit stage, an O(dt) effect in L; AMReX 6e-10, round-off / dt).  Matrix: C34 added (gate 45).  Where the correction is zero the
  results are unchanged (scratch rerun vs the 2026-10-12 store): C33 and C28 bit-identical on all
  backends (C28 triangles to round-off: gate 41's triangles are orthogonal), C32 bit-identical on
  AMReX and within 1-2 ulp on JAX / foam (their LSQ x-derivative of the x-uniform field is
  round-off, A has mixed entries).  C26 changes by 2.3e-4 (JAX, AMReX; foam 2.5e-4), 2e-4 of
  max |Q|; AMReX = JAX 1.1e-13, foam vs JAX unchanged (L1 1.72e-3).  Open: the cell gradients
  take the boundary ghosts as samples, not the LSQ boundary closure (ANALYSIS 40), on every
  backend; a distributed run's halo gradients are exchanged but no mpiN config was rerun.
- **VAM projection: approximate by construction (2026-10-13, dev/vam-projection; numerics
  unchanged).**  The splitter builds the pressure rows as the constraint with the corrector
  substituted symbolically and product-rule expanded (`split_for_pressure_structural`); the
  template discretizes that PDE term by term, with LSQ second derivatives (`P_xx`, `h_xx`, `b_xx`)
  and products of first derivatives, while correction then constraint apply the first-derivative
  LSQ twice, D(h D P).  Aux (recomputed after the correction), BC ghosts, the 1/h row scaling and
  the frozen h, b are the same on both sides, and the constraint is affine in P (one linear solve;
  Newton agrees): the one difference is the elliptic operator, a compact L_a instead of the
  composition D∘G, i.e. an approximate projection (Almgren, Bell, Szymczak 1996,
  https://doi.org/10.1137/S1064827593244213), not idempotent.  Measured on C31's wave (first
  projection, max over the corrected rows, nx 30 / 60 / 120): constraint residual of P(U) / of U
  7.4e-2 / 2.8e-2 / 1.4e-2, |P(P(U)) - P(U)| / |P(U) - U| 5.5e-2 / 1.9e-2 / 5.1e-3.  The time
  error it leaves does not shrink with the mesh: IMEX-SSP2 L1(h) dt/4 -> dt/8 differences
  1.53e-7 / 1.55e-7 / 1.19e-7 (orders 0.99 / 0.91 / 1.00), as D∘G and L_a differ by O(1) on
  grid-scale modes.
- **Exact projection, prototype** (`prototype/exact_projection.patch`, not merged): R(P) = the
  constraint (`press` at dt = 0) of `correct(corr, Q(P))`, the corrected rows' derivative aux
  recomputed.  Idempotent to 2e-15 / 2e-14 / 8e-14 (constraint residual 2e-15 / 1e-14 / 2e-13 of
  U's).  C31 JAX: ssp2 1.98 / 1.95 (was 1.15 / 0.91); t2 1.38 / 0.82 with the Lie Manning source
  (n 0: 1.95 / 1.88; was 0.81 / 0.67); t1 1.04 / 1.02.  C10 1.91 / 2.00 / 2.01 (JAX = foam); C8 on
  quads: foam vs JAX L1 8e-10 (was 1.6e-2: first derivatives only, no second-derivative stencils),
  AMReX 1.5e-9 (threshold 1e-9).  Not adopted: on C8's triangles the composed operator is
  indefinite and ill-conditioned (200 of 1440 eigenvalues with Re < 0, cond 1.2e6 against 956 for
  L_a; quads 887 against 514) and
  unpreconditioned GMRES fails (C8 `vam_tri`: JAX relative residual 3.4e-3, foam 2.2e-5), the
  spurious modes of a collocated D∘G that approximate projections avoid (Almgren, Bell,
  Crutchfield 2000, https://doi.org/10.1137/S1064827599357024).
- **What a second-order projection needs:** the exact composition with L_a (the current operator)
  as its preconditioner, i.e. a preconditioned `LinSolve` block on every backend; and the runtime
  items the prototype carries: a right-hand side at round-off (a stage already projected: IMEX-SSP2
  with a source below the Newton tolerance) breaks JAX's GMRES, which takes |b| < eps for zero
  (solve on b / |b|), and the C++ matvec R(x) - R(0) would keep no digits for a tiny x (evaluate
  |x| op(x / |x|)); the AMG assembly pattern becomes the stencil's stencil (JAX: > 63 colours) and
  AMReX's block probing needs cells 2R + 1 apart.  On the time accuracy of projection schemes and
  the pressure-increment form: Brown, Cortez, Minion 2001, https://doi.org/10.1006/jcph.2001.6715;
  Guermond, Minev, Shen 2006, https://doi.org/10.1016/j.cma.2005.10.010 (Zotero J4IJXP8A, no PDF;
  the Almgren papers are not in the library).

### 45b. In-plane viscosity: two derivation defects fixed; one physics choice open (2026-10-13)

- `equations.py:package_viscous` emitted `-Derivative(C, x_d) * d_e Q` unevaluated; for a
  state-dependent `C = nu q / h` `from_model` turned it into a compound aux (`nu*q_x_0/h_x`,
  `target_kind='unknown'`) that every backend read as 0, silently dropping -d_d(nu q/h) d_e h
  (also in m07 / m14).  Now `sp.diff(C, x_d)`.
- `system_extract.py:_classification_remainder` added +d_d(A d_e Q) although A carries the RHS
  sign: the "classified into NO slot" warning was a false positive.  Now `-=`.
- After both: the SystemModel operator of SME(0) and SME(1) + NewtonianInPlane equals the model's
  own depth integration symbolically (gate 45: 4e-16 vs h div(tau)); discrete -> SM order 2 on all
  backends; C26 moved by 5e-5.
- **Open (physics, the user's call):** (A) the model keeps the Leibniz in-plane boundary terms ->
  h div(nu (grad u + grad u^T)), not momentum-conservative on periodic domains; (B) the literature
  form div(nu h (grad u + grad u^T)) (Gerbeau-Perthame RR-4084 Lemma 2.1 eq. (9); Marche 2007
  eq. (5.19)) needs the in-plane stress in the surface / bed traction (`apply_stress_closures`,
  `_solve_traction` build sigma.n from tau_iz only) -- affects SME, VAM, ML-SME, ML-VAM.

### 45c. In-plane viscosity: the sigma-coordinate question settled (debugger, 2026-10-13)

- **Reference** (gate 46): a hand-written viscous SWE SystemModel (state h, q_x, q_y) with the
  literature term div(nu h (grad u + grad u^T)) in conservative form, straight into diffusion_matrix.
  JAX: discrete -> exact at order 1.99 / 1.98 (nx 16 / 32 / 64: 1.7e-3 / 3.1e-3 at 64).  The
  numerics are right independently of any derivation.
- **Derivation walked with pdb** (SME(0) + NewtonianInPlane, small_slope):
  1. `closures._solve_traction` at the surface: tau.n = [tau_xz, tau_yz, -tau_xz eta_x - tau_yz eta_y]
     (frame slopes) -- only the vertical shears; tau_xx, tau_xy, tau_yy never enter the traction;
     StressFree gives tau_xz(eta) = 0.
  2. After `small_slope_scaling` the frame slopes are 0; the sigma-mapped x-momentum carries the
     metric terms nu (d u~/d y) (d h/d y), nu (d u~/d x)(d b/d x), ... (in-plane stress x slopes).
  3. Input of `package_viscous` (projected, SME(0)): exactly
     -nu h d_yy(q_x/h) - nu h d_xy(q_y/h) - 2 nu h d_xx(q_x/h), i.e. RHS = h div(tau): the metric
     terms of the sigma map cancel in the projection.
- **Conclusion.** The derivation is exact for what it encodes: h div(tau) = div(h tau) - tau(eta).grad(eta)
  + tau(b).grad(b).  The literature form div(h tau) (Gerbeau-Perthame, Marche; momentum-conserving)
  needs the stress-free surface and the bed traction WITH the in-plane stress: tau_xz(eta) =
  tau_xe(eta) d_e eta (and the bed analogue), whose d_z tau_xz integral adds back exactly
  +tau(eta).grad(eta) - tau(b).grad(b).  The model keeps tau_de in the bulk but drops it from the
  traction (and small_slope zeroes the frame slope there while the sigma map keeps the physical
  slope): an inconsistent combination, not a sigma-coordinate bug.
- **Fix for (B):** with an in-plane closure, build tau.n from the full stress (tau_de and tau_dz) and
  resolve the slopes of the in-plane part to the PHYSICAL interface slopes.  Acceptance: SME(0) +
  NewtonianInPlane equals gate 46's hand-written operator symbolically.

### 45d. VAM second order: why Escalante et al. 2024 report it, what we need (2026-10-13)

Their pressure step is the same approximate projection (corrector substituted into the constraint,
compact (P_{i+1} - 2P_i + P_{i-1})/dx^2, eq. 16; 1-D collocated FD); their order tables refine space
and time together at fixed CFL, where the projection defect O(dt s/(1-s)), s ~ (k dx)^2, is O(dx^3)
and invisible -- our C10 is that test and gives 2.01 on all backends.  C31 refines dt on a fixed mesh
and sees the O(dt) floor, largest at the closed wall.  Recommended: the pressure-increment (lagged)
projection, Almgren, Bell, Crutchfield 2000 "version (1)": only `project()` changes (correct with
P^n, solve press for dP, store P^n + dP); a Fourier model keeps 2.00 at every wavenumber.  First:
the spectrum of I - (D.G) L_a^{-1} on C8 triangles and C31 away from walls.  Details:
`research/vam_second_order.md`, `research/toy/`.

### 45e. In-plane viscosity: the in-plane stress in the bed / surface traction (2026-10-13, dev/inplane-traction)

- **Change (derivation only).** `closures.py:apply_stress_closures` -> `_inplane_traction`: with an
  in-plane closure the bed and surface traction is built from the full deviatoric stress,
  (tau n)_d = (tau_dz - sum_e tau_de d_e I) n_z, with tau_de = `NewtonianInPlane.stress` (the
  sigma-mapped rho nu (d_d u_e + d_e u_d) at the trace, `ClosureState.dx_phys`) and d_e I the
  PHYSICAL interface slope (`ClosureState.interface_slopes`, d_e(b + zeta h) at zeta = 0, 1), the
  slope of the sigma map's Leibniz terms.  `_solve_traction` carries sum_e tau_de d_e I through
  the frame solve as an opaque g_d; the frame slopes of the shear part still go to 0 under
  small_slope, so tau_dz(eta) = sum_e tau_de(eta) d_e eta and tau_dz(b) = P_d + sum_e tau_de(b) d_e b.
  SME and VAM pass `base=b` to their ClosureState.  Without an in-plane closure the code path is
  unchanged.
- **pdb** (SME(0), 2 horizontal): before the closure the projected rows carry the Leibniz terms
  (nu c(1) phi(1) d_y u~(1) (b_y + h_y), ...); after closure + Simplify (frame slopes 0) no nu term
  outside an integral is left in either row.
- **Symbolic** (diffusion + nu-part of the source, aux = exact derivatives, generic smooth h, q, b
  of (x, y)).  SME(0): SM - div(h tau) = 0; SM - gate 46 = 0, flat and varying bed (gate 45 prints
  it).  SME(1): every momentum row equals the independent projection
  (2k+1)[d_e(h <phi_k tau~_de>) + <phi_k' (b_e + zeta h_e) tau~_de>] + the vertical shear.  The mean
  rows are div(h taubar), taubar = nu (grad u0 + grad u0^T) - (2 nu / h)(u1 (x) grad z_m +
  grad z_m (x) u1), z_m = b + h/2; periodic integral -6e-17 (max |op| 15): momentum-conservative.
  The q_1 rows keep the non-divergence <phi_1' ...> part (a moment, not a conserved quantity).
  One horizontal: SME(1) and VAM(1) the same (0; periodic integral 2e-16).
- **Gate 45** (JAX, discrete vs the exact div(h tau), nx 16 / 32 / 64): q_x 2.71e-2 / 6.83e-3 /
  1.71e-3 (orders 1.99 / 1.99), q_y 4.66e-2 / 1.21e-2 / 3.08e-3 (1.94 / 1.98).  The old operator
  h div(tau) differs from div(h tau) by 2.8 % / 5.9 % of max |op| on this field.
- **Goldens.** Only m07 / m14 change, only in `source`: q_0 loses its nu terms (the row is now the
  pure diffusion divergence), q_1 gets the phi_1' terms with the physical slopes.  The 4 pre-existing
  failures (m05, n02, n03, n02 canary) are unchanged.  The rederive test
  `test_inplane_h_column_matches_G_matrix_first_principles` fails the same before and after
  (A[q_k <- h] = +V, the test expects -V; A is unchanged here).
- **Matrix** (serial, against the pre-change store): C32 and C33 bit-identical (JAX); C26 moves by
  5.7e-5 (JAX, AMReX; foam 6.0e-5), 5e-5 of max |Q|; AMReX = JAX 1.2e-15, foam vs JAX L1 1.72e-3 as
  before.  C34 now checks against the exact div(h tau); PASS.
- **Open.** (1) ML-SME / ML-VAM (`apply_layer_stress_closures`) are unchanged and still give
  h_l div(tau_l) per layer.  Their internal interfaces need a single-valued interface traction
  tau_dz - tau_de d_e z_l (which layer's tau_de, or the mean) before the bed / surface can take
  `_inplane_traction` (the layer derivation also uses its own `nu` symbol).  (2) Gate 46 is the
  first literature term only.  Marche 2007 eq. (5.19) has 2 mu div(h D(u)) + 2 mu grad(h div u),
  Gerbeau, Perthame 2001 (https://doi.org/10.3934/dcdsb.2001.1.89) eq. (1) 4 mu d_x(h d_x u) in
  1-D (Zotero attachments CV8TMDFI, XMQXMC6E).  The grad(h div u) part is the viscous normal stress
  tau_zz = -2 rho nu div u in the pressure, which the hydrostatic elimination drops.  Keeping it is
  a model choice.  (3) small_slope=False with an in-plane closure derives (SME(0), one horizontal),
  but it has not been run.  Its traces carry the 1/(1 - |grad I|^2) factor of the frame solve, whose
  normal row has no tau_zz.

### 45f. In-plane viscosity: the viscous normal stress tau_zz; multilayer interface traction (2026-10-13, dev/inplane-tauzz)

- **Where tau_zz was dropped.** `Momentum`'s vertical row carries no stress at all (so
  `moment_scaling`, which keeps every tau key containing 'z', never sees a `tau_zz`); the SME
  hydrostatic step integrates d_z p = -rho g and sets p(eta) = 0.  With an in-plane closure
  tau_zz = 2 rho nu d_z w = -2 rho nu div_h u is of the same order as tau_de and was missing.
- **Literature.** Marche 2007 (https://doi.org/10.1016/j.euromechflu.2006.04.007, Zotero
  CV8TMDFI), eq. (5.19): "2 mu div(h D(u)) + 2 mu grad(h div u)"; its pressure (5.17)
  "p = p_o + (xi - z)/F_r - (2/R_e) d_x ubar - (2/R_e) d_y vbar - eps beta Delta xi + O(eps^2)",
  surface condition (3.10) "(p - p_o - (2/R_e) d_z w + beta kappa)|_{z=xi} = O(eps^2)", bed (5.18)
  with "-grad d (p + (2/R_e) div U)|_{z=d}".  Gerbeau, Perthame 2001 (Zotero XMQXMC6E), eq. (1):
  "4 mu d_x(h d_x u)".
- **Change (derivation).** `equations.inplane_normal_stress` gives tau_zz; `NewtonianInPlane`
  gained `normal_stress(s)` (its trace).  SME: the vertical row gets -d_z tau_zz / rho and the
  surface p(eta) = tau_zz(eta) (normal stress-free), so p = rho g (eta - z) + tau_zz(z).  pdb
  (SME(0), 2 horizontal): after `IntegrateZ` the surface Subs terms cancel against the BC, psol =
  rho (g (b + h - z) - 2 nu (u_x + v_y)), and the x-row nu terms are -4 d_x(nu u_x) - 2 d_x(nu v_y)
  - d_y(nu u_y) - d_y(nu v_x), i.e. tau_xx - tau_zz = 2 nu u_x + 2 nu div u.  VAM / ML-SME / ML-VAM:
  the pressure unknown is p - tau_zz = -sigma_zz (zero at the surface, continuous across layer
  interfaces), the vertical row is unchanged and the horizontal rows get +d_d tau_zz / rho
  (`add_inplane_viscous(normal_stress=True)`).  Traction: `_inplane_traction` uses
  tau_de - delta_de tau_zz (`inplane_stress`): at the surface the term is p(eta) d_d eta, at the bed
  the normal stress the tangential projection removes (Marche (5.18)).
- **Symbolic** (diffusion + nu-parts of source / NCP / flux, aux = exact derivatives, generic
  h, q, b).  SME(0): SM - div(h tau') = 0, SM - gate 46 = 0 on flat and varying bed,
  tau' = nu (grad u + grad u^T) + 2 nu (div u) I.  SME(1), SME(2) (1 horizontal), SME(1), VAM(1)
  (1 and 2 horizontals): every momentum row equals the independent projection
  (2k+1)[d_e(h <phi_k s'_de>) + <phi_k' (b_e + zeta h_e) s'_de> - <phi_k' nu d_zeta u_d / h>],
  s' = tau - tau_zz I with the physical derivatives; no unclassified terms.  Gate 46 now carries
  the 2 nu grad(h div u) term (A[q_x <- q_x, x, x] = 4 nu, A[q_x <- q_y, x, y] = 2 nu, ...).
- **Open, the user's call (level >= 1 only).** Marche eq. (3.5) keeps the shear in the vertical
  balance, "d_z p = -1/F_r^2 + (1/R_e)(d_z(div_xy U) + 2 d_z^2 w)", hence (3.11)
  "p - p_o + beta kappa = (xi - z)/F_r^2 - (1/R_e)(d_x u(z) + d_x u(xi) + d_y v(z) + d_y v(xi))";
  Gerbeau-Perthame Lemma 3.1 the same, "p = G(h - z) - nu d_x u(z) - nu d_x u(h)", and the shear
  "nu/eps^2 d_z u + nu d_x w".  For a z-independent velocity (level 0, the acceptance) both
  reduce to -2 nu div u; for level >= 1 the d_e tau_ez term and the nu d_d w part of the shear are
  of the same order as tau_zz but not implemented (here: tau_zz only), and in the multilayer
  models d_e tau_ez couples the layers non-locally (div u(eta) of the top layer in every layer).
- **Multilayer interface traction.** Each layer's own trace now carries the in-plane traction at
  every interface (`apply_layer_stress_closures`: bed / surface through `_solve_traction`,
  internal interfaces tau_l(trace) += g_own), so the layer's in-plane part is
  d_e(h_l <s'_l>).  The assembly adds the shared part -/+ sum_e s*_de d_e z_a at the top / bottom
  of the two layers at internal interface a (`interface_inplane_traction`, projected
  phi_k(side) / mu_k like the u* transfer), s* = `InterfaceFlux(quantity=InterfaceQuantity.stress)` of the two
  traces: `MeanInterface` by default, `UpwindInterface(quantity=InterfaceQuantity.stress)` by the sign of G;
  `quantity=InterfaceQuantity.velocity` (default) keeps the transfer-velocity meaning.  The interface traction
  is then single-valued in its in-plane part and the layers telescope.  ML-VAM + NewtonianInPlane
  never derived before (mode 0: "cannot route bare second-derivative term"); mode 0 now goes
  through `package_viscous`.
- **ML checks** (2 layers, level 0, l_1 symbolic): identical states q_l = l_l q: sum_l op_l -
  div(h tau') = 0 (ML-SME and ML-VAM, 1 and 2 horizontals, mean and upwind); per layer (mean)
  op_1 = d_e(h_1 s'_1) - s*.grad z_1, op_2 = d_e(h_2 s'_2) + s*.grad z_1 exactly; different layer
  states, smooth periodic fields, l_1 = 0.4: periodic integral of sum_l op_l 3e-15 (max |op|
  92-120); 1 and 3 layers (1-D): sum = 4 nu d_x(h u_x) (Gerbeau-Perthame).  Level 1 (ML-SME 1 / 2
  horizontals, mean / upwind, ML-VAM 1 horizontal) derives with 0 unclassified terms.
- **Goldens.** Only m07 / m14 change (1 horizontal: every in-plane coefficient doubles, 2 nu ->
  4 nu; the vertical shear terms are unchanged); the 4 pre-existing failures (m05, n02, n03, n02
  canary) are unchanged (identical diffs).  The rederive test
  `test_inplane_h_column_matches_G_matrix_first_principles` now expects A[q_k <- h] = V with
  4 nu (it asserted A = -V with 2 nu, a sign error noted in 45e) and passes; reference tier 13 / 13.
- **Discrete** (JAX, nx 16 / 32 / 64, max rel): gate 46 (hand-written) and C34 (gate 45, SME(0) +
  NewtonianInPlane) vs the exact div(h tau') give the same numbers, q_x 3.55e-2 / 8.99e-3 / 2.26e-3
  (orders 1.98 / 1.99), q_y 5.83e-2 / 1.52e-2 / 3.85e-3 (1.94 / 1.98).
- **Matrix** (serial, scratch stores before / after, the same JAX / foam / AMReX checkouts): C26
  moves by 1.4e-3 (1.3e-3 of max |Q|; JAX, NumPy, AMReX; foam 1.5e-3), AMReX = JAX 7e-16, NumPy =
  JAX 2e-14, foam vs JAX max rel 3.8e-3 (before 3.9e-3); C32 within 1-2 ulp (h = 1, div u = 0; the
  new / doubled A entries act on q_y = 0 or on round-off x-derivatives), `inviscid` bit-identical; C33
  bit-identical; C34 PASS against the new exact operator.  C26 / C32 / C33 / C34: 28 PASS, 0 FAIL.
- **Open.** (1) The level >= 1 / multilayer d_e tau_ez term above.  (2) `interpolate_rows` slot 5
  is still rho g h (1 - zeta) (+ the VAM modes), i.e. -sigma_zz, not p = -sigma_zz + tau_zz.  (3)
  small_slope=False with an in-plane closure is not run.  (4) The ML interface shear (the bulk
  closure's one-sided trace nu d_z u_l) is not single-valued; at level 0 it is 0.

### 45g. VAM second order on a fixed mesh: the pressure-increment projection (2026-10-13)

- **Diagnostics.**  (a) C31 with the walls 3 m from the wave (domain [-4.5, 4.5], nx 180, same dx;
  Manning off: on that domain the state stays exactly at rest far out and the JAX Manning Newton
  returns NaN there, a separate defect), standard projection: t2 1.45 / 1.27, ssp2 1.68 / 1.51
  (0.81 / 0.67 and 1.15 / 0.91 with the wall).  The wall closure is most of the defect, the interior
  approximate projection the rest.  (b) Spectrum of S = I - (D.G) L_a^-1 (dense, the derived
  operator L_a and the prototype's composition D.G): C8 quads rho(S) 0.983 (every eigenvalue lambda
  of L_a^-1 D.G in [0.017, 1.11], Re > 0), C31's mesh 0.986; C8 triangles 1.105, 158 of 1440
  eigenvalues with |1 - lambda| >= 1 (156 with Re lambda < 0).  The lagged iteration has per mode
  mu^2 - 2 (1 - lambda) mu + (1 - lambda) = 0: max |mu| 0.9986 on quads, 1.446 on triangles
  (368 modes >= 1; the standard projection: |1 - lambda| <= 1.105).  C8 `vam_tri` run to t 0.5
  (20 steps) stays bounded with either projection (the predictor's dissipation); no long-run test.
- **Implementation** (`chorin.py`, `Projection.increment`, OPT-IN (default `standard`); `projection=Projection.standard`
  keeps the old scheme bit-identically).  `project(Q, ..., lag)`: correct with the lag P^n, solve
  the existing `press` system on that state for dP (Newton from dP = 0), correct Q with P^n + dP.
  - SSP-RK stages: the lag is the stage input's P rows, not the predicted ones (the predictor's
    Rusanov flux diffuses the P rows: 15 % in one step at the Dirichlet-P cell).
  - IMEX-SSP2: the pressure is the tableau's implicit term, so the stages do not take the same
    pressure.  For DG P^n = D f, stage i removes sum_{j<i} (At_ij - A_ij c_j / A_jj) D f, hence lag
    c_i P^n with c = (0, 1) and 1 - 1/(2 g) = -0.707 for the final projection; the stored pressure is
    the step's total sum_i w_i P_i / A_ii + P_final.  The first try (lag A_ii P^n, final unlagged)
    left an O(dt) floor: orders 1.83 / 1.68 / 1.48 / 1.28 down to dt / 32.  A source-driven
    pressure part is not lagged (first order in that part; Manning on / off give the same orders).
  - foam `linsolve` evaluates op(x) as |x| op(x / |x|) (the prototype's item): the increment's small
    right-hand side made C8 `vam_newton` stop at relative residual 1.2e-5 > 1e-6.  `linsolve` is
    only the Chorin pressure solve (affine there).
- **Results** (serial, scratch store; JAX, foam, AMReX).  C31 VAM ssp2 1.97 / 1.96 (was 1.15 / 0.91;
  dt / 32: 1.90), t2 1.35 / 1.12 (the Lie source per stage caps it at 1, as with the exact
  projection; Manning off: 1.75 / 1.83), t1 1.03; AMReX `vam_ssp2_k0` vs JAX 3.0e-13.  C10 1.91 /
  2.00 / 2.01 on all three (unchanged: spatial error).  C8 vs JAX: AMReX 7.8e-11 / 7.8e-11 / 3.0e-11,
  foam L1 2.3e-10 / 2.2e-10 / 4.4e-11 / 1.9e-10 (`vam_tri`); Newton vs linear 0 (JAX, AMReX), 9.7e-11
  (foam).  C8 / C10 / C31: 29 rows PASS, 0 FAIL.  C31 thresholds: ssp2 >= 1.8 (theory 2), t2 >= 0.8.
- **Open:** triangles (rho(S) > 1: a long run on triangles may grow under either projection, faster
  with the increment); a time-dependent Dirichlet pressure needs P_b^{n+1} - P^n as the increment's
  boundary value (all cases here: P = 0); mpiN / GPU configs not rerun.
- **Default (2026-10-13): `standard`.**  On C8's triangles the lagged iteration has growing modes
  (|mu| up to 1.45 on 368 modes); a run to t = 0.4 took > 590 s with `increment` vs 254 s with
  `standard`.  `increment` is therefore opt-in (C31 uses it: VAM ssp2 1.96); triangle meshes need
  a stabilized variant (e.g. Almgren et al.'s version (5)) before it can be the default.

### 45h. The full Newtonian stress: `stress = Stress.shallow | Stress.full` (2026-10-13, dev/inplane-tauez)

- **Design (the user's).**  `Newtonian` is the incompressible Newtonian fluid,
  tau = rho nu (grad u + grad u^T), every component: `tensor` in physical (fixed-z) coordinates
  before the sigma map, `stress` / `normal_stress` / `shear` at the traces (`expression` stays the
  vertical shear rho nu d_z u).  The truncation is the shallow operator `moment_scaling`, set by the
  typed model setting `stress` (`equations.Stress`, an `Option`; on SME, VAM, MLSME, MLVAM, default
  `Stress.shallow`), one rule per component: `shallow` drops tau_de, tau_zz and the stress of the
  vertical row (the blueprints carry each as its own additive term) and, read off `m.stress` by
  the closure step, the rho nu d_d w part of the bulk tau_dz; `full` keeps the tensor.
  `NewtonianInPlane`, `ShallowInPlane` and the `in_plane` slot are gone (users: gates 31, 45,
  matrix C26 / C32 / C34, the reference tests, m07 / m14, the book; Sigma3D: `stress=None` keeps
  tau_xx open, `Stress.shallow` drops it).  The other bulk closures close only the vertical shear:
  `Stress.full` raises (`full_tensor_closures`).  ML-VAM-WD (experimental) is shallow only.
- **Blueprints.**  `Momentum` and `MomentumNonHydrostatic` declare the full tensor in every row,
  the vertical one included (d_e tau_ez + d_z tau_zz; the non-hydrostatic pressure unknown is
  -sigma_zz minus the hydrostatic part, so tau_zz sits in its horizontal rows, 45f); ML-SME's
  hand-written layer rows take the same terms through a stand-in blueprint.  `Stress.full`:
  `apply_full_tensor` fills tau_de, tau_zz and the vertical row's tau_ez before the sigma map; the
  horizontal shear tau_dz stays a closure variable for the traction (45e), its bulk
  rho nu (d_z u_d + d_d w) (w: the kinematic reconstruction in SME / ML-SME, the state in VAM /
  ML-VAM).
- **One term not kept.**  The vertical row drops rho nu d_e d_e w, the eps^2 term of Marche 2007
  eq. (3.2) that (3.5) drops (https://doi.org/10.1016/j.euromechflu.2006.04.007, Zotero CV8TMDFI):
  it is eps^2 below every other term of that row (VAM's vertical inertia included), and in VAM the
  bed-kinematic top w mode puts d^3 b into the w rows, which the extractor refuses (VAM(1) with it:
  "diffusion-compound piece -12 nu q_1 b_x^2 / h ... has no D d_x Q reading", row 5).  So the
  vertical row is (3.5)'s d_z p = -rho g + d_e(rho nu d_z u_e) + d_z tau_zz in every model.
- **Where the order-nu terms land.**  SME: the vertical row integrated from z to eta with
  p(eta) = tau_zz(eta) gives p = rho g (eta - z) - rho nu (div u(z) + div u(eta)), Marche (3.11).
  VAM: d_e tau_ez in the w rows (prognostic, package_viscous now also on them).  ML-SME: a per-layer
  vertical row for the viscous pressure Pi, integrated to the layer top Pi_top, which the assembly
  sets to the integral over the layers above (Pi = 0 at the surface, continuous at the interfaces).
  ML internal interfaces: each layer's trace is its own in-plane part; the assembly adds the shared
  T* = tau*_dz - s*_de d_e z_a (`interface_traction`, `InterfaceFlux(quantity=
  InterfaceQuantity.stress)`, mean by default, upwind): the whole interface traction is
  single-valued with `Stress.full` (closes 45f open item (4) there).
- **Defects found and fixed on the way** (each diagnosed with pdb on the derivation).
  (1) `operations._resolve_subs_safe` unwrapped `Subs(d_x u(x, z), z, b + h)` into
  `d_x u(x, b + h)`, which sympy reads as the TOTAL x-derivative: div u(eta) of (3.11) became
  d_x[u(eta)] (SME(1) q_0 row: h^2 q_0'' coefficient 3 nu instead of 4 nu).  45f's surface terms
  cancelled, so it was invisible.  Guard: no unwrap when the point depends on a differentiation
  variable.  (2) `PDETransformation` had no rule for `Subs(f(z), z, Z)` (a vertical field at an
  interface height): it now maps f and evaluates at zeta = (Z - b)/h.  (3)
  `InvertMassMatrix._pull` pulled a zeta-factor out of d_x d_zeta(zeta^2 q) (bracket integrals the
  w-row projection leaves): VAM(2) lost -6 nu d_x(q_2 / h) in the r_1 row.  Factors in a
  derivative's variables now stay inside.  (4) ML-SME: d_x d_t h of the interface kinematics in the
  w part, now from the global mass.
- **Checks.**  (1) `Stress.shallow` = main, byte for byte: golden-style renders of 23 models
  (SME 0-3, SME(1) 2-D, Manning, SWE 1-D / 2-D, VAM 0-2, VAM(1) 2-D, ML-SME 0 / 1 / mean / 2-D,
  ML-VAM 0 / 1 / upwind) derived cold, and all 26 goldens rendered cold equal main's renders
  except m07 / m14 (the `Stress.full` goldens, re-blessed: their nu = 0 operators equal main's,
  the nu terms are the new ones).  (2) `Stress.full` at level 0: SME(0) 1-D / 2-D (gate 45 / 46 /
  C26 / C32 / C34's model) and VAM(0) 1-D / 2-D render byte-identical to main's
  `NewtonianInPlane`: gate 45 symbolic SM - div(h tau') = SM - gate 46 = 0 (flat and varying bed);
  gate 46 q_x 3.55e-2 / 8.99e-3 / 2.26e-3, q_y 5.83e-2 / 1.52e-2 / 3.85e-3 (orders 1.99 / 1.98);
  C34 JAX and NumPy the same (NumPy = JAX 1.12e-11), as in 45f.  (3) Gate 47 (new; generic smooth
  h, q, r, b; aux = exact derivatives): SME(1), SME(2), SME(1) 2-D, VAM(1), VAM(2), VAM(1) 2-D,
  every momentum and w row minus the independent projection = 0, 0 unclassified terms.  Gate 48
  (new): ML-SME and ML-VAM, 2 layers, level 0 and 1 (1-D; level 0 also 2-D, ML-SME level 1 2-D):
  every row = the per-layer projection with the mean interface traction and Pi; with identical
  layer states (the single-layer profile restricted) the layer sum of the mode-0 rows = the
  single-layer SME / VAM row, also with `UpwindInterface(quantity=stress)`; 0 unclassified.  The
  rederive test of A[q_k <- h] now reads the d_xx h coefficient of an independent projection
  (SME(2), passes; reference tier 13 / 13).  (4) Core suite: 350 passed, the 3 failures of main
  (n02_vam_nsm, n03_sme_gate_guard, n02_time_canary), per directory the same counts as main.
  (5) Matrix (serial, JAX and NumPy, scratch store): C26, C32, C33, C34 all PASS with values
  equal to the committed table (73 rows identical); C36 (new: SME(1) `Stress.full`, periodic
  16^2, IMEX) runs on both, mass change 0, NumPy = JAX 1.4e-13.
- **Found, not fixed (pre-existing, shallow too).**  ML-VAM's w rows register the aux d_x(l_1 h)
  with target 'h*l_1' and `target_kind='unknown'` (main's m15-type models, non-viscous terms),
  the compound-aux pattern 45b found; backends likely read it as 0.  (Fixed: 45i.)
- **Open.**  ML-VAM w rows: the vertical balance is pointwise per layer, so its interface traces
  of tau_ez are one-sided (single-valued for continuous profiles only); a single-valued vertical
  traction there is a model choice (45j).  ML with `Stress.shallow` keeps the one-sided interface shear
  (45f item (4)), unchanged by spec.  `interpolate_rows` slot 5 is still the hydrostatic part
  (45f item (2)).  small_slope=False with the full stress is not run.

### 45i. ML-VAM: the compound aux d_x(l_1 h) of the w rows (2026-10-14, dev/mlvam-aux)

- **Proof.**  Every ML-VAM tried (2 layers level 0, 1 (mean, upwind), 2, 1-D; level 0 2-D; 3
  layers; `Stress.full`) registered `h*l_1_x` (3 layers also `h*l_2_x`, 2-D also `h*l_1_y`):
  atom d_x(l_1 h), `target_kind='unknown'`, used only in the source of the w rows r_1_0, r_2_0
  (products like q_1_0 q_1_0_x d_x(l_1 h) / h).  The Chorin predictor had a second one, `0_x`
  (B d_x P with P = 0 left as an unevaluated d_x(0)).  On a smooth state (2 layers, level 1,
  l_1 = 0.4, 100 cells) JAX `initial_aux` filled h*l_1_x = 0 (exact: l_1 h_x, up to 0.108).
  NumPy raised `KeyError: 'state_index'` in `derivative_aux` (its ghost path reads the state
  index before the target kind), so ML-VAM did not run on NumPy at all.
- **Cause.**  `derive_layer` keeps d_x h_l of the interface heights unevaluated (w at the layer
  bottom, the omega-tilde resolution); the assembly's `.subs(frac)` (h_l = l_l h) made d_x h_1 the
  compound d_x(l_1 h).  The horizontal and constraint rows carry it linearly (the extractor reads
  it into the NCP column of h), the w rows also in products with q_x (source, hence an aux).
- **Fix.**  `ml_vam.py` `_frac`: d^n h_l -> l_l d^n h before the substitution (l_l is a
  constant: chain rule), at the three row sites (horizontal, w, constraint).  `splitter.py`: the
  predictor also evaluates a derivative of a number (d_x(0) = 0), as it already did for the
  non-horizontal ones.
- **Checks.**  The six configs above: every operator entry after the fix equals the one before
  with h*l_j_d := l_j h_d (symbolically); no derivative aux with a non-state target is left,
  in the model or the split.  Goldens m15 / s03 re-blessed for that reason: every changed entry
  equals the old one after that substitution (m15: NCP [q_2_1, r_2_1, P_2_0, P_2_1 <- h] and
  source rows r_1_0, r_2_0, r_2_1; s03 predictor: the same, the 0_x term, eigenvalues_cfl), the
  aux list loses h*l_1_x (s03: and 0_x; `update_aux_variables` shifts by those rows).  New test
  `tests/model/test_mlvam_derivative_aux.py`: (1) every derivative aux of the model and its split
  targets a state; (2) the w rows' source with the LSQ aux against exact derivatives
  (d_x(l_1 h) = l_1 h'), smooth field, max rel 2.1e-3 / 5.3e-4 at nx 64 / 128 (order 2.0);
  before the fix 0.63 / 0.63.
- **Run** (2 layers, level 1, l_1 = 0.4, 1-D, 100 cells on [-1.5, 1.5], Gaussian bed 0.08, a
  0.04 wave on 0.34, u ~ 0.3, CFL 0.45, t = 0.5; P = 0 right, extrapolation left; JAX): after
  minus before, max abs / max |Q_row|: h 2.9e-5, q_l_0 1.4e-4 to 1.7e-4, q_l_1 3.8e-3 to 4.2e-3,
  r 7e-4 to 2.3e-3, P_l_0 1.6e-3 / 3.4e-3, P_1_1 1.8e-2.  NumPy (now runs) = JAX 6.6e-11.
- **Seen, not changed.**  Sigma3D (m16) registers `h*zeta_x` with `target_kind='unknown'` (the
  same pattern, zeta a coordinate there).  NumPy's `derivative_aux` raises on any non-state
  target when the BC ghosts are present (JAX skips it, reading 0).

### 45j. ML-VAM w rows: the interface trace of tau_ez (2026-10-14, dev/mlvam-aux)

- **What the rows do.**  With `Stress.full` the vertical row keeps d_e tau_ez = d_e f_e,
  f_e = rho nu d_z u_e (the eps^2 part rho nu d_e d_e w dropped, 45h), projected pointwise per
  layer: the w row (l, k) has RHS (2k+1)/rho <phi_k, h_l d_e f_e> (gate 48).  By Leibniz
  <phi_k, h_l d_e f> = d_e(h_l <phi_k f>) + <phi_k' (d_e z_(l-1) + zeta d_e h_l) f>
  - phi_k(1) f(z_l^-).grad z_l + phi_k(0) f(z_(l-1)^+).grad z_(l-1): the interface terms use the
  layer's own trace.  The ML ansatz is discontinuous at z_a, so the two layers use different
  values and the layer sum of the mode-0 rows keeps -(f(z_a^-) - f(z_a^+)).grad z_a, a vertical
  momentum source wherever the shear jumps at a sloping interface.  Level 0: f = 0 in every layer
  (u constant per layer), nothing to choose; level >= 1 only.
- **Options.**  (1) one-sided (the pointwise projection, current); (2) the mean of the two traces;
  (3) upwind by the sign of G; (2)/(3) as a convex combination, one value per interface; (4) a
  value independent of the profiles from the two layer states, e.g. a finite difference across
  the interface (Audusse et al. below); (5) a traction continuity condition: for a weak solution
  the interface traction is continuous up to the momentum flux through the moving interface
  (Garres-Diaz et al. (A.1)), which fixes the jump of the traction, not the value the projection
  needs; the value is a numerical-flux choice, (2) to (4).
- **Literature** (Zotero; equations as in the papers).
  - Garres-Diaz, Escalante, Morales de Luna, Castro Diaz 2023, "A general vertical decomposition of
    Euler equations: Multilayer-moment models" (attachment ER5MCUCF), App. A, the viscous
    multilayer-moment system: "[u omega + tau_xz - d_x(h xi + b) tau_xx]_(alpha+1/2) = 0, or
    equivalently [-tau_xz + d_x(h xi + b) tau_xx]_(alpha+1/2) = [u]_(alpha+1/2) omega_(alpha+1/2)"
    (A.1); with sigma = -tau_xz + d_x(h xi + b) tau_xx the layer integral leaves
    sigma^-_(alpha+1/2), sigma^+_(alpha-1/2) "where sigma^-, sigma^+ should be correctly defined.
    In [30], this is done by means of a centered approximation sigma~ = (sigma^+ + sigma^-)/2.
    Nevertheless, one could do other possible choices", (A.2) "sigma~ = G(sigma^-, sigma^+) =
    (sigma^+ + sigma^-)/2 - Lambda/2 (sigma^+ - sigma^-)", "Lambda allows us to include an upwind
    term".  ([30]: Fernandez-Nieto, Kone, Chacon Rebollo 2014, J. Sci. Comput., not in Zotero.)
    Horizontal momentum only (hydrostatic).
  - Fernandez-Nieto, Parisot, Penel, Sainte-Marie 2018, "A hierarchy of dispersive layer-averaged
    approximations of Euler equations" (4S8BGCKE): the pressure continuous, "[[q]]_(alpha+1/2) = 0"
    (3.2b); every interface value "f~_(alpha+1/2) = (1 - gamma) f^+ + gamma f^-, gamma in [0, 1],
    any convex combination providing a mean value at the interface" (3.3); vertical momentum
    (3.8c) "d_t(h_alpha w_alpha) + d_x(h_alpha u_alpha w_alpha) + w~_(alpha+1/2) Gamma_(alpha+1/2)
    + q_(alpha+1/2) - w~_(alpha-1/2) Gamma_(alpha-1/2) - q_(alpha-1/2) = 0": every interface term
    single-valued (inviscid).
  - Escalante, Fernandez-Nieto, Morales de Luna, Castro 2019, two-layer non-hydrostatic
    (86BQYCRP): (14) "d_t(h_1 w_1) + d_x(h_1 u_1 w_1) + (w_1 + w_2)/2 Gamma_I + p_(I^-) - p_b = 0",
    (15) "d_t(h_2 w_2) + d_x(h_2 u_2 w_2) - (w_1 + w_2)/2 Gamma_I - p_(I^+) = 0", closed by (16)
    "p_(I^+) = gamma_1 p_b + gamma_2 p_I": a two-valued interface pressure on purpose, free
    parameters for the dispersion relation ("for gamma_1 = 0, gamma_2 = 1 we recover the original
    system", the continuous one).  Inviscid.
  - Audusse, Bristeau, Perthame, Sainte-Marie 2011 (5ALPFGP8), viscous, hydrostatic: the viscous
    pressure "int d_x(2 nu d_x u) = d_x(2 nu h_alpha d_x u_alpha) + 2 nu [d_x z_j d_x
    u_j]_(j=alpha-1/2)^(j=alpha+1/2)", one value u_j per interface (the interface velocity
    (2.10), upwind by G in (2.23)); the vertical shear "using finite differences along the
    vertical ... ~ 2 nu (u_(alpha+1) - u_alpha)/(h_(alpha+1) + h_alpha) - 2 nu (u_alpha -
    u_(alpha-1))/(h_alpha + h_(alpha-1))".
  - Not found: a multilayer model with the viscous tau_ez in the vertical momentum (the
    non-hydrostatic ones are inviscid, the viscous ones hydrostatic).  The common rule: an
    interface term is single-valued (centered, convex combination, upwind, finite difference);
    the one exception is Escalante et al.'s pressure, two-valued as a dispersion tuning.
- **Recommendation: (2) / (3) through the existing stress `InterfaceFlux`.**  The horizontal rows
  already take T* = tau*_dz - s*.grad z_a from `InterfaceFlux(quantity=InterfaceQuantity.stress)`
  (mean by default, `UpwindInterface` by G): Garres-Diaz (A.2) with Lambda = 0 / sign.  The
  vertical component of the same traction at the same interface is p - tau_zz (continuous: the
  pressure unknown is -sigma_zz, 45f) plus the tau_ez part; f* from the same scheme makes the
  whole traction vector single-valued by one closure, keeps gate 48's reduction (identical layer
  states: f^- = f^+ = f*, so the ML rows still sum to VAM's), and makes the mode-0 w rows
  telescope.  (4) changes the value for continuous profiles too (breaks that reduction), and at
  level 0 it is the vertical shear between layers (45f item (4)), a different model.
- **Prototype.**  `MLVAM(vertical_traction=VerticalTraction.one_sided | shared)` (typed `Option`,
  default `one_sided`; `shared` with `Stress.shallow` raises): w row (l, k) gets
  sgn phi_k(side) (f* - f_own).grad z_a / (rho mu_k) (+ at the top of the lower layer, - at the
  bottom of the upper), the form of the u* transfer; f* by the stress `InterfaceFlux`.  Default
  byte-identical: the six 45i configs' SystemModels equal those before the change.
- **Checks.**  Gate 48 (`mlvam:L:D:ws`, new): level 1 1-D and 2-D, level 2 1-D: w rows = the
  pointwise reference (`one_sided`) / reference + the correction (`shared`), exactly; horizontal
  rows unchanged; identical layer states: `shared` = `one_sided`; mode-0 layer sum: `shared` =
  d_e sum_l h_l <f_l> - f(eta).grad eta + f(b).grad b, `one_sided` that minus
  (f(z_1^-) - f(z_1^+)).grad z_1 / rho.  0 unclassified.  Test
  `tests/model/test_mlvam_vertical_traction.py` (the residual difference = the formula; shallow
  raises).  Gate 48's other specs (ML-SME 0 / 1, ML-VAM 0 / 1, ML-VAM 1 upwind; 1-D) still 0 /
  0 after 45i and this change.
- **Effect** (45i's run with `Stress.full`, q_2 = 1.2 q_1, q_l_1 = 0.1 q_l_0 (a shear jump at
  z_1), JAX, t = 0.5).  nu = 1e-3: `shared` - `one_sided` max abs / max |Q_row| 3e-7 (h), 2e-6 to
  1.2e-5 (q), 1.1e-5 to 3.3e-5 (r), 1.1e-5 to 5.2e-5 (P); the viscous terms themselves
  (nu = 1e-3 vs 0) move the state by 1.6e-3 (h), 7e-3 to 0.25 (q), 1.4e-2 to 8.4e-2 (r), 1.5e-2 to
  2.7e-2 (P): the trace choice is 1e-4 to 2e-3 of the viscous effect here.  nu = 1e-4: both ten
  times smaller.  NumPy = JAX (`shared`, nu = 1e-2) 4.4e-10.
- **Open.**  nu = 1e-2 grows in both variants (h 0.38 -> 0.51 by t = 0.5, q_2_1 0.01 -> 0.27;
  explicit predictor, Chorin), not investigated.  With `UpwindInterface(quantity=stress)` the
  setup of a run (JAX and NumPy, `one_sided` too) did not finish in 14 min; the derivation does
  (0 unclassified, no non-state aux).
 (pyamg): the iteration count does not scale (2026-10-13)

L2P's VAM Chorin at nx 256 (65k cells), `pressure_linear_solver=LinearSolver.pyamg`, 3 constant steps
(`scratchpad/l2p_prof.py`: pyamg wrapped to count and time V-cycles; `ZOOMY_KRYLOV_LOG=1`).
Matched: one core per rank, XLA single-threaded (`--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1`), `mpiexec -bind-to core` (the env's MPI is MPICH / Hydra).

| ranks | ms / step | Krylov iterations / solve | V-cycle per rank |
|---|---|---|---|
| 1 (host GMRES + global SA-AMG) | 135 | 10 | 4.4 ms |
| 2 (JAX FGMRES, rank-local V-cycle + Nicolaides coarse) | 164 | 28 | 2.3 ms |
| 4 | 208 | 76 | 1.2 ms |

- The V-cycle scales (cost per rank halves with the rank's size); the preconditioner does not:
  the rank-local AMG drops the coupling between ranks, and one constant per rank and row is too
  small a coarse space to restore it, so the iterations grow with the rank count.
- Each distributed iteration also costs a JAX matvec with a halo exchange, two global
  reductions and a host callback; at 76 iterations that dominates.
- Unmatched runs mislead twice: with 8 XLA threads per process the serial run takes 266 ms /
  step (XLA's intra-op threading costs 2x at this size), and two ranks sharing those 8 cores
  oversubscribe (479 ms / step).
- Fix (proposed): one global AMG across the ranks, i.e. hypre BoomerAMG + FlexGMRES on the
  colour-assembled rows (as foam and AMReX), called once per solve on the host instead of once
  per iteration.  Needs a hypre build against the env's MPICH (the existing CPU build links
  OpenMPI 4.1.6, the OF13 / AMReX toolchain).

Result (dev/jax-hypre): `pressure_linear_solver=LinearSolver.hypre` (zoomy_jax `symsolver/hypre.py`: ctypes on
hypre 3.2.0, the shared `build_hypre_cuda.sh cpu-env` build against the env's MPICH; `auto` takes it
on CPU MPI ranks).  The owned rows with global columns by colouring (as `_amgx_mpi`), left-scaled by
the inverse 2x2 cell blocks, FlexGMRES + one BoomerAMG V-cycle with the foam / AMReX CPU settings
(HMIS, ext+i, hybrid symmetric GS) to the true relative residual 1e-10, the hierarchy lagged as
pyamg's (`pressure_amg_rebuild` 10).  One host callback per solve, between two mpi4jax allreduces
(data dependencies), so its collectives keep one place among mpi4jax's ordered ones on every rank;
reverse mode solves `S^T z = c`, `x = D^-T z` (hypre assembles `S^T` across the ranks).  Same matched
runs (nx 256, one core per rank, XLA single-threaded); ms / solve from the right-hand side ready to
the solution ready (colouring probes, host solve, accept gate):

| ranks | pyamg ms / step | its / solve | ms / solve | hypre ms / step | its / solve | ms / solve |
|---|---|---|---|---|---|---|
| 1 | 137 | 10 | 84 | 100 | 18 | 47 |
| 2 | 164 | 28 | 130 | 57 | 19 | 26 |
| 4 | 207 | 76 | 185 | 37 | 20 | 16 |

- A fresh hierarchy every solve (as foam / AMReX): 117 / 72 / 46 ms / step (setup 15 / 16 / 11 ms).
- Hydra's `-bind-to core` ignores `taskset` (it bound to cores 0, 2, 4, 6); these runs use
  `-bind-to user:40,42,44,46` (one socket).  pyamg reproduces the table above.
- Correctness (this worktree's store; foam / AMReX / numpy runs copied from main's): C8 / C10 / C31
  49 rows PASS, 0 FAIL (as main); C8 `vam` mpi2 / mpi4 vs serial 1.5e-11 / 2.4e-11 (gmres:
  8.7e-10); hypre vs pyamg serial: C8 4.3e-11, C10 2.5e-12, C31 VAM 1.2e-14.  C10 / C31 (serial
  only in the matrix) on 2 / 4 ranks vs serial: C10 `s1_t1_n60`, `s2_t2_n120` <= 3.6e-13, C31
  `vam_ssp2_k0`, `vam_t1_k0/k1` <= 3.6e-15, but `vam_t1_k3`, `vam_t2_k3` 1.35e-9 (gmres the same,
  mpi2 = mpi4 to 4e-16: a distributed defect at dt / 8 that predates hypre, open).  AD (C15's
  Chorin setup, dJ/dg): jvp / grad vs gmres 1.4e-13 / 3.6e-12 serial, <= 1.1e-12 on 2 ranks.
- L2P (1M cells, `matrix.py run --tier large L2P jax mpi4`, the harness's unbound ranks): 8562 ms /
  step, 21 iterations / solve; host per solve: assembly 972 ms, FlexGMRES 2930 ms, one AMG setup
  (2.3 s) in 6 solves; setup 73 s, compile 27 s, whole command 2:36 (pyamg: > 10 min; foam / AMReX
  mpi4 8661 / 8180 ms / step).  Single-threaded XLA per rank: 8948 ms / step.

## 47. Large tier L2P on main: one AMG on every backend (2026-10-13)

L2P (VAM(1) Chorin, 1M cells, 5 steps of dt 1e-4; `matrix.py run --tier large L2P`), main after
the foam / JAX hypre and the foam kernel-I/O merges.  ms / step (Krylov iterations per solve):

| backend | serial | mpi2 | mpi4 | gpu1 | gpu2 | gpu4 |
|---|---|---|---|---|---|---|
| AMReX (hypre) | 24993 (19) | 13209 (19) | 7722 (20) | 534 (38) | 459 (38) | 416 (38) |
| foam (hypre) | 27998 (19) | 13948 (20) | 6900 (19) | - | - | - |
| JAX (CPU hypre, GPU AmgX) | 23766 (19) | 14623 (20) | 8645 (21) | 1417 | 1383 | 1135 |

- Every CPU backend now runs the same solver (hypre FlexGMRES + BoomerAMG on the colour-assembled,
  cell-block-scaled operator) with flat iteration counts; all configs fit 10 min (before: none on
  AMReX / foam, JAX only on GPUs).  JAX `auto` takes hypre on the CPU whenever built (serial:
  pyamg 34725 ms / step).
- foam kernel I/O on fixed-size stack storage (zoomy_foam `dev/foam-kernel-io`): L2P serial / mpi2 /
  mpi4 33840 / 15251 / 8661 -> 27998 / 13948 / 6900 ms / step (assembly 12.2 -> 9.6 s per solve);
  bitwise-equal results.
- AMReX boxes: the default `max_grid_size` divided the longest side by (boxes wanted)^(1/dim), so
  L2P's 3200 x 320 channel got 6 boxes on 4 CPU ranks (2 / 2 / 1 / 1) and one box per GPU; it now
  follows the cell count (square domains unchanged): mpi4 9118 -> 7722 ms / step (36 boxes), and
  the Chorin GPU default asks 4 boxes per GPU (box sweep: 2 / 4 GPUs 571 -> 445 / 392 at 8 / 32
  boxes; with the rule 459 / 416 at 9 / 26).  One GPU: 513 -> 534 (7 boxes).  At 1M cells the
  GPUs are underloaded, and hypre on the GPU needs 38 iterations against 19 on the CPU (its GPU
  smoother / coarsening).  JAX GPU rows are the earlier AmgX runs.
- **Open defect:** 1M cells on 2 GPUs as 3 boxes (`max_grid_size` 1067) completes no step in 7 min
  with both GPUs busy (91 / 63 %), reproduced twice; 2, 8, 9 boxes are fine, and so are 3 boxes on
  2 GPUs at 10k cells and on 2 CPU ranks.  Not diagnosed (hypre GPU on an uneven 2 / 1 split is
  the suspect); the default no longer produces that layout.

## 48. Wave speed without a closed-form spectrum: an upper bound of rho; the faces rule: MOOD repairs at their own step, the cells' speeds under the edge source (2026-10-14, dev/wavespeed-bound)

The default wave speed of a model without closed-form eigenvalues (`zoomy_core.symsolver.WAVE_SPEED`)
was power iteration on A_n^2 (10 steps from x = 1) times 1.06.  Power iteration converges to rho
from below, so the product is an estimate, not a bound, and the CFL step and the HLL / Rusanov
speeds lose their positivity guarantee.  Measured (gate 49, below): ratio to max|eig| down to 0.43
on random ML-SME states and 0 on states of the Malpasset and C5 runs.

**The bound** (`SpectralRadius.gelfand`, the new `WAVE_SPEED`; opaque `gelfand_bound(j, delta, *A_flat)`:
`kernel_functions`, NumPy / JAX `userfunctions`, foam and AMReX `UserFunctions.H`; `power` and the
`spectral_radius` kernel are removed):

- `j` squarings, each after dividing by the max row sum: `s_i = ||C_i||_inf`, `C_(i+1) = (C_i / s_i)^2`,
  `C_0 = A`, so `C_j = A^m / prod_i s_i^(2^(j-i))`, `m = 2^j`, entries <= 1 (no overflow).
- One Collatz-Wielandt step on `|C_j|`: `x` = its row sums plus `delta` their max (positive),
  `w = max_i (|C_j| x)_i / x_i`, so `rho(C_j) <= rho(|C_j|) <= w`.  `delta =
  SystemModel.spectral_cw_offset = 2^-30`, emitted by core into every call (no literal in a
  backend kernel).
- `rho(A) <= s_0 sqrt(s_1 sqrt(... sqrt(s_(j-1) sqrt(w))))` from `rho(A)^m = rho(A^m)`; with
  `||C_j||_inf` in place of `w` this is Gelfand's `||A^m||^(1/m)`, which tends to rho.  Horn,
  Johnson 2012, https://doi.org/10.1017/CBO9781139020411: section 5.6 (rho <= every matrix norm,
  Gelfand's formula), section 8.1 (rho(A) <= rho(|A|), Collatz-Wielandt).
- Matrix products, sums, max and `sqrt` only (correctly rounded on every backend); `+inf` for a
  non-finite entry (as the `eigenvalues` kernel), 0 for a zero matrix; JAX gradients finite at a
  zero matrix.  `j = SystemModel.spectral_squarings = 7` (m = 128).
- Why `x` is bounded away from 0: with the plain row sums, a row that is round-off (q_y ~ 1e-20 in
  the quasi-1-D C10) still has a ratio of O(1), so `w` jumps with the round-off of each backend.
  C10 `s2_t1_n30`, NumPy against JAX: 1.03e-5 (plain row sums), 1.16e-12 (with 2^-30 max), 1.49e-12
  (`eig`).
- A state whose row or column of A_n is structurally zero is dropped before printing (block
  triangular with a zero block: same rho): `b` (SME, VAM), `P_0`, `P_1` (VAM predictor).  SME(0)
  2-D: 4x4 -> 3x3, VAM(1) 2-D: 10x10 -> 7x7.

**Rigour and tightness** (`gates/gate49_wave_speed_bound.py`: 65 sets, 1,890,568 matrices, the
production path `local_max_abs_eigenvalue` -> `gelfand_bound`, the NumPy and the JAX kernel each),
bound / max|eig|; median and max over rho > 1e-8 ||A||:

| set | matrices | min | median | max | power x 1.06, min |
|---|---|---|---|---|---|
| random: Gaussian, triangular (upper part 1, 1e3), Jordan blocks, nilpotent, complex pairs, V D V^-1; n = 2, 3, 5, 7, 10; scaled 1e-6 .. 1e6 | 128,000 | 1.000000000000000 | 1.000 - 1.360 | 1.569 | 0.17 |
| SME(0..3) 1-D / 2-D, VAM(1) predictor, ML-SME(2 layers, level 0 / 1): random wet, thin, dry cells and thinned (Audusse) faces | 440,000 | 1.000000000000001 | 1.003 - 1.038 | 1.093 | 0.43 |
| states of runs: Malpasset (new, GPU; old reference B, CPU), matrix store C1, C2, C4, C5, C8, C10, C14, C33, C36 | 1,322,568 | 1.000000000000000 | 1.000 - 1.003 | 1.059 | 0.000 |

- Not below 1 anywhere (rho > 1e-100 ||A||; below that, dry round-off garbage such as q ~ 1e-240
  at h = 0, the squares underflow and the bound may be 0).  No state needed the 40-digit recheck.
  NumPy = JAX to 6e-11 on the run states (5e-6 on ill-conditioned random V D V^-1: squaring
  amplifies round-off where ||C^2|| << ||C||^2); each kernel is a bound on its own.  C++ (the
  foam source, host build, C++14) = NumPy to 5.5e-14 on 20,000 Gaussian 5x5.
- Squarings and the Collatz-Wielandt (CW) step, median / max:

| states | j = 4 | j = 5 | j = 6 | **j = 7** | j = 8 | j = 7 without CW |
|---|---|---|---|---|---|---|
| SME(0) 2-D thinned faces | 1.237 / 1.840 | 1.133 / 1.379 | 1.072 / 1.184 | 1.038 / 1.093 | 1.019 / 1.048 | 1.046 / 1.095 |
| Malpasset run (new) | 1.002 / 1.351 | 1.001 / 1.162 | 1.000 / 1.078 | 1.000 / 1.038 | 1.000 / 1.019 | 1.012 / 1.040 |
| C5 run (wet / dry) | 1.025 / 1.475 | 1.013 / 1.215 | 1.006 / 1.109 | 1.003 / 1.059 | 1.002 / 1.032 | 1.009 / 1.126 |
| triangular, upper part 1e3, n = 10 | 11.26 / 22.04 | 3.42 / 5.39 | 1.85 / 2.42 | 1.36 / 1.57 | 1.17 / 1.25 | 1.60 / 1.85 |

  The CW step costs one matrix-vector product.  Highly non-normal matrices stay loose (last row),
  the model matrices do not.  j: the overestimate is a state-dependent factor on the dissipation;
  C11 (1-D smooth wave, O2, L1(h) at nx 80 / 160 against the JAX nx 1280 reference): j = 6 3.86e-4
  / 1.34e-4 (order 1.53, threshold 1.5), j = 7 3.91e-4 / 1.25e-4 (1.65), j = 8 3.96e-4 / 1.25e-4
  (1.66), numerical eig 3.97e-4 / 1.23e-4 (1.69); the former power x 1.06 3.71e-4 / 1.06e-4 (1.81),
  but at O1 its error was 1.7x that of eig (n 40: 1.41e-3 against 8.10e-4; j = 7: 8.2e-4).  Hence
  j = 7: median on run states <= 0.3 % (the former default added 6 % everywhere).
- Cost (the foam kernel, host build, -O3, one core; us / call, gelfand(7) / power(10)): n = 3 0.19 /
  0.17, n = 5 0.49 / 0.32, n = 7 0.97 / 0.47, n = 9 1.17 / 0.61, n = 10 1.43 / 0.63.  The dropped
  rows make the printed SME(0) 2-D matrix 3x3 (power ran on 4x4) and VAM(1) 7x7 (was 10x10).

**Positivity, Malpasset** (SME(1), the GUI session model without `symbolic_spectrum`, order 1, CFL
0.4, `hr`, 26,000 cells, to t = 120 s, the study's `run_case.py`).  The bound alone did not restore
it:

| run | wave speed | h_min | released water at 120 s (m^3) |
|---|---|---|---|
| B: old solver (reference) | closed form | 0 | 4,048,870 |
| C: new solver | closed form | 0 | 4,053,340 |
| before (Cnum, GPU) | power x 1.06 of A_n(KP hinv) | -1.26e-5 | 4,149,336 |
| control, GPU | Gelfand of A_n(KP hinv) | -1.34e-5 | 4,149,850 |
| control, CPU, to t = 5 s | numerical eig of A_n(KP hinv) | -1.70e-5 | - |
| **new default, CPU** (6 restarted 20 s windows, 30 snapshots) | **Gelfand of A_n(1/max(1e-14, h))** | **0** | 4,149,474 |
| new default, GPU 2: 25 snapshots / h_min every 0.05 s (2401 samples) | same | 0 / 0 | 4,149,512 |
| control, GPU, every 0.05 s | power x 1.06 of A_n(1/max(1e-14, h)) | 0 | 4,149,421 |
| control, GPU | closed form, HLL speeds forced to (-a, a) | 0 | 4,149,694 |

- Cause: A_n carries the KP `hinv = sqrt(2) h / sqrt(h^4 + max(h, eps)^4)` (eps = 1e-3).  At an
  Audusse face with h* < eps and q* = q h*/h != 0, `hinv*` is far below 1/h*: the speed of A_n(hinv*)
  misses the transport velocity q*/h* of the mass flux (on thinned Malpasset states the closed form
  exceeds rho(A_n(hinv*)) by up to 1.5e5) and the cell drains.  This is REQ-82, which gave the
  closed-form `eigenvalues` slot `1/max(1e-14, h)` (`regularize_pow`); the numerical spectrum still
  took A_n with the KP `hinv`.  Now every numerical-spectrum path (`gelfand`, `eig`, `bound`)
  evaluates A_n with `hinv` -> `1/max(1e-14, h)` (`Numerics._wave_speed_aux`; one definition,
  `operations.wave_speed_inverse_depth`), so the closed and the numerical spectrum are taken of the
  same matrix.  Golden `p01_foam_vam` (the old foam printer's `eig` path) regenerated for it.
- With that floor the former power estimate also kept h >= 0 on this run; the bound makes it a
  guarantee (power x 1.06 is down to 0 on these very states, table above).
- The +2.5 % released water of every numerical-spectrum run against C is not the wave speed: the
  closed form with Rusanov speeds gives the same (4,149,694).  Without a closed-form spectrum HLL
  has no Davis speeds and is local Lax-Friedrichs (-a, a).

**Which step the scheme needs.**  At first order the h update is `h_K' = h_K - dt / |K| sum_f |e_f|
F_h(U*_K, U*_N)` with the Audusse states (`h* <= h`, velocities kept).  For HLL / Rusanov,
`F_h <= S_f h*_K` whenever `S_f = max(|s_L|, |s_R|)` bounds the velocities of both face states, so
`h_K' >= h_K (1 - dt sum_f |e_f| S_f / |K|) >= 0` iff `dt sum_f |e_f| S_f <= |K|` (Audusse et al.
2004, https://doi.org/10.1137/S1064827503431090).  `S_f` must be the speed of the Riemann problem
the step solves at f.  The template's `dt = cfl 2 r_f / (d S_f)` (`r_f` the smaller inradius; on
triangles `|K| = r_K P_K / 2`) gives `dt sum_f |e_f| S_f <= 2 cfl |K|`: cfl <= 1/2 (the runs: 0.4).
At order 2 the update also carries the edge source, the NCP integrated inside each cell from its
state to its traces (Pares 2006, https://doi.org/10.1137/050628052): a wave term of the cell's own
states, which the step must bound as well.

- `CflWaveSpeed.cells` takes `S_f` from the two cells' states: the Audusse states keep the velocity
  and lower h, so their speed is at most the cells' (SWE / SME: u, alpha = q_1/h kept, c(h*) <=
  c(h)), and the edge source moves the cell states themselves.
- `CflWaveSpeed.faces` took `S_f` from the first stage's Riemann problems only.  Two holes, both on
  the Malpasset thesis configuration (`malpasset_jax_model.py`: SME(1) mixing length, closed-form
  spectrum, order 2 Venkatakrishnan + front detector + MOOD, `hr`, point-implicit source, CFL 0.4;
  JAX, GPU):

1. **MOOD's repair kept the order-2 candidate's step** ("the repairs keep the candidate's dt").
   Per-step replay of main from the t = 29 s write (`dt2`: the candidate's step; `own`: the faces
   rule on the repair's own Riemann problems): the first repair, t = 30.739 s, `dt2` = 12.6 ms, own
   0.41 ms (1/31).  Cell 21011 (dam abutment, b = 90 m, h = 0.548 m, mode-0 velocity at the 30 m/s
   cap, alpha_1 = q_1/h = 1.9e3 m/s) has first-order face speeds 1083, 1625, 2690 m/s; its order-2
   traces have h = -0.16 .. -0.42 m at two faces, Audusse h* = 0 on both sides, speed 0.  The full
   first-order fallback at `dt2` drained its neighbours to h = -4.28 m (cells 20370, 20498, 20754);
   at its own step h >= 0.  Fix (`residual.mood_step`): a repair takes `dt = min(candidate dt, the
   faces rule on its own first-stage speeds)` and the step returns it; an admissible candidate is
   unchanged.  Unit test: an 8-cell 1-D state (random search; bed steps, dry / thin / deep cells)
   where the repair at the candidate's step reaches h = -8.9e-3 at t = 0.01, >= 0 at its own.
2. **The edge source moved cells no Riemann problem saw.**  With fix 1 alone h stayed >= 0 at every
   write, but a wet cell whose order-2 traces are Audusse-dry on every face (zero flux, zero face
   speed) kept its first moment under the edge source at a step its own speed never limited:
   max alpha_1 (h > 1e-3 m) up to 1.9e16 m/s in transient episodes (12, 14 and 30 writes above
   1e3 m/s in the 3 runs that recorded it); in 2 of 9 runs it overflowed: a non-finite face speed
   made the global `dt` NaN, every entry NaN after one step, and the march loop ended without an
   error (t = 687 s, 939 s; the run before this one: 996 s).  At t = 687 s, cell 8585 (b = 78.9 m,
   h = 0.575 m): alpha_1 = 2.3e13 m/s, order-2 face speeds 0, 0, 0.11 m/s.  Control: the same step capped by
   the cells rule: max alpha_1 665 / 935 m/s (2 runs).  Fix (`residual.residual`): with `dt_rule`,
   order 2, an edge source and not every cell demoted, `dt = min(dt_rule(Riemann speeds),
   dt_rule(cell speeds))` (the cells rule's `cell_wavespeed`).  Order 1 and models without an NCP are
   unchanged.  Unit test: on 200 random SME(1) 1-D states the faces rule's order-2 step is <= the
   cells' CFL step (without the term it exceeds it, e.g. 1.5465e-3 > 1.5425e-3).

Malpasset to 2000 s, a write every 1 s (2001 writes; `results/wavespeed_bound/run.py`, figure
`malpasset_wavespeed.png` from `deliverable.py`):

| libraries | rule | runs | h_min (all writes) | max alpha_1 (m/s) | end |
|---|---|---|---|---|---|
| main 790dc047 | faces | 3 (to 40 s, 60 s, 2000 s) | -2.43, -2.37 m at t = 31 s; the 2000 s run stalled at t = 905 s | 1.9e3 (60 s) | 905 s |
| MOOD repair at own step only | faces | 9 | 0 | 5.9e9 .. 1.9e16 (episodes) | NaN at 687 s, 939 s (2 of 9) |
| **this branch** | **faces** | 5 | **0** | 229 .. 877 | 2000 s |
| this branch | cells | 4 | 0 | 200 .. 265 | 2000 s |
| main 790dc047 | cells | 2 | 0 | 259 | 2000 s |

**The default (cells, the session's rule) is unchanged**: the branch only touches the steps of
`dt_rule` (faces) and the kernel of models without a closed-form spectrum, neither on this path.
CPU, main against this branch, a write every 1 s: Q bitwise equal at all writes (to 20 s before
the edge-source term, to 10 s with it).  GPU to 2000 s: wet area (h > 0.05 m) max rel. difference
1.1e-3, max-depth envelope L1 rel. 8.6e-4, inundated area 17.024 / 17.030 km^2, against a GPU
repetition of the same code of 3.0e-3 / 1.5e-3 (atomics order).  Faces against cells (this branch):
1.3e-3 / 9.9e-4.

- The last resort of MOOD is positive whenever the first-order step is (cfl <= 1/2); the second
  SSP-RK2 stage uses the first stage's speeds under either rule (covered by the margin 0.4 < 0.5
  only).  Cost (GPU wall, 2000 s, incl. setup): faces 427 s (repair-only 377 s), cells 478 s.

Cross-backend matrix (`matrix/README.md`, results 2026-10-14): the store rerun from empty, 962
runs on core 082acb3, jax edc4c35, foam c4310bc, amrex a177451: 871 rows, 801 PASS, **0 FAIL**.
Moved inside their thresholds: C11 smooth O2 order 1.81 -> 1.65 (>= 1.5; the exact spectrum gives
1.69), O1 0.96 -> 1.02, C14 O1 0.59 -> 0.63 (the former power x 1.06 under-estimated the speed);
C8 cross-backend / parallel differences 1e-10 -> 1e-11.  zoomy_core: 360 small tests pass (3 new:
`tests/symsolver/test_wave_speed_bound.py`); goldens `p01_foam_vam` and `n02_vam_nsm` re-blessed
(the numerical spectrum of A_n with `1/max(1e-14, h)`, above); rederive tier 40 pass, 1 failure
that main bb5f636 has too (`test_rederive_cross_process_disk_hit_and_prebuilt`, AttributeError
`default_factory`).

Open:
- The cells rule is not an upper bound at thin cells: `eigenvalues_cfl` is the spectrum gated to 0
  for h <= `wet_dry_eps` (`gate_eigenvalues_dry`; numerical spectra with the KP `hinv`), but a thin
  cell's flux moves its velocity.  1-D reproduction (SME(0), order 2 unlimited + MOOD, a 1e-3 m film
  at 300 m/s next to a dry bed): h = -0.30 m under the cells rule with forward Euler, at a cell with
  h = 6.7e-9 m and q/h = -2.3e9 m/s; order 1 stays positive.  Not changed (it would change the
  default rule's results).
- HLL without a closed-form spectrum is Rusanov (above); bounds of the signed extremes (e.g. the
  bound of A + c I minus c) would give it Davis speeds.
- A NaN `dt` ends the march loop silently (`time < time_end` is false); the loop should raise.
- The first moment of the mixing-length SME(1) reaches alpha_1 ~ 260 m/s under the cells rule too
  (the mode-0 velocity is capped at 30 m/s, mode 1 is not).

## 49. Non-dimensional scaling pipeline: the stress truncation by order (2026-10-14, dev/nondim)

Design: `research/nondimensional_scaling.md` (P3).  Built in zoomy_core, opt-in; the hand rules
stay the default.

**What was built.**

- `model/derivation/nondim.py`: `Dim` (M, L, T, rational exponents) and `PARAMETER_DIMENSIONS`;
  `PiBasis` (Buckingham Pi: every group dimensionless, the groups independent and complete, else
  `DimensionError` naming the group or a missing one, e.g. `missing e.g. U**2*rho/S`; every
  quantity written through the repeating scales and the groups; `suggest()` gives a basis, the
  operator never chooses groups).  The operators, all `m.apply(...)` on the 3-D rows before the
  sigma map: `NonDimensionalize(scales, groups, ...)` (fields -> `scale * \check{f}` or `Affine`
  background + fluctuation, a derivative in a coordinate picks up `1/scale`, each row divided by
  its scale; every term must end as a group monomial times a field part, else "not dimensionally
  homogeneous"); `ShallowScaling(horizontal, vertical, regime=Regime(...))` (drops every term
  whose eps order, read off its group coefficient, exceeds its row's order; a second call without
  arguments repeats the orders); `InsertClosure(closures)`; `Dimensionalize()` (the exact inverse:
  the round trip is structure-preserving, the scaled frame never changes a kept term).
- `InsertClosure`: the bulk closure's physical tensor (`Newtonian.tensor`) in the scaled
  variables; its vertical shear fixes the free stress scale (`S = rho nu U / H`, so
  `De = eps U H / nu`, thesis eq. (fr-de)).  **Consistency check:** every term of every
  `tau_hat_ij` and of every bed / surface traction (`traction`, with the closure's group:
  `NavierSlip.scaling_groups` `k_l = lambda_s / (rho U)`, Marche 2007 eq. (5.1) `k_l' = eps r_0`;
  `ManningFriction` `C_n = g n^2 L / H^(4/3)`, O(1)) must be O(eps^0) or smaller under the
  regime, else `ScalingError` naming the closure and component.  Substituted everywhere except
  the horizontal rows' own shear `tau_dz`, which stays a closure variable (its traces carry the
  traction, 45e); for it the row's order selects the kept closure terms (`ClosureRecord.shear`).
- `model/models/scaling.py`: `FreeSurfaceNonDim(momentum, heights, free_surface, pressure)`, the
  thesis choices (20_methodology eqs. (scaling-variables), (pressure-scaling) with
  `Pressure.total` -> `p = rho g (s - z) + q`, `Pressure.nonhydrostatic` for VAM's / ML-SME's
  pressure, (sme-stress-scaling), (fr-de); rows divided by `U^2/L`, `U^2/H`, `U/L`; default regime
  `Fr, De = O(1)`, `e_x = O(eps)`); `ordered_truncation` sets `m.stress` from the outcome.
- `SME`, `VAM`, `MLSME`: setting `scaling=ShallowScaling(...)` replaces `moment_scaling`, the
  hydrostatic inertia drop (SME) and `apply_full_tensor`.  SME refuses a vertical order that keeps
  the vertical inertia; ML-SME sets `Pi = 0` when no vertical row keeps a stress.

**Term comparison** (rendered SystemModels, no cache; `tests/model/test_nondim_pipeline.py`):
the hand truncation of every model is reproduced **byte-identically** by one regime and order:

| model (hand rule) | regime | orders (h, v) | checked |
|---|---|---|---|
| SME `Stress.shallow` | Fr, De = O(1) (thesis) | (0, 0) | levels 0, 1, 2; 2-D level 1; SWE 1-D / 2-D + Manning |
| SME `Stress.full` | De = O(eps) (Marche (5.1): 1/Re = eps nu_0) | (1, 1) | levels 0, 1, 2; 2-D level 1 |
| SME `Stress.full` | De = O(eps^2) (Marche (3.2) -> (3.5): Re = O(1)) | (0, 0) | level 1 |
| VAM `Stress.shallow` | 1/De = O(eps), k_l = O(eps^2) | (1, 2) | levels 0, 1, 2; 2-D; unclosed (m11) |
| VAM `Stress.full` | De = O(1) or De = O(eps) | (2, 2) or (1, 2) | levels 1, 2 |
| ML-SME `Stress.shallow` | Fr, De = O(1) | (0, 0) | 2 / 3 layers, level 0 / 1, mean / upwind, 2-D |
| ML-SME `Stress.full` | De = O(eps) | (1, 1) | 2 layers, level 0 / 1, upwind stress |

The `nu d_e d_e w` term of the vertical row (dropped by hand in `apply_full_tensor`) is
`eps^4/De`, eps^2 below `nu d_e d_z u_e` in every regime: the second truncation drops exactly it
(test `test_vertical_nu_dxx_w_dropped_by_order`).  `nu d_z d_z w` (from `tau_zz = 2 rho nu d_z
w`) is `eps^2/De`, the order of `nu d_e d_z u_e`: it survives wherever the vertical stress does
(SME: through `p`, Marche (3.11); VAM: inside `P = -sigma_zz - rho g (eta - z)`).

**Where the hand truncation differs from the thesis regime (Fr, De = O(1)).**

- VAM, orders (0, 2): the vertical row keeps `-d_x tau_xz / rho = -nu d_x d_z u` (eps^2/De, the
  order of the vertical inertia VAM keeps).  VAM(1), state `[b, h, q_0, q_1, r_0, r_1, P_0, P_1]`,
  hand minus pipeline: `nonconservative_matrix[r_0, h] = -4 nu q_1 / h^2`,
  `nonconservative_matrix[r_0, q_1] = 2 nu / h`, nothing else.  The order-consistent VAM at
  De = O(1) keeps that term; the hand VAM is consistent with the weak-viscosity regime above (the
  thesis' reductions table: VAM is `1/De -> 0`), where the vertical stress is eps^3.
- SME `Stress.full` at De = O(1): keeping the in-plane stress (eps^2) consistently keeps the eps^2
  vertical inertia too, i.e. a non-hydrostatic model (orders (2, 2) are refused by SME).  Orders
  (2, 0) drop the vertical stress and with it `tau_zz`: SME(0) hand minus pipeline
  `diffusion_matrix[q_0, q_0, x, x] = 2 nu`, `[q_0, h, x, x] = -2 nu q_0 / h` and a source term,
  i.e. Marche's `2 mu grad(h div u)` of (5.19).  The hand `Stress.full` SME is the consistent
  truncation for De <= O(eps) (Marche's regime), not for De = O(1).
- ML-SME: its stand-in layer rows (hand-written) have no vertical inertia, so orders (2, 2) at
  De = O(1) reproduce `Stress.full`; (2, 0) drops the vertical stress (`Pi = 0`: no `d_e tau_ez`
  of the own layer or of the layers above), which differs at level 1, not at level 0
  (`d_z u = 0`).
- The consistency check refuses De^-1 = O(eps) with Marche's k_l = O(eps): the bed traction is then
  O(eps^-1) against the shear scale (slip length << H, a boundary layer).

**Not changed / open.**  The models keep the hand rules by default (the thesis-regime VAM
differs, above).  Bulk closures without a physical tensor (Elder, Bingham, k-epsilon, QR) are
refused by `InsertClosure` (their shallow-only closure needs a scaled form first); ML-VAM is not
wired; `small_slope_scaling` (frame slopes O(eps)) is still a separate step; SME's surface
normal-stress condition `p(eta) = tau_zz(eta)` is used when the vertical row keeps a stress (same
group, eps^2/De), not scaled as its own boundary row.

## 50. Free surface at the shore: a dry neighbour in the eta stencil (2026-10-14, dev/malpasset-pits)

**Root cause.**  The order-2 free-surface reconstruction (W = (h, eta = b + h), bed derived) took
a DRY cell's `eta = b` into the LSQ stencil and the limiter bounds of its wet neighbours.  A lake
at rest against a bank above its level then gets a free-surface slope in the shore cells.
Barth-Jespersen / minmod return phi = 0 there (the shore cell is a local minimum of eta);
Venkatakrishnan with `eps^2 = (cell size)^2` (NumPy, JAX, AMReX) does not (on Malpasset eps ~ 70 m
against metre-scale eta jumps: effectively unlimited), so `eta_f != eta_i`, the hydrostatic
reconstruction and the edge source no longer balance and the water accelerates, on Malpasset to
the mode-0 cap.  The front detector exposes it (without it MOOD redoes the inadmissible shore at
order 1).  Reproduction: `cases/pit_lake_at_rest.py` (SME(0), Gaussian pit, thesis numerics):
max |u| 0.40 m/s (quads) / 0.54 m/s (triangles) after 1 s from exact rest.

**Rule** (`reconstruction.shore_extension`, from the hydrostatic reconstruction, Audusse et al.
2004).  In the row eta a neighbour k with `h_k <= wet_dry_eps` enters cell i's stencil as
`min(eta_k, eta_i)`: the gradient difference `min(eta_k - eta_i, 0)`, the bound
`min(eta_k, eta_i)`.  The hydrostatic face state continues the wet cell's water level towards a
higher bed (`h* = max(0, eta_i - max(b_i, b_k))`); the same continuation in the stencil maps a lake
at rest to a flat eta in every wet cell for any limiter, and leaves a lower dry neighbour (the
flooding front) unchanged.  Boundary ghosts are not changed.  Printed as the 5-argument
`gradient(W, Wg, (eta, h), eps, mesh)` and the 7-argument limiter blocks.

**Backends.**  NumPy / JAX: `lsq_derivatives(..., shore)`, `ghost_bounds(..., dry)`; AMReX: `lsq`
/ `limiter_factors` with the depth row and eps.  foam (`rt.H`, new): the order-2 path is
OpenFOAM's own `leastSquares` gradient and `cellLimited` limiters, so `shore_lsq` /
`shore_limiter` are those schemes' loops (`fv::leastSquaresGrad`, `fv::cellLimitedGrad`: the
`leastSquaresVectors`, patch order, `small`, the minmod / Venkatakrishnan functions of r, k = 1)
with the one change in the eta row; every other row stays `fvc::grad`.  (Superseded by section 57: foam's order-2
gradient and limiters are the template's, with the shore form in them.)  A coupled neighbour's eta
and depth come over its processor / cyclic patch (the halo exchange of both rows), so a rank sees
the neighbour's wet/dry state.  `zoomy_foam/tests/shore_unit` (skewed 13 x 11 mesh, a cyclic pair,
walls, 26 dry cells, 3 of them across the seam): per cell against OpenFOAM's schemes on the field
with that cell's dry neighbours continued, gradient bitwise, minmod 1.1e-16, Venkatakrishnan
2.2e-16; without dry cells bitwise OpenFOAM's gradient; the h row and the halo columns unchanged.

**Measurements** (merged branch: main dadf2efd + the rule; "main" = the same cases on main's
zoomy_core 9308483 / zoomy_jax 751eb3f / zoomy_foam c93b53a / zoomy_amrex a177451).
- Pit lake (front detector on): JAX 0.40 -> 2.7e-15 (quads), 0.54 -> 1.1e-14 (triangles);
  AMReX 0.40 -> 2.4e-15, tilted on 2 ranks 5.1e-15; NumPy round-off (tests
  `zoomy_core/tests/symsolver/test_shore_lake_at_rest.py`, `zoomy_jax/tests/runtime/
  test_shore_lake_at_rest.py`).  foam (no front detector on foam yet: front_tol None) 3.0e-15 /
  1.4e-14, tilted 6.0e-15, minmod 2.8e-15, 2 and 4 ranks the same; foam was at rest before too
  (6.4e-15 / 4.9e-14): OpenFOAM's Venkatakrishnan has no eps and returns 0 at the shore minimum.
- Matrix (C3, C5, C6, C7, C9 on numpy / jax / foam / amrex, serial, mpi2, mpi4, JAX gpu1):
  0 FAIL, no row moved beyond round-off.  Against main: C3 bitwise on all four; C5 max rel
  3.6e-12 (NumPy, JAX, AMReX) / 1.2e-13 (foam): the dam-break front has its dry neighbours below
  the water, which the rule leaves alone; foam's eta-row factor is now the limiter value itself,
  not `(G.L)/(G.G)` of OpenFOAM's limited gradient (round-off).  C7 C5 parallel = serial
  <= 5.4e-15.
- Malpasset thesis configuration (JAX GPU, SME(1), 26000 triangles, t = 2000 s; vs the thesis
  run zp_l1): clamp-saturated cells 561 (main) -> 201 (pit 16, spill 125, other 60; main: pit 14,
  spill 289, other 258); transit A->B / A->C 1250 / 1300 s -> 1300 / 1300 s (observed 1140 /
  1320; snapshots every 50 s); gauge mean |err| vs observed 4.32 -> 4.23 m; shear median
  |q_1|/|q_0| at T 0.235 -> 0.218; h(T) rel L2 vs zp_l1 1.8e-2 -> 5.8e-2, inundated area
  -0.02 % -> -2.0 %, IoU 0.997 -> 0.961; mass drift <= 6e-16 both.

**Open.**  The thesis numbers move (above): the user decides before chapters change.  The
remaining 201 clamp cells are spill / flowing cells where Venkatakrishnan's `eps^2 = (cell size)^2`
leaves eta unlimited across a wet/dry front: a limiter-scale choice, not a derivation defect.
foam has no front detector (`front_cells`, dev/foam-front-diffusion), so the Malpasset numerics
do not build on foam yet.

## 51. foam: the a-priori wet/dry front detector; implicit diffusion at the FD operator's floor (2026-10-14, dev/foam-front-diffusion)

**The front detector (`front_cells`).**  `ReconstructionSpec.front_tol` (core e90f067) drives the
reconstruction only: in every stage, before the step and in every MOOD redo, a cell whose depth
extrapolated with the gradient of h and the limiter factor of eta falls below
-(1/front_tol - 1) h at a face is demoted to first order (phi = 0; Gallardo, Pares, Castro 2007,
https://doi.org/10.1016/j.jcp.2007.08.007).  It never sets a value and does not enter the CFL step.
NumPy, JAX and AMReX had the block, foam's runtime did not: every `front_tol` model failed to
build (`Solver.H: 'front_cells' was not declared in this scope`), i.e. each such matrix row would
be a FAIL.  foam `rt.H` `front_cells`: each own cell over all its faces (internal, boundary,
coupled) with foam's own G and phi, then the halo exchange: a halo column sees only the faces it
shares with this rank, so its flag comes from its owner.

The matrix had no run where the detector acts: C5 (minmod, flat bed) demotes nothing on any backend
(a bounded limiter keeps h_f >= 0 over a flat dry bed).  New C5 runs at `front_tol` 0.5 with
Venkatakrishnan (the Malpasset thesis configuration), each with (`*_front`: serial, mpi2, mpi4) and
without (`*_off`) the detector: the dam break, and Thacker's axisymmetric paraboloid bowl (SWASHES
2-D: a 1, r0 0.8, h0 0.1, [0, 4]^2, nx 48, t 0.5).  The bowl sits off the domain centre: centred,
the 2-rank cut x = 2 is a symmetry line, and foam mpi2 without the flag's exchange equalled the run
with it bitwise (the halo was never exercised).

| run | foam vs JAX, L1 rel: off -> front | detector acts (front vs off, L1 rel): JAX / foam | foam ranks vs serial (mpi2 / mpi4) | the same without the flag's exchange |
|---|---|---|---|---|
| dam | 9.3e-2 -> 1.5e-2 | 8.6e-2 / 0 | 2.5e-15 / 2.4e-15 | - |
| bowl | 4.7e-2 -> 6.3e-3 | 4.8e-2 / 2.9e-3 | 1.1e-16 / 1.1e-16 | 6.1e-6 / 7.9e-4 |

- AMReX and NumPy = JAX to round-off on all four runs (<= 4.3e-15), ranks = serial on all three
  backends; h_min = 0 everywhere.  Bowl L1(h - exact): JAX 1.4e-4 front, 2.0e-4 off; foam 1.2e-4 /
  1.1e-4.
- foam's detector does not act on the dam break: OpenFOAM's `cellLimited<Venkatakrishnan>` keeps the
  face value within the neighbour extremes (limited/allowed = (r + 2)/(r^2 + r + 2) <= 1), so over a
  flat dry bed h_f >= 0.  JAX's Venkatakrishnan (eps^2 = dx^2) overshoots where the neighbour
  differences are below dx; that overshoot was the 9.3e-2 foam-JAX difference without the detector,
  and the detector removes it.  On the bowl's sloping shore h_f = h + phi_eta G_h . r_f is bounded by
  neither limiter: both backends demote.  With the detector the foam-JAX difference is the O2 stencil
  difference of every foam O2 row (C5 n48: 1.7e-2).
- The dam break without the detector changes its mass by 1.7e-13 on JAX / NumPy / AMReX (C6,
  <= 1e-12): `clean_round_off` of the overshoot's round-off below zero; foam 1.9e-16.

**Implicit diffusion: ~15x slower than JAX.**  Serial, one core each (`taskset -c 24`), the matrix
runs unchanged, ms per step; foam's own split from `symFoam`'s summaries: right-hand side (the
explicit D + 2C with the LSQ correction; foam assembles no matrix: Jacobian-free Newton-Krylov),
linear solve (GMRES with its residual evaluations), rest (explicit step, source, aux).  JAX: the
march only (second call, no compile; C26 adaptive: over foam's 9 steps, the same CFL rule); XLA fuses
the step, no split.

| run (cells, steps) | JAX | foam before: all = rhs + linear solve + rest | foam after: all = rhs + linear solve + rest | GMRES iterations / solve | residual evaluations / step |
|---|---|---|---|---|---|
| C26 visc (20^2, 9) | 4.1 | 37.7 = 0.24 + 33.6 + 3.8 | 5.6 = 0.23 + 1.85 + 3.6 | 64 -> 6 | 369 -> 23 |
| C32 noslip n20 (8 x 20, 160) | 1.19 | 31.8 = 0.10 + 31.0 + 0.7 | 2.28 = 0.10 + 1.51 + 0.67 | 245 -> 13.2 | 727 -> 37 |
| C32 free n20 | 1.18 | 33.3 = 0.10 + 32.4 + 0.7 | 2.48 = 0.10 + 1.67 + 0.71 | 245 -> 13.6 | 727 -> 38 |
| C32 noslip_split n20 | 1.17 | 32.7 = 0.10 + 31.9 + 0.7 | 2.48 = 0.10 + 1.65 + 0.73 | 245 -> 13.4 | 727 -> 38 |
| C33 imex k2 (8, 80) | 0.12 | 0.141 (no implicit diffusion: the point-implicit source) | 0.135 | - | - |
| C34 n16, nu 1 (16^2, 1 step) | - | 12.3 = 0.22 + 8.1 + 4.0 | 5.1 = 0.24 + 0.55 + 4.3 | (123 -> 8 evaluations) | 123 -> 8 |
| C36 sme1 (16^2, 20; foam is not a matrix backend of C36) | 6.9 | 72.0 = 0.44 + 68.6 + 2.9 | 6.6 = 0.40 + 3.5 + 2.7 | 64 -> 6.4 | 324 -> 20 |

foam / JAX: before 9x (C26), 27-28x (C32), 10x (C36); after 1.4x, 1.9-2.1x, 1.0x.

- Cause: the linear solve, not assembly (nothing is assembled), the operator or the preconditioner.
  foam's Newton-Krylov takes the Jacobian action by a directional difference,
  J v = (R(x + h v) - R(x)) / h, h = sqrt(eps)(1 + |x|)/|v| (rt_common.H `newton_krylov`), whose
  round-off floor is ~sqrt(eps) relative.  The matrix cases ask `gmres_tol` 1e-13 (JAX meets it with
  the exact JVP).  C32 (diagnostic print of the first solve): GMRES's own estimate reached 1.5e-12 at
  iteration 5, every restart's true residual stayed at 1.6e-9 .. 3.0e-9, and all 60 restart cycles
  (`gmres_maxiter`) ran: 245 iterations / solve for nothing.  Same pattern C26 (true floor 1.4e-9),
  C34, C36.
- Fix (zoomy_foam `linsolve`, the plain GMRES path): after a cycle whose estimate met `ls_tol`, a true
  residual above `ls_tol` that did not halve since the previous restart ends the solve (the stall rule
  `hypre_solve` already had), with a once-warning naming the floor.  A slowly converging cycle (its
  estimate above `ls_tol`) is not affected.  Newton (`implicit_tol`) takes the same updates (2 / step;
  C36 35 -> 33 solves); the solution: after - before max rel 1e-16 (C26, C32), 2.6e-13 (C36, Newton
  1e-12), C34's L 7e-10 (= Q's round-off / dt 1e-7).
- The remaining 2x on C32: 37 residual evaluations / step at 0.037 ms (160 cells), of which the
  stall needs one more cycle (13 vs ~5 iterations / solve); the exact JVP needs printed derivatives of
  `diffusion_matrix` and the ghost kernels.

Cross-backend matrix (this worktree's store: the 2026-10-14 store of section 48 for JAX / NumPy /
AMReX; foam rerun from empty, serial / mpi2 / mpi4, on foam f37481c, jax 751eb3f, amrex a177451 and
core 31e7b85 / 9308483 (main's checkout advanced during the rerun: the nondim and ML-VAM merges); the
new C5 runs on all four backends): 917 rows (871 before), **0 FAIL**.  Changed foam rows: C28 implicit foam vs JAX (1-D
dir0 / dirL / neu / per 5.1e-11 / 8.1e-12 / 5.4e-11 / 2.7e-10 -> 6.0e-11 / 1.6e-11 / 6.0e-11 /
1.3e-10, quad per 1.5e-10 -> 3.0e-10, <= 1e-9: Newton 1e-10 at the FD floor; main's foam on the same
libraries reproduces the old values, so the stall stop moved them); C7 C26 parallel 6.6e-16 -> 5.6e-16
(round-off).  New: 46 C5 / C7 rows.  QR suite: 28 passed, 2 skipped.

Open:
- foam has no exact Jacobian action (above); AMReX's implicit diffusion (amrex::GMRES) was not
  measured here.
- No Malpasset-type run (a real DEM) in the matrix: the thesis configuration
  (`results/wavespeed_bound/run.py`) runs on JAX / GPU only.

## 52. The march raises; thin cells in the CFL step; HLL's Davis speeds without a closed form; Manning at rest (2026-10-14, dev/numerics-robust)

The four open items of section 48.

### 52a. A step that does not advance the time ends the march with an error

- **Defect.** `_time_loop` ran `While(time < time_end)`.  A NaN `dt` made `time` NaN and the loop
  ended as if done, returning the NaN state (section 48: Malpasset at t = 687 s, 939 s).  A `dt` of 0
  (the Gelfand bound and the `eigenvalues` kernel return `+inf` for a non-finite entry) never ended
  it: the Manning reproduction of 52d did not return within 300 s on main.
- **Fix** (`strategies/hyperbolic.py`, `vocabulary.check.march`).  The body takes a step only if
  `time + dt > time` (a `Cond`; else the state is kept and `going = 0`); the loop runs while `time <
  time_end` and `going`; after it `check.march(time, time_end, dt, n_step, max|Q|)` raises unless
  `time >= time_end` with a finite state (one reduction per run).  The message names the time, the
  step and its `dt`, and whether the state before it is finite ("its CFL step is not a positive
  number (a NaN or infinite wave speed)") or not ("step n - 1 produced it" / "the initial state").
  A constant-`dt` march that ends non-finite raises as well.
- **Backends.**  NumPy: `FloatingPointError`.  JAX: the check's arguments are extra outputs of the
  jitted program and `run` raises from them (also with `JAX_PLATFORMS=cuda`, which has no CPU
  device for host callbacks); where a CPU device exists a `lax.cond`-guarded debug callback raises
  during the execution (also under AD).  foam: `zs::fail` (FatalError); AMReX: `zs::fail`
  (`amrex::Abort`), and the AMR level loop checks `t + dt > t` itself; both drivers raise
  `FloatingPointError` with the runtime's line.  The AMReX `cond` unifies tuple branches entry by
  entry (expression / Field).  Legacy JAX `run_simulation` and `run_jit_steps` raise on a NaN end.
- **A NaN depth keeps its speed.**  The dt spectrum is `conditional(h <= 0, 0, e)` (52b): a NaN depth
  gives a NaN (closed form) or infinite (numerical) speed, so the march stops at the step after the
  one that produced it; `conditional(h > eps, e, 0)` gave it speed 0 and the march ran on.
- Tests: `tests/symsolver/test_march_check.py` (NaN cell: stops at t = 0, step 1; constant dt:
  non-finite end; q = 1e300; a step rule with dt = 0 or < 0 after t = 0.25: "stopped at t = 0.3 ...
  step 4"), its JAX twin (CPU; by hand also `JAX_PLATFORMS=cuda`, GPU), foam / AMReX
  `test_march_check.py` (marked large: they build).  `sp.Abs(Q, evaluate=False)`: sympy's
  `Abs.eval` walked the whole loop graph and the IMEX-SSP2 print did not return.

### 52b. The cells rule bounds every speed that moves mass

- **Derivation** (order 1, Rusanov / HLL on the Audusse states, `h*_K <= h_K`, the velocity kept):
  `F_h(U*_K, U*_N) = 1/2 h*_K (u_K.n + a_f) + 1/2 h*_N (u_N.n - a_f) <= 1/2 h*_K (|u_K| + a_f)` as
  `a_f >= |u_N.n|`, so `h_K' >= h_K (1 - dt sum_f |e_f| a_f / |K|)`: the step must bound the Riemann
  speed `a_f` of every face whose states carry mass.  A face state of a cell with `h > 0` keeps the
  cell's velocity and has `h* <= h`, so its speed is at most the cell state's; a cell with `h <= 0`
  has `h* = 0`, `q* = 0`, speed 0 in the flux.  Hence `conditional(h <= 0, 0, |lambda(U_K)|)` bounds
  every `a_f`.  A thin cell is not exempt: its velocity enters `a_f` of its faces and drains the
  neighbour through the dissipation `1/2 a_f h*_K`.
- **Before:** the gate was `h > wet_dry_eps` (default 1e-8; Malpasset 1e-3), and without a closed form
  the spectrum was taken of A_n with the KP `hinv` (`(h/eps)^2` of the speed below eps).
- **Now** (`gate_eigenvalues_dry`; `eps` default 0, an explicit `eps` keeps the old gate): 0 where
  `h <= 0`; without a closed form A_n with `hinv` -> `conditional(h > eps_KP, hinv, 1/max(1e-14, h))`
  (= 1/h: the KP value where it is 1/h, so wet cells keep their rounding).  The cells' speeds of the
  faces rule (order 2 with the edge source) and IMEX-SSP2's step read the same slot.
- **Reproduction** (`tests/symsolver/test_thin_cell_cfl.py`): SME(0) 1-D, 200 cells, order 2 without
  limiter + MOOD, forward Euler, cells rule, a 1 mm film at 300 m/s onto a dry bed.  Before: h =
  -8.0e-4 m at t = 1.0e-4 s in cell 106 (h = 2.1e-4 m), next to cell 107 with h = 7.2e-9 m < 1e-8 at
  u = 3.0e4 m/s (speed gated to 0; the Rusanov speed of their face 3e4 m/s).  After: h >= 0 at every
  write.  Order 1: positive either way (max |u| 2.1e4 m/s).  The unlimited reconstruction then grows
  thin-cell velocities to 1.7e5 m/s and the step collapses (dt 2e-19 at t = 3.6e-3 s: check.march
  raises; before: the same at t = 2.7e-3 s, after h < 0, with max |Q| 9e11).
- **Malpasset** (thesis configuration, cells rule, JAX on GPU 2, to 2000 s, `results/wavespeed_bound/
  run.py numfix_cells`): h_min 0 at all 2001 writes, volume drift 2.4e-15.  Against section 48's
  cells run: wet area max rel 1.2e-3, max-depth envelope L1 rel 9.8e-4, inundated area 17.023 /
  17.030 km^2 (section 48 faces vs cells 1.3e-3 / 9.9e-4, main vs section 48 1.1e-3 / 8.6e-4); max
  alpha_1 (h > 1e-3 m) 982 m/s (section 48: cells 265, faces 786).  Steps to 300 s
  (`results/numerics_robust/malpasset_steps.py`): 32,090 against
  27,252 with the former gate (+18 %: cells below 1e-3 m now bound the step); 2000 s wall 671 s
  against 478 s (the GPU shared with another job).

### 52c. HLL's Davis speeds without a closed-form spectrum

- **Bound.**  For any real c, `rho(A + c I) >= max_i |lambda_i + c| >= max Re lambda + c`, so
  `s_R = min(G(A + a I) - a, a) >= max Re lambda` and `s_L = max(a - G(a I - A), -a) <= min Re
  lambda`, G the Gelfand bound (section 48), `a` the face's Rusanov speed, the max over both face
  states (`Numerics._gelfand_extremes`; branch-free, every backend).  The shifted matrices are taken
  in the face's normal frame, `T A_n T^T`, each horizontal vector `(q_x, q_y)` rotated to `(q.n, q.t)`
  (a similarity): there a wall's reflected state gives a signed permutation of the inner state's
  matrix, the pair is symmetric and no mass crosses the wall.  In the lab frame it was not (G is not
  rotation invariant): mass drift 4.6e-9 on a disc (oblique walls), up to 1.6e-8 on Malpasset; now
  1.3e-16 / 3e-16 (`test_wall_pair_is_symmetric_at_an_oblique_wall`).
- **Rigour** (`gates/gate50_davis_bounds.py`): 128,000 random matrices (gate 49's sets; NumPy and JAX
  kernels) and 440,000 model matrices (the production `HLL.wave_speed_bounds`: SME(0..3) 1-D / 2-D,
  VAM(1), ML-SME; wet, thin and dry cells, Audusse faces): `s_L <= min Re lambda`, `s_R >= max Re
  lambda` everywhere, no 40-digit recheck needed.  Overestimate / a, median / max: model matrices
  0.2 - 0.7 % / 8.9 %; random up to 67 % (non-normal triangular, complex spectra).  HLL fan
  `(s_R - s_L) / 2a` on the model states, median 0.03 (thinned faces) .. 0.66 (Rusanov: 1).
- **The face speed** `max(|s_L|, |s_R|)` is wrapped in `Abs`: at a dry face the new pair gave -0 and
  the faces rule a step of -inf.
- **Positivity at order 1** (`results/numerics_robust/malpasset_order1.py`: Malpasset SME(1) mixing
  length, `hr`, point-implicit source, faces rule, CFL 0.4, to 120 s, JAX GPU; released water = water in the initially dry cells): closed form
  4,553,098 m^3; numerical spectrum with the Davis bounds 4,553,170 (+1.6e-5); with the former
  Rusanov pair 4,683,207 (+2.9 %, the +2.5 % of section 48).  h_min 0 at every write (1 s) in all
  three, mass to 3e-16.  Cost: 298 s against 68 s (six Gelfand calls per face instead of two).  Unit
  test: SME(0) 1-D dam break onto a bed step, order 1: h >= 0, mass to 1e-12.

### 52d. Manning friction at rest

- **Expression.**  The friction rows `-g n^2 hinv^(4/3) q_x sqrt(hinv^2 q_x^2 + hinv^2 q_y^2)`; their
  derivative contains `hinv^2 q_x^2 / sqrt(...)`, 0/0 at q = 0.  JAX's implicit source (Lie,
  IMEX-SSP2) takes the Newton Jacobian by AD: `sqrt'(0) * 0 = inf * 0 = NaN` in every cell exactly at
  rest once the Newton iterates (another cell moving).  NumPy (central differences) and foam /
  AMReX (one-sided differences) are finite there.  The point-implicit source
  (`SourceTreatment.linearized`, the Malpasset GUI session's) prints `sp.diff` of the source: NaN at
  rest on every backend.
- **Fix** (NSM operation `norm_derivative_at_zero`, applied by default before the linearization):
  every `S**r` in the source slots with `S >= 0` an even-power factor times a positive semidefinite
  quadratic form (a squared norm) and rational `r >= 1/2` not an integer becomes `conditional(S > 0,
  conditional(S > 0, S, 1)**r, S)`: the same value (bitwise), and the derivative 0 at `S = 0` (S has
  its minimum there, `dS = 0`): the derivative of `S**r` for `r > 1/2`, the least-norm subgradient
  for `r = 1/2`.  No eps, the model is untouched.  `conditional` now collapses on a boolean
  condition (after `subs`).
- **Reproduction** (C14's bump, Manning n 0.5, h = 1, a patch q_x = 2 at x < 3 next to cells at rest,
  to 0.05 s): before, JAX Lie: NaN after step 1 (check.march: "step 2 ... step 1 produced it"); JAX
  IMEX-SSP2 on main: every cell NaN at t_end, no error; point-implicit: NaN on NumPy, foam and AMReX.
  After: finite on every backend, cells still exactly at rest.  Tests:
  `tests/symsolver/test_norm_derivative.py` (the value bitwise equal on random states and at rest,
  the symbolic Jacobian finite at rest and NaN without the operation, the point-implicit run),
  `zoomy_jax/tests/symsolver/test_manning_at_rest.py` (Lie, IMEX-SSP2).  Golden
  `n01_swe2d_order2` re-blessed (the source rows).

### 52e. Matrix and tests

- **Matrix** (`matrix/README.md`, results 2026-10-14 dev/numerics-robust): every serial run rerun on
  NumPy, JAX, foam and AMReX (830 runs): 754 bitwise equal to the base store, 75 at round-off (JAX
  62: XLA compiles the step inside the loop's `cond` branch, and the Manning Jacobian by AD through
  the `conditional`; NumPy / foam C12 `bump_o2`: the re-expressed A_n of the cells' speed bound,
  bitwise equal without that commit; <= 7.1e-15, C10 `s2_t2_n120` JAX 2.1e-13 through the pressure
  GMRES).  Table: 872 rows, 801 PASS, 1 FAIL (base: 871, 801, 0); 6 rows moved in the third digit
  (C1 / C12 numpy = JAX, C6 C12 numpy mass, C7 / C9 JAX C1 o2_n40, C9 C26; the C4 thresholds, 10 x
  the JAX `_pert` sensitivity, in the fourth digit).  No matrix case has a
  thin cell (0 < h <= 1e-8), HLL without a closed form or a cell at rest under Manning.  The FAIL
  is C17 `dry` (AMReX AMR, regrid on the dry front), a new row: it did not return in the matrix
  process before and its gate metric was NaN on main; it now raises at step 85 (t = 0.0466 s), the
  level-0 step 84 having produced NaN next to cells with h = -7.8e-12 (below the round-off
  tolerance) and h ~ 1e-54 .. 1e-154 left by the AMR coarse-fine transfer.
- **Tests.**  zoomy_core: 376 small tests pass (new: `test_march_check`, `test_thin_cell_cfl`,
  `test_davis_bounds`, `test_norm_derivative`); goldens n01-n03, s02, s03 re-blessed (the
  `eigenvalues_cfl` rows; n01 also the source rows), rederive tier 40 pass + 1 known failure
  (`test_rederive_cross_process_disk_hit_and_prebuilt`).  zoomy_jax: `tests/symsolver` (march check,
  Manning at rest), legacy AD tests (`test_ad`, `test_ad_chorin`) pass.  foam / AMReX:
  `test_march_check` (large).  QR suite (`tests/qr_cases -m qr`): 27 pass, 3 skipped (no thesis
  checkout: the manifest snapshot; the jax / AMReX sessions in Pyodide).

Open:
- C17 `dry` (above): the AMR coarse-fine transfer is not positivity preserving (h < -tol); the
  march now stops where it produced the NaN.
- The cells rule takes `eig` (a host callback on a GPU) for models without a closed form: the
  Malpasset numerical-spectrum session under the cells rule did not reach 120 s in 25 min on the GPU
  (faces rule: 5 min).  The Gelfand bound would unify it with the Riemann speeds (wet-cell steps up
  to 6 % smaller).
- Unlimited order 2 grows thin-cell velocities without bound (the film of 52b): h stays >= 0, the
  step collapses and the march raises.
- HLL's normal frame needs the model's vector groups (`derive_momentum_field_indices`) and 2
  horizontal directions; otherwise the lab frame (symmetric only for walls along the axes).

## 53. IMEX-SSP2 with the implicit diffusion; diffusion on AMReX AMR levels (2026-10-14, dev/solver-combos)

**IMEX-SSP2 + implicit diffusion** (was `NotImplementedError` in `imex_step`).  The implicit part
g of the tableau (Pareschi, Russo 2005, Table II) is the implicit source plus the two-point part
D of `diffusion_matrix`; the rest of that diffusion's face flux (`diffusion_correction`: the
tangential / non-orthogonal part, zero in 1-D and on Cartesian faces without mixed entries) is in
the explicit part f.  Stage i: R_i = Q + sum_{j<i} (At_ij F_j + A_ij G_j), then one Newton solve
over the mesh of X - R_i - A_ii dt (D(X) + S(X)) = 0 (`implicit_stage`; the pointwise aux follows
the iterate, the derivative aux is frozen at R_i), F_i = dt f(Y_i) (the explicit residual with
`split_diffusion`), G_i = dt (S(Y_i) + D(Y_i)).  Without diffusion the stage is the per-cell root
as before (printed graph unchanged).  New vocabulary: `diffusion_tpfa`, `diffusion_correction`
(their sum is `diffusion`) and `implicit_solve` (a Newton whose residual couples face neighbours:
NumPy sparse LU of the coloured FD Jacobian, JAX `newton_root` + `linsolve` (exact jvp), foam /
AMReX Jacobian-free Newton-Krylov).  The Chorin template's projected IMEX-SSP2 stages take the same
implicit stage.  JAX: the TPFA operator is built under `ensure_compile_time_eval` (first used
inside a traced solve it leaked a tracer).

- Gate 52 (`gates/gate52_imex_diffusion.py`; JAX and NumPy identical to the digits shown), errors
  against the exact solution of the semi-discrete system at dt = 0.0125 / 2^k, k = 0..3:

| problem | errors k = 0..3 | order 2 -> 3 |
|---|---|---|
| heat 1-D periodic, nu dt / dx^2 = 12.8 at k = 0 | 2.03e-2 / 4.93e-3 / 1.22e-3 / 3.03e-4 | 2.01 |
| heat 1-D Dirichlet | 3.07e-4 / 7.62e-5 / 1.90e-5 / 4.74e-6 | 2.00 |
| heat quads periodic / Dirichlet | 1.68e-1 ... 2.43e-3 / 2.53e-3 ... 3.80e-5 | 2.02 / 2.00 |
| advection-diffusion (walls), nu 0.02: ssp2 / lie1 / lie2 | 3.10e-2 ... 6.24e-4 / 9.68e-2 ... 9.62e-3 / 1.03e-1 ... 1.28e-2 | 1.96 / 1.04 / 1.01 |
| the same, nu 0.2 (stiff): ssp2 / lie1 / lie2 | 2.59e-2 ... 1.28e-3 / 4.38e-2 ... 5.12e-3 / 4.29e-2 ... 5.31e-3 | 1.65 / 1.01 / 1.00 |

  Every advection-diffusion run equals the scheme written in matrix form (L_a, L_d probed on the
  NumPy backend; the stage solves (I - A_ii dt L_d) Y_i = R_i, Lie's Crank-Nicolson): JAX
  <= 3.6e-14, NumPy <= 4.2e-12.  The stiff ssp2 order (1.23 / 1.46 / 1.65) is IMEX-SSP2's own:
  the matrix form gives the same numbers and reaches 1.80 / 1.89 / 1.94 at dt / 16 .. / 64 (the
  implicit DIRK is not stiffly accurate: its last row (1 - 2g, g) is not w = (1/2, 1/2), so the
  stiff modes of the stages are damped but not exact; order reduction in the stiff regime, a
  property of the tableau, not of the stage solve); periodic, the two operators are
  circulant and commute, so the walls are needed to see Lie's order 1.
- C32's viscous shear decay with IMEX-SSP2 (JAX, ny 20): L_inf vs exact 3.37e-4 (= the Lie run's
  spatial error); time order (Richardson, dt / 2^k) 2.0001 / 2.0001.
- Chorin, VAM(1) + Newtonian Stress.full (the predictor's diffusion_matrix, 34 nonzeros), C10's
  wave, nu 0.01, increment projection, dt 8e-3 / 2^k, NumPy: projected IMEX-SSP2 stages L1(h)
  Richardson 1.96 / 1.96 (q_x_0: 1.93 / 1.95), Lie per stage (time_order 2) 1.14 / 0.99 (0.69 /
  0.71).  At nu 0.1 the Lie run (unchanged code) also fails (NumPy: singular cell Jacobian of the
  implicit source after a few steps; JAX: NaN): not this change.
- C++ = JAX (gate 41's heat equation, IMEX-SSP2, nx 16): AMReX 1-D 3.6e-10 (Newton 1e-10), quads
  1.1e-15; foam quads 9.0e-16.

**AMR + diffusion (AMReX).**  `diffusion_tpfa` refused AMR (`M.log`: diffusive fluxes not
refluxed).  Now every diffusion block logs its face fluxes on an AMR level: the explicit ones (the
`diffusion` block in an SSP-RK stage, the IMEX stage's `diffusion_correction` and its G_i
`diffusion_tpfa`) are added to the current stage's (Wm, Wp) entry, so they carry the stage's RK
weight (IMEX-SSP2: wt = w = 1/2, asserted in `_amr_procedures`); Crank-Nicolson's go to `M.dlog`
with their share of dt (1/2 two-point at the start, 1 the deferred correction, 1/2 two-point at
the end); residual evaluations inside implicit solves log nothing (`NoLog`).  `sync` integrates
both into the flux register (Berger, Colella 1989).  New kernels `tpfa_faces` /
`correction_faces` (the cell kernels unchanged; the correction's gradients are computed in the
ghost layer instead of halo-filled: an AMR halo fills only state / aux rows).  Two defects of the
AMR loop on the way: the levels' linear solves had no mesh (`the_mesh()` was null: every implicit
solve on AMR crashed), and `dt_of` ignored a constant `dt` (a heat equation, wave speed 0, took
one step of t_end).

- Gate 53 (`gates/gate53_amrex_amr_diffusion.py`; heat equation nu 0.05, base 16^2 + one level,
  a bump across the interface of the static patch [0.25, 0.75]^2, t 0.1; Neumann and periodic):

| path | covering level vs fine single | mass, reflux / no reflux | L1(AMR - fine) / L1(coarse - fine) |
|---|---|---|---|
| explicit slot, SSP-RK, dt 3.2e-3 | 0 / 0 | 2.1e-16 / 2.7e-4 | 0.37 / 0.38 |
| Lie (Crank-Nicolson), dt 1e-2 | 0 / 0 | 2.1e-16, 0 / 3.7e-4 | 0.59 / 0.59 |
| IMEX-SSP2, dt 1e-2 | 4.4e-16 / 5.9e-16 | 0 / 3.7e-4 | 0.59 / 0.59 |

  (Neumann / periodic.)  The implicit paths' larger ratio is a time lag at the interface, not the
  refluxing: the fine halo takes the coarse level's state after its step in every stage (the
  AMR loop's one dt, coarse first); Lie at dt / 2, / 4, / 8: 0.45 / 0.40 / 0.38, toward the explicit
  0.37 (spatial).  C32's viscous SME (Stress.full: state-dependent, mixed A; x-varying u0), Lie,
  covering level vs fine single: 0 (the momentum rows carry the bed's nonconservative product and
  are not refluxed, as without diffusion).

**Matrix** (serial, mpi2 / mpi4 where configured; no GPU; scratch store: main's store with C17, C26,
C28, C31-C34, C36 rerun and C37 new; 478 rows of the touched cases): **0 FAIL** apart from C17
`dry` (below).  C26, C28, C31, C32, C33, C34, C36: every row equals main's table to the printed
digits (the source-only IMEX-SSP2 path and the Lie paths are unchanged; C31 VAM ssp2 1.96, SWE ssp2
2.07).  C37: heat 2.00-2.02, advection-diffusion ssp2 1.96 / lie1 1.04 / lie2 1.01, every run =
the matrix-form scheme (JAX <= 3.6e-14, NumPy <= 4.2e-12), shear decay ny 20 / 40 / 80 orders 1.94
(the C32 numbers), time order 2.000, VAM(1) Stress.full Chorin ssp2 1.96 vs Lie per stage 0.99
(JAX = NumPy); foam / AMReX vs JAX <= 1.0e-14 (heat, advection-diffusion, VAM 1.9e-14; AMReX shear
4.4e-16, foam shear ratio 0.014 - 0.20 as C32); mpi2 = serial: JAX 1.3e-13, foam 8.0e-15, AMReX
7.1e-15.  C17 `diff_*`: as gate 53.  zoomy_core: 362 passed (2 new: `tests/symsolver/
test_imex_diffusion.py`); QR suite 27 passed, 3 skipped (the backend-only sessions).

**C17 `dry` (pre-existing, not changed here).**  The regrid case had no run in main's store; run
now, its final state has NaN in 796 coarse cells (mass drift NaN; `h_min` passes because it takes
`nanmin`), identical with main's libraries (core 082acb3, amrex a177451).  No NaN to t 0.044 (the
fine level covers the domain), NaN from a step after the regrid that shrinks the fine level
(t 0.046 -> 0.048 with the matrix's steps; with `dt_write` 5e-4 only in the last step): erratic in
the step sequence.  Dry coarse cells carry q != 0 at h = 0 (|q| / h up to 1e292).  Not diagnosed
further (outside these items).

**Open.**  (1) The AMR fine halo takes the coarse level after its step in every stage (time lag
O(dt) at the interface, above); time interpolation of the coarse state needs the stage time in
the halo fill.  (2) VAM(1) Stress.full at nu 0.1 on C10's wave fails with the unchanged Lie path
too (NumPy singular cell Jacobian, JAX NaN).  (3) GPU configs not rerun (C9: C26 `visc`, C17 gpu1).
(4) preCICE + Chorin: not implemented, plan in `STATUS.md` (the pressure solve couples both
domains: overlapping Schwarz inside a preCICE implicit-coupling window).

## 54. The NumPy backend distributed (MPI) (2026-10-14, dev/numpy-mpi)

`SymbolicHyperbolicSolverNumpy(comm=MPI.COMM_WORLD)` / `SymbolicChorinSolverNumpy(comm=...)`: the
JAX API (`setup` takes the global mesh, returns the rank's piece; `partition.gather`), mpi4py only.

**Design.**

- One partition, one template.  The piece and its exchange lists are JAX's
  (`mesh.distributed.distribute`, `Partition`: owned cells, two halo layers, cut faces tagged
  `halo`); the template's own `halo` / `reduce_*` blocks carry the rest.  The rule which rank
  counts a face or a cell moved from `zoomy_jax.symsolver.parallel.blocks` to
  `Partition.owned` (core); JAX uses it, bitwise unchanged (matrix C1, C8, C13, C21 mpi2).
- `numpy_backend/parallel.py`, `Distributed`: `halo` = Isend / Irecv of the exchange lists for any
  `(..., cells)` array -- the state after every stage, the aux and derivative aux after `full_aux`,
  the LSQ gradients of the diffusion correction, the pressure iterate inside its residual; the
  identity inside `no_halo` (assembly by colouring: each rank probes its own rows alone, the halo
  columns numbered by their owners, no collective inside the probes).  Cut faces: no BC kernel,
  no diffusion flux (as JAX).
- MOOD flags are not exchanged: `inadmissible` reads the halo'd candidate state, so every copy of
  a cell flags as its owner (C5 and the MOOD test = serial to 5e-15).
- Reductions: min / max gathered, then `np.min` / `np.max`: a NaN on any rank is the result on
  every rank (MPI.MIN on MPICH 4.3.2 dropped it in the test, `mpi_min_swallows`); sums gathered
  and added in rank order, the same bits on every rank, so the decisions taken on them (Newton
  stop, lagged-Jacobian refactorization, the accept gate, FGMRES exit) agree -- no rank-local
  branch in front of a collective.  `solver.integral(Q)` (mass) is a collective.
- IMEX source: the cell-local Newton runs on the owned and the halo cells (copies of the same
  inputs), its norm global: the serial number of updates.
- Implicit diffusion: the Crank-Nicolson residual halos its iterate; the Jacobian's owned rows by
  the coloured central differences (halo off), lagged as the serial `splu` (new dt, or an update
  reducing the global residual less than 10x).
- Pressure (`linsolve`): the affine operator on the owned unknowns with one global scale
  `max|x|`; accept gate on the global residual.
- Output / restart: gathered to rank 0, the serial file (h5py here is not built with MPI; with it,
  `Partition.write_fields`, as JAX); restart files read per rank (`scatter`).
- mpi4py / petsc4py are imported only by a distributed run: the serial backend and Pyodide run
  without them (test with both blocked; the QR C1 check of optional imports).

**Distributed linear solves** (`mpi_linear_solver = MpiLinearSolver.auto | petsc | krylov`; the
env has PETSc 3.24.5 + petsc4py on MPICH 4.3.2 with MUMPS, SuperLU_dist, hypre, so `auto` = PETSc):

| system | PETSc (petsc4py) | `krylov` (mpi4py only) |
|---|---|---|
| pressure, `gmres` | KSP GMRES on a shell matrix (matrix-free), no preconditioner: the serial scipy GMRES | FGMRES, CGS2, two gathers per iteration |
| pressure, `pyamg` / `auto` | owned rows assembled by colouring, left-scaled by the inverse cell blocks (as `host_amg_solve`), FGMRES + BoomerAMG, hierarchy lagged (`amg_rebuild`, `amg_growth`) | FGMRES + the rank's SA V-cycle (block Jacobi) |
| implicit diffusion | MUMPS LU of the assembled Jacobian (SuperLU_dist, else GMRES 1e-14) | FGMRES to 1e-14, block-Jacobi LU |

PETSc's GMRES default (classical Gram-Schmidt, never refined) took 4251 iterations per VAM
pressure solve against 910 refined (9k cells): the KSPs set `refine_always` (prefix
`zoomy_numpy_`, overridable in `PETSC_OPTIONS`).

**Matrix** (store = the main results, core 082acb3, plus numpy; all 228 numpy serial runs rerun:
bitwise = before): numpy `mpi2` / `mpi4` added where JAX has them (C1, C5, C8, C13, C14, C21,
C26; C35 mpi2), 0 FAIL, the other backends' rows unchanged.

| case | mpi2 vs serial | mpi4 vs serial | numpy mpi2 / mpi4 vs JAX serial |
|---|---|---|---|
| C1 o2_n40 (adaptive dt, t 2.5) | 1.4e-13 | 1.4e-13 (JAX: 1.4e-13) | 1.4e-13 |
| C5 dry dam break, MOOD | 5.4e-15 | 6.6e-15 | 3.4e-15 / 6.5e-15 |
| C8 VAM Chorin (PETSc GMRES) | 2.8e-11 | 3.0e-11 (tol 1e-9) | 5.9e-11 |
| C13 periodic O1 / O2 | 6.4e-16 | 5.8e-16 | 2.2e-15 |
| C14 IMEX Manning | 4.2e-16 | 5.1e-16 | 1.1e-15 |
| C21 Euler 3-D | <= 9.8e-16 | <= 6.7e-16 | 1.2e-15 |
| C26 implicit diffusion (MUMPS) | 5.6e-16 | 6.2e-16 | 2.3e-14 |
| C35 restart (mpi2) | restarted = continuous: 0 (bitwise) | | |

`krylov` (MATRIX_NUMPY_MPI_SOLVER=krylov): C8 2.0e-11 / 6.1e-11, C26 6.1e-16 / 6.2e-16.  `pyamg`
pressure on C8 vs serial pyamg: PETSc BoomerAMG 7.6e-11 / 6.6e-11, `krylov` per-rank SA
3.0e-11 / 1.6e-11.  Tests (`tests/symsolver/test_numpy_mpi.py`, 3 ranks): O2 + MOOD walls and O2
periodic = serial 4e-16 / 7e-16, mass integral, halo copies = owners, NaN min, `linsolve`
PETSc / krylov = serial 5e-16, HDF5 output = the serial file.

**Timings** (`gates/bench_numpy_parallel.py`, constant dt, cores 16-23 of a shared node, ms per
step; setup 9-35 s, serial Python on every rank):

| run | 1 rank | 2 | 4 | 8 |
|---|---|---|---|---|
| SWE O2 + MOOD, 200^2 | 716 | 434 | 213 | 109 (6.6x) |
| SWE O2 + MOOD, 400^2 | 2382 | | 869 | 459 (5.2x) |
| VAM(1) Chorin 300 x 30, `gmres` (PETSc / krylov) | 9668 | 6929 / 6004 | 3346 / 3639 | 2104 / 2573 |
| VAM(1) Chorin 300 x 30, `pyamg` (PETSc BoomerAMG / per-rank SA) | 438 | 213 / 277 | 133 / 289 | 98 / 219 |
| implicit diffusion 80^2 (MUMPS / krylov) | 418 | 233 / 276 | 128 / 148 | 88 / 95 |

- The unpreconditioned GMRES pressure is ~900 iterations per solve at 9k cells, whatever the
  ranks: use `pyamg` (BoomerAMG distributed: 17-20 iterations on 2-8 ranks).  The per-rank SA
  block preconditioner stops scaling at 4 ranks (no coarse correction, unlike JAX's).
- Open: setup is serial per rank (the global mesh, as JAX for a mesh object; a mesh file reads
  only the rank's cells); h5py without MPI gathers output on rank 0.

**With section 53 (integ/20261014).**  `implicit_solve` (one Newton over the mesh for an IMEX-SSP2
stage: the implicit source + the two-point diffusion) is distributed as the implicit diffusion:
the stage residual halos its iterate, the coloured FD Jacobian's owned rows are probed with the
halo off, each update a parallel LU (`_Factor`: scipy `splu` serially, `Distributed.lu`
distributed), the norm global.  NumPy serial: all 288 runs bitwise = integ's own libraries.
Ranks vs serial: C37 `shear_noslip_n20` 7.1e-15, `heat_quad_per_k0` 2.5e-12 (JAX 1.3e-13): the
Newton stops at `||R||_2 <= 1e-12` with max|Q| = 0.016, and the FD Jacobian (central
differences, round-off / h ~ 1e-11) leaves a different sub-tolerance residual on each
partition; at `implicit_tol` 1e-14 ranks = serial 8.0e-15.  The C7 threshold of numpy C37 is
therefore the Newton tolerance, 1e-10 (`NUMPY_NEWTON_C37`).  numpy = JAX: C37 4.3e-12 serial,
1.9e-12 mpi2; C32 1.6e-12; C26 2.3e-14 on every config.  Test: IMEX-SSP2 stiff
advection-diffusion on 3 ranks = serial 2e-13 (PETSc) / 7e-13 (krylov).

## 55. AMReX AMR by composite stages; the refluxed velocity; GPU streams, image, hypre smoother (2026-10-15, dev/amrex-amr-gpu)

### 55a. C17 `dry`: the reflux of a stage the fine level never saw

- **Where the depth goes negative** (depth checks after every transfer of the AMR loop: step, halo
  fill, reflux, average-down, regrid).  Interpolation and average-down keep h >= 0 (`cell_cons_interp`
  bounds every fine value by the coarse neighbours'); the reflux does not.  Step 83 (t 0.0456): the
  uncovered coarse cell (7, 3) next to the fine level has h 1.7e-22 at t^n, 1.04e-8 after its coarse
  step (inflow, through the coarse-fine face, from the covered cell, whose coarse value averages
  wetter fine cells), -7.75e-12 after the reflux (8 symmetric cells); the next coarse step makes NaN.
- **Cause.**  The fine level stepped after the coarse level's whole step, so its stage-2 halo held
  the coarse state at t^{n+1}: water the coarse step had moved into the cell across the
  coarse-fine face, which the reflux then replaces by the fine fluxes, left the halo into the fine
  level.  For one stage (`time_order` 1) the end-of-step reflux is the composite forward Euler step
  and h >= 0 (mass 0); for SSP-RK2 it is not (also at order 1: base 64, NaN at step 860).
- **Fix (composite stages).**  The template prints the SSP-RK step cut into its stages (core
  `_amr_stages`: `amr_stage<i>` = `a_i Q + (1 - a_i)(Y + dt L(Y))`, MOOD's `_cells` / `_o1`
  variants, `amr_bad`, `amr_implicit` (Lie), `amr_post`); the AMR loop runs stage i on every level
  coarse to fine, refluxes the stage's face terms (share `1 - a_i`) and averages down, then fills
  the finer halos from the coarser level's state of the same stage.  Each stage is then a forward
  Euler step of the scheme on the composite grid (a coarse cell next to the fine level updated with
  the fine faces' fluxes of the same stage state).  MOOD judges the composite: the inadmissible
  cells of every level (covered coarse cells excluded; a coarse cell's demotion carried into the
  fine ghosts inside it), then every cell at first order with piecewise-constant halos.  That last
  tier is the first-order scheme on the composite grid: a fine face's ghost state is the coarse
  cell's own, so h' >= h_K (1 - dt sum_f |f| a_f / |K|) >= 0 under the levels' CFL step (52b; a
  fine face of half the area under the fine level's own CFL bound).  IMEX-SSP2 keeps the whole-step
  loop (55d).
- C17 `dry`: mass 0, h_min 0 (91 steps); before: NaN at step 85.

### 55b. The long-run hang: the reflux moved the depth, not the momentum

- Only the conservative rows (b, h) are refluxed (AMREX.md round 5: refluxing the SWE momentum broke
  the lake at rest).  A thin refluxed cell kept its momentum while its depth dropped: q/h up to 1e136
  in the cells, and the cells rule (52b, 1/h for h < 1e-8) took that velocity into the step.  dry
  at base 64 (forward Euler, t 0.1): 9,556 steps, 539 s; the fine single level 386 steps.
- **Fix** (`amr_keep`, applied to the cells whose refluxed rows changed): a refluxed cell keeps the
  reconstruction variables of the rows not refluxed (its velocity), the momentum follows the
  corrected depth.  The lake at rest is untouched (no mass flux, no cell changes), a cell refluxed
  to h = 0 gets q = 0.  The momentum's pressure / NCP part of the flux difference stays unrefluxed,
  as before.
- dry base 64, t 0.1: 364 steps (time_order 1), 363 (2); new C17 row `dry_long` (steps / the fine
  single level's: 0.94, threshold 1.5).  Order 1, base 32, t 0.3: 1,612 -> 320 steps (SSP-RK2).

### 55c. GPU: 3 boxes on 2 GPUs, the image, hypre's smoother

- **Stall** (L2P 1M cells, `max_grid_size` 1067, 2 GPUs).  The first pressure solve's FlexGMRES
  iterated on NaN (seconds per iteration); the assembled matrix had 100,107 singular 2 x 2 cell
  blocks (row sums -1e25) in the second box of the rank with two boxes, a different count each run.
  The march runs in a no-sync region (no synchronisation at the end of MFIter loops); AMReX does
  not fuse its MultiFab operations (setVal, Copy, ...) on a rank with 2 to 8 boxes larger than 65^3
  cells, so they run box by box on separate streams and raced the fused kernels that follow on
  stream 0 (here: the probe vectors of the assembly).  2, 8, 9 boxes (fused, or one box per rank)
  and the 10k / 256k-cell layouts never met it.  `amrex.max_gpu_streams = 1` -> correct matrix, 20
  iterations.  **Fix:** symAmrex defaults to one GPU stream (kernels are one launch per box; no
  loss measured).
- **The image** (`zoomy_amrex.sif`, one image for CPU and CUDA) has no libnvJitLink, which
  libcusparse needs and its apt package does not pull in: every hypre-CUDA build failed to link
  (C8 gpu1 / gpu2).  Recipes (`Dockerfile`, `.def`) install `libnvjitlink-12-6`; verified with
  this image plus that package (`containers/zoomy_amrex/zoomy_amrex.sif` of this worktree).  The
  published image needs the rebuild.
- **GPU selection**: `apptainer --cleanenv` dropped `CUDA_VISIBLE_DEVICES`: every GPU run took GPU
  0 (and 1).  The driver passes it (`--env`).
- **hypre iterations** (38 GPU vs 19 CPU): the GPU smoother was l1-Jacobi, the CPU's hybrid
  symmetric Gauss-Seidel; HMIS has no device path in hypre (PMIS on the GPU).  With the CPU's
  smoother on the GPU (triangular solves of the local block): 20 iterations, but L2P solve
  1.88 s per solve against 0.36 s (2 GPUs; 1 GPU 2.38 s against 0.25 s).  Other device smoothers
  (2 GPUs, iterations / ms per solve): two-stage GS (11) 30.6 / 407, (12) 29 / 433, Chebyshev (16)
  23 / 281 (setup 218 instead of 140), hybrid GS forward (3) 28 / 1417.

### 55d. AMR implicit time lag

- Lie (gate 53, `impl`): the composite stages give the fine halo the coarse stage state in the
  explicit stages and the Crank-Nicolson explicit half; L1(AMR - fine) / L1(coarse - fine) 0.59 ->
  0.376 (Neumann) / 0.388 (periodic), at the explicit path's 0.367 / 0.380 (spatial).
- IMEX-SSP2 (0.59, unchanged): its stages are implicit solves; cutting them needs per stage the
  level's implicit solve with the halo from the coarser level's stage state, and the stage's
  F_i and G_i corrected separately (the later stages weight them by At_ij and A_ij, the end by wt =
  w): two face logs per stage and an average-down of the increments.  Not done (~150 lines).

**Matrix** (main's store with C17 all configs, C5 / C8 / C26 / C37 AMReX on every config and their
JAX serial references rerun; the `.sif` above): every AMReX row of C5, C6, C7, C8, C9, C17, C26,
C37 passes (0 FAIL).  Changed AMReX rows: C17 `dry` (new: mass 0, h_min 0; mpi2 / gpu1 = serial),
`dry_long` (new), `diff_*_impl` 0.59 -> 0.376 / 0.388 (55d), box energy 2.2e-16 -> 0, `boxnr`
(the no-reflux control) mass / energy 5.91e-4 / 1.44e-3 -> 6.15e-4 / 1.46e-3; wave16 / wave32 L1
5.27e-3 / 9.15e-4 to the printed digits (order 2.53); C9 C8 `vam` GPU vs CPU 7.2e-12 -> 8.8e-12
(the smoother); covering cases, lake, `diff_*` covering and mass rows unchanged.  The C5 / C6 / C7
/ C9 C14 rows that moved in the last digits are integ's (single-level AMReX, untouched here).  The
table's 13 non-AMReX FAILs are the mixed store (NumPy / foam / JAX ranks from main against integ's
JAX serial: C5 1.1e-2, C8 0.47, C26 2.3e-3); `table.md` not committed.  zoomy_core 397 passed;
QR 27 passed, 3 skipped.

domains: overlapping Schwarz inside a preCICE implicit-coupling window).  Done: section 56.

## 56. The Chorin march coupled over preCICE: overlapping Schwarz inside an implicit coupling window (2026-10-14, dev/precice-chorin)

`ChorinTemplate.precice_march` was refused (`NotImplementedError`): a Chorin step's pressure solve
couples every cell of both domains, so no single exchange of interface states reproduces it (the
hyperbolic coupling exchanges the O1 states next to the coupled faces once per window, C16).

**Design.**  Each participant's mesh reaches W cell rings across the interface (the overlap; its
outer boundary carries the coupled tag; `zoomy_core.mesh.overlap.overlap_layers`: the W face rings
from that tag, the next W rings of own cells the mirror).  One step per window of an implicit
preCICE scheme; every coupling iteration runs the whole step from the window start, which the
template keeps in the loop carry (the checkpoint):

- the overlap holds the peer's state at the window start (`couple.read`, relative read time 0:
  preCICE's window-start sample, the converged previous window);
- predictor and aux run over the overlap too (redundantly: its outer rings, next to the overlap's
  own boundary condition, are invalid, so W must cover the own cells' stencils);
- the pressure solve keeps the overlap's pressure at the peer's of its last iteration
  (`couple.read_iterate`, the window end): `pressure(dirichlet=(mask, G))` solves the block system
  whose overlap rows are the identity, `[A_oo A_oh; 0 I] [P_o; P_h] = [-R(0)_o; G]`, i.e. the own
  equations with the overlap pressure as Dirichlet data (the subdomain solve of Schwarz; Newton:
  the update zero there); the increment projection takes `G - P^n` for `dP`;
- the participant writes all state rows of its mirror cells; `couple.repeat` after `advance`
  (preCICE's `requiresReadingCheckpoint`) restarts the window.

At the fixed point every own-cell equation of the step reads the peer's values: the single-domain
step, to the coupling and pressure tolerances.  The interface flux is the same Riemann problem on
both sides (identical states and aux), so the mass is conserved.  The exchanged data are the
n_state rows of the mirror cells (the full-state rule of the hyperbolic coupling, on W rings);
with several SSP-RK projection stages (time_order 2, 3) then each stage's pressure rows, the
Dirichlet data of that stage's solve (`step(dirichlet=(mask, (P_1, .., P_m)), pressures=[])`;
the window start reads the state rows only).  The overlap must cover the stencils of every
stage: `ChorinTemplate.overlap_rings()` = 2 per stage + 1, enforced by every driver (measured
below; reconstruction order 2 is refused, its rings not established).  New vocabulary: blocks `overlap_cells`, `overlap_states`, `overlap_fill`; effects
`couple.read_iterate`, `couple.repeat`.  `PreciceAdapter` moved to zoomy_core
(`symsolver/precice_adapter.py`, re-exported by the JAX runtime) and drives the NumPy backend too;
the C++ effects (`rt_common.H`) answer the checkpoint queries and write / read vertex subsets;
foam takes the overlap and mirror cells from the driver (OpenFOAM numbering), AMReX the slabs of W
layers at the coupled side (single rank).  Refused: MOOD (a global demotion decision),
IMEX-SSP2's projected stages, an implicit diffusion (a second global solve), several ranks per
participant, reconstruction order 2, fewer than `overlap_rings()` rings.

**Gate 54** (`gates/gate54_chorin_precice.py`): VAM(1), one horizontal dimension, C10's wave
(H 0.34, a 0.01, x0 -0.5) over the bump on [-1.5, 1.5], 60 cells, left q = r = 0, right P = 0, O1,
time_order 1, dt = window = 2^-7 to t 0.25 (32 windows); split at x = 0 into 30 | 30 cells, W = 3;
pressure GMRES 1e-13 (full restart, every backend), coupling convergence |change| <= 1e-12
relative or 1e-13 absolute on every exchanged row; the single domain on the same backend.

| backend | scheme | max\|coupled - single\| | \|mass coupled - single\| | iterations / window |
|---|---|---|---|---|
| JAX | serial-implicit, no acceleration | 5.4e-15 | 0 | 19-23 |
| JAX | serial-implicit, IQN-ILS on P | 6.9e-16 | 0 | 6-7 |
| JAX | parallel-implicit, IQN-ILS on P | 9.2e-16 | 0 | 7-8 |
| JAX, increment projection | serial-implicit, IQN-ILS | 4.7e-16 | 0 | 6-7 |
| NumPy | serial-implicit, IQN-ILS | 6.3e-16 | 0 | 6-7 |
| foam (OF13 image) | serial-implicit, IQN-ILS | 9.8e-16 | 1.8e-16 | 6-7 |
| AMReX (image) | serial-implicit, IQN-ILS | 2.8e-16 | 1.8e-16 | 6-7 |

(The single domain loses 2.2e-5 of mass through its open right end; the coupled pair the same.)

| variant (serial-implicit, IQN-ILS) | W | JAX | NumPy | foam | AMReX | iterations |
|---|---|---|---|---|---|---|
| SSP-RK2 stages (`GATE54_TIME_ORDER=2`) | 5 | 6.1e-16 | 6.0e-16 | 8.7e-16 | 3.1e-16 | 8-9 |
| 2-D: C10's quads, ny 2, VAM(1) 2 horizontal dims (`GATE54_DIM=2`) | 3 | 8.5e-16 | 1.3e-15 | 1.35e-15 | 4.0e-16 | 6-8 |

(|mass coupled - single| <= 3.6e-17 in all of them; foam increment 6.9e-16.)  Overlap width,
max|coupled - single| (NumPy; standard / increment projection): time_order 1: W 1 1.4e-3 / -, W 2
5.5e-15 / 5.3e-8, W 3 round-off / 4.7e-16; time_order 2: W 3 2.3e-8 / -, W 4 8.6e-16 / 1.1e-13
(deterministic: unchanged at pressure tolerance 1e-15 and coupling tolerance 1e-10), W 5 6.0e-16 /
2.9e-16; time_order 3: W 5 9.1e-15 / -, W 6 1.1e-15 / 9.0e-16, W 7 1.1e-15 / 9.0e-16.  The
predictor's flux and its derivative aux each reach one ring per stage, the increment projection's
lagged correction (before the solve) one more in some stages: `overlap_rings` = 2 m + 1 is exact
in every measured case (the increment needs it at time_order 1 and 2).  A too thin overlap is not
a convergence failure but a wrong fixed point (1e-13 .. 1e-3), so the drivers refuse it.
The iteration count does not depend on W: the subdomain solve
covers the own cells only, the overlap supplying Dirichlet data one ring deep, so this is
minimal-overlap Schwarz (block Gauss-Seidel / Jacobi between the two subdomains; the measured
contraction 0.32 per Gauss-Seidel iteration is the operator's, not the overlap's).  IQN-ILS on the
pressure rows (the data the iterations read) brings it to 6-7.  A true overlap (solving the inner
overlap rings too, whose equations are exact when their predictor state is valid) would contract
with the overlap width; not built.

zoomy_core test `tests/symsolver/test_overlap_schwarz.py`: the rings of `overlap_layers` (1-D,
quads), and the coupled pair in process (NumPy, two threads, an in-memory parallel-implicit
coupling without acceleration, 24 cells, 4 steps) = the single domain to < 1e-12.

**Matrix C16**: rows `vam_jax`, `vam_jax_increment`, `vam_jax_t2`, `vam_jax_2d`, `vam_numpy`,
`vam_foam`, `vam_foam_2d`, `vam_amrex`, `vam_amrex_2d` (gate 54, serial, W = `overlap_rings`):
coupled vs single <= 1e-12 (the coupling tolerance), |mass coupled - single| <= 1e-14, iterations
per window (information): 0 FAIL.

**foam build cache race** (fixed): two participants of one binary (the coupled Left and Right
print the same program) built it concurrently; one ran the binary while the other linked it
(`Permission denied`) and the cache kept a 0-byte `symFoam`.  `FoamDriver.build` now locks the
build directory and links to `symFoam.tmp`, renamed into place (AMReX's pattern).  The hyperbolic pairs (jax_jax, jax_foam, foam_foam,
amrex_amrex, amrex_jax; serial and mpi2, rerun on the changed adapter / C++ headers) equal the
table to the printed digits.  zoomy_core 397 passed (default tier) + the new test; QR 27 passed, 3
skipped; zoomy_jax `test_lifted_shared_face` 3 passed.

**Open.**  (1) IMEX-SSP2's projected stages (three projections per step, the final one on the
combination) and reconstruction order 2 (its overlap rings).  (2) Several ranks per participant
(the overlap's cells on several ranks).  (3) 3-D and unstructured overlaps over preCICE (the rings
and the AMReX slabs are general; gate 54 runs 1-D and C10's quads).  (4) A true overlap (solve the
inner overlap rings) to cut the iterations without acceleration.

## 57. Oblique walls in the walls' normal frames; foam's O2 gradient is the template's (2026-10-14, dev/oblique-walls)

**Oblique walls.**  A `Wall` reflects the normal momentum.  On a face not aligned with an axis its
ghost couples q_x and q_y, so section 41 classed both rows as ghost rows (no constraint), and the
decoupled ghost fit of the base table (section 28: a ghost informs only the pure monomials of its
axis) took the axis of x_f - x_c, not the line the mirror ghost lies on.
- Rows (`boundary_closure.face_closure` -> `RowClosure(kinds, groups, rotated)`): a vector group
  (the BCs' `momentum_field_indices`) is classified in each face's normal frame wherever that fixes
  more rows than the Cartesian frame: n.q Dirichlet, t.q Neumann (2-D t = (-n_y, n_x); 3-D the rows
  of I - n n^T, whose squares sum to those of any orthonormal tangent basis).  Axis-aligned faces
  keep their per-row kinds: on every matrix mesh but C38's the tables are unchanged.
- Fit (`geometry_xp._vector_tables`): such a group is one vector class, one joint constrained fit
  of its d components (block-diagonal samples; per face the frame rows F q as ghost samples,
  Dirichlet values or Neumann normal derivatives; the null-space form of section 41).  Its d x d
  block tables follow the row classes in `lsq_closure`; a vector row's `lsq_closure_class` entry
  is its terms ((src_row, class), ...) (`closure_terms`, `closure_rows`), read by core
  `lsq_closure_derivatives` (NumPy), the JAX `derivative_aux` / `gradient`, foam's `Kernels.H`
  aux entries (two partner pairs) and `Fit`.  AMReX is Cartesian: no vector class.
- Base fit (`_wall_frames`): the mirror ghost lies on the face normal, so it informs the
  pure-normal monomials of the face frame.  A cell whose sampled boundary faces are oblique fits in
  the frame Q = R(k 90 deg) R(-phi_0) (|Q - I| <= 45 deg) if one Q puts all its face normals on
  axes (else Cartesian) and maps the coefficients back with the monomial rotation T (c = T^T a,
  `_monomial_rotation`, differentiable in the nodes).  2-D; axis-aligned cells unchanged.
- Checks (gate 57, `gates/gate57_oblique_walls.py`; matrix C38): a `Wall` field (n.q = 0,
  d(t.q)/dn = 0, generic at the walls: an even part in q_n, an odd cubic part in q_t, which a ghost
  sample cannot reproduce); orders of the derivatives of q_x, q_y over the boundary cells, last
  pair (the interior as gate 44):

| mesh (nx) | p | closure, k = 1 / 2 / 3 | ghost sample, k = 1 / 2 / 3 |
|---|---|---|---|
| box 10 x 4 rotated by 30 deg, quads (48 -> 96) | 2 | 2.01 / 0.86 | 1.0 / 0.0 |
| | 3 | 2.93 / 1.92 / 0.91 | 1.0 / 0.0 / -1.0 |
| same box, triangles (48 -> 96) | 2 | 1.93 / 0.89 | 0.9 / 0.1 |
| | 3 | 3.01 / 1.95 / 0.96 | 1.0 / 0.0 / -1.0 |
| annulus 1 <= r <= 2, polar quads, curved walls (96 -> 192) | 2 | 1.90 / 0.96 | 0.8 / 0.0 |
| | 3 | 3.02 / 1.92 / 1.07 | 1.0 / 0.0 / -0.5 |

  Polynomial vector fields are reproduced to 1e-11 on rotated quads, jittered quads and triangles
  (p 2, 3; corners with two frames) and on rotated hexahedra (p 2; edges, corners)
  (`test_gate_1a_core_lsq_closure.py`).  The rotated mesh: the same stencils (the axis rule
  extended all 68 wall stencils of the rotated box, nx 24, by 1 to 5 cells), every derivative
  tensor the rotated one to 7e-13; an O2 run (SME(0), LSQ degree 2, no limiter, 40 steps) of the walled box at 30 deg =
  at 0 deg: max rel h / q 2.8e-3 / 3.2e-2 before, 3.9e-16 / 1.5e-15 after (JAX, NumPy; foam C38).
- Limits.  The run takes `SpectralRadius.eig`: the default bound of section 48 works on |A_n| and
  depends on the momentum's basis (4.4e-7 / 4.1e-6 with it).  A curved wall is its polygon: on polar
  quads a wall cell's foot is its face's midpoint (normal e_r there) and the gate's field carries the
  BC there to O(h^4); on triangles the foot is off the midpoint, the polygon's normal O(h) off the
  wall's, and free slip is first order whatever the closure (geometry).  The base fit's frames are
  2-D (3-D oblique cells stay Cartesian; the vector closure is 3-D).  Annulus p = 3: the interior
  degree-3 fits two cells off the walls (10 samples, 9 monomials) do not converge (d3 ~1e2), on
  main and without walls (the selection of section 37; information in C38).

**foam's O2 gradient on stretched wall triangles** (section 43; C12 `sub` O2 at a fixed ny 2).
foam L1(h) 3.0e-3 / 7.8e-4 / 1.5e-3 at n 25 / 50 / 100; JAX 6.9e-3 / 7.7e-4 / 1.3e-4.
- Which block: foam printed `gradient` and the limiters with OpenFOAM's leastSquares and
  cellLimited.  OpenFOAM's gradient with the template's limiter: 3.0e-3 / 7.8e-4 / 1.5e-3
  (unchanged); the template's gradient with OpenFOAM's limiter: 5.2e-3 / 9.1e-4 / 1.6e-4.  The
  gradient, not the limiter.  Not an instability: max |q_y| saturates at 2.4e-2 by t = 2 (JAX
  1.6e-3), a steady state with 15x the transverse momentum over the bump.
- Why: OpenFOAM 13's leastSquares (`leastSquaresVectors.C`) weights a neighbour by |S_f| (1 - w) /
  |d|^2 and puts the boundary face value at the foot n (n.(C_f - C)) with weight |S_f| / d_n^2, face
  neighbours only.  On these triangles it turns a field of x alone into a wall-normal derivative of
  4.2e-3 / 1.6e-3 / 5.0e-4 at n 100 / 200 / 400 (order 1.4 - 1.7); zoomy_core's fit (vertex rings
  inside, face neighbours and the decoupled mirror ghost at the walls, unweighted) 1.7e-3 / 4.2e-4 /
  1.0e-4 (order 2).  The spurious transverse pressure force drives the q_y above.  Neither
  conditioning nor missing constraint rows: at degree 1 there is no closure.
- Fix: foam's `gradient` is zoomy_core's table through the `Fit` (`constant/zoomyStencil` now
  carries each reconstruction row's closure entry, so at degree >= 2 the reconstruction rows close
  as on JAX, vector rows included); foam's minmod, Barth-Jespersen and Venkatakrishnan are the
  template's (face-neighbour and ghost bounds, as JAX and NumPy); OpenFOAM's `limited(...)`
  schemes are gone.  C12 fixed ny: foam = JAX to 1.4e-15 / 5.2e-15 / 3.8e-10 (n 100: round-off
  through limiter switches), L1 6.77e-3 / 7.65e-4 / 1.31e-4 (order 3.1 / 2.5).
- The shore form of section 50 (merge of integ/20261014) lives in the same two blocks, as
  zoomy_core's `lsq_derivatives(..., shore)` and the runtimes' `ghost_bounds(..., dry)` do it:
  `Fit::gradient` (the base fit only; a closure fit keeps its samples, as on JAX) sends the depth
  row with the values through the Fit's distribution map and takes min(eta_k - eta_i, 0) for a
  stencil cell with h_k <= eps; `limit` bounds row eta with min(eta_k, eta_i) for a dry face
  neighbour (halo columns on coupled patches).  OpenFOAM's shore loops (`shore_lsq`,
  `shore_limiter`) and `tests/shore_unit` (their check against OpenFOAM's schemes) are gone.
  Pit lake at rest (`cases/pit_lake_at_rest.py`, now `PIT_BACKEND=foam`; front detector on): foam
  serial venk 2.3e-15, venk_tri 1.1e-14, tilted 5.2e-15 / 8.4e-15, wet, minmod, BJ, no_front,
  order 1 <= 4.2e-15; 2 ranks the same; without the rule (the same build, shore arguments
  ignored) 0.40 / 0.52 m/s, as JAX on main: foam's Venkatakrishnan now has eps^2 = (cell size)^2
  too, so the rule is what keeps the lake at rest.
- Found on the way (zoomy_core): the decoupled ghost's axis was argmax |x_f - x_c|.  On a wall
  triangle stretched along the wall that is the tangential axis, and the ghost informed nothing:
  wall-normal derivative of an x-only field 0.16 on the n 25 mesh (dx = 2 dy, a tie).  The axis is
  now the normal's (the frame's on oblique walls): 2.7e-2.  Gates 39 (72 / 72), 42 (18 / 18), 44
  (quads and triangles) pass.

Cross-backend matrix (`matrix/README.md`, results 2026-10-15): the store rerun from empty on core
9f739cf, jax 43c1db9, foam a37c9c8, amrex a177451, the CPU configs (serial, mpi2, mpi4, mpi8; no
GPU here): 959 runs, 935 rows, 939 PASS cells, **2 FAIL**, both C17 `dry` (AMReX AMR, regrid on the
dry front; a run added to C17 after the last full matrix): mass drift nan, its coarse output
carries NaN cells; the same nan with main's libraries (h_min 0.1209 both), not touched here.
Against the store of section 48: JAX 295 of 300 runs bitwise identical (the rest: C15 `chorin_rev`
2e-12 relative, C35 `typo` messages), NumPy and AMReX all identical; foam's 88 O2 runs changed to
foam = JAX: C1 O2 L1 diff 7e-16, C2 L1 2.6e-4 -> 3.8e-16, C4 sub / trans O2 max rel 3.6e-14 /
3.1e-12, C5 1.7e-2 -> 1.9e-15, C11 sub O2 3.8e-6 (JAX's own sensitivity 2.2e-5), C23 O2 4.5e-14,
C26 1.6e-3 -> 3.6e-16, C32 |foam - JAX| / |JAX - exact| 0.19 / 0.021 / 0.014 -> 2e-12; foam's
orders are JAX's (C12 bump O2 1.92 -> 1.69, sub O2 2.76 -> 2.42, C19 o2minmod 1.87 -> 2.79).  The
checks that measured a shrinking foam - JAX difference measure foam = JAX instead: max rel <=
10 x JAX's round-off sensitivity where the `*_pert` runs exist (as AMReX), C23 O2 max rel <= 1e-12.
New rows: C12 `subny2` (the fixed-ny case: JAX, foam, NumPy order 2.55; foam = JAX 3.8e-10 <= 10 x
the sensitivity 2.7e-10) and C38 (all PASS).  zoomy_core: 375 small tests pass (goldens included);
QR suite (`-m qr`, from the main tree's tests with these libraries first): 28 pass, 2 skipped.

After the merge of integ/20261014 (core bc90274, jax a8eb5d4, foam d5cbe75, amrex 514d9e0): gates
39 (72 / 72), 42 (18 / 18), 44 (all), 57 (orders, rotation 3.9e-16 / 1.5e-15) pass; zoomy_core
414 small tests pass; QR suite 28 pass, 2 skipped.  Matrix rows of C2, C3, C5, C6, C12, C19, C26,
C32, C37, C38 on JAX and foam, serial and mpi2, from an empty store: 215 rows, 189 PASS cells,
0 FAIL.  JAX is integ's bitwise (C5 bowl checked); foam = JAX now also on integ's new C5 front
runs (bowl / dam, front and off: 5e-16 / 1.8e-15; before 6.3e-3 / 4.7e-2 / 1.5e-2 / 9.3e-2) and
C37 (<= 1.4e-14), and the dam's front detector acts on foam as on JAX (front vs off 8.6e-2,
before 8e-17).

Open:
- The decoupled ghost fit's frames are 2-D; a 3-D oblique wall keeps the Cartesian decoupling.
- Free slip on a polygonal approximation of a curved wall is first order off the face midpoints
  (triangles): a curved-boundary treatment, not a closure question.
- Degree-3 interior fits on polar quads (10 samples) do not converge (annulus p = 3).
- The default wave-speed bound (section 48) is not rotation invariant.
- C17 `dry`: the measure (or the AMR output) of the dry-front regrid run.

## 58. Library defects from the thesis-case migrations (2026-10-14, dev/migration-defects)

Seven defects the migrations found, each a silent wrong answer or a run that failed where the
scheme is fine.  Numbers: NumPy unless stated; the matrix rows are in the results section of
`matrix/README.md`.

**1. The hydrostatic reconstruction with the pressure in the flux.**  Audusse et al. (2004)
restore, per face side, the flux a lake at rest keeps: `S~ = Pi(q_i) - Pi(q_i*)`.  The code read
`Pi` from `hydrostatic_pressure`, but the hand-written `SWE` keeps `g h^2 / 2` in its `flux`
(`P = 0`): `S~ = 0`, and `PositiveRusanov` / `PositiveHLL` / `PositiveNonconservativeHLL` (the
path integral runs between the reconstructed states, where `b* - b* = 0`) lost the bed slope
entirely; `PositiveNonconservativeRusanov` integrated the raw states.  Lake at rest over
`b = 0.2 exp(-(x-5)^2/2)`, 40 cells, 20 steps of 0.01, max |hu|:

| flux | before: SWE O1 | SME(0) O1 | SWE O2 | SME(0) O2 | after (all, O1 / O2) |
|---|---|---|---|---|---|
| PositiveRusanov | 8.5e-2 | 0 | 8.8e-2 | 8.6e-2 | <= 1.3e-16 |
| PositiveHLL | 8.5e-2 | 1.8e-16 | 8.8e-2 | 8.6e-2 | <= 2.3e-16 |
| PositiveNonconservativeHLL | 8.5e-2 | 1.8e-16 | 5.0e-3 | 1.9e-16 | <= 2.3e-16 |
| PositiveNonconservativeRusanov | 1.2e-2 | 0 | 2.5e-3 | 1.3e-16 | <= 1.3e-16 |
| PositiveQuasilinearRusanov | 0 | 8.6e-2 | 4.2e-18 | 5.0e-3 | <= 4.3e-18 |
| Rusanov, HLL, HLLC | 8.0e-2 (bed dropped) | 8.0e-2 | | | raise at setup |

The fix is one derivation: `Pi` is the physical flux `F + P` at the lake at rest of the face
state (depth and bed kept, every other row zero; `Numerics.rest_flux_definition`), the flux the
conservative numerical flux evaluates at the reconstructed states.  For every derived model
(SME 0-2, ML-SME, VAM(1) predictor, MalpassetSWE) it is `P` symbolically (their `F` vanishes at
rest), so their kernels are unchanged; for `SWE` it is `g h^2 / 2`.  Two more defects of the same
scheme surfaced in the sweep: (a) at order 2 `PositiveRusanov` / `PositiveHLL` had no in-cell bed
term (the reconstructed bed varies inside the cell; the path-conservative classes carry it in
`numerical_cell_edge_source`): they now add Audusse's centred source
`g (h_c + h_e)/2 (b_e - b_c)` (the bed column of `B` by the same Gauss rule, exact for `g h`);
(b) `PositiveQuasilinearRusanov` has no conservative flux (the quasilinear path integral between
the reconstructed states already carries `g h* (eta_R - eta_L)`, zero at rest), so its `S~` is
zero: adding it counted the pressure jump twice (not WB, and inconsistent).  A class whose face
terms do not discretize a live column of the nonconservative matrix now raises at setup, naming
the flux and the columns (`check_numerics`): Rusanov / HLL / HLLC on any model with a bed;
`PositiveRusanov` / `PositiveHLL` on a model whose `B` couples more than the bed (SME(1)).
`NonconservativeRusanov` / `QuasilinearRusanov` stay offered: consistent, not well-balanced by
construction (Rusanov dissipation on the depth jump; 3.5e-3 above on both models alike).  A flow
over the bump (40 steps): SWE (pressure in `F`) and SME(0) (in `P`) now agree to 1.6e-13 (HLL,
Davis speeds since 52c), 1.7e-6 (the Rusanov family: SME(0) has no closed-form spectrum, its
dissipation speed is the Gelfand bound) and exactly (`NonconservativeRoe`).  Before, SWE + PositiveHLL was off by O(0.1) (the river case's 57 %).  Matrix: C3 on
plain SWE with four Positive* fluxes, O1 / O2, all four backends: max |u| <= 2.5e-15.

**2. The periodic seam's `face_aux`.**  NumPy and JAX gave a boundary face the inner cell's aux on
both sides, also at a periodic seam, where the Riemann problem's outer state is the partner's
trace: a face kernel reading aux (the regularised `hinv` of `NonconservativeRusanov`; the
Positive* classes rebuild it from the reconstructed depth, which hid the defect in C13) took
`hu_R * hinv_L`.  One step, 40 cells, seam moved by 7 cells: the 2 seam cells' momentum differed
by 1.1e-4; now 0.  The seam face now reads the partner cell's aux and CFL speed
(`fc_other`, `fc_cfl`), the reconstruction's seam ghost the partner's aux.  foam (cyclic patches
are coupled faces with a halo column) and AMReX (a periodic side is no boundary) were right.
Matrix: C13 with `NonconservativeRusanov`, shift by L/2: 0 on all four backends.

**3. The sliver step at the end time.**  The time is a floating-point sum: ten steps of 0.005 end
at 0.05 - 6.9e-18 (one step of 1e-3 to 2: 1.1e-13), and the march took a further step of that
size; the Chorin projection (`1/dt^2` in its pressure operator) failed there (VAM(1), 1-D:
relative residual 9.4e-3 > the gate).  `landed(dt, cap)` merges a leftover to the next instant
(end, write, checkpoint) below `LANDING_TOL = 1e-6` of the step into the step: after n steps the
sum is off by up to ~n^2 u in units of the step, so 1e-6 merges every such residue up to n ~ 1e5
(worst case) and lengthens the merged step by at most 1e-6 (no CFL margin is that thin).  A
genuine remainder (0.05 / 0.004) is still its own step.  In the template (`face_rule`, the cells
rule): every backend; runs without a sliver are bitwise unchanged.

**4. JAX `initial_aux`** converts array-likes (`jnp.asarray`; tracers pass): a NumPy `Q` reached
the vmapped BC kernels of the derivative aux and raised `TracerArrayConversionError`.

**5. `equilibrium_reconstruction`** `'bernoulli'` / `'projected_bernoulli'` (moving-equilibrium
well-balancing) were ignored by the symbolic solvers: setup now raises `NotImplementedError`;
`'audusse'` with a Riemann class without a hydrostatic reconstruction raises `ValueError`.

**6. The default pressure solve on a periodic channel.**  Not a null space and not the smoother:
the assembled solvers (NumPy / JAX `pyamg`, JAX `hypre`, `amgx`) colour-assemble the operator on
a sparsity pattern, and that pattern was the mesh's LSQ stencil (`mesh.lsq_neighbors`), which
ends at a periodic seam, while the derivative aux the operator is built from samples the
topology's stencil (`topo.lsq_neighbors`, wrapped across the seam since ANALYSIS 33) and the
cells behind the boundary ghosts (calibration case, 8 cells: cell 0's pattern {0, 1}, its fit
samples 7 too; 2-D: a seam cell samples both cells across).  The matrix lacked the cross-seam
entries, GMRES converged on it, and the true residual stayed O(1): 0.78 (calibration VAM(3)),
2.2 (VAM(1) wave, 1-D, 16 cells), 0.39 (2-D channel, C8 `per_auto`); the same solve on a dense
pattern converged.  `operator_neighbors(topo, mesh)` is the samples of the fit: the topology's
stencil and the cell behind each boundary-face sample (`boundary_face_cells`: inner, or the
partner across the seam); checked against the dense operator (2-D, 16 and 48 cells: no entry
outside it).  NumPy pyamg = GMRES to 3e-12; JAX pyamg / hypre / auto (= hypre on the CPU) failed
before (the accept gate in the host callback) and equal NumPy's GMRES to 3e-12 now.  Matrix: C8
`per` (each backend's default solver on a periodic VAM(1) channel; foam / AMReX: hypre, right
before too) and `per_auto` (JAX, NumPy: 'auto' vs GMRES).

**7. Derivative aux no backend computes.**  An `aux_registry` entry of target `unknown` (a
compound such as `h*zeta`), a `limited_derivative`, or (foam / AMReX) a derivative of an aux
function was read as 0 (JAX), skipped (NumPy `derivative_aux`) or dropped from `Kernels.H`
(foam / AMReX).  Setup now raises, naming the aux and its target (`check_derivative_aux`; NumPy and
JAX compute state and function targets, foam and AMReX state targets).  The one model that hit
it, the full-stress `Sigma3D` (`d_x zc` of its in-plane stress term, `zc = b + h zeta`), takes
`d_x b + zeta d_x h` by the chain rule (as ML-VAM's layer depths, 45i): golden m16 changes
`h*zeta_x` (target unknown) to `h_x` (state) with `zeta` in the source term; equal after
`h*zeta_x := zeta h_x`.  No matrix case has another target than a state row.

**Matrix** (`matrix/README.md`, results 2026-10-14 dev/migration-defects): full rerun, serial
on all four backends, mpi2 / mpi4 on JAX, foam, AMReX: 845 PASS, 0 FAIL, 44 new rows (C3 plain
SWE, C13 seam aux, C8 periodic pressure); three rows moved at round-off (C14 1.50e-12 ->
1.52e-12, C31 1.08e-13 -> 1.16e-13), every other row equal to 1e-6.  zoomy_core: small 451 pass,
large / rederive only the known m05 golden and cache-correctness failures; QR (`-m qr`): 20 pass,
10 skip (Pyodide not installed here).  After the merge of integ/20261014 (52's step guard
`Cond(time_new > time, ...)` and `check.march` in `_time_loop`, the nested checkpoint `Cond`;
`landed` acts on the step inside `step`, so both compose: a merged sliver gives `time_new = time
+ cap`, i.e. the instant itself): zoomy_core small 473 pass, the same two known large / rederive
failures; C3 / C8 / C13 / C14 / C31 / C37 serial + mpi2 all PASS; QR 20 pass, 10 skip.

### 58a. Moving-equilibrium well-balancing on the symbolic solvers: a plan

*What the old solver does* (`zoomy_core/fvm/bernoulli_wb.py`, NumPy only, 1-D only): per face,
each side's state is reconstructed to the common bed `b* = max(b_L, b_R)` preserving the
discharge `q` and the per-streamline Bernoulli head `H(s) = u^2/2 + g (b + h)` (`bernoulli`:
discharge-fraction streamline label; `projected_bernoulli`: layered, sigma-parameterised
`|u|` weighting with signed velocity, Newton on the surface, the moments reprojected by
quadrature); the Riemann solver and the moment NCP act on the reconstructed pair and the
consistency source is the jump of the full conservative flux `Fc(q) - Fc(q*)` (the mass row
cancels: `q` is preserved).  At level 0 the reconstruction is a cubic (closed form); it is exact
for frictionless steady flows only (`H` is not conserved with friction).

*Literature.*  The general form (Castro, Pares and co-workers; Gomez-Bueno et al. 2023, Zotero
785VK5FP): a fully well-balanced reconstruction finds in each cell the stationary solution of the
local steady ODE `F(U)_x + B U_x = S(U) H_x` (friction in `S`) with the cell's average, reconstructs
only the fluctuation around it and subtracts the stationary flux / source; without a closed form
the local ODE is integrated numerically (collocation RK).  Koellermeier and Pimentel-Garcia 2022
(8FC5TJIS) is the blueprint for the shallow-water moment equations, Pimentel-Garcia's friction
chapter (D93ZEV5W) integrates `U_x = -(J_F + B)^{-1} (S H_x + R)` to the interfaces.  The Bernoulli
reconstruction above is the frictionless closed-form corner of this.

*Where it would go.*  (1) A model operator, not a solver option: the SystemModel already carries
`quasilinear_matrix`, `source` and the bed role, so `steady_ode_rhs(U, b_x) = -(A_n)^{-1}(S + B_b
b_x)` is derivable symbolically (a `solve` kernel per cell, as `SourceTreatment.linearized`).
(2) A reconstruction mode `ReconstructionSpec(equilibrium=Equilibrium.moving)` in
`discretization/reconstruction.py`: per face, integrate the local ODE from the cell to the face
centre (a fixed number of RK stages in the template: no new block; or a `Root` for the closed-form
Bernoulli corner) to get `q_e*`, then add the limited fluctuation at order 2.  (3) A Riemann class
whose consistency flux is the full `Fc` at the reconstructed states (`consistency_flux_definition`
of item 1 generalises to it: `Pi = F + P` without the rest projection), plus the cell term
`Fc(q_e*) - Fc(q_c)` minus the source integral along the same ODE path (zero for the exact
stationary solution).  (4) Tests: MacDonald-type steady states (prescribed `h(x)`, the bed
integrated from it: the exact discrete steady state is known), frictionless and with Manning,
levels 0-2; the lake at rest as its `u = 0` corner; every backend through the template.  2-D
needs the streamline (a velocity vector), as in the old code's 1-D restriction.

## 59. JAX Chorin on the GPU: the cells-CFL eigensolve and the pressure solve (2026-10-14, dev/jax-chorin-perf)

Case: thesis `escalante_vam_bump/run_mlvam2_jax_jit.py` (ML-VAM(2) level 1, 14 states, 60 cells,
RIEMANN=hr ORDER=2 TORDER=2, MOOD, constant dt 1e-3), copy of thesis main, GPU 0 (L40S),
`taskset -c 32-39`, float64.  Library: main a2ba0f57 (zoomy_core 9308483, zoomy_jax 468fd7f).

**Reproduction.**  t = 3 (3000 steps): 671 s in all = build 20 s + `setup` 115-140 s + one XLA
compile of the march 270-300 s + 265 s of steps; warm steps 0.088 s/step, flat to t = 3, the
same with `on_write` every 100 steps.  `jax_log_compiles`: one compile of the march, none per
step (the template is one jitted `lax.while_loop`).  The step4 report's "~1 s/step" does not
reproduce; the legacy solver today (`ChorinSplitVAMSolverJax.run_jit_steps`, the thesis script
of 40c26762, same GPU) runs 0.21 s/step.

**Profile before** (nsys, 50 warm steps; per step):

| cost | GPU time | launches / copies / syncs |
|---|---|---|
| `jnp.linalg.eigvals` = cuSOLVER `Xgeev`, 2 x 61 14x14 matrices (`face_wavespeed` of `eigenvalues_cfl`: the cells rule) | ~60 ms | 17000 / 2300 D2H / 3700 |
| AmgX (FGMRES + aggregation AMG), 2 pressure solves | ~3 ms | 1800 / 450 / 640 (~12 ms of host time) |
| the rest (XLA fusions in CUDA graphs: predictor, MOOD, colouring assembly, aux) | ~3 ms | |

`Xgeev` is a host-driven routine called once per matrix; the CFL check of a constant-dt run reads
the speeds every 10th step only, but they are computed every step (outside the `Cond`; the
template is integ/20261014's, not touched here).

**Fixes** (zoomy_jax; zoomy_core only the enum member):

1. `zoomy_jax/fvm/eigvals.py`: `eigvals_device`, LAPACK `dgeev`'s eigenvalue path (`dgebal`
   isolation + power-of-two scaling, `dgehd2`, `dlahqr` with the Ahues-Tisseur deflation and
   `dlanv2`) written on whole `N x N` blocks with masks (N = 16 for n = 14): one Triton kernel
   (Pallas, a program per matrix, the QR loop inside the kernel) on CUDA, the same code vmapped
   elsewhere.  `eigvals` = LAPACK on the CPU (unchanged), `eigvals_device` on a GPU; used by the
   `eigenvalues` userfunction and the numerical-eigenvalue fallbacks of `JaxRuntime`.  Its
   jvp / vjp are `jnp.linalg.eigvals`' (section 44 unchanged).  `face_wavespeed` evaluates both
   sides of every face in one batch (one eigensolve per step instead of two).  122 matrices:
   cuSOLVER 67 ms, LAPACK on the host 8 ms, the kernel 0.9 ms (n = 30: 137 / 5.9 ms; n = 40,
   8 warps: 180 / 34 ms; above n = 64 the XLA loop); vs LAPACK <= 2e-14 |A|
   (unit test: random, structured, defective, badly scaled; batch-independent bitwise).  Not a
   bound: `gelfand_bound` (section 48) for the cells rule would be cheaper but changes dt; left
   as an option.
2. `LinearSolver.dense`: an LU (`jnp.linalg.solve`) of the colour-assembled pressure matrix on the
   device, inside `custom_linear_solve` (transpose solve for reverse AD).  `auto` on a GPU, serial:
   dense up to `pressure_dense_max` = 1024 unknowns, AmgX above (dense 240: 0.6 ms, 1000: 4.9 ms,
   2000: 14 ms; AmgX ~6 ms per solve at 240).

**After** (warm steps, same bench):

| state | s/step |
|---|---|
| main | 0.088 |
| + device eig (vmapped XLA, 4.7 ms per eigensolve) | 0.021 |
| + dense pressure LU | 0.0082 |
| + Triton eig kernel | 0.0046 |
| legacy `run_jit_steps` today | 0.21 |

Split after (nsys, node-level, per step): 4.3 ms GPU busy of ~4.6 ms: XLA fusions 2.6 ms (960
kernels: predictor, MOOD, assembly, aux), pressure LU 1.1 ms (2 x cuSOLVER `getrf` 240x240),
eigensolve 0.6 ms (1 kernel); 7 host syncs (the loop and `Cond` predicates).  The case to t = 3:
671 -> 432 s, now build + setup + compile (~430 s) against 14 s of steps.

**Results.**  The case's 30 frames to t = 3: the eig changes bitwise identical (at constant dt the
speeds only feed the CFL check); dense LU vs AmgX (rtol 1e-9): final state rel L2 3.6e-12, max
over the frames 5.9e-11.  Matrix (per-worktree store, C8 C10 C31 jax serial, C8 gpu1): 12 PASS,
0 FAIL (C9 C8 GPU vs serial 1.8e-11); C8 serial = main bitwise (4 runs: CPU keeps LAPACK, and the
one-batch face speeds are per cell the same arithmetic); C8 gpu1 vs main gpu1 2.9e-12 (adaptive
dt from the new eigensolve); C8 gpu1 with `LinearSolver.dense` vs serial GMRES 1.3e-11.  Tests:
zoomy_core 375 pass; zoomy_jax 26 failures, the same ids as on main; QR 27 pass, 3 skip.

**Open.**  (a) The march's XLA executable is never in the persistent cache: jax writes no entry for
an executable with host callbacks (`check.cfl`, the linear-solve accept gate, `io.write`), so every
run compiles 270-300 s.  (b) `setup` 115-140 s: sympy in the Riemann solver build
(`riemann_solvers._model_eval` / `face_max_abs_eigenvalue` substitutions, cProfile).  (c) The
pressure LU ignores the block-banded structure (1-D: 1.1 ms/step).  (d) At constant dt the cells
speeds are evaluated every step for a check every 10th (template).

## 60. Steffler VAM: the spurious wall pressure and the 5280-cell divergence were the oblique walls (2026-10-15, dev/vam-wall-pressure)

Thesis migration defect (`cases/migration_status/steffler.md`): `SymbolicChorinSolverJax` on the
Steffler bend (VAM(1), dim 3, `elder_fix` + Chezy) put |P_0| up to 425 into the inner-wall cells
at 2150 cells (o1, t = 20 s; the legacy `ChorinSplitVAMSolverJax` <= 4e-4) and diverged at 5280
cells within ~15 steps for every dt and pressure solver.  No library change here: section 57
(dev/oblique-walls, on integ2) removes it.  Matched runs, the thesis driver
`steffler_vam_jax.py --level 1 --bulk elder_fix --bottom chezy --psolver hypre`, legacy dt,
before (integ: core cda425d, jax 53a6fcc; * the migration's own runs, `matrix_symsolver`, on the
libraries of that day) vs integ2 (core b56c274, jax 9f620e8):

| run | before | integ2 | legacy (`matrix_elder`) |
|---|---|---|---|
| o1 5280, dt 1.18e-3 | NaN at t = 0.0175 (15 steps; P_0 6.7e2 at r = 3.158, t = 0.01) | finite to t = 6 (5080 steps); max P_0 1.7e-3 (t = 2), 1.1e-3 (t = 6) | 1.3e-3 (t = 2), 8.5e-4 (t = 6) |
| o2 5280 | diverges* | finite to t = 2.5 (2120 steps), P_0 1.6e-2 | 1.5e-2 (t = 2) |
| o1 2150, wall cells, t = 8 / 20 | 0.84 / 425* | 7.2e-4 / 8.9e-4 | 4.1e-4 / 2.9e-4 |
| o2 500, wall cells, t = 10 / 40 / 60 | 6.7e-2 / 5.4e-2 / 4.7e-2* | 1.3e-3 / 1.0e-4 / 8.4e-5 | 4.5e-4 / 8.9e-5 / 7.9e-5 (t = 50) |

Mechanism (cause the integ libraries had).  The pressure rows read the predictor's divergence
`q_x_k_x + q_y_k_y` from degree-2 LSQ fits.  On the bend's walls (oblique faces) the `Wall` ghost
couples q_x and q_y, so section 41 classed both rows as ghost rows, and the base table's decoupled
ghost informed the axis of x_f - x_c instead of the face normal.  A solid-body rotation
q_k = c_k h w (-y, x) (divergence free, tangential to the circular walls) gets a wall divergence
of 1.5e-3 (k = 0) / 4.5e-4 (k = 1) on integ against a derivative scale h w = 6e-3 (25 %); integ2
(normal-frame vector closure, n.q Dirichlet, t.q Neumann) 4.5e-5 / 1.4e-5, interior 3e-17 on
both.  The pressure solve and its correction are therefore not a projection at those cells:
iterating `ChorinTemplate.project` on that state (no predictor, 5280 cells) multiplies the bend
wall pressure by 1.7-2.9 per projection on integ (2.3e-2 -> 4.3e-1 in 5, at the inner-wall row
r = 3.158 where the runs blew up), and by 0.35 on integ2 (3.6e-3 -> 6.1e-5).  A growth per
projection does not depend on dt, which is what the reported dt / solver sweeps saw; at 2150
cells the factor is near 1 (slow growth from t = 5 s, saturation at 425).

Remaining difference to the legacy (factor 2-3 in the startup front's P_0 at the wall, the late
o2 values equal): not the wall closure of the divergence.  The legacy differentiated the momentum
with extrapolated wall samples (`u_bf=None`); integ2 with those samples (the press system's wall
ghosts of the q rows set to the inner cell, o1 2150, `--wave-speed bound`) gives P_0 1.3e-3 at
t = 7 against 1.0e-3 without: the no-penetration closure is the better one and stays.  The legacy
used other predictor numerics (Perron wave speed, its stencils; steffler.md, causes 1-2).

Open (not a closure defect of this case): free slip on a curved wall is `d u_t/dn = kappa u_t`
(zero shear), the vector closure imposes `d(t.q)/dn = 0` (the flat-wall form): O(h kappa) at the
walls, the order of the 4.5e-5 above.

## 61. One wave-speed default, the bound in the face's normal frame (2026-10-21, dev/wavespeed-frame)

- **Rule.**  Closed-form eigenvalues where the model provides them, else the Gelfand bound
  (section 48) of A_n evaluated in the face's normal frame (`system_model.wave_speed_bound`:
  `normal_frame` rotates each horizontal vector group to (q.n, q.t), as HLL's Davis speeds of
  52c already did).  Default `SystemModel.spectral_radius_method = SpectralRadius.gelfand` for
  every consumer and solver (symbolic and legacy): Rusanov / Positive* speeds, HLL Davis speeds,
  the faces and the cells CFL rules (`eigenvalues_cfl` of `gate_eigenvalues_dry`, new
  `bound_wave_speed` for transport systems without a depth gate), MOOD's repair step, IMEX-SSP2,
  Chorin, coupling (via the Riemann kernels).  `SpectralRadius.eig` (numerical eigenvalues,
  incl. the JAX device eigensolver of section 59) stays an opt-in; `characteristic_ghost` and
  NonconservativeRoe still decompose A_n (not a wave speed).
- **Removed** (no use left): the Perron bound (`SpectralRadius.bound`,
  `use_spectral_radius_bound`, `spectral_radius_bound`), the SME level >= 2 dt cone (now the
  bound), `zoomy_core.symsolver.WAVE_SPEED` / `default_wave_speed` (the default lives in the
  SystemModel).  `symbolic_spectrum()` as a late operation re-gates a stale dt spectrum.
- **Checks.**  Gate 57 rotation (box 30 deg = 0 deg, SME(0) O2, default wave speed): main 4.4e-7
  / 4.1e-6 (h / q, FAIL), now 3.9e-16 / 1.9e-15 (JAX, NumPy); C38 with adaptive steps (new rows
  `box_r*_faces`, `box_r*_cells`): <= 1.5e-15 on NumPy, JAX, foam.  Gate 49 (models, now also
  the cells rule's slot): 660,000 matrices, min ratio 1.000000000000000, max 1.049; gate 50
  568,000, PASS.  Thin-cell test: the bound is 1.044 x the near-defective pair u +- 1e-4 at
  u = 300 (eig and closed form exact).
- **Tests.**  zoomy_core 502 pass, rederive 41 pass; goldens n01-n03, s02, s03, p01 re-blessed
  (eigenvalues_cfl / Riemann speed rows: gelfand_bound instead of eigenvalues); prebuilt cache
  rebuilt (SME keys).  zoomy_jax: legacy-solver refs re-blessed (with `SpectralRadius.eig` they
  match the old ones; max|diff| 6e-7 .. 3e-5); remaining failures timing gates as on main.
  QR 28 pass, 2 skipped.  Matrix: not rerun in full (main does it after the merge); a partial
  serial / mpi2 store comparison against main showed Cartesian cases at round-off (C1-C4, C11,
  C13, C14, C26, C31, C37 <= 3e-9), triangles C12 <= 1.5e-6, C5 n48 4.6e-6, C10 s2 <= 5.9e-6,
  Euler C24 bitwise.
- Open: 3-D (3 directions) and reduced systems keep the lab frame (no unique tangent basis).

## 62. The dry-neighbour rule of section 50 is free-surface specific (2026-10-21, dev/wavespeed-frame)

Audited, no change needed: the template activates it only through `bed_rows` (roles.depth and
roles.bed declared, reconstruction W = (b, b + h), bed derived) and passes the rows and eps as
block arguments; NumPy / JAX / foam (`Fit::gradient`, `limit`) / AMReX (`gradient_shore`,
`limiter_factors`) take them from those arguments, no fixed index.  Without the roles the printed
blocks are the plain 3-argument gradient and 5-argument limiters (new test
`test_shore_rule_follows_the_free_surface_roles`).  Pit lake at rest (JAX): all variants
<= 1.1e-14.

## 63. VAM's default truncation is the order-consistent one; `VAMWeakViscosity`; SME / ML-SME on the pipeline; ML-VAM shared interface traction (2026-10-21, dev/vam-default)

User decisions (on ANALYSIS 49 and 45j): VAM is truncated by order in the thesis regime
`Fr, De = O(1)` by default; the thesis results (the hand rule = the weak-viscosity limit
`1/De = O(eps)`, `k_l = O(eps^2)`) stay reproducible through a separate class defined by the
pipeline, not by the hand code path, which goes; SME / ML-SME switch too if no result changes;
ML-VAM's interface trace of tau_ez is the mean of the two traces by default.

**What changed** (zoomy_core `dev/vam-default`).

- `VAM.default_scaling()`: `ShallowScaling(horizontal=0, vertical=2)` in the default regime
  (`Fr, De = O(1)`, `e_x = O(eps)`, NavierSlip `k_l = O(eps)`), with `Stress.full` (2, 2).
  `scaling=None` (the default) means this; an explicit `ShallowScaling` still overrides it.  The
  hand path (`moment_scaling` + `apply_full_tensor` in `VAM.derive_model`) is deleted: VAM always
  runs `ordered_truncation`.
- `VAMWeakViscosity(VAM)`: `default_scaling()` = orders (1, 2) in `Regime(De=-1, k_l=2, C_n=1)`:
  `1/De = O(eps)`, the bed drag at the order of the shear (NavierSlip `k_l = O(eps^2)` as in 49;
  Manning `C_n = O(eps)`: at `C_n = O(1)` the consistency check refuses ManningFriction, its
  traction `C_n De u|u|` being O(eps^-1) against the shear scale; at `C_n = O(eps^2)` the drag is
  dropped as beyond the order).  `Stress.full` is refused (`ValueError`: nothing more survives at
  these orders; use `VAM(stress=Stress.full)`).  Exported from `zoomy_core.model.models`.
- `InsertClosure` takes (i) a bulk closure without a `tensor` (it gives only the vertical shear:
  Elder, Bingham, k-epsilon, the thesis' calibrated eddy viscosities): nothing is substituted
  before the projection, it closes the horizontal rows' `tau_dz` afterwards as before, and the
  order must keep no other stress component, else `ScalingError` naming the closure and
  suggesting `VAMWeakViscosity` / SME (`ClosureRecord.shear_only`); (ii) bed / surface closures
  without scaling information (`dimensions`, `scaling_groups`: `RoughWall`, `WallFunctionBed`,
  `StressFree`): their traction enters after the projection as before, without the consistency
  check; `ClosureRecord.unchecked` and the derivation history name them.  Before, both raised,
  which kept SME / VAM with Elder, Bingham or RoughWall off the pipeline.
- `SME.default_scaling()`: orders (0, 0) at `Fr, De = O(1)` for `Stress.shallow`; `None` (the
  hand rule) for `Stress.full` (open, below).  `MLSME.default_scaling()`: (0, 0), `Stress.full`
  (2, 2); ML-SME's hand path deleted.
- ML-VAM: `vertical_traction` default `VerticalTraction.shared` (the stress `InterfaceFlux`
  value of the two traces: the mean, or `UpwindInterface(quantity=stress)` if listed); with
  `Stress.shallow` the w rows carry no tau_ez, so `shared` and `one_sided` are the same model
  (no longer a `ValueError`).  New `VerticalTraction.function` with
  `vertical_traction_function=(below, above) -> value` (sympy expressions of the two one-sided
  traces; ~10 lines in `ml_vam.py`); `function` without the callable, the callable without
  `function`, or a callable with `Stress.shallow` raise.  Cache keys: a two-argument callable is
  keyed by its effect on two probe symbols (`cache_keys._canonical`; before: its bytecode, blind
  to constants).
- `CACHE_VERSION` v10 (a v9 `VAM` entry holds the hand rule under the same spec key); prebuilt
  cache rebuilt.

**Deleted.**  VAM's `moment_scaling` / `apply_full_tensor` branch and its `scaling is None`
switch; ML-SME's `moment_scaling` / `apply_full_tensor` branch and its `pipeline` / `with_pi`
switches.  The functions stay: SME `Stress.full`, ML-VAM, ML-VAM-WD, KESME, QRKESME and Sigma3D
still call them (none of these is on the pipeline).

**VAM(1) rows that differ** (`dimension=2`, `[Newtonian(), NavierSlip(), StressFree()]`, state
`[b, h, q_0, q_1, r_0, r_1, P_0, P_1]`; goldens m12 = `VAMWeakViscosity`, m17 = `VAM`; rendered
SystemModels, everything else byte-identical):

```
VAM only (VAMWeakViscosity's r_0 row has no non-conservative entry):
nonconservative_matrix[4,1,0] = Mul(Integer(4), Pow(Symbol('h'), Integer(-2)), Symbol('nu'), Symbol('q_1'))
nonconservative_matrix[4,3,0] = Mul(Integer(-1), Integer(2), Pow(Symbol('h'), Integer(-1)), Symbol('nu'))
```

i.e. B[r_0, h] = 4 nu q_1 / h^2, B[r_0, q_1] = -2 nu / h (VAMWeakViscosity minus VAM:
-4 nu q_1 / h^2, 2 nu / h, as in 49): the depth average of `-d_x tau_xz / rho = -nu d_x d_z u`
with `d_z u = 2 q_1 / h^2` for the linear profile.  The r_1 row is unchanged
(`<phi_1, d_x d_z u> = 0` at level 1).  Test `test_vam_minus_weak_viscosity`.

**Byte-identity of the switches** (rendered SystemModels, no cache; main vs dev/vam-default): every
config identical.

| model | configs |
|---|---|
| `VAMWeakViscosity` = former `VAM` | levels 0, 1, 2 with `[Newtonian, NavierSlip, StressFree]`; `[Newtonian, StressFree]` 1-D and 2-D; Newtonian + Manning; Elder(friction=RoughWall) + RoughWall; NavierSlip only (unclosed bulk); unclosed; goldens m11, m12, m13, S02, N02, P01 unchanged; the thesis Steffler `make_model(model="vam")`, levels 1, 2, bulk elder / elder_live / elder_fix / ghamry / newtonian / none x bed chezy / navier, + nu_h: all rows identical (one `bc_kernel` line reads `Float(1.0)*n0` or `n0` depending on the build order within one process, with either library) |
| SME (pipeline (0, 0)) = hand rule | levels 0, 1, 2; 2-D; SWE + Manning; ElderSME; Elder(friction) + RoughWall; Bingham (quadrature 8); Newtonian + RoughWall; NavierSlip only; unclosed; `Stress.full` (hand, unchanged); goldens m01-m08 |
| ML-SME (pipeline) = hand rule | 2 layers level 1 (mean / upwind), 3 layers level 0, 2 layers level 0 2-D, the thesis Bingham plug layer (`level=[2, 0]`, quadrature 12), `Stress.full` level 1; golden m10 |

**Refusals that are new.**  `VAM` with a shear-only bulk closure (Elder, calibrated): the
vertical row keeps `d_x tau_xz`, which the closure cannot give -> `ScalingError` naming
`VAMWeakViscosity`.  `VAMWeakViscosity(stress=Stress.full)` -> `ValueError`.  The default VAM
without a bulk closure (unclosed, NavierSlip only) keeps the unclosed `d_x tau_xz` modes in its
w rows (an open system as before, with more open terms).

**SME `Stress.full`: options (not changed).**  The hand rule is the consistent truncation for
`De <= O(eps)` (Marche 2007 (5.1); byte-identical to the pipeline at orders (1, 1),
`Regime(De=1)`, test `sme1_full_marche`), not for the thesis regime.
(a) Keep the result, move it onto the pipeline: `default_scaling()` = (1, 1) at `Regime(De=1)`
for `Stress.full`; byte-identical, deletes SME's last hand path (`moment_scaling`,
`apply_full_tensor`, the hydrostatic `zdrop`); `Stress.full` then reads "Marche's regime".
(b) The thesis regime: keeping the in-plane stress at `De = O(1)` needs horizontal order 2;
(2, 2) keeps the eps^2 vertical inertia (refused by the hydrostatic SME), (2, 0) drops the
vertical stress and tau_zz: SME(0) gains Marche's `2 nu grad(h div u)`
(`diffusion_matrix[q_0, q_0, x, x] = 2 nu`, `[q_0, h, x, x] = -2 nu q_0 / h`, a source term, 49).
C26, C32 (`u_t = nu u_yy`, "no factor 2"), C34 and C36 are built on `SME(stress=Stress.full)`
and would move.
(c) (a) plus a separate class for (b) (as `VAMWeakViscosity` for VAM), or (b) as the default and
a separate class for the present one.

**Thesis** (`mig/vam-default`, worktree `~/git/wt-thvamdefault`): every non-legacy case script and
notebook under `cases/`, `notebooks/` that builds or subclasses VAM uses `VAMWeakViscosity`
(Python NAME tokens only; strings, comments, chapters untouched; jupytext pairs both; the GUI case
cells' `class_path` and `cases/vam/bump/make_cases.py`), so every thesis number keeps the hand
rule; at nu = 0 (most VAM cases) both classes give the same numbers.

**Zoomy cases.**  Matrix, gates, docs examples: `VAM` (nu = 0 everywhere but C37's
`Stress.full`, which is unchanged); the GUI's VAM card (nu = 0.1, a demo) takes the new default.
gate48 iterates `(one_sided, shared)` explicitly.  Matrix: `MATRIX_ONLY=<regex>` restricts a
command to the matching run keys.

**Checks.**  zoomy_core 505 passed (default tier); large tier: m05_sme_2d fails on main as well
(eigenvalues None vs the stored closed form), unrelated.  zoomy_jax: only timing gates fail
(ahs26_small, ritter_dry, bc_runtime, lake_at_rest; a different subset on each run, on main and on
the branch, under load).  QR: 27 passed, 3 skipped (2 by design: jax / amrex solver cards in Pyodide;
the manifest test: no thesis checkout in the worktree).  Matrix (private store, rerun of the VAM /
Chorin rows; stopped by scope after C31 numpy / jax): C8 (all runs; numpy, jax, foam, amrex;
serial + mpi2), C10 (all, serial), C15 chorin (jax serial + mpi2), C16 vam_* (numpy, jax), C31
vam_* (numpy, jax): 0 FAIL; changed rows only at the pressure-tolerance level (e.g. C8 vam AMReX
vs JAX 1.8e-11 -> 4.6e-11, foam L1 1.1e-11 -> 2.9e-11; nu = 0 in these cases, so only the CSE /
operation order of the kernels moved).  Not rerun: C16 / C31 foam + amrex, C35 c8_*, C37 vam_*,
mpi4 / gpu configs (main reruns the matrix after the merge).  gate 48: MLSME 0 / 1, MLVAM 0 / 1
all 0 (its per-layer reference is the one-sided trace, now pinned explicitly).

**Open.**  ML-VAM is not on the pipeline: with `Stress.shallow` its w rows keep the hand rule,
so ML-VAM(1 layer) reduces to `VAMWeakViscosity`, not to the new `VAM` (gate 48's reductions use
`Stress.full`, unchanged).  Wiring ML-VAM (per-layer `ordered_truncation`, as ML-SME) would make
the VAM family consistent.  SME `Stress.full` above.  No `VAMWeakViscosity` GUI model card.

## 64. HLL NaN on a dry bed (thesis `speed`, step 5): a dry-dry face's speed was -0 (2026-10-21, dev/hll-dry-nan)

Report (thesis `cases/speed`, `migration_status/step5.md`): `PositiveNonconservativeHLL` (`hr`) on the
circular dam break onto an exactly dry bed (SWE 2-D, closed-form eigenvalues) gave t = NaN on NumPy
and foam (phases 1-2, o1 and o2 + MOOD), finite on JAX and AMReX.  The step-5 port ran on zoomy_core
9308483 (2026-10-14).

- **Reproduction** (speed's `run.py` copied, 8 x 8 or 32 x 32, t 0.01, NumPy, `np.seterr(invalid=
  "raise")`): the first invalid operation is `dt*(source - divergence)` in the march (`0 * inf`);
  one line earlier the printed CFL step `dt = reduce_min(cfl r_f / S)` is `-inf`: `S`, the
  `numerical_face_sides_speed` row of `HLL.face_wave_speed`, is `-0.0` at a dry-dry face, `1/-0 =
  -inf`, time `-inf`.  The HLL flux's `1/(s_R+ - s_L- + 1e-14)` guard is not involved.
- **Cause.**  `face_wave_speed = Max(-Min(s_L, 0), Max(s_R, 0))`.  Dry-dry: `s_L = s_R = 0`, so
  `-Min(s_L, 0) = -0.0`, `Max(s_R, 0) = +0.0`; sympy stores the arguments as `(Max(s_R, 0),
  -Min(s_L, 0))`, i.e. the face speed is `max(+0, -0)`, and the tie of signed zeros is resolved per
  backend by the callee its printer emits:

  | backend | printed max | tie rule | `max(+0, -0)` | 1/S, dt |
  |---|---|---|---|---|
  | NumPy (lambdify) | `numpy.maximum` | second argument | -0 | -inf, NaN |
  | foam (kernel lane) | `Foam::max` = `(s1 > s2) ? s1 : s2` | second | -0 | -inf, NaN |
  | AMReX (kernel lane) | `amrex::max` = `std::max` = `(a < b) ? b : a` | first | +0 | +inf, finite |
  | JAX | `jnp.maximum` (XLA max: -0 < +0) | +0 | +0 | +inf, finite |

  foam matches NumPy because `Foam::max` and `numpy.maximum` share the tie rule, AMReX and JAX
  differ for two different reasons.  Not a `where` evaluating both branches, not `std::max`
  swallowing a NaN: the values are equal and only the sign bit differs.
- **Fix (already on main).**  `HLL.face_wave_speed` returns `Abs(Max(...))` (core e82751e, section
  52c, merged with dev/numerics-robust 7d086ff, 2026-10-14 after 9308483): `|.|` is `+0` on every
  backend, the speed is `max(|s_L|, |s_R|) >= 0` by definition, so it falls out of the scheme (no
  clamping, no tuning).  The step-5 port predates it.  Today's main (core fe048d8): speed n 32 t 0.1
  o1 / o2 + MOOD, `hr`, PASS on all four; numpy vs jax / amrex / foam max rel 2.5e-16 / 3.3e-16 /
  2.2e-16 (o1), 2.3e-14 / 2.4e-14 / 2.5e-14 (o2); mass 1.5e-16.  On main a step of -inf no longer
  passes silently either: the march stops (`time_4 > time_3` false) and `check_march` raises.
- **This branch: regression only** (core 04a953b) (the face speed is the wave-speed code of dev/wavespeed-frame; no
  library change here).  `tests/symsolver/test_hll_dry_face.py`: the NumPy-printed face speed of
  `PositiveHLL` / `PositiveNonconservativeHLL` at a dry-dry face is `+0` (signbit) for five normals,
  and the SWE 2-D dry dam break (8 x 8, o1, both fluxes) reaches t_end finite under
  `np.errstate(invalid="raise")` with mass to 1e-12; all 4 fail on 9308483 (-0.0, `FloatingPointError:
  invalid value encountered in multiply`), pass on main.  Matrix: C5 gains the plain-SWE dry dam break
  `swe_{PositiveHLL, PositiveNonconservativeHLL}_o{1,2}` (C5's geometry, nx 48, t 0.02, CFL 0.45,
  O2 + MOOD; `cases.dry_swe`) on the four backends: h_min >= 0, t_end reached, AMReX / foam = JAX to
  round-off, numpy = JAX and C6 mass via the generic checks.  C3 + C5 serial, four backends
  (private store): 123 PASS, 0 FAIL; new rows: t_end exact everywhere, h_min 0, AMReX / foam vs JAX
  9.6e-16 / 2.7e-15 (O1), 8.0e-14 / 8.8e-14 (O2 + MOOD), both fluxes; C6 C5 mass 1.5e-13 (all four).
- **Note** (not changed): the kernel lanes still differ in their `max` tie rule (AMReX first, NumPy /
  foam / the solver lane's `maximum` second), so any printed expression whose value can be a signed
  zero that is then divided by is backend-dependent; an `Abs` (or a sum `x + 0.0`) at the source is the
  portable form.

## 65. JAX start-up: what the 270-300 s compile and the 115-140 s setup are; the cache with host callbacks; xreplace in the Riemann build (2026-10-21, dev/jax-compile)

Reproduction: `cases/jax_compile_time.py` (setup split, the march's trace / lowering / XLA compile,
its size, its callbacks, cache hits; `BREAKDOWN=` equations per kernel, `PROFILE=1` cProfile).
Cases: C1 `o2_n40` (hyperbolic), C8 `vam` (VAM(1) Chorin), ANALYSIS 59's Escalante bump (ML-VAM(2)
level 1, 14 states, 60 cells, hr, O2 + MOOD, SSP-RK2, dt 1e-3, `on_write`).  `taskset -c 24-31`,
GPU 1 (L40S), jax / jaxlib 0.9.2, float64.

**The march program** (before; one program per run, the `lax.while_loop` of the template):

| case | device | trace | lower | XLA compile | jaxpr eqns | StableHLO ops | HLO after opt. | callbacks |
|---|---|---|---|---|---|---|---|---|
| C1 o2_n40 | CPU | 2.6 s | 0.6 s | 2.7 s | 12.1k | 10.7k | 9.9k | 1 |
| C8 vam | CPU | 2.4 s | 0.7 s | 6.8 s | 23.6k | 20.8k | 30.4k | 2 |
| C8 vam | GPU | 2.2 s | 0.7 s | 51.8 s | 23.6k | 20.8k | 27.9k | 2 |
| Escalante | CPU | 13.0 s | 3.6 s | 33.6 s | 113.6k | 75.0k | 191k | 12 |
| Escalante | GPU | 12.5 s | 3.6 s | 300-303 s | 113.7k | 75.8k | 163k | 10 |
| Escalante, legacy `run_jit_steps` (thesis 40c26762) | GPU | 7.6 s | 1.9 s | 228 s | | | 88k | |
| same, legacy | CPU | 12.3 s | 2.7 s | 44-54 s | | | 127k | |

- **Expected for its size.**  The GPU compile is ~1.8 ms per optimised HLO instruction (C8 and the
  Escalante march alike; the CPU ~0.18 ms), and the legacy solver, with 54 % of the instructions,
  compiled in 228 s.  XLA's split (`TF_CPP_VMODULE=gpu_compiler,nvptx_compiler,hlo_pass_pipeline`):
  Escalante HLO passes 241 s, of them **`priority-fusion` 220 s** (superlinear: C8 15 of 18 s),
  IR emission 24 s, LLVM -> PTX 7-8 s, ptxas 9 s; legacy: priority-fusion 148 s.  The fusion pass is
  serial: `--xla_gpu_force_compilation_parallelism=8` 215 s.  Changing fusion (other passes,
  limits) changes the kernels, so not the results bitwise; not done.
- **What the program is.**  55 % of the Escalante equations are the lambdified sympy kernels (already
  CSE'd), inlined at each call site: `numerical_cell_edge_source` 26.9k (6 calls: 2 SSP-RK stages x
  {O2 predictor, MOOD's O1 re-step, ...}), the face kernel `numerical_face_sides(_speed)` 22.4k
  (6 calls), the pressure / corrector slot operators 17.9k; the LSQ derivative aux ~10k, the limiter
  ~5k, the eigensolve ~2.5k.  Not constants: hoisted_jit passes 1991 arrays (155 kB) as arguments,
  XLA constant folding is off (section 20), geometry is an input (section 26).  The way to a
  smaller program is the template's: one kernel call per stage for O2 and the MOOD fallback (one
  batch over both face-state sets) and the stages as a loop instead of unrolled; each halves the
  kernel copies.  Not done here (template, all backends).
- **Host callbacks block the persistent cache: confirmed, all kinds.**  In jax 0.9.2 every Python
  callback (`jax.debug.callback` ordered or not, `io_callback`, `pure_callback`) lowers to an
  `xla_ffi_python_{cpu,gpu}_callback` custom call and registers its callable with the compile;
  `compiler._cache_write` writes nothing for an executable with any ("uses host callbacks").  The
  march has: `check.cfl` (in a `cond`), `check.march` (ordered; whenever a CPU device exists, so
  in every run), `io.write` (`on_write`, `output`), `io.save_state` / `io.load_state`, the linear
  solves' accept gate (in a `cond`), host linear solvers (PyAMG / hypre `pure_callback`), preCICE
  (`io_callback`).  The cache key is the same in two processes (C1: identical); all small
  programs of the setup do hit (C1 147 of 148, C8 195 of 196).
- **Setup.**  Escalante 117-140 s, of which `Numerics._model_eval` 99 s: `sympy.subs` of the
  model operators (quasilinear matrix, numerical eigenvalues, NCP matrix) at the face states, 155
  calls (56 distinct).  C8: 20 of 31 s.

**Fixes.**

1. zoomy_core `riemann_solvers._substitute`: `_model_eval` substitutes through `xreplace` when
   that builds the same tree as `subs`: every key a `Symbol`, no value containing a moved key
   (sequential `subs` would substitute into it), no `Derivative`, `Subs`, integral / sum, `cot`,
   `AccumBounds` (whose `_eval_subs` differs from a rebuild); an immutable array through its flat
   entries (its own `has` compares arrays elementwise).  Else `subs` as before.  Checked call by
   call against `subs` (sympy `==`, entry by entry): C1 o2_n40, C8 vam / vam_tri / per, C10
   s2_t2_n30: 609 calls, 0 differences; 3.0 s against 95.7 s.
2. zoomy_jax `symsolver/compile_cache.py`: inside `callback_cache()` (the first call of
   `hoisted_jit`, i.e. every march compile) an executable with host callbacks is written to
   `jax_compilation_cache_dir` under jax's own cache key (`zoomy-cb-<key>`, atomic write) and a hit
   is loaded with `backend.deserialize_executable(..., host_callbacks=)`, the callables of the
   present lowering (the custom calls name them by index, the same order for the same module).  It
   wraps `jax._src.compiler.compile_or_get_cached` (checked signature; any failure: jax's own
   compile); active only with a cache directory set (`JAX_COMPILATION_CACHE_DIR`), opt-out
   `ZOOMY_JAX_CALLBACK_CACHE=0`.  Test `tests/symsolver/test_compile_cache.py` (ordered and
   in-`cond` callbacks fire after a load, result bitwise).

**Before / after** (Escalante to t = 0.1, GPU 1; `JAX_COMPILATION_CACHE_DIR` set):

| | setup | trace + lower | XLA compile | run (100 steps incl. compile) |
|---|---|---|---|---|
| main | 117-120 s | 16 s | 300-303 s | 320-323 s |
| fix, first run (cache written) | 20.7 s | 16 s | 303 s | 323 s |
| fix, next runs (cache hit) | 12.5 s | 9.6 s | 12.5 s (load) | 24.2 s |

The final state of the cache-hit run equals the compiling run's bitwise; frames written by
`on_write` the same.  Two separate GPU compiles of the same program differ by 3.8e-15 (rel; GPU
atomics / autotuning), so GPU runs are compared to round-off, CPU runs bitwise.  C1 CPU: compile
3.4 s -> 0.31 s, final state bitwise.

**Acceptance** (private store, jax serial + gpu1, vs main's code run the same day): C1 (6 runs),
C8 (6), C10 (9) serial **bitwise**, with the cache written (run 1) and loaded (run 2); gpu1 C1
o2_n40 1.4e-13 / 5.6e-14, C8 vam 1.4e-11 / 7.6e-12 (rel; GPU round-off, section 59's 1e-9 tolerance;
main vs main on gpu1 not rerun).  Wall per run, cache loaded: C8 serial 52-75 s -> 29-44 s, C8
gpu1 110 -> 38 s, C10 56-133 s -> 27-90 s.  XLA:CPU logs `cpu_aot_loader` feature warnings
(+prefer-no-gather / scatter) when it loads cached CPU programs (jax's own entries too).
Not run (wrap-up): zoomy_core / zoomy_jax suites, QR suite.

## 66. Two silent ignores of the JAX solver are errors: `ad_mode` after `setup`, snapshots in reverse mode (2026-10-21, dev/ad-traps)

Defects (`zoomy_jax/symsolver/solver.py`, `runtime.py`):

1. `ad_mode` is read once, in `_setup` (passed to `R.runtime`, which picks `lax.while_loop` or the
   checkpointed equinox loop).  `solver.ad_mode = AdMode.reverse` after `setup` changed nothing;
   `jax.grad` then failed deep inside the plain `lax.while_loop`.
2. `AdMode.reverse` skipped `on_write` (`io_write`: `ad_mode == "forward"` guard) and, through
   `output`, the HDF5 snapshots, with no message.

Fix (no change otherwise; the guard in `io_write` is gone because `runtime` now refuses the case):

- `setup` (hyperbolic and Chorin) calls `_check_ad_options` first: `ad_mode=AdMode.reverse` with
  `on_write` or `output` set raises `ValueError` (naming the options set), before `_writer()`
  opens the HDF5 file.  `R.runtime` itself raises for reverse with `on_write`.
- `_setup` records `_built_ad_mode`; `run` raises `ValueError` when `ad_mode` differs ("ad_mode is
  read once, at setup. Set ad_mode before setup").

Test `tests/symsolver/test_ad_traps.py` (6 cases, 1-D SME(0) dam break, 20 cells, CPU, ~90 s mostly
compile): fails 4 of 6 on the old code; mode change in both directions, reverse with `on_write` /
`output`, set-before-setup `jax.grad` equals the jvp, forward `on_write` unchanged.  Bitwise:
final `(Q, Qaux)` and the reverse gradient hash identically before / after (forward, reverse,
forward with `on_write`).  Gate 2: forward vs central FD 9e-10 ... 4e-7, reverse vs jvp <= 1e-14
(J = 2.5325231268e+01, 6.0680499168e-01).  QR suite: 27 passed, 3 skipped by design (no thesis
checkout in the worktree; two solver cards need the jax / amrex backends).

Still silent, not changed (outside the request): `checkpoint=` (restart file) in reverse mode
skips `on_save` the same way; `ad_checkpoints` changed after `setup` is likewise ignored.

## 67. Function hooks: user functions inside a model's operators (2026-10-21, dev/function-hooks)

A model's operators may call `hook("name")` (`zoomy_core.symsolver.hooks`): an opaque SymPy function
(`_is_hook`, real) that the derivation keeps opaque (never promoted to an aux field:
`expose_aux_atoms`, `Model._is_field_application`, `system_extract._collect_fields` skip it).  The
shared setting `HyperbolicSettings.functions` (name -> callable / `sp.Lambda` / `HookFunction`;
names identifiers, values callables) implements it.  Gradients enter as the existing
derivative-aux symbols.  Typical use: a closure trained elsewhere (fixed weights).

- **NumPy** (`numpy_backend/kernels.py`): the hooks join the lambdify module; **JAX**
  (`jax_runtime._lambdify_array`, the one lambdify site incl. `runtime._numerics_face`): the same,
  per cell under `vmap`.  Setup checks names both ways (`check_functions`: a hook without a
  function, a function no operator calls; "did you mean").
- **Derivatives.**  Wherever the pipeline differentiates an operator (source Jacobians, eager in
  the SystemModel; the quasilinear matrix and closed-form eigenvalues of a hook flux) a hook gives
  `Derivative(f(..), a_i)` / `Subs(Derivative(..))` (now allowed past the REQ-130 trace guard when
  it is a hook's).  `Hooks.lower` prints it per `HookFunction.derivative`:
  `HookDerivative.zero` -> 0; user partials (a list, one per argument, or one callable returning
  all) -> called like the function (first derivatives only); `HookDerivative.auto` (default for a
  plain callable) -> a `sp.Lambda` is inlined and SymPy differentiates it (every Python backend), a
  callable is differentiated by `jax.grad` (nested per order) on JAX, refused on NumPy: the kernel
  becomes a stub and setup raises `HookError` if the printed program calls it (`nsm.<slot>`), so an
  unused Jacobian kernel costs nothing.  The template's IMEX (`Root`) differentiates the residual
  itself (JAX AD, NumPy FD), so explicit / `Imex.lie` / `Imex.ssp2` need no declared derivative;
  `SourceTreatment.linearized` prints dS/dQ and does.
- **foam / AMReX** (`CxxBackend._prepare`, `CxxChorinBackend._prepare`): `functions` set, or a
  hook in any kernel of the systems -> `NotImplementedError` naming the function, the operators
  that call it and "supported on numpy, jax", before printing.
- **Default unchanged.**  Without `functions` no hook code runs (no scan: it costs 5 s on SME level
  4).  Printed programs of C1 and C14 (o1, o2; NumPy, JAX, foam and AMReX host programs) and the
  NumPy kernel sources: sha256 identical on main and the branch.

Tests (`zoomy_jax/tests/symsolver/test_function_hooks.py`, 14, 30 s): 1-D SWE, Manning as
`manning(h, hu)`, 50 cells, t 0.5.  Hook vs built-in Manning, NumPy and JAX: explicit 0, `Imex.lie`
<= 3.6e-16, `Imex.ssp2` 0 (the friction moves Q by > 1e-4).  `SourceTreatment.linearized`: user
partials (both forms), `sp.Lambda` = built-in <= 3.3e-16 on both, JAX `auto` = user partials
<= 1e-13, `zero` runs and differs by 6.0e-7 (the dropped friction Jacobian), NumPy `auto` of a
callable raises at setup.  Hook flux (`pflux(h, hu)` = the momentum flux) on JAX, dam break o1 / o2:
1.1e-15 / 2.1e-15 (eigenvalues by `jax.grad` of the hook).  foam and AMReX `run` raise.  Tutorial
`docs/book/tutorials/hooks/function_hook.ipynb` (JAX function, assert hook = built-in).
