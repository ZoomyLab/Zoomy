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

`wave_speed="cells"` (default): eigenvalues of the cell states of each face
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
- Default `pressure_linear_solver="auto"`: amgx on a GPU with jaxamg.

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
- `pressure_linear_solver="auto"`: amgx on a GPU, pyamg on the CPU, when
  installed; else jax.scipy GMRES.  Chorin gates 7/8/12/13 pass on pyamg
  (vs old 8.7e-13, AD rev-jvp 4.8e-12, foam 1.0e-12).

## 15. Reconstruction pair (h, eta)

`ReconstructionSpec.bed = "derived"` (default): with the model's
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

`SystemModel.spectral_radius_method = "eig" | "power" | "bound"`; "power":
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
- **IMEX-SSP2(2,2,2)** (Pareschi & Russo 2005) as `imex="ssp2"`: SWE + Manning time order 2.01-2.08
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
  phi_k(side) / mu_k like the u* transfer), s* = `InterfaceFlux(quantity="stress")` of the two
  traces: `MeanInterface` by default, `UpwindInterface(quantity="stress")` by the sign of G;
  `quantity="velocity"` (default) keeps the transfer-velocity meaning.  The interface traction
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
- **Implementation** (`chorin.py`, `Projection.increment`, OPT-IN (default `standard`); `projection="standard"`
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
  the compound-aux pattern 45b found; backends likely read it as 0.
- **Open.**  ML-VAM w rows: the vertical balance is pointwise per layer, so its interface traces
  of tau_ez are one-sided (single-valued for continuous profiles only); a single-valued vertical
  traction there is a model choice.  ML with `Stress.shallow` keeps the one-sided interface shear
  (45f item (4)), unchanged by spec.  `interpolate_rows` slot 5 is still the hydrostatic part
  (45f item (2)).  small_slope=False with the full stress is not run.

## 46. JAX distributed pressure solve (pyamg): the iteration count does not scale (2026-10-13)

L2P's VAM Chorin at nx 256 (65k cells), `pressure_linear_solver="pyamg"`, 3 constant steps
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

Result (dev/jax-hypre): `pressure_linear_solver="hypre"` (zoomy_jax `symsolver/hypre.py`: ctypes on
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
