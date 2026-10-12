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
