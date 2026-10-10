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

`zoomy_core/symsolver/reconstruction.py`; blocks in `vocabulary.py`, bodies in
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
