# Solver template: one framework for all backends (design for approval)

Status: DRAFT for user approval, 2026-10-07. Implementation happens on a feature branch only after approval.
Supersedes `2026-07-20-solver-flowcharts-phase1.md` (v6). v6 was never adopted by a backend. This design reuses its IR
code (`zoomy_core/solver/ir.py`, `march.py`, `external.py`, walkers `transformation/procedure_{c,python}.py`) and
extends it.

Evidence base (session b5667895, 2026-10-07):
- 6 backend investigations: v6 core, numpy, jax + preCICE, OpenFOAM, AMReX + dmplex, firedrake/fenicsx/dune.
- 7 stress tests: implicit/DAE, preCICE, AD, splitting/Chorin, AMR/MPI/GPU, FEM/UFL, adversarial critic.
- 4 feasibility prototypes (scratch only): traced IR, C walker, numpy shared bodies + BC contract, jax AD mechanisms.
- Notes: `scratchpad/synthesis_notes.md`. Scripts: `scratchpad/feas_*`, `scratchpad/critic`.

---

## 0. Decision and principle

- **One template framework for all backends**: the user's decision of 2026-10-07.
- **Variants are fine.** Not every combination must work. A *solver variant* is one feasible combination of template +
  stages + plugins + backend. Impossible combinations are refused **at build time** by capability checks. The goal is
  shared core code and aligned backend contracts, not one solver that does everything.
- **What is shared:**
  - the template (time loop, step, stage sequencing, MOOD, guards, output cadence);
  - the block contracts (signatures, semantics, comm class);
  - the Python block bodies (one `xp` body for numpy and jax);
  - the time-scheme data (tableaus);
  - coupling physics (ghosts, interface fluxes);
  - constants (`march_constants`).
- **What stays per backend:**
  - native bodies (mesh loops in C++, halos, Krylov/PETSc/AMReX solvers, IO, preCICE API calls);
  - backend shells where the backend owns structure (AMReX level recursion, PETSc TS as a `NativeStep`, firedrake forms).

## 1. What the evidence says

| Area | Verdict | Key numbers |
|---|---|---|
| Splitting / Chorin | GAIN | 5 Chorin implementations with 5 semantics → 1 stage list; ~4600 → ~1900 driver LOC; time order 2 for free in C++ |
| Implicit / IMEX / DAE | GAIN | implicit source composed 4 different ways across 6 backends today; ~5900 → ~1700 LOC; AD tests (ARS232/343, BDF2, ROS2) fwd=rev=FD ≤ 3e-10 |
| preCICE | GAIN (5 additions) | foam + jax ~2.9k → ~1.0k LOC; ~52 % of `PreciceManager.H` is backend-independent physics; a new participant costs ~300 LOC, not a 1.1k clone |
| AD | GAIN | AD constraints cost numpy/C++/FEM nothing at runtime; two of them fix existing numpy/jax divergences |
| AMR / MPI / GPU | GAIN (3 bounded limits) | Berger–Oliger subcycling stays a backend shell; jax AMR needs padding or a recompile per regrid; regrid is not differentiable |
| FEM / UFL | firedrake GAIN, dune GAIN (fwd only), fenicsx EXCLUDE | firedrake −850 LOC; a shared UFL form builder fixes 3 dune bugs; DG1 becomes 2nd order in time |
| Critic | layers sound; favoured a Python-only template | concerns: debuggability of generated code, size of IR, C++ shells. Addressed in §3.4, §9, §11 |
| Feasibility: traced IR | PASS | Select / Guard / Callback / Root / checkpointed While on the real step: bitwise vs references; traced loop 2.3 → 0.8 ms/step; Chorin composite bitwise; traced `time_end` 3 → 1 compile |
| Feasibility: C walker | PASS after 8 fixes | real emitted march compiles (`-Wall -Wextra`), runs 10 steps, matches an independent reference to 4e-16; ~400 LOC core |
| Feasibility: shared numpy/jax bodies | PASS | one body: numpy 45× (1-D) / 13× (2-D) faster and bitwise equal to today's numpy; jax.jit of the same body is faster than today's jax solver |

## 2. Three layers

```
TEMPLATE (core IR, written once)        what happens in which order: march, step, stages, MOOD, guards, output
   │  statements: Seq Assign Call Hook While Repeat Select Root LinSolve Guard Callback IfStatic ForEachFace
   ▼
STAGES + PLUGINS (composition)          which physics/features: Hyperbolic, ImplicitSource, ImplicitDiffusion,
   │                                    Elliptic, Pointwise, ...; PreciceCoupling, Observer, Output
   ▼
BACKEND (lowering + bodies + caps)      how it runs: numpy, jax-fwd, jax-rev, jax-sharded, foam, amrex, dmplex,
                                        firedrake, dune; capability set checked at build time
```

## 3. Template IR

### 3.1 Statement kinds

| Statement | Meaning | numpy | jax (fwd / rev) | C++ | firedrake |
|---|---|---|---|---|---|
| `Seq(...)` | sequence | — | — | — | — |
| `Assign(x, expr)` | scalar or whole-array assignment | `x = expr` | `x = expr` (functional) | elementwise backend hook (MultiFab `LinComb`, field algebra, loop) | `Function.assign` (handle mode) |
| `Call(proc, in, out)` | call a block or sub-Procedure | call | call | call | call |
| `Hook(name, in, out, default)` | named extension point; plugins fill it, default body otherwise | as Call | as Call | as Call | as Call |
| `IfStatic(flag, a, b)` | build-time branch | resolved away | resolved away | resolved away | resolved away |
| `While(cond, body)` | runtime loop, dynamic count | `while` | `lax.while_loop` / equinox checkpointed (rev) | `while` | `while` |
| `Repeat(n, body)` | runtime loop, static count | `for` | `lax.scan` / `fori_loop` (static) | `for` | `for` |
| `Select(pred, a, b)` | runtime branch | `if` | `lax.cond` | `if` | `if` |
| `Root(residual_proc, x0, opts)` | nonlinear solve; residual is a sub-Procedure (closure) | Newton | `custom_root` (implicit function theorem) | Newton / SNES | `NonlinearVariationalSolver` |
| `LinSolve(matvec_proc, rhs, opts)` | linear solve; matvec is a sub-Procedure | scipy gmres / LU / banded | `custom_linear_solve` + gmres / tridiagonal | KSP / AMReX GMRES / BiCGStab | `LinearVariationalSolver` |
| `Guard(ok, msg)` | fatal check | `raise` | `eqx.error_if` (+ ordered debug print of values) | abort with diagnostics | `raise` |
| `Callback(fn, in, out, kind)` | side effect: idempotent / stateful / returns-data | call | ordered `io_callback` with zero tangent; stateful ones only at host level | call | call under `stop_annotating` |
| `ForEachFace(count, body)` | face traversal (only where core emits face bodies) | vectorised | `vmap`; count static | backend loop (`ParallelFor`, `forAll`, `for`) | UFL `dS`/`ds` |

### 3.2 Nesting
- **Build-time composition**, unrolled with zero runtime cost:
  - Procedures call Procedures;
  - Python loops emit repeated statements (RK stages, Strang halves);
  - `IfStatic` is resolved at build time.
- **Runtime nesting** to any depth with `While`, `Repeat`, `Select`, `Root` and `LinSolve`. Example: coupling window
  `While` ⊃ time-step `While` ⊃ step ⊃ stages ⊃ `Root` ⊃ Krylov.

### 3.3 Host / compiled boundary rule
- The outermost loop that contains a **stateful** Callback runs on the host. Examples: the preCICE window loop, an AMR
  epoch loop.
- The innermost loop free of them is compiled as one unit (one jitted `While`).
- C++ has no boundary.
- Reverse AD is allowed only inside compiled units. Gradients chain per window or per epoch.

### 3.4 Readable generated code (answers the critic's debugging concern)
- The Python walker emits **source files**, not closure towers. Example: `build/generated/step_ssprk2_jax.py`, with
  comments naming the template statement each line comes from. Stack traces end in readable Python.
- The C walker emits C++ text (already does).
- `solver.show()` prints the assembled template for teaching and review.

### 3.5 State carried through the loop (one pytree / struct)
`time, iteration, dt, dt_prev, Q, Qaux, acc (observers), i_snapshot, history (multistep), stats (Newton/Krylov
iterations, rel_resid), warm starts, P (plugin state), carry (stage-owned fields, e.g. a separate pressure)`.
- A rollback is `Select(reload, ck, state)` over the **whole** state. That makes implicit-coupling rollback, step
  rejection and restart correct by construction. This fixes a foam bug: its checkpoint does not restore the write
  counter.

## 4. Block contract (ExternalProcedures)

Each block declares: inputs → outputs, semantics, comm class (`none` | `neighbor` | `global`), and who implements it.
"xp" means one shared body for numpy and jax.

| Block | Signature (in → out) | Comm | Implemented by | Replaces today |
|---|---|---|---|---|
| `prepare_mesh` | (mesh, nsm) → (topology, geom) | none | topology: numpy; geom: xp (geometry from node coordinates, differentiable) | `base_mesh` loops, `lsq_mesh`, `convert_mesh_to_jax`, 4× `d_face`, 4× face inradius |
| `build_operators` | (nsm, topology) → ops | none | per backend (printers) | runtime builders |
| `initialize_state` | (nsm, geom) → (Q, Qaux) | none | xp / native | — |
| `compute_dt` | (Q, Qaux, p, geom) → dt_cfl | global | xp (λ per face, local CFL, safe divide) + `reduce_min` | 4+ dt laws |
| `halo_bc` | (Q, Qaux, p, t_stage, bdata) → bf_values | neighbor | xp (batched BC kernel) / native (FillPatch, processor patches, DMGlobalToLocal) | per-face loops; `_halo_wrap` |
| `reconstruct` | (Q, Qaux, bf_values, o1) → (Q_L, Q_R, grad) | none | xp (LSQ-MUSCL, limiters, eta/W, positivity) | 7 numpy/jax class pairs |
| `flux_pass` | (Q_L, Q_R, Qaux_L, Qaux_R, p, t_stage) → (F, Dp, Dm) | none | xp (batched Riemann kernels) / native | numpy per-face loop; jax vmap |
| `cell_ncp` | (Q, grad, Qaux, p) → cell_term | none | xp | 2 different discretisations |
| `gather_update` | (Q, F, Dp, Dm, cell_term, Qaux, p, t_stage, dt) → (Q_cand, troubled) | none | xp (`scatter_add`) / native | scatter code ×5 |
| `update_aux` | (Q, Qaux, p, t) → Qaux | neighbor (gradient aux) | xp | `update_qaux` ×3 |
| `update_variables` | (Q, Qaux, p, t, dt) → Q | none | xp | `update_q` ×3 |
| `source` / `diffusion_op` | model kernels | none | printers | — |
| `elliptic_residual` | (P, Q, Qaux_in, p_stage, t, dt) → R (sub-Procedure for `LinSolve`) | neighbor | xp / native (+ assembled blocks for AMG / PETSc Pmat) | 5 pressure steps |
| `write_fields` | (Q, Qaux, time, i_snapshot) → () | — | Callback, idempotent by i_snapshot | 4 writers |
| `gather_interface`, `couple_*` | see §7 | global | plugin bodies per backend | `PreciceManager.H`, `precice_solver.py` |

Required changes recorded by the investigations:
- ExternalProcedure declarations become the **single typed source** of signatures. Delete `SOLVER_ARG_SLOTS`; check arity
  and kind at print time.
- `Procedure.results` is added; values in/out are passed once; a context handle gives bodies access to workspace.

## 5. Hooks (default = no-op)
`face_fixup` (coupling), `post_stage` (DG limiter, positivity), `post_step` (`update_variables`), `observe`,
`face_sync` / `flux_accumulate` / `level_sync` (AMR), `couple_*` (via plugin), `precond` (Krylov).

## 6. Time schemes (data in core, unrolled at build time)

| Scheme | Data | Structure |
|---|---|---|
| Euler, SSPRK2, SSPRK3 | Shu-Osher (a_k, c_k) | `Q^(k) = a_k·Q0 + (1−a_k)·stage(Q^(k−1), t + c_k·dt)` |
| IMEX-ARK (ARS232/343, ARKIMEX-2e) | Butcher (A_E, A_I, b, c) | per stage: `Root` if a_ii ≠ 0, then explicit stage |
| Rosenbrock | (α, Γ, b, c, γ) | `LinSolve` per stage (needs the `jvp` primitive) |
| BDF-k | multistep coefficients + `history` | `Root` on the full residual; start-up via `Select` |
| Lie / Strang | `Compose[(scheme, dt_frac)]` | pre / post stage lists |
| Chorin | `RK(SSP2, Compose[Hyperbolic(Euler), Elliptic, Pointwise])` | composite stage per RK stage |
| Sub-cycling (static) | `Repeat(n, step(dt/n))` | — |
| `NativeStep` | backend owns the step (PETSc TS, Irksome IRK, dune femdg) | no inner hooks; capability-limited |

Stage times are always `t + c_k·dt` for boundary conditions, sources and aux. This fixes AMReX (stage 2 at tⁿ) and foam
(all stages at tⁿ⁺¹), which are first order in time for time-dependent sources today.

## 7. Plugins

**PreciceCoupling.** Fills `couple_init / ongoing / read / write / advance / finalize`, `window_terms` and `face_fixup`,
and carries `P`. The march gains an outer window loop (see the example in §10).

Core writes the coupling physics **once**, symbolically or as xp bodies:
- characteristic, fullstate and Froude ghosts;
- the DGH interface flux;
- the remaining mass-row variants;
- σ-lift and projection;
- frame rotation.

The backend body is only API + mesh registration + interface gather.

Template additions it requires:
- **A1** nested loops with a host/compiled boundary, and a traced end time;
- **A2** plugin state in the checkpoint;
- **A3** a `bdata` slot in the BC kernel;
- **A4** coupled hooks for the elliptic and implicit-diffusion stages;
- **A5** writes gated on "committed".

**Observer.** `init / update` carried in the state (already in production jax).

**Output.** The write cadence is drift-free (stamps recomputed, not accumulated), and initial and final snapshots are
written.

## 8. Backends and capabilities

| Backend | Lowering | Capabilities | Refused |
|---|---|---|---|
| numpy | host loop, xp bodies, scipy solves | implicit, Chorin, coupling (host), pyodide | GPU, SPMD, AD |
| jax-fwd | `lax.while_loop`, xp bodies | jvp, GPU, coupling (host-window), in-trace output | — |
| jax-rev | equinox checkpointed While | vjp, `Root`/`LinSolve` via implicit function theorem, observers, idempotent output | stateful callbacks in the compiled loop, traced loop counts, in-trace preCICE, `NativeStep` |
| jax-sharded | `shard_map`, halo `ppermute`, `reduce_min` = `all_gather` + `min` (`pmin` has no AD rule), `inner` = `psum` | SPMD, GPU, (AD unproven) | non-replicated predicates |
| foam | printed march in `zoomyFoam`, native bodies | MPI, preCICE, IMEX-ARK, Chorin | AD, GPU |
| amrex (via the amrex steward) | printed stage executor; level shell native | GPU (hyperbolic), MPI, AMR with one global dt, implicit source, Chorin | AD, AMR subcycling + projection / coupling (until a shell exists) |
| dmplex | printed march; PETSc as `Root`/`LinSolve` (option a) or `NativeStep` (option b) | MPI, implicit | AD (TSAdjoint later), GPU |
| firedrake | host loop, handle mode, UFL form builder, `ExplicitStage` body | implicit forms, AMR rebind, MPI, reverse AD via pyadjoint (with limits) | jax trace, GPU |
| dune (molGalerkin) | host loop, shared UFL builder | explicit, forward only | AD |
| fenicsx | — | — | excluded (dead code) |

Static rules checked by `build_solver`:
- **Replicated predicates:** any `Select`, `While`, `Root` or `Guard` guarding a communicating block must use a `reduce_*`
  output. This catches the foam MOOD MPI deadlock and the jax `shard_map` MOOD predicate.
- **Reverse mode:** static counts; no stateful callbacks inside the compiled loop.
- **Coloured Jacobians:** must use a structurally symmetric pattern. A J-only pattern gives wrong reverse gradients
  (tested).

## 9. Solver variants (initial catalogue = today's capabilities)

Presets keep today's names, so most thesis scripts keep working.

| Preset (name kept) | Template | Default stages | Typical backends |
|---|---|---|---|
| `HyperbolicSolver` | `march(rk_step(tableau(order), explicit_stage))` | Hyperbolic | all |
| `IMEXSourceSolver` | + implicit stages (Lie or ARK, see D2) | Hyperbolic, ImplicitDiffusion, ImplicitSource | numpy, jax, foam, amrex, dmplex, firedrake |
| `ChorinSplitVAMSolver` | RK over composite stage | Hyperbolic(Euler), Elliptic, Pointwise | numpy, jax, foam, amrex, dmplex |
| `DAESolver` | ARK on the full residual (`Root` + projection) | ImplicitFull | numpy, jax |
| any + `plugins=[PreciceCoupling(...)]` | window loop | — | numpy, jax-fwd, foam |
| any + `backend="jax-reverse"` | checkpointed loop | — | jax |
| `NativeStep` variants | backend-owned step | — | dmplex PETSc TS, Irksome, dune femdg |
| Sigma3D | opaque stage owning its mesh | — | numpy only |

## 10. User-side examples

**Hyperbolic, order 1 or 2.** The order comes from the NSM, as today:
```python
nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"))
solver = HyperbolicSolver(time_end=10.0, cfl=0.9, backend="jax")          # backend="numpy" | "jax-reverse" | ...
Q, Qaux = solver.solve(mesh, nsm)
```

**With preCICE:**
```python
solver = HyperbolicSolver(time_end=10.0, cfl=0.9, backend="jax",
                          plugins=[PreciceCoupling("precice-config.xml", participant="SME", interface="right")])
```

**Template internals** (core, written once). The generated step and march code is shown in the chat answer of
2026-10-07:
```python
def explicit_stage():                         # Q_cand = Q + dt·L(Q; t_stage)
    return Procedure("explicit_stage", inputs=(Q, Qaux, p, t_stage, dt), outputs=(Q_cand,), body=[
        Call("halo_bc", (Q, Qaux, p, t_stage, bdata), out=(bf,)),
        Call("reconstruct", (Q, Qaux, bf, o1), out=(Q_L, Q_R, grad)),
        Call("flux_pass", (Q_L, Q_R, Qaux, p, t_stage), out=(F, Dp, Dm)),
        Hook("face_fixup", (F, Dp, Dm, P), out=(F, Dp, Dm)),
        Call("cell_ncp", (Q, grad, Qaux, p), out=(cell_term,)),
        Call("gather_update", (Q, F, Dp, Dm, cell_term, Qaux, p, t_stage, dt), out=(Q_cand, troubled))])

def rk_step(tableau, stage):                  # build-time unrolled Shu-Osher
    body = [Assign(Q0, Q)]
    for a_k, c_k in tableau:
        body += [Assign(t_stage, time + c_k*dt), Call(stage, (Q, Qaux, p, t_stage, dt), out=(Q_cand, troubled)),
                 Assign(Q, a_k*Q0 + (1 - a_k)*Q_cand), Call("update_aux", (Q, Qaux, p, t_stage), out=(Qaux,))]
    body += [Call("reduce_any", (troubled,), out=(mood,)),
             Select(mood, then=[Assign(Q, Q0), Assign(o1, 1), *stage_blocks_again]),   # MOOD: re-enter stages at O1
             Guard(all_admissible(Q), "MOOD could not restore positivity"),
             Hook("post_step", (Q, Qaux, p, time, dt), out=(Q, Qaux))]
    return Procedure("step", ...)

def march(step, coupled):                     # loop in a loop when coupled
    inner = While(time < t_until, [Call("compute_dt", ..., out=(dt,)), Assign(dt, Min(dt, t_until - time)),
                                   Guard(finite(dt) & (dt > 0), "dt not admissible"),
                                   Call(step, ...), Assign(time, time + dt), Hook("observe", ...),
                                   Callback("write_fields", ..., when=due(time))])
    if not coupled:
        return Procedure("march", body=[Assign(t_until, t_end), inner])
    return Procedure("march", body=[Callback("couple_init"), While(Callback("couple_ongoing"), [
        Assign(ck, STATE), Callback("couple_read", (time,), out=(peer, t_until)),
        Call("window_terms", (Q, Qaux, p, peer), out=(P,)), inner,
        Callback("couple_write", (Q,)), Callback("couple_advance", (t_until,), out=(reload,)),
        Select(reload, then=[Assign(STATE, ck)])]), Callback("couple_finalize")])
```

**Chorin (today's `ChorinSplitVAMSolverJax`), user side:**
```python
split = VAM(level=1, dimension=2, closures=[Newtonian(), StressFree()], boundary_conditions=bcs).chorin_split()
solver = ChorinSplitVAMSolver(split, time_end=20.0, cfl=0.45, time_order=2,
                              reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"),
                              elliptic=EllipticOptions(tol=1e-10, restart="auto", precond=None),
                              backend="jax-reverse", checkpoints=64,
                              plugins=[Observer(running_max("h"))])
Q, Qaux, acc = solver.run(mesh)
```

**Chorin, what the preset assembles** (core):
```python
stages = [Hyperbolic(split.predictor, time="euler", source="lie"),              # explicit_stage on the predictor NSM
          Elliptic(split.pressure,                                             # LinSolve over elliptic_residual
                   solve=LinSolve(kind="krylov", tol=c_elliptic_tol, restart=c_elliptic_restart),
                   guard="rel_resid <= c_elliptic_accept"),
          Pointwise(split.corrector)]                                          # update_variables of the corrector
step = rk_step(SSPRK2, composite_stage(stages))   # each RK stage: aux refresh per stage → predictor → pressure → corrector
march(step, coupled=False)
```
- Per-stage parameter vectors are gathered by name from one traced vector, with `dt` as an explicit slot.
- Per-stage aux pools are rebuilt from Q at each stage.
- Today's ~1100-line jax Chorin file becomes ~90 lines of shared stage bodies plus generic machinery.

## 11. Decisions needed from you (recommended default first)

| # | Question | Options (recommended first) | Consequence |
|---|---|---|---|
| D1 | MOOD semantics | (a) cell-wise O1 repair, then whole-step O1; still troubled → **Guard (fatal)**. (b) as (a) but commit silently (jax today). (c) whole-step only (v6) | Feasibility: whole-step O1 at CFL 0.9 is NOT positivity-preserving in 1-D (Ritter h_min −1e-3), so (c) alone is unsafe |
| D2 | Default IMEX composition | (a) keep Lie after RK as the default (today's numpy/jax results), ARK as an option. (b) ARK default (2nd order with stiff sources) | (b) moves IMEX results |
| D3 | Stage time | `t + c_k·dt` everywhere | AMReX/foam results with time-dependent BCs or sources move |
| D4 | CFL dimension | from the mesh dimension (1-D uses 1) | legacy numpy 1-D goldens shift |
| D5 | BC kernel convention | all kernels return **ghost** values (Dirichlet → 2v − Q). Alternative: per-branch flag | fixes an 8 % non-converging boundary-gradient error; the Riemann state at Dirichlet faces changes |
| D6 | Aux on the BC side | recompute from the ghost state (numpy) | jax results move ~1e-3 at boundaries that change h |
| D7 | Stage-2 aux refresh | yes (jax) | numpy results move; numpy currently collapses to 1st order in time with VK |
| D8 | Chorin nesting | outer SSP-RK over the composite stage (jax / Escalante) | numpy Chorin results move |
| D9 | Failures (dt, elliptic, Newton) | **Guard (fatal)** with emitted tolerances | no more silent `break` / warnings |
| D10 | Snapshots | initial + final written by the template | — |
| D11 | FEM scope | firedrake + dune (molGalerkin) in; fenicsx deleted; dune femdg = optional `NativeStep` | — |
| D12 | Keep as variants vs retire | keep DAE (ported), Sigma3D (opaque, numpy-only); retire `SplittingSolver` (→ Chorin path), `precice_solver.py` (→ plugin), v6 `solver2` / `march_emitted` (superseded) | deletions |
| D13 | C++ order | foam first (owned by core?), dmplex next, AMReX via steward request; each gated: emitted march compiles and matches the native march (o1, o2, MOOD, IMEX, coupling) | — |
| D14 | Tests (your rule: discussed first) | conformance suite per backend (list in §13) | — |
| D15 | General PDE core with free-surface as a specialisation package | yes: roles + admissibility kernel + face_equilibrium hook + specialisation-owned default NSM ops (§18) | free-surface results unchanged; name-collision bugs removed |

## 12. Plan (feature branch `solver-template` in zoomy_core, zoomy_jax, then the C++ / FEM repos)

| Phase | Content | Gate |
|---|---|---|
| P0 | IR: Select, Root, LinSolve, Guard, Callback, Repeat, Hook, closures, `Procedure.results`, readable source emission, capability checker, C walker fixes (8), typed block registry | unit checks of lowering on numpy + jax + C (compile) |
| P1 | Shared xp bodies: geometry, BC (batched), reconstruction (all limiters), flux / gather, dt; numpy + jax backends; `HyperbolicSolver` preset | numpy and jax variants match today's jax where semantics agree (bitwise per backend), and the decisions D3–D7 are applied with before/after numbers |
| P2 | Implicit (Root / LinSolve, IMEX, DAE), Chorin composite, observers, jax-rev / jax-sharded variants | today's AD tests reproduce (fwd = rev = FD) |
| P3 | PreciceCoupling plugin (Python) + foam participant body; coupling physics in core | L1L1 twin bitwise; SME↔VOF case reproduces |
| P4 | C++: foam printed march, dmplex, AMReX (request to the steward); firedrake / dune | per-backend suite + images to you before switching (v6 gating rule) |
| P5 | Delete old solvers per backend after your approval | — |

Effort (critic's estimate for the full-IR route): ~4000 new LOC, ~3500 LOC body deduplication, ~9–13 weeks.
- ~111 thesis scripts import zoomy_jax and 172 use numpy internals. Presets keep the names; a numpy facade keeps
  `.step(dt)` working.
- Scripts using private internals (`_rt_*`, `_sim_*`) are converted case by case. The thesis repo is on another
  session's branch, so its conversions are coordinated separately.

## 13. Proposed new tests (for your approval)
- Template lowering: each statement kind on numpy / jax-fwd / jax-rev / C (compile + run) — small.
- Block conformance per backend:
  - temporal-order MMS (catches stage-time drift);
  - stiff-relaxation IMEX order;
  - MOOD outcome (h ≥ 0, exact mass);
  - BC routing with four different BCs;
  - BC convention on a linear field.
- AD: fwd = rev = FD for each jax variant (exists; to be ported).
- Coupling: in-process twin bitwise; rollback restores the write counter.

## 14. Risks
- **Readability of generated code.** Mitigation: emitted source with statement comments; `solver.show()`.
- **C++ shells.** AMReX recursion, PETSc TS and foam `Time` own structure. Mitigation: `NativeStep` + backend shells +
  capability refusals; per-backend gate before switching.
- **Result changes from unifying semantics (D1–D8).** Every change is reported with before/after numbers; nothing is
  baselined silently.
- **Ownership.** AMReX changes only via steward requests; the superrepo and thesis are on another session's branch, so
  work goes on the feature branch via worktrees.
- **Unmeasured items.**
  - Guard/cond cost on GPU;
  - AD under SPMD;
  - pyadjoint checkpointing with an adaptive loop;
  - nvcc behaviour of emitted orchestration;
  - pyodide end-to-end (node is present, pyodide is not installed).

## 15. Bugs found during the investigation
| Bug | State |
|---|---|
| jax Venkatakrishnan limiter never limited decreasing faces | fixed, zoomy_jax `77a87c6` (φ now equals numpy to 2e-16) |
| jax 2-D BC routing by position instead of name (24/32 faces wrong on a 4-BC channel) | fix in progress (background) |
| BC kernels mix face/ghost conventions; jax limiter bounds double-convert | D5 |
| foam MOOD can deadlock under MPI (rank-local predicate) | open (via template rule / request) |
| foam implicit coupling rollback does not restore the write counter (writes an unconverged frame) | open (template fixes by construction) |
| AMReX/foam stage time: 1st order in time for time-dependent sources | D3 |
| AMReX `UpdateState` before `FillPatch`: coarse–fine ghost aux one stage stale; "FluxRegister from final Fface" wrong at order 2 | open (request) |
| AMReX Chorin `LoopOnCpu` over device data (CUDA) | open (request) |
| firedrake DG1 first order in time; no t_end clamp; accumulating write cadence; dead `_setup` with NameError | open (P4) |
| dune: no DG volume terms for k ≥ 1; one BC for all boundaries at a frozen time; source / aux evaluated at t=0, x=0 | open (P4) |
| `precice_solver.py` broken at 5 levels | to be replaced by the plugin |
| v6 `solver2` eta path broken; `reduce_dt` NaN in reverse on dry faces | superseded |
| jax `_advance` min-combines the t_end and coupling window clamps (D7 of v6) | fixed by the template |

## 16. In-process multi-model coupling (no preCICE), added 2026-10-07

This is the in-process twin of the preCICE plugin: the same window / sub-cycling / rollback template, with exchange by
mesh-to-mesh transfer blocks instead of preCICE API calls. Evidence that it works: jax in-process SME↔SME twins are
already bit-exact (`thesis/cases/sme_coupling_jax` v1/v3), and the interlevel face contribution sits in `solver_jax.py:123`.

New building blocks:
- **Model namespaces.** The state holds one sub-state per model; blocks are qualified per model (`fluid.explicit_stage`,
  `heat.implicit_diffusion`).
- **`Transfer(src.field → dst.slot, interface, method)`.** Setup builds the mapping matrices on the host (nearest /
  linear / RBF / conservative supermesh); runtime applies them as a gather plus weights (xp) or CSR (C++).
- **`bdata` BC slot.** Shared with preCICE (A3).
- **Convergence and acceleration blocks** for implicit coupling: relaxation, Aitken, IQN-ILS.
- **Monolithic option.** `Root` over the interface unknowns (fixed point → `custom_root`), which makes implicit coupling
  differentiable by the implicit function theorem.

Result:
- one generated solver when all models share a backend;
- jax: one jit, so XLA fuses across models, and reverse AD works through the coupled system;
- C++: one executable, as OpenFOAM `chtMultiRegionFoam` does natively;
- mixed backends: preCICE.

## 17. Generated-code quality and CSE, added 2026-10-07

- **Kernel arithmetic.** SymPy CSE already runs per kernel (`generic_c.py:664,680`; `procedure_c.py:155,170`; jax
  lambdify `cse=True`, `jax_runtime.py:175`).
  - Printer fixes needed: emit `sqrt` / `rsqrt` for exponents ±1/2 (today `pow(x, 1/2)`; measured 2× slower on GPU);
    `x*x` chains for small integer powers; a CSE'd reciprocal for `1/pow`.
- **Optional symbolic fusion of pointwise chains** (reconstructed face states → numerical flux → fluctuations) into one
  expression with a joint CSE pass. This is not possible across gathers, limiters or reductions.
- **Memory passes.** The block decomposition costs extra sweeps. "Fused" vs "face-stored" becomes a backend lowering
  option; AMReX measured face-stored as 1.24× faster than fused on an L40S, so neither is always better.
- **Data layout.** Generated code uses the backend-native containers (Array4, `volScalarField`, `Vec`) via typed slots,
  so the layout is the same as hand-written code.
- **Orchestration cost.** Negligible.

## 18. General PDE core, free-surface as a specialisation (D15), added 2026-10-07

Inventory of every free-surface assumption in generic paths (G = generic, D = free-surface but data-driven,
H = hard-coded; counts are distinct code sites):

| Layer | G | D | H |
|---|---|---|---|
| core | 16 | 22 | 6 |
| numpy | 10 | 11 | 7 |
| jax | 12 | 8 | 6 (3 in legacy/dead/v6 code) |
| foam | 8 | 4 | 4 |
| amrex | 6 | 2 | 8 |
| dmplex | 4 | 2 | 6 |
| firedrake | 6 | 3 | 1 |
| dune | 2 | 2 | 2 |

Measured with non-free-surface models (advection, Burgers, Euler Sod):
- numpy crashes with `KeyError 'h'` (`solver_numpy.py:813`). With one lookup patched, all three models run at order 1 and 2.
- jax runs all three, matching numpy.
- **Silent wrong answers** from name-based inference:
  - a tracer named `h` gets the dry-cell CFL gate, so dt = `dt_max` (5.0) instead of 0.0045;
  - a tracer named `b` loses Rusanov dissipation and blows up to ±2900.
- Not free-surface related:
  - a state named `T` crashes the jax BC kernel (reserved-name collision);
  - numpy scalar diffusion needs a parameter literally named `nu`;
  - heat decay is 0.704 vs the exact 0.674 (not investigated);
  - periodic order-2 runs lose mass (~1e-4).

Real roadblocks and their mechanisms:
1. **Admissibility hard-wired as h ≥ 0** (template contract, numpy, jax, foam, amrex, dmplex). Fix: an NSM-emitted
   `admissible(Q, Qaux, p)` constraint list ([h] for SWE; [ρ, p(Q)] for Euler) plus a finiteness check. MOOD and
   positivity scaling loop over it.
2. **Roles inferred from names** (`h`, `b`, `g`, `hinv`, `nu`). Fix: model-declared roles `depth`, `bed`/stationary
   (`SystemModel.stationary_indices` already exists), `positive`, momentum groups, gravity. The analytic sign
   assumptions seed `positive` before the NSM strips them. Free-surface default NSM operations (KP `hinv`, dry gate)
   move into the specialisation.
3. **Well-balancing inside generic solvers** (numpy `bernoulli_wb` branch, jax audusse/bernoulli/steady-ODE branch,
   amrex driver Audusse, dmplex eta B=0/H=1). Fix: a `face_equilibrium` hook whose body is model-supplied: printed
   W-maps / Positive* kernels (already generic) or a registered free-surface procedure.
4. **C++ drivers hard-code the SWE layout:**
   - amrex: `static_assert` b=0, h=1; g = `p(0,0)`; `ErrorEst`; clamp. Changes go via the amrex steward.
   - dmplex: dry relaxation of the implicit Jacobian and `VacuumLimiter` on row 1 in the default stack.

   Fix: role-gated code (`if constexpr (Model::has_depth_role)`), named parameter accessors, plugin-supplied
   refinement indicators.
5. **Cosmetic:** slot names (`hydrostatic_pressure`, `[b,h,u,v,w,p]` lift), GUI cards, plot helpers, `wet_dry_eps`
   constants.

Scope consequences:
- §7's coupling physics (ghosts, mass rows, σ-lift) becomes a free-surface coupling specialisation of the generic
  plugin.
- §13's MOOD test is stated against the admissibility set.
- Chorin is VAM-only today (`splitter.py:103` requires `h`), so general projection methods (e.g. incompressible NS)
  need a generic split API.
