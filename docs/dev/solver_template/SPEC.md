# Solver template — implementation spec (Stage 1: JAX)

Binding spec for every agent working on `feature/solver-template`. Background: `DESIGN.md` (decision document),
`synthesis_notes.md` (all investigation + feasibility findings), `prototypes/` (working scratch prototypes:
`feas_ir/ir_ext.py` + `march_ext.py` = new statement kinds + traced march; `feas_np/shared_body.py` = one xp step body
for numpy+jax; `feas_c/feas_printer.py` = the 8 C-walker fixes; `critic/b_template.py` = 25-line template + backend).

## 0. Rules (binding)

- **Where:** worktree `~/git/Zoomy-solver-template`, branch `feature/solver-template` in the superrepo, every
  submodule and `thesis`. NEVER touch `~/git/Zoomy` (another agent works there on `dune-backend`). Before every commit:
  `git branch --show-current` must print `feature/solver-template`. Brief commit subjects, NO Co-Authored-By / trailers,
  no amend, no force, no push. Stage only your own files.
- **Python:** `W=~/git/Zoomy-solver-template; PYTHONNOUSERSITE=1 PYTHONPATH=$W/library/zoomy_core:$W/library/zoomy_jax
  micromamba run -n zoomy python` — verify `zoomy_core.__file__` / `zoomy_jax.__file__` resolve into `$W`. x64 on.
  No pytest-xdist.
- **No shortcuts.** Every solver (hyperbolic, IMEX, Chorin splitting, preCICE, AD) complete, general and
  dimension-agnostic (1-D and 2-D horizontal; unstructured LSQ meshes; any number of state rows).
- **Generalise and fix on the go.** The goal is NOT to reproduce old code; it is to still solve everything we solve
  today, more generally/correctly. Hard-coded indices, name-based roles, misplaced well-balancing, small defects:
  fix them in the new code. Report every intentional result change with before/after numbers.
- **Old solvers stay untouched and importable** until Stage 2 passes. New code lives in new modules. Presets reuse the
  familiar class names in the NEW namespace (`zoomy_jax.template.HyperbolicSolver`, …).
- **Tests:** gate tests ONLY in `tests/template_gates/` of the repo they test, files `test_gate_<substage>_*.py`,
  pytest marker `template_gate` (register it). Small and fast; long ones additionally `large`. They are consolidated
  with the user at the end — do not add tests anywhere else, do not modify existing tests.
- **Architecture rules:** models stay analytic (no eps in models); numerical choices live in the NSM / stage options;
  printers are syntax-only; operators are Functions. `zoomy_core` never imports jax (pyodide). jax specifics live in
  `zoomy_jax`. Generic code never names `h`, `b`, `g`, `hinv`, `nu` — use roles (§4).
- **Readable generated code:** Python lowering emits source files (not closure towers); stack traces must end in
  readable generated files that name the template statement they came from.

## 1. Decisions (user, 2026-10-07)

| # | Decision |
|---|---|
| D1 | MOOD: troubled = not admissible (§4) or non-finite. Order-2 candidate → recompute troubled cells at order 1 → if still troubled, whole step at order 1 → if still troubled: **Guard error** (implementation bug: order 1 is admissible under the CFL). |
| D2 | IMEX default = implicit stages after the RK step (Lie, cheaper); true IMEX-ARK (ARS232/ARS343, per-stage implicit) available as an option. |
| D3 | Stage k evaluates BCs, sources, aux at `t + c_k·dt` (tableau column c). Overridable by a custom stage. |
| D4 | CFL: local per-face law `dt = min_f CFL·2r_f / (d·(2k+1)·|λ_f|)` with d = mesh (horizontal) dimension, k = reconstruction degree; capped by dt_max; one implementation for all. |
| D5 | BC kernels return GHOST values (Dirichlet → 2v − Q_inner, Wall reflection, Extrapolation = Q_inner, …). Gradient, limiter bounds and Riemann right state all consume the ghost state consistently. A backend needing face values overrides its BC body. |
| D6 | Aux on the boundary/ghost side is recomputed from the ghost state with the model's aux update rule. |
| D7 | Aux refreshed between RK stages (update_aux at t + c_k·dt). |
| D8 | Chorin/projection: outer RK over the composite stage [hyperbolic(Euler) → elliptic → pointwise]. |
| D9 | Failures (dt not finite/≤0, elliptic/Newton/Krylov not converged) → Guard error; tolerances are emitted constants. |

## 2. Layers and modules

```
authoring classes   zoomy_core/solver/template/   Scheme, Stage, Plugin, Solver, presets (backend-agnostic)
IR                  zoomy_core/solver/ir.py       statements + Procedure + typed block registry (extend existing)
walkers             zoomy_core/transformation/procedure_python.py → emits Python SOURCE for a backend
                    zoomy_core/transformation/procedure_c.py      → C++ (keep printable; stub-compile check)
shared bodies       zoomy_core/solver/bodies/     one xp body per block (numpy & jax), backend primitives injected
jax backend         zoomy_jax/template/           JaxBackend(mode="forward"|"reverse", checkpoints), primitives,
                                                  jax-only bodies (if any), presets re-exported for users
```
(Exact file names may be refined; keep this split: authoring/IR/walkers/bodies in core, jax in zoomy_jax.)

### 2.1 Authoring classes
- `Scheme`: `ShuOsher(stages=[(a_k, c_k), …])` with EULER, SSPRK2, SSPRK3 (c = 0 | 0,1 | 0,1,½); default from the
  reconstruction order but independent of it; `Imex(kind="lie"|"ark", tableau=ARS232|ARS343)` (Butcher A_E, A_I, b, c);
  `Compose(...)` for splitting. Unrolled at build time into statements.
- `Stage` (build(ctx) → statements): `Hyperbolic` (explicit stage), `ImplicitSource`, `ImplicitDiffusion`, `Elliptic`,
  `Pointwise`, `CompositeStage([...])`, user subclasses; per-stage NSM/runtime, parameter gather by name, explicit dt.
- `Plugin`: fills hooks + adds carry: `Output` (drift-free cadence, initial + final snapshot), `Observer`,
  `InProcessCoupling`, `PreciceCoupling`.
- `Solver(scheme, stages, plugins, backend, time_end, cfl, …)`: `build()` → Procedure tree → walker → callable;
  `solve(mesh, nsm)`, `run_simulation(Q0, Qaux0=None, parameters=None, time0=0.0)`, `run_observed(...)`,
  `show()` prints the assembled template. Variants = catalogue entries; no capability-checking code.

### 2.2 IR statements (extend `zoomy_core/solver/ir.py`; prototypes in `prototypes/feas_ir/ir_ext.py`)
`Seq, Assign, Call, Hook(name, default), IfStatic, While(cond, body, carry), Repeat(n_static, body),
Select(pred, then, else), Root(residual_proc, x0, opts), LinSolve(matvec_proc, rhs, opts), Guard(ok, msg, values),
Callback(fn, in, out, kind="idempotent"|"stateful"|"returns")`, `ForEachFace` (static count).
- `Procedure(name, inputs, outputs, body)`; procedures may be passed as arguments (closures) to Root/LinSolve.
- Block registry = the single typed source of signatures (name, inputs, outputs, kinds, comm class
  none|neighbor|global); arity/kind checked when building. Remove `SOLVER_ARG_SLOTS`.
- `_resolve` recursion into new nodes; int-typed slots stay ints (counters).
- Host/compiled boundary: the outermost loop containing stateful Callbacks runs on the host; the innermost loop free
  of them is one compiled unit.

### 2.3 Python walker
Emits `generated/<solver>_<proc>_<backend>.py` (cache dir), comments per statement, imports the backend primitive
object `B` and the bodies; loads via importlib. Lowering via the backend: `B.while_loop`, `B.repeat`, `B.cond`,
`B.root`, `B.linsolve`, `B.guard`, `B.callback`, `B.jit`, `B.xp`, `B.scatter_add/min/max`, `B.reduce_min/any/sum`,
`B.inner`.

### 2.4 jax backend
forward: `lax.while_loop`; reverse: `equinox.internal.while_loop(kind="checkpointed", checkpoints=K)`;
Select → `lax.cond`; Root → `implicit_ad.newton_root` (custom_root); LinSolve → custom_linear_solve + gmres with
`inner`; Guard → `eqx.error_if` + ordered debug print of values; Callback (idempotent) → ordered `io_callback` in a
`custom_jvp` zero-tangent wrapper; stateful callbacks only at host level (the reverse backend has no lowering for
them). `reduce_min` AD-safe (all_gather+min under SPMD; pmin has no AD rule). Safe double-where divides in dt and any
x/λ expressions (reverse NaN on dry/zero faces otherwise). One compile per configuration; `time_end`, `time0`,
write interval TRACED (not in the cache key).

## 3. Blocks (shared xp bodies; signature order `(dt, time, Q, Qaux, p, …)` style as in the template)

| Block | Notes |
|---|---|
| `prepare_mesh(mesh) → (topology, geom)` | topology numpy (integer, static); geometry = vectorised xp from node coordinates (volumes, normals, face areas/centres, cell centres, ghost centres, inradius, face inradius, LSQ matrices with REQ-174b ghost decoupling). Prototype: `prototypes/` (2-D geometry 25 lines, LSQ 30 lines, bitwise vs FVMMesh/LSQMesh). Differentiable w.r.t. node coordinates (AD gate in 1d). |
| `halo_bc` / BC values | batched BC kernel over all boundary faces (vectorise the BC Piecewise; `prototypes/feas_np/f4_batched_bc.py`); routing by tag NAME (`boundary_face_branches`, zoomy_core 4c7168b); D5 ghost convention; D6 ghost aux. No Python loops over faces. |
| `reconstruct` | constant, LSQ-MUSCL with venkatakrishnan / barth_jespersen / minmod / van_albada / none; reconstruction variables via the model's W(Q)/Q(W) maps (eta etc.); positivity scaling over the admissibility list; `o1` mask for MOOD. |
| `flux_pass` | batched numerical flux + fluctuations (`nsm.riemann`), path-conservative NCP, per-face. |
| `cell_ncp` | emitted `numerical_cell_edge_source` (Gauss–Legendre) form. |
| `gather_update` | scatter (B.scatter_add) of fluxes/fluctuations, cell term, explicit source/diffusion → Q_cand, troubled mask. |
| `compute_dt` | D4 law; safe divide; dt_max cap; `eigenvalues_cfl` slot. |
| `update_aux`, `update_variables` | full-length aux contract, LSQ derivative aux, time/position threaded. |
| `write_fields` | idempotent by snapshot index (HDF5 store as today); `Observer` as in `zoomy_jax/fvm/observers.py`. |
| `face_equilibrium` hook | well-balancing (Audusse HR, Bernoulli, steady-ODE) as model-supplied kernels/registered procedures, NOT in generic bodies. |

## 4. Roles and admissibility (general PDE core)
- `SystemModel`/NSM carry declared roles: `depth`, `bed`, `stationary` (rows with zero flux; use
  `SystemModel.stationary_indices`), `positive`, `momentum` groups, `gravity` parameter. Free-surface model builders
  declare them; generic code reads roles only.
- NSM emits `admissible(Q, Qaux, p)` = list of expressions that must stay ≥ 0 (SWE/SME: [h]; Euler: [ρ, p(Q)]) — used by
  MOOD (D1) and positivity scaling. No literal `h<0` anywhere in generic code.
- Free-surface default NSM operations (KP hinv desingularisation, dry eigenvalue gate) are applied by the free-surface
  specialisation based on the `depth` role, not by name.
- Name-collision regressions to avoid (found 2026-10-07): a tracer named `h` must not get the dry gate; a tracer named
  `b` must keep Rusanov dissipation; a state named `T` must work in the jax BC kernel.

## 5. Stage 1 sub-stages and gates (each gate = `tests/template_gates/test_gate_<substage>_*.py`, marker `template_gate`)

- **1a core + hyperbolic.** IR + walkers + authoring classes + roles/admissibility + all blocks above + MOOD (D1) +
  guards (D9) + output/observer + jax backend (forward AND reverse) + `HyperbolicSolver` preset.
  Gates: lowering of every statement kind on jax-fwd/jax-rev; C stub-compile check of the emitted template still
  passes (`prototypes/feas_c`); SWASHES Stoker (o1,o2) and Ritter dry (o2, MOOD) L1 vs analytic within today's
  values or better; lake at rest over a bump to roundoff; 2-D pulse with four different BCs on create_2d and a gmsh
  mesh; jvp == central FD and grad == jvp (1e-8) on 1-D and 2-D; non-free-surface models (linear advection order
  ≥ 1.9, Burgers, Euler Sod mass exact); tracer-name regressions (§4); timing ≤ today's jax solver per step; the
  `thesis/cases/verification/swe_bump_ad` configuration vs experiment (report RMS).
- **1b IMEX.** Root/LinSolve lowering; `Imex(kind="lie")` default + `kind="ark"` (ARS232/343); implicit source
  (local + global Newton–Krylov, aux policy), implicit diffusion (CN, state-dependent Newton). Gates: stiff relaxation
  order (ARK 2nd order, Lie 1st), today's IMEX results, AD fwd=rev=FD through implicit stages.
- **1c Chorin/projection.** Generic projection stage via `split_for_pressure_structural`; VAM as specialisation;
  null-space option for pure-Neumann pressure (incompressible NS). Gates: VAM bump vs experiment (`thesis/cases/
  verification/vam_bump`) unchanged or better; AD fwd=rev; one incompressible-NS-style test (e.g. 2-D Taylor–Green
  or lid-free periodic decay) if feasible within the model API.
- **1d AD completion.** Observers/QoIs, checkpoint control, traced time_end, AD through Root/LinSolve, **geometry as
  input** (node coordinates → geometry inside the differentiated function). Gates: all existing `test_ad*` scenarios
  reproduced in the new framework; d/d(node coords) fwd = rev = FD on a small 2-D channel.
- **1e coupling.** `InProcessCoupling` (two models, different meshes, Transfer blocks, explicit Jacobi/Gauss–Seidel and
  implicit fixed-point with Aitken; window loop + rollback) then `PreciceCoupling` (host window loop, jax↔jax two
  processes, then jax ↔ the existing foam solver as partner). Gates: in-process SME↔SME twin bitwise vs monolithic;
  L1L1 twin; rollback restores the write counter.

## 6. Performance knowledge to apply
Batched BC kernels (160×), batched LSQ gradient (50×), one compile per configuration, no Python loops over faces in
traced code, analytic eigenvalues where available, `march_log_interval` default off on accelerators, SymPy CSE per
kernel; printer fixes: `sqrt`/`rsqrt` for exponents ±½, `x*x` for small integer powers.
