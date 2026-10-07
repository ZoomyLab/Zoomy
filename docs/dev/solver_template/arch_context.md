# Architecture under evaluation (three layers) — shared context for the stress tests

Goal of the user: a NEW modular overhaul (not "did we follow the July v6 plan"). Commit to this structure only if
every current and future feature (explicit FV, IMEX/implicit, DAE, Chorin/projection splitting, general operator
splitting, preCICE coupling, AD forward+reverse, AMR, MPI/SPMD/GPU, FEM/UFL backends, geometry AD) can be expressed
with LOW overhead compared with today, and the structure does not limit future features.

## Layers
1. TEMPLATE — the time march written ONCE (as a symbolic Procedure IR in zoomy_core, lowered by walkers; the C++
   walker prints C++, the Python walker builds numpy or jax callables). It contains only structure + named hook calls:
   march: While(proceed) { dt = compute_dt; dt = couple_window(dt); couple_read; step(stage list); mood/select;
   update_variables; update_aux; couple_write/advance; observe; write_fields }. No feature code inside.
   Proposed new statement kinds (in addition to existing Assign, Call, IfStatic(build-time), ForEachFace, While):
   Select (runtime branch: lax.cond / if), Root (Newton root; jax custom_root), Guard (fatal check: eqx.error_if /
   raise), Callback (ordered side effect: io_callback+custom_jvp / direct call / C++ call), plus default (no-op)
   bodies for hooks.
2. STAGES + PLUGINS — composition chosen at build time: stage list per RK stage (Hyperbolic, ImplicitSource,
   ImplicitDiffusion, PressureProjection/elliptic, Corrector/pointwise, ...); plugins fill hook bodies (PreciceCoupling,
   Observer, output). Opaque blocks are ExternalProcedures: name+signature declared in core, body per backend
   (Python bodies written ONCE for numpy and jax via an array namespace xp + ~6 primitives: scatter_add/min/max,
   lower_cond, lower_while, lower_root, linear_solve, callback).
3. BACKEND — how it executes: numpy (pyodide-safe, never imports jax), jax-forward (lax.while_loop), jax-reverse
   (equinox checkpointed while loop; reverse-only), C++ backends (foam, amrex, dmplex: native bodies for mesh loops,
   halos, solvers, IO), FEM/UFL (firedrake/dune/fenicsx: bodies are form assembly + PETSc solves).
   Each backend declares a CAPABILITY set; build_solver refuses unsupported combinations at build time
   (e.g. jax-reverse + in-trace preCICE predicate; amrex + reverse AD).

User example of the intended API:
  solver = build_solver(nsm, backend=JaxReverseBackend(checkpoints=64),
                        stages=[Hyperbolic(), ImplicitSource()], plugins=[Observer(...)])

## Facts established today (investigations + in-memory feasibility tests)
- Production jax HyperbolicSolver (zoomy_jax/fvm/solver_jax.py) already has the three-layer shape as CLASS METHODS:
  template = _compile_time_loop + _advance; hooks with default bodies = _couple_*; IMEX (solver_imex_jax.py) and
  Chorin (solver_chorin_vam_jax.py) run on the same loop overriding only setup_simulation/step; observers
  (fvm/observers.py) carried in the loop state; ad_mode forward/reverse; implicit stages via lax.custom_root
  (fvm/implicit_ad.py); capability refusal _require_forward_ad. fwd = rev = FD tests exist (tests/runtime/test_ad*.py).
- The July v6 IR (zoomy_core/solver/{ir,march,external}.py, walkers transformation/procedure_{c,python}.py) is used
  by no backend; IR tests were deleted 2026-07-21; jax v6 track (fvm/solver2, march_emitted.py) is far behind
  production. C walker predicted not to compile the real march (undeclared temporaries, const outputs, array assigns,
  arity mismatches).
- In-memory tests: emitted v6 step inside equinox checkpointed While: grad = jvp = FD, bitwise to host march;
  custom_root inside a body under that loop works; eqx.error_if guard works under fwd/rev; io_callback wrapped in
  custom_jvp works forward and is REPLAYED under reverse (fine for idempotent output); a STATEFUL io_callback loop
  predicate (preCICE is_coupling_ongoing) under reverse runs the WRONG number of steps; ForEachFace with traced
  count fails reverse (counts must be static); single xp body for numpy+jax: bitwise per backend, few ulp across.
- numpy: 30 behavioural differences vs jax (MOOD semantics x3, implicit stage order, stage-2 aux refresh, CFL
  dimension, Chorin nesting, BC ghost convention, tag routing, ...); batched numpy LSQ gradient 50x faster than the
  current per-cell loop; numpy never imports jax today.
- foam (zoomy_foam): hand-written march ~= template; PreciceManager.H 1142 LOC (implicit coupling with clock rewind,
  ghost modes, frozen mass row); IMEX-ARK (ARS232/343) per stage; Chorin app; MPI; native fvMesh; 30 s builds.
- amrex: Evolve/Advance map 1:1 to template; stored face arrays; Berger-Oliger AMR without FluxRegister, substeps
  not quantized; GPU for hyperbolic; Chorin separate executables; hard-coded SWE row layout remnants; owned by the
  amrex steward (changes via request).
- dmplex: PETSc TS owns stages (TSSSP/ARKIMEX/BDF2); flat face loop = ForEachFace; MPI partial; no tests; owned by core.
- firedrake: UFL forms (hand-written face form), Lie split explicit + implicit Newton; pyadjoint possible but unused;
  DG1 is first order in time; AMR via animate. fenicsx: dead. dune (branch dune-backend): same UFL face form,
  hand SSP-RK2, best structural fit for template.
- numpy DAESolver (zoomy_core/fvm/solver_dae_numpy.py, ~630 LOC): IMEX-ARK with dense FD Jacobian + graph colouring.
