"""`EmittedMarchSolver` — the jax adapter for the march EMITTED by core.

This file is the architecture test's jax half.  It contains **no scheme
logic**: every `ExternalProcedure` body it
supplies is a thin positional shim onto the EXISTING
`zoomy_jax.fvm.solver2.blocks` implementation.  The march STRUCTURE — the
loop, the step head, the frozen dt, the stage sequence, the build-time
branches, the write cadence, the honesty guards and every constant — comes
from `zoomy_core.solver.march.emit_march` and is lowered by
`ProcedureBuilder`.

If this solver reproduces `MarchSolver` bitwise, the emitted march IS the
hand-written one and the other backends become ports.

Nothing in `solver2` is modified; `EmittedMarchSolver` subclasses
`MarchSolver` purely to inherit the setup path (`setup_march` builds
`MeshRT` / `Ops`) and the reference blocks.

Two lowering choices, both forced and both worth stating
-------------------------------------------------------
1. **The step is built on the jax backend, the march on numpy.**  The
   `While` lowers to `lax.while_loop` on the jax backend, and a traced
   loop CANNOT host the FATAL dt guard — the abort would be staged out.  The
   reference march is a plain python `while` for exactly this reason, so the
   emitted march is built with `ProcedureBuilder("numpy")`: at march level
   the only arithmetic is the scalar `time < t_end` predicate, all array
   work happens inside the jax-built step.  This mirrors the reference's two
   host dispatches per step (head + step).
2. **The MOOD redo reuses `MarchSolver.stage_loop`.**  The sanctioned redo
   re-enters the SAME stage sequence at runtime with `o1=1`.  The IR's only
   branch is build-time and `o1` is not a declared argument slot, so the
   emitted step cannot re-enter itself; the redo body therefore calls the
   reference stage loop, which is the same code the emitted sequence shims
   onto block by block.
"""
from __future__ import annotations

from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from zoomy_core.solver.external import assert_procedure_bodies
from zoomy_core.solver.march import emit_march
from zoomy_core.transformation.procedure_python import ProcedureBuilder

from zoomy_jax.fvm.solver2 import blocks as B
from zoomy_jax.fvm.solver2.context import build_operators, prepare_mesh
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.solver2.state import MarchState


@partial(jax.jit, static_argnums=(4, 5, 6))
def _reduce_dt_jit(lam_lo, lam_hi, inradius_f, dt_max, cfl, dimension, degree,
                   clamp):
    """`blocks.reduce_dt` under `jit` — see the note at the call site.

    `cfl` / `dimension` / `degree` are static because they are EMITTED
    constants: they are baked into the compiled program exactly as the
    reference bakes the closed-over `self._cfl`.
    """
    return B.reduce_dt(lam_lo, lam_hi, inradius_f, dt_max, CFL=cfl,
                       dimension=dimension, degree=degree,
                       t_remaining=clamp, dt_window=None)


class _NotCoupled:
    """Bodies for the coupling surfaces on an UNCOUPLED backend.

    They exist so the `REQUIRED_PROCEDURES` contract is met without
    pretending coupling works: the face fixup is the STRICT no-op the design
    demands, everything else RAISES if it is ever reached (it never is with
    `coupled=False`, which is a build-time flag).
    """

    @staticmethod
    def face_fixup(Fface, Dp, Dm, n_faces):
        return Fface, Dp, Dm

    @staticmethod
    def _unavailable(name):
        def body(*_a, **_k):
            raise NotImplementedError(
                f"{name}: zoomy_jax has no preCICE participant; the march was "
                "built with coupled=False, so this body is unreachable by "
                "construction — it exists only to satisfy the "
                "REQUIRED_PROCEDURES contract.")
        return body


class EmittedMarchSolver(MarchSolver):
    """`MarchSolver` whose march comes from core instead of `march.py`.

    Every `param` knob, the whole setup path and all block bodies are
    inherited / reused; only the assembly changes.
    """

    #: Set by :meth:`_build_step`; the emitted program is kept so a run can
    #: print exactly which constants and build flags produced it.
    program = None

    # ── the external body table ────────────────────────────────────────────

    def _external_bodies(self, MeshRT, Ops_, *, on_write=None,
                         max_steps=1_000_000):
        """Positional shims onto `solver2.blocks` — no scheme logic here."""
        nc = MeshRT.n_cells
        cell_centers = MeshRT.cell_centers[:, :nc]
        stage_loop = self.stage_loop        # the reference stage loop (MOOD)

        def _source(Q, Qaux, p, t, dt):
            if Ops_.source is None:
                return None
            return Ops_.source(Q[:, :nc], Qaux[:, :nc], p,
                               time=t, dt=dt, position=cell_centers)

        def cell_ncp(Q, Qaux, p, lim_grad):
            """Amendment 10, the cell-interior NCP integral.

            The only body without a one-to-one counterpart in `blocks.py`:
            in the reference it is inline in `MarchSolver._build_step`.  It
            is reproduced verbatim (no `|cell|` division — the volume factor
            cancels against the per-unit-volume residual).
            """
            if lim_grad is None or Ops_.nonconservative_matrix is None:
                return None
            Ball = Ops_.nonconservative_matrix(Q, Qaux, p)
            ncp = jnp.einsum("ijdc,jdc->ic", Ball, lim_grad)
            return jnp.zeros_like(Q).at[:, :ncp.shape[1]].subtract(ncp)

        def assert_progress(time, iteration, dt, t_end):
            if int(iteration) < int(max_steps):
                return None
            raise FloatingPointError(
                f"march exceeded max_steps = {max_steps} at t = {float(time)} "
                f"/ {float(t_end)} with dt = {float(dt):.6e} — the timestep "
                f"collapsed towards zero without ever becoming non-positive, "
                f"so the dt guard could not see it. This is a REPORTED "
                f"FINDING about the scheme/configuration; the march does not "
                f"silently spin and the CFL is not reduced.")

        def write_fields(Q, Qaux, time, i_snapshot, do_write):
            if on_write is not None and bool(do_write):
                on_write(MarchState(time, 0, int(i_snapshot), Q, Qaux))

        bodies = {
            # ── setup / translation ────────────────────────────────────────
            "solver_prepare_mesh": prepare_mesh,
            "solver_build_operators": build_operators,
            "solver_initialize_state": self.setup_march,
            "solver_adapt_mesh": _NotCoupled._unavailable("solver_adapt_mesh"),
            "solver_reduce_min": lambda arr, n: jnp.min(arr),

            # ── step head ──────────────────────────────────────────────────
            # Both head bodies are JITTED.  This is not an optimisation: the
            # reference compiles dt_pass + reduce_dt as ONE jitted `head`, and
            # an eagerly-evaluated eigenvalue sweep rounds differently from
            # the XLA-fused one (measured: the lam_lo/lam_hi arrays differ
            # from step 1 on, carrying ~1 ULP into dt and 8.7e-19 into Q after
            # 19 steps).  Jitting each body restores bit-for-bit agreement.
            # c_eps_h is accepted and ignored: core's regularize_pow already
            # baked max(eps_h, h) into the derived eigenvalues kernel, so
            # re-applying it here would floor the spectrum twice.
            "solver_dt_pass": jax.jit(
                lambda Q, Qaux, p, c_eps_h: B.dt_pass(Q, Qaux, p, MeshRT,
                                                      Ops_)),
            "solver_reduce_dt": lambda lo, hi, r, nf, cfl, dim, degf, dtmax, clamp: (
                _reduce_dt_jit(lo, hi, r, dtmax, float(cfl), float(dim),
                               (int(degf) - 1) // 2, clamp)),
            "solver_apply_dt_floor": lambda dt, floor: jnp.maximum(dt, floor),
            "solver_assert_dt_admissible": lambda dt, lo, hi, r, t, it: (
                B.assert_dt_admissible(dt, lo, hi, r, time=t, iteration=it)),
            "solver_assert_march_progress": assert_progress,

            # ── stage bodies ───────────────────────────────────────────────
            "solver_stage_base": lambda Q: Q,
            "solver_clear_troubled": lambda Q: jnp.zeros((Q.shape[1],),
                                                         dtype=bool),
            "solver_merge_troubled": lambda a, b: a | b,
            "solver_halo_bc": lambda Q, Qaux, p, t: B.halo_bc(
                Q, Qaux, p, MeshRT, t, Ops_),
            "solver_reconstruct": lambda Q, Qaux, bf, o1: B.reconstruct(
                Q, Qaux, MeshRT, Ops_, bf, o1=bool(o1)),
            "solver_cell_ncp": cell_ncp,
            "solver_no_cell_term": lambda Q: None,
            "solver_flux_pass": lambda Q, Qaux, p, t, Q_L, Q_R: B.flux_pass(
                Q, Qaux, p, MeshRT, Ops_, t, (Q_L, Q_R)),
            "solver_gather_update": (
                lambda Qk, F, Dp, Dm, dt, beta, cell_term, Qaux, p, t:
                B.gather_update(Qk, F, Dp, Dm, MeshRT, dt, beta,
                                source=_source(Qk, Qaux, p, t, dt),
                                cell_term=cell_term,
                                h_index=Ops_.h_index)),
            # OUTCOME semantics: the strategy is the backend's.  h_bound and
            # require_finite are the emitted detector constants; the reference
            # predicate already IS (h < 0) | ~isfinite, so they are asserted
            # rather than re-applied — a backend that widened the bound would
            # trip this.
            "solver_mood_resolve": (
                lambda troubled, Q0, Q_cand, Qaux, p, t, dt, h_bound, finite:
                B.mood_resolve(troubled, Q0, Q_cand, Qaux, p, t, dt, MeshRT,
                               Ops_, stage_loop)),
            "solver_implicit_source": _NotCoupled._unavailable(
                "solver_implicit_source"),
            "solver_implicit_diffusion": _NotCoupled._unavailable(
                "solver_implicit_diffusion"),

            # ── post-step ──────────────────────────────────────────────────
            "solver_update_variables": lambda Q, Qaux, p, t, dt: (
                B.update_variables(Q, Qaux, p, t, dt, MeshRT, Ops_)),
            "solver_update_aux": lambda Q, Qaux, p, t, dt: (
                B.update_aux(Q, Qaux, p, t, dt, MeshRT, Ops_)),

            # ── io / coupling ──────────────────────────────────────────────
            "solver_write_fields": write_fields,
            "solver_coupling_face_fixup": _NotCoupled.face_fixup,
            "solver_coupling_checkpoint": lambda *a: None,
        }
        for nm in ("solver_coupling_init", "solver_coupling_read",
                   "solver_coupling_write", "solver_coupling_advance",
                   "solver_coupling_finalize"):
            bodies[nm] = _NotCoupled._unavailable(nm)

        assert_procedure_bodies(bodies, "zoomy_jax (emitted march)")
        return bodies

    # ── build ──────────────────────────────────────────────────────────────

    def _build_step(self):
        """Build the reference blocks (for the MOOD redo) and then lower the
        EMITTED march on top of them."""
        super()._build_step()          # gives self.stage_loop / the reference
        MeshRT, Ops_ = self.MeshRT, self.Ops

        self.program = emit_march(
            self.nsm,
            cfl=self._cfl,
            dimension=(MeshRT.dimension if self._dt_dimension is None
                       else self._dt_dimension),
            dt_max=self._dt_max,
            mood=bool(self.mood_redo),
            coupled=False,
            write_output=True,
            interior_ncp=bool(Ops_.use_interior_ncp),
            order=int(Ops_.order),
        )
        if tuple(self.program.tableau) != tuple(Ops_.tableau):
            raise ValueError(
                f"the EMITTED tableau {self.program.tableau} disagrees with "
                f"the reference {Ops_.tableau} — the stage arithmetic is core "
                "DATA and the two must be the same table")
        self._rebuild_callables()

    def _rebuild_callables(self, *, on_write=None, max_steps=1_000_000):
        bodies = self._external_bodies(self.MeshRT, self.Ops,
                                       on_write=on_write, max_steps=max_steps)

        # ``ProcedureBuilder`` COPIES its externals table, so each nested
        # procedure is built with a table that already contains the ones it
        # calls: should_write -> step -> march.
        sw = ProcedureBuilder("numpy", externals=bodies).build(
            self.program.should_write)

        def should_write(time, dt, i_snapshot, write_interval):
            env = sw(time=time, dt=dt, i_snapshot=i_snapshot,
                     write_interval=write_interval)
            return env["do_write"], int(env["i_snapshot"])

        bodies["solver_should_write"] = should_write

        step_fn = ProcedureBuilder("jax", externals=bodies).build(
            self.program.step)
        step_jit = jax.jit(lambda Q, Qaux, p, time, iteration, dt: (
            lambda e: (e["Q"], e["Qaux"], e["time"], e["iteration"],
                       e["troubled"]))(
            step_fn(Q=Q, Qaux=Qaux, p=p, time=time, iteration=iteration,
                    dt=dt)))
        bodies["solver_hyperbolic_step"] = step_jit

        self.emitted_step = step_jit
        self.emitted_march = ProcedureBuilder("numpy", externals=bodies).build(
            self.program.march)
        return self.emitted_march

    # ── the march ──────────────────────────────────────────────────────────

    def march(self, S, *, time_end=None, n_steps=None, dt_window=None,
              write_interval=None, on_write=None, parameters=None,
              max_steps=1_000_000):
        """Run the EMITTED march.  Same signature and same return shape as
        `MarchSolver.march`, so the twin gate can call either.

        `n_steps` is not part of the emitted loop predicate (the design's
        `proceed` is `time < t_end`); a step-count run is therefore driven
        by the equivalent `t_end` and is REJECTED rather than silently
        approximated.
        """
        if n_steps is not None:
            raise NotImplementedError(
                "the emitted march's loop predicate is the design's "
                "`proceed` (time < t_end); an n_steps run has no emitted "
                "equivalent and is not approximated here.")
        p = self._rt_parameters if parameters is None else parameters
        time_end = self.time_end if time_end is None else time_end
        march_fn = self._rebuild_callables(
            on_write=(on_write if write_interval is not None else None),
            max_steps=max_steps)

        t0 = perf_counter()
        env = march_fn(
            Q=S.Q, Qaux=S.Qaux, p=p, time=float(S.time),
            iteration=int(S.iteration), i_snapshot=int(S.i_snapshot),
            t_end=float(time_end),
            dt_window=dt_window,
            # No cadence => a stamp no march can reach.  NOT ``inf``: the
            # first stamp is ``0 * write_interval`` and ``0 * inf`` is NaN,
            # which would make the gate compare against NaN forever.
            write_interval=(1e300 if write_interval is None
                            else write_interval),
            inradius_f=self.MeshRT.inradius_f,
            n_faces=self.MeshRT.n_faces,
        )
        S = MarchState(env["time"], int(env["iteration"]),
                       int(env["i_snapshot"]), env["Q"], env["Qaux"])
        jax.block_until_ready(S.Q)
        stats = {"steps": int(env["iteration"]), "wall": perf_counter() - t0}
        return S, stats

    # ── reporting ──────────────────────────────────────────────────────────

    def describe_emitted(self) -> str:
        """The emitted program's flags + constants, for the pre-march print."""
        if self.program is None:
            return "emitted march not built yet (call setup_march first)"
        return self.program.report()
