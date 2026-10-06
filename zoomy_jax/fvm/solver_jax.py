"""JAX FVM solvers: JIT-compiled time stepping.

Solver hierarchy (mirrors NumPy):
    HyperbolicSolver  (explicit flux + source)
      setup_simulation  — mesh/model → JAX operators (closures)
      step              — single explicit timestep
      run_simulation    — jax.lax.while_loop over step
      solve             — init + setup + run

Inherits param definitions from NumPy ``HyperbolicSolverNumpy`` but
overrides all computational methods with JAX implementations.
"""

import os
from time import time as gettime

import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

import param

from zoomy_core.misc.logger_config import logger

import zoomy_core.misc.io as io
import zoomy_core.misc.misc as _misc
import zoomy_jax.misc.io as jax_io
import h5py
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_core.fvm.solver_numpy import Solver as SolverNumpy
from zoomy_jax.transformation.jax_runtime import JaxRuntime
from zoomy_jax.mesh.mesh import convert_mesh_to_jax, lsq_gradient_per_field
from zoomy_core.mesh import ensure_lsq_mesh


def _write_field_names(filepath, names):
    """Stamp ``/fields@names`` (state names in ``Q`` row order) on the store at
    ``filepath`` (absolute), byte-identical to what
    ``zoomy_core.misc.io.write_mesh_to_hdf5(..., names=...)`` writes for the
    numpy solvers -- the convention ``zoomy_plotting.read_hdf5`` resolves.
    Written once at setup; the per-snapshot writer appends into the same
    ``/fields`` group."""
    with h5py.File(filepath, "a") as f:
        f.require_group("fields").attrs["names"] = np.array(
            [str(n) for n in names], dtype="S")


# ── Inter-level coupling: THE PRODUCER of ``coupled_dm`` ────────────────────
#
# ``flux_operator`` below already CONSUMES a precomputed face contribution at
# masked coupled boundary faces (``face_contrib_bnd`` is replaced wholesale by
# ``coupled_dm``), threaded per window through ``step(..., coupled=(...))``.
# Nothing produced it.  These two functions do.

def interlevel_pad_low_state(q_low, n_high):
    """Lift a LOW participant's interface state ``(n_low, n_faces)`` into the
    HIGH space ``(n_high, n_faces)`` by zero-padding the missing moments.

    The padding is a FORMALITY at run time: the emitted kernel already had
    literal ``sp.Integer(0)`` substituted into rows ``[n_low, n_high)`` at
    BUILD time (``Numerics._zero_padded_low_state``), so those inputs are dead
    and CSE deleted every subtree they fed.  The array is padded anyway
    because the emitted signature is sized from ``n_dof_q`` on every backend
    (foam/amrex size ``Q_minus`` from it, and the C printers' argument
    whitelists carry exactly six face keys) — so a short array is not
    passable, while a padded one costs nothing.
    """
    q_low = jnp.asarray(q_low)
    n_low = q_low.shape[0]
    if n_low == n_high:
        return q_low
    if n_low > n_high:
        raise ValueError(
            f"interlevel_pad_low_state: low state has {n_low} rows, more than "
            f"the high space's {n_high}.")
    return jnp.concatenate(
        [q_low, jnp.zeros((n_high - n_low,) + q_low.shape[1:], q_low.dtype)],
        axis=0)


def assert_interface_bed_is_flat(q_low, q_high, bed_index, *, tol=0.0):
    """ASSERT ``Δb = 0`` across a coupling interface — the scope condition of
    the discharge ledger, checked rather than assumed.

    The ``q_0`` row telescopes across an inter-level face ONLY on a flat bed:
    its nonconservative row carries the live bed column ``B[q_0][b] = g·h``,
    so the two sides' contributions sum to ``-B[q_0][b]·Δb``, which is zero
    iff ``Δb = 0``.  MEASURED on SME(0)↔SME(1): residual ``0.0`` exactly with
    a flat bed, ``7.0e-01`` across a 0.4 m step, while ``h`` stays ``0.0`` in
    both.  So this is not a formality — it is the difference between a
    conservation claim that holds and one that is simply false.

    Raises ``ValueError`` naming the worst face.  A no-op on traced (jit)
    inputs, where a data-dependent check cannot be made; call it from the
    window driver, which holds concrete arrays.
    """
    try:
        db = np.asarray(jnp.asarray(q_high)[bed_index]
                        - jnp.asarray(q_low)[bed_index], dtype=float)
    except (jax.errors.TracerArrayConversionError, TypeError):
        return                       # traced: not checkable here, by design
    worst = float(np.abs(db).max()) if db.size else 0.0
    if worst > tol:
        face = int(np.argmax(np.abs(db)))
        raise ValueError(
            f"coupling interface is NOT flat: max|Δb| = {worst:.6e} at "
            f"boundary face {face} (tol {tol:g}).  The inter-level face "
            "operator conserves the DEPTH row across any bed, but the "
            "discharge row q_0 telescopes only when Δb = 0 — its "
            "nonconservative row carries the live bed column B[q_0][b] = g·h, "
            "leaving a residual -B[q_0][b]·Δb.  Either flatten the bed at the "
            "interface or drop q_0 from the conservation claim; do not assume "
            "it.")


def interlevel_face_contribution(rt_flux, rt_fluctuations, side, q_low, q_high,
                                 aux_low, aux_high, parameters, normal,
                                 *, n_low=None, flat_bed_row=None):
    """THE SHARED INTERFACE FACE CONTRIBUTION, for ONE side of an inter-level
    coupling — the ``coupled_dm`` that ``flux_operator`` consumes.

    ``rt_flux`` / ``rt_fluctuations`` are the HIGH participant's registered
    ``numerical_flux`` / ``numerical_fluctuations`` runtime kernels — BOTH arms
    of the interface are evaluated with the HIGH model's operator, which is the
    whole content of the fix.  ``side`` is ``"low"`` or ``"high"``.

    Returns an array of shape ``(rows_of_this_participant, n_faces)``, aligned
    to the boundary-face axis exactly as ``coupled_mask`` is — i.e. directly
    the ``coupled_dm`` argument of ``step(..., coupled=(coupled_dm,
    coupled_mask))``.

    ONE EVALUATION, TWO PROJECTIONS — AND THIS IS THE WHOLE POINT.
    The kernel is called ONCE, at the CANONICAL orientation ``normal = low →
    high``.  Both sides are then read out of that single packed answer:

        side="low"   ->  ( F + D_minus )[0 : n_low]
        side="high"  ->  ( -F + D_plus )

    which is precisely the interior-face convention this solver already uses
    (``dQ[A] -= (F + Dm)``, ``dQ[B] -= (-F + Dp)``), so a coupled face is
    discretised identically to an interior one.

    DO NOT "let both participants build the identical pair with their own
    outward normal".  That is the trap this function exists to close, and the
    older comment on the consumer said exactly the wrong thing.  ``nHat`` is
    normalised as ``n/(|n| + SMALL)``, so on an x-normal face ``nHat.x()`` is
    ``1 - 2.2e-16`` rather than ``1.0``, and ``0.5·(a+b)·nx`` then differs from
    ``0.5·(a·nx + b·nx)`` in the last bit — MEASURED to break bit-equality on
    ~62% of states (``swePreciceCoupling.C``).  Two independent evaluations of
    "the same" Riemann problem are not bit-identical; one evaluation and two
    sign projections is.

    THE TRUNCATION IS FREE.  The low participant reads rows ``0:n_low`` of the
    high-space answer and nothing else: no second kernel, no restriction
    operator, no projection matrix.  That is legitimate because of the
    hierarchy identity (V&K eq 3.66) — the high model's flux at the zero-padded
    low state restricted to the first ``n_L`` rows IS the low model's own flux.

    NO NEW EMITTED KERNEL.  This calls the participant's ALREADY-REGISTERED
    ``numerical_flux`` / ``numerical_fluctuations``, which take an explicit
    face pair; the zero-padding is a pure caller-side action.  So the whole
    construction needs no printer change, no new registered operator, and no
    new SystemModel API.  Proved in
    ``zoomy_core/tests/fvm/test_interlevel_telescoping.py`` over a state sweep.

    CONSERVATION IS ROW-SCOPED — AND THE SCOPE IS CHECKED, NOT ASSUMED.
    The two contributions sum to ``-(D_plus + D_minus) = -A_int·ΔQ``, which
    vanishes exactly on the rows whose nonconservative row is identically zero.
    For the K&T SME family that is ``h`` (and the stationary ``b``): ``h``
    telescopes across ANY bed, but the discharge row ``q_0`` telescopes only
    across a FLAT bed, because its nonconservative row carries the live bed
    column ``B[q_0][b] = g·h``.  Pass ``flat_bed_row=<index of b>`` to have that
    asserted here per window (see :func:`assert_interface_bed_is_flat`);
    without it the mass ledger still holds and the discharge one is not
    claimed.
    """
    if side not in ("low", "high"):
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")
    if flat_bed_row is not None:
        assert_interface_bed_is_flat(q_low, q_high, int(flat_bed_row))

    q_high = jnp.asarray(q_high)
    n_high = q_high.shape[0]
    q_low_padded = interlevel_pad_low_state(q_low, n_high)

    # THE single evaluation, at the canonical low->high orientation.
    args = (q_low_padded, q_high, aux_low, aux_high, parameters, normal)
    F = jnp.asarray(rt_flux(*args)).reshape(n_high, -1)
    fluct = jnp.asarray(rt_fluctuations(*args))
    Dp = jnp.asarray(fluct[0]).reshape(n_high, -1)
    Dm = jnp.asarray(fluct[1]).reshape(n_high, -1)

    if side == "low":
        rows = n_low if n_low is not None else jnp.shape(q_low)[0]
        return (F + Dm)[0:int(rows)]
    return -F + Dp


# ── Logging callbacks (used by io_callback inside JIT) ──────────────────────

def log_callback_hyperbolic(iteration, time, dt, time_stamp, log_every=10):
    """Log callback hyperbolic."""
    if iteration % log_every == 0:
        logger.info(
            f"iteration: {int(iteration)}, time: {float(time):.6f}, "
            f"dt: {float(dt):.6f}, next write at time: {float(time_stamp):.6f}"
        )
    return None


def log_callback_poisson(iteration, res):
    """Log callback poisson."""
    logger.debug(
        f"Newton iterations: {iteration}, final residual norm: {jnp.linalg.norm(res):.3e}"
    )
    return None


def log_callback_execution_time(time):
    """Log callback execution time."""
    logger.info(f"Finished simulation with in {time:.3f} seconds")
    return None


# ── Newton solver (for PoissonSolver) ───────────────────────────────────────

def newton_solver(residual):
    """Newton solver."""
    def Jv(Q, U):
        """Jv."""
        return jax.jvp(lambda q: residual(q), (Q,), (U,))[1]

    @jax.jit
    @partial(jax.named_call, name="preconditioner")
    def compute_diagonal_of_jacobian(Q):
        """Compute diagonal of jacobian."""
        ndof, N = Q.shape

        def compute_entry(i, j):
            """Compute entry."""
            e = jnp.zeros_like(Q).at[i, j].set(1.0)
            J_e = Jv(Q, e)
            return J_e[i, j]

        def outer_loop(i, diag):
            """Outer loop."""
            def inner_loop(j, d):
                """Inner loop."""
                val = compute_entry(i, j)
                return d.at[i, j].set(val)

            return jax.lax.fori_loop(0, N, inner_loop, diag)

        diag_init = jnp.zeros_like(Q)
        return jax.lax.fori_loop(0, ndof, outer_loop, diag_init)

    @jax.jit
    @partial(jax.named_call, name="newton_solver")
    def newton_solve(Q):
        """Newton solve."""
        def cond_fun(state):
            """Cond fun."""
            _, r, i = state
            maxiter = 10
            return jnp.logical_and(jnp.linalg.norm(r) > 1e-6, i < maxiter)

        def body_fun(state):
            """Body fun."""
            Q, r, i = state

            def lin_op(v):
                """Lin op."""
                return Jv(Q, v)

            delta, info = gmres(
                lin_op,
                -r,
                x0=jnp.zeros_like(Q),
                maxiter=10,
                solve_method="incremental",
                restart=100,
                tol=1e-6,
            )

            def backtrack(alpha, Q, delta, r):
                """Backtrack."""
                def cond(val):
                    """Cond."""
                    alpha, _, _ = val
                    return alpha > 1e-3

                def body(val):
                    """Body."""
                    alpha, Q_curr, r_curr = val
                    Qnew = Q_curr + alpha * delta
                    r_new = residual(Qnew)
                    improved = jnp.linalg.norm(r_new) < jnp.linalg.norm(r_curr)

                    return jax.lax.cond(
                        improved,
                        lambda _: (0.0, Qnew, r_new),  # Accept and stop
                        lambda _: (alpha * 0.5, Q_curr, r_curr),  # Retry
                        operand=None,
                    )

                return jax.lax.while_loop(cond, body, (alpha, Q, r))[1:]

            Q_new, r_new = backtrack(1.0, Q, delta, r)

            return (Q_new, r_new, i + 1)

        r0 = residual(Q)
        init_state = (Q, r0, 0)

        Q_final, res, i = jax.lax.while_loop(cond_fun, body_fun, init_state)

        jax.experimental.io_callback(log_callback_poisson, None, i, res)

        return Q_final

    return newton_solve


# ── HyperbolicSolver ────────────────────────────────────────────────────────

class HyperbolicSolver(HyperbolicSolverNumpy):
    """JAX HyperbolicSolver — JIT-compiled explicit time stepping.

    Follows the setup_simulation / step / run_simulation pattern.
    Inherits param definitions from the NumPy base class.
    """

    # ── Free-surface-aware reconstruction (Audusse-Bouchut 2005,
    # Kurganov-Petrova 2007, Xing-Zhang 2013).  When
    # ``free_surface_h_index`` is set AND ``reconstruction_variables``
    # is one of ``"eta"`` / ``"xz"``, the order≥2 reconstruction
    # switches from plain conservative-variable LSQ-MUSCL to a
    # wet/dry-aware variant.  At order=1 these flags have no effect
    # (constant reconstruction is already cell-wise positive).
    # Caller-driven — the solver does not auto-detect from state
    # variable names.
    free_surface_h_index = param.Integer(default=None, allow_None=True,
        doc="State-vector index of h.  Enables wet-dry MUSCL fallback.")
    free_surface_b_index = param.Integer(default=0, allow_None=True,
        doc="State-vector index of bathymetry b.  Used only when "
            "``reconstruction_variables='eta'`` to form η = h + b.")
    # REQ-48: the wet/dry threshold is NOT a solver parameter — it is the
    # NSM-owned canonical parameter ``wet_dry_eps`` (populated from the model,
    # also read by the FVM riemann solver).  The eta reconstruction reads it
    # from ``nsm.parameter_values.wet_dry_eps``; models that declare none let
    # the reconstruction use its own default.
    free_surface_momentum_indices = param.List(default=None, allow_None=True,
        doc="Indices of momentum components.  Defaults to "
            "``h_index+1 … h_index+1+dim``.")
    reconstruction_variables = param.Selector(
        default="conservative", objects=["conservative", "eta", "xz"],
        doc="What to reconstruct at order≥2.  ``conservative`` = the "
            "plain LSQ-MUSCL on (b, h, hu, hv).  ``eta`` = primitive-"
            "variable MUSCL on (b, η = h + b, hu, hv), the standard "
            "Audusse-Bouchut 2005 / Kurganov-Petrova 2007 well-balanced "
            "wet/dry recipe.  ``xz`` = Xing-Zhang 2013 cell-mean "
            "positivity scaling on the conservative reconstruction.")
    positivity_method = param.Selector(
        default="", objects=["", "zhang_shu"],
        doc="A-priori cell-mean positivity at order≥2.  '' = none; "
            "'zhang_shu' = Xing-Zhang-Shu 2010 deviation cap so h≥0 holds for "
            "the cell mean under CFL ≤ 1/(2k+1), with NO a-posteriori step and "
            "NO dt-halving.  Currently wired for ``reconstruction_variables="
            "'eta'`` (the well-balanced wet/dry recipe).")
    front_theta_tol = param.Number(default=None, allow_None=True, bounds=(0, 1),
        doc="A-priori wet/dry-front PRE-DETECTOR (order≥2, eta path).  None = "
            "off.  A value ∈ (0, 1] demotes to genuine 1st order (φ:=0) every "
            "cell whose positivity indicator θ_front < tol — i.e. the wet/dry "
            "shoreline + steep h extrema (Gallardo-Parés-Castro).  Reuses the "
            "_pp_theta indicator.  Lets the run use the linear-stability CFL "
            "instead of the a-priori-positivity 1/(2k+1); pair with ``mood``.")
    mood = param.Boolean(default=True,
        doc="A-posteriori MOOD corrector — THE DEFAULT order≥2 positivity "
            "mechanism (user directive 2026-07-27: MOOD, not Zhang-Shu, so there "
            "is never confusion about which guard/CFL is active).  ON by default; "
            "set False only to compare against an unguarded O2.  Engages only on "
            "the eta reconstruction (``reconstruction_variables='eta'``), the one "
            "path that exposes per-cell O1 demotion + an emitted-W O1-HR "
            "fallback; on any other reconstruction it is inert (a purely-wet "
            "conservative O2 run needs no wet/dry net).  Mechanism (per step, no "
            "dt-halving ever): take the SSP-RK2 O2 candidate; flag inadmissible "
            "cells (PAD h<0 + CAD non-finite); recompute those at O1 Audusse-HR "
            "(φ:=0 → the piecewise-constant emitted-W reconstruction, positivity-"
            "preserving by the XZS 1st-order lemma + Audusse face clip, "
            "conservative via the shared face flux).  If whole-CELL demotion "
            "still leaves an inadmissible cell (a draining front where an O2 wet "
            "neighbour fluxes a denormal −h into the demoted cell — which the "
            "next step's sqrt(g·h) eigenvalue turns into a NaN), escalate to a "
            "whole-STEP O1-HR redo (amrex whole-step O1 MOOD), which matches the "
            "pure-O1 march and is provably admissible.  Replaces the a-priori "
            "Xing-Zhang-Shu cap (``positivity_method='zhang_shu'`` + CFL≤1/(2k+1)) "
            "as the default — that code stays available but is no longer the gate.")
    march_log_interval = param.Integer(default=None, allow_None=True,
        bounds=(0, None),
        doc="Steps between the in-loop progress prints, or 0 to silence them.  "
            "``None`` (default) = AUTO: silent on an ACCELERATOR backend, every "
            "10 steps on CPU.\n\n"
            "The print sits inside the jitted ``lax.while_loop``, so every "
            "firing is a device->host round trip that serialises the march.  On "
            "a GPU that dominates: MEASURED on Malpasset (26k cells, order 1, "
            "CFL 0.4, L40S) 5.95 -> 0.63 s of wall per simulated second, i.e. "
            "9.4x, with ``nvidia-smi`` reading 0% utilisation while the print "
            "was live and 99% once it was gone.  A whole 2500 s order-2 march "
            "went 6.4 -> 1.14 s/sim-s (5.6x; smaller than the microbenchmark "
            "because a larger dt fires the callback less often per simulated "
            "second).  Auto-off on accelerators keeps the progress trace where "
            "it is cheap (CPU debugging) and drops it where it is not.  The "
            "branch is resolved in PYTHON at trace time, so when it is off "
            "nothing is emitted into the jaxpr at all.")
    force_steady_ode_wb = param.Boolean(default=False,
        doc="Force the fully-WB steady-ODE moving-equilibrium reconstruction "
            "even for a frictionless level-0 model that would otherwise take the "
            "closed-form Bernoulli fast path.  R=0 ⇒ the steady ODE integrates to "
            "the SAME equilibrium as the closed form, so this exists only to "
            "compare the two moving-equilibrium WB reconstructions head-to-head "
            "on ONE model (cid 157, wb_comparison Case A).  Default off ⇒ every "
            "existing golden is unaffected.  1-D only (the ndim≠1 guard already "
            "raises for the ODE path).")

    def __init__(self, **kwargs):
        # Riemann numerics are now owned by the NSM (nsm.riemann is the
        # symbolic class; nsm.build_numerics() instantiates it).  The
        # JaxRuntime built in setup_simulation lambdifies that
        # Numerics into jit-vmapped per-face callables.  No more legacy
        # _jax_flux / _jax_nc_flux kwargs.
        super().__init__(**kwargs)

    # ── Runtime creation ────────────────────────────────────────────────

    def initialize(self, mesh, model):
        """Allocate state arrays of shape ``(n_var, n_inner_cells)`` —
        no ghost cells.  Boundary values are computed inline inside
        the flux operator from the indexed BC kernel; the LSQ-MUSCL
        reconstruction sees them via the LSQ-augmented stencil
        (``lsq_boundary_face_neighbors``) and uses the BC face values
        as ``Q_R`` at boundary faces (NumPy parity).

        Matches the NumPy contract: ``Q.shape[1] == mesh.n_inner_cells``.
        """
        Q, Qaux = super().initialize(mesh, model)
        return Q, Qaux

    def create_runtime(self, Q, Qaux, mesh, model):
        """Create the JAX runtime over the NSM.

        ``model`` here is ``self.nsm.sm`` — the SystemModel inside the
        NSM that ``setup_simulation`` resolved.  Builds a
        :class:`JaxRuntime` (jit-vmapped per-cell + per-face operators)
        and converts the LSQ mesh to its JAX form.
        """
        jax_mesh = convert_mesh_to_jax(mesh)
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        runtime = JaxRuntime.from_nsm(self.nsm)
        # Parameters are live on the runtime; snapshot for the runtime
        # state stored on self.
        parameters = runtime.parameters
        return Q, Qaux, parameters, jax_mesh, runtime

    # ── State update (vmap over cells) ──────────────────────────────────

    def update_q(self, Q, Qaux, mesh, model, parameters):
        """Apply the SystemModel's per-cell ``update_variables`` to every cell.

        Uses ``model.update_variables`` as a **full-grid** callable —
        the JaxRuntime now lambdifies + vmaps that slot once at
        construction time, so this is a single JIT'd call (was: a
        Python-side ``jax.vmap`` per invocation that retraces every step).
        ``None`` on the runtime means the SystemModel had no
        ``update_variables`` slot ⇒ identity; we short-circuit.
        """
        fn = getattr(model, "update_variables", None)
        if fn is None:
            return Q
        return fn(Q, Qaux, parameters)

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model,
                    parameters, time, dt):
        """Apply the SystemModel's per-cell ``update_aux_variables`` to every
        cell — the aux leg of ``post_step``, symmetric to :meth:`update_q`.

        The JaxRuntime lambdifies + vmaps the ``update_aux_variables`` slot
        once at construction (a single JIT'd, traceable call), so e.g. the
        KP-desingularized ``hinv = √2·h/√(h⁴+max(h,eps)⁴)`` is recomputed from
        the current ``h`` every step.  ``None`` on the runtime ⇒ the model
        declared no per-cell aux formula ⇒ identity; short-circuit.

        This OVERRIDES the inherited NumPy ``Solver.update_qaux`` registry
        walker, which is a guaranteed no-op on JAX (no ``self.sm`` /
        ``_chain_systemmodel``) and is not JIT-traceable.  The other
        post_step/residual call-sites (setup, predictor, Newton) all route
        through this one method, so they pick the aux update up uniformly.
        """
        out = Qaux
        # (1) LOCAL aux formula leg (e.g. KP hinv).  The runtime lambdifies the
        # SystemModel's ``update_aux_variables`` as a FULL-LENGTH ``(n_aux, ·)``
        # vector — each row is either an algebraic formula (``hinv = KP(h)``) or
        # a passthrough of its own aux symbol (the SystemModel sizes it to
        # ``len(aux_state)`` via ``_to_matrix``).  Writing it as a PREFIX slice
        # silently mis-places the algebraic rows onto the leading
        # (derivative/LSQ-gradient) rows whenever the model APPENDS an aux (e.g.
        # ``hinv`` registered last): that row is then clobbered by the
        # derivative walk (2) below, ``hinv`` stays 0, and every ``hinv``-scaled
        # moment source/friction vanishes → ``SME(level≥1)`` degenerates to SWE
        # (task 0017 — was worked around case-side by emitting a full-length
        # passthrough vector).  So enforce the full-length contract and replace
        # by row, never by prefix.
        fn = getattr(model, "update_aux_variables", None)
        if fn is not None:
            # REQ-185: update_aux_variables is declared
            # ``(Q, Qaux, p, time, position)`` — thread the current ``time``
            # (a rain-rate aux ``r_o = Piecewise((rate, t<T_rain),(0,True))``
            # binds it) and the per-cell centres (a manufactured spatial aux
            # binds position).  ``mesh.cell_centers`` is 3-padded and sliced to
            # this Q's column count; state-only aux (KP ``hinv``) ignores both.
            local = fn(Q, Qaux, parameters, time=time,
                       position=mesh.cell_centers[:, :Q.shape[-1]])
            if local.shape[0] != out.shape[0]:
                raise ValueError(
                    f"update_aux_variables must be full-length "
                    f"({out.shape[0]} aux rows) so each algebraic aux lands on "
                    f"its own row — got {local.shape[0]} rows.  A partial vector "
                    f"is prefix-written and silently degenerates SME(≥1)→SWE "
                    f"(task 0017); emit a passthrough on every non-formula row.")
            out = local
        # (2) DERIVATIVE aux leg — the non-local LSQ-gradient rows the
        # SystemModel gathered in aux_registry, via the shared walk (the SINGLE
        # source the Chorin per-pool refresh also calls).
        out = self._walk_derivative_aux(getattr(model, "sm", None), out, Q, mesh)
        return out

    @staticmethod
    def _walk_derivative_aux(sm, Qaux, Q, mesh, *, kinds=("derivative",),
                             target_kinds=("state", "function"),
                             registry=None):
        """Fill ``aux_registry`` derivative rows of ONE SystemModel via the
        shared BC-aware LSQ kernel — the single source the canonical
        :meth:`update_qaux` AND ``ChorinSplitVAMSolverJax._refresh_aux_for_sm``
        both call.  ``kinds`` / ``target_kinds`` keep each caller's exact scope
        (canonical: spatial ``derivative``, state+function; Chorin: also
        ``limited_derivative``, state only).  ``registry`` overrides the entry
        list (the Chorin caller passes ``aux_registry + aux_input_registry`` —
        see the splitter's ``_partition_pressure_aux``).  jit-traceable — the
        registry is static so the loop unrolls at trace time."""
        out = Qaux
        if registry is None:
            registry = (getattr(sm, "aux_registry", None)
                        if sm is not None else None)
        if not registry:
            return out
        for e in registry:
            if e["kind"] not in kinds:
                continue
            tk = e["target_kind"]
            if tk not in target_kinds:
                continue
            field = (Q[e["state_index"]] if tk == "state"
                     else out[e["function_row"]])
            grad = lsq_gradient_per_field(
                mesh, field, u_bf=None, multi_index=e["multi_index"])
            out = out.at[e["row"], :grad.shape[0]].set(grad)
        return out

    # ── Operator builders ───────────────────────────────────────────────

    def get_compute_source(self, mesh, runtime):
        """JIT-compiled source operator using JaxRuntime.source (which
        is itself jit-vmap'd over the cell axis).

        REQ-185: the timestepping calls this as
        ``(dt, time, Q, Qaux, parameters, dQ)`` and threads the current
        simulation ``time``, the step ``dt`` and the inner-cell centre
        positions into ``JaxRuntime.source`` (declared
        ``source(Q, Qaux, p, time, dt, position)``), so a time/space-dependent
        source binds them.  An autonomous source ignores the extra args
        (the runtime defaults are the coordinate origin — bit-identical)."""
        nc = mesh.n_inner_cells
        rt_source = runtime.source
        # Inner-cell centre positions (3, nc) — the length-3 position VECTOR X
        # of the REQ-185 source signature; ``mesh.cell_centers`` is 3-padded.
        cell_centers = mesh.cell_centers[:, :nc]

        @jax.jit
        @partial(jax.named_call, name="source")
        def compute_source(dt, time, Q, Qaux, parameters, dQ):
            S = rt_source(Q[:, :nc], Qaux[:, :nc], parameters,
                          time=time, dt=dt, position=cell_centers)
            return dQ.at[:, :nc].set(S)

        return compute_source

    def get_apply_boundary_conditions(self, mesh, runtime):
        """JIT-compiled BC operator that fills ghost cells via the
        indexed kernel on JaxRuntime.boundary_conditions."""
        rt_bc = runtime.boundary_conditions
        if rt_bc is None or int(getattr(mesh, "n_boundary_faces", 0)) == 0:
            # No BC kernel, or a mesh with no boundary faces (e.g. an interior
            # SPMD partition — all faces connect to halo cells, refreshed by
            # halo exchange).  The BC operator is identity; tracing the fori_loop
            # over a size-0 boundary_face array would otherwise gather OOB.
            return lambda time, Q, Qaux, parameters: Q

        @jax.jit
        @partial(jax.named_call, name="apply_boundary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            def loop_body(i, Q):
                i = jnp.asarray(i, dtype=jnp.int32)
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = jnp.linalg.norm(position - position_ghost)
                q_ghost = rt_bc(
                    i_bc_func, time, position, distance,
                    q_cell, qaux_cell, parameters, normal,
                )
                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)
                return Q

            return jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)

        return apply_boundary_conditions

    def get_compute_max_abs_eigenvalue(self, mesh, runtime):
        """PER-FACE max |eigenvalue|, using JaxRuntime.eigenvalues
        (jit-vmap'd per cell).  Returns an ``(n_faces,)`` array (NOT a global
        scalar) so the CFL couples each face's wave speed to its LOCAL cell
        size; ``compute_dt`` takes the min over faces -> ``min_f(h/λ)``, instead
        of the over-conservative ``global_min(h)/global_max(λ)``."""
        # cid 210: the dt spectrum is an NSM OUTPUT, not a backend decision.
        # ``eigenvalues_cfl`` is what ``gate_eigenvalues_dry`` publishes — dry
        # cells zeroed for the timestep, with the Rusanov dissipation left
        # physical on ``eigenvalues`` — and it is an NSM default for
        # depth-carrying transport systems.  Read it when present, else fall
        # back, exactly as the numpy consumer does.  This backend applies NO dry
        # threshold of its own: it evaluates the operator it is handed.  Before
        # this, jax read the RAW spectrum and gated nothing, so a near-dry
        # cell's u = q/h drove the CFL: on a 96x96 radial dry dam break that
        # collapsed dt to ~1.5e-08 and the run took 1,353,670 steps instead of
        # ~40.
        rt_eig = getattr(runtime, "eigenvalues_cfl", None) or runtime.eigenvalues

        @jax.jit
        @partial(jax.named_call, name="max_abs_eigenvalue")
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            iA = mesh.face_cells[0]
            iB = mesh.face_cells[1]
            normal = mesh.face_normals
            evA = jnp.abs(rt_eig(Q[:, iA], Qaux[:, iA], parameters, normal))
            evB = jnp.abs(rt_eig(Q[:, iB], Qaux[:, iB], parameters, normal))
            # Reduce the eigenvalue axis, KEEP the face axis (last).
            if evA.ndim > 1:
                red = tuple(range(evA.ndim - 1))
                evA = jnp.max(evA, axis=red)
                evB = jnp.max(evB, axis=red)
            return jnp.maximum(evA, evB)

        return compute_max_abs_eigenvalue

    def _build_reconstruction(self, mesh, symbolic_model, runtime=None):
        """Build JAX face reconstruction.

        First-order: piecewise-constant.  Second-order: the LSQ-augmented
        MUSCL (``LSQMUSCLReconstructionJAX``) — mirrors NumPy
        ``LSQMUSCLReconstruction``, takes ``Q`` of shape
        ``(n_var, n_inner_cells)`` and uses the BC-provided
        ``bf_face_values`` for both the limiter bounds and ``Q_R`` at
        boundary faces.

        Free-surface / positivity-preserving path: when
        ``self.free_surface_h_index`` is set the order≥2 branch
        instead uses :class:`PositivityPreservingLSQMUSCLJAX`, which
        adds a per-cell Xing–Zhang 2013 slope-scaling pass on top of
        the standard limiter so the reconstructed ``h`` at every face
        of every cell stays ``≥ 0``.  Combined with the standard CFL
        bound (``dt · α / h_inradius ≤ 1/(2k+1)`` for degree-``k``
        reconstruction, so ``1/3`` for linear LSQ-MUSCL), this is
        sufficient to keep the cell-mean ``h`` non-negative over each
        explicit step — the gap that the face-clamp + dry-cell-
        fallback in :class:`FreeSurfaceLSQMUSCLJAX` leaves open over
        long horizons on real wet/dry meshes (Malpasset T = 100 s).
        """
        from zoomy_jax.fvm.reconstruction_jax import (
            ConstantReconstruction, LSQMUSCLReconstructionJAX,
            PositivityPreservingLSQMUSCLJAX,
            EmittedWLSQMUSCLJAX,
        )
        dim = symbolic_model.dimension
        # MANDATE 6a: there is no ``eps_wet`` to plumb any more.  The ``eta``
        # reconstruction consumes the EMITTED reconstruction-variable pair,
        # whose ``1/h`` is already core's KP-desingularized ``hinv`` at the
        # canonical ``wet_dry_eps`` — so the threshold lives in core, is
        # emitted, and cannot be contradicted by a backend default.  The old
        # code read ``wet_dry_eps`` from the NSM and FELL BACK to the class
        # literal ``1e-3`` when the model declared none; SWE declares none, so
        # the fallback is what actually ran, and it was the dry-bed order-2
        # collapse.
        # The ONLY wet/dry threshold this solver will pass to the
        # reconstruction: the model's EMITTED ``wet_dry_eps``.  Absent => None
        # => the reconstruction does no dry-cell demotion (and says so).  There
        # is deliberately no fallback literal; the old fallback WAS the bug.
        emitted_wet_dry_eps = None
        for _z in (getattr(self.nsm, "parameter_values", None),
                   getattr(self.nsm, "parameters", None)):
            if _z is not None and getattr(_z, "contains", None) \
                    and _z.contains("wet_dry_eps"):
                try:
                    emitted_wet_dry_eps = float(getattr(_z, "wet_dry_eps"))
                    break
                except (TypeError, ValueError):
                    pass
        order = self.nsm.reconstruction.order
        mode = self.reconstruction_variables
        limiter = self.nsm.reconstruction.limiter
        # The Audusse-Bouchut η/HR free-surface reconstruction runs at BOTH
        # orders.  At order 1 it is piecewise-constant in ``W`` (zero slope) but
        # STILL applies the free-surface face-state machinery — the W-space
        # Audusse clip ``η_f := max(η_f, b_f)`` (⇒ ``h_f = max(0, η_f−b_f) ≥ 0``)
        # and the emitted inverse ``q_f = u_f·h_f`` (dry-face momentum vanishes
        # as a matched pair).  That clip is what makes first order positivity-
        # preserving at a draining wet/dry front; the plain
        # ``ConstantReconstruction`` order-1 fallback below omits it (the flux is
        # ``NonconservativeRusanov``, which carries no hydrostatic
        # reconstruction), so O1 pushed ``h < 0`` and NaN'd (2-D Thacker planar).
        # O2 is untouched: ``piecewise_constant`` is False and the a-priori
        # positivity / front knobs pass through exactly as before.
        if mode == "eta":
            assert self.free_surface_h_index is not None, (
                "reconstruction_variables='eta' requires "
                "free_surface_h_index to be set")
            if runtime.state_from_reconstruction_uses_aux:
                raise NotImplementedError(
                    "This model's emitted ``state_from_reconstruction`` "
                    "reads AUX variables, but the face pass has "
                    "only CELL-centre aux — feeding it there would "
                    "silently use the wrong aux at every face.  Reported, "
                    "not worked around.")
            return EmittedWLSQMUSCLJAX(
                mesh, dim,
                runtime.reconstruction_variables,
                runtime.state_from_reconstruction,
                aux_of_q=getattr(runtime, "update_aux_variables", None),
                eps_wet=emitted_wet_dry_eps,
                b_index=int(self.free_surface_b_index),
                h_index=int(self.free_surface_h_index),
                momentum_indices=self.free_surface_momentum_indices,
                positivity=(self.positivity_method or None) if order >= 2 else None,
                front_tol=self.front_theta_tol if order >= 2 else None,
                piecewise_constant=(order < 2),
                limiter=limiter,
            )
        if order >= 2:
            if mode == "xz":
                assert self.free_surface_h_index is not None, (
                    "reconstruction_variables='xz' requires "
                    "free_surface_h_index to be set")
                return PositivityPreservingLSQMUSCLJAX(
                    mesh, dim,
                    h_index=int(self.free_surface_h_index),
                    momentum_indices=self.free_surface_momentum_indices,
                    limiter=limiter,
                )
            # mode == "conservative" — plain LSQ-MUSCL on (b, h, hu, hv).
            # Back-compat: if free_surface_h_index is set without an
            # explicit mode, default to xz (matches the prior behavior
            # introduced when free_surface_h_index was first added).
            if self.free_surface_h_index is not None:
                return PositivityPreservingLSQMUSCLJAX(
                    mesh, dim,
                    h_index=int(self.free_surface_h_index),
                    momentum_indices=self.free_surface_momentum_indices,
                    limiter=limiter,
                )
            return LSQMUSCLReconstructionJAX(
                mesh, dim, limiter=limiter)
        return ConstantReconstruction(mesh, dim)

    def get_flux_operator(self, mesh, runtime):
        """Flux operator — literal NumPy parity (S5b).

        * ``Q`` has shape ``(n_var, n_inner_cells)`` — no ghost cells.
        * Boundary face values are evaluated inline from the BC kernel
          (vmap'd over the boundary face axis).
        * The LSQ-MUSCL reconstruction is fed ``Q + bf_values`` so its
          limiter bounds and ``Q_R`` at boundary faces are physically
          consistent.
        * Interior + boundary faces are accumulated in separate
          accumulators (split loops).
        """
        reconstruct = self._build_reconstruction(mesh, runtime.sm, runtime)
        rt_num_flux = runtime.numerical_flux
        rt_num_fluct = runtime.numerical_fluctuations
        rt_bc = runtime.boundary_conditions

        # Whether the reconstruction honours a per-cell ``force_o1`` demotion
        # mask (the eta path does) — required by the a-posteriori MOOD corrector.
        support_o1 = bool(getattr(reconstruct, "supports_force_o1", False))
        # MOOD is the DEFAULT order≥2 positivity mechanism (``mood`` defaults
        # True), but it can only engage where the reconstruction exposes per-cell
        # O1 demotion — i.e. ``reconstruction_variables='eta'`` (the only one with
        # an emitted-W O1-HR fallback).  On any other reconstruction there is
        # nothing to demote to, so MOOD is inert: a purely-wet conservative O2 run
        # needs no wet/dry positivity net, and the default-on flag must NOT trip
        # every such run.  Gate on ``support_o1`` and record it for the step
        # driver — no raise (the old ``mood=True and not support_o1`` error would
        # now fire on every conservative O2 march, since the default is True).
        self._mood_active = bool(
            self.mood and support_o1 and self.nsm.reconstruction.order >= 2)

        # Cell-interior non-conservative integral (path-conservative,
        # order ≥ 2) — mirrors NumPy ``solver_numpy.get_flux_operator``:
        # ∫_cell B(Q)·∂_x Q dx ≈ B(Q_c)·s_c with the limited slope s_c.
        # REQUIRED for well-balancing at order ≥ 2 (the face fluctuations
        # carry only the inter-cell jump; this is the intra-cell smooth
        # part).  Active only when the reconstruction exposes its limited
        # gradient (plain LSQ-MUSCL); order 1 has slope ≡ 0.
        rt_ncm = getattr(runtime, "nonconservative_matrix", None)
        # THE EMITTED interior nonconservative term (core
        # ``numerical_cell_edge_source``).  Preferred over the inline volume
        # form above when the runtime provides it: it is the SAME
        # Gauss-Legendre path integral the face fluctuations use, so the two
        # halves of one nonconservative product share a quadrature — which is
        # what the order-2 well-balancing cancellation depends on.
        rt_ces = getattr(runtime, "numerical_cell_edge_source", None)
        # The emitted-W reconstruction needs Qaux + parameters at call time
        # (its ``W`` map reads the desingularized ``hinv`` aux), plus the
        # emitted LOCAL aux formula to rebuild aux on ghost/boundary states.
        from zoomy_jax.fvm.reconstruction_jax import EmittedWLSQMUSCLJAX
        needs_emitted_w = isinstance(reconstruct, EmittedWLSQMUSCLJAX)
        rt_uav = getattr(runtime, "update_aux_variables", None)

        # ── Moving-equilibrium (Bernoulli) well-balanced reconstruction ──────
        # Reconstruct the interface states to the common bed b* preserving the
        # discharge q and the per-column specific energy, then add the
        # conservative-flux-jump source Fc(Q)−Fc(Q*) (Fc = flux + hydrostatic
        # pressure).  The reconstruction MATH lives in core
        # (``reconstruct_bernoulli_level0`` + the ``newton_solve`` opaque
        # kernel) — this backend supplies the loop; the Fc jump REUSES the
        # emitted ``flux`` / ``hydrostatic_pressure`` kernels.  Mirrors numpy's
        # ``solver_numpy`` step 3c.  Level-0 (SWE) 1-D only (v1): the streamline
        # head is not yet a depth-resolved velocity vector.
        _eqr = getattr(runtime.sm, "equilibrium_reconstruction", "none")
        # cid 214: 'audusse' is the CORE lake-at-rest hydrostatic reconstruction
        # (``bernoulli_wb._reconstruct_audusse``, selected by the model's
        # ``equilibrium_reconstruction``).  This backend used to REFUSE it and
        # tell the caller to use numpy, while carrying its own Audusse-style
        # clip inside the reconstruction — i.e. reimplementing a scheme core
        # already owned.  A backend supplies the LOOP; core supplies the SCHEME.
        # It now routes to the same kernel numpy uses, which is backend-agnostic
        # (``xp``) for exactly this reason.
        bern_enabled = _eqr in ("audusse", "bernoulli", "projected_bernoulli")
        bern_recon = None
        bern_ode = False
        _bb = _bh = _bndim = None
        rt_flux = rt_press = None
        if _eqr == "audusse":
            # Lake-at-rest hydrostatic reconstruction — CORE's kernel, reached
            # FIRST because it is dimension- and level-agnostic and so takes
            # none of the 1-D / level-0 restrictions the moving-equilibrium
            # modes below carry (those would raise on a 2-D friction case
            # before ever reaching the reconstruction).
            from zoomy_core.fvm.bernoulli_wb import (
                build_bernoulli_config as _build_wb_cfg,
                reconstruct as _core_reconstruct)
            _bstate = [str(x) for x in runtime.sm.state]
            _bh = _bstate.index("h")
            _bb = _bstate.index("b")
            _bndim = int(runtime.sm.flux.shape[1])
            rt_flux = runtime.flux
            rt_press = runtime.hydrostatic_pressure
            _aud_cfg = _build_wb_cfg(runtime.sm, mode="audusse")

            def bern_recon(Qf, bstar):
                return _core_reconstruct(Qf, bstar, _aud_cfg, jnp)

        elif bern_enabled:
            import sympy as _sp
            from zoomy_core.fvm.bernoulli_wb import (
                reconstruct_bernoulli_level0, make_steady_ode_slope,
                moving_equilibrium_cells)
            from zoomy_jax.fvm.userfunctions import (newton_solve, solve as _solve_op,
                                                     solve_steady_ode)
            # Indices from the STATE order (== Q rows == quasilinear rows/cols).
            _bstate = [str(s) for s in runtime.sm.state]
            if "h" not in _bstate or "b" not in _bstate:
                raise ValueError("bernoulli WB needs 'h' and 'b' fields")
            _bh = _bstate.index("h")
            _bb = _bstate.index("b")
            _bq = [i for i, nm in enumerate(_bstate) if nm not in ("h", "b")]
            _bg = float(runtime.sm.parameter_values["g"])
            _bndim = int(runtime.sm.flux.shape[1])
            rt_flux = runtime.flux
            rt_press = runtime.hydrostatic_pressure
            # WHICH well-balanced kernel to run is MODEL knowledge, resolved by
            # the system model (resolve_equilibrium_kernel).  This backend used
            # to rebuild that decision itself: it reconstructed the moment level
            # by counting non-h/b state fields and ran sympy.simplify over the
            # source to sniff for friction.  A solver has no business knowing
            # what a "level" is -- that is the model's story -- so it now just
            # reads the answer.  Plain Python at trace time: nothing about level
            # or source reaches generated jax code.
            _wb_kernel = runtime.sm.resolve_equilibrium_kernel(
                force_steady_ode=bool(self.force_steady_ode_wb))
            bern_ode = (_wb_kernel == "steady_ode")
            if runtime.sm.flux.shape[1] != 1 and bern_ode:
                raise NotImplementedError(
                    "moving-equilibrium ODE WB (friction / level ≥ 1) is 1-D "
                    "only (v1).")

            if not bern_ode:
                _bq0 = _bq[0]

                def bern_recon(Qf, bstar):
                    return reconstruct_bernoulli_level0(
                        Qf, bstar, _bh, _bb, _bq0, _bg, jnp, newton_solve)
            else:
                # FULLY well-balanced moving-equilibrium reconstruction via the
                # local stationary ODE U_x = -(J_F+B)^{-1}(S H_x + R) (friction R
                # inside the RHS): integrate the emitted operators to the
                # intercells with ``solve_steady_ode`` (implicit midpoint), and
                # DISABLE the separate additive source — friction is carried by
                # the equilibrium U* alone (paper eq 28, midpoint quadrature).
                self._bern_ode = True
                _n_state = len(_bstate)
                rt_quasi = getattr(runtime, "quasilinear_matrix", None)
                rt_src = runtime.source
                if rt_quasi is None:
                    raise NotImplementedError(
                        "moving-equilibrium ODE WB needs the emitted "
                        "quasilinear_matrix (J_F+B); this runtime has none.")
                # 1-D left/right neighbour cell indices (fc0=left, fc1=right by
                # face normalisation); boundary cells point to themselves and are
                # excluded from the fluctuation slope (order-1 there).
                _ncinner = int(mesh.n_inner_cells)
                _fc0 = np.asarray(mesh.face_cells[0])
                _fc1 = np.asarray(mesh.face_cells[1])
                _bfset = set(int(f) for f in np.asarray(
                    mesh.boundary_face_face_indices))
                _int_faces = [f for f in range(int(mesh.n_faces))
                              if f not in _bfset]
                _left = np.arange(_ncinner, dtype=np.int64)
                _right = np.arange(_ncinner, dtype=np.int64)
                for _f in _int_faces:
                    _a, _b2 = int(_fc0[_f]), int(_fc1[_f])
                    _right[_a] = _b2
                    _left[_b2] = _a
                _interior_cell = (_left != np.arange(_ncinner)) & \
                                 (_right != np.arange(_ncinner))
                _left_j = jnp.asarray(_left)
                _right_j = jnp.asarray(_right)
                _int_cell_j = jnp.asarray(_interior_cell)
                _xc = np.asarray(mesh.cell_centers[0, :_ncinner])
                _xc_j = jnp.asarray(_xc)
                _dxc = _xc[_right] - _xc[_left]
                _dxc = np.where(_dxc == 0.0, 1.0, _dxc)   # boundary guard
                _dxc_j = jnp.asarray(_dxc)
                _cw_j = jnp.asarray(np.asarray(mesh.cell_volumes)[:_ncinner])
                rt_uav_ode = getattr(runtime, "update_aux_variables", None)
        # ``hasattr`` is NOT a valid test here and was the bug: it succeeds on
        # an INHERITED base-class ``reconstruct_with_grad``, so a subclass that
        # puts its positivity / wet-dry / WB treatment only in ``__call__``
        # (as PositivityPreservingLSQMUSCLJAX and FreeSurfaceLSQMUSCLJAX both
        # did) silently routed the order≥2 NCP path through the UNTREATED base
        # implementation.  Measured on a 1-D draining bed: the two paths
        # returned face states differing by 3.125e-03, with min h_face = 0 via
        # ``__call__`` but −3.125e-03 (NEGATIVE DEPTH) via the inherited
        # ``reconstruct_with_grad``.  The guard below was written to catch
        # exactly this and was defeated by attribute inheritance.
        #
        # A class opts in by providing its OWN ``reconstruct_with_grad``, or —
        # if it deliberately reuses a parent's implementation unchanged — by
        # declaring ``supports_grad_recon = True`` in its own class body.
        # Inheriting either one is not opting in.
        _cls = type(reconstruct)
        _grad_recon_ok = (
            "reconstruct_with_grad" in _cls.__dict__
            or bool(_cls.__dict__.get("supports_grad_recon", False)))
        use_interior_ncp = bool(
            self.nsm.reconstruction.order >= 2
            and rt_ncm is not None
            and (_grad_recon_ok or rt_ces is not None))
        if (self.nsm.reconstruction.order >= 2 and rt_ncm is not None
                and not use_interior_ncp):
            # Silently dropping the interior NCP at order >= 2 loses
            # well-balancing for NCP-bearing models (review note on
            # e67fc78): any reconstruction variant used here must expose
            # its limited gradient like LSQMUSCLReconstructionJAX does.
            raise NotImplementedError(
                f"order-{self.nsm.reconstruction.order} with a nonzero "
                "nonconservative_matrix requires the reconstruction to "
                "provide its OWN reconstruct_with_grad (limited cell "
                "gradient) for the cell-interior NCP integral; "
                f"{_cls.__name__} does not define one. Inheriting it from a "
                "base class does NOT count: a subclass that only overrides "
                "__call__ would run the base reconstruction with its "
                "positivity / wet-dry / well-balancing treatment skipped. "
                "Either implement reconstruct_with_grad on "
                f"{_cls.__name__}, or — if reusing the parent's is genuinely "
                "correct — declare supports_grad_recon = True on it.")

        # ── Explicit diffusion (REQ-50) ──────────────────────────────────
        # dQ += ∇·(A:∇Q) from the model's rank-4 ``diffusion_matrix_explicit``
        # (the explicit twin; else ``diffusion_matrix`` evaluated at Qⁿ),
        # applied via the SAME dense TPFA divergence the IMEX dense path uses
        # (REQ-109's DenseDiffusionOperatorJAX).  This carries cross-variable /
        # state-dependent tensors — e.g. the malpasset chain-rule cross term
        # ``A[2,1]=-D·u`` — unlike the numpy reference's DIAGONAL-only explicit
        # path (``solver_numpy._diffusion_in_flux`` per-variable
        # ``explicit_with_bc``).  ``JaxRuntime`` already lambdifies both tensors
        # (``None`` when the model carries neither), so this is a no-op for
        # models without diffusion.  Interior only (``bf_grads=None``); the
        # diffusive wall flux is left to a follow-up (the convective BC handles
        # boundaries, and A∝u ⇒ 0 at a no-slip wall / lake-at-rest anyway).
        rt_diff_expl = getattr(runtime, "diffusion_matrix_explicit", None)
        rt_diff = getattr(runtime, "diffusion_matrix", None)
        rt_diff_fn = rt_diff_expl if rt_diff_expl is not None else rt_diff
        dense_diff_op = None
        if rt_diff_fn is not None:
            from zoomy_jax.fvm.reconstruction_jax import DenseDiffusionOperatorJAX
            dense_diff_op = DenseDiffusionOperatorJAX(
                mesh, mesh.dimension, getattr(runtime, "n_state", 0),
                state_dependent=False)

        # Precompute face index arrays.
        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])
        bf_face_idx = np.asarray(mesh.boundary_face_face_indices)
        bf_func_no = np.asarray(mesh.boundary_face_function_numbers)
        n_bf = int(mesh.n_boundary_faces)
        n_faces = int(mesh.n_faces)
        bf_set = set(int(f) for f in bf_face_idx)
        interior_faces = np.array(
            [f for f in range(n_faces) if f not in bf_set], dtype=np.int32)

        iA_int = jnp.asarray(fc0[interior_faces])
        iB_int = jnp.asarray(fc1[interior_faces])
        iInner_bnd = jnp.asarray(fc0[bf_face_idx])
        interior_faces_j = jnp.asarray(interior_faces)
        boundary_faces_j = jnp.asarray(bf_face_idx)
        bf_func_no_j = jnp.asarray(bf_func_no, dtype=jnp.int32)

        face_centers_j = jnp.asarray(mesh.face_centers)
        face_normals_j = jnp.asarray(mesh.face_normals)
        face_volumes_j = jnp.asarray(mesh.face_volumes)
        cell_volumes_j = jnp.asarray(mesh.cell_volumes)
        cell_centers_j = jnp.asarray(mesh.cell_centers)
        boundary_face_cells_j = jnp.asarray(
            mesh.boundary_face_cells, dtype=jnp.int32)

        # Periodic-seam faces (REQ-116): after ``resolve_periodic_bcs``, a
        # periodic boundary face has ``boundary_face_cells`` remapped to the
        # PARTNER cell (≠ ``face_cells[0]``, the this-side cell).  The Periodic
        # BC kernel is a pass-through, so at these faces the ghost state Q_R is
        # the partner cell's state — NOT ``BC(Q_L)`` (which would silently
        # degrade the wrap to extrapolation).  Static mask ⇒ zero per-step cost
        # (and the whole branch is dropped at trace time when no periodic BC).
        periodic_bf_np = (np.asarray(mesh.boundary_face_cells) != fc0[bf_face_idx]
                          if n_bf > 0 else np.zeros(0, dtype=bool))
        has_periodic = bool(periodic_bf_np.any())
        periodic_mask_j = jnp.asarray(periodic_bf_np)

        # Distance from boundary face to inner cell (precomputed).
        d_face = np.asarray([
            np.linalg.norm(
                np.asarray(mesh.face_centers)[bf_face_idx[i], :]
                - np.asarray(mesh.cell_centers)[:, mesh.boundary_face_cells[i]])
            for i in range(n_bf)
        ]) if n_bf > 0 else np.zeros(0)
        d_face_j = jnp.asarray(d_face)

        @jax.jit
        @partial(jax.named_call, name="Flux")
        def flux_operator(dt, time, Q, Qaux, parameters, dQ, force_o1=None,
                          coupled_dm=None, coupled_mask=None):
            dQ = jnp.zeros_like(dQ)

            def _fc_proj(Qstate, qaux_face, normals_faces):
                # Conservative flux Fc = flux + hydrostatic_pressure at ``Qstate``,
                # projected on the face normal (Σ_d Fc_d·n_d).  Reuses the EMITTED
                # kernels; recomputes the LOCAL aux (``hinv`` = 1/h) from Qstate so
                # a reconstructed h* carries the consistent 1/h*.
                aux = (rt_uav(Qstate, qaux_face, parameters)
                       if rt_uav is not None else qaux_face)
                fc = (rt_flux(Qstate, aux, parameters)
                      + rt_press(Qstate, aux, parameters))    # (n_eq, n_dim, nf)
                return jnp.einsum("edf,df->ef", fc, normals_faces[:_bndim])

            # 1. Boundary face values via BC kernel (vmap over boundary faces).
            if rt_bc is not None and n_bf > 0:
                def _per_bf(i):
                    cell = boundary_face_cells_j[i]
                    fidx = boundary_faces_j[i]
                    return rt_bc(
                        bf_func_no_j[i], time,
                        face_centers_j[fidx, :], d_face_j[i],
                        Q[:, cell], Qaux[:, cell],
                        parameters, face_normals_j[:, fidx],
                    )
                # vmap over the boundary faces.  The generated BC kernel is a
                # sympy Piecewise over ``i_bc_func`` (lowers to a jnp.where
                # chain), which vectorises cleanly under vmap.  A Python loop
                # here instead UNROLLED n_bf copies of the whole BC kernel into
                # the JIT graph — quadratic compile + a slow per-step boundary
                # pass on meshes with a large perimeter (e.g. Malpasset).
                bf_values = jax.vmap(_per_bf)(jnp.arange(n_bf)).T
            else:
                bf_values = jnp.zeros((Q.shape[0], max(n_bf, 1)))

            # 2. Reconstruct with bf_values (LSQ-MUSCL sees boundary
            # face values for limiter bounds + Q_R override).
            o1kw = {"force_o1": force_o1} if support_o1 else {}
            if needs_emitted_w:
                # The EMITTED reconstruction map reads aux (``hinv``), so the
                # reconstruction needs Qaux + parameters, and the boundary face
                # states need aux CONSISTENT WITH THE GHOST STATE.  Recompute
                # it from ``bf_values`` via the model's own emitted local aux
                # formula — the same thing core's numpy solver does in
                # ``_ghost_aux``.  Reusing the inner cell's aux would pair a
                # ghost ``h`` with the interior's ``hinv``.
                if n_bf > 0 and rt_uav is not None:
                    bf_aux = rt_uav(bf_values, Qaux[:, iInner_bnd], parameters)
                else:
                    bf_aux = jnp.zeros((Qaux.shape[0], bf_values.shape[1]),
                                       dtype=bf_values.dtype)
                o1kw = dict(o1kw, Qaux=Qaux, parameters=parameters,
                            bf_aux=bf_aux)
            if use_interior_ncp and rt_ces is None:
                Q_L, Q_R, lim_grad = reconstruct.reconstruct_with_grad(
                    Q, bf_values, **o1kw)
                # LEGACY VOLUME FORM — used only by a runtime that predates the
                # EMITTED ``numerical_cell_edge_source``.  dQ_c −= Σ_d
                # B(Q_c)[:,:,d]·s_c[:,d]; no |cell| division, the volume factor
                # cancels against the per-unit-volume residual.
                #
                # It evaluates B at the CELL MEAN only.  That is exact for a B
                # LINEAR in the state (SWE's g·h — which is why Audusse's
                # centred term is a closed form) and an approximation for every
                # other one: hinv, q²/h, the moment couplings.  It also lived
                # ONLY here, so foam/amrex/firedrake had no interior term at
                # all.  Both reasons are why it moved into core as a shared
                # Gauss-Legendre path integral.
                B_all = rt_ncm(Q, Qaux, parameters)   # (n_eq, n_state, dim, nc)
                interior_ncp = jnp.einsum("ijdc,jdc->ic", B_all, lim_grad)
                dQ = dQ.at[:, :interior_ncp.shape[1]].subtract(interior_ncp)
            else:
                Q_L, Q_R = reconstruct(Q, bf_values, **o1kw)
            normals = face_normals_j
            face_volumes = face_volumes_j

            # 3. Symbolic Riemann (jit-vmap'd over face axis).
            qauxA = Qaux[:, iA_int]
            qauxB = Qaux[:, iB_int]
            normals_int = normals[:, interior_faces_j]
            fv_int = face_volumes[interior_faces_j]
            cvA_int = cell_volumes_j[iA_int]
            cvB_int = cell_volumes_j[iB_int]
            Q_L_int = Q_L[:, interior_faces_j]
            Q_R_int = Q_R[:, interior_faces_j]

            # ── FULLY well-balanced moving-equilibrium (Pimentel-García) ──
            # Per-cell local stationary solution U*(x) = U_i ± (Δx/2)K_i with
            # K_i = -(J_F+B)_VV^{-1}(S·H_x + R) (friction R baked into K), and
            # the minmod-limited slope of the FLUCTUATION V = U − U*.  Both from
            # the emitted operators + ``solve_steady_ode``.
            if bern_ode:
                Q_in = Q[:, :_ncinner]
                b_row = Q_in[_bb]
                hx_c = (b_row[_right_j] - b_row[_left_j]) / _dxc_j

                def _aux_of(U):
                    return (rt_uav_ode(U, Qaux[:, :_ncinner], parameters)
                            if rt_uav_ode is not None else Qaux[:, :_ncinner])

                def _quasi(U):
                    return rt_quasi(U, _aux_of(U), parameters)

                def _srcU(U):
                    return rt_src(U, _aux_of(U), parameters)

                _slope = make_steady_ode_slope(
                    _quasi, _srcU, _bb, _solve_op, jnp, _n_state, ndim=0)
                _K_c = _slope(Q_in, hx_c)
                Ustar_L_c, Ustar_R_c, sigma_c = moving_equilibrium_cells(
                    Q_in, hx_c, _cw_j, _bb, _slope, solve_steady_ode, jnp,
                    Q_left=Q_in[:, _left_j], Q_right=Q_in[:, _right_j],
                    order=int(self.nsm.reconstruction.order))
                sigma_c = sigma_c * _int_cell_j[None, :]   # O1 at boundary cells
                _half_c = 0.5 * _cw_j

            # 3c. Bernoulli moving-equilibrium WB (interior): reconstruct both
            # interface states to the common bed b*, run the Riemann + moment
            # NCP on the RECONSTRUCTED states (they coincide at equilibrium, so
            # the numerical flux + fluctuation vanish there), and add the
            # conservative-flux-jump source Fc(Q₀)−Fc(Q*).  The interior NCP
            # (step 3b) keeps the ORIGINAL LSQ face traces (bernoulli-
            # independent, like numpy's cell-centre grad form).
            QL_flux, QR_flux = Q_L_int, Q_R_int
            if bern_ode:
                # Riemann on the ACTUAL traces P = U* + fluctuation; the
                # conservative-flux correction subtracts the equilibrium flux
                # Fc(U*) (paper eq 28) so it vanishes at a discrete stationary
                # state (P = U*, and the two sides' U* coincide).
                UsL_int = Ustar_R_c[:, iA_int]           # iA right intercell eq
                UsR_int = Ustar_L_c[:, iB_int]           # iB left  intercell eq
                QL_flux = UsL_int + _half_c[iA_int] * sigma_c[:, iA_int]
                QR_flux = UsR_int - _half_c[iB_int] * sigma_c[:, iB_int]
                sL_int = -_fc_proj(UsL_int, qauxA, normals_int)
                sR_int = -_fc_proj(UsR_int, qauxB, normals_int)
                # Aux (hinv=1/h) MUST track the reconstructed face state, not the
                # cell mean: the Riemann flux hinv·q² would otherwise carry the
                # cell's 1/h against the extrapolated h — an O(Δx) desync that
                # ÷Δx leaves an O(1) momentum residual and breaks WB.
                qauxA_f = (rt_uav(QL_flux, qauxA, parameters)
                           if rt_uav is not None else qauxA)
                qauxB_f = (rt_uav(QR_flux, qauxB, parameters)
                           if rt_uav is not None else qauxB)
            elif bern_enabled:
                bstar_int = jnp.maximum(Q_L_int[_bb], Q_R_int[_bb])
                QL_flux = bern_recon(Q_L_int, bstar_int)
                QR_flux = bern_recon(Q_R_int, bstar_int)
                sL_int = (_fc_proj(Q_L_int, qauxA, normals_int)
                          - _fc_proj(QL_flux, qauxA, normals_int))
                sR_int = (_fc_proj(Q_R_int, qauxB, normals_int)
                          - _fc_proj(QR_flux, qauxB, normals_int))
                qauxA_f, qauxB_f = qauxA, qauxB
            else:
                qauxA_f, qauxB_f = qauxA, qauxB

            F_num_int = rt_num_flux(
                QL_flux, QR_flux, qauxA_f, qauxB_f, parameters, normals_int)
            fluct_int = rt_num_fluct(
                QL_flux, QR_flux, qauxA_f, qauxB_f, parameters, normals_int)
            Dp_int = fluct_int[0]
            Dm_int = fluct_int[1]

            dQ = dQ.at[:, iA_int].subtract(
                (F_num_int + Dm_int) * fv_int / cvA_int)
            dQ = dQ.at[:, iB_int].subtract(
                (-F_num_int + Dp_int) * fv_int / cvB_int)
            if bern_enabled or bern_ode:
                dQ = dQ.at[:, iA_int].subtract(sL_int * fv_int / cvA_int)
                dQ = dQ.at[:, iB_int].add(sR_int * fv_int / cvB_int)

            # 3b'. FULLY-WB in-cell nonconservative term on the FLUCTUATION slope
            # (paper eq 28): −B(U_i)·σ_i, zero at a discrete stationary state.
            # Replaces the standard reconstruction-slope NCP for the ODE path.
            if bern_ode and rt_ncm is not None:
                B_cell = rt_ncm(Q_in, Qaux[:, :_ncinner], parameters)
                ncp_c = jnp.einsum("ijdc,jdc->ic", B_cell, sigma_c[:, None, :])
                dQ = dQ.at[:, :_ncinner].subtract(ncp_c)

            # 3b. EMITTED interior nonconservative term, once per (cell, edge).
            # ``int_{T_ij} B(W)·grad W dx`` in surface form: with a linear
            # reconstruction grad W is constant on the subcell, so the integral
            # collapses to the segment path integral between the CELL MEAN and
            # the cell's own reconstructed EDGE value.  Each side uses its own
            # cell mean and its own edge trace, with the normal pointing OUT of
            # that cell — so B's normal is negated.  Vanishes identically at
            # order 1, where Q_L == Q[iA] and Q_R == Q[iB].
            if use_interior_ncp and rt_ces is not None and not bern_ode:
                ces_A = rt_ces(Q[:, iA_int], Q_L_int, qauxA, qauxA,
                               parameters, normals_int)
                ces_B = rt_ces(Q[:, iB_int], Q_R_int, qauxB, qauxB,
                               parameters, -normals_int)
                dQ = dQ.at[:, iA_int].subtract(ces_A * fv_int / cvA_int)
                dQ = dQ.at[:, iB_int].subtract(ces_B * fv_int / cvB_int)

            # 4. Boundary face loop — one inner cell only.
            if n_bf > 0:
                qauxBnd = Qaux[:, iInner_bnd]
                normals_bnd = normals[:, boundary_faces_j]
                fv_bnd = face_volumes[boundary_faces_j]
                cv_bnd = cell_volumes_j[iInner_bnd]
                Q_L_bnd = Q_L[:, boundary_faces_j]
                # Q_R at boundary = BC(Q_L_at_face) — recompute from the
                # reconstructed inner state to keep the limiter and BC
                # consistent at face center (mirrors NumPy override).
                def _per_bf_R(i):
                    cell = boundary_face_cells_j[i]
                    fidx = boundary_faces_j[i]
                    return rt_bc(
                        bf_func_no_j[i], time,
                        face_centers_j[fidx, :], d_face_j[i],
                        Q_L_bnd[:, i], Qaux[:, cell],
                        parameters, face_normals_j[:, fidx],
                    )
                Q_R_bnd_list = [_per_bf_R(i) for i in range(n_bf)]
                Q_R_bnd = jnp.stack(Q_R_bnd_list, axis=-1)

                # Periodic seam (REQ-116): Q_R is the opposite-side (partner)
                # cell state, already remapped into ``boundary_face_cells``;
                # the BC-kernel override above (fed Q_L) would turn the wrap
                # back into extrapolation.  Mirrors numpy's periodic_bf branch.
                # ``has_periodic`` is a static python bool → this whole block is
                # dropped at trace time for non-periodic meshes.
                if has_periodic:
                    Q_partner = Q[:, boundary_face_cells_j]
                    Q_R_bnd = jnp.where(
                        periodic_mask_j[None, :], Q_partner, Q_R_bnd)

                # 4c. Bernoulli moving-equilibrium WB (boundary): reconstruct the
                # inner + ghost traces to b*, run the Riemann on them, and add
                # the inner-side conservative-flux-jump source (only the inner
                # cell exists to receive it — mirrors numpy step 4b/WB).
                QL_bnd_flux, QR_bnd_flux = Q_L_bnd, Q_R_bnd
                qauxBnd_fL = qauxBnd_fR = qauxBnd
                if bern_ode:
                    # Inner cell's equilibrium trace toward its boundary face
                    # (sign = outward-normal x-component): U* + fluctuation vs the
                    # BC ghost; the flux correction subtracts Fc(U*_bnd).
                    _s_bnd = jnp.sign(normals_bnd[0])              # (n_bf,)
                    _UsR_b = Ustar_R_c[:, iInner_bnd]
                    _UsL_b = Ustar_L_c[:, iInner_bnd]
                    Us_bnd = jnp.where(_s_bnd[None, :] > 0, _UsR_b, _UsL_b)
                    _hf = (_s_bnd * _half_c[iInner_bnd])[None, :]
                    QL_bnd_flux = Us_bnd + _hf * sigma_c[:, iInner_bnd]
                    # Rebuild the ghost from the EQUILIBRIUM trace, not the raw
                    # inner mean: the BC's extrapolated (non-prescribed) slots
                    # must carry the equilibrium so the ghost coincides with the
                    # inner U* at a steady state — otherwise the extrapolated h
                    # sits an O(Δx) step off U*_bnd and leaves an O(1) boundary
                    # residual that swamps the interior WB.
                    def _bf_ghost_ode(i):
                        cell = boundary_face_cells_j[i]
                        fidx = boundary_faces_j[i]
                        return rt_bc(
                            bf_func_no_j[i], time,
                            face_centers_j[fidx, :], d_face_j[i],
                            Us_bnd[:, i], Qaux[:, cell],
                            parameters, face_normals_j[:, fidx])
                    QR_bnd_flux = jnp.stack(
                        [_bf_ghost_ode(i) for i in range(n_bf)], axis=-1)
                    if has_periodic:
                        QR_bnd_flux = jnp.where(
                            periodic_mask_j[None, :],
                            Q[:, boundary_face_cells_j], QR_bnd_flux)
                    sL_bnd = -_fc_proj(Us_bnd, qauxBnd, normals_bnd)
                    qauxBnd_fL = (rt_uav(QL_bnd_flux, qauxBnd, parameters)
                                  if rt_uav is not None else qauxBnd)
                    qauxBnd_fR = (rt_uav(QR_bnd_flux, qauxBnd, parameters)
                                  if rt_uav is not None else qauxBnd)
                elif bern_enabled:
                    bstar_bnd = jnp.maximum(Q_L_bnd[_bb], Q_R_bnd[_bb])
                    QL_bnd_flux = bern_recon(Q_L_bnd, bstar_bnd)
                    QR_bnd_flux = bern_recon(Q_R_bnd, bstar_bnd)
                    sL_bnd = (_fc_proj(Q_L_bnd, qauxBnd, normals_bnd)
                              - _fc_proj(QL_bnd_flux, qauxBnd, normals_bnd))

                F_num_bnd = rt_num_flux(
                    QL_bnd_flux, QR_bnd_flux, qauxBnd_fL, qauxBnd_fR, parameters,
                    normals_bnd)
                fluct_bnd = rt_num_fluct(
                    QL_bnd_flux, QR_bnd_flux, qauxBnd_fL, qauxBnd_fR, parameters,
                    normals_bnd)
                Dm_bnd = fluct_bnd[1]
                # Coupled-BC flux injection (conservative characteristic
                # coupling, Dahm-Giesselmann-Helzel / PreciceManager.H
                # applyCoupledFaceFlux).  At coupled boundary faces the face
                # contribution is the SHARED Riemann answer computed in the
                # agreed high-fidelity model space and passed in per window
                # (``coupled_dm``, aligned to the boundary-face axis).
                # ``coupled_mask`` selects the coupled faces; normal faces keep
                # the ghost flux.  Both static-None by default -> the branch is
                # dropped at trace time and behaviour is unchanged.
                #
                # PRODUCED by ``interlevel_face_contribution`` (module top).
                # This comment USED TO SAY "both participants build the
                # identical pair with their own outward normal, so the two
                # contributions telescope".  That is exactly the thing NOT to
                # do: ``nHat`` is normalised as ``n/(|n| + SMALL)``, so an
                # x-normal face carries ``nHat.x() = 1 - 2.2e-16`` and the two
                # sides' algebraically-equal expressions differ in the last bit
                # -- measured to break bit-equality on ~62% of states
                # (``swePreciceCoupling.C``).  The producer instead evaluates
                # the face ONCE at the canonical low->high orientation and
                # hands each side a sign projection of that ONE answer
                # (``F + D_minus`` low, ``-F + D_plus`` high).
                face_contrib_bnd = F_num_bnd + Dm_bnd
                if coupled_dm is not None:
                    face_contrib_bnd = jnp.where(
                        coupled_mask[None, :], coupled_dm, face_contrib_bnd)
                dQ = dQ.at[:, iInner_bnd].subtract(
                    face_contrib_bnd * fv_bnd / cv_bnd)
                if bern_enabled or bern_ode:
                    dQ = dQ.at[:, iInner_bnd].subtract(sL_bnd * fv_bnd / cv_bnd)

                # 4b. EMITTED interior nonconservative term at the BOUNDARY
                # face.  A boundary cell's in-cell integral has TWO halves: the
                # interior-face half (added in step 3b) and this boundary-face
                # half.  Omitting it left boundary cells with an incomplete
                # nonconservative integral — measured directly as an O(1)
                # residual error in the boundary strips while the INTERIOR
                # matched the conservative operator to O(h^2).  Over a march
                # that boundary error propagates inward and caps a smooth
                # quasilinear run at ~1.85 instead of 2.  The pair is (boundary
                # cell mean, that cell's reconstructed boundary-face value) with
                # the OUTWARD normal — same shape as the interior side.
                if use_interior_ncp and rt_ces is not None and not bern_ode:
                    ces_bnd = rt_ces(Q[:, iInner_bnd], Q_L_bnd,
                                     qauxBnd, qauxBnd, parameters, normals_bnd)
                    dQ = dQ.at[:, iInner_bnd].subtract(
                        ces_bnd * fv_bnd / cv_bnd)

            # 5. Explicit diffusion (REQ-50): dQ += ∇·(A:∇Q).  A evaluated at
            # the current state (explicit); the per-volume divergence matches
            # the flux residual's ``/cv`` convention and the contract's
            # ``+∂_x(A·∂_x Q)`` sign.  Interior faces only (bf_grads=None).
            if dense_diff_op is not None:
                A_cells = dense_diff_op._as_cell_tensor(
                    rt_diff_fn(Q, Qaux, parameters))
                dQ = dQ.at[:, :dense_diff_op.nc].add(
                    dense_diff_op._divergence(Q, A_cells, bf_grads=None))

            return dQ

        return flux_operator

    # ── setup_simulation / step / run_simulation ────────────────────────

    def setup_simulation(self, mesh, model):
        """Build all JAX operators from mesh and model.

        ``model`` may be a :class:`Model`, a :class:`SystemModel`, or a
        :class:`NumericalSystemModel`.  All numerical knobs
        (``reconstruction.order``, ``reconstruction.limiter``,
        ``regularization.eigenvalue_eps``) live on the NSM; the mesh
        stencil uses ``nsm.resolved_lsq_degree()``.

        Until the JAX runtime is wired through
        ``JaxRuntimeModel.from_system_model`` (S4b), callers passing a
        bare SystemModel or an NSM built from one must do so with a
        source Model — JAX's :class:`Kernel` + Model-based
        ``JaxRuntimeModel`` paths still need it.

        Returns
        -------
        Q, Qaux : jnp.ndarray
            Initial state arrays on device.
        """
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        # REQ-190: fill the adaptive strategy's ``dt_max`` cap from the NSM's
        # standard ``dt_max`` unless the caller passed an explicit cap (explicit
        # wins).  jax shares core's ``timestepping.adaptive``, so a wave-free
        # (fully-dry) domain — every gated ``|λ| = 0`` → local CFL limits ``+inf``
        # — then steps at ``dt_max`` instead of leaking ``inf``, identically to
        # the numpy solver.  No-op for strategies without the hook (constant dt).
        timestepping.apply_default_dt_max(self.compute_dt, nsm.dt_max)
        mesh = ensure_lsq_mesh(mesh, nsm)
        # Periodic BCs (REQ-116): remap ``boundary_face_cells`` at the periodic
        # seam to the partner cell across the wrap, exactly as numpy does
        # (solver_numpy.py `setup_simulation`).  Must run on the NumPy mesh
        # BEFORE ``create_runtime``/``convert_mesh_to_jax`` (which copies the
        # patched array) — resolving on the already-converted jax mesh is too
        # late.  Without it the jax FVM applied an extrapolation/wall flux at
        # the seam (periodic roll waves decayed instead of growing).
        bcs_obj = (source_model.boundary_conditions if source_model is not None
                   else getattr(nsm.sm, "_bc_source", None))
        if hasattr(mesh, "resolve_periodic_bcs") and bcs_obj is not None:
            mesh.resolve_periodic_bcs(bcs_obj)
        # `initialize` only needs the SystemModel's shape info + the
        # initial_conditions if the SM carries them.  Source Model is
        # no longer required — JaxRuntime is the runtime.
        # Size (Q, Qaux) from the NSM's SystemModel (aux like ``hinv`` is
        # promoted at the NSM level; a raw source Model reports 0 aux and yields
        # a (0, nc) Qaux that mismatches the NSM-built operators — cid=101).
        # ``nsm.sm`` carries the ICs, so Q is filled correctly; for a pre-built
        # NSM caller source_model is None already, so this is a no-op there.
        Q, Qaux = self.initialize(mesh, nsm.sm)
        Q, Qaux, parameters, jax_mesh, runtime_model = self.create_runtime(
            Q, Qaux, mesh, nsm.sm,
        )

        # Store runtime state for step / run_simulation
        self._rt_mesh = jax_mesh
        # Keep the source LSQMesh for HDF5 output — MeshJAX has no
        # write_to_hdf5; the original NumPy LSQMesh is the only thing
        # that knows how to serialise itself (mesh + LSQ stencils).
        self._rt_mesh_np = mesh
        self._rt_model = runtime_model
        self._rt_parameters = parameters

        # Build operators (closures over jax_mesh / runtime_model)
        self._rt_flux_op = self.get_flux_operator(jax_mesh, runtime_model)
        self._rt_source_op = self.get_compute_source(jax_mesh, runtime_model)
        self._rt_bc_op = self.get_apply_boundary_conditions(jax_mesh, runtime_model)
        self._rt_eigenvalue_op = self.get_compute_max_abs_eigenvalue(
            jax_mesh, runtime_model
        )

        # Precompute PER-FACE cell size for the LOCAL CFL: couple each face's
        # wave speed (per-face) to the smaller of its two adjacent cells'
        # inradii; compute_dt takes min over faces -> min_f(h/λ), NOT the
        # over-conservative global_min(inradius)/global_max(λ).
        _ir = jax_mesh.cell_inradius
        self._rt_inradius_face = jnp.minimum(
            _ir[jax_mesh.face_cells[0]], _ir[jax_mesh.face_cells[1]])

        # Initial aux update
        Qaux = self.update_qaux(
            Q, Qaux, Q, Qaux, jax_mesh, runtime_model, parameters, 0.0, 1.0
        )

        # Apply initial BCs
        Q = self._rt_bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)

        # Move to device
        Q = jax.device_put(Q)
        Qaux = jax.device_put(Qaux)
        self._rt_mesh = jax.device_put(jax_mesh)

        return Q, Qaux

    def _explicit_hyperbolic_step(self, dt, time, Q, Qaux, parameters, flux, nc,
                                  *, time_euler=False, aux_of_q=None,
                                  source=None):
        """The explicit hyperbolic (flux) sub-step, MOOD-aware.  Factored out of
        ``step`` so the IMEX solver's explicit stage reuses the SAME a-priori
        front pre-detector (which rides inside the reconstruction) AND
        a-posteriori MOOD corrector (the ``lax.cond`` re-run here).  ``flux`` is
        the ``(dt, time, Q, Qaux, p, dQ, force_o1)`` operator; ``nc`` = #inner
        cells (for the troubled-cell mask).  order≥2 = SSP-RK2 (Heun); order 1 =
        explicit Euler.

        ``source`` (optional, ``(dt, time, Q, Qaux, p, dQ)`` operator): when
        given, its contribution is folded into the RK RHS at EACH stage
        (mirroring ``solver_numpy`` ``rhs = flux + source``) so a source-carrying
        row reaches the flux's design order instead of the 1st-order Godunov
        split a separate explicit-Euler source sub-step would impose (cid 151).
        The base explicit ``step`` passes the source here; the IMEX loop leaves
        it ``None`` and steps the source IMPLICITLY afterwards; the Chorin-VAM
        predictor leaves it ``None`` and adds the source in its own splitting
        cycle (Escalante eq. 11).  ``None`` ⇒ flux-only, byte-identical to
        before.  The point-implicit ``source_treatment="linearized"`` update is
        baked INTO the source operator itself (NSM rewrites ``sm.source`` into
        ``(I − dt·J)⁻¹ S``), so folding it here linearizes inside each RK stage —
        no second split.

        ``time_euler=True`` keeps the order-≥2 SPATIAL reconstruction (and the
        whole MOOD apparatus) but advances with a SINGLE Euler step in time.
        That is the shape a projection/splitting scheme needs: the outer RK
        stages live in the *splitting* driver, and each stage's hyperbolic
        substep must be one Euler step or the split operators are applied at
        the wrong nesting (Escalante et al. 2024, JCP 504:112882, eq. 11 — the
        per-stage hyperbolic update is Euler, second order coming from the
        reconstruction).  Default ``False`` ⇒ every pre-existing caller keeps
        the RK2-inside-the-step behaviour, bit-identically.

        SPMD: the flux is halo-wrapped here (once), so EVERY solver that reuses
        this sub-step — the base explicit ``step`` AND the IMEX explicit stage —
        gets across-partition halo exchange before each flux evaluation with no
        extra wiring.  Identity when not sharded (``_halo_exchange`` unset).
        """
        flux = self._halo_wrap(flux)
        if self.nsm.reconstruction.order >= 2:
            # SSP-RK2 (Heun).  ``force_o1`` (or None) is threaded through both
            # stages so the a-posteriori MOOD re-run demotes the SAME troubled
            # cells in both — the a-priori front pre-detector lives inside the
            # reconstruction and applies on every call regardless.
            def _rk2(force_o1):
                Q0 = Q
                dQ = flux(dt, time, Q0, Qaux, parameters,
                          jnp.zeros_like(Q), force_o1)
                if source is not None:
                    dQ = dQ + source(dt, time, Q0, Qaux, parameters,
                                     jnp.zeros_like(Q))
                Q1 = Q0 + dt * dQ
                # REFRESH THE AUX AGAINST THE STAGE STATE.  Qaux is a FUNCTION
                # of Q (SME/VAM carry the desingularized ``hinv`` = f(h)), so
                # feeding stage 2 the aux computed from Q0 evaluates the flux
                # at an inconsistent state: an O(dt) error that drags the whole
                # scheme back to FIRST order in time no matter how good the
                # spatial reconstruction is.
                #
                # Measured on smooth flat-bed SME(1) before this fix: with the
                # diffusive minmod limiter the spatial error masked it
                # (h 1.73 / q_0 1.78), but with venkatakrishnan -- which is
                # 2nd-order at extrema, so the spatial error shrinks -- every
                # row collapsed toward 1 (h 1.12 / q_0 1.16 / q_1 0.97). A
                # "better limiter makes convergence worse" inversion is the
                # signature of a time-integration defect, not a spatial one.
                Qaux1 = (aux_of_q(Q1, Qaux, parameters)
                         if aux_of_q is not None else Qaux)
                dQ = flux(dt, time + dt, Q1, Qaux1, parameters,
                          jnp.zeros_like(Q), force_o1)
                if source is not None:
                    # Stage 2 uses the REFRESHED aux (consistent with the flux)
                    # and the stage-2 time level ``time + dt`` — a
                    # time-dependent source samples both levels like numpy's RK2.
                    dQ = dQ + source(dt, time + dt, Q1, Qaux1, parameters,
                                     jnp.zeros_like(Q))
                Q2 = Q1 + dt * dQ
                return 0.5 * (Q0 + Q2)

            def _euler(force_o1):
                dQ = flux(dt, time, Q, Qaux, parameters,
                          jnp.zeros_like(Q), force_o1)
                if source is not None:
                    dQ = dQ + source(dt, time, Q, Qaux, parameters,
                                     jnp.zeros_like(Q))
                return Q + dt * dQ

            _advance = _euler if time_euler else _rk2

            if getattr(self, "_mood_active", False):
                # ── A-posteriori MOOD — THE DEFAULT order-2 positivity net ──
                # (user directive 2026-07-27; ``mood`` defaults True, engages on
                # the eta path only — ``_mood_active`` set in get_flux_operator.)
                #
                #   1. O2 candidate (the SSP-RK2 above).
                #   2. Flag inadmissible cells: PAD h<0 + CAD non-finite.  (No
                #      relaxed-DMP — a DMP bound is invalid in the presence of a
                #      source, and the physical admissibility set is exactly h≥0
                #      + finite.)
                #   3. Recompute the FLAGGED cells at O1 Audusse-HR: ``force_o1``
                #      zeroes their reconstruction slope → the piecewise-constant
                #      emitted-W reconstruction (Audusse η→h clip + emitted
                #      inverse q=u·h), positivity-preserving by the XZS 1st-order
                #      lemma and conservative through the shared face flux.  This
                #      is the whole-CELL order drop — O2 is kept in the wet
                #      interior (matters for Malpasset/Thacker), only front cells
                #      fall to O1.
                #   4. SAFETY NET (amrex whole-step O1 MOOD): if the whole-cell
                #      redo STILL leaves an inadmissible cell — a capless draining
                #      front where an O2 wet neighbour fluxes a denormal −h
                #      (≈ −1e-142) into the demoted cell, which the NEXT step's
                #      unguarded sqrt(g·h) eigenvalue would turn into a
                #      domain-wide NaN (the cid-155 Ritter-O2 failure) — escalate
                #      to a whole-STEP O1-HR redo (force_o1 everywhere).  That
                #      matches the pure-O1 march (h ≥ 0 exactly) and is provably
                #      admissible under the CFL.  No dt reduction anywhere.
                #
                # Smooth everywhere-wet cases never flag ⇒ ``jnp.any`` is False ⇒
                # the pure-O2 candidate is returned, bit-identical: MOOD is a
                # no-op there.  Only the taken lax.cond branch executes (scalar
                # predicate), so the safety net costs nothing off the front.
                h_idx = int(self.free_surface_h_index)

                def _troubled(Qc):
                    return ((Qc[h_idx, :nc] < 0.0)
                            | (~jnp.isfinite(Qc[:, :nc]).all(axis=0)))

                Q_cand = _advance(None)

                def _repair():
                    Q_wc = _advance(_troubled(Q_cand))          # whole-cell O1-HR
                    return jax.lax.cond(
                        jnp.any(_troubled(Q_wc)),
                        lambda: _advance(jnp.ones((nc,), dtype=bool)),  # whole-step
                        lambda: Q_wc)

                return jax.lax.cond(
                    jnp.any(_troubled(Q_cand)), _repair, lambda: Q_cand)
            return _advance(None)
        dQ = flux(dt, time, Q, Qaux, parameters, jnp.zeros_like(Q))
        if source is not None:
            dQ = dQ + source(dt, time, Q, Qaux, parameters, jnp.zeros_like(Q))
        return Q + dt * dQ

    def _halo_wrap(self, flux):
        """SPMD hook.  When ``self._halo_exchange`` is set (the solver is being
        run inside a ``jax.shard_map`` over a partitioned mesh), refresh the
        halo slabs of ``Q`` **and** ``Qaux`` before *every* flux evaluation —
        so the reconstruction + Riemann across the partition boundary see the
        neighbor cells (``ppermute``).  Because the wrap sits on the flux op
        itself, every stage (order-1 Euler, both SSP-RK2 stages, a MOOD re-run)
        gets fresh halos with no change to the RK logic.  ``None`` (single
        device) ⇒ returns ``flux`` unchanged: identity, zero cost, no behaviour
        change on the non-sharded path."""
        he = getattr(self, "_halo_exchange", None)
        if he is None:
            return flux

        def wrapped(dt, time, Q, Qaux, parameters, dQ, *rest):
            return flux(dt, time, he(Q), he(Qaux), parameters, dQ, *rest)

        return wrapped

    def step(self, dt, time, Q, Qaux, parameters=None, coupled=None):
        """Perform a single explicit time step.

        ``coupled`` (optional) = ``(coupled_dm, coupled_mask)`` for
        conservative characteristic interface coupling: ``coupled_dm`` is the
        shared high-fidelity Riemann face flux ``F + D_minus`` (state rows x
        boundary faces) and ``coupled_mask`` (boundary faces) flags the coupled
        faces.  Threaded into the jitted flux operator so a coupled boundary
        face uses this RP flux instead of its ghost flux.  ``None`` (all
        existing callers) => uncoupled, bit-identical to before.

        Each RK stage starts by refreshing the boundary ghost cells from
        the *current* interior state — mirroring the numpy solver, which
        evaluates the indexed BC kernel inside its flux operator. Without
        per-stage BCs the second RK stage reads stale ghosts from the
        previous outer step, contaminating a few cells next to each
        boundary and stripping the global L2 of its 2nd-order rate even
        though the interior is fine (interior errs ≈ O(dx²), boundary
        errs ≈ O(dx)).
        """
        # ``parameters=None`` → the baked runtime params (all existing
        # callers); passing a traced array makes the step differentiable
        # w.r.t. physical parameters (forward-mode jvp/jacfwd).
        parameters = self._rt_parameters if parameters is None else parameters
        flux = self._rt_flux_op
        if coupled is not None:
            # Bind the per-window coupled-face flux into the flux operator so the
            # RK/MOOD driver's fixed ``flux(dt,time,Q,Qaux,p,dQ[,force_o1])``
            # call signature is preserved (the coupled data rides as kwargs).
            _coupled_dm, _coupled_mask = coupled
            _flux_base = flux

            def flux(*a, **k):
                return _flux_base(*a, coupled_dm=_coupled_dm,
                                  coupled_mask=_coupled_mask, **k)

        # BC is now applied INSIDE flux_operator at the right time
        # (just before reconstruction), so step() no longer pre-fills
        # ghost cells.  Mirrors the NumPy solver — one BC kernel
        # invocation per face per stage, no stale-ghost reads.
        #
        # The explicit hyperbolic (flux) sub-step, MOOD-aware — factored into
        # ``_explicit_hyperbolic_step`` so the IMEX solver reuses the SAME
        # front pre-detector + MOOD corrector for its explicit stage.  The
        # source is FOLDED into that step's RK RHS at the flux's order
        # (``source=`` below), mirroring ``solver_numpy`` ``rhs = flux + source``
        # — NOT stepped separately.  The old separate ``ode.RK1(source)`` was a
        # 1st-order Godunov split: it injected an O(dt)=O(dx) error into every
        # source-carrying row, capping MMS/steady friction/moment-source cases
        # at 1st order even with a 2nd-order flux (cid 151).
        #
        # REQ-185: ``time`` is threaded into the step so a time-dependent source
        # binds ``t`` (dt + cell positions are added by ``get_compute_source``);
        # folding samples the source at both RK time levels like numpy.
        #
        # The FULLY well-balanced ODE moving-equilibrium path carries the source
        # (friction) INSIDE the equilibrium reconstruction U* (paper eq 28,
        # midpoint quadrature → no additive source term), so no source is folded
        # there to avoid double-counting.
        source_op = (None if getattr(self, "_bern_ode", False)
                     else self._rt_source_op)
        Q = self._explicit_hyperbolic_step(
            dt, time, Q, Qaux, parameters, flux,
            int(self._rt_mesh.n_inner_cells),
            source=source_op,
            aux_of_q=getattr(self._rt_model, "update_aux_variables", None))

        return Q

    def post_step(self, time, dt, Q, Qold, Qaux, parameters=None):
        """Post-step processing: BCs, update_q, update_qaux.

        Separated from ``step`` so that subclasses (e.g. IMEX) can insert
        implicit solves between the explicit step and the post-processing.

        Parameters
        ----------
        time : scalar
            Current simulation time (after dt advance).
        dt : scalar
            Time step size.
        Q : jnp.ndarray
            State after explicit step.
        Qold : jnp.ndarray
            State before the step (for aux updates).
        Qaux : jnp.ndarray
            Auxiliary state before the step.

        Returns
        -------
        Q_new, Qaux_new : jnp.ndarray
            Fully updated state and auxiliary arrays.
        """
        mesh = self._rt_mesh
        model = self._rt_model
        parameters = self._rt_parameters if parameters is None else parameters

        # No separate post-step BC application — the BC kernel runs
        # inside flux_operator at the right moment of each RK stage.

        # Update variables (clamp, etc.)
        Q = self.update_q(Q, Qaux, mesh, model, parameters)

        # Update auxiliary variables
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qaux, mesh, model, parameters, time, dt
        )

        return Q, Qaux

    # ── per-step building block + coupling hooks ────────────────────────
    # These make the time loop composable: the uncoupled solver and a
    # coupled (preCICE) subclass share the SAME physics per step
    # (``_advance``) and differ only in the loop wrapper + the two hooks.
    def _advance(self, time, Q, Qaux, *, time_end, parameters=None):
        """One time step: adaptive dt → explicit RK ``step`` → ``post_step``,
        with coupling hooks interleaved.

        The reusable per-step building block.  Uncoupled, the hooks are
        no-ops/identity so this is the plain physics step; a coupled
        (preCICE) subclass fills the hooks and reuses everything else.
        Returns ``(time_new, dt, Q, Qaux)``.
        """
        Qold = Q
        dt = self.compute_timestep(Q, Qaux, parameters)
        dt = jnp.minimum(dt, time_end - time)
        dt = self._couple_dt(dt)                          # cap by partner dt
        Q = self._couple_pre_step(time, dt, Q, Qaux)      # read partner → BC
        # The shared interface face contribution for THIS window, if a coupled
        # subclass produced one (see ``interlevel_face_contribution``).  None
        # uncoupled -> ``step`` takes its existing ``coupled=None`` path and
        # the flux operator drops the branch at trace time: bit-identical.
        coupled = self._couple_face_contribution(time, dt, Q, Qaux)
        Q_stepped = self.step(dt, time, Q, Qaux, parameters, coupled=coupled)
        Q, Qaux = self.post_step(
            time + dt, dt, Q_stepped, Qold, Qaux, parameters)
        self._couple_post_step(time + dt, dt, Q, Qaux)    # write partner + advance
        return time + dt, dt, Q, Qaux

    # ── coupling hooks (no-op/identity uncoupled; preCICE overrides) ────
    def _couple_setup(self):
        """Bootstrap (preCICE participant + mesh + handshake) BEFORE the loop."""
        return None

    def _couple_dt(self, dt):
        """Cap dt by the partner/coupling timestep.  Identity uncoupled."""
        return dt

    def _couple_pre_step(self, time, dt, Q, Qaux):
        """Read partner data and inject it into the boundary state BEFORE the
        explicit step.  Returns Q unchanged uncoupled."""
        return Q

    def _couple_face_contribution(self, time, dt, Q, Qaux):
        """The ``(coupled_dm, coupled_mask)`` pair for THIS window, or ``None``.

        A coupled subclass builds it with :func:`interlevel_face_contribution`
        — ONE evaluation of the shared inter-level face at the canonical
        low→high orientation, projected to this participant's side — and
        returns ``(coupled_dm, coupled_mask)`` with shapes
        ``(rows_of_this_participant, n_boundary_faces)`` and
        ``(n_boundary_faces,)``.

        ``None`` uncoupled, which is what every existing caller gets: ``step``
        keeps its ``coupled=None`` path and ``flux_operator`` drops the whole
        injection branch at trace time, so an uncoupled march is bit-identical
        to before this hook existed."""
        return None

    def _couple_post_step(self, time, dt, Q, Qaux):
        """Write this step's result to the partner + advance the coupling.
        No-op uncoupled."""
        return None

    def _couple_proceed(self, time, time_end):
        """Loop-continue predicate.  ``time < time_end`` uncoupled; a coupled
        subclass returns 'coupling ongoing'."""
        return time < time_end

    def compute_timestep(self, Q, Qaux, parameters=None):
        """Compute the adaptive time step using the stored eigenvalue operator.

        JIT-compatible. Uses ``self.compute_dt`` (from param) with the
        precomputed eigenvalue operator and min inradius.

        Returns
        -------
        dt : scalar
        """
        return self.compute_dt(
            Q, Qaux,
            (self._rt_parameters if parameters is None else parameters),
            self._rt_inradius_face, self._rt_eigenvalue_op,
        )

    def run_simulation(self, Q, Qaux, write_output=True, parameters=None):
        """JIT-compiled time loop using jax.lax.while_loop.

        Calls ``compute_timestep`` -> ``step`` -> ``post_step`` in a
        while_loop until ``time >= time_end``.

        Parameters
        ----------
        Q, Qaux : jnp.ndarray
            Initial state (from ``setup_simulation``).
        write_output : bool
            Whether to write snapshots to HDF5.
        parameters : jnp.ndarray, optional
            Physical-parameter vector.  ``None`` → the baked
            ``self._rt_parameters`` (all existing callers).  Pass a traced
            array to make the whole run differentiable w.r.t. parameters
            (``jax.jvp``/``jacfwd``; forward mode works through the
            ``while_loop``).

        Returns
        -------
        Q, Qaux : jnp.ndarray
            Final state.
        """
        parameters = (self._rt_parameters if parameters is None
                      else parameters)
        mesh = self._rt_mesh

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            # MeshJAX has no write_to_hdf5; use the source NumPy
            # LSQMesh stashed in setup_simulation.  init_output_directory
            # prepends main_dir; write_to_hdf5 expects an absolute path
            # (h5py.File doesn't auto-resolve relative-to-main_dir).
            _abs_h5 = os.path.join(_misc.get_main_directory(), output_hdf5_path)
            self._rt_mesh_np.write_to_hdf5(_abs_h5)
            _write_field_names(_abs_h5, self.nsm.state)   # /fields@names
            io.save_settings(self.settings)
            save_fields = jax_io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
                """Save field."""
                return i_snapshot

        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        i_snapshot = save_fields(0.0, 0.0, 0.0, Q, Qaux)
        last_written = 0.0
        time_end = self.time_end

        # Coupling bootstrap hook — no-op for an uncoupled run; a preCICE
        # subclass overrides it (participant init + mesh + handshake).
        self._couple_setup()

        # Capture the per-step building block + loop predicate for the closure.
        _advance = self._advance
        _couple_proceed = self._couple_proceed
        # Progress-print cadence, resolved HERE (python, pre-trace) so an
        # accelerator march stages no host callback at all.  None => auto: the
        # trace is worth its cost on CPU and is 9.4x on a GPU, so default it off
        # wherever the backend is not the CPU.
        _log_every = self.march_log_interval
        if _log_every is None:
            _log_every = 10 if jax.default_backend() == "cpu" else 0
        _log_every = int(_log_every)

        # ── compile the march ONCE ───────────────────────────────────────
        # ``time_loop`` used to be defined (and ``jax.jit``-wrapped) fresh on
        # every ``run_simulation`` call.  jit caches on the FUNCTION OBJECT, so
        # a new closure per call meant a full XLA recompilation of the whole
        # march every call -- invisible to a single-shot solve, brutal to any
        # driver that calls run_simulation repeatedly (frame/window producers,
        # restart chains, coupled sub-cycling).  MEASURED with JAX_LOG_COMPILES
        # on Malpasset 26k: a 4-window run logged FOUR ``Compiling
        # jit(time_loop)`` events at ~70 s each, and the same 400 s of
        # simulation cost 400 s wall at 1 window against 968 s at 8 (2.4x, i.e.
        # ~81 s per extra window -- all of it recompilation).
        #
        # Cache the compiled loop on the instance, keyed by everything the
        # trace bakes in as a Python constant.  Anything that changes the
        # generated code is in the key, so a genuine change still recompiles;
        # repeat calls with the same configuration reuse the executable.
        # ``parameters`` / Q / Qaux stay TRACED arguments, so adaptive dt --
        # recomputed from the state every step inside the loop -- is unaffected.
        # NB: do NOT key on ``id()`` of ``save_fields`` / ``_advance`` /
        # ``_couple_proceed``.  All three are rebuilt per call -- ``self._advance``
        # materialises a FRESH bound method on every attribute access -- so an
        # id-based key never hits and the cache silently does nothing (measured:
        # still 4 compiles for 4 windows).  Key on the values that actually
        # change the emitted code instead; the callables are functionally fixed
        # for a given solver + output configuration.
        _cache_key = (float(time_end), float(dt_snapshot), int(_log_every),
                      bool(write_output))
        _cached = getattr(self, "_time_loop_cache", None)
        if _cached is not None and _cached[0] == _cache_key:
            return_cached = _cached[1]
        else:
            return_cached = None

        @jax.jit
        @partial(jax.named_call, name="time_loop")
        def time_loop(time, iteration, i_snapshot, last_written, Q, Qaux,
                      parameters):
            """JIT-compiled time loop."""
            loop_val = (time, iteration, i_snapshot, last_written, Q, Qaux)

            @partial(jax.named_call, name="time_step")
            def loop_body(init_value):
                """Single iteration of the time loop."""
                time, iteration, i_snapshot, last_written, Qnew, Qauxnew = init_value

                # One step of physics (dt → step → post_step), the shared
                # building block.
                time_new, dt, Q_final, Qaux_final = _advance(
                    time, Qnew, Qauxnew, time_end=time_end,
                    parameters=parameters,
                )

                iteration_new = iteration + 1
                time_stamp = i_snapshot * dt_snapshot

                i_snapshot_new = save_fields(
                    time_new, time_stamp, i_snapshot, Q_final, Qaux_final
                )
                last_written_new = jnp.where(
                    i_snapshot_new > i_snapshot, time_new, last_written)

                # jax.debug.print is transparent under autodiff; the previous
                # jax.experimental.io_callback has NO JVP rule and blocked
                # jax.jvp / jax.grad through the whole time loop (forward-mode
                # sensitivity, segment-wise per write).
                #
                # It is still a HOST CALLBACK inside the jitted while_loop, so
                # every firing round-trips to the host and serialises the march
                # -- 9.4x on a GPU (see ``march_log_interval``).  ``_log_every``
                # is a PYTHON int resolved before tracing: 0 => the callback is
                # never staged into the jaxpr, not merely predicated off.
                if _log_every:
                    jax.lax.cond(
                        (iteration_new % _log_every) == 0,
                        lambda: jax.debug.print(
                            "iteration: {i}, time: {t:.6f}, dt: {d:.6f}",
                            i=iteration_new, t=time_new, d=dt),
                        lambda: None,
                    )

                return (time_new, iteration_new, i_snapshot_new,
                        last_written_new, Q_final, Qaux_final)

            def proceed(loop_val):
                """Check if simulation should continue."""
                time, iteration, i_snapshot, last_written, Qnew, Qaux = loop_val
                return _couple_proceed(time, time_end)

            (time, iteration, i_snapshot, last_written, Qnew, Qauxnew) = (
                jax.lax.while_loop(proceed, loop_body, loop_val)
            )

            # The end state.  The write stamps are ``i_snapshot * dt_snapshot``
            # with ``dt_snapshot = time_end / (snapshots - 1)``, so the last
            # stamp can land a roundoff ABOVE time_end (0.1 / 11 * 11 =
            # 0.1 + 1.4e-17) and the loop ends with the final state unwritten
            # (12 snapshots to 0.1 s stored 0.0918 s last).  Same fix as the
            # numpy loop (zoomy_core 129327d): write the end state iff the
            # last write is older than ``time``.  The gate is a ``lax.cond``
            # on that predicate (not a sentinel stamp fed to ``save_fields``,
            # whose ``time < stamp`` skip test is False for a NaN ``time``
            # and would write a duplicate record after a blow-up); a NaN
            # ``time`` compares False here, exactly as in the numpy loop.
            i_snapshot = jax.lax.cond(
                time > last_written,
                lambda: save_fields(time, time, i_snapshot, Qnew, Qauxnew),
                lambda: i_snapshot,
            )

            return Qnew, Qauxnew

        if return_cached is not None:
            time_loop = return_cached
        else:
            self._time_loop_cache = (_cache_key, time_loop)

        Q, Qaux = time_loop(0.0, 0.0, i_snapshot, last_written, Q, Qaux,
                            parameters)
        return Q, Qaux

    # ── solve (top-level entry point) ───────────────────────────────────

    def solve(self, mesh, model, write_output=True):
        """Full solve: initialize -> setup -> run.

        This is the main entry point, compatible with the NumPy solver
        interface. Calls ``setup_simulation`` then ``run_simulation``.
        """
        time_start = gettime()
        Q, Qaux = self.setup_simulation(mesh, model)
        Q, Qaux = self.run_simulation(Q, Qaux, write_output=write_output)
        jax.experimental.io_callback(
            log_callback_execution_time, None, gettime() - time_start
        )
        return Q, Qaux


# ── PoissonSolver ───────────────────────────────────────────────────────────

class PoissonSolver(SolverNumpy):
    """PoissonSolver. (class)."""
    def get_residual(
        self, Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt
    ):
        """Get residual."""
        def residual(Q):
            """Residual."""
            qaux = self.update_qaux(
                Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
            )
            q = boundary_operator(time, Q, qaux, parameters)
            res = model.residual(q, qaux, parameters)
            res = res.at[:, mesh.n_inner_cells :].set(0.0)
            return res

        return residual

    @jax.jit
    @partial(jax.named_call, name="poission_solver")
    def solve(self, mesh, model, write_output=True):
        """Solve."""
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)

        # dummy values for a consistent interface
        i_snapshot = 0.0
        time = 0.0
        time_next_snapshot = 0.0
        dt = 0.0

        Qold = Q
        Qauxold = Qaux

        boundary_operator = self.get_apply_boundary_conditions(mesh, model)

        Q = boundary_operator(time, Q, Qaux, parameters)
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        if write_output:
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            # MeshJAX has no write_to_hdf5; use the source NumPy
            # LSQMesh stashed in setup_simulation.  init_output_directory
            # prepends main_dir; write_to_hdf5 expects an absolute path
            # (h5py.File doesn't auto-resolve relative-to-main_dir).
            _abs_h5 = os.path.join(_misc.get_main_directory(), output_hdf5_path)
            self._rt_mesh_np.write_to_hdf5(_abs_h5)
            _write_field_names(_abs_h5, self.nsm.state)   # /fields@names
            save_fields = jax_io.get_save_fields(output_hdf5_path, True)
        else:

            def save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux):
                """Save fields."""
                return i_snapshot

        residual = self.get_residual(
            Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt
        )
        newton_solve = newton_solver(residual)

        time_start = gettime()

        Q = newton_solve(Q)

        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        i_snapshot = save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux)

        jax.experimental.io_callback(
            log_callback_execution_time, None, gettime() - time_start
        )

        return Q, Qaux
