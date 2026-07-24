"""JAX IMEX solver: explicit flux + implicit diffusion + implicit source.

Extends ``HyperbolicSolver`` (JAX) with:
- RK2 for explicit stage when reconstruction_order >= 2
- Implicit diffusion via Crank-Nicolson (DiffusionOperatorJAX)
- Implicit source stepping: local (cell-wise Newton) or global (Newton-GMRES)

All operations are JIT-compatible inside ``jax.lax.while_loop``.
"""

import os
import time as _time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

import zoomy_jax.fvm.ode as ode
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
import zoomy_core.misc.io as io
import zoomy_jax.misc.io as jax_io
from zoomy_core.misc import misc as _misc
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin
from zoomy_core.misc.logger_config import logger

from zoomy_jax.fvm.jvp_jax import analytic_source_jvp_jax
from zoomy_jax.mesh.mesh import (
    compute_derivatives, convert_mesh_to_jax, lsq_gradient_per_field)
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.numerics import NumericalSystemModel
from zoomy_jax.transformation.jax_runtime import JaxRuntime
from zoomy_jax.transformation.to_jax import JaxRuntimeModel


@dataclass
class IMEXStats:
    """Statistics from an IMEX solve."""
    n_steps: int = 0
    source_mode: str = "auto"
    implicit_calls: int = 0
    implicit_time_s: float = 0.0
    init_time_s: float = 0.0
    compile_time_s: float = 0.0
    runtime_only_s: float = 0.0
    total_time_s: float = 0.0


def _param_value(model, name, default=None):
    """Extract a scalar parameter value from a symbolic model.

    The canonical Model carries Symbols in ``model.parameters`` and the
    numeric values in ``model.parameter_values``. Read from the values
    side; fall back to the symbolic side only for legacy models that
    populate ``parameters`` numerically.
    """
    pv = getattr(model, "parameter_values", None)
    if pv is not None and pv.contains(name):
        return float(getattr(pv, name))
    ps = getattr(model, "parameters", None)
    if ps is not None and ps.contains(name):
        try:
            return float(getattr(ps, name))
        except (TypeError, ValueError):
            return default
    return default


def _build_diffusion_operators_jax(mesh_numpy, symbolic_model, dim, n_vars):
    """Build JAX DiffusionOperator for each variable (if model has diffusion).

    Parameters
    ----------
    mesh_numpy : BaseMesh / LSQMesh
        The original NumPy mesh (before JAX conversion). Needed for
        assembling the Laplacian with Python loops.
    symbolic_model : Model
        The symbolic model (has diffusive_flux, parameters, etc.)
    dim : int
    n_vars : int

    Returns
    -------
    dict or None
        {var_index: DiffusionOperatorJAX} (scalar path), {"__dense__": op}
        (dense/state-dependent path, REQ-109), or None if no diffusion.
    """
    # ── Dense / state-dependent path (REQ-109) ───────────────────────────
    # If the model carries a general rank-4 ``diffusion_matrix`` with any
    # off-diagonal (i≠j) / cross-derivative (d≠e) entry OR any state/aux
    # dependence, build the single DenseDiffusionOperatorJAX and let it
    # consume the runtime ``diffusion_matrix`` implicitly (Newton+GMRES for the
    # state-dependent case).  Classification is SHARED with numpy
    # (HyperbolicSolver._classify_diffusion) so both backends route identically
    # — mirror of core reconstruction.py::DenseDiffusionOperator (57f5e8f).
    import sympy as sp
    sym_A = getattr(symbolic_model, "diffusion_matrix", None)
    if sym_A is not None and not all(
            sp.simplify(e) == 0 for e in sp.flatten(sym_A)):
        from zoomy_core.fvm.solver_numpy import HyperbolicSolver
        needs_dense, state_dependent, _diag = \
            HyperbolicSolver._classify_diffusion(sym_A, symbolic_model)
        if needs_dense or state_dependent:
            from zoomy_jax.fvm.reconstruction_jax import DenseDiffusionOperatorJAX
            op = DenseDiffusionOperatorJAX(
                mesh_numpy, dim, n_vars, state_dependent=state_dependent)
            return {"__dense__": op}

    # ── Scalar path (constant diagonal): existing per-variable ν·L ───────
    if not hasattr(symbolic_model, 'diffusive_flux'):
        return None
    sym_dflux = symbolic_model.diffusive_flux()
    is_zero = hasattr(sym_dflux, 'tolist') and all(
        e == 0 for row in sym_dflux.tolist()
        for e in (row if isinstance(row, list) else [row])
    )
    if is_zero:
        return None
    nu_val = _param_value(symbolic_model, "nu", default=0.0)
    if nu_val <= 0:
        return None
    from zoomy_jax.fvm.reconstruction_jax import DiffusionOperatorJAX
    return {v: DiffusionOperatorJAX(mesh_numpy, dim, nu=nu_val)
            for v in range(n_vars)}


class IMEXSourceSolverJax(DerivativeAwareSolverMixin, HyperbolicSolver):
    """Pure-JAX IMEX solver with implicit diffusion and source stepping.

    The entire time integration compiles into a single XLA program via
    ``jax.lax.while_loop``.

    Features:
    - RK2 for explicit advection when reconstruction_order >= 2
    - Crank-Nicolson implicit diffusion via DiffusionOperatorJAX
    - Local sources  -> cell-wise Newton via ``jax.lax.fori_loop``
    - Global sources -> Newton-GMRES via ``jax.lax.while_loop``
    """

    source_mode = "auto"          # "auto" | "local" | "global"
    implicit_tol = 1e-8
    implicit_maxiter = 6
    gmres_tol = 1e-7
    gmres_maxiter = 30
    jv_backend = "ad"             # "analytic" | "fd" | "ad"
    fd_eps = 1e-7

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(**kwargs)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=self.source_mode))

    def create_runtime(self, Q, Qaux, mesh, model):
        """Create runtime.

        The runtime is the canonical :class:`JaxRuntime`, lambdified from
        the NSM's SystemModel matrices. This is the single bridge between
        the symbolic side (``model.parameters`` carries Symbols,
        ``model.parameter_values`` carries the numeric values) and the
        numerical side (``runtime.parameters`` is a jnp array of values).
        Earlier IMEX-jax revisions used :class:`JaxRuntimeModel(model)`,
        which lambdified the Model's pre-aux-substitution functions and
        choked on bare ``Derivative(b, x)`` / ``tau(t, x, σ)`` atoms.
        """
        mesh = ensure_lsq_mesh(mesh, model)
        # Periodic BCs are resolved in ``solve`` via the canonical _bc_source
        # container BEFORE create_runtime is called.  The old reach here,
        # ``model.boundary_conditions``, is the lambdified BC-kernel Function on
        # the SystemModel/NSM path (not a BoundaryConditions container) and
        # crashed resolve_periodic_bcs — so it is intentionally not repeated.
        jax_mesh = convert_mesh_to_jax(mesh)
        Q = jnp.asarray(Q)
        Qaux = jnp.asarray(Qaux)
        if not isinstance(model, NumericalSystemModel):
            self.nsm = NumericalSystemModel.from_system_model(model)
        else:
            self.nsm = model
        runtime_model = JaxRuntime.from_nsm(self.nsm)
        parameters = runtime_model.parameters
        return Q, Qaux, parameters, jax_mesh, runtime_model

    def _resolve_source_mode(self, model):
        """Resolve source mode."""
        if self.source_mode in ("local", "global"):
            return self.source_mode
        sym = model.model if hasattr(model, "model") else model
        has = hasattr(sym, "derivative_specs") and bool(sym.derivative_specs)
        return "global" if has else "local"

    def _resolve_jv_backend(self):
        """Resolve Jacobian-vector product backend."""
        return self.jv_backend if self.jv_backend in ("analytic", "fd", "ad") else "ad"

    # ── pure closures for JIT tracing ────────────────────────────────────

    @staticmethod
    def _build_update_qaux(runtime_model, mesh, parameters=None):
        """Return ``(Q, Qaux, Qold, dt) -> Qaux_new``, fully JAX-traceable.

        UNIFIED with the canonical jax ``HyperbolicSolver.update_qaux``
        (solver_jax.py): three composed legs, all no-ops when absent so the
        builder works for any model.

        1. LOCAL ``update_aux_variables`` prefix leg (e.g. KP ``hinv``) —
           previously MISSING here, so canonical aux models had their aux
           frozen at the IC across the whole IMEX run.
        2. NON-LOCAL ``aux_registry`` spatial-derivative leg via the shared
           BC-aware ``lsq_gradient_per_field`` (state + function-aux targets).
        3. LEGACY ``derivative_specs`` overlay — kept for StructuredDerivative/
           Green-Naghdi models, the ONLY path that can express a time
           derivative ``(Q-Qold)/dt`` (``aux_registry`` encodes spatial order
           only).  Additive: a no-op for canonical aux_registry models.

        ``parameters`` defaults to ``runtime_model.parameters`` so the
        gnn_blueprint children that call ``_build_update_qaux(rmodel, jmesh)``
        inherit the improved closure with no call-site change.
        """
        if parameters is None:
            parameters = getattr(runtime_model, "parameters", None)
        local_fn = getattr(runtime_model, "update_aux_variables", None)
        sm = getattr(runtime_model, "sm", None)
        registry = [
            e for e in (getattr(sm, "aux_registry", None) or [])
            if e.get("kind") == "derivative"
            and e.get("target_kind") in ("state", "function")
        ]
        sym = runtime_model.model if hasattr(runtime_model, "model") else runtime_model
        specs = list(getattr(sym, "derivative_specs", None) or [])
        field_idx = ({n: i for i, n in enumerate(sym.variables.keys())}
                     if specs else {})
        key_idx = getattr(sym, "derivative_key_to_index", {}) if specs else {}

        if local_fn is None and not registry and not specs:
            return lambda Q, Qaux, Qold, dt: Qaux

        def _uq(Q, Qaux, Qold, dt):
            out = Qaux
            # (1) local update_aux_variables prefix leg
            if local_fn is not None:
                local = local_fn(Q, Qaux, parameters)
                out = out.at[:local.shape[0]].set(local)
            # (2) aux_registry spatial-derivative leg (BC-aware LSQ)
            for e in registry:
                if e["target_kind"] == "state":
                    field = Q[e["state_index"]]
                else:
                    field = out[e["function_row"]]
                grad = lsq_gradient_per_field(
                    mesh, field, u_bf=None, multi_index=e["multi_index"])
                out = out.at[e["row"], :grad.shape[0]].set(grad)
            # (3) legacy derivative_specs overlay (incl. time derivative)
            for sp in specs:
                i_aux = key_idx[sp.key]
                i_q = field_idx[sp.field]
                n_t = sum(a == "t" for a in sp.axes)
                n_x = sum(a == "x" for a in sp.axes)
                data = Q[i_q]
                if n_t == 1:
                    data = (Q[i_q] - Qold[i_q]) / jnp.maximum(dt, 1e-14)
                if n_x > 0:
                    data = compute_derivatives(
                        data, mesh, derivatives_multi_index=[[n_x]])[:, 0]
                out = out.at[i_aux, :].set(data)
            return out

        return _uq

    @staticmethod
    def _build_implicit_local(runtime_model, parameters, n_vars, maxiter):
        """Cell-wise (diagonal) Newton via fori_loop.

        When the model carries an algebraic aux formula (e.g. the
        KP-desingularized ``hinv = KP(h)``), the implicit source is solved with
        that aux RECOMPUTED from the current iterate, and the per-cell Jacobian
        is taken by AD *through* that recompute — so the chain term
        ``∂S/∂aux·∂aux/∂Q`` (e.g. ``∂hinv/∂h ~ -1/h²``) is included.  The old
        path froze ``Qaux`` and used ``∂S/∂Q`` only, so the SME(1) friction
        solve was inconsistent (solved the wrong equation with a stale Jacobian)
        — task A.  jax AD chain-rules the already-lowered ``source``∘
        ``update_aux_variables`` exactly.

        Preferred (symbolic) route — used when the model carries an algebraic
        aux AND both symbolic source jacobians are lowered: assemble the
        consistent implicit Jacobian as ``∂S/∂Q + ∂S/∂aux·∂aux/∂Q`` from the two
        symbolic ``source_jacobian_wrt_{variables,aux_variables}`` blocks, with
        only ``∂aux/∂Q`` taken by AD of the cheap ``update_aux_variables`` map.
        This avoids AD through ``source`` entirely and is the exact building
        block an additive-RK (ARK) implicit stage needs.  If a symbolic block is
        missing (e.g. a model that never channeled ∂S/∂aux), it falls back to
        AD through ``source``∘``update_aux_variables``; models with no algebraic
        aux fall back to the frozen ``∂S/∂Q`` form."""
        uav = getattr(runtime_model, "update_aux_variables", None)
        sjq = getattr(runtime_model, "source_jacobian_wrt_variables", None)
        sja = getattr(runtime_model, "source_jacobian_wrt_aux_variables", None)

        def _impl(Qexp, Qaux, dt):
            if uav is not None and sjq is not None and sja is not None:
                # Cell-local aux re-derived from the iterate (frozen ``Qaux``
                # column supplies the non-local passthrough rows; only the
                # algebraic rows — e.g. KP ``hinv`` — depend on Q).
                def _aux_of(qv, qav):
                    return uav(qv[:, None], qav[:, None], parameters)[:, 0]
                def _S(qv, qav):
                    Qa = _aux_of(qv, qav)[:, None]
                    return runtime_model.source(qv[:, None], Qa, parameters)[:, 0]
                S_of = lambda Q: jax.vmap(
                    _S, in_axes=(1, 1), out_axes=1)(Q, Qaux)

                def _Jcell(qv, qav):
                    Qac = _aux_of(qv, qav)[:, None]          # (n_aux,1)
                    Jsq = sjq(qv[:, None], Qac, parameters)[:, :, 0]   # (n_eq,n_state)
                    Jsa = sja(qv[:, None], Qac, parameters)[:, :, 0]   # (n_eq,n_aux)
                    Jaq = jax.jacfwd(_aux_of, argnums=0)(qv, qav)      # (n_aux,n_state)
                    return Jsq + Jsa @ Jaq                            # (n_eq,n_state)
                J_of = lambda Q: jax.vmap(
                    _Jcell, in_axes=(1, 1), out_axes=2)(Q, Qaux)
            elif uav is not None:
                # AD through source∘update_aux (one column at a time; AD zeros
                # the passthrough rows that don't depend on Q).
                def _g(qv, qav):
                    Qa = uav(qv[:, None], qav[:, None], parameters)
                    return runtime_model.source(
                        qv[:, None], Qa, parameters)[:, 0]
                S_of = lambda Q: jax.vmap(
                    _g, in_axes=(1, 1), out_axes=1)(Q, Qaux)
                J_of = lambda Q: jax.vmap(
                    jax.jacfwd(_g, argnums=0), in_axes=(1, 1), out_axes=2)(Q, Qaux)
            else:
                S_of = lambda Q: runtime_model.source(Q, Qaux, parameters)
                J_of = lambda Q: runtime_model.source_jacobian_wrt_variables(
                    Q, Qaux, parameters)

            def body(i, Q):
                S = S_of(Q)
                Jq = J_of(Q)
                A = jnp.eye(n_vars, dtype=Q.dtype)[:, :, None] - dt * Jq
                R = Q - Qexp - dt * S
                d = jax.vmap(lambda Ac, rc: jnp.linalg.solve(Ac, -rc),
                             in_axes=(2, 1), out_axes=1)(A, R)
                return Q + d
            return jax.lax.fori_loop(0, maxiter, body, Qexp)
        return _impl

    @staticmethod
    def _build_implicit_global(runtime_model, mesh, parameters,
                               boundary_op, update_qaux, symbolic_model,
                               maxiter, tol, gmres_tol, gmres_maxiter,
                               jv_backend, fd_eps):
        """Newton-GMRES via while_loop with early exit on convergence."""
        def _impl(Qexp, Qauxold, Qold, time_now, dt):
            q_shape = Qexp.shape

            def residual(Qstate):
                Qa = update_qaux(Qstate, Qauxold, Qold, dt)
                Qbc = boundary_op(time_now, Qstate, Qa, parameters)
                S = runtime_model.source(Qbc, Qa, parameters)
                return Qbc - Qexp - dt * S

            if jv_backend == "ad":
                def matvec(Qc, _Rc, v):
                    V = v.reshape(q_shape)
                    return jax.jvp(residual, (Qc,), (V,))[1].reshape(-1)
            elif jv_backend == "fd":
                def matvec(Qc, Rc, v):
                    V = v.reshape(q_shape)
                    return ((residual(Qc + fd_eps * V) - Rc) / fd_eps).reshape(-1)
            else:
                def matvec(Qc, _Rc, v):
                    V = v.reshape(q_shape)
                    Qa = update_qaux(Qc, Qauxold, Qold, dt)
                    jvs = analytic_source_jvp_jax(
                        runtime_model, symbolic_model, Qc, Qa, V,
                        mesh, dt, include_chain_rule=True)
                    return (V - dt * jvs).reshape(-1)

            def cond(state):
                _, _, rnorm, i = state
                return jnp.logical_and(i < maxiter, rnorm > tol)

            def body(state):
                Q, R, _rnorm, i = state
                b = (-R).reshape(-1)
                delta, info = jax_gmres(
                    lambda v: matvec(Q, R, v), b,
                    atol=0.0, tol=gmres_tol, maxiter=gmres_maxiter)
                delta = jnp.where(info == 0, delta, b)
                Qn = boundary_op(time_now, Q + delta.reshape(q_shape),
                                 Qauxold, parameters)
                Rn = residual(Qn)
                return (Qn, Rn, jnp.linalg.norm(Rn.reshape(-1)), i + 1)

            R0 = residual(Qexp)
            init = (Qexp, R0, jnp.linalg.norm(R0.reshape(-1)), jnp.int32(0))
            return jax.lax.while_loop(cond, body, init)[0]

        return _impl

    @staticmethod
    def _build_implicit_diffusion(diffusion_ops_jax, bc_op, parameters,
                                  runtime_model=None, gmres_tol=1e-7,
                                  gmres_maxiter=30, newton_maxiter=8,
                                  newton_tol=1e-8):
        """Build a JIT-compatible implicit diffusion closure.

        Parameters
        ----------
        diffusion_ops_jax : dict or None
            ``{var_index: DiffusionOperatorJAX}`` (scalar path) or
            ``{"__dense__": DenseDiffusionOperatorJAX}`` (dense path, REQ-109).
        bc_op : callable
            Boundary condition operator.
        parameters : jnp.ndarray
            Model parameters.
        runtime_model : JaxRuntime, optional
            Needed for the dense path — supplies the lowered
            ``diffusion_matrix(Q, Qaux, p)`` as the operator's ``A_fn``.
        gmres_tol, gmres_maxiter, newton_maxiter, newton_tol :
            Implicit-solve knobs threaded to the dense operator (the scalar
            path uses its own dense-matrix solve).

        Returns
        -------
        callable or None
            (Q, Qaux, dt, time) -> Q_diffused, or None if no diffusion.
        """
        if diffusion_ops_jax is None:
            return None

        # ── Dense / state-dependent path (REQ-109) ───────────────────────
        if "__dense__" in diffusion_ops_jax:
            op = diffusion_ops_jax["__dense__"]
            dmat = getattr(runtime_model, "diffusion_matrix", None)
            if dmat is None:
                raise ValueError(
                    "dense implicit diffusion needs runtime.diffusion_matrix, "
                    "but the runtime did not lower it (REQ-109).")
            nc = op.nc

            def _apply_diffusion(Q, Qaux, dt, time):
                # Operate on inner cells only; the runtime diffusion_matrix
                # vmaps over the cell axis, so slice Q/Qaux to [:nc].  Qaux/p
                # are frozen across the Newton iterates (only Q varies), per
                # the numpy reference (solver_imex_numpy).
                Q_in = Q[:, :nc]
                Qaux_in = Qaux[:, :nc] if Qaux.shape[0] > 0 else Qaux

                def _A(Qs):
                    return dmat(Qs, Qaux_in, parameters)

                Q_new = op.implicit_solve(
                    Q_in, dt, _A, bf_grads=None,
                    tol=gmres_tol, maxiter=gmres_maxiter,
                    newton_maxiter=newton_maxiter, newton_tol=newton_tol)
                Q = Q.at[:, :nc].set(Q_new)
                # Boundaries: refill ghosts via the BC op (periodic / Neumann),
                # exactly as the scalar jax path does after its solve.
                Q = bc_op(time, Q, Qaux, parameters)
                return Q

            return _apply_diffusion

        # ── Scalar path (constant diagonal ν·L, per variable) ────────────
        var_indices = sorted(diffusion_ops_jax.keys())
        ops = [diffusion_ops_jax[v] for v in var_indices]

        def _apply_diffusion(Q, Qaux, dt, time):
            for v_idx, op in zip(var_indices, ops):
                u_diffused = op.implicit_solve(Q[v_idx, :], dt)
                Q = Q.at[v_idx, :].set(u_diffused)
            Q = bc_op(time, Q, Qaux, parameters)
            return Q

        return _apply_diffusion

    # ── main entry point ─────────────────────────────────────────────────

    def solve(self, mesh, model, write_output=False):
        """Full IMEX solve: setup + JIT-compiled time loop."""
        t0_wall = _time.time()

        # Keep reference to NumPy mesh for diffusion operator assembly.
        # Coerce to the NSM and resolve periodic BCs the canonical way (mirrors
        # solver_numpy.py:1016-1023 / solver_jax.py:682): the periodic topology
        # remap walks the BoundaryConditions CONTAINER, which lives at
        # ``sm._bc_source`` on the declarative path (or
        # ``source_model.boundary_conditions`` for a raw Model) — NOT
        # ``sm.boundary_conditions``, which is the lambdified BC-kernel Function
        # (a sympy ``Function`` with no ``boundary_conditions_list``; reaching it
        # was the old IMEX bug that predated the NSM→_bc_source→runtime path).
        mesh_numpy = ensure_lsq_mesh(mesh, model)
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        bcs_obj = (source_model.boundary_conditions if source_model is not None
                   else getattr(nsm.sm, "_bc_source", None))
        if bcs_obj is not None and hasattr(mesh_numpy, "resolve_periodic_bcs"):
            mesh_numpy.resolve_periodic_bcs(bcs_obj)

        # Size (Q, Qaux) from the NSM's SystemModel, NOT the raw source Model:
        # aux like the KP ``hinv`` is auto-promoted at the NSM level, so a raw
        # Model reports 0 aux and yields a (0, nc) Qaux while the runtime
        # operators (built from the NSM) expect the promoted aux — an empty-axis
        # gather in the per-cell operator (cid=101).  ``nsm.sm`` carries the
        # ICs too (NSM-by-construction deepcopy), so this also fills Q correctly.
        # For a pre-built-NSM caller source_model is None already, so this only
        # corrects the raw-Model entry path.
        Q, Qaux = self.initialize(mesh_numpy, nsm.sm)
        Q, Qaux, parameters, jmesh, rmodel = self.create_runtime(
            Q, Qaux, mesh_numpy, nsm)

        source_mode = self._resolve_source_mode(rmodel)
        jv_backend = self._resolve_jv_backend()
        n_vars = Q.shape[0]

        # Canonical state hygiene leg (update_variables / update_q), previously
        # never applied in the IMEX loop.  None (identity) for models without
        # a per-cell transform; a clamp/ramp for those that declare one.
        uv = getattr(rmodel, "update_variables", None)

        uq = self._build_update_qaux(rmodel, jmesh, parameters)
        if uv is not None:                       # update_q THEN update_qaux
            Q = uv(Q, Qaux, parameters)
        Qaux = uq(Q, Qaux, Q, jnp.asarray(1.0, dtype=Q.dtype))

        ev_op = self.get_compute_max_abs_eigenvalue(jmesh, rmodel)
        flux_op = self.get_flux_operator(jmesh, rmodel)
        bc_op = self.get_apply_boundary_conditions(jmesh, rmodel)

        Qnew = bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)
        Qauxnew = Qaux

        h_face = jnp.minimum(
            jmesh.cell_inradius[jmesh.face_cells[0]],
            jmesh.cell_inradius[jmesh.face_cells[1]],
        ).min()
        time_end = jnp.asarray(self.time_end, dtype=Q.dtype)
        compute_dt_fn = self.compute_dt
        # REQ-109: the symbolic handle carrying the sympy ``diffusion_matrix``
        # (a ZArray) is ``rmodel.sm`` on a ``JaxRuntime`` — ``rmodel`` itself
        # exposes the jax-LOWERED ``diffusion_matrix`` (a PjitFunction), which
        # ``_build_diffusion_operators_jax`` cannot flatten.  Resolve to the
        # SystemModel so the dense classifier reads the symbolic tensor.
        sym_model = (rmodel.model if hasattr(rmodel, "model")
                     else getattr(rmodel, "sm", None))

        # Build diffusion operators from the NumPy mesh
        symbolic_model = sym_model if sym_model is not None else rmodel
        diff_ops = _build_diffusion_operators_jax(
            mesh_numpy, symbolic_model,
            symbolic_model.dimension, n_vars)

        # Build implicit diffusion closure (dense path also needs the runtime
        # diffusion_matrix + implicit-solve knobs; REQ-109)
        impl_diffusion = self._build_implicit_diffusion(
            diff_ops, bc_op, parameters, rmodel,
            self.gmres_tol, self.gmres_maxiter,
            self.implicit_maxiter, self.implicit_tol)

        if source_mode == "local":
            impl = self._build_implicit_local(
                rmodel, parameters, n_vars, self.implicit_maxiter)
        else:
            impl = self._build_implicit_global(
                rmodel, jmesh, parameters, bc_op, uq, sym_model,
                self.implicit_maxiter, self.implicit_tol,
                self.gmres_tol, self.gmres_maxiter,
                jv_backend, self.fd_eps)

        # Select ODE stepper: RK2 for 2nd-order, RK1 for 1st-order
        ode_step = ode.RK2 if self.nsm.reconstruction.order >= 2 else ode.RK1

        # ── HDF5 snapshot output (AD-safe, mirrors HyperbolicSolver) ──────
        # IMEX previously wrote NO field snapshots (only a checkpoint sidecar),
        # so runs were not post-processable. Re-enable the canonical writer:
        # jax_io.get_save_fields uses pure_callback wrapped in custom_jvp (JVP
        # returns 0), transparent under autodiff — same path solver_jax uses.
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory,
                f"{self.settings.output.filename}.h5")
            io.init_output_directory(
                self.settings.output.directory,
                self.settings.output.clean_directory)
            mesh_numpy.write_to_hdf5(
                os.path.join(_misc.get_main_directory(), output_hdf5_path))
            io.save_settings(self.settings)
            save_fields = jax_io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
                return i_snapshot
        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        # initial snapshot (writes iteration_0 + the mesh group) → start counter
        i_snapshot0 = save_fields(0.0, 0.0, 0.0, Qnew, Qauxnew)

        # ── build the full loop ──────────────────────────────────────────

        def run_loop(Q0, Qa0, isnap0, max_steps):
            init = (jnp.asarray(0.0, dtype=Q0.dtype),
                    Q0, Qa0,
                    jnp.asarray(0, dtype=jnp.int32),
                    isnap0)

            def cond(s):
                return jnp.logical_and(s[0] < time_end, s[3] < max_steps)

            def step(s):
                t, Qn, Qan, ns, isnap = s
                dt = compute_dt_fn(
                    Qn, Qan, parameters, h_face, ev_op)
                dt = jnp.minimum(dt, time_end - t)

                # Explicit hyperbolic step — reuse the base solver's MOOD-aware
                # _explicit_hyperbolic_step so the IMEX explicit stage gets the
                # SAME a-priori front pre-detector + a-posteriori MOOD corrector
                # (RK2 for order≥2, Euler for order 1).  The IMPLICIT source
                # below then handles the stiff friction that the old explicit
                # RK1 source could not — together: MOOD = flux positivity,
                # IMEX = source stiffness (orthogonal axes).
                Qe = self._explicit_hyperbolic_step(
                    dt, t, Qn, Qan, parameters, flux_op,
                    int(jmesh.n_inner_cells))
                Qe = bc_op(t, Qe, Qan, parameters)

                # Implicit diffusion step (Crank-Nicolson)
                if impl_diffusion is not None:
                    Qe = impl_diffusion(Qe, Qan, dt, t)

                # Implicit source step
                if source_mode == "local":
                    Qa_imp = uq(Qe, Qan, Qn, dt)
                    Qi = impl(Qe, Qa_imp, dt)
                else:
                    Qi = impl(Qe, Qan, Qn, t, dt)

                Qn2 = bc_op(t + dt, Qi, Qan, parameters)
                if uv is not None:               # update_q THEN update_qaux
                    Qn2 = uv(Qn2, Qan, parameters)
                Qan2 = uq(Qn2, Qan, Qn, dt)

                t_new = t + dt
                ns_new = ns + 1
                # AD-safe HDF5 snapshot at the next evenly-spaced time stamp.
                isnap_new = save_fields(
                    t_new, isnap * dt_snapshot, isnap, Qn2, Qan2)
                # jax.debug.print is transparent under autodiff (unlike io_callback).
                jax.lax.cond(
                    (ns_new % 10) == 0,
                    lambda: jax.debug.print(
                        "iteration: {i}, time: {t:.6f}, dt: {d:.6f}",
                        i=ns_new, t=t_new, d=dt),
                    lambda: None)
                return (t_new, Qn2, Qan2, ns_new, isnap_new)

            return jax.lax.while_loop(cond, step, init)

        t_init = _time.time() - t0_wall

        # AOT compile
        has_diff = impl_diffusion is not None
        logger.info(
            f"JIT-compiling IMEX loop (source_mode={source_mode}, "
            f"jv={jv_backend}, ode={'RK2' if self.nsm.reconstruction.order >= 2 else 'RK1'}, "
            f"diffusion={'CN' if has_diff else 'none'}) ...")
        t_c0 = _time.time()
        lowered = jax.jit(run_loop).lower(
            Qnew, Qauxnew, i_snapshot0, jnp.int32(0))
        compiled = lowered.compile()
        t_compile = _time.time() - t_c0
        logger.info(f"Compilation done in {t_compile:.2f}s")

        # Execute
        t_r0 = _time.time()
        state = compiled(Qnew, Qauxnew, i_snapshot0, jnp.int32(2**30))
        _, Qnew, Qauxnew, n_steps, _isnap = state
        Qnew.block_until_ready()
        t_run = _time.time() - t_r0

        stats = IMEXStats(
            source_mode=source_mode,
            n_steps=int(n_steps),
            init_time_s=t_init,
            compile_time_s=t_compile,
            runtime_only_s=t_run,
            total_time_s=t_init + t_compile + t_run,
        )
        object.__setattr__(self, "last_stats", stats)
        logger.info(
            f"IMEX-JAX done: {int(n_steps)} steps, "
            f"compile={t_compile:.2f}s, run={t_run:.2f}s, "
            f"total={stats.total_time_s:.2f}s"
        )
        return jnp.asarray(Qnew), jnp.asarray(Qauxnew)
