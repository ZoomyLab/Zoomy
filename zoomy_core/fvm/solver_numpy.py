"""FVM solver for hyperbolic PDE systems (numpy backend).

Uses the symbolic Riemann solver (riemann_solvers.py) for flux computation.
No dependency on legacy flux.py or nonconservative_flux.py.

Solver hierarchy:
    Solver (base: init, create_runtime, BCs)
      -> HyperbolicSolver (explicit time stepping + symbolic Riemann flux)
           -> setup_simulation(mesh, model)
           -> run_simulation() -> Q, Qaux
           -> step(dt): apply_bcs -> reconstruct -> flux -> ode_step -> update_state
"""

import os
from time import time as gettime

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.io as io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.fvm.riemann_solvers import (
    PositiveNonconservativeRusanov,
    NonconservativeRusanov,
)


_EMPTY_AUX = np.array([])
_EMPTY_AUX2 = np.empty((0, 0))
# REQ-185: coordinate origin for the single position VECTOR argument ``X`` of
# the cell operators, used when a caller has no coordinate context (exact for
# coordinate-independent operators).  Binds x→0, y→0, z→0.
_ORIGIN3 = np.zeros(3)


# -- Field detection helpers ---------------------------------------------------

def _var_index(model, name, fallback=None):
    keys = list(model.variables.keys())
    if name in keys:
        return keys.index(name)
    if fallback is not None:
        return fallback
    raise KeyError(f"Variable '{name}' not found in model variables: {keys}")


def _param_value(model, name, default=None):
    # SystemModel keeps numeric values in ``parameter_values`` (``parameters``
    # holds Symbols); a Model keeps values in ``parameters``.
    pv = getattr(model, "parameter_values", None)
    if pv is not None and hasattr(pv, "contains") and pv.contains(name):
        return float(getattr(pv, name))
    params = getattr(model, "parameters", None)
    if params is not None and hasattr(params, "contains") and params.contains(name):
        try:
            return float(getattr(params, name))
        except (TypeError, ValueError):
            return default
    return default


def _detect_scaled_q_indices(model):
    if hasattr(model, "numerics_scaled_q_indices"):
        return model.numerics_scaled_q_indices
    keys = list(model.variables.keys())
    excluded = set()
    for name in ["b", "h"]:
        if name in keys:
            excluded.add(keys.index(name))
    return [i for i in range(model.n_variables) if i not in excluded]


def _nsm_explicit_numerics(solver, symbolic_model):
    """Return the NSM-owned Riemann numerics when the case EXPLICITLY chose
    one, else ``None`` so each solver keeps its own default (REQ-157).

    The Riemann solver is owned by the :class:`NumericalSystemModel`
    (``nsm.riemann`` + ``nsm.build_numerics()``) — jax and dmplex both honour
    it.  numpy historically hard-coded its own class and silently ignored the
    NSM's choice, so a case that built the NSM with e.g.
    ``PositiveNonconservativeHLL`` still ran Rusanov on numpy (a *different*
    discretisation than every other backend).  We defer only when the choice
    was explicit (``nsm.riemann_explicit``), so a defaulted NSM leaves the
    solver's own default (plain / positive / Roe) untouched."""
    nsm = getattr(solver, "nsm", None)
    if nsm is None or not getattr(nsm, "riemann_explicit", False):
        return None
    riemann = getattr(nsm, "riemann", None)
    if riemann is None:
        return None
    scaled = getattr(nsm, "scaled_q_indices", None)
    if scaled is None:
        scaled = _detect_scaled_q_indices(symbolic_model)
    return riemann(symbolic_model, scaled_q_indices=scaled)


# -- Boundary condition helpers ------------------------------------------------

def _compute_bf_face_gradients(Q, Qaux, bc_indices, bc_grad_fn, bf_cells,
                               bf_fidx, d_face, normals_arr, face_centers,
                               n_bf, n_vars, has_aux, time, parameters):
    """Compute per-variable boundary face-normal gradients via the
    indexed ``boundary_gradients(bc_idx, time, position, distance, Q,
    Qaux, p, normal)`` kernel.

    Returns dict ``{var_index: ndarray(n_bf,)}`` for use by
    ``DiffusionOperatorV2``."""
    bf_grads = {v: np.zeros(n_bf) for v in range(n_vars)}
    for i_bf in range(n_bf):
        q_inner = Q[:, bf_cells[i_bf]]
        qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else np.array([])
        fidx = bf_fidx[i_bf]
        normal = normals_arr[:, fidx]
        position = face_centers[fidx, :]
        fg = bc_grad_fn(
            bc_indices[i_bf], time, position, d_face[i_bf],
            q_inner, qaux_inner, parameters, normal,
        )
        for v in range(n_vars):
            bf_grads[v][i_bf] = fg[v]
    return bf_grads


# -- Helpers -------------------------------------------------------------------

def _coerce_to_system_model(model):
    """Return a :class:`SystemModel` view of ``model``.

    Accepts a :class:`Model` (calls :meth:`SystemModel.from_model`),
    a :class:`SystemModel` (returned unchanged), or a
    :class:`NumericalSystemModel` (returns ``nsm.sm``). This is the
    canonical source of truth for ``state`` / ``aux_state`` sizing
    after :meth:`SystemModel.from_model.expose_aux_atoms` has run.
    """
    if isinstance(model, NumericalSystemModel):
        return model.sm
    if isinstance(model, SystemModel):
        return model
    return SystemModel.from_model(model)


# -- Base Solver ---------------------------------------------------------------

def _spectral_bound(A, mode="numerical"):
    """``max|lambda|`` for a BATCH of matrices ``A`` of shape ``(n, m, m)``.

    ``mode``:

    * ``"gershgorin"`` — ``||A||_inf = max_i sum_j |A_ij|``.  A RIGOROUS upper
      bound on the spectral radius (union of Gershgorin discs), computed with
      two reductions and NO eigensolver.  Always valid; conservative, so it
      over-estimates the wave speed (safe: more Rusanov dissipation, smaller
      dt).
    * ``"symmetric"`` — ``rho((A + A^T)/2)`` via the SYMMETRIC eigensolver.
      Valid when the spectrum is REAL, i.e. while the system is hyperbolic:
      then ``lambda`` lies in the numerical range, so
      ``lambda in [lambda_min(A_sym), lambda_max(A_sym)]``.  It bounds only the
      REAL PART in general -- for a matrix with complex eigenvalues (loss of
      hyperbolicity) it can UNDER-estimate, e.g. ``[[0,1],[-1,0]]`` has
      ``rho = 1`` but a zero symmetric part.  Tighter than Gershgorin and uses
      the well-supported ``eigvalsh``.
    * anything else — the exact non-symmetric spectrum (LAPACK/XLA ``eig``).

    Measured (120 x 14 x 14, one call): numpy eig 8.09 ms, jax-CPU eig 8.55 ms,
    jax-GPU eig 114.8 ms, eigvalsh 0.76-5.3 ms, Gershgorin 0.38-0.48 ms.  The
    non-symmetric eig has no efficient XLA lowering, so the bound modes are the
    only ones that are cheap on GPU.
    """
    if mode == "gershgorin":
        return np.abs(A).sum(axis=2).max(axis=1)
    if mode == "symmetric":
        A_sym = 0.5 * (A + np.transpose(A, (0, 2, 1)))
        return np.abs(np.linalg.eigvalsh(A_sym)).max(axis=1)
    return np.abs(np.real(np.linalg.eigvals(A))).max(axis=1)


class Solver(param.Parameterized):
    """Base solver class: initialization, runtime creation, boundary conditions."""

    settings = param.Parameter(default=None, doc="Run configuration (Settings object)")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.settings is None:
            self.settings = Settings.default()
        else:
            defaults = Settings.default()
            defaults.update(self.settings)
            self.settings = defaults

    def initialize(self, mesh, model):
        """Allocate (Q, Qaux) of shape ``(n_state, n_inner_cells)``
        and ``(n_aux_state, n_inner_cells)``.

        The aux size MUST come from the SystemModel's ``aux_state``,
        not from the source Model's ``aux_variables``. Canonical
        derivation classes (SME, VAM, MLME, MLVAM) leave
        ``aux_variables`` empty and rely on
        :meth:`SystemModel.from_model` ``expose_aux_atoms`` to
        auto-promote bathymetry / derivative / function atoms into
        ``sm.aux_state``. The Model-side count is therefore wrong
        for any model that uses auto-promotion. We normalize to a
        SystemModel here so the allocation matches the runtime's
        view (which is also SystemModel-based).
        """
        nc = mesh.n_inner_cells
        # Resolve the SystemModel-side view of state + aux. Accepts a
        # Model (auto-promote), a SystemModel (use directly), or an NSM
        # (unwrap to ``nsm.sm``).
        sm = _coerce_to_system_model(model)
        n_variables = len(sm.state)
        n_aux_variables = len(sm.aux_state)
        Q = np.empty((n_variables, nc), dtype=float)
        Qaux = np.empty((n_aux_variables, nc), dtype=float)
        return Q, Qaux

    def create_runtime(self, Q, Qaux, mesh, model):
        """Build the numpy runtime from a :class:`SystemModel`.

        Contract: ``model`` is a :class:`SystemModel` — the
        self-contained numerical model.  Every ``setup_simulation``
        normalises its input to a SystemModel before reaching here.
        The runtime comes from
        :meth:`NumpyRuntimeModel.from_system_model`; the numeric
        parameter array from ``parameter_values``.  Numerical
        regularisation, if wanted, is a separate
        ``SystemModel → SystemModel`` pass applied *before* this point.
        """
        Q, Qaux = np.asarray(Q), np.asarray(Qaux)
        parameters = np.array(
            list(model.parameter_values.values()), dtype=float)
        runtime_model = NumpyRuntimeModel.from_system_model(model)
        return Q, Qaux, parameters, mesh, runtime_model

    def update_q(self, Q, Qaux, mesh, model, parameters, dt):
        """Apply ``model.update_variables`` (h-clamp, momentum ramp) at
        each cell — evaluated for all cells in one batched call.

        ``update_variables`` is carried through the SystemModel and
        exposed on the runtime as ``update_variables(Q, Qaux, p, dt)`` —
        the identity for models with no per-cell transform (``dt`` then
        ignored).  It is ``None`` only for SystemModels assembled directly
        without one (e.g. split sub-systems); then this is a genuine no-op,
        not a legacy fallback.

        Batching is exact: the lowered kernel is a lambdified numpy
        expression evaluated element-wise (Min/Max → ``minimum``/``maximum``,
        Piecewise → ``select``) and each cell's rows depend only on that
        cell's own column, so ``(n_vars, n_cells)`` in gives
        ``(n_rows, n_cells)`` out with no cross-cell accumulation.
        ``Q[:, :] = vals[:n_vars]`` keeps the per-cell prefix/broadcast
        semantics of ``.ravel()[:n_vars]``."""
        update = getattr(model, "update_variables", None)
        if update is None:
            return Q
        n_vars, n_cells = Q.shape
        aux = Qaux if Qaux.shape[0] > 0 else np.empty((0, n_cells))
        try:
            vals = np.asarray(update(Q, aux, parameters, dt), dtype=float)
            if vals.ndim != 2 or vals.shape[1] != n_cells:
                # Only the element-wise block (n_rows, n_cells) is accepted.
                # Any other shape is ambiguous or non-local — a bare scalar
                # row comes back (n_cells,) (indistinguishable from n_cells
                # scalar rows), a cross-cell reduction ``[q.sum()]`` comes
                # back (1,) — so it is NOT normalised; the per-cell loop is
                # the only exact evaluation for such a callable.
                raise ValueError(
                    f"batched update_variables returned {vals.shape}")
        except (ValueError, TypeError):
            # Fallback: a callable that does not broadcast over cells or
            # does not return the (n_rows, n_cells) block — the per-cell
            # loop, identical to the pre-batching behaviour.
            for c in range(n_cells):
                a = Qaux[:, c] if Qaux.shape[0] > 0 else _EMPTY_AUX
                Q[:, c] = np.asarray(
                    update(Q[:, c], a, parameters, dt), dtype=float,
                ).ravel()[:n_vars]
            return Q
        Q[:, :] = vals[:n_vars]
        return Q

    def _apply_local_aux_formula(self, model, Qaux, Q, parameters, time,
                                 cell_centers=None, *, copy=False):
        """Apply the lowered ``update_aux_variables`` per-cell formula (e.g.
        the KP-desingularized ``hinv``, or a time-dependent rain rate) to
        ``Qaux``.

        REQ-185: ``update_aux_variables`` is lowered as
        ``(Q, Qaux, p, t, x, y, z)`` — a time-dependent aux (rain rate
        ``r_o = Piecewise((rate, t<T_rain),(0,True))``) binds ``t``; a spatial
        aux binds the cell centre.  ``cell_centers`` is the mesh's
        ``(3, n_cells)`` array; when absent the coordinate origin is used
        (exact for coordinate-independent aux such as ``hinv``).

        ``callable`` (not merely ``is not None``): a SystemModel used directly
        (DAE path) carries ``update_aux_variables`` as a symbolic ZArray coerced
        from ``None`` — non-callable, NOT a per-cell formula.  Only a lowered
        runtime callable is a genuine local-aux leg; the placeholder is skipped
        (returns ``Qaux`` unchanged).  ``copy=True`` writes into a fresh array
        (the canonical :meth:`update_qaux` contract); ``copy=False`` refreshes
        in place (the Chorin per-pool refresh)."""
        local_fn = getattr(model, "update_aux_variables", None)
        if not callable(local_fn) or Qaux.shape[0] == 0:
            return Qaux
        out = np.array(Qaux, copy=True) if copy else Qaux
        n_cells = Q.shape[1]
        # Batched evaluation over all cells.  Exact because the lowered kernel
        # is a lambdified element-wise numpy expression whose constant /
        # time-only rows are already ``c*ones_like(<anchor state>)``
        # (vectorize.uniform_rank), so (n_vars, n_cells) / (n_aux, n_cells) /
        # (3, n_cells) in gives (n_rows, n_cells) out; every cell's rows are
        # computed from its own OLD column, exactly as the loop did.  The
        # origin is broadcast to (3, n_cells) — NOT (3, 1) — because a
        # position-only row (``x**2``) is not anchor-wrapped and would come
        # back (1,) beside (n_cells,) rows (ragged → ValueError).
        # ``cell_centers`` carries ghost columns after the inner ones.
        pos = (cell_centers[:, :n_cells] if cell_centers is not None
               else np.broadcast_to(_ORIGIN3[:, None], (3, n_cells)))
        try:
            vals = np.asarray(local_fn(Q, Qaux, parameters, time, pos),
                              dtype=float)
            if vals.ndim != 2 or vals.shape[1] != n_cells:
                # Only the element-wise block (n_rows, n_cells) is accepted
                # (what the lowered ``array([...])`` kernel returns).  Any
                # other shape is ambiguous or non-local — a bare scalar row
                # comes back (n_cells,) (indistinguishable from n_cells
                # scalar rows), a cross-cell reduction ``[q.sum()]`` comes
                # back (1,) — so it is NOT normalised; the per-cell loop is
                # the only exact evaluation for such a callable.
                raise ValueError(
                    f"batched update_aux_variables returned {vals.shape}")
        except (ValueError, TypeError):
            # Fallback: a formula that cannot broadcast over cells or does
            # not return the (n_rows, n_cells) block — the per-cell loop,
            # identical to the pre-batching behaviour.
            for c in range(n_cells):
                pos_c = (cell_centers[:, c] if cell_centers is not None
                         else _ORIGIN3)
                v = np.asarray(
                    local_fn(Q[:, c], Qaux[:, c], parameters, time, pos_c),
                    dtype=float).ravel()
                out[:v.shape[0], c] = v
            return out
        out[:vals.shape[0], :] = vals
        return out

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        """Fill the auxiliary vector with BOTH legs the SystemModel gathered:

        1. the LOCAL per-cell aux formula — the lowered
           ``update_aux_variables`` (e.g. KP ``hinv``), applied per cell (one
           batched call) like :meth:`update_q`.  Absent/``None`` for models
           with no aux formula
           (identity), so this leg is then skipped.
        2. the NON-LOCAL derivative-aux rows — ``kind == 'derivative'`` entries
           of ``aux_registry`` filled via ``LSQMesh.compute_derivatives`` (state
           ``Q`` for state derivatives, the WORKING aux for function-aux
           derivatives).

        Subclasses override to supply the ``kind == 'function'`` rows and call
        ``super().update_qaux(...)``.  No-op if neither leg is present.
        """
        # (1) LOCAL aux formula leg — per-cell (only present when the model
        # declares a real, non-identity update_aux_variables).  REQ-185: thread
        # the current ``time`` and the mesh cell centres so a time/space-
        # dependent aux (rain rate, manufactured source) binds them.
        out = self._apply_local_aux_formula(
            model, Qaux, Q, parameters, time,
            getattr(mesh, "cell_centers", None), copy=True)
        # (2) DERIVATIVE aux leg — the non-local LSQ-gradient rows the
        # SystemModel gathered in aux_registry, via the shared walk (the SINGLE
        # source the Chorin per-pool refresh also calls).
        return self._walk_derivative_aux(
            self._sm_from_solver_or_model(model), out, Q, mesh)

    def _walk_derivative_aux(self, sm, Qaux, Q, mesh, *,
                             kinds=("derivative",),
                             limiter_fn=None,
                             copy=True, registry=None):
        """Fill ``aux_registry`` derivative rows of ONE SystemModel by LSQ —
        the single source the canonical :meth:`update_qaux` AND
        ``ChorinSplitVAMSolver.update_aux_variables`` (per sub-system pool)
        both call.

        ``kinds`` / ``limiter_fn`` keep each caller's exact scope: the
        canonical post-step walks plain spatial ``derivative`` rows with no
        limiter (defaults → byte-identical to the old inline leg); the Chorin
        pools also walk ``limited_derivative`` rows and apply a TVD limiter
        (``limiter_fn`` resolved from the model-declared
        ``entry['limiter_scheme']``).  State-derivative targets read ``Q``;
        function-aux targets read the working aux; boundary faces use
        extrapolation (Neumann-zero).  ``copy=True`` returns a fresh array
        (canonical: don't clobber the caller's Qaux); ``copy=False`` mutates
        in place (Chorin: refresh the pool array itself).

        ``registry`` overrides ``sm.aux_registry`` — the Chorin pressure
        pool passes ``aux_registry + aux_input_registry`` so its private
        Qaux pool gets BOTH the live pressure derivatives and the frozen
        predictor-produced input derivatives filled in one pass (the
        printer-facing ``aux_registry`` stays minimal)."""
        if registry is None:
            registry = (getattr(sm, "aux_registry", None)
                        if sm is not None else None)
        if not registry:
            return Qaux
        out = np.array(Qaux, copy=True) if copy else Qaux
        nc = Q.shape[1]
        u_full = np.zeros(mesh.n_cells)
        for entry in registry:
            if entry["kind"] not in kinds:
                continue
            row = entry["row"]
            if row >= out.shape[0]:
                continue
            mi = entry["multi_index"]
            tk = entry["target_kind"]
            if tk == "state":
                u_full[:nc] = Q[entry["state_index"], :]
            elif tk == "function":
                u_full[:nc] = out[entry["function_row"], :]
            else:
                # Unknown target — leave whatever the caller put there.
                continue
            grad = mesh.compute_derivatives(
                u_full, degree=max(mi), derivatives_multi_index=[mi],
                u_boundary_face="extrapolation",
            )[:nc, 0]
            if limiter_fn is not None and entry["kind"] == "limited_derivative":
                scheme = entry.get("limiter_scheme")
                if scheme is not None:
                    grad = limiter_fn(u_full[:nc], grad, mi, scheme, mesh)
            out[row, :] = grad
        return out

    def _sm_from_solver_or_model(self, model):
        """Look up the SystemModel: prefer ``self.sm`` (when the
        solver was built that way), else ``model._chain_systemmodel``
        if present, else None."""
        if hasattr(self, "sm") and self.sm is not None:
            return self.sm
        return getattr(model, "_chain_systemmodel", None)

    # -- NSM coercion --------------------------------------------------

    def _coerce_to_nsm(self, model):
        """Normalise ``model`` to a (NumericalSystemModel, source-Model)
        pair.  Auto-promotes Model / SystemModel by building an NSM
        with default specs (``ReconstructionSpec(order=1)``,
        ``eigenvalue_eps=1e-8``, etc.) — to override
        defaults, build the NSM explicitly and pass that in.

        Lives on the BASE ``Solver`` because it is the one front door that
        makes a solver's aux/state sizing agree with the kernels
        ``NumpyRuntimeModel.from_system_model`` compiles (which promotes to
        an NSM internally, registering e.g. the KP ``hinv`` aux).  Every
        solver that allocates buffers must go through it.

        ``source_model`` is the *original* :class:`Model` when one was
        passed (needed for ``resolve_periodic_bcs``, which walks
        ``boundary_conditions_list`` — a field that lives on the Model
        object and not on the lambdified SystemModel BC kernel).  When
        the caller supplied a bare SystemModel or NSM, ``source_model``
        is ``None``."""
        if isinstance(model, NumericalSystemModel):
            return model, None
        nsm = NumericalSystemModel.from_system_model(model)
        source_model = None if isinstance(model, SystemModel) else model
        return nsm, source_model


# -- HyperbolicSolver ----------------------------------------------------------

class HyperbolicSolver(Solver):
    """Explicit time-stepping solver using the symbolic Riemann solver.

    Core methods:
        setup_simulation(mesh, model) -- build all operators once
        run_simulation()              -- time loop: compute_dt -> step -> output
        step(dt)                      -- one timestep (readable, no if-clauses)
        solve(mesh, model)            -- convenience: setup + run
    """

    time_end = param.Number(default=0.1, bounds=(0, None), doc="Simulation end time")
    min_dt = param.Number(default=1e-6, bounds=(0, None), doc="Minimum allowed timestep")
    compute_dt = param.Parameter(default=None, doc="Time-stepping strategy (callable)")
    _diffusion_in_flux = True  # explicit diffusion in flux operator
    # Numerical knobs (``reconstruction.order``, ``reconstruction.limiter``,
    # ``eigenvalue_eps``) live on the
    # :class:`NumericalSystemModel`; the solver reads ``self.nsm.*`` after
    # ``setup_simulation``.  Constructors no longer accept those kwargs —
    # pass an NSM (or a Model / SystemModel that gets auto-promoted to
    # one with default specs).

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.compute_dt is None:
            self.compute_dt = timestepping.adaptive(CFL=0.9)
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        self.settings = defaults

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        nc = mesh.n_inner_cells
        Q = model.initial_conditions.apply(mesh.cell_centers[:, :nc], Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers[:, :nc], Qaux)
        return Q, Qaux

    # -- Symbolic model helpers ----------------------------------------

    def _get_symbolic_model(self, model):
        return model.model if hasattr(model, "model") else model

    # -- Max abs eigenvalue (for the CFL/dt estimate ONLY) -------------
    #
    # cid=209: this is the dt consumer, and it is the ONLY one that reads a
    # gated spectrum.  When ``gate_eigenvalues_dry`` was opted in it published
    # ``eigenvalues_cfl`` (dry cells zeroed) — read THAT here so a dry cell
    # cannot drive dt.  The Rusanov face dissipation lives in
    # ``numerics.riemann_solvers.local_max_abs_eigenvalue`` and deliberately
    # keeps reading the ungated ``eigenvalues``: ``s_max`` IS the artificial
    # viscosity of the positivity argument, and zeroing it at the wet/dry front
    # ran those faces centrally and exhausted the MOOD cascade.

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        keys = list(symbolic_model.variables.keys())
        fi_h = keys.index("h") if "h" in keys else None
        dim = symbolic_model.dimension
        normals = mesh.face_normals[:dim, :]
        has_aux = symbolic_model.n_aux_variables > 0

        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        # Split faces into interior and boundary
        int_mask = (iA < nc) & (iB < nc)
        bnd_mask = ~int_mask
        interior_faces = np.where(int_mask)[0]
        boundary_faces = np.where(bnd_mask)[0]
        iA_int = iA[interior_faces]
        iB_int = iB[interior_faces]
        # face_cells[0] is guaranteed to be the inner cell at boundary faces
        iInner_bnd = iA[boundary_faces]

        eig_mode = getattr(symbolic_model, "eigenvalue_mode", "symbolic")
        if (eig_mode != "numerical"
                and getattr(symbolic_model, "eigenvalues", None) is not None):
            # BATCHED symbolic wave speeds: one vectorized evaluation of the
            # registered closed-form spectrum (e.g. the SME β-HSWME
            # eigenvalues) per face side — no per-face Python loop and no
            # LAPACK at all.
            rt = NumpyRuntimeModel.from_system_model(symbolic_model)
            # Gated slot when a dry gate was opted in, else the raw spectrum.
            eig_fn = getattr(rt, "eigenvalues_cfl", None) or rt.eigenvalues
            n_int = normals[:, interior_faces]
            n_bnd = normals[:, boundary_faces]
            n_vars_s = symbolic_model.n_variables

            def _side_max(Q, Qaux, parameters, cells, ns):
                q_s = Q[:, cells]
                a_s = Qaux[:, cells] if Qaux.shape[0] > 0 else _EMPTY_AUX2
                evs = np.asarray(eig_fn(q_s, a_s, parameters, ns), float)
                evs = evs.reshape(n_vars_s, -1)
                return np.abs(evs).max(axis=0)

            def compute_max_eigenvalue(Q, Qaux, parameters):
                max_ev = np.zeros(mesh.n_faces)
                m_A = _side_max(Q, Qaux, parameters, iA_int, n_int)
                m_B = _side_max(Q, Qaux, parameters, iB_int, n_int)
                max_ev[interior_faces] = np.maximum(m_A, m_B)
                max_ev[boundary_faces] = _side_max(
                    Q, Qaux, parameters, iInner_bnd, n_bnd)
                return max_ev
            return compute_max_eigenvalue

        # BATCHED numerical wave speeds (model carries no closed-form
        # spectrum — ``eigenvalues is None``): one vectorized quasilinear
        # evaluation over all cells + one stacked np.linalg.eigvals over
        # all face-side matrices, two orders of magnitude cheaper than a
        # per-face Python loop.  This is the dt path and it stays EXACT; the
        # flux dissipation takes the cheap ``spectral_radius_bound`` instead
        # (``riemann_solvers.local_max_abs_eigenvalue``), because that one is
        # evaluated per face inside emitted C where an eigensolve is not
        # batchable.
        rt = NumpyRuntimeModel.from_system_model(symbolic_model)
        ql_fn = rt.quasilinear_matrix
        n_vars = symbolic_model.n_variables
        eps_reg = self.nsm.eigenvalue_eps
        keys = list(symbolic_model.variables.keys())
        reg = eps_reg * np.eye(n_vars)
        if fi_h is not None and "b" in keys:
            reg[keys.index("b"), keys.index("b")] = 0.0
        n_int = normals[:, interior_faces]          # (dim, n_if)
        n_bnd = normals[:, boundary_faces]

        def compute_max_eigenvalue(Q, Qaux, parameters):
            ql = np.asarray(ql_fn(Q, Qaux, parameters), dtype=float)
            nc_ = Q.shape[1]
            if ql.shape == (n_vars, n_vars, nc_, dim):
                # vectorize wrapper appends the cell axis before dim
                ql = np.transpose(ql, (0, 1, 3, 2))
            elif ql.shape != (n_vars, n_vars, dim, nc_):
                raise ValueError(
                    f"unexpected quasilinear shape {ql.shape}")
            # (n, n, dim, nc); A·n per face side → (n_faces_side, n, n)
            A_A = np.einsum("ijdk,dk->kij", ql[:, :, :, iA_int], n_int)
            A_B = np.einsum("ijdk,dk->kij", ql[:, :, :, iB_int], n_int)
            A_I = np.einsum("ijdk,dk->kij", ql[:, :, :, iInner_bnd], n_bnd)
            A_all = np.concatenate([A_A, A_B, A_I], axis=0) + reg[None]
            # REQ-168 ADDENDUM 2 (third site): batched LAPACK raises on
            # non-finite entries.  A non-finite COMMITTED cell state at dt
            # time maps to a +inf wave speed — dt clamps to zero, an honest
            # halt instead of a crash; genuinely dry cells are still zeroed
            # by the dry-cell skip below, exactly as before.
            okA = np.isfinite(A_all).all(axis=(1, 2))
            m = np.full(A_all.shape[0], np.inf)
            if okA.any():
                m[okA] = _spectral_bound(A_all[okA], eig_mode)
            n_if = len(interior_faces)
            max_ev = np.zeros(mesh.n_faces)
            max_ev[interior_faces] = np.maximum(m[:n_if], m[n_if:2 * n_if])
            max_ev[boundary_faces] = m[2 * n_if:]
            return max_ev
        return compute_max_eigenvalue

    # -- Flux operator (symbolic Riemann solver) -----------------------

    def _build_numerics(self, symbolic_model):
        """Build the symbolic Riemann solver. Override for SWE-specific variants.

        Well-balancing (``equilibrium_reconstruction`` in {'audusse','bernoulli'})
        is applied as an interface reconstruction + well-balancing source in the
        flux operator (the unified ``reconstruct_equilibrium`` hook), on top of
        the standard non-conservative Rusanov numerics — so both modes go through
        the SAME path (Audusse is the lake-at-rest check on Bernoulli's wiring).
        The source differs per mode (built in ``bernoulli_wb``): Audusse uses the
        PRESSURE jump only (momentum-only S̃ — zero mass row, no convective/aux
        dependence, dimension-agnostic), Bernoulli the full conservative-flux
        jump (its discharge-preserving reconstruction zeroes the mass row).

        When the NSM explicitly carries a Riemann choice, honour it (REQ-157)
        — this is the same contract jax/dmplex follow; the NSM owns the
        numerics."""
        nsm_numerics = _nsm_explicit_numerics(self, symbolic_model)
        if nsm_numerics is not None:
            return nsm_numerics
        return NonconservativeRusanov(symbolic_model)

    def _build_diffusion_operators(self, mesh, symbolic_model, dim, n_vars):
        """Build the diffusion operators from the SystemModel's rank-4
        ``diffusion_matrix`` ``A(Q, Qaux, p)`` (shape ``(n_eq, n_state,
        n_dim, n_dim)``) — the constitutive tensor in ``div(A : grad Q)``.

        Two paths, selected by the STRUCTURE of ``A`` (:meth:`_classify_diffusion`):

        * **Scalar path** (constant DIAGONAL ``A[v, v, d, d]``): the per-variable
          :class:`DiffusionOperatorV2` (``ν × L``), returned as a ``{v: op}`` dict
          and consumed by the existing per-variable implicit/explicit loops.
          Gated on ``ν > 0`` (SME source-viscosity / VAM inviscid runs untouched).
        * **Dense path** (any off-diagonal ``A[i, j]`` ``i≠j``, cross-derivative
          ``d≠e``, OR state-dependent entry): a single
          :class:`DenseDiffusionOperator` stored on ``self._dense_diffusion``.
          It assembles + solves the GENERAL ``∇·(A:∇Q)`` (cross-variable, evaluated
          at ``Q^{n+1}`` — Newton when ``A`` depends on the state), which the IMEX
          solver drives in ``_apply_implicit_diffusion``.  No ``ν`` gate (the
          diffusivity lives in the tensor, e.g. μ(I) ν-moments carried as aux).

        Returns the scalar ``{v: op}`` dict (or ``None``); the dense operator, when
        built, is on ``self._dense_diffusion`` (else ``None``)."""
        self._dense_diffusion = None
        sym_A = getattr(symbolic_model, 'diffusion_matrix', None)
        if sym_A is None:
            return None
        if all(sp.simplify(e) == 0 for e in sp.flatten(sym_A)):
            return None

        needs_dense, state_dependent, diagonal = self._classify_diffusion(
            sym_A, symbolic_model)

        if needs_dense or state_dependent:
            # GENERAL dense / state-dependent diffusion — the model's own
            # diffusion_matrix, assembled + solved implicitly (no scalar-ν
            # reduction, no parabolic CFL).  Consumed only by the IMEX
            # implicit-diffusion step; the runtime ``diffusion_matrix``
            # callable is fetched at apply time from ``self._sim_model``.
            from zoomy_core.fvm.reconstruction import DenseDiffusionOperator
            self._dense_diffusion = DenseDiffusionOperator(
                mesh, dim, n_vars, state_dependent=state_dependent)
            return None

        # Scalar-ν diagonal path (unchanged, bit-for-bit reference).
        nu_val = _param_value(symbolic_model, "nu", default=0.0)
        if nu_val <= 0:
            return None
        from zoomy_core.fvm.reconstruction import DiffusionOperatorV2
        diffusivity = {v: sp.sympify(-diagonal[v]) for v in diagonal}
        state = list(symbolic_model.variables.get_list())
        aux = list(symbolic_model.aux_variables.get_list())
        par = list(symbolic_model.parameters.get_list())
        self._diffusivity_fns = {
            v: sp.lambdify(state + aux + par, expr, "numpy")
            for v, expr in diffusivity.items()}
        return {v: DiffusionOperatorV2(mesh, dim, nu=nu_val) for v in diffusivity}

    @staticmethod
    def _classify_diffusion(sym_A, symbolic_model):
        """Classify a rank-4 ``diffusion_matrix`` for path selection.

        Returns ``(needs_dense, state_dependent, diagonal)``:

        * ``needs_dense`` — any non-zero off-diagonal (``i≠j``) or
          cross-derivative (``d≠e``) entry (the scalar operator only sums
          ``A[v,v,d,d]``).
        * ``state_dependent`` — any entry depends on a state or aux symbol
          (⇒ the implicit solve is nonlinear ⇒ Newton).
        * ``diagonal`` — ``{v: Σ_d A[v,v,d,d]}`` (the scalar-path diffusivity
          source, only meaningful when neither flag is set)."""
        import itertools
        n_eq, n_st, n_d = sym_A.shape[0], sym_A.shape[1], sym_A.shape[2]
        state_syms = set(symbolic_model.variables.get_list()) | set(
            symbolic_model.aux_variables.get_list())
        needs_dense = False
        state_dependent = False
        diagonal = {}
        for i, j, d, e in itertools.product(range(n_eq), range(n_st),
                                            range(n_d), range(n_d)):
            a = sp.sympify(sym_A[i, j, d, e])
            if a == 0:
                continue
            if a.free_symbols & state_syms:
                state_dependent = True
            if i == j and d == e:
                diagonal[i] = sp.sympify(diagonal.get(i, 0)) + a
            else:
                needs_dense = True
        return needs_dense, state_dependent, diagonal

    def _build_reconstruction(self, mesh, symbolic_model):
        """Build ghost-cell-free face reconstruction."""
        from zoomy_core.fvm.reconstruction import (
            ConstantReconstructionV2, LSQMUSCLReconstruction,
            PositivityPreservingLSQMUSCL,
        )
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            # ``positivity`` routing (REQ-152):
            #   * ``"zhang_shu"`` — A-PRIORI Xing–Zhang–Shu cell-mean cap: scale
            #     the depth-row deviation so the reconstructed h stays ≥ 0 at
            #     every face (conservative — the mean is untouched).  Guarantees
            #     h ≥ 0 only under CFL ≤ 1/(2k+1).
            #   * ``"mood"`` — A-POSTERIORI MOOD (matches jax/dmplex): the plain
            #     LSQ-MUSCL reconstruction (force_o1-aware) rides at the run CFL;
            #     ``HyperbolicSolver.step`` re-steps troubled cells at order 1
            #     (constant reconstruction) so positivity comes from the
            #     order-1 Xing–Zhang lemma — NO deviation cap, NO depth clamp.
            positivity = getattr(self.nsm.reconstruction, "positivity", "")
            h_idx = _var_index(symbolic_model, "h")
            if positivity == "zhang_shu" and h_idx is not None:
                return PositivityPreservingLSQMUSCL(
                    mesh, dim, h_index=h_idx,
                    limiter=self.nsm.reconstruction.limiter)
            return LSQMUSCLReconstruction(
                mesh, dim, limiter=self.nsm.reconstruction.limiter)
        return ConstantReconstructionV2(mesh, dim)

    def get_flux_operator(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        dim = symbolic_model.dimension

        numerics = self._build_numerics(symbolic_model)
        # ``local_max_abs_eigenvalue`` is a REAL expression of the model's
        # normal-projected eigenvalues — the closed-form spectrum, or the opaque
        # numerical ``eigenvalues`` kernel when the model has none (REQ-167) —
        # so the numerical flux lowers directly through the UserFunctions table.
        # Let the numerics declare any extra backend kernels it needs (e.g.
        # the Roe |A| dissipation), merged in before lambdification.
        extra_kernels = getattr(numerics, "runtime_kernels", None)
        if callable(extra_kernels):
            for _kname, _kfn in extra_kernels(symbolic_model).items():
                NumpyRuntimeSymbolic.module[_kname] = _kfn
        runtime_numerics = numerics.to_runtime_numpy()
        use_batched = bool(getattr(numerics, "supports_batched_faces", False))

        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        normals_arr = mesh.face_normals[:dim, :]
        face_volumes = mesh.face_volumes
        cell_volumes = mesh.cell_volumes
        n_vars = symbolic_model.n_variables
        has_aux = symbolic_model.n_aux_variables > 0

        # Build reconstruction (ghost-cell-free)
        reconstruct = self._build_reconstruction(mesh, symbolic_model)
        # A-posteriori MOOD wiring (REQ-152): whether ``positivity="mood"`` is
        # requested AND the reconstruction honours a per-cell ``force_o1``
        # demotion mask (zero its slopes → constant reconstruction).  Read by
        # ``step`` to run the a-posteriori re-step; stored here so IMEX / coupled
        # subclasses that reuse this flux operator inherit the same corrector.
        self._support_o1 = bool(getattr(reconstruct, "supports_force_o1", False))
        self._free_surface_h_index = _var_index(symbolic_model, "h")
        self._mood = (
            getattr(self.nsm.reconstruction, "positivity", "") == "mood")

        # Runtime non-conservative matrix B(Q) for the cell-interior NCP
        # integral (path-conservative consistency at order ≥ 2).  Built only
        # when a higher-order reconstruction carries a slope — order 1 has none.
        ncm = None
        if self.nsm.reconstruction.order >= 2:
            from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
            ncm = NumpyRuntimeModel.from_system_model(
                symbolic_model).nonconservative_matrix

        # Bernoulli moving-equilibrium WB: reconstruct interface states to the
        # common bed b* preserving (q, H(s)); the flux + moment NCP act on the
        # reconstructed states and the conservative-flux-jump source (below)
        # well-balances the moving steady state.  Built once.
        _bern = None
        _eqr = getattr(self.sm, "equilibrium_reconstruction", "none")
        # Unified WB wiring: audusse AND bernoulli go through the SAME hook
        # (interface reconstruction + well-balancing source on the standard
        # NonconservativeRusanov numerics).  'audusse' is dimension-agnostic
        # (face-based HR + the momentum-only PRESSURE-jump source).  The MOVING
        # bernoulli reconstruction is 1-D for now (its per-streamline Bernoulli
        # head is not yet a 2-D velocity vector).
        if _eqr in ("audusse", "bernoulli", "projected_bernoulli"):
            if dim != 1 and _eqr != "audusse":
                raise NotImplementedError(
                    "bernoulli moving-equilibrium WB is 1-D only (v1); "
                    "use equilibrium_reconstruction='audusse' for 2-D lake-at-rest.")
            from zoomy_core.fvm.bernoulli_wb import (
                build_bernoulli_config, reconstruct as _bern_recon)
            _bern = build_bernoulli_config(symbolic_model, mode=_eqr)

        # Build diffusion operators (ghost-cell-free)
        self._diffusion_ops = self._build_diffusion_operators(mesh, symbolic_model, dim, n_vars)

        # Precompute face index sets — split into interior and boundary
        bf_face_set = set(mesh.boundary_face_face_indices)
        interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        boundary_faces = mesh.boundary_face_face_indices.copy()
        iA_int = iA[interior_faces]
        iB_int = iB[interior_faces]
        # THIS-side cell at boundary faces (for reconstruction + flux accumulation).
        # Use face_cells[0] (guaranteed by normalization), NOT boundary_face_cells
        # (which may be remapped to the opposite cell for periodic BCs).
        iInner_bnd = iA[boundary_faces]  # = face_cells[0] at boundary faces
        cvA_int = cell_volumes[iA_int]
        cvB_int = cell_volumes[iB_int]
        cvInner_bnd = cell_volumes[iInner_bnd]
        fv_int = face_volumes[interior_faces]
        fv_bnd = face_volumes[boundary_faces]

        # Per-boundary-face indices into the indexed BC function +
        # precompute d_face.  No Python BC-object list — the BC is
        # ``runtime_model.boundary_conditions(bc_idx, time, position,
        # distance, Q_cell, Qaux_cell, parameters, normal) → q_face``.
        n_bf = mesh.n_boundary_faces
        bf_cells = mesh.boundary_face_cells
        bf_fidx = mesh.boundary_face_face_indices
        bc_indices = boundary_face_branches(
            mesh, getattr(symbolic_model, "_bc_source", None)
            or getattr(model, "_bc_source", None))
        bc_fn = model.boundary_conditions
        face_centers = mesh.face_centers
        # LOCAL algebraic aux map (e.g. ``hinv = kp_hinv(h)``).  Used to keep
        # the GHOST (right) side of a boundary face's aux CONSISTENT with its
        # ghost STATE: the reconstruction/BC fills the right state ``Q_R`` (the
        # opposite cell for a periodic wrap, the ghost state for a Dirichlet /
        # extrapolation BC) but the aux was the inner cell's for both sides.
        # With a state-dependent aux this desyncs ``aux_R`` from ``Q_R`` so the
        # Rusanov dissipation stops telescoping across the wrap face (the
        # periodic mass leak).  Recomputing it from ``Q_R`` via the model's own
        # ``update_aux_variables`` is generic for every BC type; the non-local
        # derivative-aux rows are identity-passthrough here and keep the inner
        # value (best available — a single ghost state has no stencil).
        local_aux_fn = getattr(model, "update_aux_variables", None)

        def _ghost_aux(q_right, qaux_inner, parameters, time, position=_ORIGIN3):
            """Aux consistent with the ghost state ``q_right``.  REQ-185:
            ``update_aux_variables`` is lowered as ``(Q, Qaux, p, t, x)``; pass
            the current ``time`` and the (boundary-face) position vector."""
            if local_aux_fn is None or not has_aux:
                return qaux_inner
            return np.asarray(
                local_aux_fn(q_right, qaux_inner, parameters, time, position),
                dtype=float).reshape(-1)

        d_face = np.array([
            np.linalg.norm(
                mesh.face_centers[bf_fidx[i], :dim] - mesh.cell_centers[:dim, bf_cells[i]]
            )
            for i in range(n_bf)
        ]) if n_bf > 0 else np.array([])

        # Faces remapped by ``resolve_periodic_bcs`` carry the OPPOSITE-side
        # cell in ``boundary_face_cells`` (≠ face_cells[0], the this-side
        # cell).  The Periodic kernel is a pass-through, so the face value
        # is that opposite cell's state; feeding it Q_L in step 3 below
        # would silently turn the wrap back into extrapolation.
        periodic_bf = (np.asarray(bf_cells) != np.asarray(iA[bf_fidx])
                       if n_bf > 0 else np.zeros(0, dtype=bool))

        # Store BC metadata for IMEX access — both kernels live on the
        # runtime; diffusion path uses the gradient one.
        self._bc_indices = bc_indices
        self._bc_fn = bc_fn
        self._bc_grad_fn = model.boundary_gradients
        self._bf_cells = bf_cells
        self._bf_fidx = bf_fidx
        self._d_face = d_face
        self._n_bf = n_bf

        support_o1 = bool(getattr(reconstruct, "supports_force_o1", False))
        # Opt-in (mirrors ``support_o1`` above): only a reconstruction that
        # wraps the model's OWN WB forward/inverse maps (``PrimitiveReconstruction``)
        # declares ``needs_aux`` — every other reconstruction class is
        # untouched by the block below.
        support_aux = bool(getattr(reconstruct, "needs_aux", False))

        def flux_operator(dt, time, Q, Qaux, parameters, dQ, force_o1=None):
            dQ = np.zeros((n_vars, nc))

            # 1. Compute boundary face values via the indexed BC kernel.
            bf_values = np.zeros((n_vars, n_bf))
            aux_bf = (np.zeros((Qaux.shape[0], n_bf), dtype=float)
                      if support_aux else None)
            for i_bf in range(n_bf):
                q_inner = Q[:, bf_cells[i_bf]]
                qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else _EMPTY_AUX
                fidx = bf_fidx[i_bf]
                normal = normals_arr[:, fidx]
                position = face_centers[fidx, :]
                bf_values[:, i_bf] = bc_fn(
                    bc_indices[i_bf], time, position, d_face[i_bf],
                    q_inner, qaux_inner, parameters, normal,
                )
                if support_aux:
                    # ``bf_values[:, i_bf]`` is a BC-TRANSFORMED state, not
                    # the inner cell's own (a Dirichlet/characteristic BC can
                    # prescribe a different h) — an algebraic aux (e.g. the
                    # KP-desingularised ``hinv = 1/h``) must be recomputed
                    # FROM IT, exactly like the ghost-state aux ``_ghost_aux``
                    # already recomputes for the post-reconstruction BC
                    # override below.  Reusing ``qaux_inner`` unchanged would
                    # silently divide by the WRONG h at this boundary face.
                    aux_bf[:, i_bf] = _ghost_aux(
                        bf_values[:, i_bf], qaux_inner, parameters, time,
                        position)

            # 2. Reconstruct (uses bf_values for limiter bounds + Q_R placeholder).
            # ``force_o1`` (a-posteriori MOOD re-step, when the reconstruction
            # supports it) zeroes the slopes of flagged cells → constant
            # reconstruction; ``None`` on every normal / order-1 path.
            o1kw = {"force_o1": force_o1} if support_o1 else {}
            # ``Qaux`` / ``aux_bf`` — only a reconstruction that declared
            # ``needs_aux`` (PrimitiveReconstruction) receives them; the model's
            # own declared ``aux_state`` drives their row layout end to end, so
            # a model with zero aux rows degrades to ``(0, n)`` arrays here.
            auxkw = {"Qaux": Qaux, "Qaux_bf": aux_bf} if support_aux else {}
            Q_L, Q_R = reconstruct(Q, bf_values, **o1kw, **auxkw)

            # 3. Override Q_R at boundary faces with BC(Q_L) — except
            # periodic faces, whose face value is the opposite-side cell
            # state (already in bf_values via the remapped bf_cells).
            for i_bf in range(n_bf):
                fidx = bf_fidx[i_bf]
                if periodic_bf[i_bf]:
                    Q_R[:, fidx] = Q[:, bf_cells[i_bf]]
                    continue
                qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else _EMPTY_AUX
                normal = normals_arr[:, fidx]
                position = face_centers[fidx, :]
                Q_R[:, fidx] = bc_fn(
                    bc_indices[i_bf], time, position, d_face[i_bf],
                    Q_L[:, fidx], qaux_inner, parameters, normal,
                )

            # 3c. Bernoulli moving-equilibrium WB reconstruction (all faces):
            # replace interface states by their reconstruction onto the common
            # bed b* preserving (q, H(s)); stash the conservative-flux-jump
            # source Fc(Q)−Fc(Q*) to add during the face scatter below.
            _bern_src = None
            if _bern is not None:
                _bi = _bern["b"]
                _bstar = np.maximum(Q_L[_bi, :], Q_R[_bi, :])
                _QL0, _QR0 = Q_L.copy(), Q_R.copy()
                Q_L = _bern_recon(Q_L, _bstar, _bern)
                Q_R = _bern_recon(Q_R, _bstar, _bern)
                _fc = _bern["fc"]
                # conservative-flux-jump source Fc·n (projected on the per-face
                # normal -> dimension-agnostic; 1-D recovers the direction-0 flux)
                _bern_src = (_fc(_QL0, normals_arr) - _fc(Q_L, normals_arr),
                             _fc(_QR0, normals_arr) - _fc(Q_R, normals_arr))

            # 3b. Cell-interior non-conservative integral (path-conservative,
            # order ≥ 2): ∫_cell B(Q)·∂_x Q dx ≈ B(Q_c)·s_c, with B evaluated
            # at the cell-centre state and s_c the limited reconstruction slope
            # (depth row = ∂_x η − ∂_x b for free-surface).  The face
            # fluctuations below carry only the inter-cell jump; this is the
            # intra-cell smooth part.  Skipped exactly at order 1 (slope = 0).
            # REQUIRED for well-balancing at order ≥ 2 — it telescopes the
            # reconstructed hydrostatic-pressure variation against the bed
            # slope.  The |cell| factor cancels the per-unit-volume residual
            # normalisation, so there is no division by cell volume.
            grad = getattr(reconstruct, "_limited_grad", None)
            if ncm is not None and grad is not None and grad.any():
                for c in range(nc):
                    qa_c = Qaux[:, c] if has_aux else _EMPTY_AUX
                    B_c = np.asarray(ncm(Q[:, c], qa_c, parameters), dtype=float)
                    if B_c.ndim == 2 and dim == 1:
                        B_c = B_c[..., None]
                    for d in range(dim):
                        dQ[:, c] -= B_c[:, :, d] @ grad[:, d, c]

            # 4. BATCHED face Riemann solves: the lambdified kernels (whose
            # eigenvalue-based dissipation broadcasts over a trailing
            # face axis) — one call per kernel instead of one per face.
            # Scatter with np.add.at (unbuffered) so repeated cell indices
            # accumulate correctly on unstructured meshes.  Numerics classes
            # opt IN via ``supports_batched_faces`` (verified bit-identical
            # for the plain NCP-Rusanov; the HR variants keep the loop until
            # their kernels are batch-validated).

            # 4a. Interior faces — both cells valid
            if use_batched and len(interior_faces):
                qA = Q_L[:, interior_faces]
                qB = Q_R[:, interior_faces]
                qauxA = Qaux[:, iA_int] if has_aux else _EMPTY_AUX2
                qauxB = Qaux[:, iB_int] if has_aux else _EMPTY_AUX2
                n_f = normals_arr[:, interior_faces]
                fluct = np.asarray(runtime_numerics.numerical_fluctuations(
                    qA, qB, qauxA, qauxB, parameters, n_f), dtype=float
                ).reshape(2 * n_vars, -1)
                # ``numerical_flux`` returns the flux alone — (n_vars) rows.
                # dt comes per CELL from ``compute_max_abs_eigenvalue``.
                num_flux = np.asarray(runtime_numerics.numerical_flux(
                    qA, qB, qauxA, qauxB, parameters, n_f), dtype=float
                ).reshape(n_vars, -1)
                Dp = fluct[:n_vars]
                Dm = fluct[n_vars:]
                np.add.at(dQ.T, iA_int,
                          (-(num_flux + Dm) * fv_int / cvA_int).T)
                np.add.at(dQ.T, iB_int,
                          (-(-num_flux + Dp) * fv_int / cvB_int).T)
                if _bern_src is not None:
                    sL, sR = _bern_src
                    np.add.at(dQ.T, iA_int, (-sL[:, interior_faces] * fv_int / cvA_int).T)
                    np.add.at(dQ.T, iB_int, (sR[:, interior_faces] * fv_int / cvB_int).T)

            # 4b. Boundary faces — one inner cell only
            if use_batched and len(boundary_faces):
                qA = Q_L[:, boundary_faces]
                qB = Q_R[:, boundary_faces]
                qauxI = Qaux[:, iInner_bnd] if has_aux else _EMPTY_AUX2
                # Right-side aux consistent with the ghost state ``qB``
                # (recomputed from ``Q_R`` so a wrapped/Dirichlet ghost carries
                # ``aux_R = update_aux_variables(Q_R)``, not the inner aux).
                if has_aux and local_aux_fn is not None:
                    qauxR = np.empty_like(qauxI)
                    for j in range(qB.shape[1]):
                        qauxR[:, j] = _ghost_aux(
                            qB[:, j], qauxI[:, j], parameters, time,
                            face_centers[boundary_faces[j], :])
                else:
                    qauxR = qauxI
                n_f = normals_arr[:, boundary_faces]
                fluct = np.asarray(runtime_numerics.numerical_fluctuations(
                    qA, qB, qauxI, qauxR, parameters, n_f), dtype=float
                ).reshape(2 * n_vars, -1)
                num_flux = np.asarray(runtime_numerics.numerical_flux(
                    qA, qB, qauxI, qauxR, parameters, n_f), dtype=float
                ).reshape(n_vars, -1)
                Dm = fluct[n_vars:]
                np.add.at(dQ.T, iInner_bnd,
                          (-(num_flux + Dm) * fv_bnd / cvInner_bnd).T)
                if _bern_src is not None:
                    sL, _sR = _bern_src
                    np.add.at(dQ.T, iInner_bnd,
                              (-sL[:, boundary_faces] * fv_bnd / cvInner_bnd).T)

            # 4-loop. Per-face fallback for numerics without batch support.
            if not use_batched:
                for fi in range(len(interior_faces)):
                    f = interior_faces[fi]
                    qA = Q_L[:, f]
                    qB = Q_R[:, f]
                    qauxA = Qaux[:, iA_int[fi]] if has_aux else _EMPTY_AUX
                    qauxB = Qaux[:, iB_int[fi]] if has_aux else _EMPTY_AUX
                    n = normals_arr[:, f]
                    fluct = np.asarray(
                        runtime_numerics.numerical_fluctuations(
                            qA, qB, qauxA, qauxB, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    num_flux = np.asarray(
                        runtime_numerics.numerical_flux(
                            qA, qB, qauxA, qauxB, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    Dp = fluct[:n_vars]
                    Dm = fluct[n_vars:]
                    dQ[:, iA_int[fi]] -= (num_flux + Dm) * fv_int[fi] / cvA_int[fi]
                    dQ[:, iB_int[fi]] -= (-num_flux + Dp) * fv_int[fi] / cvB_int[fi]
                for bi in range(len(boundary_faces)):
                    f = boundary_faces[bi]
                    qA = Q_L[:, f]
                    qB = Q_R[:, f]
                    qauxA = Qaux[:, iInner_bnd[bi]] if has_aux else _EMPTY_AUX
                    # Right-side aux consistent with the ghost state qB.
                    qauxR = (_ghost_aux(qB, qauxA, parameters, time,
                                        face_centers[f, :])
                             if has_aux else _EMPTY_AUX)
                    n = normals_arr[:, f]
                    fluct = np.asarray(
                        runtime_numerics.numerical_fluctuations(
                            qA, qB, qauxA, qauxR, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    num_flux = np.asarray(
                        runtime_numerics.numerical_flux(
                            qA, qB, qauxA, qauxR, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    Dm = fluct[n_vars:]
                    dQ[:, iInner_bnd[bi]] -= (num_flux + Dm) * fv_bnd[bi] / cvInner_bnd[bi]

            # 5. Explicit diffusion with boundary contributions
            if self._diffusion_ops is not None and self._diffusion_in_flux:
                bf_grads = _compute_bf_face_gradients(
                    Q, Qaux, bc_indices, self._bc_grad_fn, bf_cells, bf_fidx,
                    d_face, normals_arr, face_centers, n_bf, n_vars, has_aux,
                    time, parameters,
                )
                fns = getattr(self, "_diffusivity_fns", {})
                for v, diff_op in self._diffusion_ops.items():
                    nu_cell = None
                    if v in fns:                 # state-dependent diffusivity −A[v,v,d,d]
                        val = fns[v](*Q, *Qaux, *parameters)
                        nu_cell = np.broadcast_to(
                            np.asarray(val, dtype=float), (Q.shape[1],)).astype(float)
                    dQ[v, :] += diff_op.explicit_with_bc(
                        Q[v, :], bf_grads[v], nu_cell=nu_cell)

            return dQ
        return flux_operator

    # -- Source operator -----------------------------------------------

    def get_compute_source(self, mesh, model):
        """Compound source operator: evaluates both the implicit
        ``source`` slot and the explicit ``source_explicit`` slot at
        the current state and sums them.

        This backend is explicit-only (FE/SSP-RK), so the IMEX split
        a Firedrake backend respects is collapsed here — *all* source
        contributions go to the RHS evaluated at the current state.
        Backends that genuinely support IMEX (e.g. Firedrake) keep
        ``source`` in the source-step Newton and ``source_explicit``
        in the convective step.
        """
        has_explicit = hasattr(model, "source_explicit")
        # REQ-185: the ``source`` operator is lowered as
        # ``(Q, Qaux, p, t, [dt], x)`` — time and the position VECTOR ``x``
        # (cell centres, ``(dim, n_cells)``) are always passed; ``dt`` is passed
        # explicitly when ``source_needs_dt`` (i.e. ``dt`` is a first-class
        # source arg rather than a baked model parameter).  Cell centres are
        # captured once here.  ``source_explicit`` is NOT in this cut's scope and
        # keeps its ``(Q, Qaux, p)`` signature (padded coordinate args).
        source_needs_dt = getattr(model, "source_needs_dt", False)
        # INNER cells only.  ``mesh.cell_centers`` is GHOST-PADDED while the
        # ``Q``/``Qaux`` handed to the source are not (measured on a 200-cell
        # 1-D mesh: Q (4, 200), Qaux (5, 200), cell_centers (3, 202)), so a
        # position-dependent source mixed 202-length position terms with
        # 200-length state terms and died with
        #     operands could not be broadcast together with shapes (200,) (202,)
        # That is every MMS case on this backend -- the manufactured source is
        # position-dependent by construction.
        _cc = getattr(mesh, "cell_centers", None)
        _ni = getattr(mesh, "n_inner_cells", None)
        cell_centers = (_cc[:, :_ni] if _cc is not None and _ni is not None
                        and getattr(_cc, "ndim", 0) == 2 and _cc.shape[1] != _ni
                        else _cc)

        def compute_source(dt, time, Q, Qaux, parameters, dQ):
            if source_needs_dt:
                dQ = model.source(Q[:, :], Qaux[:, :], parameters,
                                  time, dt, cell_centers)
            else:
                dQ = model.source(Q[:, :], Qaux[:, :], parameters,
                                  time, cell_centers)
            if has_explicit:
                dQ = dQ + model.source_explicit(Q[:, :], Qaux[:, :], parameters)
            return dQ
        return compute_source

    def get_compute_source_jacobian_wrt_variables(self, mesh, model):
        def compute(dt, Q, Qaux, parameters, dQ):
            return model.source_jacobian_wrt_variables(Q, Qaux, parameters)
        return compute

    # -- Setup / Step / Run / Solve ------------------------------------

    def _compute_face_inradius(self, mesh):
        """Per-face inradius for the LOCAL CFL: each face carries the min of
        its two adjacent cells' inradii (boundary faces: the inner cell's),
        so ``compute_dt`` pairs every face's own size with its own local
        wave speed — never the global smallest cell with the global fastest
        wave.  Shared by every solver setup (base, IMEX, splitting)."""
        nc = mesh.n_inner_cells
        inner_face_mask = (mesh.face_cells[0] < nc) & (mesh.face_cells[1] < nc)
        face_inradius = np.full(mesh.n_faces, np.inf)
        face_inradius[inner_face_mask] = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, inner_face_mask]],
            mesh.cell_inradius[mesh.face_cells[1, inner_face_mask]],
        )
        if (~inner_face_mask).any():
            face_inradius[~inner_face_mask] = mesh.cell_inradius[
                mesh.face_cells[0, ~inner_face_mask]]
        return face_inradius

    def setup_simulation(self, mesh, model, write_output=True):
        """Build all operators once. Stores simulation state on self.

        ``model`` may be a :class:`Model`, a :class:`SystemModel`, or a
        :class:`NumericalSystemModel`.  Plain models are auto-promoted
        through Model → SystemModel → NSM internally; when an NSM is
        passed directly its numerical slots (reconstruction order,
        limiter, regularization) seed the solver attributes and its
        auto-resolved LSQ degree drives the mesh stencil.
        """
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        # REQ-190: the timestep cap is a STANDARD NSM parameter.  Fill the
        # adaptive strategy's dt_max from ``nsm.dt_max`` unless the caller
        # passed an explicit cap (explicit wins); a wave-free (fully-dry)
        # domain then steps at dt_max instead of leaking ``inf``.
        timestepping.apply_default_dt_max(self.compute_dt, nsm.dt_max)
        mesh = ensure_lsq_mesh(mesh, nsm)
        # Periodic-BC topology resolution needs the ``BoundaryConditions``
        # object (it iterates ``.boundary_conditions_list`` for tagged
        # periodic pairs).  Done here, *before* normalising to a
        # SystemModel.  Production Models carry the object directly;
        # declarative SystemModels retain it as ``_bc_source`` (set by
        # ``attach_boundary_conditions``) — without this fallback a
        # Periodic BC on the declarative path was SILENTLY ignored and
        # the boundary acted as extrapolation (waves leave, mass leaks).
        bcs_obj = None
        if source_model is not None:
            bcs_obj = source_model.boundary_conditions
        else:
            bcs_obj = getattr(nsm.sm, "_bc_source", None)
        if hasattr(mesh, "resolve_periodic_bcs") and bcs_obj is not None:
            mesh.resolve_periodic_bcs(bcs_obj)
        model = nsm.sm
        self.sm = model
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        # Store simulation state
        self._sim_mesh = mesh
        self._sim_model = model
        self._sim_parameters = parameters
        self._sim_Q = Q
        self._sim_Qaux = Qaux
        self._sim_time = 0.0

        # Build operators (once) — no ghost-cell BC operator needed
        self._sim_compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
        self._sim_flux_operator = self.get_flux_operator(mesh, model)
        self._sim_source_operator = self.get_compute_source(mesh, model)

        # Apply initial variable updates (no step taken yet → dt = 0; clamps
        # are dt-independent, the corrector path is never on the setup leg).
        self._sim_Q = self.update_q(self._sim_Q, Qaux, mesh, model,
                                    parameters, 0.0)

        self._sim_face_inradius = self._compute_face_inradius(mesh)

        # Output setup
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            io.write_mesh_to_hdf5(output_hdf5_path, mesh,   # self-describing output
                                  names=[str(s) for s in self.nsm.state])
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            self._sim_save_fields = lambda time, time_stamp, i_snapshot, Q, Qaux: i_snapshot

    def _mood_cascade(self, rk2, Qnew, h_idx, mesh, dt):
        """A-posteriori MOOD repair: demote the troubled CELLS, then, if that is
        not enough, the whole step — ONE design shared with the jax backend.

        Two levels, and no iteration:

        1. flag the inadmissible cells (``h < 0`` or non-finite) and re-step
           with those reconstructing at order 1;
        2. if any remain, re-step with the WHOLE domain at order 1.  That is the
           ``d = 0`` end of the Clain-Diot-Loubère cascade — the monotone
           first-order scheme, positivity-preserving under its CFL bound — so it
           terminates the repair by construction rather than by a growth
           argument.

        This REPLACES the EPD_2 stencil cascade (cid 210/214), which demoted a
        troubled cell together with its face neighbours and iterated.  EPD_2 is
        the right reading of Clain, Diot & Loubère (*JCP* 230(10):4028-4050,
        2011) Remark 11 for a smooth/shock problem, where the neighbour carries
        a polynomial whose degree must be coupled across the shared face.  At a
        WET/DRY front it is actively harmful, because the neighbour it demotes
        is dry: measured on jax, EPD_2 cost 5.8x the compute (230 steps vs 40)
        at 46x smaller dt and left only 216 of 9216 cells exactly dry against
        3408, by pushing a numerical film of h ~ 1e-11 carrying momentum
        ~2000x disproportionate to its depth.  Mass and positivity stayed
        perfect throughout, which is why nothing caught it.

        Both backends now carry this design, so a numpy/jax comparison measures
        the MODEL rather than two different repair strategies.  numpy RAISES
        where jax commits the first-order state — jax cannot raise from inside
        ``jit``, and the order-1 march is provably admissible, so committing it
        is honest there; here the inadmissible state is kept as evidence and no
        depth is ever clamped.
        """
        n = Qnew.shape[1]

        def _troubled(Qc):
            return (Qc[h_idx, :] < 0.0) | ~np.isfinite(Qc).all(axis=0)

        troubled = _troubled(Qnew)
        if not troubled.any():
            return Qnew

        # 1. demote the troubled cells and re-step.
        Q_wc = rk2(troubled)
        if not _troubled(Q_wc).any():
            return Q_wc

        # 2. parachute: whole-STEP order 1.  This is the d = 0 end of the
        #    Clain-Diot-Loubere cascade -- the monotone first-order scheme,
        #    positivity-preserving under its CFL bound -- so it terminates the
        #    repair by construction.
        Q_all = rk2(np.ones(n, dtype=bool))
        if not _troubled(Q_all).any():
            return Q_all

        bad = np.flatnonzero(_troubled(Q_all))
        raise FloatingPointError(
            f"MOOD repair exhausted at dt={dt:g}: {bad.size} cell(s) remain "
            f"inadmissible after a WHOLE-STEP order-1 re-step "
            f"(cells {bad[:8].tolist()}{'…' if bad.size > 8 else ''}, "
            f"min h = {np.nanmin(Q_all[h_idx, bad]):.6e}).\n"
            "The first-order scheme is positivity-preserving under its CFL "
            "bound, so this is a real defect, not a tolerance: check the CFL, "
            "the boundary conditions, and any operation that gates the wave "
            "speed used for the flux dissipation.  Nothing is clamped here on "
            "purpose — the inadmissible state is the evidence.")


    def step(self, dt):
        """One explicit timestep (ghost-cell-free).

        BCs are evaluated inline inside the flux operator — no separate
        ghost-cell filling step. Loops are split into interior/boundary.

        O1 (RK1): flux → advance
        O2 (RK2/Heun): flux → advance → flux → average
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        mesh = self._sim_mesh
        model = self._sim_model
        flux = self._sim_flux_operator
        source = self._sim_source_operator

        def rhs(t, Q_in, force_o1=None):
            """Total RHS = flux + source.  ``force_o1`` (a-posteriori MOOD)
            demotes flagged cells to order 1 inside the *flux* reconstruction
            only; the source is slope-free so it is unaffected."""
            dQ_f = flux(dt, t, Q_in, Qaux, parameters, np.zeros_like(Q_in),
                        force_o1)
            dQ_s = source(dt, t, Q_in, Qaux, parameters, np.zeros_like(Q_in))
            return dQ_f + dQ_s

        if self.nsm.reconstruction.order >= 2:
            # SSP-RK2 (Heun) — flux + source per stage.  ``force_o1`` (or None)
            # is threaded through BOTH stages so the a-posteriori MOOD re-run
            # demotes the SAME cells in both.
            Q0 = np.array(Q)

            def _rk2(force_o1):
                Q1 = Q + dt * rhs(time_now, Q, force_o1)
                Q2 = Q1 + dt * rhs(time_now + dt, Q1, force_o1)
                return 0.5 * (Q0 + Q2)

            Qnew = _rk2(None)
            # A-posteriori MOOD (REQ-152): detect troubled cells on the order-2
            # candidate — PAD (h < 0) + CAD (non-finite) — and re-run with the
            # troubled STENCIL demoted to order 1, iterating until the candidate
            # is admissible.  See :meth:`_mood_cascade` for why the stencil and
            # not the cell.  NO depth is ever clamped/floored.
            h_idx = getattr(self, "_free_surface_h_index", None)
            if (getattr(self, "_mood", False)
                    and getattr(self, "_support_o1", False)
                    and h_idx is not None):
                Qnew = self._mood_cascade(_rk2, Qnew, h_idx, mesh, dt)
        else:
            # RK1
            Qnew = Q + dt * rhs(time_now, Q)

        # Variable updates (clamp, etc.)
        Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters, dt)

        # Auxiliary variable updates
        Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model,
                                    parameters, time_now, dt)

        # Commit new state
        self._sim_Q = Qnew
        self._sim_Qaux = Qauxnew

    def run_simulation(self):
        """Time loop: compute_dt -> step -> post_step -> output."""
        t_start = gettime()
        time_now = 0.0
        next_write_at = 0.0
        i_snapshot = 0.0
        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        iteration = 0

        # Initial snapshot, the same convention as the jax and DAE loops: the
        # first record of the store is the initial condition at t = 0, not the
        # state after the first step.
        i_snapshot = self._sim_save_fields(
            time_now, next_write_at, i_snapshot, self._sim_Q, self._sim_Qaux,
        )
        last_written = time_now
        next_write_at += dt_snapshot

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, self._sim_parameters,
                self._sim_face_inradius, self._sim_compute_max_abs_eigenvalue,
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                break

            self._sim_time = time_now
            self.step(dt)

            time_now += dt
            iteration += 1

            if time_now >= next_write_at:
                i_snapshot = self._sim_save_fields(
                    time_now, next_write_at, i_snapshot, self._sim_Q, self._sim_Qaux,
                )
                last_written = time_now
                next_write_at += dt_snapshot

            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {iteration}, time: {time_now:.6f}, "
                    f"dt: {dt:.6f}, next write at time: {next_write_at:.6f}"
                )

        # The end state. The write schedule is accumulated by addition, so
        # its last entry can land a roundoff above time_end and the loop
        # ends with the final state unwritten (a 40-snapshot march to 0.5 s
        # ended its store at 0.487 s).
        if time_now > last_written:
            i_snapshot = self._sim_save_fields(
                time_now, time_now, i_snapshot, self._sim_Q, self._sim_Qaux,
            )

        self._sim_time = time_now
        logger.info(f"Finished simulation with in {gettime() - t_start:.3f} seconds")
        return self._sim_Q, self._sim_Qaux

    def solve(self, mesh, model, write_output=True):
        """Convenience: setup_simulation + run_simulation."""
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()


# -- Free-surface variant ------------------------------------------------------

def _build_free_surface_numerics(symbolic_model):
    """Build positive Rusanov for free-surface models.  ``h`` and
    ``b`` are auto-located in ``model.variables`` or
    ``model.aux_variables`` via :class:`FieldHandle` — no explicit
    ``field_map`` is needed."""
    scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
    return PositiveNonconservativeRusanov(
        symbolic_model,
        scaled_q_indices=scaled_q_indices,
    )


class _SurfaceReconAdapter:
    """Adapt :class:`SurfaceReconstruction` to the explicit flux operator's
    ``reconstruct(Q, bf) → (Q_L, Q_R)`` interface.

    ``SurfaceReconstruction`` reconstructs the free-surface elevation
    ``η = h + b`` (flat at lake-at-rest → zero limited slope) and returns
    ``(Q_L, Q_R, b_L, b_R)`` with the per-side depth recovered as ``h = η − b``;
    the explicit flux operator wants only ``(Q_L, Q_R)`` and reads the limited
    slope of the *actual* state (depth row ``∂η − ∂b``) off ``_limited_grad``
    for the cell-interior non-conservative integral.  ``b`` lives in the
    conservative state here, so no ``Qaux`` is needed at reconstruction time.
    """

    def __init__(self, surf):
        self.surf = surf
        self._limited_grad = None
        # Forward the a-posteriori MOOD ``force_o1`` capability of the wrapped
        # surface reconstruction (which forwards it to its LSQ base).
        self.supports_force_o1 = bool(
            getattr(surf, "supports_force_o1", False))

    def __call__(self, Q, bf, phi=None, force_o1=None):
        Q_L, Q_R, _b_L, _b_R = self.surf(Q, None, bf, phi=phi,
                                         force_o1=force_o1)
        self._limited_grad = self.surf._limited_grad
        return Q_L, Q_R


class FreeSurfaceFlowSolver(HyperbolicSolver):
    """Explicit FVM for free-surface flows (SWE, SME, VAM).

    Uses positive (hydrostatic reconstruction) Rusanov with wet/dry handling.
    Requires model variables 'b' and 'h'.

    At reconstruction order ≥ 2 it reconstructs the free-surface elevation
    ``η = h + b`` (so the limiter is well-balanced) and the base solver adds
    the cell-interior non-conservative integral — together these give exact
    well-balancing at lake-at-rest to machine precision.
    """

    def _build_numerics(self, symbolic_model):
        nsm_numerics = _nsm_explicit_numerics(self, symbolic_model)
        if nsm_numerics is not None:
            return nsm_numerics
        return _build_free_surface_numerics(symbolic_model)

    def _build_reconstruction(self, mesh, symbolic_model):
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            h_idx = _var_index(symbolic_model, "h")
            b_idx = _var_index(symbolic_model, "b")
            # Well-balanced surface-elevation reconstruction needs b in the
            # conservative state (recover h = η − b per side).  When the model
            # has no in-state bathymetry, fall back to the wet/dry conservative
            # MUSCL (not WB at order 2, but the only sensible option there).
            if b_idx is not None:
                from zoomy_core.fvm.reconstruction import (
                    LSQMUSCLReconstruction, SurfaceReconstruction,
                )
                base = LSQMUSCLReconstruction(
                    mesh, dim, limiter=self.nsm.reconstruction.limiter)
                surf = SurfaceReconstruction(
                    base, h_index=h_idx, b_index=b_idx, b_in_state=True)
                return _SurfaceReconAdapter(surf)
            from zoomy_core.fvm.reconstruction import FreeSurfaceLSQMUSCL
            # The RECONSTRUCTION's wet/dry epsilon — a different consumer from
            # the (now NSM-emitted) CFL spectrum, but the same rule: the value
            # is the model's own ``wet_dry_eps``, never a backend literal.
            from zoomy_core.systemmodel.operations import _DEFAULT_WET_DRY_EPS
            eps_wet = _param_value(symbolic_model, "wet_dry_eps",
                                   default=_DEFAULT_WET_DRY_EPS)
            return FreeSurfaceLSQMUSCL(
                mesh, dim, h_index=h_idx, eps_wet=eps_wet,
                limiter=self.nsm.reconstruction.limiter)
        return super()._build_reconstruction(mesh, symbolic_model)


def _build_roe_numerics(symbolic_model):
    """Path-conservative Roe (matrix |A| dissipation) for free-surface models;
    auto-locates ``h``/``b`` and the depth-scaled momentum rows like the
    Rusanov variant."""
    from zoomy_core.fvm.riemann_solvers import NonconservativeRoe
    scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
    return NonconservativeRoe(symbolic_model, scaled_q_indices=scaled_q_indices)


class RoeFreeSurfaceFlowSolver(FreeSurfaceFlowSolver):
    """:class:`FreeSurfaceFlowSolver` using the path-conservative Roe scheme
    (matrix ``|A|`` dissipation via runtime numerical eigendecomposition)
    instead of scalar Rusanov dissipation — sharper shocks/contacts at the
    same reconstruction order.  Everything else (hydrostatic reconstruction,
    wet/dry handling, time-stepping) is unchanged."""

    def _build_numerics(self, symbolic_model):
        nsm_numerics = _nsm_explicit_numerics(self, symbolic_model)
        if nsm_numerics is not None:
            return nsm_numerics
        return _build_roe_numerics(symbolic_model)
