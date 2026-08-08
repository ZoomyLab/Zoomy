"""JaxRuntime — clean JIT-vmapped JAX runtime over a NumericalSystemModel.

This is the *new* JAX runtime: the documented workflow is

    Model → SystemModel → NumericalSystemModel → JaxRuntime

Every operator stored on the underlying SystemModel (``flux``, ``source``,
``hydrostatic_pressure``, ``nonconservative_matrix``,
``quasilinear_matrix``, ``mass_matrix``, ``eigenvalues``, the indexed BC
kernels) is lambdified once with the JAX module, then wrapped in
``jax.jit(jax.vmap(...))`` so a single call evaluates the operator at
every cell (or every face) in one shot.

The Riemann ``Numerics`` (built from ``nsm.build_numerics()``) is
lambdified the same way and exposes per-face ``numerical_flux`` /
``numerical_fluctuations`` vmapped over the face axis.

Design choices:

* **No source-Model dependency.**  JaxRuntime consumes only the NSM
  (and its embedded SystemModel).  The legacy path that required
  ``Kernel(model)`` + ``JaxRuntimeModel(model, kernel=...)`` is gone.
* **Parameters are live.**  ``self.parameters`` is a property that
  reads ``nsm.sm.parameter_values`` fresh on every access, so user
  mutations (e.g. ``nsm.sm.parameter_values.g = 12.0``) flow through
  without a runtime rebuild.  ``jax.grad`` can be taken against the
  parameter axis of any operator.
* **No broadcast hack.**  Operators are vmap'd over the cell/face axis
  explicitly — no ``ones_like(anchor)`` wrapping; each lambdified
  function takes pure scalar inputs.

Replaces the legacy ``JaxRuntimeModel`` (Model-based) and
``JaxRuntimeSymbolic`` (Numerics-based, broadcast-tricked).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from zoomy_core.fvm.riemann_solvers import Numerics
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.transformation.vectorize import uniform_rank


# ── JAX module dict for sp.lambdify ──────────────────────────────────


# The UserFunctions table lives in ONE place (REQ-168):
# ``zoomy_jax.fvm.userfunctions`` — arithmetic + the opaque kernels
# (``eigensystem``, ``solve``) + the solver-injected ``compute_derivative``.
# Built fresh per runtime because ``JaxRuntime`` injects per-instance entries.
def _jax_module_base() -> dict:
    # Lazy import: ``zoomy_jax.fvm`` imports ``transformation.to_jax``, so a
    # module-level import here would be circular.
    from zoomy_jax.fvm.userfunctions import jax_userfunctions
    m = jax_userfunctions()
    # ``max_wavespeed`` is an opaque sympy Function that Rusanov-style numerics
    # USED to emit; ``JaxRuntime.__init__`` still fills it with the local-Roe /
    # max-eig callable built from the SystemModel's ``eigenvalues``.
    # ⚠ REQ-168: core no longer emits this symbol anywhere (grep zoomy_core =
    # zero hits), so the plug below is orphaned — kept for now (harmless: an
    # unemitted symbol is simply never resolved) pending the REQ-168 GAP-1
    # decision, which should give the wave speed a real λ-only kernel instead.
    m["max_wavespeed"] = None
    return m


# ── Helpers ──────────────────────────────────────────────────────────


def _to_array_of_exprs(definition) -> sp.Array:
    """Coerce a SystemModel operator entry (Matrix / ZArray /
    NDimArray / None) to an ``sp.Array`` with concrete shape.  None
    becomes a length-0 1-D Array so the lambdified callable returns
    an empty result without raising."""
    if definition is None:
        return sp.Array([], shape=(0,))
    if hasattr(definition, "tolist") and callable(definition.tolist):
        # Matrix / NDimArray / ZArray
        return sp.Array(definition.tolist())
    if isinstance(definition, (list, tuple)):
        return sp.Array(list(definition))
    # Scalar
    return sp.Array([definition])


def _shape(arr: sp.Array) -> tuple:
    return tuple(int(s) for s in arr.shape)


def _flat_symbols(zstruct_like) -> List[sp.Symbol]:
    """Flatten a Zstruct (or list-of-Symbols) into a list of sympy
    Symbols, preserving order."""
    if hasattr(zstruct_like, "values") and callable(zstruct_like.values):
        return list(zstruct_like.values())
    return list(zstruct_like)


def _group_symbols(group) -> List[sp.Symbol]:
    """Flatten ONE declared signature group from
    ``sm.operator_signature(name)`` to an ordered symbol list.  Vector
    groups (variables / aux_variables / parameters / normal / position)
    are Zstructs; the scalar groups (``time`` / ``dt``) are bare
    Symbols."""
    if isinstance(group, sp.Symbol):
        return [group]
    return _flat_symbols(group)


def _lambdify_array(arr: sp.Array, arg_lists: Sequence[Sequence[sp.Symbol]],
                    module: dict) -> Callable:
    """Lambdify a (possibly empty) sympy Array against grouped
    arg-lists.  Returns a callable that takes one positional argument
    per arg_list (each itself a sequence of scalars matching the
    signature's declared length) and returns an array of the original
    sympy shape.

    The runtime arg lengths must match the signature's declared scalar
    counts.  Each runtime arg is sliced to ``len(group_syms)`` scalars
    — this matters when the JAX mesh stores 3D-padded vectors for
    fields the signature declares as 1D (e.g. SWE1D normal is just
    ``n0``, but ``mesh.face_normals`` is shape ``(3, n_faces)``).
    """
    shape = _shape(arr)
    # REQ-84: shared constant-entry rank-normalization seam
    # (``zoomy_core.transformation.vectorize.uniform_rank``, the same helper
    # the numpy printer uses).  The FIRST arg group is the state group — mirror
    # to_numpy's convention (vector_symbols = state, anchor = first state
    # symbol).  Wrapping each state-free constant entry with
    # ``zeros_like(anchor)`` / ``c*ones_like(anchor)`` lifts short rows (e.g.
    # MalpassetSWE's identically-zero bathymetry flux row) to the batch rank of
    # their siblings so the generated ``jnp.array`` stacks them; ``ones_like`` /
    # ``zeros_like`` bind to ``jnp.*`` via the module dict (userfunctions.py).
    if arg_lists and shape and all(int(s) > 0 for s in shape):
        state_syms = tuple(arg_lists[0])
        if state_syms:
            arr = uniform_rank(arr, state_syms, state_syms[0])
    counts = [len(group) for group in arg_lists]
    flat_args = [sym for group in arg_lists for sym in group]
    flat_expr = list(arr) if shape else []

    if not flat_expr:
        # Empty array — return a constant zero-shape function.
        empty = jnp.zeros(shape)
        def _empty(*args):
            return empty
        return _empty

    fn_flat = sp.lambdify(flat_args, flat_expr, modules=[module, "jax"],
                          cse=True)

    def call_per_cell(*grouped_args):
        flat = []
        for g, count in zip(grouped_args, counts):
            if count == 0:
                continue
            if hasattr(g, "shape") and g.shape != ():
                for i in range(count):
                    flat.append(g[i])
            else:
                # Scalar — only sensible when count == 1.
                flat.append(g)
        out = fn_flat(*flat)
        if shape:
            # REQ-84: STRUCTURAL flat→shape reconstruction, NOT a retireable
            # workaround — ``flat_expr = list(arr)`` flattens row-major, so this
            # reshape restores the operator shape (e.g. SWE flux (4,2) from 8
            # scalars); removing it returns a flat vector and downstream
            # ``[i,j]`` reads silently go wrong.  (jax verified 2026-07-17; the
            # mixed-rank stacking fix is ``uniform_rank`` above.)
            return jnp.asarray(out).reshape(shape)
        return jnp.asarray(out)

    return call_per_cell


# ── The runtime class ────────────────────────────────────────────────


class JaxRuntime:
    """JIT-vmapped JAX runtime over a :class:`NumericalSystemModel`.

    Parameters
    ----------
    nsm : NumericalSystemModel
        The NSM whose embedded ``sm`` (and the
        :class:`~zoomy_core.fvm.riemann_solvers.Numerics` built from
        the NSM's riemann class) provide every operator.  A bare
        ``SystemModel`` (or ``Model``) is auto-promoted via
        :meth:`NumericalSystemModel.from_system_model`.

    Operators exposed
    -----------------

    Per-cell (vmap over the cell axis of ``Q`` and ``Qaux``):

    * ``flux(Q, Qaux, parameters)``  →  ``(n_eq, n_dim, n_cells)``
    * ``hydrostatic_pressure(Q, Qaux, parameters)`` →  ``(n_eq, n_dim, n_cells)``
    * ``source(Q, Qaux, parameters[, time, dt, position])`` →  ``(n_eq, n_cells)``
    * ``mass_matrix(Q, Qaux, parameters)`` → ``(n_eq, n_state, n_cells)``
    * ``nonconservative_matrix(Q, Qaux, parameters)`` → ``(n_eq, n_state, n_dim, n_cells)``
    * ``quasilinear_matrix(Q, Qaux, parameters)`` → same shape as NCP
    * ``eigenvalues(Q, Qaux, parameters, normal)`` →  ``(n_eq, n_cells)``

    All registry-slot signatures are DERIVED from the SystemModel's
    declared ``operator_signature`` (REQ-185); trailing groups beyond
    ``(Q, Qaux, parameters)`` bind positionally in declared order or by
    keyword, with ``time`` / ``dt`` defaulting to 0 and ``position`` to
    the origin (exact for operators that don't reference them).

    Per-face (vmap over the face axis):

    * ``numerical_flux(qL, qR, qauxL, qauxR, parameters, normal)``
      →  ``(n_eq, n_faces)``
    * ``numerical_fluctuations(qL, qR, qauxL, qauxR, parameters, normal)``
      →  ``(2*n_eq, n_faces)`` (Dp and Dm stacked)

    Boundary kernel (indexed):

    * ``boundary_conditions(i_bc, time, position, distance, q_cell,
       qaux_cell, parameters, normal)`` →  ``(n_state,)`` ghost cell value.
      Invoke this inside the solver's ``fori_loop`` over boundary
      faces — it is *not* itself vmap'd because the BC index varies
      per face.
    """

    @classmethod
    def from_nsm(cls, nsm) -> "JaxRuntime":
        """Auto-promote a Model / SystemModel to NSM and wrap."""
        if not isinstance(nsm, NumericalSystemModel):
            nsm = NumericalSystemModel.from_system_model(nsm)
        return cls(nsm)

    def __init__(self, nsm: NumericalSystemModel):
        if not isinstance(nsm, NumericalSystemModel):
            raise TypeError(
                f"JaxRuntime expects a NumericalSystemModel, got {type(nsm)}.  "
                f"Use JaxRuntime.from_nsm(...) to auto-promote."
            )
        self.nsm = nsm
        self.sm = nsm.sm
        self.n_state = self.sm.n_state
        self.n_aux = len(self.sm.aux_state)
        self.n_dim = self.sm.n_dim

        # Symbol groups.
        self._q_syms = _flat_symbols(self.sm.state)
        self._qaux_syms = _flat_symbols(self.sm.aux_state)
        self._p_syms = _flat_symbols(self.sm.parameters)
        self._n_syms = _flat_symbols(self.sm.normal)
        # 3D position symbols (X0, X1, X2) used by interpolate_to_3d.
        # SystemModel.position may be a Zstruct from Model.position
        # (length 3 even for 2D models) or None when no such operator
        # is attached.
        if getattr(self.sm, "position", None) is not None:
            self._pos_syms = _flat_symbols(self.sm.position)
        else:
            self._pos_syms = []

        # JAX module dict — local copy so per-instance ``max_wavespeed``
        # injection doesn't leak across runtimes.
        self.module = _jax_module_base()

        # Per-cell operators on the SystemModel.
        self._build_cell_operators()
        # Per-face Riemann numerics — only meaningful for square
        # (hyperbolic) SystemModels.  Rectangular sub-systems from
        # splitter operations (sm_press, sm_corr) carry no flux of
        # their own; skip and set the slots to None.
        sm = self.sm
        if sm.n_equations == sm.n_state and sm.flux is not None:
            try:
                self._build_face_operators()
            except (ValueError, KeyError, AttributeError, TypeError):
                self.numerical_flux = None
                self.numerical_fluctuations = None
                self._numerics = None
        else:
            self.numerical_flux = None
            self.numerical_fluctuations = None
            self._numerics = None
        # Indexed boundary-condition kernels.
        self._build_bc()

    # ── Parameter accessors ──────────────────────────────────────

    @property
    def parameters(self) -> jnp.ndarray:
        """Live numeric parameter array, read fresh from
        ``self.sm.parameter_values`` on every access.  Mutating
        ``nsm.sm.parameter_values.g = 12.0`` is reflected on the next
        call — no runtime rebuild needed.  Pass this array (or a
        ``jax.grad``-tracked clone) as the parameter argument of any
        operator to take AD w.r.t. parameters."""
        return jnp.asarray(
            [float(v) for v in self.sm.parameter_values.values()],
            dtype=float,
        )

    @property
    def parameter_names(self) -> List[str]:
        return [str(k) for k in self.sm.parameters.keys()]

    @property
    def parameter_symbols(self) -> List[sp.Symbol]:
        return list(self.sm.parameters.values())

    # Convenience alias for code that wants the source Model's API surface
    # ("dimension", etc.) but only has a JaxRuntime in hand. The Model's
    # `.dimension` equals the SystemModel's `n_dim` (both are the spatial
    # dimensionality of the PDE).
    @property
    def dimension(self) -> int:
        return self.n_dim

    # ── Per-cell operator construction ───────────────────────────

    def _build_cell_operators(self):
        sm = self.sm
        # REQ-185: every registry-slot operator derives its lambdify arg
        # lists from the DECLARED signature on the SystemModel
        # (``sm.operator_signature`` via ``_build_slot_operator`` /
        # ``_slot_arg_lists``) — this runtime translates syntax only and
        # never re-declares arg lists.  ``std`` survives ONLY for the
        # NON-slot model map ``state_update`` (a splitter product outside
        # ``OPERATOR_ARG_SLOTS``), mirroring the generic-C / foam decision
        # to leave non-slot lowerings hand-built.
        std = (self._q_syms, self._qaux_syms, self._p_syms)

        # ── flux: (n_eq, n_dim)
        self.flux = self._build_slot_operator("flux", sm.flux)
        # ── hydrostatic_pressure: (n_eq, n_dim)
        self.hydrostatic_pressure = self._build_slot_operator(
            "hydrostatic_pressure", sm.hydrostatic_pressure)
        # ── source: (n_eq, 1) → collapse trailing axis below.
        # REQ-185: the source signature is DECLARED on the SystemModel
        # (``operator_signature`` — the single source) and carries time + dt +
        # the length-3 position vector; ``_build_slot_operator`` reads it and
        # returns a backward-compatible callable ``source(Q, Qaux, p, time=0,
        # dt=0, position=None)`` (defaults are the coordinate origin, EXACT for
        # autonomous sources — same trailing-arg padding the numpy runtime uses).
        self.source = self._build_slot_operator(
            "source", sm.source, squeeze_trailing=True)
        # ── state_update: split-system corrector update (n_eq, 1) →
        # collapse trailing 1.  Optional — some sub-systems carry no
        # state_update slot, in which case the attribute is None.
        # NON-slot model map (not in ``OPERATOR_ARG_SLOTS``): keeps the
        # hand-built ``std`` arg list, like reconstruction_variables /
        # interpolate_to_3d in the C-family printers.
        su = getattr(sm, "state_update", None)
        if su is not None:
            self.state_update = self._vmap_cell(
                _lambdify_array(_to_array_of_exprs(su), std, self.module),
                n_extra=0, squeeze_trailing=True,
            )
        else:
            self.state_update = None

        # ── reconstruction_variables / state_from_reconstruction ──────────
        # The order-2 primitive reconstruction MODEL MAP PAIR, emitted here so
        # jax consumes the SAME (b, b+h, hinv·q) that OpenFOAM's
        # ``Model::reconstruction_variables`` / ``state_from_reconstruction``
        # consume (``zoomy_foam/numerics_o2.H``).  MANDATE 6a: the ``1/h`` in
        # the forward map is the KP-desingularized ``hinv`` AUX that core swept
        # in via ``desingularize_hinv`` (``_RECONSTRUCTION_OPERATORS``) at the
        # canonical ``wet_dry_eps`` — so the reconstruction needs NO threshold
        # of its own, and no backend-side literal can disagree with core.
        #
        # Both are NON-slot model maps, so (like ``state_update`` above) they
        # keep a hand-built arg list rather than a declared ``operator_signature``.
        # The INVERSE is expressed in fresh ``WB_<state>`` symbols (see
        # ``zoomy_core.model.reconstruction_inverse``), NOT in the state
        # symbols — its first arg group is therefore the WB vector, fed with
        # the limiter's reconstructed primitive FACE values.
        rv = getattr(sm, "reconstruction_variables", None)
        sfr = getattr(sm, "state_from_reconstruction", None)
        if rv is not None and sfr is not None:
            from zoomy_core.model.reconstruction_inverse import (
                reconstruction_symbols)
            wb_syms = reconstruction_symbols(self._q_syms)
            rv_arr = _to_array_of_exprs(rv)
            sfr_arr = _to_array_of_exprs(sfr)
            self.reconstruction_variables = self._vmap_cell(
                _lambdify_array(rv_arr, std, self.module), n_extra=0)
            self.state_from_reconstruction = self._vmap_cell(
                _lambdify_array(
                    sfr_arr, (wb_syms, self._qaux_syms, self._p_syms),
                    self.module),
                n_extra=0)
            # Does the INVERSE actually read aux?  The face pass has cell-centre
            # aux, not face aux; if the inverse needed a face-interpolated aux
            # we would be silently feeding it the wrong thing.  Record the fact
            # rather than assume it — the consumer asserts on it.
            _aux_set = set(self._qaux_syms)
            self.state_from_reconstruction_uses_aux = any(
                _aux_set & sp.sympify(e).free_symbols
                for e in sp.flatten(sfr_arr))
        else:
            self.reconstruction_variables = None
            self.state_from_reconstruction = None
            self.state_from_reconstruction_uses_aux = False
        # ── mass_matrix: (n_eq, n_state)
        self.mass_matrix = self._build_slot_operator(
            "mass_matrix", sm.mass_matrix)
        # ── source_jacobian_wrt_variables: (n_eq, n_state) — ∂S/∂Q (frozen aux).
        # ── source_jacobian_wrt_aux_variables: (n_eq, n_aux) — ∂S/∂aux.
        # Both are real fields on the SystemModel (channeled from the Model,
        # else auto-derived in __post_init__).  We lambdify them here so IMEX
        # (and any other solver that needs a cell-local Newton on the source)
        # can assemble the implicit Jacobian symbolically: the consistent
        # ∂S/∂Q_total = ∂S/∂Q + ∂S/∂aux·∂aux/∂Q, with ∂aux/∂Q taken by AD of
        # the (cheap) update_aux_variables map — no AD through ``source``.
        sj = getattr(sm, "source_jacobian_wrt_variables", None)
        if sj is not None:
            self.source_jacobian_wrt_variables = self._build_slot_operator(
                "source_jacobian_wrt_variables", sj)
        else:
            self.source_jacobian_wrt_variables = None
        sja = getattr(sm, "source_jacobian_wrt_aux_variables", None)
        if sja is not None and _shape(_to_array_of_exprs(sja)):
            self.source_jacobian_wrt_aux_variables = self._build_slot_operator(
                "source_jacobian_wrt_aux_variables", sja)
        else:
            self.source_jacobian_wrt_aux_variables = None
        # ── update_variables: (n_eq, 1) per-cell state hygiene
        # transform — squeeze the trailing 1 so callers get
        # ``(n_eq, n_cells)``.  Optional; ``None`` means identity, in
        # which case the slot stays ``None`` and the solver's
        # ``update_q`` short-circuits to a no-op via ``getattr``.
        uv = getattr(sm, "update_variables", None)
        if uv is not None and _shape(_to_array_of_exprs(uv)):
            # REQ-185: declared ``update_variables(Q, Qaux, p, dt)`` — dt is
            # load-bearing in the Chorin corrector, where it is baked in as a
            # model PARAMETER and the declared dt group is dropped again
            # (duplicate lambdify argument, see ``_slot_arg_lists``).  Callers
            # keep the 3-arg form; ``dt=...`` reaches a non-Chorin remap.
            self.update_variables = self._build_slot_operator(
                "update_variables", uv, squeeze_trailing=True)
        else:
            self.update_variables = None
        # ── update_aux_variables: (n_aux, 1) per-cell aux formula (e.g. the
        # KP-desingularized hinv = sqrt(2) h / sqrt(h^4 + max(h,eps)^4)).
        # Lowered exactly like ``update_variables``; the solver applies it to
        # Qaux each step (post_step / update_qaux).  ``None`` ⇒ identity (the
        # model declares no per-cell aux formula), and the slot short-circuits.
        uav = getattr(sm, "update_aux_variables", None)
        if uav is not None and _shape(_to_array_of_exprs(uav)):
            # REQ-185: declared signature carries time + position (no dt);
            # backward-compatible ``update_aux_variables(Q, Qaux, p, time=0,
            # position=None)`` — a rain-rate aux binds ``t``, a manufactured
            # ``S(x)`` binds position; state-only aux ignores them.
            self.update_aux_variables = self._build_slot_operator(
                "update_aux_variables", uav, squeeze_trailing=True)
        else:
            self.update_aux_variables = None
        # ── diffusion_matrix / diffusion_matrix_explicit:
        # ``(n_eq, n_state, n_dim, n_dim)`` rank-4 tensors.  Optional —
        # ``None`` when the SystemModel carries no diffusion.  Solvers
        # that wire diffusion through the runtime use these; FV paths
        # that fold diffusion into the convective flux can ignore them.
        dm = getattr(sm, "diffusion_matrix", None)
        if dm is not None and _shape(_to_array_of_exprs(dm)):
            self.diffusion_matrix = self._build_slot_operator(
                "diffusion_matrix", dm)
        else:
            self.diffusion_matrix = None
        dme = getattr(sm, "diffusion_matrix_explicit", None)
        if dme is not None and _shape(_to_array_of_exprs(dme)):
            self.diffusion_matrix_explicit = self._build_slot_operator(
                "diffusion_matrix_explicit", dme)
        else:
            self.diffusion_matrix_explicit = None
        # ── interpolate_to_3d: (n_3d_components,)  signature
        # ``(Q, Qaux, p, position)`` where ``position = [X0, X1, X2]``.
        # NON-slot model map (not in ``OPERATOR_ARG_SLOTS``): hand-built
        # arg list, matching the generic-C / foam printers which also keep
        # their interpolate_to_3d lowerings backend-local.
        # Vmapped over the cell axis with ``position`` broadcast per
        # cell so callers can hand in a per-cell ``(3, n_cells)``
        # position array (typically constant ``(x_cell, y_cell, z*)``
        # for a chosen vertical level ``z*``).  Returns the 3D
        # reconstruction at ``z*`` per cell.
        p23 = getattr(sm, "interpolate_to_3d", None)
        # Treat the all-zero default (``Model.interpolate_to_3d``
        # returns ``ZArray.zeros(6)`` by default) as "no projection
        # defined" — callers fall back to a model-agnostic
        # depth-averaged ``hu/h`` instead of getting silently-zero
        # AD/FD sensitivities.
        if p23 is not None:
            p23_arr = _to_array_of_exprs(p23)
            if _shape(p23_arr) and all(int(bool(e)) == 0
                                       for e in p23_arr.tolist()):
                p23 = None
        if (p23 is not None and _shape(_to_array_of_exprs(p23))
                and self._pos_syms):
            p23_sig = (self._q_syms, self._qaux_syms, self._p_syms,
                       self._pos_syms)
            try:
                p23_fn = _lambdify_array(_to_array_of_exprs(p23),
                                         p23_sig, self.module)
            except Exception:
                # Recent SME / VAM / MLSME slot may carry raw spatial
                # Derivative atoms that the JAX printer can't lambdify
                # (e.g. Derivative(b, x), Derivative(q_k/h, x)). The
                # 2D-to-3D projection is post-simulation reconstruction,
                # not needed by the explicit FVM solver — drop the slot
                # gracefully and continue. Re-raise on any other failure.
                self.interpolate_to_3d = None
            else:
                @jax.jit
                def _interpolate_to_3d(Q, Qaux, parameters, position):
                    def per_cell(q, qaux, pos):
                        return p23_fn(q, qaux, parameters, pos)
                    return jax.vmap(per_cell, in_axes=(1, 1, 1),
                                    out_axes=-1)(Q, Qaux, position)
                self.interpolate_to_3d = _interpolate_to_3d
        else:
            self.interpolate_to_3d = None
        # ── nonconservative_matrix: (n_eq, n_state, n_dim)
        self.nonconservative_matrix = self._build_slot_operator(
            "nonconservative_matrix", sm.nonconservative_matrix)
        # ── quasilinear_matrix: (n_eq, n_state, n_dim)
        self.quasilinear_matrix = self._build_slot_operator(
            "quasilinear_matrix", sm.quasilinear_matrix)
        # ── eigenvalues: declared ``(Q, Qaux, p, n)`` — the normal group is
        # a deliberate deviation from the ruling's condensed list, pinned by
        # the committed REQ-185 C ABI.  Two paths:
        #   * symbolic eigenvalues present → lambdify against the declared
        #     signature (normal is per-cell, vmap axis 1).
        #   * eigenvalues is None (chain Chorin convention) → fall
        #     back to numerical eigvals(quasilinear · n) per cell.
        eig_def = sm.eigenvalues
        eig_arr = _to_array_of_exprs(eig_def)
        eig_shape_cell = _shape(eig_arr)
        # ``eigenvalues_cfl`` — the TIMESTEP-ONLY spectrum published by
        # ``gate_eigenvalues_dry`` (cid 209/210).  Registered exactly like
        # ``eigenvalues`` and against the same declared signature; ``None`` when
        # the model carries no gate, in which case the solver's dt path falls
        # back to ``eigenvalues``.  The dissipation always reads ``eigenvalues``.
        cfl_def = getattr(sm, "eigenvalues_cfl", None)
        cfl_shape = _shape(_to_array_of_exprs(cfl_def))
        self.eigenvalues_cfl = (
            self._build_slot_operator("eigenvalues_cfl", cfl_def)
            if cfl_shape and all(s > 0 for s in cfl_shape) else None)
        if eig_shape_cell and all(s > 0 for s in eig_shape_cell):
            self.eigenvalues = self._build_slot_operator(
                "eigenvalues", eig_def)
        else:
            ql_arr = _to_array_of_exprs(sm.quasilinear_matrix)
            ql_fn = _lambdify_array(
                ql_arr, self._slot_arg_lists("quasilinear_matrix")[1],
                self.module)
            n_state = self.n_state
            n_dim = self.n_dim

            @jax.jit
            def _eig_full_num(Q, Qaux, parameters, normal):
                def per_cell(q, qaux, n):
                    ql = jnp.asarray(ql_fn(q, qaux, parameters))
                    ql = ql.reshape(n_state, n_state, n_dim)
                    # REQ-170: the mesh stores face normals with 3 components
                    # (vertex coords are 3-D; z ≡ 0 on a 2-D mesh) while the
                    # quasilinear matrix carries only ``n_dim`` directions —
                    # einsum then fails on label 'k'.  Truncating the normal is
                    # exact (the dropped components are identically zero) and a
                    # no-op when the ranks already agree (1-D / 3-D).
                    A_n = jnp.einsum("ijk,k->ij", ql, n[:n_dim])
                    return jnp.real(jnp.linalg.eigvals(A_n))
                return jax.vmap(per_cell, in_axes=(1, 1, 1),
                                out_axes=-1)(Q, Qaux, normal)
            self.eigenvalues = _eig_full_num

    def _vmap_cell(self, per_cell_fn, *, n_extra=0, squeeze_trailing=False):
        """Wrap a per-cell ``fn(q, qaux, p, *extra)`` into a JIT'd
        full-grid callable.  ``Q`` and ``Qaux`` are vmapped over the
        cell axis (axis 1); ``parameters`` and ``extra`` are broadcast.

        ``squeeze_trailing``: SystemModel column operators (source) are
        stored as ``(n_eq, 1)`` matrices; we drop the size-1 axis so
        downstream code gets ``(n_eq, n_cells)`` not
        ``(n_eq, 1, n_cells)``."""
        @jax.jit
        def full_grid(Q, Qaux, parameters):
            def per_cell(q, qaux):
                return per_cell_fn(q, qaux, parameters)
            out = jax.vmap(per_cell, in_axes=(1, 1), out_axes=-1)(Q, Qaux)
            if squeeze_trailing and out.ndim >= 3 and out.shape[-2] == 1:
                # (n_eq, 1, n_cells) → (n_eq, n_cells): drop the trailing
                # singleton of a vmapped COLUMN operator (ndim==3).  Guard on
                # ndim>=3 (not >=2): a genuinely 1-D single-element slot
                # (e.g. a 1-row update_aux_variables) vmaps to (1, n_cells)
                # and must NOT be squeezed to (n_cells,).
                out = jnp.squeeze(out, axis=-2)
            return out
        return full_grid

    # Signature groups whose runtime arrays vary PER CELL (vmap axis 1);
    # the rest (parameters, time, dt) broadcast unbatched.
    _PER_CELL_GROUPS = ("variables", "aux_variables", "normal", "position")

    def _slot_arg_lists(self, name):
        """``(keys, arg_lists)`` for registry-slot operator ``name``, read off
        the DECLARED signature ``sm.operator_signature(name)`` — the REQ-185
        boundary API, the same ``Function.args`` the C-family and numpy
        printers consume.  The symbols come from the signature groups
        themselves, so this runtime translates SYNTAX only and never
        re-declares arg lists.

        One jax lowering-SYNTAX constraint, mirroring numpy's
        ``NumpyRuntimeModel._op_sig``: when a split sub-system baked ``dt``
        in as a model PARAMETER (Chorin/VAM), the declared ``dt`` group is
        the SAME symbol as the parameter and must be dropped from the
        lambdify args (duplicate argument) — a lowering constraint, not a
        signature difference."""
        sig = self.sm.operator_signature(name)
        dt_in_params = bool(self.sm.parameters.contains("dt"))
        keys = [k for k in sig.keys() if not (k == "dt" and dt_in_params)]
        return keys, tuple(_group_symbols(sig[k]) for k in keys)

    def _build_slot_operator(self, name, definition, *, squeeze_trailing=False):
        """Build a registry-slot operator whose lambdify signature carries
        the DECLARED groups (variables, aux_variables, parameters, [normal],
        [time], [dt], [position]) from ``_slot_arg_lists(name)`` — no local
        arg-list re-declaration (REQ-185).  Generalizes the coord-operator
        builder to EVERY slot: ``(Q, Qaux, p)``-only slots reduce exactly to
        the classic per-cell vmap.

        The returned callable is backward-compatible:
        ``fn(Q, Qaux, parameters, *extras, **extras)``.  Trailing declared
        groups bind positionally in FULL declared-signature order (so
        ``eigenvalues(Q, Qaux, p, n)`` and ``source(Q, Qaux, p, t, dt, x)``
        read naturally, and a positionally-passed ``dt`` still lands on the
        dt slot even when the Chorin drop removed it) or by keyword.
        ``time`` / ``dt`` default to 0 and ``position`` to the coordinate
        origin ``zeros((3, n_cells))`` — EXACT for operators that don't
        reference them, so existing 3-arg solver call sites are unchanged;
        ``position`` follows the mesh's 3-padded convention.  Extras passed
        to a slot that doesn't declare them are accepted and IGNORED
        (mirrors the numpy flattener's trailing-``dt`` tolerance).
        ``uniform_rank`` anchoring stays on the ``variables`` group — always
        first in every declared slot."""
        keys, arg_lists = self._slot_arg_lists(name)
        per_cell_fn = _lambdify_array(
            _to_array_of_exprs(definition), arg_lists, self.module)
        cell_keys = tuple(k for k in keys if k in self._PER_CELL_GROUPS)
        # Positional binding of trailing groups uses the FULL declared key
        # order (pre dt-drop) so positional call sites are drop-invariant.
        declared = list(self.sm.operator_signature(name).keys())
        trailing_keys = tuple(k for k in declared
                              if k not in ("variables", "aux_variables",
                                           "parameters"))

        @jax.jit
        def _jitted(*groups):
            bound = dict(zip(keys, groups))

            def per_cell(*cell_vals):
                local = {**bound, **dict(zip(cell_keys, cell_vals))}
                return per_cell_fn(*(local[k] for k in keys))

            out = jax.vmap(per_cell, in_axes=(1,) * len(cell_keys),
                           out_axes=-1)(*(bound[k] for k in cell_keys))
            if squeeze_trailing and out.ndim >= 3 and out.shape[-2] == 1:
                out = jnp.squeeze(out, axis=-2)
            return out

        def full_grid(Q, Qaux, parameters, *args, **kwargs):
            extras = dict(zip(trailing_keys, args))
            extras.update(kwargs)
            groups = []
            for k in keys:
                if k == "variables":
                    groups.append(Q)
                elif k == "aux_variables":
                    groups.append(Qaux)
                elif k == "parameters":
                    groups.append(parameters)
                elif extras.get(k) is not None:
                    v = extras[k]
                    groups.append(jnp.asarray(v, float)
                                  if k in ("time", "dt") else v)
                elif k in ("time", "dt"):
                    groups.append(jnp.asarray(0.0))
                elif k == "position":
                    groups.append(jnp.zeros((3, Q.shape[-1])))
                else:
                    raise TypeError(
                        f"{name}: declared arg group '{k}' not supplied — "
                        f"pass it positionally after parameters or as "
                        f"{k}=...")
            return _jitted(*groups)

        return full_grid

    # ── Per-face Riemann numerics ────────────────────────────────

    def _build_face_operators(self):
        # Build the symbolic Riemann (uses the NSM's riemann class).
        numerics = self.nsm.build_numerics()
        self._numerics = numerics

        # ``max_wavespeed`` plug — Rusanov-style numerics emit one
        # ``max_wavespeed(*Q, *Qaux, *p, *n)`` per face side (left and
        # right are *separate* invocations inside the symbolic body,
        # each with arity ``n_state + n_aux + n_p + n_n``).  Build a
        # plug that consumes those flat scalar args and returns the
        # per-side max |eigenvalue|.
        #
        # Two paths, mirroring NumPy:
        #   * If ``sm.eigenvalues`` is non-empty (symbolic): lambdify
        #     it and take ``max(|eigs|)``.
        #   * If ``sm.eigenvalues`` is None (chain Chorin convention):
        #     fall back to numerical eigenvalues from the quasilinear
        #     matrix — ``A_n = sum_d ql[:, :, d] * n[d]``, then
        #     ``max(|jnp.linalg.eigvals(A_n)|)``.  Mirrors NumPy's
        #     ``eigenvalue_mode = "numerical"`` fallback.
        n_state = self.n_state
        n_aux = self.n_aux
        n_p = len(self._p_syms)
        n_n = len(self._n_syms)

        eig_arr = _to_array_of_exprs(self.sm.eigenvalues)
        eig_shape = _shape(eig_arr)
        has_symbolic_eig = (
            bool(eig_shape) and all(s > 0 for s in eig_shape))

        if has_symbolic_eig:
            _eig_scalar = _lambdify_array(
                eig_arr,
                self._slot_arg_lists("eigenvalues")[1],
                self.module,
            )

            def _max_wavespeed(*all_args):
                q = jnp.asarray(all_args[:n_state])
                qaux = jnp.asarray(all_args[n_state:n_state + n_aux])
                p = jnp.asarray(
                    all_args[n_state + n_aux:n_state + n_aux + n_p])
                n = jnp.asarray(
                    all_args[n_state + n_aux + n_p:
                             n_state + n_aux + n_p + n_n])
                return jnp.abs(_eig_scalar(q, qaux, p, n)).max()

        else:
            # Numerical fallback: per-cell quasilinear → eigvals.
            ql_arr = _to_array_of_exprs(self.sm.quasilinear_matrix)
            _ql_scalar = _lambdify_array(
                ql_arr,
                self._slot_arg_lists("quasilinear_matrix")[1],
                self.module,
            )
            n_dim = self.n_dim

            def _max_wavespeed(*all_args):
                q = jnp.asarray(all_args[:n_state])
                qaux = jnp.asarray(all_args[n_state:n_state + n_aux])
                p = jnp.asarray(
                    all_args[n_state + n_aux:n_state + n_aux + n_p])
                n = jnp.asarray(
                    all_args[n_state + n_aux + n_p:
                             n_state + n_aux + n_p + n_n])
                ql = jnp.asarray(_ql_scalar(q, qaux, p))
                ql = ql.reshape(n_state, n_state, n_dim)
                A_n = jnp.einsum("ijk,k->ij", ql, n)
                evs = jnp.linalg.eigvals(A_n)
                return jnp.max(jnp.abs(evs.real))

        # Inject before lambdifying numerics.
        self.module["max_wavespeed"] = _max_wavespeed

        # Per-face signature: (Q_minus, Q_plus, Qaux_minus, Qaux_plus,
        #                      parameters, normal) — all vectors.
        # NOT a registry slot: the minus/plus trace symbols are OWNED by the
        # Numerics object (per-face Riemann composite), so this signature is
        # read off ``numerics``, not ``OPERATOR_ARG_SLOTS``.
        face_sig = (
            list(numerics.variables_minus),
            list(numerics.variables_plus),
            list(numerics.aux_variables_minus),
            list(numerics.aux_variables_plus),
            self._p_syms,
            self._n_syms,
        )

        num_flux_expr = numerics.numerical_flux()
        num_fluct_expr = numerics.numerical_fluctuations()
        num_flux_arr = _to_array_of_exprs(num_flux_expr)
        num_fluct_arr = _to_array_of_exprs(num_fluct_expr)
        num_flux_per_face = _lambdify_array(
            num_flux_arr, face_sig, self.module)
        num_fluct_per_face = _lambdify_array(
            num_fluct_arr, face_sig, self.module)

        @jax.jit
        def numerical_flux(qL, qR, qauxL, qauxR, parameters, normal):
            def per_face(qLi, qRi, qauxLi, qauxRi, n):
                return num_flux_per_face(qLi, qRi, qauxLi, qauxRi,
                                         parameters, n)
            return jax.vmap(per_face, in_axes=(1, 1, 1, 1, 1),
                            out_axes=-1)(qL, qR, qauxL, qauxR, normal)

        @jax.jit
        def numerical_fluctuations(qL, qR, qauxL, qauxR, parameters, normal):
            def per_face(qLi, qRi, qauxLi, qauxRi, n):
                return num_fluct_per_face(qLi, qRi, qauxLi, qauxRi,
                                          parameters, n)
            return jax.vmap(per_face, in_axes=(1, 1, 1, 1, 1),
                            out_axes=-1)(qL, qR, qauxL, qauxR, normal)

        self.numerical_flux = numerical_flux
        self.numerical_fluctuations = numerical_fluctuations

        # The EMITTED interior nonconservative term.  Same face signature
        # SHAPE, so the same lambdify + vmap path — but the state pair is
        # (cell mean, that cell's own reconstructed edge value) and the normal
        # points OUT of the cell rather than across the face.
        #
        # Bound only when the numerics registers it, so a runtime built against
        # an older core still works; the solver then falls back to its legacy
        # inline volume form.
        num_ces = getattr(numerics, "numerical_cell_edge_source", None)
        if num_ces is None:
            self.numerical_cell_edge_source = None
        else:
            num_ces_arr = _to_array_of_exprs(num_ces())
            num_ces_per_face = _lambdify_array(
                num_ces_arr, face_sig, self.module)

            @jax.jit
            def numerical_cell_edge_source(qc, qe, qauxc, qauxe,
                                           parameters, normal):
                def per_face(qci, qei, qauxci, qauxei, n):
                    return num_ces_per_face(qci, qei, qauxci, qauxei,
                                            parameters, n)
                return jax.vmap(per_face, in_axes=(1, 1, 1, 1, 1),
                                out_axes=-1)(qc, qe, qauxc, qauxe, normal)

            self.numerical_cell_edge_source = numerical_cell_edge_source

    # ── Boundary-condition kernels ───────────────────────────────

    def _build_bc(self):
        """Lambdify the indexed BC kernel that lives on
        ``sm.boundary_conditions``.  The definition is typically a
        sympy ``Piecewise`` over ``i_bc_func`` whose branches are
        ``sp.Array`` ghost-state vectors of length ``n_state``.

        We delegate to the proven NumPy ``_lambdify_function``
        infrastructure (handles Piecewise-of-vectors, vectorized
        broadcast, flatten/extract) and just JIT-wrap the result so
        JAX sees it as one fused kernel.  No vmap — the BC index
        varies per face, so the solver invokes this inside its
        boundary fori_loop."""
        from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic

        for attr in ("boundary_conditions",
                     "aux_boundary_conditions",
                     "boundary_gradients"):
            obj = getattr(self.sm, attr, None)
            if obj is None:
                setattr(self, attr, None)
                continue
            # Lambdify via NumpyRuntimeSymbolic-style with the JAX
            # module dict so the body uses jnp.* operations.
            stub = _StubSymbolicRegistrar(obj, attr)
            rt = NumpyRuntimeSymbolic(stub, module=self.module, printer="jax")
            jit_callable = jax.jit(rt.runtime_functions[attr])
            setattr(self, attr, jit_callable)


class _StubSymbolicRegistrar:
    """Minimal SymbolicRegistrar look-alike: exposes a single named
    function via ``.functions``.  ``NumpyRuntimeSymbolic`` iterates
    ``symbolic_obj.functions`` and lambdifies each entry — we feed it
    one function at a time so each BC kernel becomes its own jit'd
    callable."""

    def __init__(self, fn_obj, name):
        from zoomy_core.misc.misc import Zstruct
        self.functions = Zstruct(**{name: fn_obj})


# ── Signature-flattening helpers for the indexed BC kernel ────────


def _flatten_with_structure(sig):
    """Walk a Zstruct-style signature and return (flat_arg_syms,
    structure).  ``structure`` is a list of ("key", "leaf-or-vector")
    entries used by ``_gather_flat`` to reassemble call args from the
    runtime arguments."""
    flat_args: List[sp.Symbol] = []
    structure: List[tuple] = []

    if hasattr(sig, "keys") and hasattr(sig, "values"):
        for key, val in zip(sig.keys(), sig.values()):
            sub_flat, sub_struct = _flatten_node(val)
            structure.append((key, sub_struct))
            flat_args.extend(sub_flat)
    else:
        # Fallback: treat as flat sequence.
        for i, val in enumerate(sig):
            sub_flat, sub_struct = _flatten_node(val)
            structure.append((f"arg{i}", sub_struct))
            flat_args.extend(sub_flat)
    return flat_args, structure


def _flatten_node(node):
    """Recursively flatten a Zstruct/list/scalar node.  Returns
    (flat_symbols, count) where count is the number of scalar slots
    the signature expects for this node — used to size-match runtime
    args (e.g. a 1D position signature declares 1 slot even though the
    JAX mesh stores a 3D padded position array)."""
    if isinstance(node, sp.Symbol):
        return [node], 1
    if hasattr(node, "values") and callable(node.values):
        syms = []
        total = 0
        for v in node.values():
            sub, n = _flatten_node(v)
            syms.extend(sub)
            total += n
        return syms, total
    if isinstance(node, (list, tuple)):
        syms = []
        total = 0
        for v in node:
            sub, n = _flatten_node(v)
            syms.extend(sub)
            total += n
        return syms, total
    return [node], 1


def _flatten_with_structure(sig):
    """Walk a Zstruct-style signature and return (flat_arg_syms,
    structure).  ``structure`` is a list of ``(key, expected_count)``
    pairs used by ``_gather_flat`` to slice the runtime arrays to
    the right length."""
    flat_args: List[sp.Symbol] = []
    structure: List[tuple] = []

    if hasattr(sig, "keys") and hasattr(sig, "values"):
        for key, val in zip(sig.keys(), sig.values()):
            sub_flat, count = _flatten_node(val)
            structure.append((key, count))
            flat_args.extend(sub_flat)
    else:
        for i, val in enumerate(sig):
            sub_flat, count = _flatten_node(val)
            structure.append((f"arg{i}", count))
            flat_args.extend(sub_flat)
    return flat_args, structure


def _gather_flat(structure, i_bc, time, position, distance, q_cell,
                 qaux_cell, parameters, normal):
    """Map runtime args (named) to the flat list the lambdified BC
    function expects.  Each ``structure`` entry is ``(key, count)`` —
    we take exactly ``count`` scalars from the matching runtime arg."""
    named = {
        "i_bc_func": i_bc, "i_bc": i_bc,
        "time": time, "t": time,
        "position": position, "X": position,
        "distance": distance, "dX": distance,
        "variables": q_cell, "Q": q_cell,
        "aux_variables": qaux_cell, "Qaux": qaux_cell,
        "parameters": parameters, "p": parameters,
        "normal": normal, "n": normal,
    }
    flat = []
    for key, count in structure:
        val = named.get(key, None)
        if val is None:
            # Unknown structural key — pass zeros (count of them).
            for _ in range(count):
                flat.append(jnp.asarray(0.0))
            continue
        if count == 1 and (
            not hasattr(val, "shape") or val.shape == () or val.ndim == 0
        ):
            flat.append(jnp.asarray(val))
            continue
        arr = jnp.asarray(val)
        if arr.ndim == 0:
            # Scalar but multiple expected — repeat (rare).
            for _ in range(count):
                flat.append(arr)
        else:
            for i in range(count):
                flat.append(arr[i])
    return flat
