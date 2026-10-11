"""Module `zoomy_core.model.boundary_conditions`."""

import functools
import numpy as np
from time import time as get_time
import sympy
from sympy import Matrix
import param
from typing import Callable, List

from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function


# ── per-field ``on=`` masking, rooted on the base BoundaryCondition ──────────
# Every BC carries an ``on`` selector (a field name / family / alias / "all").
# A BC must only write the ghost slots it OWNS; the rest fall back to the
# interior (extrapolation) for values and to zero-Neumann for gradients.  This
# masking is implemented ONCE here and applied to EVERY BC subclass via
# ``BoundaryCondition.__init_subclass__`` — so ``on=`` works natively for all
# BCs (no PerFieldBoundary wrapper needed for the common single-BC-per-tag
# case).  The slot indices are bound against the model state by ``bind_on``.
#
# Map: method name -> (index of the interior-state arg in *args, is_gradient,
# symbolic).  Value methods default non-owned slots to the interior; gradient
# methods default them to 0 (zero Neumann).  ``compute_*`` are the symbolic
# (codegen / ZArray) kernels; ``face_*`` are the numeric (numpy) kernels.
_BC_ON_METHODS = {
    "compute_boundary_condition": (3, False, True),
    "compute_boundary_gradient":  (3, True,  True),
    "compute_ghost_state":        (3, False, True),
    "face_value":                 (0, False, False),
    "face_state":                 (0, False, False),
    "face_gradient":              (0, True,  False),
}


def _wrap_on_method(fn, interior_idx, is_grad, symbolic):
    """Wrap a BC value/gradient producer so its result is applied ONLY to the
    slots in ``self._on_slots`` (``None`` ⇒ owns all ⇒ pass through unchanged,
    bit-identical to the un-wrapped behaviour)."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kw):
        full = fn(self, *args, **kw)
        slots = getattr(self, "_on_slots", None)
        if slots is None:                       # owns everything → no masking
            return full
        n = len(full)
        if symbolic:
            base = ZArray.zeros(n) if is_grad else ZArray(args[interior_idx])
        else:
            arr = np.asarray(full, dtype=float)
            base = (np.zeros_like(arr) if is_grad
                    else np.asarray(args[interior_idx], dtype=float).copy())
        for s in slots:
            base[s] = full[s]
        return base
    wrapped._on_wrapped = True
    return wrapped


# --- Helper Function (Unchanged) ---
def _sympy_interpolate_data(time, timeline, data):
    """Internal helper `_sympy_interpolate_data`."""
    assert timeline.shape[0] == data.shape[0]
    conditions = (((data[0], time <= timeline[0])),)
    for i in range(timeline.shape[0] - 1):
        t0 = timeline[i]
        t1 = timeline[i + 1]
        y0 = data[i]
        y1 = data[i + 1]
        conditions += (
            (-(time - t1) / (t1 - t0) * y0 + (time - t0) / (t1 - t0) * y1, time <= t1),
        )
    conditions += (((data[-1], time > timeline[-1])),)
    return sympy.Piecewise(*conditions)


# --- Base Class ---
class BoundaryCondition(param.Parameterized):
    """
    Default implementation. The required data for the 'ghost cell' is the data
    from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.

    Flat-list interface: a BC carries a positional ``tag`` (the boundary patch)
    and an ``on`` selector naming the FIELD(S) it applies to — a state field name
    (``"h"``, ``"q_0"``), a family base (``"q"``, ``"k"``, ``"T"`` → every
    ``<base>_i`` slot), a model-registered alias (``"momentum"``), or ``"all"``
    (default).  ``on`` is resolved generically against the model's own declared
    state by :func:`resolve_per_field`, so ANY future field is addressable by
    name with no hard-coding.  Pass a flat list of these to a model and different
    fields get different BCs at the same tag (momentum reflects, depth
    extrapolates, …).
    """

    tag = param.String(default="bc")
    on = param.String(default="all", doc=(
        "Field/family/group this BC applies to ('h', 'q', 'q_0', 'momentum', "
        "'all'); resolved against the model's declared state."))

    # Resolved slot indices this BC owns (None ⇒ all ⇒ no masking).  Set by
    # ``bind_on`` once the model state is known (see SystemModel.attach_*).
    _on_slots = None

    #: How :meth:`compute_boundary_condition`'s value is to be read by the
    #: GHOST convention (decision D5, :meth:`compute_ghost_state`):
    #: ``"ghost"`` — the value IS the ghost state (extrapolation, reflection,
    #: periodic partner, incoming-characteristic ghost, data ghost);
    #: ``"face"`` — the value is the state ON the face, so the ghost is the
    #: linear extrapolation through it, ``2·face − Q_inner`` (Dirichlet,
    #: inflow/outflow, user lambdas).
    boundary_value_kind = "ghost"

    def __init_subclass__(cls, **kw):
        """Apply the ``on=`` mask to EVERY subclass automatically: each
        value/gradient producer defined on the subclass is wrapped so it only
        writes the slots the BC owns.  Inherited (un-overridden) producers are
        already wrapped on the base class, so they mask too."""
        super().__init_subclass__(**kw)
        for name, (idx, is_grad, symbolic) in _BC_ON_METHODS.items():
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_on_wrapped", False):
                setattr(cls, name, _wrap_on_method(fn, idx, is_grad, symbolic))

    def __init__(self, tag=None, on=None, **params):
        if tag is not None:
            params["tag"] = tag
        if on is not None:
            params["on"] = on
        super().__init__(**params)

    def bind_on(self, state_names, aliases=None):
        """Resolve this BC's ``on=`` selector to the state-slot indices it owns
        (``None`` ⇒ ``"all"`` ⇒ owns everything, masking disabled).  Idempotent;
        called by ``SystemModel.attach_boundary_conditions`` against the declared
        state so the per-field mask is honored at EVERY entry point."""
        on = self.on or "all"
        if on in ("all", ""):
            self._on_slots = None
        else:
            self._on_slots = _resolve_on(
                on, list(state_names), aliases or {"momentum": "q"})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic builder for the boundary *value* — used by
        :meth:`BoundaryConditions.get_boundary_condition_function` to
        assemble the indexed ``Piecewise`` BC kernel."""
        raise NotImplementedError(
            "BoundaryCondition is a virtual class. Use one of its derived classes!"
        )

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic builder for the boundary *normal-direction gradient*
        ``∂Q/∂n`` at the face — used by
        :meth:`BoundaryConditions.get_boundary_gradient_function` to
        assemble the indexed ``Piecewise`` boundary-gradient kernel
        (consumed by the diffusion path).

        Default: zero Neumann.  Subclasses override when a non-zero
        gradient is part of the BC (e.g. prescribed-flux Robin BCs)."""
        n_vars = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        return ZArray.zeros(n_vars)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Symbolic GHOST state of this BC (decision D5) — the ONE value the
        new solver template's gradient, limiter bounds and Riemann right state
        all consume.  The ghost sits at the mirror point ``x_c + 2(x_f − x_c)``
        of the inner cell, so a BC that prescribes a FACE value ``v`` has the
        ghost ``2v − Q_inner`` (exact for a linear field); a BC whose value is
        already a ghost (reflection, extrapolation, periodic partner) returns
        it unchanged — see :attr:`boundary_value_kind`.  ``bdata`` is the
        per-face boundary-data vector (coupling); ignored unless a subclass
        reads it.  Non-owned slots (``on=``) are the interior value."""
        v = ZArray(self.compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal))
        if self.boundary_value_kind == "face":
            q = ZArray(Q)
            return ZArray([2 * v[i] - q[i] for i in range(len(q))])
        return v

    def face_state(self, Q_face, Qaux_face, normal, parameters):
        """Compute the boundary-side Riemann state from the reconstructed face value.

        Called at boundary faces inside the flux operator.
        ``Q_face`` is the MUSCL-reconstructed interior state at the face.
        Returns the state that the Riemann solver sees on the boundary side.

        Default: same as Q_face (Neumann / zero-flux).

        .. deprecated:: Use ``face_value`` instead.
        """
        return Q_face.copy()

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Boundary-side Riemann state for convective flux (ghost-cell-free).

        Parameters
        ----------
        Q_inner : ndarray, shape (n_vars,)
            Cell-center values of the adjacent interior cell.
        Qaux_inner : ndarray, shape (n_aux,)
            Auxiliary variables of the adjacent interior cell.
        normal : ndarray, shape (dim,)
            Outward unit face normal.
        d_face : float
            Distance from interior cell center to face center.
        time : float
            Current simulation time.
        parameters : ndarray, shape (n_params,)
            Model parameters.

        Returns
        -------
        ndarray, shape (n_vars,)
            The state that the Riemann solver sees on the boundary side.
        """
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face,
                      time, parameters):
        """Face-normal gradient dQ/dn for diffusive flux at boundary.

        Parameters
        ----------
        Q_inner : ndarray, shape (n_vars,)
            Cell-center values of the adjacent interior cell.
        Q_face : ndarray, shape (n_vars,)
            The ``face_value`` result (already computed upstream).
        Qaux_inner : ndarray, shape (n_aux,)
            Auxiliary variables of the adjacent interior cell.
        normal : ndarray, shape (dim,)
            Outward unit face normal.
        d_face : float
            Distance from interior cell center to face center.
        time : float
            Current simulation time.
        parameters : ndarray, shape (n_params,)
            Model parameters.

        Returns
        -------
        ndarray, shape (n_vars,)
            Face-normal gradient at the boundary face.
        """
        Q_inner = np.asarray(Q_inner, dtype=float)
        Q_face = np.asarray(Q_face, dtype=float)
        return (Q_face - Q_inner) / max(d_face, 1e-30)


# Wrap the base class's OWN producers too (``__init_subclass__`` fires only for
# subclasses, not for ``BoundaryCondition`` itself), so inherited methods mask.
for _name, (_idx, _grad, _sym) in _BC_ON_METHODS.items():
    _fn = BoundaryCondition.__dict__.get(_name)
    if _fn is not None and not getattr(_fn, "_on_wrapped", False):
        setattr(BoundaryCondition, _name,
                _wrap_on_method(_fn, _idx, _grad, _sym))


# --- Derived Boundary Conditions (Unchanged) ---


class Extrapolation(BoundaryCondition):
    """Extrapolation. (class)."""
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        return ZArray(Q)

    def face_state(self, Q_face, Qaux_face, normal, parameters):
        """Extrapolation: boundary state = interior face state (zero flux)."""
        return Q_face.copy()

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Extrapolation: boundary state = interior cell state."""
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face,
                      time, parameters):
        """Extrapolation: zero Neumann (no normal gradient)."""
        return np.zeros_like(np.asarray(Q_inner, dtype=float))


class ZeroNeumann(Extrapolation):
    """Zero-gradient (∂Q/∂n = 0) — the named alias of :class:`Extrapolation`.
    Ghost = interior cell value; zero normal gradient.  Per-field via ``on=``."""


class Dirichlet(BoundaryCondition):
    """Prescribed VALUE on the field(s) this BC covers (``on=``).  ``value`` is a
    scalar (or symbolic expression on the codegen path) imposed on every slot the
    BC owns; other slots are untouched (the :class:`PerFieldBoundary` picks only
    this BC's slots).  Example: ``Dirichlet("left", on="h", value=1.5)``."""

    value = param.Number(default=0.0)
    boundary_value_kind = "face"

    def __init__(self, tag=None, on=None, value=None, **params):
        if value is not None:
            params["value"] = value
        super().__init__(tag=tag, on=on, **params)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        out = ZArray(Q)
        for i in range(n):
            out[i] = self.value
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        return np.full(np.asarray(Q_inner, dtype=float).shape, float(self.value))


class Flux(BoundaryCondition):
    """Prescribed normal-gradient (Neumann / Robin flux ``∂Q/∂n = gradient``) on
    the field(s) this BC covers (``on=``).  The ghost VALUE extrapolates; the
    boundary GRADIENT is set to ``gradient`` (consumed by the diffusion path).

    ``gradient`` is general and, like ``on=`` / ``tag``, needs NO live objects —
    it is one of:

    * a **number** — constant Neumann, e.g. ``Flux("top", on="mom", gradient=0)``
      (stress-free);
    * a **string formula** in field / parameter NAMES — the recommended Robin
      form, e.g. Navier slip ``Flux("bottom", on="mom", gradient="lambda_s*h*mom/nu")``;
      the names are resolved against ``sm`` at attach time, exactly the way
      ``on="mom"`` is resolved against the declared state — so the BC is written
      before the SystemModel exists;
    * a **callable** ``(Q, Qaux, normal, d, t, p) → value`` (numpy escape hatch);
    * a **sympy expression** (free symbols matched to ``sm`` by name).

    :meth:`resolve` (called from ``attach_boundary_conditions``) binds the names
    to the SystemModel's ``(state, aux, parameters)`` — storing the resolved
    expression for the codegen path and a lambda for the numpy path, so
    :meth:`face_gradient` evaluates it from the adjacent interior cell.  Only
    this BC's ``on=`` slots receive the gradient (root masking)."""

    gradient = param.Parameter(default=0.0)

    def __init__(self, tag=None, on=None, gradient=None, **params):
        if gradient is not None:
            params["gradient"] = gradient
        super().__init__(tag=tag, on=on, **params)
        self._resolved_grad = None     # sympy expr in sm symbols (codegen)
        self._grad_lambda = None       # numpy evaluator (state, aux, params)

    def resolve(self, sm):
        """Resolve a string / sympy ``gradient`` against ``sm`` BY NAME (numbers
        and callables pass through).  ``"lambda_s*h*mom/nu"`` → the expression in
        this SystemModel's actual state / aux / parameter symbols; then lambdify
        for the numpy path."""
        g = self.gradient
        if callable(g) and not isinstance(g, sympy.Basic):
            return
        syms = (list(sm.variables.get_list())
                + list(sm.aux_variables.get_list())
                + list(sm.parameters.get_list()))
        name_map = {str(s): s for s in syms}
        if isinstance(g, str):
            expr = sympy.sympify(g, locals=name_map)
        elif isinstance(g, sympy.Basic):
            expr = g.subs({fs: name_map[fs.name] for fs in g.free_symbols
                           if fs.name in name_map})
        else:
            return                                          # plain number
        self._resolved_grad = expr
        if not expr.is_number:
            self._grad_lambda = sympy.lambdify(syms, expr, "numpy")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)                      # value extrapolates

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Ghost of a prescribed normal gradient ``g``: ``Q + 2·dX·g`` (the
        mirror point lies ``2·dX`` along the outward normal), exact for a
        linear field with ``∂_n u = g`` on an orthogonal face."""
        q = ZArray(Q)
        g = ZArray(self.compute_boundary_gradient(
            time, X, dX, Q, Qaux, parameters, normal))
        return ZArray([q[i] + 2 * dX * g[i] for i in range(len(q))])

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        out = ZArray.zeros(n)
        g = self._resolved_grad if self._resolved_grad is not None else self.gradient
        g = g if isinstance(g, sympy.Basic) else sympy.sympify(
            g if not callable(g) else 0)
        for i in range(n):
            out[i] = g
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters):
        g = self.gradient
        if callable(g) and not isinstance(g, sympy.Basic):
            val = g(Q_inner, Qaux_inner, normal, d_face, time, parameters)
        elif self._grad_lambda is not None:                 # string / symbolic
            val = self._grad_lambda(*np.asarray(Q_inner, dtype=float),
                                    *np.asarray(Qaux_inner, dtype=float),
                                    *np.asarray(parameters, dtype=float))
        elif self._resolved_grad is not None:               # resolved constant
            val = float(self._resolved_grad)
        else:                                               # plain number
            val = float(g)
        return np.full(np.asarray(Q_inner, dtype=float).shape, float(val))


class Coupled(BoundaryCondition):
    """preCICE-coupled boundary patch.

    The kernel is a regular symbolic fallback (default: extrapolation —
    copy the inner state), printed into ``Model.H`` like any other BC.
    At runtime the solver intercepts this patch and overwrites its
    boundary values with the preCICE-exchanged data; the fallback is
    used before preCICE delivers data (initial step, decoupled runs).

    The interface always exchanges the **full** ``interpolate_to_3d``
    field set ``[b, h, u, v, w, p]`` sampled on a uniform vertical
    ``z``-grid — no per-field selection.  Direction, mapping and
    coupling scheme live entirely in ``precice-config.xml``; the BC only
    binds the patch to a named preCICE mesh.

    Parameters
    ----------
    tag : str
        OpenFOAM patch name (matches ``constant/polyMesh/boundary``).
    mesh_name : str
        preCICE mesh name (matches an entry in ``precice-config.xml``).
    """

    mesh_name = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Fallback (extrapolation) — overwritten by preCICE at runtime."""
        return ZArray(Q)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Coupled ghost = the per-face boundary DATA supplied by the coupling
        (``bdata``, filled by the coupling plugin; the template default is the
        BC input state, i.e. extrapolation before the peer delivers)."""
        if bdata is None:
            return ZArray(Q)
        return ZArray(list(bdata))

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Fallback face value = interior cell (extrapolation)."""
        return np.asarray(Q_inner, dtype=float).copy()


class InflowOutflow(BoundaryCondition):
    """Inflow / outflow boundary — prescribe selected state fields,
    extrapolate the rest.

    ``prescribe_fields`` maps a **state index** to a *spec*.  A spec is
    either a bare value (scalar or symbolic expression — the field is
    replaced by it) or a ``dict`` carrying a ``"mode"`` key:

    * ``{"mode": "replace", "value": v}`` — ``Q_out[k] = v``.
    * ``{"mode": "inlet_outlet", "value": v}`` — OpenFOAM ``inletOutlet``
      style: the field takes ``v`` where the interior is inflowing
      (``Q[k] >= 0``) and keeps the interior value otherwise (so
      backflow is never overridden):
      ``Q_out[k] = conditional(Q[k] >= 0, v, Q[k])``.  The index ``k``
      is assumed to be a momentum component whose positive direction
      points into the domain.
    * ``{"mode": "blend", "target": t, "weight": w}`` — a *soft* inflow:
      ``Q_out[k] = Q[k] + max(0, t - Q[k]) * w``.  ``w = 0`` means "no
      inflow" (the field stays at its interior value); ``w = 1`` pulls
      it fully to ``t``.

    Every ``value`` / ``target`` / ``weight`` is a scalar or a symbolic
    expression — typically built by the codegen driver from model
    parameters (e.g. a piecewise-linear Q(t) timeline interpolated over
    a fixed block of timeline parameters).  The BC plugs the
    expressions in; it never builds them and it never reads data.
    """

    prescribe_fields = param.Dict(default={})
    boundary_value_kind = "face"

    @staticmethod
    def _coerce(v):
        """Symbolic expressions and numbers pass through unchanged; a
        legacy eval-able string spec is evaluated."""
        if isinstance(v, sympy.Basic):
            return v
        try:
            return float(v)
        except (ValueError, TypeError):
            return float(eval(v))

    def _symbolic_field(self, k, spec, Q):
        """Symbolic ghost value for prescribed state index ``k``."""
        from zoomy_core.model.kernel_functions import conditional

        if not isinstance(spec, dict):
            return self._coerce(spec)
        mode = spec.get("mode", "replace")
        if mode == "replace":
            return self._coerce(spec["value"])
        if mode == "inlet_outlet":
            v = self._coerce(spec["value"])
            return conditional(sympy.GreaterThan(Q[k], 0), v, Q[k])
        if mode == "blend":
            target = self._coerce(spec["target"])
            weight = self._coerce(spec["weight"])
            return Q[k] + sympy.Max(0, target - Q[k]) * weight
        raise ValueError(
            f"InflowOutflow: unknown prescribe mode {mode!r} "
            f"(expected 'replace', 'inlet_outlet' or 'blend')"
        )

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, spec in self.prescribe_fields.items():
            Qout[k] = self._symbolic_field(k, spec, Q)
        return Qout

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """InflowOutflow (numpy path): prescribe selected fields, rest
        from the interior.  Supports constant ``replace`` specs only —
        the symbolic ``inlet_outlet`` / ``blend`` modes and
        parameter-valued prescriptions flow through the codegen path
        (``compute_boundary_condition``)."""
        Q_out = np.asarray(Q_inner, dtype=float).copy()
        for k, spec in self.prescribe_fields.items():
            if isinstance(spec, dict):
                if spec.get("mode", "replace") != "replace":
                    raise NotImplementedError(
                        "InflowOutflow.face_value supports only constant "
                        "'replace' specs; use the codegen path "
                        f"(compute_boundary_condition) for mode "
                        f"{spec.get('mode')!r}."
                    )
                spec = spec["value"]
            try:
                Q_out[k] = float(spec)
            except (ValueError, TypeError):
                Q_out[k] = float(eval(spec))
        return Q_out


class Lambda(BoundaryCondition):
    """User-function FACE values: ``prescribe_fields`` maps a slot to
    ``f(time, X, dX, Q, Qaux, p, n)`` evaluated at the face centre ``X``."""
    prescribe_fields = param.Dict(default={})
    boundary_value_kind = "face"

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            Qout[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


class FromModel(BoundaryCondition):
    """Boundary values DERIVED IN THE MODEL.

    References a boundary definition the declarative derivation registered as
    a function group, e.g.::

        m.register_group("boundary:wall", state_index, boundary_value_expr)

    ``SystemModel.from_model`` parses these into ``sm.boundary_specs[<name>]``
    (state-symbol expressions per state slot);
    ``SystemModel.attach_boundary_conditions`` then calls :meth:`resolve` on
    every ``FromModel`` entry — after which this BC has the SAME signature
    and kernel path as every other :class:`BoundaryCondition`.  Slots the
    model definition does not prescribe extrapolate (ghost = inner).

    Usage::

        BoundaryConditions([FromModel(tag="left", definition="wall"),
                            Extrapolation(tag="right")])
    """

    definition = param.String(default="", doc="name registered in the model "
                              "via register_group('boundary:<name>', …)")
    prescribe_fields = param.Dict(default={}, doc="state_index → sympy expr "
                                  "in the SystemModel's state symbols; "
                                  "filled by resolve()")

    def resolve(self, sm):
        """Pull the named definition off ``sm.boundary_specs`` (the wrapper
        model-definition → boundary function)."""
        specs = getattr(sm, "boundary_specs", None) or {}
        if self.definition not in specs:
            raise KeyError(
                f"FromModel: no boundary definition {self.definition!r} on "
                f"this SystemModel; available: {sorted(specs)}. Register it "
                "in the derivation via "
                f"register_group('boundary:{self.definition}', index, expr).")
        self.prescribe_fields = dict(specs[self.definition])
        self._state_syms = list(sm.state)
        # A model-derived boundary is a GEOMETRIC statement: a wall reflects
        # the NORMAL component and leaves the tangential one alone, so the
        # registered expression carries the face-normal symbols ``n_d`` just
        # like every normal-projected operator.  Capture them here exactly as
        # ``Characteristic.resolve`` does; without this substitution the n_d
        # survive as free symbols and the only registration that can be
        # written is a normal-BLIND full reversal.
        self._normal_syms = list(sm.normal.values()) if sm.normal else []
        return self

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        if not self.prescribe_fields:
            raise RuntimeError(
                f"FromModel(tag={self.tag!r}, definition={self.definition!r}) "
                "is unresolved — attach it via "
                "SystemModel.attach_boundary_conditions (or call .resolve(sm)).")
        Qv = ZArray(Q)
        Qout = ZArray(Q)
        smap = {s: Qv[i] for i, s in enumerate(self._state_syms)}
        for i, ns in enumerate(getattr(self, "_normal_syms", None) or []):
            smap[ns] = normal[i]
        for k, expr in self.prescribe_fields.items():
            Qout[k] = sympy.sympify(expr).xreplace(smap)
        return Qout


class Characteristic(BoundaryCondition):
    """Incoming-characteristic ghost — the data-level analogue of Roe
    upwinding.

    With ``A·n = R Λ L`` the eigendecomposition of the normal-projected
    quasilinear matrix at the interior state, the ghost is

        ``Q_ghost = Q + P⁻·(Q_target − Q)``,   ``P⁻ = R · 1_{λ<0} · L``,

    i.e. only the INCOMING characteristic fields (λ < 0 w.r.t. the outward
    normal) carry the target's information; the outgoing content is the
    interior's, so the face Riemann problem reflects nothing — even when the
    target data is inconsistent (mid-coupling-iteration peer data, far-field
    guesses).  Built symbolically over the SystemModel's OPAQUE
    ``eigensystem`` kernel (numerical eigendecomposition per backend), so it
    works for systems without a closed-form spectrum (SME N≥2, K&T).

    The target defaults to extrapolation (= interior ⇒ pure outflow);
    override slots via ``prescribe_fields`` (slot → callable, Lambda-style)
    or override :meth:`target_state` in subclasses (far-field, wall,
    coupled).  Resolved against the SystemModel by
    ``attach_boundary_conditions`` (needs the eigensystem + state layout).
    """

    prescribe_fields = param.Dict(default={}, doc="state_index → callable("
                                  "time, X, dX, Q, Qaux, p, n) for the "
                                  "TARGET state in that slot")

    def resolve(self, sm):
        self._state_syms = list(sm.state)
        self._normal_syms = list(sm.normal.values())
        self._es = [sympy.sympify(e) for e in sm.eigensystem]
        return self

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        """The state whose incoming-characteristic content is imposed.
        Default: extrapolation base overridden by ``prescribe_fields``."""
        target = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            target[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return target

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        if getattr(self, "_es", None) is None:
            raise RuntimeError(
                f"Characteristic(tag={self.tag!r}) is unresolved — attach "
                "via SystemModel.attach_boundary_conditions (or call "
                ".resolve(sm)).")
        n_st = len(self._state_syms)
        Qv = ZArray(Q)
        smap = {s: Qv[i] for i, s in enumerate(self._state_syms)}
        for i, ns in enumerate(self._normal_syms):
            smap[ns] = normal[i]
        ES = [e.xreplace(smap) for e in self._es]
        lam = ES[:n_st]
        R = Matrix(n_st, n_st,
                   lambda i, j: ES[n_st + i * n_st + j])
        L = Matrix(n_st, n_st,
                   lambda i, j: ES[n_st + n_st * n_st + i * n_st + j])
        target = self.target_state(time, X, dX, Q, Qaux, parameters, normal)
        dQ = Matrix(n_st, 1, lambda i, _j: target[i] - Qv[i])
        sel = sympy.diag(*[sympy.Piecewise((1, lam[k] < 0), (0, True))
                           for k in range(n_st)])
        corr = R * sel * (L * dQ)
        out = ZArray(Q)
        for i in range(n_st):
            out[i] = Qv[i] + corr[i, 0]
        return out


class CharacteristicFarField(Characteristic):
    """Far-field / open boundary: the incoming characteristics carry a
    prescribed far-field state, the outgoing ones leave undisturbed
    (Thompson-style non-reflecting in-/outflow)."""

    far_field = param.Parameter(default=None, doc="callable(n_state) → "
                                "far-field state vector, or a sequence")

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        if self.far_field is None:
            return ZArray(Q)
        vals = (self.far_field(len(self._state_syms))
                if callable(self.far_field) else self.far_field)
        target = ZArray(Q)
        for i, v in enumerate(vals):
            target[i] = v
        return target


def derive_momentum_field_indices(sm):
    """DERIVE the wall's momentum index groups from the model's OWN variable
    registry, so no caller has to know the state layout.

    The registry is ``sm.interpolate_to_3d`` — the canonical
    ``[b, h, u, v, w, p](ζ)`` lift that every derived family registers.  Row 0
    is ``b`` and row 1 is ``h`` (the GEOMETRY slots, never momentum); rows 2
    and 3 are the horizontal velocities ``u`` and ``v``.  The state entries a
    velocity row depends on — minus the geometry slots — ARE that direction's
    momentum components, **in state order**, so position ``j`` means the same
    moment / layer in every direction.

    Returns the groups in the convention :class:`Wall` consumes: a list of
    index groups, each group one VECTOR across the horizontal directions
    (``[[q_x_0, q_y_0], [q_x_1, q_y_1], …]``), i.e. the transpose of the
    per-direction lists.  A 1-D model yields singleton groups.

    A system that DECLARES its momentum role (``sm.declare_roles(momentum=…)``,
    e.g. the compressible Euler system, which has no lift) is read from that
    declaration instead.

    Raises with the model's own state printed if the registry is missing or
    the per-direction lists are ragged — guessing an index layout is exactly
    the failure this function exists to remove.
    """
    from zoomy_core.systemmodel.roles import declared_role_record
    if declared_role_record(sm).get("momentum"):
        declared = sm.roles.momentum
        if declared:
            return [list(t) for t in zip(*declared)]
    rows = getattr(sm, "interpolate_to_3d", None)
    if rows is None:
        raise RuntimeError(
            "Wall: cannot derive momentum_field_indices — this SystemModel "
            "registers no 'interpolate_to_3d' (the [b,h,u,v,w,p] lift).  Pin "
            "the groups explicitly, e.g. Wall(momentum_field_indices=[[2,3]]).")
    rows = list(rows)
    state = list(sm.state)
    if len(rows) < 3:
        raise RuntimeError(
            f"Wall: interpolate_to_3d has only {len(rows)} rows — no "
            "horizontal velocity row to derive momentum slots from.")
    geom = set()
    for slot in (0, 1):                      # b, h → geometry, never momentum
        geom |= {i for i, s in enumerate(state)
                 if s in sympy.sympify(rows[slot]).free_symbols}
    per_direction = []
    for slot in (2, 3):                      # u, v → horizontal directions
        if slot >= len(rows):
            break
        free = sympy.sympify(rows[slot]).free_symbols
        idx = [i for i, s in enumerate(state) if s in free and i not in geom]
        if idx:
            per_direction.append(idx)
    if not per_direction:
        raise RuntimeError(
            "Wall: no horizontal velocity row of interpolate_to_3d depends on "
            f"any non-geometry state slot; state = {[str(s) for s in state]}.")
    n = len(per_direction[0])
    if any(len(g) != n for g in per_direction):
        raise RuntimeError(
            f"Wall: ragged momentum groups {per_direction} for state "
            f"{[str(s) for s in state]} — the x and y rows do not carry the "
            "same number of moments, so they cannot be paired into vectors.  "
            "Pin momentum_field_indices explicitly for this model.")
    if len(set(sum(per_direction, []))) != n * len(per_direction):
        raise RuntimeError(
            f"Wall: momentum groups {per_direction} share state slots between "
            "directions (tensorial/rotated ansatz) — pin "
            "momentum_field_indices explicitly for this model.")
    return [list(t) for t in zip(*per_direction)]


class _DerivedMomentumIndices:
    """Mixin: ``momentum_field_indices=None`` means DERIVE (ruling 9396582 —
    pinned-only, derive otherwise).  An explicit pin always wins."""

    def resolve(self, sm):
        sup = getattr(super(), "resolve", None)
        if sup is not None:
            sup(sm)
        if self.momentum_field_indices is None:
            self.momentum_field_indices = derive_momentum_field_indices(sm)
        return self

    def _require_indices(self):
        if self.momentum_field_indices is None:
            raise RuntimeError(
                f"{type(self).__name__}(tag={self.tag!r}) is unresolved: "
                "momentum_field_indices is None and no SystemModel was "
                "attached.  Attach via SystemModel.attach_boundary_conditions "
                "(or call .resolve(sm)), or pin the groups explicitly.")
        return self.momentum_field_indices


class CharacteristicWall(_DerivedMomentumIndices, Characteristic):
    """Wall built from the characteristic tool: the target is the MIRROR
    state (normal momentum reflected); only its incoming content is imposed,
    which enforces u·n → 0 at the face without over-specifying the outgoing
    fields.

    ``momentum_field_indices`` defaults to ``None`` = DERIVE from the model's
    variable registry at resolve time (see
    :func:`derive_momentum_field_indices`).  An explicit pin wins.
    """

    momentum_field_indices = param.List(default=None, allow_None=True)

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        q = ZArray(Q)
        out = ZArray(Q)
        for indices in self._require_indices():
            dim = len(indices)
            n_vec = Matrix(normal[:dim])
            momentum = Matrix([q[k] for k in indices])
            normal_momentum = momentum.dot(n_vec)
            mirrored = momentum - 2 * normal_momentum * n_vec
            for i, idx in enumerate(indices):
                out[idx] = mirrored[i]
        return out


class FromData(BoundaryCondition):
    """FromData. (class)."""
    prescribe_fields = param.Dict(default={})
    timeline = param.Array(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    """CharacteristicReflective. (class)."""
    R = param.Parameter(default=None)
    L = param.Parameter(default=None)
    D = param.Parameter(default=None)
    S = param.Parameter(default=None)
    M = param.Parameter(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        q = Matrix(Q)
        q_n = self.S @ q
        W_int = self.L @ q_n
        W_bc = W_int.copy()
        MW = self.M @ W_int
        for i in range(W_int.rows):
            lam = self.D[i, i]
            cond = sympy.GreaterThan(-lam, 0, evaluate=False)
            W_bc[i, 0] = sympy.Function("conditional")(cond, MW[i, 0], W_int[i, 0])
        q_n_bc = self.R @ W_bc
        q_bc = sympy.simplify(self.S.inv() @ q_n_bc)
        out = ZArray.zeros(len(q_bc))
        for i in range(len(q_bc)):
            out[i] = sympy.Function("conditional")(
                sympy.GreaterThan(q[0], 1e-4), q_bc[i, 0], q[i, 0]
            )
        return out


class Wall(_DerivedMomentumIndices, BoundaryCondition):
    """Free-slip / partially-permeable wall.

    ``momentum_field_indices`` is a list of index GROUPS, each group one
    momentum VECTOR across the mesh's horizontal directions
    (``[[q_x_0, q_y_0], [q_x_1, q_y_1], …]``).  It defaults to ``None``, which
    means DERIVE the groups from the model's own variable registry at resolve
    time — see :func:`derive_momentum_field_indices`.  An explicit pin always
    wins, for models with no registry (hand-built) or an exotic layout.

    The former default ``[[1, 2]]`` was correct only for the hand-built SWE
    layout ``[h, hu, hv]``; on the derived layout ``[b, h, q…]`` it selected
    the DEPTH as the first momentum component (ShapeError in 1-D, and in 2-D a
    silently negative ghost depth at a lake at rest against a wall).
    """
    momentum_field_indices = param.List(default=None, allow_None=True)
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        q = ZArray(Q)
        groups = self._require_indices()
        dim = len(groups[0])
        n_vec = Matrix(normal[:dim])
        out = ZArray(Q)
        momentum_list_wall = []
        for indices in groups:
            momentum = Matrix([q[k] for k in indices])
            normal_momentum_coef = momentum.dot(n_vec)
            transverse_momentum = momentum - normal_momentum_coef * n_vec
            momentum_wall = (
                self.wall_slip * transverse_momentum
                - (1 - self.permeability) * normal_momentum_coef * n_vec
            )
            momentum_list_wall.append(momentum_wall)
        for indices, momentum_wall in zip(groups, momentum_list_wall):
            for i, idx in enumerate(indices):
                out[idx] = (1 - self.blending) * momentum_wall[i] + self.blending * q[
                    idx
                ]
        return out

    def face_state(self, Q_face, Qaux_face, normal, parameters):
        """Wall: reflect normal momentum component of the reconstructed face value."""
        Q_wall = Q_face.copy()
        for indices in self._require_indices():
            dim = len(indices)
            n_vec = normal[:dim]
            mom = np.array([Q_face[k] for k in indices], dtype=float)
            normal_component = np.dot(mom, n_vec)
            for i, k in enumerate(indices):
                Q_wall[k] = (self.wall_slip * (mom[i] - normal_component * n_vec[i])
                             - (1 - self.permeability) * normal_component * n_vec[i])
        return Q_wall

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Wall: reflect normal momentum, copy scalars.

        Momentum decomposition:
            u_wall = slip * u_tangential - (1 - perm) * u_normal
        Scalars are extrapolated (copied from interior).
        """
        Q_inner = np.asarray(Q_inner, dtype=float)
        Q_wall = Q_inner.copy()
        for indices in self._require_indices():
            dim = len(indices)
            n_vec = np.asarray(normal[:dim], dtype=float)
            mom = np.array([Q_inner[k] for k in indices], dtype=float)
            normal_component = np.dot(mom, n_vec)
            for i, k in enumerate(indices):
                Q_wall[k] = (self.wall_slip * (mom[i] - normal_component * n_vec[i])
                             - (1 - self.permeability) * normal_component * n_vec[i])
        return Q_wall


class RoughWall(Wall):
    """RoughWall. (class)."""
    CsW = param.Number(default=0.5)
    Ks = param.Number(default=0.001)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        slip_length = dX * sympy.ln((dX * self.CsW) / self.Ks)
        f = dX / slip_length
        wall_slip = (1 - f) / (1 + f)
        original_slip = self.wall_slip
        self.wall_slip = wall_slip
        res = super().compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal
        )
        self.wall_slip = original_slip
        return res

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """RoughWall: Wall reflection with distance-dependent slip."""
        slip_length = d_face * np.log(max((d_face * self.CsW) / self.Ks, 1e-30))
        f = d_face / max(slip_length, 1e-30)
        original_slip = self.wall_slip
        self.wall_slip = (1 - f) / (1 + f)
        result = super().face_value(
            Q_inner, Qaux_inner, normal, d_face, time, parameters
        )
        self.wall_slip = original_slip
        return result


class Periodic(BoundaryCondition):
    """Periodic. (class)."""
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        return ZArray(Q)


class WindStress(BoundaryCondition):
    """Mixed Neumann + Dirichlet BC — POM-style wind-stress surface.

    Some state rows get a prescribed *value* (Dirichlet); others get a
    prescribed *face-normal gradient* (Neumann).  Rows mentioned in
    neither default to zero-gradient extrapolation.  Symbolic specs in
    state / aux / parameter symbols are supported on the codegen and
    Firedrake-DG paths (boundary kernels lambdify the resulting
    expressions); the numpy paths require numeric specs.

    Built for the canonical POM surface BC (BM87 §2.15)::

        ρ_o K_M ∂_z U = τ_o            ← Neumann on the velocity row
        ∂_z T = 0                       ← Neumann on the tracer row
        q² = B₁^(2/3) u_τs²             ← Dirichlet on the q² row
        q²ℓ = 0                         ← Dirichlet on the q²ℓ row

    For the velocity Neumann gradient ``∂_z U = u_*² / K_M`` the user
    pulls the symbolic ``K_M`` from the model so the BC stays
    consistent with the interior closure::

        sm = SystemModel.from_model(my_col)
        K_M_sym = sm.diffusion_matrix[0, 0, 0, 0]
        bc = WindStress(
            tag="surface",
            prescribe_gradients={0: sm.parameters.u_star**2 / K_M_sym,
                                 1: 0},
            prescribe_values={2: sm.parameters.B1 ** sp.Rational(2, 3)
                                * sm.parameters.u_star**2,
                              3: 0},
        )

    Same mechanism handles the bottom BC (no-slip + zero heat flux +
    bottom-stress q² + zero q²ℓ) with different per-row specs.
    """

    prescribe_values = param.Dict(default={})
    prescribe_gradients = param.Dict(default={})

    @staticmethod
    def _coerce(v):
        if isinstance(v, sympy.Basic):
            return v
        return sympy.sympify(v)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Dirichlet rows take the prescribed value; others extrapolate."""
        out = ZArray(Q)
        for k, spec in self.prescribe_values.items():
            out[k] = self._coerce(spec)
        return out

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        """Neumann rows take the prescribed gradient; others zero."""
        n_vars = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        out = ZArray.zeros(n_vars)
        for k, spec in self.prescribe_gradients.items():
            out[k] = self._coerce(spec)
        return out

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Dirichlet rows ``2v − Q``, Neumann rows ``Q + 2·dX·g``, the rest
        extrapolate."""
        out = ZArray(Q)
        q = ZArray(Q)
        for k, spec in self.prescribe_values.items():
            out[k] = 2 * self._coerce(spec) - q[k]
        for k, spec in self.prescribe_gradients.items():
            out[k] = q[k] + 2 * dX * self._coerce(spec)
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy: Dirichlet rows get the prescribed value, others extrapolate.
        Only numeric specs work here; symbolic specs need to flow through
        the codegen / Firedrake-DG path."""
        Q_out = np.asarray(Q_inner, dtype=float).copy()
        for k, spec in self.prescribe_values.items():
            try:
                Q_out[k] = float(spec)
            except (ValueError, TypeError) as e:
                raise NotImplementedError(
                    f"WindStress.face_value: symbolic spec {spec!r} for "
                    f"row {k} is not lowerable through the numpy path; use "
                    "compute_boundary_condition / codegen instead."
                ) from e
        return Q_out

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face,
                      time, parameters):
        """Numpy: Neumann rows get the prescribed gradient; others zero.
        Numeric specs only (same caveat as ``face_value``)."""
        out = np.zeros_like(np.asarray(Q_inner, dtype=float))
        for k, spec in self.prescribe_gradients.items():
            try:
                out[k] = float(spec)
            except (ValueError, TypeError) as e:
                raise NotImplementedError(
                    f"WindStress.face_gradient: symbolic spec {spec!r} for "
                    f"row {k} is not lowerable through the numpy path."
                ) from e
        return out


# --- System-Aware Boundary Conditions ---
#
# These are applied via system.boundary_conditions.apply(SystemWall(), tag="right").
# They dispatch per equation type: scalar → Extrapolation, momentum → reflection.


class SystemExtrapolation:
    """Apply Extrapolation to all equations in the system."""

    def __init__(self, tag=None):
        self.tag = tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(eq_name, Extrapolation(tag=t), tag=t)


class SystemPeriodic:
    """Apply Periodic to all equations in the system."""

    def __init__(self, tag=None, periodic_to_physical_tag=""):
        self.tag = tag
        self.periodic_to_physical_tag = periodic_to_physical_tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(
                eq_name,
                Periodic(tag=t, periodic_to_physical_tag=self.periodic_to_physical_tag),
                tag=t,
            )


class SystemWall:
    """System-aware wall BC: Extrapolation for scalars, reflection for momentum.

    Applied via ``system.boundary_conditions.apply(SystemWall(), tag="right")``.
    The Wall BC holds a reference to the system and reads its equations
    to determine scalar vs momentum fields automatically.

    Parameters
    ----------
    tag : str
        Boundary tag (e.g. "right", "bottom").
    permeability : float
        0 = impermeable (default), 1 = fully permeable.
    wall_slip : float
        1 = free-slip (default), 0 = no-slip.
    """

    def __init__(self, tag=None, permeability=0.0, wall_slip=1.0):
        self.tag = tag
        self.permeability = permeability
        self.wall_slip = wall_slip

    def apply_to_system_bcs(self, system_bcs, tag=None, system=None):
        t = tag or self.tag
        momentum_eqs = [n for n in system_bcs.equation_names if "momentum" in n]
        scalar_eqs = [n for n in system_bcs.equation_names if n not in momentum_eqs]

        # Scalars: extrapolation WITHOUT gradient (preserves hydrostatic balance)
        for eq_name in scalar_eqs:
            system_bcs.set(eq_name, Extrapolation(tag=t, use_gradient=False), tag=t)

        # Momentum: wall reflection WITH gradient (for O2 boundary accuracy)
        wall_bc = WallMomentumBC(
            tag=t,
            system_bcs=system_bcs,
            permeability=self.permeability,
            wall_slip=self.wall_slip,
            use_gradient=True,
        )
        for eq_name in momentum_eqs:
            system_bcs.set(eq_name, wall_bc, tag=t)


class WallMomentumBC:
    """Wall BC for momentum equations — reads the system to build reflection.

    Holds a reference to the ``SystemBoundaryConditions`` (and thus knows
    which equations exist).  At compile time, determines the momentum
    vector grouping automatically from the equation names.

    The normal/tangential decomposition works for any system derived from
    INS: SWE (hu), SME (hu0, hu1, ...), VAM (hu, hv, hw moments), full INS.
    """

    def __init__(self, tag, system_bcs, permeability=0.0, wall_slip=1.0,
                 use_gradient=True):
        self.tag = tag
        self._system_bcs = system_bcs
        self.permeability = permeability
        self.wall_slip = wall_slip
        self.use_gradient = use_gradient

    @property
    def momentum_equations(self):
        """Momentum equations in the current system."""
        return [n for n in self._system_bcs.equation_names if "momentum" in n]

    def __repr__(self):
        return (f"WallMomentumBC(eqs={self.momentum_equations}, "
                f"perm={self.permeability}, slip={self.wall_slip})")


# --- Compiler: System BCs → legacy BoundaryConditions ---


def compile_system_bcs(system_bcs, equation_variable_map, dimension):
    """Translate system-aware BCs into the legacy BoundaryConditions container.

    Reads per-equation, per-tag BCs from ``system_bcs`` and produces
    a ``BoundaryConditions`` with one entry per tag. The
    ``equation_variable_map`` maps equation names to variable indices
    so the Wall BC knows which indices form the momentum vector.

    Parameters
    ----------
    system_bcs : SystemBoundaryConditions
    equation_variable_map : dict
        ``{equation_name: [var_index, ...]}``
    dimension : int
        Model dimension (1 or 2 for horizontal).

    Returns
    -------
    BoundaryConditions
    """
    bc_list = []
    for tag in system_bcs.tags:
        bcs_for_tag = system_bcs.get_all(tag)
        if not bcs_for_tag:
            continue

        # Check if any equation has a WallMomentumBC for this tag
        wall_bc = None
        all_extrap = True
        all_periodic = True
        for eq_name, bc in bcs_for_tag.items():
            if isinstance(bc, WallMomentumBC):
                wall_bc = bc
                all_extrap = False
                all_periodic = False
            elif isinstance(bc, Periodic):
                all_extrap = False
            elif isinstance(bc, Extrapolation):
                all_periodic = False
            else:
                all_extrap = False
                all_periodic = False

        if all_periodic:
            bc_obj = next(iter(bcs_for_tag.values()))
            bc_list.append(Periodic(
                tag=tag,
                periodic_to_physical_tag=getattr(bc_obj, 'periodic_to_physical_tag', ''),
            ))
        elif wall_bc is not None:
            # Build momentum_field_indices from equation_variable_map.
            # Group momentum equation indices by moment index:
            # For level=1, 1D: x_momentum = [2, 3] → [[2], [3]]
            # For level=1, 2D: x_momentum = [2, 3], y_momentum = [4, 5]
            #   → [[2, 4], [3, 5]]  (pairs for normal/tangential)
            mom_eqs = wall_bc.momentum_equations
            mom_indices_per_eq = [equation_variable_map.get(eq, []) for eq in mom_eqs]

            if len(mom_eqs) == 0:
                # No momentum equations — just extrapolation
                bc_list.append(Extrapolation(tag=tag))
            elif len(mom_eqs) == 1:
                # 1D: each variable is its own "vector" (scalar momentum)
                bc_list.append(Wall(
                    tag=tag,
                    momentum_field_indices=[[idx] for idx in mom_indices_per_eq[0]],
                    permeability=wall_bc.permeability,
                    wall_slip=wall_bc.wall_slip,
                ))
            else:
                # 2D+: group by moment index across equations
                n_per_eq = len(mom_indices_per_eq[0])
                groups = []
                for k in range(n_per_eq):
                    group = [indices[k] for indices in mom_indices_per_eq
                             if k < len(indices)]
                    groups.append(group)
                bc_list.append(Wall(
                    tag=tag,
                    momentum_field_indices=groups,
                    permeability=wall_bc.permeability,
                    wall_slip=wall_bc.wall_slip,
                ))
        elif all_extrap:
            bc_list.append(Extrapolation(tag=tag))
        else:
            # Mixed — default to extrapolation
            bc_list.append(Extrapolation(tag=tag))

    result = BoundaryConditions(bc_list)

    # Build per-tag gradient variable indices: which variables should get
    # gradient extrapolation at each boundary tag.
    grad_indices = {}
    for tag in system_bcs.tags:
        indices = []
        for eq_name, bc in system_bcs.get_all(tag).items():
            if getattr(bc, 'use_gradient', True):
                indices.extend(equation_variable_map.get(eq_name, []))
        grad_indices[tag] = sorted(set(indices))
    result._gradient_variable_indices = grad_indices

    return result


# --- Container Class ---


class BoundaryConditions(param.Parameterized):
    """BoundaryConditions. (class)."""
    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    def __init__(self, boundary_conditions=None, **params):
        """Initialize the instance."""
        if boundary_conditions is not None:
            params["boundary_conditions_list"] = boundary_conditions
        elif "boundary_conditions" in params:
            params["boundary_conditions_list"] = params.pop("boundary_conditions")
        super().__init__(**params)
        if self.boundary_conditions_list:
            self.boundary_conditions_list.sort(key=lambda bc: bc.tag)
        self._boundary_functions = [
            bc.compute_boundary_condition for bc in self.boundary_conditions_list
        ]
        self._boundary_gradient_functions = [
            bc.compute_boundary_gradient for bc in self.boundary_conditions_list
        ]
        self._boundary_tags = [bc.tag for bc in self.boundary_conditions_list]

    @property
    def list_sorted_function_names(self):
        """List sorted function names."""
        return self._boundary_tags

    @property
    def boundary_conditions_list_dict(self):
        """Boundary conditions list dict."""
        return {bc.tag: bc for bc in self.boundary_conditions_list}

    # [FIX] Added 'function_name' argument with default "boundary_conditions"
    def get_boundary_condition_function(
        self,
        time,
        X,
        dX,
        Q,
        Qaux,
        parameters,
        normal,
        function_name="boundary_conditions",
    ):
        """Get boundary condition function."""
        bc_idx = sympy.Symbol("bc_idx", integer=True)

        if not self._boundary_functions:
            bc_func_expr = ZArray(Q.get_list())
        else:
            conditions = []
            for i, func in enumerate(self._boundary_functions):
                res = func(
                    time,
                    X.get_list(),
                    dX,
                    Q.get_list(),
                    Qaux.get_list(),
                    parameters.get_list(),
                    normal.get_list(),
                )
                conditions.append((res, sympy.Eq(bc_idx, i)))

            bc_func_expr = sympy.Piecewise(*conditions)

        # [FIX] Use the passed name here
        func = Function(
            name=function_name,
            args=Zstruct(
                idx=bc_idx,
                time=time,
                position=X,
                distance=dX,
                variables=Q,
                aux_variables=Qaux,
                parameters=parameters,
                normal=normal,
            ),
            definition=bc_func_expr,
        )
        return func

    def get_boundary_gradient_function(
        self,
        time,
        X,
        dX,
        Q,
        Qaux,
        parameters,
        normal,
        function_name="boundary_gradients",
    ):
        """Indexed symbolic kernel for the boundary face-normal gradient
        ``∂Q/∂n``.

        Same shape and contract as
        :meth:`get_boundary_condition_function`: a ``Function`` whose
        first argument is the BC index, with a ``Piecewise`` body
        dispatching to each BC subclass's
        :meth:`BoundaryCondition.compute_boundary_gradient`.  The
        default per BC is zero Neumann (matches ``Extrapolation``'s
        gradient); subclasses override when a non-zero gradient is
        part of the BC."""
        bc_idx = sympy.Symbol("bc_idx", integer=True)

        if not self._boundary_gradient_functions:
            grad_expr = ZArray.zeros(len(Q.get_list()))
        else:
            conditions = []
            for i, func in enumerate(self._boundary_gradient_functions):
                res = func(
                    time,
                    X.get_list(),
                    dX,
                    Q.get_list(),
                    Qaux.get_list(),
                    parameters.get_list(),
                    normal.get_list(),
                )
                conditions.append((res, sympy.Eq(bc_idx, i)))
            grad_expr = sympy.Piecewise(*conditions)

        return Function(
            name=function_name,
            args=Zstruct(
                idx=bc_idx,
                time=time,
                position=X,
                distance=dX,
                variables=Q,
                aux_variables=Qaux,
                parameters=parameters,
                normal=normal,
            ),
            definition=grad_expr,
        )


    def ghost_state_branches(self, time, X, dX, Q, Qaux, parameters, normal,
                             bdata):
        """Per-branch symbolic GHOST states (decision D5): one ``ZArray`` per
        sorted BC (branch ``i`` serves tag ``list_sorted_function_names[i]``).
        An empty container yields the single extrapolation branch."""
        args = (time, X.get_list(), dX, Q.get_list(), Qaux.get_list(),
                parameters.get_list(), normal.get_list(), bdata.get_list())
        if not self.boundary_conditions_list:
            return [ZArray(Q.get_list())]
        return [ZArray(bc.compute_ghost_state(*args))
                for bc in self.boundary_conditions_list]

    def get_boundary_ghost_function(self, time, X, dX, Q, Qaux, parameters,
                                    normal, bdata,
                                    function_name="boundary_ghost_state"):
        """Indexed symbolic GHOST-state kernel — same contract as
        :meth:`get_boundary_condition_function` plus the per-face boundary
        data group ``boundary_data`` (coupling), returning the D5 ghost of the
        BC branch ``idx``."""
        bc_idx = sympy.Symbol("bc_idx", integer=True)
        branches = self.ghost_state_branches(time, X, dX, Q, Qaux, parameters,
                                             normal, bdata)
        if len(branches) == 1 and not self.boundary_conditions_list:
            expr = branches[0]
        else:
            expr = sympy.Piecewise(*[(res, sympy.Eq(bc_idx, i))
                                     for i, res in enumerate(branches)])
        return Function(
            name=function_name,
            args=Zstruct(idx=bc_idx, time=time, position=X, distance=dX,
                         variables=Q, aux_variables=Qaux,
                         parameters=parameters, normal=normal,
                         boundary_data=bdata),
            definition=expr,
        )


def boundary_data_symbols(n_state):
    """The per-face boundary-data group ``bdata_0 … bdata_{n-1}`` of the
    ghost-state kernel (coupling data, one entry per state row)."""
    z = Zstruct(**{f"bdata_{i}": sympy.Symbol(f"bdata_{i}", real=True)
                   for i in range(int(n_state))})
    z._symbolic_name = "bdata"
    return z


def build_boundary_ghost_kernel(sm):
    """The D5 GHOST-state kernel of ``sm`` as ``(function, branches)``.

    ``function`` is the indexed ``boundary_ghost_state(idx, time, position,
    distance, variables, aux_variables, parameters, normal, boundary_data)``
    kernel; ``branches[i]`` is the ghost ``ZArray`` of BC branch ``i`` (the
    sorted-tag order :func:`boundary_face_branches` routes to), over the SAME
    argument symbols (``function.args``).

    Built from the system's own ``BoundaryConditions`` container
    (``sm._bc_source``) in the system's CURRENT symbols, so an NSM gets its
    assumption-free state and its registered aux.  A system carrying a bare
    kernel without its container falls back to that kernel's branches,
    read as ghosts."""
    bcs = getattr(sm, "_bc_source", None)
    bdata = boundary_data_symbols(sm.n_state)
    if isinstance(bcs, BoundaryConditions):
        X = sm.position_struct()
        dX = sympy.Symbol("distance", real=True)
        args = (sm.time, X, dX, sm.variables, sm.aux_variables, sm.parameters,
                sm.normal, bdata)
        fn = bcs.get_boundary_ghost_function(*args)
        branches = bcs.ghost_state_branches(*args)
        return fn, branches
    kernel = getattr(sm, "boundary_conditions", None)
    if kernel is None:
        raise ValueError("build_boundary_ghost_kernel: the system carries no "
                         "boundary conditions")
    definition = kernel.definition
    if isinstance(definition, sympy.Piecewise):
        branches = [ZArray(list(sympy.flatten(e))) for e, _c in definition.args]
    else:
        branches = [ZArray(list(sympy.flatten(definition)))]
    a = kernel.args
    fargs = Zstruct(**{k: a[k] for k in a.keys()}, boundary_data=bdata)
    return Function(name="boundary_ghost_state", args=fargs,
                    definition=definition), branches


def boundary_face_branches(mesh, bcs):
    """BC-kernel branch index of every boundary face of ``mesh``, matched BY
    TAG NAME — the ``bc_idx`` every solver passes to the lowered
    ``boundary_conditions`` / ``boundary_gradients`` kernels.

    :class:`BoundaryConditions` sorts its BCs by tag, so kernel branch ``i``
    serves tag ``bcs.list_sorted_function_names[i]``.  The mesh numbers its
    tags on its own: ``mesh.boundary_face_function_numbers`` index
    ``mesh.boundary_conditions_sorted_names``, ordered by position on
    ``create_2d`` (left, right, bottom, top) and by gmsh physical id on
    ``from_msh``.  The raw mesh number is therefore NOT a branch index — read
    as one, ``create_2d`` sends ``bottom`` faces to the ``right`` BC.

    ``bcs is None`` (a SystemModel built from a bare kernel, without its
    ``BoundaryConditions`` source) carries no tag names; the mesh numbering is
    then the kernel's only contract and is returned unchanged.  So is an EMPTY
    container: its kernel has no branches (``ZArray(Q)``, extrapolation on
    every face) and never reads ``bc_idx``.  A boundary face whose tag has no
    BC, or a tag declared twice, raises -- except the reserved tag ``halo``
    (faces a distributed run's partition cuts, no BC): branch -1."""
    n_bf = int(mesh.n_boundary_faces)
    numbers = np.asarray(mesh.boundary_face_function_numbers, dtype=int)[:n_bf]
    tags = [] if bcs is None else list(bcs.list_sorted_function_names)
    if not tags or n_bf == 0:
        return numbers
    branch = {}
    for i, tag in enumerate(tags):
        if tag in branch:
            raise ValueError(
                f"boundary tag {tag!r} is declared by more than one BC "
                f"branch {tags}; give each tag one BoundaryCondition")
        branch[tag] = i
    names = list(mesh.boundary_conditions_sorted_names)
    used = sorted(set(numbers.tolist()))
    if used and used[-1] >= len(names):
        raise ValueError(
            f"mesh boundary faces use tag numbers {used} but the mesh names "
            f"only {names}; cannot route them to BCs by name")
    missing = [names[k] for k in used
               if names[k] not in branch and names[k] != "halo"]
    if missing:
        raise ValueError(
            f"mesh boundary tags {missing} have no boundary condition "
            f"(BCs are given for {tags})")
    to_branch = np.array([branch.get(nm, -1) for nm in names], dtype=int)
    return to_branch[numbers]


# ─────────────────────────────────────────────────────────────────────────────
# Per-field boundary conditions — the flat-list interface.
#
#   bcs = [Wall("left", on="momentum"),     # momentum slots reflect
#          Extrapolation("left", on="h"),   # depth extrapolates
#          Extrapolation("right")]          # on="all" → every slot
#   SME(level=2, closures=[...], boundary_conditions=bcs)
#
# Different fields get different BCs at the SAME tag.  `on` is resolved
# GENERICALLY against the model's declared state (any future field — tracer,
# energy, temperature — is addressable by its name), unclaimed slots default to
# Extrapolation, and two BCs claiming the same (tag, field) is an error.
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_on(on, state_names, aliases):
    """Resolve an ``on`` selector to a list of state-slot indices, generically.

    ``"all"`` → every slot; an exact field name → that slot; a family base
    (``"q"`` → ``q_0, q_1, …``); an alias (``"momentum"`` → ``"q"``)."""
    on = aliases.get(on, on)
    if on in ("all", "", None):
        return list(range(len(state_names)))
    if on in state_names:
        return [state_names.index(on)]
    fam = [i for i, nm in enumerate(state_names)
           if nm == on or nm.startswith(on + "_")]
    if fam:
        return fam
    raise ValueError(
        f"boundary condition on={on!r}: not a state field, family base, or "
        f"alias. Known fields: {state_names}; aliases: {sorted(aliases)}")


def _momentum_groups(slots, state_names):
    """Group momentum slots into VECTORS per moment level — dimension-agnostic.

    Slots sharing a trailing moment index are the (x[, y[, z]]) components of ONE
    momentum vector; a :class:`Wall` decomposes each group into normal+transverse
    w.r.t. the face normal and reflects ONLY the normal component (transverse
    keeps slip).  1-D → one component per group (``[[q_0],[q_1],…]`` → each
    reflects); 2-D/3-D → ``[[q_x_i, q_y_i, …], …]`` ordered by direction name."""
    import re
    by_level = {}
    for s in slots:
        nm = state_names[s]
        m = re.search(r"(\d+)$", nm)
        lvl = int(m.group(1)) if m else 0
        by_level.setdefault(lvl, []).append(s)
    return [sorted(by_level[l], key=lambda s: state_names[s])
            for l in sorted(by_level)]


class PerFieldBoundary(BoundaryCondition):
    """Composite BC for one tag: each state slot delegates to the BC assigned to
    its field.  Built by :func:`resolve_per_field`; unclaimed slots fall back to
    :class:`Extrapolation`.  Reuses the existing BC engines unchanged — it just
    picks, per slot, the ghost value/gradient produced by that slot's BC."""

    def __init__(self, tag=None, slot_bc=None, n_state=0, **params):
        super().__init__(tag=tag, **params)
        self._slot_bc = slot_bc or {}        # {slot_index: BoundaryCondition}
        self._n = n_state

    def _per_slot(self, method, base, *args):
        cache = {}
        for slot, bc in self._slot_bc.items():
            key = id(bc)
            if key not in cache:
                cache[key] = getattr(bc, method)(*args)
            base[slot] = cache[key][slot]
        return base

    # symbolic kernels (codegen path)
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return self._per_slot("compute_boundary_condition", ZArray(Q),
                              time, X, dX, Q, Qaux, parameters, normal)

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        return self._per_slot("compute_boundary_gradient", ZArray.zeros(n),
                              time, X, dX, Q, Qaux, parameters, normal)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        return self._per_slot("compute_ghost_state", ZArray(Q),
                              time, X, dX, Q, Qaux, parameters, normal, bdata)

    # numeric kernels (numpy path)
    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        return self._per_slot("face_value",
                              np.asarray(Q_inner, dtype=float).copy(),
                              Q_inner, Qaux_inner, normal, d_face, time, parameters)

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters):
        return self._per_slot("face_gradient",
                              np.zeros_like(np.asarray(Q_inner, dtype=float)),
                              Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters)


def resolve_per_field(bc_list, state_names, aliases=None):
    """Build a :class:`BoundaryConditions` from a FLAT LIST of per-field BCs.

    For each tag, group the ``(tag, on)`` BCs into a :class:`PerFieldBoundary`:
    every state slot is served by the BC whose ``on`` covers it (a :class:`Wall`
    gets its ``momentum_field_indices`` set to the slots it owns so it reflects
    exactly those); unclaimed slots default to :class:`Extrapolation`; two BCs on
    the same ``(tag, field)`` raise."""
    aliases = aliases or {}
    n = len(state_names)
    by_tag = {}
    for bc in bc_list:
        by_tag.setdefault(bc.tag, []).append(bc)
    out = []
    # WHOLE-PATCH BCs own the entire boundary and are detected by TYPE at the
    # tag level by the runtime (Periodic → mesh.resolve_periodic_bcs;
    # Coupled → the preCICE/OpenFOAM path).  They must pass through UNWRAPPED —
    # composing them into a PerFieldBoundary would hide them from those
    # detectors.  They cannot be mixed with per-field BCs at the same tag.
    whole_patch = (Periodic, Coupled)
    for tag, bcs in by_tag.items():
        wp = [bc for bc in bcs if isinstance(bc, whole_patch)]
        if wp:
            if len(bcs) > 1:
                raise ValueError(
                    f"tag {tag!r}: {type(wp[0]).__name__} is a whole-patch "
                    f"boundary condition — it cannot be combined with per-field "
                    f"BCs at the same tag")
            out.append(wp[0])                 # pass through, unwrapped
            continue
        slot_bc = {}
        for bc in bcs:
            slots = _resolve_on(bc.on, state_names, aliases)
            if isinstance(bc, Wall):
                # group momentum components into VECTORS per moment level so the
                # Wall decomposes into normal/transverse and reflects only the
                # NORMAL component — dimension-agnostic (1-D, 2-D, 3-D).
                bc.momentum_field_indices = _momentum_groups(slots, state_names)
            for s in slots:
                if s in slot_bc and slot_bc[s] is not bc:
                    raise ValueError(
                        f"conflicting boundary conditions on tag {tag!r} field "
                        f"{state_names[s]!r}: {type(slot_bc[s]).__name__} and "
                        f"{type(bc).__name__}")
                slot_bc[s] = bc
        default = Extrapolation(tag=tag)
        for s in range(n):
            slot_bc.setdefault(s, default)
        out.append(PerFieldBoundary(tag=tag, slot_bc=slot_bc, n_state=n))
    return BoundaryConditions(out)


def resolve_and_attach(sm, boundary_conditions, aux_bcs=None, aliases=None):
    """Attach boundary conditions to a built SystemModel, accepting EITHER the
    legacy :class:`BoundaryConditions` container OR the NEW flat per-field list
    (``[Wall("left", on="momentum"), …]``), which is resolved against ``sm``'s
    declared state via :func:`resolve_per_field`.  Shared by every model's
    ``system_model``."""
    if isinstance(boundary_conditions, list):
        boundary_conditions = resolve_per_field(
            boundary_conditions, [str(s) for s in sm.state],
            aliases if aliases is not None else {"momentum": "q"})
    if boundary_conditions is not None:
        sm.attach_boundary_conditions(boundary_conditions, aux_bcs=aux_bcs)
