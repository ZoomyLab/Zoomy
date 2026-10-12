"""Boundary conditions of the Model layer: per-tag BC classes, the BoundaryConditions container
and the kernels it lowers to. They are attached to a SystemModel, which turns them into the
boundary / ghost-state kernels the solver templates call.
"""

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
    """Wrap a BC producer so its result is written only to the slots in ``self._on_slots``
    (None: owns all, pass through).
    """
    @functools.wraps(fn)
    def wrapped(self, *args, **kw):
        """Call the producer and mask its result to the owned slots."""
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
    """Piecewise-linear sympy interpolation of ``data`` over ``timeline``, constant outside the
    range.
    """
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
    """Base class of a boundary condition: a ``tag`` (patch) and an ``on`` selector naming the
    state field(s) it owns. A flat list of BCs gives different fields different BCs at one tag;
    ``on`` is resolved against the model state by :func:`resolve_per_field`.
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
        """Wrap every value/gradient producer of a subclass so it honours the ``on=`` mask."""
        super().__init_subclass__(**kw)
        for name, (idx, is_grad, symbolic) in _BC_ON_METHODS.items():
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_on_wrapped", False):
                setattr(cls, name, _wrap_on_method(fn, idx, is_grad, symbolic))

    def __init__(self, tag=None, on=None, **params):
        """Set ``tag`` and ``on`` (both optional) and forward the remaining parameters."""
        if tag is not None:
            params["tag"] = tag
        if on is not None:
            params["on"] = on
        super().__init__(**params)

    def bind_on(self, state_names, aliases=None):
        """Resolve ``on`` to the owned state-slot indices (None: all, no masking); called by
        ``SystemModel.attach_boundary_conditions``.
        """
        on = self.on or "all"
        if on in ("all", ""):
            self._on_slots = None
        else:
            self._on_slots = _resolve_on(
                on, list(state_names), aliases or {"momentum": "q"})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value (the outer face state seen by the Riemann problem), assembled
        into the indexed ``boundary_conditions`` kernel.
        """
        raise NotImplementedError(
            "BoundaryCondition is a virtual class. Use one of its derived classes!"
        )

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic face-normal gradient ``dQ/dn`` (default: zero Neumann), assembled into the
        ``boundary_gradients`` kernel.
        """
        n_vars = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        return ZArray.zeros(n_vars)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Symbolic ghost state at the mirror image of the inner cell centre; a
        ``boundary_value_kind == "face"`` value ``v`` gives ``2v - Q``.

        Args:
            bdata: per-face boundary data (coupling), read only by subclasses that need it.
        """
        v = ZArray(self.compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal))
        if self.boundary_value_kind == "face":
            q = ZArray(Q)
            return ZArray([2 * v[i] - q[i] for i in range(len(q))])
        return v

    def face_state(self, Q_face, Qaux_face, normal, parameters):
        """Deprecated numpy boundary-side Riemann state from the reconstructed face value
        (default: copy); use ``face_value``.
        """
        return Q_face.copy()

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy boundary-side Riemann state of the adjacent interior cell (default: copy of
        ``Q_inner``).

        Args:
            d_face: distance from the interior cell centre to the face centre.

        Returns:
            State of shape ``(n_vars,)`` seen by the Riemann solver on the boundary side.
        """
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face,
                      time, parameters):
        """Numpy face-normal gradient ``dQ/dn`` for the diffusive flux (default:
        ``(Q_face - Q_inner) / d_face``).

        Args:
            Q_face: the already computed ``face_value`` result.

        Returns:
            Gradient of shape ``(n_vars,)``.
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
    """Ghost state equals the interior state (zero normal gradient)."""
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value: the inner state."""
        return ZArray(Q)

    def face_state(self, Q_face, Qaux_face, normal, parameters):
        """Boundary state equals the interior face state."""
        return Q_face.copy()

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Boundary state equals the interior cell state."""
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face,
                      time, parameters):
        """Zero normal gradient."""
        return np.zeros_like(np.asarray(Q_inner, dtype=float))


class ZeroNeumann(Extrapolation):
    """Named alias of :class:`Extrapolation` (``dQ/dn = 0``)."""


class Dirichlet(BoundaryCondition):
    """Prescribed value on the field(s) selected by ``on=``, e.g.
    ``Dirichlet("left", on="h", value=1.5)``.
    """

    value = param.Number(default=0.0)
    boundary_value_kind = "face"

    def __init__(self, tag=None, on=None, value=None, **params):
        """Set the prescribed ``value`` besides ``tag`` and ``on``."""
        if value is not None:
            params["value"] = value
        super().__init__(tag=tag, on=on, **params)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value: ``value`` in every slot."""
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        out = ZArray(Q)
        for i in range(n):
            out[i] = self.value
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy boundary state: ``value`` in every slot."""
        return np.full(np.asarray(Q_inner, dtype=float).shape, float(self.value))


class Flux(BoundaryCondition):
    """Prescribed normal gradient ``dQ/dn = gradient`` on the field(s) selected by ``on=``; the
    ghost value extrapolates. ``gradient`` is a number, a string formula in field/parameter
    names (e.g. Navier slip ``"lambda_s*h*mom/nu"``), a sympy expression or a numpy callable;
    names are bound by :meth:`resolve`.
    """

    gradient = param.Parameter(default=0.0)

    def __init__(self, tag=None, on=None, gradient=None, **params):
        """Set the ``gradient`` specification besides ``tag`` and ``on``."""
        if gradient is not None:
            params["gradient"] = gradient
        super().__init__(tag=tag, on=on, **params)
        self._resolved_grad = None     # sympy expr in sm symbols (codegen)
        self._grad_lambda = None       # numpy evaluator (state, aux, params)

    def resolve(self, sm):
        """Bind a string / sympy ``gradient`` to the symbols of ``sm`` by name and lambdify it
        for the numpy path.
        """
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
        """Symbolic boundary value: the inner state."""
        return ZArray(Q)                      # value extrapolates

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Ghost of the prescribed gradient ``g``: ``Q + 2*dX*g``."""
        q = ZArray(Q)
        g = ZArray(self.compute_boundary_gradient(
            time, X, dX, Q, Qaux, parameters, normal))
        return ZArray([q[i] + 2 * dX * g[i] for i in range(len(q))])

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic prescribed gradient in every slot."""
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        out = ZArray.zeros(n)
        g = self._resolved_grad if self._resolved_grad is not None else self.gradient
        g = g if isinstance(g, sympy.Basic) else sympy.sympify(
            g if not callable(g) else 0)
        for i in range(n):
            out[i] = g
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy boundary state: the interior cell state."""
        return np.asarray(Q_inner, dtype=float).copy()

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters):
        """Numpy prescribed gradient, evaluated at the interior cell."""
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
    """preCICE-coupled boundary patch; the kernel is an extrapolation fallback that the solver
    overwrites with the exchanged data at runtime.

    Args:
        mesh_name: preCICE mesh name of the patch (as in ``precice-config.xml``).
    """

    mesh_name = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Fallback boundary value (extrapolation), overwritten by preCICE at runtime."""
        return ZArray(Q)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Ghost equals the per-face coupling data ``bdata`` (the inner state if absent)."""
        if bdata is None:
            return ZArray(Q)
        return ZArray(list(bdata))

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Fallback face value: the interior cell state."""
        return np.asarray(Q_inner, dtype=float).copy()


class InflowOutflow(BoundaryCondition):
    """Prescribe selected state fields, extrapolate the rest. ``prescribe_fields`` maps a state
    index to a value (scalar or expression) or to a dict with ``mode`` in ``replace`` (value),
    ``inlet_outlet`` (value only where inflowing) or ``blend`` (soft pull to ``target`` with
    ``weight``).
    """

    prescribe_fields = param.Dict(default={})
    boundary_value_kind = "face"

    @staticmethod
    def _coerce(v):
        """Pass expressions and numbers through; evaluate a string spec."""
        if isinstance(v, sympy.Basic):
            return v
        try:
            return float(v)
        except (ValueError, TypeError):
            return float(eval(v))

    def _symbolic_field(self, k, spec, Q):
        """Symbolic ghost value of state index ``k`` for its spec."""
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
        """Symbolic boundary value with the prescribed fields replaced."""
        Qout = ZArray(Q)
        for k, spec in self.prescribe_fields.items():
            Qout[k] = self._symbolic_field(k, spec, Q)
        return Qout

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy path: constant ``replace`` specs only; other modes go through the codegen path."""
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
    """Face values from user functions: ``prescribe_fields`` maps a slot to
    ``f(time, X, dX, Q, Qaux, p, n)`` at the face centre.
    """
    prescribe_fields = param.Dict(default={})
    boundary_value_kind = "face"

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value with the prescribed slots from the user functions."""
        Qout = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            Qout[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


class FromModel(BoundaryCondition):
    """Boundary values derived in the model: references a ``boundary:<name>`` function group
    registered by the derivation. :meth:`resolve` (called by
    ``SystemModel.attach_boundary_conditions``) binds it to ``sm.boundary_specs``; slots the
    definition does not prescribe extrapolate.
    """

    definition = param.String(default="", doc="name registered in the model "
                              "via register_group('boundary:<name>', …)")
    prescribe_fields = param.Dict(default={}, doc="state_index → sympy expr "
                                  "in the SystemModel's state symbols; "
                                  "filled by resolve()")

    def resolve(self, sm):
        """Take the named definition from ``sm.boundary_specs`` and capture the state and
        face-normal symbols.
        """
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
        """Symbolic boundary value: the model definition with state and normal symbols
        substituted.
        """
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
    """Incoming-characteristic ghost: only the characteristic fields with ``lambda < 0`` take
    the target state, ``Q + R 1_{lambda<0} L (Q_target - Q)``, so the Riemann problem reflects
    nothing. Built on the SystemModel's opaque ``eigensystem`` kernel; the target is the inner
    state unless overridden by ``prescribe_fields`` or :meth:`target_state`.
    """

    prescribe_fields = param.Dict(default={}, doc="state_index → callable("
                                  "time, X, dX, Q, Qaux, p, n) for the "
                                  "TARGET state in that slot")

    def resolve(self, sm):
        """Bind the state symbols, normal symbols and eigensystem of ``sm``."""
        self._state_syms = list(sm.state)
        self._normal_syms = list(sm.normal.values())
        self._es = [sympy.sympify(e) for e in sm.eigensystem]
        return self

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        """State whose incoming content is imposed (default: inner state overridden by
        ``prescribe_fields``).
        """
        target = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            target[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return target

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value from the eigendecomposition of the normal-projected system."""
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
    """Non-reflecting far-field boundary: incoming characteristics carry ``far_field``, outgoing
    ones leave undisturbed.
    """

    far_field = param.Parameter(default=None, doc="callable(n_state) → "
                                "far-field state vector, or a sequence")

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        """Far-field state vector (the inner state if ``far_field`` is None)."""
        if self.far_field is None:
            return ZArray(Q)
        vals = (self.far_field(len(self._state_syms))
                if callable(self.far_field) else self.far_field)
        target = ZArray(Q)
        for i, v in enumerate(vals):
            target[i] = v
        return target


def derive_momentum_field_indices(sm):
    """Derive the momentum index groups (one vector per group, across the horizontal directions)
    from the registry ``sm.interpolate_to_3d``, or from declared roles.

    Returns:
        Groups such as ``[[q_x_0, q_y_0], [q_x_1, q_y_1], ...]``; raises if the registry is
        missing or ragged.
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
    """Mixin: ``momentum_field_indices=None`` means derive at resolve time; an explicit pin
    wins.
    """

    def resolve(self, sm):
        """Derive ``momentum_field_indices`` from ``sm`` unless pinned."""
        sup = getattr(super(), "resolve", None)
        if sup is not None:
            sup(sm)
        if self.momentum_field_indices is None:
            self.momentum_field_indices = derive_momentum_field_indices(sm)
        return self

    def _require_indices(self):
        """Return the momentum groups, raising if still unresolved."""
        if self.momentum_field_indices is None:
            raise RuntimeError(
                f"{type(self).__name__}(tag={self.tag!r}) is unresolved: "
                "momentum_field_indices is None and no SystemModel was "
                "attached.  Attach via SystemModel.attach_boundary_conditions "
                "(or call .resolve(sm)), or pin the groups explicitly.")
        return self.momentum_field_indices


class CharacteristicWall(_DerivedMomentumIndices, Characteristic):
    """Wall from the characteristic tool: the target is the mirror state (normal momentum
    reflected) and only its incoming content is imposed. ``momentum_field_indices=None`` derives
    the groups at resolve time (see :func:`derive_momentum_field_indices`).
    """

    momentum_field_indices = param.List(default=None, allow_None=True)

    def target_state(self, time, X, dX, Q, Qaux, parameters, normal):
        """Inner state with the normal momentum of each group reflected."""
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
    """Face values from a time series: ``prescribe_fields`` maps a slot to data interpolated
    over ``timeline``.
    """
    prescribe_fields = param.Dict(default={})
    timeline = param.Array(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic ghost ``2*interp - Q`` for the prescribed slots."""
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    """Reflective boundary in characteristic variables: incoming ones (``D <= 0``) are mapped
    through ``M``. ``R, L, D`` are the user-supplied eigen-decomposition and ``S`` the rotation
    into the normal frame.
    """
    R = param.Parameter(default=None)
    L = param.Parameter(default=None)
    D = param.Parameter(default=None)
    S = param.Parameter(default=None)
    M = param.Parameter(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value: incoming characteristic variables reflected through ``M``,
        guarded against dry cells.
        """
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
    """Free-slip / partially permeable wall: per momentum vector,
    ``u_wall = slip * u_tangential - (1 - permeability) * u_normal``; scalars extrapolate.
    ``momentum_field_indices`` lists index groups (one vector per group); None derives them from
    the model at resolve time, an explicit pin wins.
    """
    momentum_field_indices = param.List(default=None, allow_None=True)
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic wall ghost: reflected normal momentum, slip on the tangential part, blended
        with the inner state by ``blending``.
        """
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
        """Numpy wall state from the reconstructed face value."""
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
        """Numpy wall state: reflected normal momentum, scalars copied from the interior."""
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
    """:class:`Wall` with a distance-dependent slip from a log-law roughness (``CsW``, ``Ks``)."""
    CsW = param.Number(default=0.5)
    Ks = param.Number(default=0.001)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic wall ghost with the roughness slip ``(1 - f) / (1 + f)``."""
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
        """Numpy wall state with the roughness slip."""
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
    """Periodic boundary; the runtime pairs it with ``periodic_to_physical_tag``
    (``mesh.resolve_periodic_bcs``), the kernel is an extrapolation fallback.
    """
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Fallback boundary value: the inner state."""
        return ZArray(Q)


class WindStress(BoundaryCondition):
    """Mixed Dirichlet and Neumann boundary (POM-style wind-stress surface):
    ``prescribe_values`` fixes rows, ``prescribe_gradients`` fixes face-normal gradients, other
    rows extrapolate. Symbolic specs work on the codegen and Firedrake-DG paths, the numpy paths
    need numbers.
    """

    prescribe_values = param.Dict(default={})
    prescribe_gradients = param.Dict(default={})

    @staticmethod
    def _coerce(v):
        """Sympify a spec value."""
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
        """Dirichlet rows ``2v - Q``, Neumann rows ``Q + 2*dX*g``, the rest extrapolate."""
        out = ZArray(Q)
        q = ZArray(Q)
        for k, spec in self.prescribe_values.items():
            out[k] = 2 * self._coerce(spec) - q[k]
        for k, spec in self.prescribe_gradients.items():
            out[k] = q[k] + 2 * dX * self._coerce(spec)
        return out

    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy: Dirichlet rows take the prescribed (numeric) value; others extrapolate."""
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
        """Numpy: Neumann rows take the prescribed (numeric) gradient; others zero."""
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
    """System-aware BC: :class:`Extrapolation` on every equation."""

    def __init__(self, tag=None):
        """Store the boundary ``tag``."""
        self.tag = tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        """Set :class:`Extrapolation` for every equation of ``system_bcs`` at the tag."""
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(eq_name, Extrapolation(tag=t), tag=t)


class SystemPeriodic:
    """System-aware BC: :class:`Periodic` on every equation."""

    def __init__(self, tag=None, periodic_to_physical_tag=""):
        """Store the ``tag`` and ``periodic_to_physical_tag``."""
        self.tag = tag
        self.periodic_to_physical_tag = periodic_to_physical_tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        """Set :class:`Periodic` for every equation of ``system_bcs`` at the tag."""
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(
                eq_name,
                Periodic(tag=t, periodic_to_physical_tag=self.periodic_to_physical_tag),
                tag=t,
            )


class SystemWall:
    """System-aware wall: extrapolation (without gradient) for scalar equations, reflection
    (:class:`WallMomentumBC`) for momentum equations.

    Args:
        permeability: 0 impermeable (default), 1 fully permeable.
        wall_slip: 1 free-slip (default), 0 no-slip.
    """

    def __init__(self, tag=None, permeability=0.0, wall_slip=1.0):
        """Store tag, permeability and wall slip."""
        self.tag = tag
        self.permeability = permeability
        self.wall_slip = wall_slip

    def apply_to_system_bcs(self, system_bcs, tag=None, system=None):
        """Set extrapolation on scalar equations and :class:`WallMomentumBC` on momentum
        equations at the tag.
        """
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
    """Wall BC for the momentum equations of a system; :func:`compile_system_bcs` groups their
    variables into momentum vectors.
    """

    def __init__(self, tag, system_bcs, permeability=0.0, wall_slip=1.0,
                 use_gradient=True):
        """Store the tag, the system BC container and the wall parameters."""
        self.tag = tag
        self._system_bcs = system_bcs
        self.permeability = permeability
        self.wall_slip = wall_slip
        self.use_gradient = use_gradient

    @property
    def momentum_equations(self):
        """Momentum equations of the current system."""
        return [n for n in self._system_bcs.equation_names if "momentum" in n]

    def __repr__(self):
        """Short summary of the momentum equations and wall parameters."""
        return (f"WallMomentumBC(eqs={self.momentum_equations}, "
                f"perm={self.permeability}, slip={self.wall_slip})")


# --- Compiler: System BCs → legacy BoundaryConditions ---


def compile_system_bcs(system_bcs, equation_variable_map, dimension):
    """Translate system-aware BCs into a :class:`BoundaryConditions` container with one BC per
    tag.

    Args:
        system_bcs: per-equation, per-tag BCs.
        equation_variable_map: ``{equation_name: [var_index, ...]}``, used to group momentum
            indices.
        dimension: horizontal model dimension (1 or 2).

    Returns:
        The container, with ``_gradient_variable_indices`` per tag.
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
    """Container of the per-tag BCs, sorted by tag; builds the indexed boundary, gradient and
    ghost-state kernels.
    """
    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    def __init__(self, boundary_conditions=None, **params):
        """Take the list of BCs (``boundary_conditions``), sort by tag and collect their
        kernels.
        """
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
        """Tags in kernel-branch order."""
        return self._boundary_tags

    @property
    def boundary_conditions_list_dict(self):
        """BCs keyed by tag."""
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
        """Indexed symbolic ``boundary_conditions`` kernel: a Piecewise over the BC index
        dispatching to each BC's boundary value.
        """
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
        """Indexed symbolic ``boundary_gradients`` kernel (face-normal gradient), same contract
        as :meth:`get_boundary_condition_function`.
        """
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
        """Per-BC symbolic ghost states (branch ``i`` serves tag ``i``); an empty container
        gives one extrapolation branch.
        """
        args = (time, X.get_list(), dX, Q.get_list(), Qaux.get_list(),
                parameters.get_list(), normal.get_list(), bdata.get_list())
        if not self.boundary_conditions_list:
            return [ZArray(Q.get_list())]
        return [ZArray(bc.compute_ghost_state(*args))
                for bc in self.boundary_conditions_list]

    def get_boundary_ghost_function(self, time, X, dX, Q, Qaux, parameters,
                                    normal, bdata,
                                    function_name="boundary_ghost_state"):
        """Indexed symbolic ghost-state kernel; as :meth:`get_boundary_condition_function` plus
        the per-face ``boundary_data`` group.
        """
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
    """Per-face boundary-data group ``bdata_0 .. bdata_{n-1}`` (coupling data, one entry per
    state row) of the ghost-state kernel.
    """
    z = Zstruct(**{f"bdata_{i}": sympy.Symbol(f"bdata_{i}", real=True)
                   for i in range(int(n_state))})
    z._symbolic_name = "bdata"
    return z


def build_boundary_ghost_kernel(sm):
    """Build the ghost-state kernel of ``sm`` from its ``BoundaryConditions`` source, in the
    system's current symbols.

    Returns:
        ``(function, branches)``: the indexed ``boundary_ghost_state`` kernel and the ghost
            ``ZArray`` per BC branch.
    """
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
    """BC-kernel branch index of every boundary face of ``mesh``, matched by tag name (the
    mesh's own tag numbers are not branch indices). A face whose tag has no BC raises, except
    the reserved ``halo`` tag (branch -1); without ``bcs`` the mesh numbering is returned
    unchanged.
    """
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
    """Resolve an ``on`` selector (``"all"``, field name, family base, alias) to state-slot
    indices.
    """
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
    """Group momentum slots into vectors per moment level (the x, y, z components of one
    moment), as :class:`Wall` expects.
    """
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
    """Composite BC for one tag: each state slot takes the value/gradient of the BC assigned to
    its field. Built by :func:`resolve_per_field`; unclaimed slots fall back to
    :class:`Extrapolation`.
    """

    def __init__(self, tag=None, slot_bc=None, n_state=0, **params):
        """Store the ``{slot: BC}`` map and the state size."""
        super().__init__(tag=tag, **params)
        self._slot_bc = slot_bc or {}        # {slot_index: BoundaryCondition}
        self._n = n_state

    def _per_slot(self, method, base, *args):
        """Fill ``base`` slot by slot from the result of ``method`` on each slot's BC."""
        cache = {}
        for slot, bc in self._slot_bc.items():
            key = id(bc)
            if key not in cache:
                cache[key] = getattr(bc, method)(*args)
            base[slot] = cache[key][slot]
        return base

    # symbolic kernels (codegen path)
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary value, per slot from the owning BC."""
        return self._per_slot("compute_boundary_condition", ZArray(Q),
                              time, X, dX, Q, Qaux, parameters, normal)

    def compute_boundary_gradient(self, time, X, dX, Q, Qaux, parameters, normal):
        """Symbolic boundary gradient, per slot from the owning BC."""
        n = len(Q.get_list()) if hasattr(Q, "get_list") else len(Q)
        return self._per_slot("compute_boundary_gradient", ZArray.zeros(n),
                              time, X, dX, Q, Qaux, parameters, normal)

    def compute_ghost_state(self, time, X, dX, Q, Qaux, parameters, normal,
                            bdata=None):
        """Symbolic ghost state, per slot from the owning BC."""
        return self._per_slot("compute_ghost_state", ZArray(Q),
                              time, X, dX, Q, Qaux, parameters, normal, bdata)

    # numeric kernels (numpy path)
    def face_value(self, Q_inner, Qaux_inner, normal, d_face, time, parameters):
        """Numpy boundary state, per slot from the owning BC."""
        return self._per_slot("face_value",
                              np.asarray(Q_inner, dtype=float).copy(),
                              Q_inner, Qaux_inner, normal, d_face, time, parameters)

    def face_gradient(self, Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters):
        """Numpy face gradient, per slot from the owning BC."""
        return self._per_slot("face_gradient",
                              np.zeros_like(np.asarray(Q_inner, dtype=float)),
                              Q_inner, Q_face, Qaux_inner, normal, d_face, time, parameters)


def resolve_per_field(bc_list, state_names, aliases=None):
    """Build a :class:`BoundaryConditions` from a flat list of per-field BCs: one
    :class:`PerFieldBoundary` per tag, unclaimed slots extrapolate, two BCs on one (tag, field)
    raise. Whole-patch BCs (:class:`Periodic`, :class:`Coupled`) pass through unwrapped.
    """
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
    """Attach boundary conditions (a :class:`BoundaryConditions` container or a flat per-field
    list) to the SystemModel ``sm``; shared by every model's ``system_model``.
    """
    if isinstance(boundary_conditions, list):
        boundary_conditions = resolve_per_field(
            boundary_conditions, [str(s) for s in sm.state],
            aliases if aliases is not None else {"momentum": "q"})
    if boundary_conditions is not None:
        sm.attach_boundary_conditions(boundary_conditions, aux_bcs=aux_bcs)
