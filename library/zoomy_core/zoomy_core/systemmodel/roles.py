"""Declared state roles of a `SystemModel` (depth, bed, stationary, positive, momentum,
gravity), stored as `sm.history` records of field names.
Generic code reads `Roles` instead of guessing from field names; a model with no declaration
gets no free-surface specialisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import sympy as sp

#: ``history`` record name of a role declaration.
ROLE_RECORD = "declare_roles"

_ROLE_KEYS = ("depth", "bed", "stationary", "positive", "momentum", "gravity")
_SIGN_PREDICATES = ("positive", "nonnegative")
_UNSET = object()


@dataclass(frozen=True)
class Roles:
    """Resolved roles: row indices into `sm.state` for depth, bed, stationary, positive and
    momentum rows, plus the gravity parameter name.
    """

    depth: Optional[int] = None
    bed: Optional[int] = None
    stationary: Tuple[int, ...] = ()
    positive: Tuple[int, ...] = ()
    momentum: Tuple[Tuple[int, ...], ...] = ()
    gravity: Optional[str] = None
    declared: bool = False
    declared_keys: frozenset = field(default=frozenset(), repr=False)
    names: Tuple[str, ...] = field(default=(), repr=False)

    @property
    def depth_declared(self) -> bool:
        """True iff the model made a statement about its depth row (possibly "none")."""
        return "depth" in self.declared_keys

    @property
    def is_free_surface(self) -> bool:
        """True iff the model declares a depth row."""
        return self.depth is not None

    @property
    def momentum_vectors(self) -> Tuple[Tuple[int, ...], ...]:
        """Momentum rows grouped per vector (one tuple per moment/layer, one entry per direction);
        the transpose of `momentum`.
        """
        if not self.momentum:
            return ()
        return tuple(tuple(v) for v in zip(*self.momentum))

    def as_names(self) -> dict:
        """The roles with row indices replaced by state names (diagnostics)."""
        nm = self.names

        def one(i):
            """State name of row `i` (`None` stays `None`)."""
            return None if i is None else nm[i]
        return {
            "depth": one(self.depth), "bed": one(self.bed),
            "stationary": [nm[i] for i in self.stationary],
            "positive": [nm[i] for i in self.positive],
            "momentum": [[nm[i] for i in d] for d in self.momentum],
            "gravity": self.gravity, "declared": self.declared,
        }


def _state_names(sm) -> list:
    """State names of `sm` as strings."""
    return [str(s) for s in sm.state]


def _as_name(sm, ref) -> str:
    """Normalise a row reference (name, Symbol, applied Function or index) to a validated state
    name.
    """
    names = _state_names(sm)
    if isinstance(ref, bool):
        raise TypeError(f"role reference {ref!r} is not a state field")
    if isinstance(ref, int):
        if not 0 <= ref < len(names):
            raise ValueError(f"role row {ref} out of range for state {names}")
        return names[ref]
    if isinstance(ref, sp.Basic) and not isinstance(ref, sp.Symbol):
        ref = getattr(ref, "func", ref)
        ref = getattr(ref, "__name__", None) or str(ref)
    name = str(ref)
    if name not in names:
        raise ValueError(
            f"role field {name!r} is not a state field; state = {names}")
    return name


def _carries_sign(sym) -> bool:
    """True if the symbol carries a positive or non-negative assumption."""
    return any(getattr(sym, f"is_{p}", None) is True for p in _SIGN_PREDICATES)


def assumption_positive_names(sm) -> list:
    """State fields whose symbol carries a positive or non-negative assumption.
    Valid on a `SystemModel` only, because the `NumericalSystemModel` strips state
    assumptions.
    """
    return [str(s) for s in sm.state
            if isinstance(s, sp.Symbol) and _carries_sign(s)]


def declare_roles(sm, *, depth=_UNSET, bed=_UNSET, stationary=_UNSET,
                  positive=_UNSET, momentum=_UNSET, gravity=_UNSET,
                  description: Optional[str] = None):
    """Record a role declaration on `sm` (a `history` record) and return `sm`.

    Args:
        depth, bed, stationary, positive, momentum, gravity: only the keys passed are recorded;
            `None` clears a scalar role.
        description: label of the history record.
    """
    rec = {}
    if depth is not _UNSET:
        rec["depth"] = None if depth is None else _as_name(sm, depth)
    if bed is not _UNSET:
        rec["bed"] = None if bed is None else _as_name(sm, bed)
    if stationary is not _UNSET:
        rec["stationary"] = (None if stationary is None
                             else [_as_name(sm, r) for r in stationary])
    if positive is not _UNSET:
        rec["positive"] = (None if positive is None
                           else [_as_name(sm, r) for r in positive])
    if momentum is not _UNSET:
        if momentum is None:
            rec["momentum"] = None
        else:
            per_dir = [[_as_name(sm, r) for r in d] for d in momentum]
            n = {len(d) for d in per_dir}
            if len(n) > 1:
                raise ValueError(
                    f"momentum groups {per_dir} are ragged: every direction "
                    "must list the same number of components")
            rec["momentum"] = per_dir
    if gravity is not _UNSET:
        if gravity is not None:
            gravity = str(gravity)
            if not sm.parameters.contains(gravity):
                raise ValueError(
                    f"gravity parameter {gravity!r} is not a parameter of "
                    f"this system; parameters = {list(sm.parameters.keys())}")
        rec["gravity"] = gravity
    if not rec:
        return sm
    sm.history.append({
        "name": ROLE_RECORD,
        "description": description or "declared state roles",
        "roles": rec,
    })
    return sm


def declared_role_record(sm) -> dict:
    """All role declarations on `sm` merged in order (later wins per key)."""
    merged = {}
    for entry in getattr(sm, "history", None) or []:
        if isinstance(entry, dict) and entry.get("name") == ROLE_RECORD:
            merged.update(entry.get("roles") or {})
    return merged


def resolve_roles(sm) -> Roles:
    """Resolve the declared roles against the current state; undeclared
    `stationary`/`positive` fall back to structure and sign assumptions, never to names.
    """
    names = _state_names(sm)
    rec = declared_role_record(sm)

    def idx(name):
        """Row index of state name `name` (`None` if absent)."""
        return names.index(name) if (name is not None and name in names) else None

    def idx_list(lst):
        """Row indices of the names in `lst` that are still in the state."""
        return tuple(i for i in (idx(n) for n in lst) if i is not None)

    if rec.get("stationary") is not None:
        stationary = idx_list(rec["stationary"])
    else:
        stationary = tuple(sorted(sm.stationary_indices))
    if rec.get("positive") is not None:
        positive = idx_list(rec["positive"])
    else:
        positive = tuple(i for i, s in enumerate(sm.state)
                         if isinstance(s, sp.Symbol) and _carries_sign(s))
    momentum = ()
    if rec.get("momentum"):
        per_dir = [idx_list(d) for d in rec["momentum"]]
        if per_dir and len({len(d) for d in per_dir}) == 1 and per_dir[0]:
            momentum = tuple(tuple(d) for d in per_dir)
    gravity = rec.get("gravity")
    if gravity is not None and not sm.parameters.contains(gravity):
        gravity = None
    return Roles(depth=idx(rec.get("depth")), bed=idx(rec.get("bed")),
                 stationary=stationary, positive=positive, momentum=momentum,
                 gravity=gravity, declared=bool(rec),
                 declared_keys=frozenset(rec), names=tuple(names))


def derive_momentum_directions(sm, depth=None) -> Optional[list]:
    """Momentum rows per direction as name lists, derived from the model's `interpolate_to_3d`
    lift or, failing that, from the depth row's mass-balance flux; `None` if neither applies.
    """
    from zoomy_core.model.boundary_conditions import (
        derive_momentum_field_indices)
    names = _state_names(sm)
    try:
        vectors = derive_momentum_field_indices(sm)
        per_dir = [list(d) for d in zip(*vectors)]
        return [[names[i] for i in d] for d in per_dir]
    except Exception:
        pass
    if depth is None:
        return None
    i_h = names.index(depth)
    e2s = list(sm.equation_to_state_index)
    if i_h not in e2s:
        return None
    row = e2s.index(i_h)
    per_dir = []
    for d in range(sm.n_dim):
        expr = sp.sympify(sm.flux[row, d])
        if sm.hydrostatic_pressure is not None:
            expr = expr + sp.sympify(sm.hydrostatic_pressure[row, d])
        hits = [names[k] for k, s in enumerate(sm.state)
                if k != i_h and sp.diff(expr, s) != 0]
        per_dir.append(hits)
    if not per_dir or not per_dir[0] or len({len(p) for p in per_dir}) != 1:
        return None
    return per_dir


def declare_free_surface_roles(sm, *, depth, bed=None, gravity=None,
                               momentum="derive", positive="assumptions",
                               description=None):
    """Declare the free-surface roles (depth, bed as stationary row, gravity, momentum groups,
    positive rows) and tag `sm` as `free_surface`.

    Returns:
        `sm`.
    """
    depth_name = _as_name(sm, depth)
    bed_name = None if bed is None else _as_name(sm, bed)
    if momentum == "derive":
        momentum = derive_momentum_directions(sm, depth=depth_name)
    if positive == "assumptions":
        positive = assumption_positive_names(sm)
        if depth_name not in positive:
            positive = [depth_name] + positive
    kw = dict(depth=depth_name, bed=bed_name,
              stationary=[] if bed_name is None else [bed_name],
              positive=positive, gravity=gravity)
    if momentum is not None:
        kw["momentum"] = momentum
    declare_roles(sm, description=description or "free-surface roles", **kw)
    from zoomy_core.systemmodel.tags import ModelTag, declare_tags
    declare_tags(sm, {ModelTag.free_surface}, "implied by free-surface roles")
    return sm


def infer_sigma_map_roles(sm, model) -> bool:
    """Declare depth and bed from a sigma-mapped model's relation `z = b + h*zeta`; returns True
    when a declaration was made.
    """
    rel = getattr(model, "coord_relations", None) or {}
    zeta = getattr(model, "_vertical", None) or getattr(model, "vertical", None)
    if not rel or zeta is None:
        return False
    names = set(_state_names(sm))

    def field_name(expr):
        """Name of `expr` if it is a single state field, else `None`."""
        expr = sp.sympify(expr)
        if isinstance(expr, sp.Symbol):
            return str(expr) if str(expr) in names else None
        if isinstance(expr, sp.core.function.AppliedUndef):
            n = expr.func.__name__
            return n if n in names else None
        return None

    for rhs in rel.values():
        rhs = sp.sympify(rhs)
        if zeta not in rhs.free_symbols:
            continue
        d = field_name(sp.expand(sp.diff(rhs, zeta)))
        b0 = sp.expand(rhs.subs(zeta, 0))
        # the bed may be given data (aux), not a state row: depth only then
        b = None if b0 == 0 else field_name(b0)
        if d is None:
            continue
        declare_free_surface_roles(
            sm, depth=d, bed=b, momentum="derive",
            description="free-surface roles from the sigma map z = b + h*zeta")
        return True
    return False


__all__ = [
    "ROLE_RECORD", "Roles", "declare_roles", "declared_role_record",
    "resolve_roles", "declare_free_surface_roles", "infer_sigma_map_roles",
    "assumption_positive_names", "derive_momentum_directions",
]
