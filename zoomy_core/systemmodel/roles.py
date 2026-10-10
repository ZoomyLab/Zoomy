"""Declared state ROLES of a SystemModel — the general-PDE replacement for
name-based inference (``h``, ``b``, ``g``, ``hinv``).

A role is a fact the MODEL states about its own state vector:

* ``depth``      — the row carrying the water depth (free-surface models only);
* ``bed``        — the row carrying the bathymetry;
* ``stationary`` — rows whose evolution is identically zero (no flux, no
  source, no dissipation may act on them);
* ``positive``   — rows the model's analytic sign assumptions declare
  positive / non-negative;
* ``momentum``   — the momentum rows per horizontal direction:
  ``momentum[d]`` lists the direction-``d`` components, aligned across ``d``
  (``momentum[0][k]`` and ``momentum[1][k]`` are the x/y components of the
  SAME vector — one moment / layer / mode);
* ``gravity``    — the NAME of the gravity parameter.

Generic code reads :class:`Roles` only.  A model with no declaration (a
tracer, Burgers, Euler) has ``depth is None`` and gets none of the
free-surface specialisations, whatever its fields happen to be called.

Storage: a declaration is a record in ``sm.history`` (``{"name":
"declare_roles", "roles": {...}}``) holding FIELD NAMES, resolved to row
indices against the current ``sm.state`` on every read.  The history is the
one attribute every SystemModel copy carries along (``deepcopy``, the
assumption-free NSM twin, the Chorin splitter stages, the padded predictor of
the old Chorin solvers), so the declaration survives every hand-built copy
without each construction site having to know about roles.  Later records
override earlier ones key by key.
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
    """Resolved roles: row INDICES into ``sm.state`` (see module docstring)."""

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
        """True iff the model made a statement about its depth row (possibly
        "none").  Legacy name-based fallbacks of the free-surface-only
        components (Audusse HR solvers, opt-in depth operations) apply ONLY
        to systems that never made such a statement."""
        return "depth" in self.declared_keys

    @property
    def is_free_surface(self) -> bool:
        """True iff the model declares a depth row."""
        return self.depth is not None

    @property
    def momentum_vectors(self) -> Tuple[Tuple[int, ...], ...]:
        """The momentum rows grouped per VECTOR (one tuple per moment/layer,
        one entry per direction) — the transpose of :attr:`momentum`."""
        if not self.momentum:
            return ()
        return tuple(tuple(v) for v in zip(*self.momentum))

    def as_names(self) -> dict:
        """The roles with row indices replaced by state names (diagnostics)."""
        nm = self.names

        def one(i):
            return None if i is None else nm[i]
        return {
            "depth": one(self.depth), "bed": one(self.bed),
            "stationary": [nm[i] for i in self.stationary],
            "positive": [nm[i] for i in self.positive],
            "momentum": [[nm[i] for i in d] for d in self.momentum],
            "gravity": self.gravity, "declared": self.declared,
        }


def _state_names(sm) -> list:
    return [str(s) for s in sm.state]


def _as_name(sm, ref) -> str:
    """Normalise a row reference (name, Symbol, applied Function, index) to the
    state NAME, validating that it exists."""
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
    return any(getattr(sym, f"is_{p}", None) is True for p in _SIGN_PREDICATES)


def assumption_positive_names(sm) -> list:
    """State fields whose SYMBOL carries a positive / non-negative assumption.

    Valid on a SystemModel only: the NSM strips state assumptions at its
    boundary (:func:`~zoomy_core.numerics.numerical_system_model.
    construct_numerical`), so this must be read BEFORE that — which is what
    :func:`declare_roles` does when it records ``positive``."""
    return [str(s) for s in sm.state
            if isinstance(s, sp.Symbol) and _carries_sign(s)]


def declare_roles(sm, *, depth=_UNSET, bed=_UNSET, stationary=_UNSET,
                  positive=_UNSET, momentum=_UNSET, gravity=_UNSET,
                  description: Optional[str] = None):
    """Record a role declaration on ``sm`` (a history record, see the module
    docstring).  Only the keys passed are recorded; ``None`` explicitly clears
    a scalar role.  Field references may be names, Symbols, applied field
    Functions or row indices; they are validated and stored as NAMES.
    Returns ``sm``."""
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
    """All role declarations on ``sm`` merged in order (later wins per key)."""
    merged = {}
    for entry in getattr(sm, "history", None) or []:
        if isinstance(entry, dict) and entry.get("name") == ROLE_RECORD:
            merged.update(entry.get("roles") or {})
    return merged


def resolve_roles(sm) -> Roles:
    """Resolve the declared roles against the CURRENT state.

    Undeclared keys fall back to STRUCTURE, never to names:

    * ``stationary`` → :attr:`SystemModel.stationary_indices` (rows that are
      identically zero in every operator slot);
    * ``positive``   → the state symbols' sign assumptions (empty on an NSM,
      whose state symbols are assumption-free by construction).

    A declared field that is no longer in the state (e.g. after a change of
    state variables) resolves to "absent"."""
    names = _state_names(sm)
    rec = declared_role_record(sm)

    def idx(name):
        return names.index(name) if (name is not None and name in names) else None

    def idx_list(lst):
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
    """The momentum rows per direction, DERIVED — never guessed from names.

    1. From the model's own ``interpolate_to_3d`` lift (the ``[b, h, u, v,
       w, p]`` registry every derived free-surface family registers) via
       :func:`~zoomy_core.model.boundary_conditions.derive_momentum_field_indices`.
    2. Without a usable lift, from the MASS BALANCE: the direction-``d``
       momentum rows are the state rows the depth row's flux depends on
       (``∂_t h + ∂_d Σ q_d = 0``; a multilayer system sums its layers).

    Returns per-direction NAME lists, or ``None`` when neither applies."""
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
    """The free-surface specialisation: declare ``depth`` / ``bed`` /
    ``gravity``, the bed as the stationary row, the momentum groups (derived
    from the model's lift unless given) and ``positive`` (the analytic sign
    assumptions plus the depth).  The admissibility list of a system that
    declares none defaults to ``[depth]`` (see
    :meth:`SystemModel.admissible_constraints`).  Returns ``sm``."""
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
    declare_tags(sm, {ModelTag.FREE_SURFACE}, "implied by free-surface roles")
    return sm


def infer_sigma_map_roles(sm, model) -> bool:
    """Declare depth / bed from a σ-mapped derivation model's coordinate
    relation ``z = b + h·ζ`` (recorded by ``PDETransformation`` in
    ``model.coord_relations``).  STRUCTURAL: the depth is the coefficient of
    the reference coordinate, the bed the relation at ``ζ = 0``; both must be
    single state fields.  Returns True when a declaration was made."""
    rel = getattr(model, "coord_relations", None) or {}
    zeta = getattr(model, "_vertical", None) or getattr(model, "vertical", None)
    if not rel or zeta is None:
        return False
    names = set(_state_names(sm))

    def field_name(expr):
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
