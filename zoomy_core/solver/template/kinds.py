"""Value kinds of the solver template: the type vocabulary every port, local
and constant is checked against.

A kind names the SHAPE CLASS and the element type of a value.  It is what the
type checker compares at every ``Call`` (arity + kind), what the C walker turns
into a declaration, and what decides whether an ``Assign`` is a scalar
assignment or a whole-array elementwise one.

Scalars
    ``real``  floating-point scalar (time, dt, tolerances)
    ``int``   integer scalar (iteration, i_snapshot, counters)
    ``bool``  logical scalar (predicates)

Arrays (element type in brackets; ``nv`` state rows, ``na`` aux rows, ``nc``
cells, ``nf`` faces, ``nb`` boundary faces, ``d`` horizontal dimension)
    ``cell_state``  (nv, nc)    [real]  Q, Q0, Q_cand, cell_term
    ``cell_aux``    (na, nc)    [real]  Qaux
    ``params``      (np,)       [real]  physical parameters
    ``face_state``  (nv, nf)    [real]  Q_L, Q_R, F, Dp, Dm
    ``face_aux``    (na, nf)    [real]  Qaux_L, Qaux_R
    ``face_scalar`` (nf,)       [real]  per-face wave speeds, inradius
    ``face_mask``   (nf,)       [bool]
    ``cell_scalar`` (nc,)       [real]
    ``cell_mask``   (nc,)       [bool]  troubled, o1
    ``cell_grad``   (nv, d, nc) [real]  limited gradient
    ``bc_state``    (nv, nb)    [real]  ghost states (D5 ghost convention)
    ``bc_aux``      (na, nb)    [real]  ghost aux (D6)
    ``array``       any shape   [real]  generic (toy procedures, user blocks)
    ``mask``        any shape   [bool]  generic

Opaque (pytrees / handles; never elementwise)
    ``geom``    geometry pytree (differentiable node-coordinate products)
    ``obs``     observer accumulator pytree
    ``plugin``  plugin carry pytree
    ``any``     wildcard (setup blocks taking host objects)
"""
from __future__ import annotations

import dataclasses

SCALAR_KINDS = ("real", "int", "bool")

#: array kind -> element type ("real" | "bool")
ARRAY_KINDS = {
    "cell_state": "real",
    "cell_aux": "real",
    "params": "real",
    "face_state": "real",
    "face_aux": "real",
    "face_scalar": "real",
    "face_mask": "bool",
    "cell_scalar": "real",
    "cell_mask": "bool",
    "cell_grad": "real",
    "bc_state": "real",
    "bc_aux": "real",
    "array": "real",
    "mask": "bool",
}

OPAQUE_KINDS = ("geom", "obs", "plugin", "any")

ALL_KINDS = frozenset(SCALAR_KINDS) | frozenset(ARRAY_KINDS) | frozenset(OPAQUE_KINDS)

#: real array kind -> the bool array kind of the same shape (elementwise
#: comparisons / isfinite produce it).  Kinds without an entry have no
#: materialisable mask; their elementwise predicates may only appear directly
#: inside a reduction (``AnyTrue`` / ``AllTrue``).
MASK_OF = {
    "cell_scalar": "cell_mask",
    "cell_mask": "cell_mask",
    "face_scalar": "face_mask",
    "face_mask": "face_mask",
    "array": "mask",
    "mask": "mask",
}

#: kinds that carry a trailing cell axis (``MaskLike`` reads it)
CELL_AXIS_KINDS = frozenset({"cell_state", "cell_aux", "cell_scalar",
                             "cell_mask", "cell_grad", "array", "mask"})

#: Transient kind of an elementwise predicate over an array without a mask
#: kind (e.g. ``IsFinite(Q)``): legal only as the argument of a reduction.
TRANSIENT_BOOL = "~bool"


class KindError(TypeError):
    """A value is used with a kind its port or context does not accept."""


def check_kind(kind: str) -> str:
    if kind not in ALL_KINDS:
        raise KindError(
            f"unknown kind {kind!r}; known kinds: {sorted(ALL_KINDS)}")
    return kind


def is_scalar(kind: str) -> bool:
    return kind in SCALAR_KINDS


def is_array(kind: str) -> bool:
    return kind in ARRAY_KINDS


def is_opaque(kind: str) -> bool:
    return kind in OPAQUE_KINDS


def element_type(kind: str) -> str:
    """``real`` / ``int`` / ``bool`` of a scalar or array kind."""
    if kind in SCALAR_KINDS:
        return kind
    if kind in ARRAY_KINDS:
        return ARRAY_KINDS[kind]
    raise KindError(f"kind {kind!r} has no element type (opaque)")


def compatible(actual: str, declared: str) -> bool:
    """May a value of kind ``actual`` be passed where ``declared`` is expected?

    Exact match, plus: ``int`` -> ``real`` promotion; ``bool`` -> ``int``;
    the wildcards ``any`` (anything) and ``array`` / ``mask`` (any array of
    the same element type)."""
    if actual == declared or declared == "any" or actual == "any":
        return True
    if declared == "real" and actual in ("int", "bool"):
        return True
    if declared == "int" and actual == "bool":
        return True
    if declared in ("array", "mask") and actual in ARRAY_KINDS:
        return ARRAY_KINDS[actual] == ARRAY_KINDS[declared]
    if actual in ("array", "mask") and declared in ARRAY_KINDS:
        return ARRAY_KINDS[actual] == ARRAY_KINDS[declared]
    return False


@dataclasses.dataclass(frozen=True)
class Port:
    """A typed, named slot of a procedure or block signature."""

    name: str
    kind: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"port name must be an identifier, got {self.name!r}")
        check_kind(self.kind)

    def __str__(self):
        return f"{self.name}: {self.kind}"


def ports(spec) -> tuple:
    """Normalise a port list.

    Accepts ``Port`` objects, ``(name, kind)`` pairs, ``"name:kind"`` strings
    or a whitespace-separated ``"a:real b:int"`` string."""
    if spec is None:
        return ()
    if isinstance(spec, str):
        spec = spec.split()
    out = []
    for p in spec:
        if isinstance(p, Port):
            out.append(p)
        elif isinstance(p, str):
            name, _, kind = p.partition(":")
            if not kind:
                raise ValueError(f"port {p!r} needs a kind: 'name:kind'")
            out.append(Port(name.strip(), kind.strip()))
        else:
            name, kind = p
            out.append(Port(name, kind))
    names = [p.name for p in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate port name(s) {dupes}")
    return tuple(out)
