"""Model-derived wall boundary definitions: the free-slip wall as a reflection of the momentum
vector about the face normal.
Registered on the derivation Model and substituted per face by `FromModel` in the boundary
layer.
"""
from __future__ import annotations

from zoomy_core.systemmodel.system_model import face_normal_symbols

__all__ = ["free_slip_wall_rows", "register_free_slip_wall"]


def free_slip_wall_rows(vectors):
    """Return the ghost value `q - 2 n (n.q)` for every momentum component of a free-slip
    wall.

    Args:
        vectors: iterable of momentum vectors, each a sequence of per-direction fields (one
            entry per mesh direction).
    Returns:
        `{state field: ghost value}` in the canonical face-normal symbols.
    """
    rows = {}
    for vec in vectors:
        vec = list(vec)
        nrm = face_normal_symbols(len(vec))
        qn = sum(nd * qd for nd, qd in zip(nrm, vec))
        for nd, qd in zip(nrm, vec):
            rows[qd] = qd - 2 * nd * qn
    return rows


def register_free_slip_wall(m, vectors, *, name="wall"):
    """Register `free_slip_wall_rows` as `boundary:<name>` on the derivation model `m`;
    `h` and `b` stay unprescribed (extrapolated).
    """
    for field, value in free_slip_wall_rows(vectors).items():
        m.register_group(f"boundary:{name}", field, value)
    return m
