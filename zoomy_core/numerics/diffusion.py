"""Structure of a system's diffusion tensor — read off ``diffusion_matrix``
ONLY (the new-template replacement for the numpy solver's scalar path, which
keys its coefficient on a parameter literally named ``nu``).

The tensor ``A(Q, Qaux, p)`` of shape ``(n_eq, n_state, n_dim, n_dim)``
enters the balance law as ``∇·(A : ∇Q)``.  :func:`classify_diffusion`
answers which discretisation it needs:

* ``"none"``       — identically zero;
* ``"isotropic"``  — ``A[i, j, d, e] = κ_i δ_ij δ_de``: one scalar
  coefficient per row (TPFA ``κ·L`` per row, the cheap path);
* ``"diagonal"``   — row-diagonal (``i = j``) and axis-diagonal (``d = e``)
  but direction-dependent coefficients;
* ``"dense"``      — any cross-row (``i ≠ j``) or cross-derivative
  (``d ≠ e``) entry (the general ``∇·(A:∇Q)`` assembly).

``state_dependent`` flags a coefficient that reads a state or aux symbol
(the implicit solve is then nonlinear).  The coefficients are the tensor's
OWN entries (sign as the model wrote them); nothing here looks a parameter
up by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import sympy as sp


@dataclass(frozen=True)
class DiffusionStructure:
    kind: str                                   # none|isotropic|diagonal|dense
    state_dependent: bool = False
    coefficients: dict = field(default_factory=dict)   # row -> expr (isotropic)
    axis_coefficients: dict = field(default_factory=dict)  # (row, d) -> expr


def classify_diffusion(sm, slot: str = "diffusion_matrix") -> DiffusionStructure:
    """Classify ``sm.<slot>`` (see the module docstring)."""
    A = getattr(sm, slot, None)
    if A is None:
        return DiffusionStructure("none")
    shape = tuple(int(s) for s in getattr(A, "shape", ()))
    if len(shape) != 4:
        raise ValueError(f"{slot}: expected a rank-4 tensor, got shape {shape}")
    n_eq, n_st, n_d, _ = shape
    live = set(sm.state) | set(sm.aux_state)
    dense = False
    state_dep = False
    axis = {}
    any_entry = False
    for i, j, d, e in product(range(n_eq), range(n_st), range(n_d), range(n_d)):
        a = sp.sympify(A[i, j, d, e])
        if a == 0:
            continue
        any_entry = True
        if a.free_symbols & live:
            state_dep = True
        if i == j and d == e:
            axis[(i, d)] = a
        else:
            dense = True
    if not any_entry:
        return DiffusionStructure("none")
    if dense:
        return DiffusionStructure("dense", state_dep, {}, axis)
    rows = sorted({i for i, _d in axis})
    coeff = {}
    isotropic = True
    for i in rows:
        vals = [axis.get((i, d), sp.S.Zero) for d in range(n_d)]
        if any(sp.simplify(v - vals[0]) != 0 for v in vals[1:]):
            isotropic = False
            break
        coeff[i] = vals[0]
    if isotropic:
        return DiffusionStructure("isotropic", state_dep, coeff, axis)
    return DiffusionStructure("diagonal", state_dep, {}, axis)


__all__ = ["DiffusionStructure", "classify_diffusion"]
