"""Compressible Euler equations of an ideal gas, built directly as a
:class:`SystemModel` (no Model derivation, no depth integration).

State ``Q = (rho, rho_u[, rho_v[, rho_w]], E)`` in ``dimension`` = 1, 2 or 3
space dimensions, pressure ``p = (gamma - 1)(E - |m|^2 / (2 rho))``:

    d_t rho   + div(m)                      = 0
    d_t m     + div(m (x) m / rho + p I)    = 0
    d_t E     + div((E + p) m / rho)        = 0

Declared alongside the operators:

* eigenvalues along the face normal ``n``: ``u.n - c``, ``u.n`` (``dimension``
  times), ``u.n + c``, ``c = sqrt(gamma p / rho)``;
* roles: the momentum rows per direction (what :class:`~zoomy_core.model.
  boundary_conditions.Wall` reflects); no depth -- none of the free-surface
  specialisations apply;
* admissibility ``[rho, p]`` (MOOD's troubled-cell test);
* reconstruction variables ``W = (rho, u_d, p)`` and their inverse (the NSM's
  ``ReconstructionSpec(variables="conservative")`` reconstructs ``Q`` instead).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.reconstruction_inverse import reconstruction_symbols
from zoomy_core.systemmodel.system_model import SystemModel, face_normal_symbols

_MOMENTUM = ("rho_u", "rho_v", "rho_w")


def compressible_euler(dimension: int = 3, gamma: float = 1.4,
                       boundary_conditions=None,
                       initial_conditions=None) -> SystemModel:
    """The ideal-gas Euler system in ``dimension`` space dimensions.
    ``boundary_conditions``: a ``BoundaryConditions`` (or list) of the
    existing BC classes; ``initial_conditions``: an ``initial_conditions``
    object (e.g. ``IC.UserFunction``)."""
    dim = int(dimension)
    if dim not in (1, 2, 3):
        raise ValueError(f"dimension {dimension!r}: 1, 2 or 3")
    space = [sp.Symbol(c, real=True) for c in "xyz"[:dim]]
    rho = sp.Symbol("rho", real=True)
    m = [sp.Symbol(_MOMENTUM[d], real=True) for d in range(dim)]
    E = sp.Symbol("E", real=True)
    state = [rho, *m, E]
    g = sp.Symbol("gamma", positive=True)
    params = Zstruct(gamma=g)
    params._symbolic_name = "p"

    u = [md / rho for md in m]
    kinetic = sum(md * ud for md, ud in zip(m, u)) / 2
    p = (g - 1) * (E - kinetic)
    n_eq = dim + 2
    F = sp.zeros(n_eq, dim)
    for d in range(dim):
        F[0, d] = m[d]
        for k in range(dim):
            F[1 + k, d] = m[k] * u[d] + (p if k == d else 0)
        F[n_eq - 1, d] = (E + p) * u[d]

    n = face_normal_symbols(dim)
    un = sum(ud * nd for ud, nd in zip(u, n))
    c = sp.sqrt(g * p / rho)
    eigenvalues = sp.Matrix([un - c] + [un] * dim + [un + c])

    # reconstruction pair: W = (rho, u_d, p) and Q(W) in the WB_<state> symbols
    W = reconstruction_symbols(state)
    w_rho, w_u, w_p = W[0], W[1:1 + dim], W[n_eq - 1]
    to_w = ZArray([rho, *u, p])                                  # (n_state,)
    to_q = ZArray([w_rho, *[w_rho * wu for wu in w_u],
                   w_p / (g - 1) + w_rho * sum(wu ** 2 for wu in w_u) / 2])

    sm = SystemModel(
        time=sp.Symbol("t", real=True), space=space, state=state, aux_state=[],
        parameters=params,
        flux=F, hydrostatic_pressure=sp.zeros(n_eq, dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(n_eq, n_eq, dim),
        source=sp.zeros(n_eq, 1), mass_matrix=sp.eye(n_eq),
        eigenvalues=eigenvalues,
        parameter_values=Zstruct(gamma=float(gamma)),
        reconstruction_variables=to_w, state_from_reconstruction=to_q,
        initial_conditions=initial_conditions)
    sm.history.append({"name": "compressible_euler",
                       "description": f"ideal-gas Euler, {dim}-D"})
    sm.declare_roles(depth=None, momentum=[[md] for md in m])
    sm.declare_admissible([rho, p])
    if boundary_conditions is not None:
        sm.attach_boundary_conditions(boundary_conditions)
    return sm


def pressure(sm: SystemModel):
    """``p(Q)`` of a :func:`compressible_euler` system (sympy)."""
    return sm.admissible[1]


__all__ = ["compressible_euler", "pressure"]
