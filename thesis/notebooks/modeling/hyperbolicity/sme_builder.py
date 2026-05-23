"""SME PDESystem builder — composed from ``zoomy_core.derivation``.

SME at level L is the Galerkin projection of σ-coord shallow water with:

* ``HydrostaticFlow`` — non-hydrostatic remainder ``p ≡ 0``;
  z-momentum is dropped (w is determined by depth-integrated continuity).
* ``PolynomialAnsatz(M=L, N_w=-1, N_p=-1)`` — shifted Legendre on
  ``[0, 1]`` for the velocity.
* ``GalerkinProjection(w_mode='from_continuity')`` — ω(ξ) is built
  from continuity.

After the j-th x-momentum projections come back, we substitute the
``∂_t h`` rule from continuity j=0 to remove residual time-derivatives
of h that surface in the IBP integrand.

The result is the standard K&T 2019 SME at level L (verified at
L = 0, 1, 2 by ``kt2019_verification.py`` and the analysis-library
hyperbolicity test).
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp

from zoomy_core.analysis import PDESystem
from zoomy_core.derivation import (
    HydrostaticFlow,
    PolynomialAnsatz,
    GalerkinProjection,
)


# Re-export coordinate symbols for callers that want them.
flow = HydrostaticFlow.with_defaults()
t, x, xi, g = flow.t, flow.x, flow.xi, flow.g


def build_sme_pde_system(level: int, *, flat_bottom: bool = True
                         ) -> Tuple[PDESystem, sp.Function, list]:
    """Build the SME nonlinear PDESystem at the given Galerkin level.

    Equations (M + 2 of them):
        - continuity j = 0
        - x-momentum j = 0..M (with ``∂_t h`` substituted out)

    Returns ``(system, h_field, [u_0, …, u_M])``.
    """
    M = level
    flow = HydrostaticFlow.with_defaults()
    if flat_bottom:
        # Inject ∂_x b = 0 by replacing the b-function's spatial
        # derivative everywhere downstream.  Easier: substitute b → -H
        # at the very end via with_substitutions(); for the BUILDER we
        # leave b symbolic and the analysis caller decides.
        pass

    ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                              M=M, N_w=-1, N_p=-1)
    proj = GalerkinProjection(flow=flow, ansatz=ansatz,
                              w_mode="from_continuity")

    # Continuity j = 0 (the evolution equation for h).
    cont_j0 = proj.project_continuity(0)
    # x-momentum j = 0..M.
    xmom = [proj.project_x_momentum(j) for j in range(M + 1)]

    # Apply ∂_t h substitution from continuity j = 0 to eliminate
    # residual ∂_t h atoms produced by the ω polynomial inside the IBP
    # integrand.  Repeated xreplace + expand until no ∂_t h remains.
    dt_h_atom = sp.Derivative(flow.h, flow.t)
    dt_h_rhs = -sp.Derivative(flow.h * ansatz.u_coeffs[0], flow.x)
    xmom_subbed = []
    for eq in xmom:
        prev = None
        cur = sp.expand(eq.doit())
        while prev != cur:
            prev = cur
            cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}))
            cur = sp.expand(cur.doit())
        xmom_subbed.append(cur)

    fields = [flow.h] + list(ansatz.u_coeffs)
    equations = [cont_j0] + xmom_subbed

    if flat_bottom:
        equations = [eq.subs({sp.Derivative(flow.b, flow.x): 0,
                              sp.Derivative(flow.b, flow.t): 0})
                     for eq in equations]

    return PDESystem(
        equations=equations,
        fields=fields,
        time=flow.t,
        space=[flow.x],
        parameters={flow.g: flow.g},
    ), flow.h, list(ansatz.u_coeffs)
