"""VAM PDESystem builder — composed from ``zoomy_core.derivation``.

Mirrors ``tutorials/sme/sme_builder.py``.  The model logic that used to
live in ``escalante2024_generic.py`` is now expressed as the
composition

  NonHydrostaticFlow  +  PolynomialAnsatz(M, N_w=N, N_p=N)
                      +  GalerkinProjection(w_mode='state')
                      +  KBC bottom + surface BC closures.

Two output modes:

* ``eliminate_closures=False`` (default) — return the full PDESystem
  with the algebraic closures (``w_N - rhs = 0``, ``p_N - rhs = 0``)
  as separate equations.  Suitable for dispersion analysis (the
  closures appear as algebraic rows in the plane-wave matrix).

* ``eliminate_closures=True`` — substitute the algebraic closures
  into every other equation, so the returned PDESystem has no
  algebraic rows and ``M_t`` is **non-singular**.  This is the form
  needed for clean hyperbolicity sampling — the principal-symbol
  pencil then reproduces the paper's eigenvalue formulas (eq 12 of
  Escalante 2024).
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp

from zoomy_core.analysis import PDESystem
from zoomy_core.derivation import (
    NonHydrostaticFlow,
    PolynomialAnsatz,
    GalerkinProjection,
    kbc_bottom_solve_w_N,
    surface_bc_solve_p_N,
)


# Re-export the canonical symbols for callers.
flow = NonHydrostaticFlow.with_defaults()
t, x, xi, g = flow.t, flow.x, flow.xi, flow.g
h, b = flow.h, flow.b


def build_vam_pde_system(M: int, N: int, *,
                         eliminate_closures: bool = False,
                         hyperbolic_predictor: bool = False,
                         ) -> Tuple[PDESystem, list, list, list]:
    """Build the VAM nonlinear PDESystem at degree (M, N).

    Args:
        M: degree of the velocity ansatz (u_0..u_M).
        N: degree of w and p ansatzes (w_0..w_N; p_0..p_N).
        eliminate_closures:
            False — keep ``w_N`` and ``p_N`` as state variables, with
                    their algebraic closures included as separate
                    equations (full DAE form, suitable for dispersion).
            True  — pre-substitute the closures into every equation,
                    so ``w_N`` and ``p_N`` no longer appear; the system
                    is purely differential, ``M_t`` is non-singular.
        hyperbolic_predictor:
            If True, additionally drop the non-hydrostatic pressure
            terms ``T(P)`` (set every ``p_i`` to zero before
            projection).  This produces the **underlying hyperbolic
            system** of eq (7) of Escalante 2024 — what the
            projection-correction predictor (eq 11) actually advances.
            Hyperbolicity sampling on this form reproduces the paper's
            eigenvalues from eq (12).  The full DAE form (default) has
            additional pressure-related modes which are determined
            algebraically at each time step rather than evolved.

    Returns ``(system, u_fields, w_fields, p_fields)``.  When
    ``hyperbolic_predictor=True`` the ``p_fields`` is empty.
    """
    flow = NonHydrostaticFlow.with_defaults()
    # Hyperbolic-predictor mode drops the non-hydrostatic pressure
    # entirely — easiest done by giving the ansatz no p coefficients
    # so ``p ≡ 0`` and the corresponding p_N closure / pressure terms
    # all collapse.
    N_p = -1 if hyperbolic_predictor else N
    ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                              M=M, N_w=N, N_p=N_p)
    proj = GalerkinProjection(flow=flow, ansatz=ansatz, w_mode="state")

    # --- Equations ---
    # Continuity j = 0.
    cont_j0 = proj.project_continuity(0)
    # x-mom j = 0..M.
    xmom = [proj.project_x_momentum(j) for j in range(M + 1)]
    # z-mom j = 0..N-1 (j=N is the algebraic closure for w_N via KBC bottom).
    zmom = [proj.project_z_momentum(j) for j in range(N)]
    # Continuity j = 1..N (algebraic-in-time constraints).  Skipped in
    # hyperbolic_predictor mode: those constraints are enforced by the
    # corrector (Poisson) step, not by the predictor.
    cont_constraints = ([proj.project_continuity(j) for j in range(1, N + 1)]
                        if not hyperbolic_predictor else [])

    # Algebraic closures (RHS expressions).
    w_N_rhs = kbc_bottom_solve_w_N(ansatz, flow)
    p_N_rhs = (surface_bc_solve_p_N(ansatz, flow)
               if not hyperbolic_predictor else None)

    if eliminate_closures:
        repl = {ansatz.w_coeffs[N]: w_N_rhs}
        if not hyperbolic_predictor:
            repl[ansatz.p_coeffs[N]] = p_N_rhs
        # Substitute closures into every projected equation.
        diff_eqs = xmom + zmom + cont_constraints
        diff_eqs = [sp.expand(eq.xreplace(repl).doit()) for eq in diff_eqs]
        # Apply ∂_t h substitution from cont j=0 to remove residual
        # ∂_t h atoms in the projection equations (the same bug-3
        # closure pattern as SME).  Do NOT substitute it INTO cont j=0
        # itself (that's the equation we're using).
        cont_j0_subbed = sp.expand(cont_j0.xreplace(repl).doit())
        dt_h_atom = sp.Derivative(flow.h, flow.t)
        dt_h_rhs = -sp.Derivative(flow.h * ansatz.u_coeffs[0], flow.x)
        diff_eqs_subbed = []
        for eq in diff_eqs:
            prev = None
            cur = sp.expand(eq.doit())
            while prev != cur:
                prev = cur
                cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}))
                cur = sp.expand(cur.doit())
            diff_eqs_subbed.append(cur)
        u_fields = list(ansatz.u_coeffs)
        w_fields = list(ansatz.w_coeffs[:N])           # drop w_N
        p_fields = (list(ansatz.p_coeffs[:N]) if not hyperbolic_predictor
                    else [])
        fields = [flow.h] + u_fields + w_fields + p_fields
        equations = [cont_j0_subbed] + diff_eqs_subbed
    else:
        eq_w_N = ansatz.w_coeffs[N] - w_N_rhs
        u_fields = list(ansatz.u_coeffs)
        w_fields = list(ansatz.w_coeffs)
        p_fields = list(ansatz.p_coeffs)
        fields = [flow.h] + u_fields + w_fields + p_fields
        algebraic_eqs = [eq_w_N]
        if not hyperbolic_predictor:
            algebraic_eqs.append(ansatz.p_coeffs[N] - p_N_rhs)
        equations = ([cont_j0] + xmom + zmom + cont_constraints
                     + algebraic_eqs)

    return (PDESystem(equations=equations,
                      fields=fields,
                      time=flow.t,
                      space=[flow.x],
                      parameters={flow.g: flow.g}),
            u_fields, w_fields, p_fields)
