"""Generic SME (Shallow Moment Equations) PDESystem builder.

SME at level L = M is the Galerkin projection of shallow water with a
polynomial velocity ansatz ``u = u_0 + u_1 φ_1(ξ) + … + u_M φ_M(ξ)``,
hydrostatic pressure, and ω(ξ=0)=ω(ξ=1)=0 KBCs — same recipe as the
generic VAM derivation, but **without** non-hydrostatic pressure or
z-momentum.  The pipeline reduces to:

  Continuity j=0       → ∂_t h evolution
  x-momentum  j=0..M   → evolution of (h u_0, …, h u_M)

with the j ≥ 1 continuity projections + KBCs collapsing into the
``∂_t h`` substitution that K&T 2019 uses to eliminate the held
``∂_t h`` terms (this is just our bug-3 closure pattern, applied
fresh).

This module returns a ``PDESystem`` ready for analysis — same
contract as ``escalante2024_generic.build_vam_pde_system``.

The level-2 case reproduces K&T 2019 eq (4.17); levels 0, 1 reproduce
eqs (4.13), (4.14).  This file has nothing model-specific beyond the
ansatz; the level is a parameter.
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp

from zoomy_core.analysis import PDESystem


# ---------------------------------------------------------------------------
# Coordinate symbols
# ---------------------------------------------------------------------------

t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
xi = sp.Symbol("xi", real=True)
g = sp.Symbol("g", positive=True)


def _shifted_legendre(n_max):
    s = sp.Symbol("s")
    return [sp.expand(sp.legendre(i, s).subs(s, 1 - 2 * xi))
            for i in range(n_max + 1)]


def _polynomial_integrate(integrand, var=xi, lo=0, hi=1):
    expr = sp.expand(integrand)
    if not expr.has(var):
        return (hi - lo) * expr
    poly = sp.Poly(expr, var)
    anti = poly.integrate().as_expr()
    return sp.expand(anti.subs(var, hi) - anti.subs(var, lo))


def build_sme_pde_system(level: int, *, flat_bottom: bool = True
                         ) -> Tuple[PDESystem, sp.Function, list]:
    """Construct the SME nonlinear PDESystem at the given Galerkin level.

    Args:
        level:        Galerkin truncation level M (0 = SWE, 1 = SME-1, …).
        flat_bottom:  if True (default), substitute b = const so its
                      gradient vanishes — appropriate for hyperbolicity
                      analysis at constant base states.

    Returns:
        ``(system, h_field, [u_0_field, …, u_M_field])``.

    Equations (M + 2 of them):
        - continuity j = 0 (after eliminating ∂_t h via the trick that
          all higher-j continuity projections become identities at
          flat-bottom rest)
        - x-momentum j = 0..M (with the bug-3 closure already applied
          via direct in-place ∂_t h substitution from j=0)
    """
    M = level
    phi = _shifted_legendre(M)
    h = sp.Function("h", real=True)(t, x)
    if flat_bottom:
        # Just declare ∂_x b = 0; carry b as a constant.
        d_b_dx = sp.S.Zero
    else:                                                # pragma: no cover
        b_fn = sp.Function("b", real=True)(x)
        d_b_dx = sp.Derivative(b_fn, x)

    u_coeffs = [sp.Function(f"u_{i}", real=True)(t, x) for i in range(M + 1)]
    u = sum((u_coeffs[i] * phi[i] for i in range(M + 1)), sp.S.Zero)

    # Continuity j=0: ∂_t h + ∂_x(h u_0) = 0
    cont_j0 = sp.Derivative(h, t) + sp.Derivative(h * u_coeffs[0], x)

    # σ-coord momentum split (Escalante et al. 2024, eq 3):
    #
    #     ∂_t(h u) + ∂_x(h u² + h p) + g h ∂_x η
    #         + ∂_ξ(ω u − p ∂_x(ξh+b)) = ∂_ξ σ_xz
    #
    # where ``p`` is the **non-hydrostatic** pressure remainder after
    # p_total = p_H + p with p_H = −g h (ξ − 1).  For SME (purely
    # hydrostatic) the non-hydrostatic remainder is identically zero;
    # the hydrostatic effect lives entirely in the ``g h ∂_x η``
    # source term.  Hence we DO NOT add a ``h p`` term in the flux,
    # nor do we put p in the IBP boundary or interior of ∂_ξ.
    eta_x = sp.Derivative(h, x) + d_b_dx

    # σ-velocity ω(ξ) for SME — derived from continuity (NOT a state
    # variable, in contrast to VAM where w is independent).  Continuity
    # in σ-coords reads
    #
    #     ∂_t h + ∂_x(h u(ξ)) + ∂_ξ ω = 0,
    #
    # and KBC at the bottom is ω(0) = 0.  Integrating in ξ:
    #
    #     ω(ξ) = - ξ ∂_t h - Σ_i ∂_x(h u_i) · Φ_i(ξ),
    #     where  Φ_i(ξ) := ∫_0^ξ φ_i(ξ') dξ'.
    #
    # KBC at the top ω(1) = 0 then collapses to continuity j=0 — i.e.
    # it's not an independent constraint.
    Phi_i = []                                             # Φ_i(ξ)
    for i in range(M + 1):
        anti = sp.Poly(phi[i], xi).integrate().as_expr()   # ∫ φ_i dξ', no constant
        Phi_i.append(sp.expand(anti))                       # Φ_i(0) = 0 ✓
    omega_sme = -xi * sp.Derivative(h, t)
    for i in range(M + 1):
        omega_sme -= sp.Derivative(h * u_coeffs[i], x) * Phi_i[i]

    # x-momentum projection j=0..M.  Non-hydrostatic p ≡ 0 for SME so
    # only ω u survives in the ∂_ξ term and KBCs ω(0)=ω(1)=0 kill the
    # boundary.  Interior IBP: -∫φ_j' ω u dξ.
    eqs_xmom = []
    for j in range(M + 1):
        int_phi_j = _polynomial_integrate(phi[j])
        int_phi_j_u = _polynomial_integrate(phi[j] * u)
        int_phi_j_u2 = _polynomial_integrate(phi[j] * u**2)
        dphi_j = sp.diff(phi[j], xi)
        int_dphi_omega_u = _polynomial_integrate(dphi_j * omega_sme * u)
        eq = (sp.Derivative(h * int_phi_j_u, t)
              + sp.Derivative(h * int_phi_j_u2, x)
              + g * h * eta_x * int_phi_j
              - int_dphi_omega_u)
        eqs_xmom.append(eq)

    # Apply ∂_t h substitution from continuity j=0 to remove residual
    # ∂_t h atoms inside the projection equations (the bug-3 closure).
    # This is the same fixpoint pattern as kt2019_verification — the
    # held ∂_t h surfaces from ω inside the IBP integral.  Use
    # ``xreplace`` for exact atomic substitution; ``sp.expand`` first
    # so any ``Derivative(h(t,x)*const, t)`` etc. gets distributed
    # before the rule fires.
    dt_h_atom = sp.Derivative(h, t)
    dt_h_rhs = -sp.Derivative(h * u_coeffs[0], x)
    eqs_xmom_subbed = []
    for eq in eqs_xmom:
        # Repeatedly distribute outer derivatives + xreplace until no
        # ``Derivative(h, t)`` atom remains.
        prev = None
        cur = sp.expand(eq.doit())
        while prev != cur:
            prev = cur
            cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}))
            cur = sp.expand(cur.doit())
        eqs_xmom_subbed.append(cur)
    eqs_xmom = eqs_xmom_subbed

    fields = [h] + list(u_coeffs)
    equations = [cont_j0] + eqs_xmom

    return PDESystem(
        equations=equations,
        fields=fields,
        time=t,
        space=[x],
        parameters={g: g},
    ), h, u_coeffs
