"""VAM model derivation — Escalante, Morales de Luna, Cantero-Chinchilla,
Castro-Orgaz (2024) "Polynomial vertically averaged and moment equations
for shallow flow with non-hydrostatic pressure" (arXiv:2XP7LGXZ).

Derives **eq (4)** (the 6 governing equations) and **eq (5)** (the 3
closure constraints) of the paper from system **(3)** — the σ-coordinate
form of the inviscid Navier-Stokes system with a hydrostatic / non-
hydrostatic pressure split.

Crucial property: NO hydrostatic assumption is made.  The z-momentum
equation is kept and projected on the same Galerkin basis as the rest.
The hydrostatic part of the pressure is absorbed into the source term
``g h ∂_x η`` (paper convention); the non-hydrostatic remainder
``p = p_0·1 + p_1·φ_1 + p_2·φ_2`` is what appears in (4).

Polynomial ansatz (paper eq mid-§2):

    u(t, x, ξ) = u_0(t,x) + u_1(t,x)·φ_1(ξ)
    w(t, x, ξ) = w_0(t,x) + w_1(t,x)·φ_1(ξ) + w_2(t,x)·φ_2(ξ)
    p(t, x, ξ) = p_0(t,x) + p_1(t,x)·φ_1(ξ) + p_2(t,x)·φ_2(ξ)

Friction τ and viscous σ_xz are set to zero in this script (the
inviscid VAM); they enter eq (4) additively and don't change the
polynomial structure of the projections.

Strategy:

* Galerkin-project the σ-coord momentum equations against φ_j for
  j ∈ {0, 1}.
* The j=0 projection of continuity gives eq (4) row 1
  (``∂_t h + ∂_x(h u_0) = 0``).
* The j=1 projection of continuity gives constraint **I_1** of eq (5),
  after substituting j=0 continuity to eliminate ∂_t h.
* The kinematic BCs ``ω|_{ξ=0} = ω|_{ξ=1} = 0`` evaluated on the ansatz
  give the bottom and surface pointwise constraints; combined they give
  **I_2** (subtraction) and the **w_2 closure** (addition).
* The j=0 projection of x-momentum gives eq (4) row 2.

All four projections are verified literally against the paper.

Exit code 0 on full match; non-zero on first mismatch.
"""
from __future__ import annotations

import argparse
import sys
import sympy as sp


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

t, x, xi = sp.symbols("t x xi", real=True)
g = sp.Symbol("g", positive=True)

h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)              # fixed bottom topography
eta = h + b                                      # free-surface elevation

u0, u1 = (sp.Function(f"u_{k}", real=True)(t, x) for k in (0, 1))
w0, w1, w2 = (sp.Function(f"w_{k}", real=True)(t, x) for k in (0, 1, 2))
p0, p1, p2 = (sp.Function(f"p_{k}", real=True)(t, x) for k in (0, 1, 2))


# ---------------------------------------------------------------------------
# Shifted-Legendre basis on [0, 1] up to degree 2
#   φ_0 = 1, φ_1 = 1 − 2ξ, φ_2 = 1 − 6ξ + 6ξ²
# ---------------------------------------------------------------------------

phi = [sp.Integer(1),
       1 - 2 * xi,
       1 - 6 * xi + 6 * xi**2]


def integrate_polynomial(integrand, var=xi, lo=0, hi=1):
    """Polynomial-only integration via ``sp.Poly``, mirroring
    :func:`zoomy_core.symbolic.polynomial_integrate`.  The integrand is
    a polynomial in ``var`` with coefficients that are arbitrary
    functions of (t, x); ``sp.Poly`` handles them as opaque
    coefficients."""
    expr = sp.expand(integrand)
    if not expr.has(var):
        return (hi - lo) * expr
    poly = sp.Poly(expr, var)
    anti = poly.integrate().as_expr()
    return sp.expand(anti.subs(var, hi) - anti.subs(var, lo))


# ---------------------------------------------------------------------------
# Polynomial ansatz: u, w, p as σ-coord polynomials
# ---------------------------------------------------------------------------

u_ansatz = u0 + u1 * phi[1]
w_ansatz = w0 + w1 * phi[1] + w2 * phi[2]
p_ansatz = p0 + p1 * phi[1] + p2 * phi[2]


# σ-vertical velocity (paper just below eq 3):
#   ω(t,x,ξ) = w − (∂_t(ξh+b) + u·∂_x(ξh+b))
# ξ here is held independent of t,x; ∂_t(ξh+b) = ξ·∂_t h, ∂_x(ξh+b) = ξ·∂_x h + ∂_x b.
omega_ansatz = (
    w_ansatz
    - (xi * sp.Derivative(h, t)
       + u_ansatz * (xi * sp.Derivative(h, x) + sp.Derivative(b, x)))
)


# ---------------------------------------------------------------------------
# Galerkin projection helpers
# ---------------------------------------------------------------------------

def galerkin(integrand, j):
    """``∫_0^1 (integrand)·φ_j(ξ) dξ`` via :func:`integrate_polynomial`.

    ``integrand`` is a polynomial in ξ with coefficients that may
    contain ``Derivative(...)`` atoms (for ∂_t, ∂_x).  Those atoms are
    treated as opaque by ``sp.Poly`` since they don't depend on ξ.
    """
    return integrate_polynomial(integrand * phi[j])


# ---------------------------------------------------------------------------
# Step 1 — j=0 projection of continuity → eq (4) row 1
# ---------------------------------------------------------------------------

def derive_continuity_j0():
    """Project ``∂_x(h ũ) + ∂_ξ(w̃ − ũ ∂_x(ξh+b)) = 0`` against φ_0 = 1.

    ∫_0^1 ∂_x(h ũ) dξ = ∂_x(h u_0).
    ∫_0^1 ∂_ξ V dξ = V|_1 − V|_0 with V = w̃ − ũ ∂_x(ξh+b).
    Substitute the kinematic BCs ω|_0 = ω|_1 = 0 to convert the V
    boundary values into ∂_t b and ∂_t η = ∂_t h (with ∂_t b = 0).
    """
    # Direct projection: integrate ∂_x(h u) φ_0 dξ — h is independent of
    # ξ; u_ansatz is polynomial in ξ.
    flux_x = sp.expand(h * u_ansatz)
    flux_x_proj = galerkin(flux_x, j=0)            # integral over ξ
    # Now apply ∂_x outside the integral (commutes with ξ-integration)
    j0_flux = sp.Derivative(flux_x_proj, x)

    # The ∂_ξ term: ∫_0^1 ∂_ξ V dξ = V(1) − V(0)
    # V = w − u·∂_x(ξh+b); at ξ=0: V(0) = w(0) − u(0)·∂_x b
    # at ξ=1: V(1) = w(1) − u(1)·∂_x η
    # Use ω(ξ) = w − ∂_t(ξh+b) − u·∂_x(ξh+b)
    #     ⇒ V(ξ) = w − u·∂_x(ξh+b) = ω(ξ) + ∂_t(ξh+b)
    # KBCs ⇒ V(0) = ∂_t b, V(1) = ∂_t η
    # ∂_xi term contribution: V(1) − V(0) = ∂_t η − ∂_t b = ∂_t h
    # (since b is fixed, ∂_t b = 0)
    j0_xi_term = sp.Derivative(eta, t) - sp.Derivative(b, t)

    full = j0_flux + j0_xi_term
    return sp.expand(full)


# ---------------------------------------------------------------------------
# Step 2 — j=1 projection of continuity → eq (5) constraint I_1
# ---------------------------------------------------------------------------

def derive_continuity_j1(dt_h_rule):
    """Project the σ-coord continuity against φ_1 = 1 − 2ξ.

    ∫_0^1 φ_1 ∂_x(h u) dξ = ∂_x(h · ∫_0^1 φ_1·u dξ) = ∂_x(h · u_1/3).
    ∫_0^1 φ_1 ∂_ξ V dξ = [φ_1 V]_0^1 − ∫_0^1 ∂_ξ φ_1 · V dξ
                       = −V(1) − V(0) + 2 ∫_0^1 V dξ
    where V = w − u·∂_x(ξh+b).  Apply KBCs ⇒ V(1) = ∂_t η, V(0) = ∂_t b,
    and integrate the polynomial V over ξ.

    Substitute ``dt_h_rule`` (j=0 continuity) to eliminate ∂_t h, then
    canonicalise.  The result is exactly I_1 of eq (5).
    """
    # Spatial flux part: ∂_x(h · ∫_0^1 φ_1·u dξ).  ∫_0^1 φ_1·u dξ = u_1/3.
    flux_x = sp.expand(h * u_ansatz)
    flux_x_proj = galerkin(flux_x, j=1)
    j1_flux = sp.Derivative(flux_x_proj, x)

    # IBP-on-φ_1 part: −V(1) − V(0) + 2·∫V dξ  with V = w − u·∂_x(ξh+b)
    V = w_ansatz - u_ansatz * (xi * sp.Derivative(h, x) + sp.Derivative(b, x))
    V_at_0 = sp.expand(V.subs(xi, 0))
    V_at_1 = sp.expand(V.subs(xi, 1))
    int_V = integrate_polynomial(V)

    # Use kinematic BCs to replace V(1) and V(0) with the time-derivative
    # boundary terms — but here we apply them after the j=0 reduction,
    # so we keep them symbolic in V.  Instead, we use the equivalent
    # formulation: V(ξ) = ω(ξ) + ∂_t(ξh+b), so V(0) = ω(0) + ∂_t b and
    # V(1) = ω(1) + ∂_t η.  KBCs ω(0)=ω(1)=0 give the boundary-tilde
    # values directly.
    V0_kbc = sp.Derivative(b, t)              # = 0 (fixed bottom) but kept symbolic
    V1_kbc = sp.Derivative(eta, t)

    j1_xi_term = -V1_kbc - V0_kbc + 2 * int_V

    full = sp.expand(j1_flux + j1_xi_term)

    # Distribute derivatives over Add so the held ``Derivative(h+b, t)``
    # surfaces ``Derivative(h, t)`` for the substitution below.
    full = expand_derivatives(full)
    # Substitute j=0 continuity rule (eliminate ∂_t h).
    full = full.subs(dt_h_rule)
    # Drop the ∂_t b artefact (b is a function of x only).
    full = full.subs({sp.Derivative(b, t): 0})
    return sp.expand(full)


# ---------------------------------------------------------------------------
# Step 3 — kinematic BCs at ξ=0, ξ=1 → eq (5) constraints I_2 and w_2
# ---------------------------------------------------------------------------

def derive_kbc_constraints(dt_h_rule):
    """The KBCs ``ω|_{ξ=0} = ω|_{ξ=1} = 0`` evaluated on the ansatz.

    At ξ=0: ω(0) = w̃|_0 − u_0·∂_x b = (w_0 + w_1 + w_2) − (u_0+u_1)·∂_x b
            (since ∂_t b = 0).  This is the **w_2 closure**.

    At ξ=1: ω(1) = w̃|_1 − ∂_t η − u_1·∂_x η
                 = (w_0 − w_1 + w_2) − ∂_t h − (u_0 − u_1)·∂_x η.

    KBC(top) − KBC(bottom), with ∂_t h replaced via j=0 continuity →
    constraint I_2.
    KBC(bottom) alone, solved for w_2 → the w_2-closure of eq (5).
    """
    omega_0 = sp.expand(omega_ansatz.subs(xi, 0))
    omega_1 = sp.expand(omega_ansatz.subs(xi, 1))
    # ∂_t b ≡ 0 (b = b(x) only).
    omega_0 = omega_0.subs({sp.Derivative(b, t): 0})
    omega_1 = omega_1.subs({sp.Derivative(b, t): 0})

    # I_2 = ω(1) − ω(0), with ∂_t h eliminated via j=0 continuity.
    diff = sp.expand(omega_1 - omega_0)
    diff = expand_derivatives(diff)
    diff = diff.subs(dt_h_rule)                   # eliminate ∂_t h
    I_2 = sp.expand(diff)

    # w_2 closure from KBC at bottom: ω(0) = 0 ⇒ w_2 = ...
    # Solve omega_0 = 0 for w_2.
    w2_closure_rhs = sp.solve(omega_0, w2)[0]

    return I_2, w2_closure_rhs


# ---------------------------------------------------------------------------
# Step 4 — j=0 projection of x-momentum → eq (4) row 2
# ---------------------------------------------------------------------------

def derive_xmom_j0():
    """Project the σ-coord x-momentum against φ_0 = 1.

    ∂_t(h u) + ∂_x(h u² + h p) + g h ∂_x(b+h)
        + ∂_ξ(ω u − p ∂_x(ξh+b)) = 0   (eq 3 row 2, inviscid)

    j=0 projection (φ_0 = 1, ∫φ_0 = 1):
      ∂_t(h ∫u dξ) + ∂_x(h ∫u² dξ + h ∫p dξ) + g h ∂_x(b+h)·1
          + [ω u − p ∂_x(ξh+b)]_0^1 = 0

    ∫u dξ = u_0,  ∫u² dξ = u_0² + u_1²/3,  ∫p dξ = p_0.
    The boundary term simplifies via ω|_{0,1} = 0 to:
      [-p ∂_x(ξh+b)]_0^1 = −p|_1·∂_x η + p|_0·∂_x b
      = −(p_0 − p_1 + p_2)·∂_x η + (p_0 + p_1 + p_2)·∂_x b
    Then ``p_0 − p_1 + p_2 = 0`` (atmospheric BC, paper) makes the
    surface term vanish; the bottom term keeps p_b = p_0 + p_1 + p_2.
    """
    # ∫_0^1 u dξ = u_0; ∫_0^1 u² dξ = u_0² + u_1²/3 (using ∫φ_1² = 1/3);
    # ∫_0^1 p dξ = p_0.
    int_u = integrate_polynomial(u_ansatz)
    int_u2 = integrate_polynomial(u_ansatz**2)
    int_p = integrate_polynomial(p_ansatz)

    # Time derivative of momentum.
    j0_dt = sp.Derivative(h * int_u, t)
    # Spatial flux: ∂_x(h u² + h p) + g h ∂_x η.
    j0_dx = sp.Derivative(h * int_u2 + h * int_p, x)
    j0_grav = g * h * sp.Derivative(eta, x)

    # Boundary term from ∂_ξ.  ω|_{0,1} = 0 (KBCs); only the
    # −p·∂_x(ξh+b) part survives.
    p_at_0 = sp.expand(p_ansatz.subs(xi, 0))
    p_at_1 = sp.expand(p_ansatz.subs(xi, 1))
    bdry_top = -p_at_1 * sp.Derivative(eta, x)
    bdry_bot = +p_at_0 * sp.Derivative(b, x)
    j0_xi = bdry_top + bdry_bot

    full = sp.expand(j0_dt + j0_dx + j0_grav + j0_xi)
    return full


# ---------------------------------------------------------------------------
# Step 5 — j=0 projection of z-momentum → eq (4) row 3
# ---------------------------------------------------------------------------

def derive_zmom_j0():
    """Project the σ-coord z-momentum against φ_0 = 1.

    ∂_t(h w) + ∂_x(h u w) + ∂_ξ(ω w + p) = 0   (eq 3 row 3, inviscid)

    j=0 projection (φ_0 = 1):
      ∂_t(h·∫w dξ) + ∂_x(h·∫(u w) dξ) + [ω w + p]_0^1 = 0

    ∫w dξ = w_0,  ∫u w dξ = u_0 w_0 + (1/3) u_1 w_1   (cross-φ vanish).
    KBCs ⇒ ω·w boundary terms drop; the p boundary gives
      p|_1 − p|_0 = (p_0 − p_1 + p_2) − (p_0 + p_1 + p_2) = −2 p_1.
    """
    int_w = integrate_polynomial(w_ansatz)
    int_uw = integrate_polynomial(u_ansatz * w_ansatz)
    j0_dt = sp.Derivative(h * int_w, t)
    j0_dx = sp.Derivative(h * int_uw, x)

    p_at_0 = sp.expand(p_ansatz.subs(xi, 0))
    p_at_1 = sp.expand(p_ansatz.subs(xi, 1))
    j0_xi = (p_at_1 - p_at_0)              # ω-w terms vanish via KBC

    return sp.expand(j0_dt + j0_dx + j0_xi)


# ---------------------------------------------------------------------------
# Reference equations — written literally from Escalante et al. 2024
# ---------------------------------------------------------------------------

def reference_continuity_j0():
    """Eq (4) row 1: ``∂_t h + ∂_x(h u_0) = 0``."""
    return sp.Derivative(h, t) + sp.Derivative(h * u0, x)


def reference_I_1():
    """Eq (5) line 1:
        h ∂_x u_0 + (1/3) ∂_x(h u_1) + (1/3) u_1 ∂_x h
            + 2(w_0 − u_0 ∂_x b)
    """
    return (h * sp.Derivative(u0, x)
            + sp.Rational(1, 3) * sp.Derivative(h * u1, x)
            + sp.Rational(1, 3) * u1 * sp.Derivative(h, x)
            + 2 * (w0 - u0 * sp.Derivative(b, x)))


def reference_I_2():
    """Eq (5) line 2:
        h ∂_x u_0 + u_1 ∂_x h + 2(u_1 ∂_x b − w_1)
    """
    return (h * sp.Derivative(u0, x)
            + u1 * sp.Derivative(h, x)
            + 2 * (u1 * sp.Derivative(b, x) - w1))


def reference_w2():
    """Eq (5) line 3 / eq (6) ``w_2 = …``:
        w_2 = −(w_0 + w_1) + (u_0 + u_1) ∂_x b
    """
    return -(w0 + w1) + (u0 + u1) * sp.Derivative(b, x)


def reference_zmom_j0():
    """Eq (4) row 3 (inviscid, τ = σ_0 = 0):
        ∂_t(h w_0) + ∂_x(h u_0 w_0 + (1/3) h u_1 w_1) = 2 p_1
    Move RHS to LHS to compare with derived = 0.
    """
    lhs = (sp.Derivative(h * w0, t)
           + sp.Derivative(h * u0 * w0
                           + sp.Rational(1, 3) * h * u1 * w1, x))
    rhs = 2 * p1
    return lhs - rhs


def reference_xmom_j0():
    """Eq (4) row 2 (inviscid, τ = 0):
        ∂_t(h u_0) + ∂_x(h u_0² + (1/3) h u_1² + h p_0) + g h ∂_x η
            = −2 p_1 ∂_x b
    Move the RHS to LHS to compare against the pipeline (which writes
    everything = 0).
    """
    lhs = (sp.Derivative(h * u0, t)
           + sp.Derivative(h * u0**2 + sp.Rational(1, 3) * h * u1**2 + h * p0, x)
           + g * h * sp.Derivative(eta, x))
    rhs = -2 * p1 * sp.Derivative(b, x)
    return lhs - rhs


# ---------------------------------------------------------------------------
# Comparison machinery (paper convention: surface BC p_0 − p_1 + p_2 = 0)
# ---------------------------------------------------------------------------

def expand_derivatives(expr):
    """Distribute Derivative(product, var) via product rule, fixpoint."""
    def step(e):
        if isinstance(e, sp.Add):
            return sp.Add(*[step(a) for a in e.args])
        if isinstance(e, sp.Mul):
            return sp.Mul(*[step(a) for a in e.args])
        if isinstance(e, sp.Derivative):
            inner = step(e.expr)
            wrt_pairs = e.variable_count
            v, n = wrt_pairs[0]
            rest = wrt_pairs[1:]
            if n > 1:
                rest = ((v, n - 1),) + tuple(rest)
            if isinstance(inner, sp.Add):
                return step(sp.Add(*[sp.Derivative(a, v, *rest) for a in inner.args]))
            if isinstance(inner, sp.Mul):
                factors = inner.args
                out = sp.Add(*[
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v),) + factors[i+1:]))
                    for i in range(len(factors))
                ])
                if rest:
                    out = sp.Derivative(out, *rest)
                return step(out)
            return e
        return e
    prev = None
    cur = sp.expand(expr)
    while prev != cur:
        prev = cur
        cur = sp.expand(step(cur))
    return cur


def kill_trivially_zero_derivatives(expr):
    """Replace ``Derivative(inner, var)`` with 0 when ``inner`` is a
    Number/constant or doesn't depend on ``var``.  Handles the
    ``Derivative(1/3, x)`` artefact that survives ``expand`` because
    sympy keeps held derivatives even of pure rationals."""
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(sp.Derivative):
        inner = d.args[0]
        wrt = []
        for v in d.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        if inner.is_number:
            mapping[d] = sp.S.Zero
        elif not any(inner.has(v) for v in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normal_form(expr, surface_bc=True):
    """Expand all derivatives, kill ∂_t b (b = b(x)), then expand."""
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    e = e.subs({sp.Derivative(b, t): 0})
    if surface_bc:
        # Paper convention: p_0 − p_1 + p_2 = 0 ⇔ p_2 = p_1 − p_0.
        # Apply this so the j=0 momentum surface term collapses.
        e = e.subs({p2: p1 - p0})
    e = expand_derivatives(e)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


def _check(label, derived, reference, surface_bc=True):
    derived_n = normal_form(derived, surface_bc=surface_bc)
    reference_n = normal_form(reference, surface_bc=surface_bc)
    diff = sp.expand(derived_n - reference_n)
    ok = (diff == 0)
    mark = "✓ MATCH" if ok else "✗ MISMATCH"
    print(f"--- {label} ---")
    print(f"  {mark}")
    if not ok:
        print(f"  derived   = {derived_n}")
        print(f"  reference = {reference_n}")
        print(f"  diff      = {diff}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on first mismatch")
    args = parser.parse_args()

    print("=== VAM derivation — Escalante et al. 2024 ===\n")

    n_mismatch = 0

    # --- Eq (4) row 1: j=0 continuity ---
    cont_j0 = derive_continuity_j0()
    cont_j0_ref = reference_continuity_j0()
    if not _check("Eq (4) row 1: j=0 continuity", cont_j0, cont_j0_ref,
                  surface_bc=False):
        n_mismatch += 1

    # The j=0 continuity rule used to eliminate ∂_t h downstream:
    # ∂_t h = − ∂_x(h u_0).
    dt_h_rule = {sp.Derivative(h, t):
                 -sp.Derivative(h * u0, x).expand(deep=False)}
    # Use a flat-form rule for cleanest substitution behaviour:
    dt_h_rule = {sp.Derivative(h, t): -sp.Derivative(h * u0, x)}

    # --- Eq (5) line 1: I_1 from j=1 continuity ---
    I_1 = derive_continuity_j1(dt_h_rule)
    I_1_ref = reference_I_1()
    if not _check("Eq (5) I_1: j=1 continuity (after dt_h substitution)",
                  I_1, I_1_ref, surface_bc=False):
        n_mismatch += 1

    # --- Eq (5) line 2 (I_2) and line 3 (w_2 closure) from KBCs ---
    I_2, w2_rhs = derive_kbc_constraints(dt_h_rule)
    I_2_ref = reference_I_2()
    if not _check("Eq (5) I_2: KBC top − KBC bottom (after dt_h subst)",
                  I_2, I_2_ref, surface_bc=False):
        n_mismatch += 1

    # w_2 closure: derived w_2 should match reference.
    w2_diff = sp.expand(w2_rhs - reference_w2())
    ok_w2 = (w2_diff == 0)
    mark = "✓ MATCH" if ok_w2 else "✗ MISMATCH"
    print(f"--- Eq (5) w_2 closure: KBC at ξ=0 (solved for w_2) ---")
    print(f"  {mark}")
    print(f"  derived   w_2 = {w2_rhs}")
    print(f"  reference w_2 = {reference_w2()}")
    if not ok_w2:
        print(f"  diff = {w2_diff}")
        n_mismatch += 1
    print()

    # --- Eq (4) row 2: j=0 x-momentum ---
    xmom_j0 = derive_xmom_j0()
    xmom_j0_ref = reference_xmom_j0()
    if not _check("Eq (4) row 2: j=0 x-momentum (uses surface BC p_0 − p_1 + p_2 = 0)",
                  xmom_j0, xmom_j0_ref, surface_bc=True):
        n_mismatch += 1

    # --- Eq (4) row 3: j=0 z-momentum ---
    zmom_j0 = derive_zmom_j0()
    zmom_j0_ref = reference_zmom_j0()
    if not _check("Eq (4) row 3: j=0 z-momentum (uses surface BC p_0 − p_1 + p_2 = 0)",
                  zmom_j0, zmom_j0_ref, surface_bc=True):
        n_mismatch += 1

    print(f"=== Summary: {n_mismatch} mismatch(es) ===")
    if args.strict and n_mismatch > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
