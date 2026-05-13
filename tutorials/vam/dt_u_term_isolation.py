# ---
# title: "Chain isolation of the ∂_t u term — where does the spurious -U_1 ∂_t h come from?"
# author: Ingo Steldermann
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# # Chain isolation: ∂_t u with affine test argument
#
# When we run the VAM x-momentum chain with **affine** test argument
# ``φ_u_j((z-b)/h)`` instead of the z-only form ``φ_u_j(z)``, the
# resulting ``xmom_j0`` row contains an extra ``-U_1 ∂_t h`` term that
# does **not** appear in the conservative-form Escalante reference
# ``∂_t (h U_0) + ∂_x(h U_0² + (1/3) h U_1²) + g h ∂_x η + 2 P_1 ∂_x b / ρ = 0``.
#
# To pinpoint where this term enters, we isolate just the ``∂_t u``
# piece (no advection, no pressure, no gravity) and walk it through
# the chain step-by-step using ``sys.describe()`` to make every
# intermediate visible.
#
# **Setup.**  Single-leaf system whose only equation is
# ``φ_u_0((z-b)/h) · ∂_t u = 0``.  Then push it through the standard
# chain: ProductRule → Integrate → InterfaceKBC at the surface →
# ``∂_t b → 0`` → AffineProjection → Expand → EvaluateIntegrals.
# At each step we print the leaf so we can see exactly where the
# ``-U_1 ∂_t h`` shows up.

# +
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection, EvaluateIntegrals, Expand, FullINS, InterfaceKBC,
    Integrate, Inviscid, Multiply, ProductRule, StateSpace,
    Expression,
)
from zoomy_core.model.models.derived_system import System
# -


# ## 1. Setup
#
# State, basis, coefficients.  We use VAM(M=1) for the velocity
# expansion: ``u = U_0 + U_1 · φ_u_1(ζ)``.

# +
state = StateSpace(dimension=2)
z = state.z
basis_u = Legendre_shifted(level=1, symbol="phi_u")
coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
            for k in range(2)]
# Test functions in the opaque ζ-Function form: ``phi_u_k(state.zeta)``
# where ``state.zeta = Function("zeta")(t, x, z)`` carries full
# (t, x, z) dependence.  Sympy's native chain rule fires cleanly through
# this opaque head; ``ProductRule(variables=[t, x, z])`` then produces
# the conservative-form pre-distribution
# ``phi(ζ) ∂_t u → ∂_t(phi(ζ)·u) − u·phi'(ζ)·∂_t ζ`` for free, and
# ``Integrate(method="auto")`` fires Leibniz on ``∂_t F`` because the
# coefficient is now constant 1.
test_phi_u = Zstruct(
    **{f"phi_{k}": basis_u.phi[k](state.zeta) for k in range(2)})
# -


# ## 2. Build a single-equation system: just ∂_t u
#
# ``System("dt_u_only", state)`` with one leaf
# ``∂_t u(t, x, z) = 0`` (a placeholder; the equation has no
# physical meaning on its own — we just want to track what the
# chain does to it).

# +
sys = System("dt_u_only", state)
sys._set("dt_u", Expression(sp.Derivative(state.u, state.t), name="dt_u"))
sys.describe()
# -


# ## 3. Step 1 — Multiply by φ_u_0((z-b)/h)
#
# Scalar multiplication by the j=0 test factor.

# +
sys.dt_u.apply(Multiply(test_phi_u.phi_0))
sys.describe()
# -


# ## 4. Step 2 — ProductRule on [x, z]
#
# Distributes ``∂_v(coeff · F)`` for ``v ∈ {x, z}``.  Since ``∂_t u``
# carries no ``∂_x`` or ``∂_z``, this should be a no-op on the
# integrand.  But ``φ_u_0((z-b)/h)`` is z-dependent, and after the
# product rule we'd see chain-rule contributions if the chain
# anticipates them.

# +
sys.apply(ProductRule(variables=[state.t, state.x, z]))
sys.describe()
# -


# ## 5. Step 3 — Integrate over [b, η]
#
# ``method="auto"``: the integrand is ``φ_u_0((z-b)/h) · ∂_t u``.
# Auto-mode looks for a derivative shape — finds ``∂_t``.
# ``_extract_derivative(integrand, t)`` decomposes the integrand
# as ``coeff · ∂_t F``: ``coeff = φ_u_0((z-b)/h)``,
# ``F = u(t, x, z)``.  The check is whether ``coeff`` is ``t``-
# independent — and **here it is not**: ``φ_u_0((z-b)/h)`` depends
# on ``b(t, x)`` and ``h(t, x)``, so the auto-mode rejects the
# Leibniz route and falls through to ``direct``: an unevaluated
# ``Integral(...)``.
#
# (Compare with the z-only test argument ``φ_u_0(z)``: there
# ``coeff = φ_u_0(z)`` is t-independent, so Leibniz fires, producing
# ``∂_t ∫ φ u dz - φ(η) u|_η ∂_t η + φ(b) u|_b ∂_t b`` — the
# conservative-form path.)

# +
sys.apply(Integrate(z, state.b, state.eta, method="auto"))
sys.describe()
# -


# ## 6. Step 4 — InterfaceKBC at the surface and the bottom
#
# These substitute ``w|_η = ∂_t b + ∂_t h + u|_η ∂_x η`` and
# ``w|_b = u|_b ∂_x b``.  The current integrand has no ``w``, so this
# should be a no-op.

# +
sys.apply(InterfaceKBC(state, state.b)).simplify()
sys.apply(InterfaceKBC(state, state.eta)).simplify()
sys.describe()
# -


# ## 7. Step 5 — AffineProjection (z → ζh + b)
#
# Substitutes the integration variable.  The integrand becomes
# ``φ_u_0(ζ) · ∂_t u(t, x, ζh + b) · h`` (the ``h`` from
# ``dz = h dζ``).
#
# **Key observation.**  ``∂_t u(t, x, ζh + b)`` is sympy's
# ``Derivative(u(t, x, ζh + b), t)``.  When sympy chain-rules this,
# it produces:
#
#     u_t(t, x, ζh + b)  +  u_z(t, x, ζh + b) · ∂_t(ζh + b)
#                        =  u_t  +  u_z · (ζ ∂_t h + ∂_t b)
#
# with ``∂_t b → 0`` later.  So **the affine projection introduces
# a chain-rule contribution proportional to ``ζ ∂_t h``** —
# multiplied by ``u_z``.  This is the term we suspect is producing
# the spurious ``-U_1 ∂_t h``.

# +
sys.apply(AffineProjection(state, rewrite_basis_args=False))
sys.describe()
# -


# ## 8. Step 6 — Expand u into modes (u = U_0 + U_1 · φ_u_1(ζ))
#
# After expansion, the integrand reduces to a polynomial in ``ζ``
# times ``∂_t U_k`` (the coefficient time derivatives).  No
# Leibniz boundary corrections were generated in §5, so all that
# remains is the bulk volume integration of ``∂_t U_k`` weighted
# by ``∫_0^1 φ_u_0(ζ) · φ_u_k(ζ) dζ = δ_{0k} / (2·0+1) = δ_{0k}``.
# The result will be just ``h · ∂_t U_0`` — the time-derivative
# acting on the j=0 mode of the bulk-projected velocity.

# +
sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
sys.describe()
# -


# ## 9. Step 7 — EvaluateIntegrals + ∂_t b → 0
#
# The polynomial integrals collapse and the chain-rule
# contribution materialises as a closed-form expression in
# ``(U_0, U_1, h, ∂_t h)``.

# +
sys.apply(EvaluateIntegrals(state)).simplify()
sys.doit()
sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()
sys.describe()
# -


# ## 10. Comparison with what Leibniz on t requires
#
# The mathematically correct value of
# ``∫_b^η φ_u_0((z-b)/h) · ∂_t u dz`` via Leibniz on t (with
# t-dependent boundaries ``b(t,x)``, ``η(t,x)``) is:
#
#     ∂_t [ ∫_b^η φ_u_0((z-b)/h) · u dz ]
#       − φ_u_0(1) · u|_η · ∂_t η
#       + φ_u_0(0) · u|_b · ∂_t b
#
# With ``φ_u_0 ≡ 1`` (shifted Legendre P_0), ``∂_t b = 0``, and
# ``∂_t η = ∂_t h``:
#
#     = ∂_t (h U_0)  −  u|_η · ∂_t h
#     = ∂_t (h U_0)  −  (U_0 + U_1·φ_u_1(1)) · ∂_t h
#
# For Zoomy's shifted-Legendre convention ``φ_u_1(ζ) = 1 − 2ζ``,
# ``φ_u_1(1) = −1``, so ``u|_η = U_0 − U_1`` and the correct
# Leibniz answer is:
#
#     ∂_t (h U_0)  −  (U_0 − U_1) · ∂_t h
#       = h ∂_t U_0  +  U_1 · ∂_t h
#
# The chain produces just ``h ∂_t U_0`` — the **bulk** integration
# only.  The Leibniz boundary corrections ``−u|_η ∂_t η`` and
# ``+u|_b ∂_t b`` are **missing** because ``Integrate(method="auto")``
# fell through to ``direct`` in §5 (refused Leibniz because the
# t-dependent coefficient ``φ_u_0((z-b)/h)`` failed
# ``_extract_derivative``'s coeff-independence check).

# +
h, U_0, U_1 = state.h, coeffs_u[0], coeffs_u[1]
correct_leibniz = (
    sp.Derivative(h * U_0, state.t).doit()
    - (U_0 + U_1 * basis_u.eval(1, sp.S.One)) * sp.Derivative(h, state.t)
)
actual = sys._tree.dt_u.expr.doit()
diff = sp.simplify(sp.expand(actual - correct_leibniz))
print("Correct Leibniz answer:")
sp.pprint(sp.expand(correct_leibniz))
print()
print("Chain output (actual):")
sp.pprint(sp.expand(actual))
print()
print("Diff (actual − correct):")
sp.pprint(diff)
print()
print("This diff shows the *missing* Leibniz boundary terms that")
print("would appear if Integrate could fire the t-Leibniz route on")
print("a t-dependent test coefficient.")
# -


# ## 11. Discussion
#
# **Root cause — pinpointed in §5.**  ``Integrate(method="auto")``
# decides whether to fire Leibniz on a derivative variable ``v`` by
# decomposing the integrand as ``coeff · ∂_v F`` via
# ``_extract_derivative`` ([ins_generator.py:5185](
# ../../library/zoomy_core/zoomy_core/model/models/ins_generator.py#L5185
# )).  That decomposition succeeds only when ``coeff`` is
# **independent of ``v``**.
#
# With z-only test argument ``φ_u_0(z)``: ``coeff = φ_u_0(z)`` is
# t-independent → Leibniz on t fires → produces the textbook
# Leibniz formula
# ``∂_t ∫_b^η φ · u dz − φ(η) · u|_η · ∂_t η + φ(b) · u|_b · ∂_t b``.
# After AffineProjection rewrites ``φ(η) → φ(1)``, ``φ(b) → φ(0)``,
# and EvaluateIntegrals collapses everything, the conservative
# form ``∂_t(h U_0) − u|_η ∂_t h + …`` materialises with all
# boundary corrections.
#
# With affine test argument ``φ_u_0((z-b)/h)``:
# ``coeff = φ_u_0((z-b)/h)`` depends on ``b(t,x)`` and ``h(t,x)``
# → coeff is t-dependent → ``_extract_derivative`` refuses → auto
# routes to ``direct`` (unevaluated ``Integral(...)``) → no
# Leibniz boundary terms generated → bulk volume only.  The
# missing ``−u|_η ∂_t h`` is the ``-U_1 ∂_t h`` we observe in
# ``xmom_j0`` of the full chain.
#
# **Where the fix should land.**  Three candidate designs, in
# order of increasing scope:
#
# 1. **Relax ``_extract_derivative``'s coeff check for `t`-dependence
#    that flows only through `b` and `h`.**  When ``coeff``
#    depends on ``v`` only via the variable-domain functions
#    ``b(t,x)`` and ``h(t,x)``, fire Leibniz anyway — the
#    boundary derivatives ``∂_t b``, ``∂_t h`` end up as
#    ``∂_t η = ∂_t b + ∂_t h`` and ``∂_t b`` factors that
#    InterfaceKBC at η / b already knows how to substitute.
#    Smallest change; preserves the auto-mode contract.
#
# 2. **Make AffineProjection emit Leibniz boundary terms when it
#    finds an unevaluated ``Integral(... · ∂_v F, …)`` whose coeff
#    is t-dependent only via ``(z-b)/h``.**  Push the boundary
#    bookkeeping into AffineProjection rather than Integrate.
#    Lets Integrate stay a pure volume operator.
#
# 3. **Pre-distribute via a new ``ProductRule(variables=[t])``.**
#    Apply ``ProductRule`` over ``t`` *before* Integrate so
#    ``coeff · ∂_t u`` becomes ``∂_t(coeff · u) − ∂_t(coeff) · u``;
#    the first piece is a conservative ``∂_t(h-projected-mass)``
#    and the second a boundary-source term that
#    InterfaceKBC absorbs.  Largest change but architecturally
#    most uniform — momentum and continuity then go through the
#    *same* primitives with no special-case.
#
# **Until that's in place**, the working pattern in
# ``vam_galerkin.derive_model`` is the mixed-test-arg convention:
#
# - **Continuity** uses ``φ_p_k((z-b)/h)`` — needed for the
#   ``-∫ u · ∂_x φ_j|_z dz`` chain-rule volume term at j ≥ 1.
#   Continuity has no ``∂_t`` term in its integrand, so the
#   coeff-dependence issue from §5 doesn't bite.
# - **Momentum** uses ``φ_u_k(z)`` / ``φ_w_k(z)`` (z-only), so
#   ``coeff = φ(z)`` is t-independent and the Leibniz route
#   fires correctly on the ``∂_t u`` and ``∂_t w`` integrands.
