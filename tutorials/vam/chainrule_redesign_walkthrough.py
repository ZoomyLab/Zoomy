# ---
# title: "VAM chain-rule redesign — step-by-step diagnostic"
# author: Ingo Steldermann
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
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

# # VAM chain-rule redesign — diagnostic walkthrough
#
# Companion to
# [thesis/chapters/derivation_vam_chainrule_redesign_notes.md](
# ../../thesis/chapters/derivation_vam_chainrule_redesign_notes.md).
#
# **Goal of this notebook.** Step through the redesigned VAM(1, 2, 2)
# chain — explicit affine test argument ``φ_k((z-b)/h)`` plus
# ``ProductRule(variables=[x, z])`` plus
# ``AffineProjection(rewrite_basis_args=False)`` — and show, via
# ``describe()`` at every step, what the equations look like.  At the
# end, compare ``continuity.test_0`` (which should become the standard
# mass equation) and ``continuity.test_1`` (which should become
# Escalante's ``I_1`` after substituting ``∂_t h`` via mass) against
# the expected forms.
#
# The current main-branch ``VAMModelGalerkin`` uses a master-formula
# workaround for the cont-projections.  This notebook *bypasses* that
# workaround and exercises the chain primitives directly, so we can
# locate the step where the chain-rule contribution gets lost or
# spurious terms enter.
#
# **How to use.** Convert via ``jupytext --to ipynb
# tutorials/vam/chainrule_redesign_walkthrough.py`` and run cell by
# cell, inspecting the ``describe()`` output between steps.

# +
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection, EvaluateIntegrals, Expand, FullINS, InterfaceKBC,
    Integrate, Inviscid, Multiply, ProductRule, StateSpace,
)
# -


# ## 1. Setup: VAM(1, 2, 2) state, bases, coefficients
#
# Same setup as ``VAMModelGalerkin._chain_M=1, N_w=2, N_p=2``.  The
# **only difference vs. main** is the test-function argument:
# ``basis.phi[k]((z-b)/h)`` instead of ``basis.phi[k](z)``.

# +
M, N_w, N_p = 1, 2, 2

state = StateSpace(dimension=2)
z = state.z
basis_u = Legendre_shifted(level=M,   symbol="phi_u")
basis_w = Legendre_shifted(level=N_w, symbol="phi_w")
basis_p = Legendre_shifted(level=N_p, symbol="phi_p")

coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
            for k in range(M + 1)]
coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
            for k in range(N_w + 1)]
coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
            for k in range(N_p + 1)]

# Explicit affine argument — this is the redesign.
zeta_arg = (z - state.b) / state.h
test_phi_u = Zstruct(
    **{f"phi_{k}": basis_u.phi[k](zeta_arg) for k in range(M + 1)})
test_phi_w = Zstruct(
    **{f"phi_{k}": basis_w.phi[k](zeta_arg) for k in range(N_w)})
test_phi_cont = Zstruct(
    **{f"phi_{k}": basis_p.phi[k](zeta_arg) for k in range(N_p + 1)})
# -


# ## 2. Step 1 — Full INS + drop viscosity + pressure split
#
# Standard.  ``state.p = ρ g (η − z) + p_NH`` and the viscous tensor
# is zeroed.  ``sys.continuity`` and ``sys.momentum.{x,z}`` are scalar
# leaves at this point.

# +
sys = FullINS(state)
sys.apply(Inviscid(state)).simplify()
p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
          ).simplify()

sys.describe()
# -


# ## 3. Step 2 — Project against test functions
#
# Multiply continuity, momentum.x, momentum.z by their test-function
# Zstructs (``test_0..test_{N_p}``, ``test_0..test_M``,
# ``test_0..test_{N_w-1}``).  After this, ``sys._tree.continuity``
# is a Zstruct with three children at j = 0, 1, 2.

# +
sys.continuity.apply(Multiply(test_phi_cont, outer=True))
sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

sys.describe()
# -


# ## 4. Step 3 — ProductRule on [x, z]
#
# Rewrites every ``coeff · ∂_v F`` to ``∂_v(coeff · F) − ∂_v(coeff) · F``
# where ``v ∈ {x, z}``.  The chain rule on
# ``∂_x(φ_k((z-b)/h))|_z = φ_k'(ζ) · ∂_x ζ|_z`` should fire here
# automatically because the test-function argument is explicit.

# +
sys.apply(ProductRule(variables=[state.x, z]))

sys.describe()
# -


# ## 5. Step 4 — Integrate (depth-integrate over [b, η])
#
# Per-term auto-mode dispatch:
# - ``∂_x F`` → Leibniz: ``∂_x ∫ F dz − F|_η ∂_x η + F|_b ∂_x b``
# - ``∂_z F`` → fundamental theorem: ``F|_η − F|_b``
# - everything else → unevaluated ``Integral(...)``.

# +
sys.apply(Integrate(z, state.b, state.eta, method="auto"))

sys.describe()
# -


# ## 6. Step 5 — InterfaceKBC at bottom + surface
#
# Substitutes the boundary ``u·w`` cross-terms via the kinematic BCs:
# ``w|_η − u|_η ∂_x η = ∂_t h``, ``w|_b − u|_b ∂_x b = 0``.

# +
sys.apply(InterfaceKBC(state, state.b)).simplify()
sys.apply(InterfaceKBC(state, state.eta)).simplify()

sys.describe()
# -


# ## 7. Step 5b — Surface BC for ``p_NH`` (field level)
#
# ``p_NH(η) = 0`` substituted at the integrand level so the Leibniz
# boundary terms involving the surface pressure vanish.

# +
sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

sys.describe()
# -


# ## 8. Step 6 — AffineProjection (z → ζh + b in integrals)
#
# With ``rewrite_basis_args=False`` because the test-function args
# are already in affine form ``(z-b)/h``.  The internal
# ``_simplify_basis_args`` pass canonicalises ``((ζh+b)-b)/h → ζ``.

# +
sys.apply(AffineProjection(state, rewrite_basis_args=False))

sys.describe()
# -


# ## 9. Step 7 — Expand ``u``, ``w``, ``p_NH`` into modes
#
# Each field becomes ``Σ_k coeff_k · φ_k(ζ)``.

# +
sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w, state=state))
sys.apply(Expand(p_NH, basis=basis_p, coefficients=coeffs_p, state=state))

sys.describe()
# -


# ## 10. Step 8 — EvaluateIntegrals (sympy.integrate + resolve_atoms)
#
# Every ``Integral(polynomial_in_ζ, (ζ, 0, 1))`` collapses to a scalar.
# Opaque basis atoms ``φ_k(arg)`` resolve to their concrete polynomials.

# +
sys.apply(EvaluateIntegrals(state)).simplify()

sys.describe()
# -


# ## 11. Step 9 — Drop ``∂_t b`` (static bottom)
#
# Applied **after** ``EvaluateIntegrals`` plus a tree-wide ``.doit()``.
# The surface KBC introduces ``∂_t η = ∂_t b + ∂_t h``, which lives
# inside compound atoms like ``Derivative(c·b, t)`` (with ``c`` a
# constant from EvaluateIntegrals).  ``.doit()`` distributes the
# constant out so the bare ``Derivative(b, t)`` atom is produced and
# the substitution rule matches.
# Note: ``b(t, x)`` stays a function of both arguments — only the
# time-derivative is zeroed, so transient bottom topography can be
# reintroduced by removing this substitution.

# +
sys.doit()
sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

sys.describe()
# -


# ## 12. Step 10 — Bottom-KBC closure for ``W_2``
#
# ``Σ_k W_k φ_k(0) − (Σ_k U_k φ_k(0)) ∂_x b = 0`` solved for ``W_2``
# and substituted everywhere (``φ_k(0) = 1`` for shifted Legendre).

# +
u_at_b = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero)
             for k in range(M + 1))
w_at_b = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero)
             for k in range(N_w + 1))
bot_kbc = w_at_b - u_at_b * sp.Derivative(state.b, state.x).doit()
w_top_sol = sp.solve(bot_kbc, coeffs_w[N_w])[0]
sys.apply({coeffs_w[N_w]: w_top_sol}).simplify()

sys.describe()
# -


# ## 13. Step 11 — Surface-BC closure for ``P_2``
#
# ``Σ_k P_k φ_k(1) = 0`` solved for ``P_2`` and substituted everywhere.

# +
p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)
               for k in range(N_p + 1))
p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()

sys.describe()
# -


# ## 14. Compare ``continuity.test_0`` against the expected mass equation
#
# Mass should be ``∂_t h + ∂_x(h U_0) = 0``.  Any non-zero diff
# pinpoints terms that survived from the chain that shouldn't have.

# +
h, b = state.h, state.b
t, x = state.t, state.x

mass_actual = sys._tree.continuity.test_0.expr
mass_expected = sp.Derivative(h, t) + sp.Derivative(h * coeffs_u[0], x).doit()
diff_mass = sp.simplify(sp.expand(mass_actual - mass_expected))

print("Expected (mass):")
sp.pprint(mass_expected)
print()
print("Actual (continuity.test_0):")
sp.pprint(sp.expand(mass_actual))
print()
print("Diff (actual − expected):")
sp.pprint(diff_mass)
# -


# ## 15. Compare ``continuity.test_1`` against Escalante's ``I_1``
#
# After substituting ``∂_t h = -∂_x(h U_0)`` via mass, ``cont_j1``
# should match Escalante eq. (5):
# ``I_1 = h ∂_x U_0 + (1/3) ∂_x(h U_1) + (1/3) U_1 ∂_x h
#         + 2(W_0 - U_0 ∂_x b)``.
#
# The diff identifies which terms of ``I_1`` are missing or which
# extras are present — directly pinpointing where the chain-rule
# contribution is lost.

# +
cont1_actual = sys._tree.continuity.test_1.expr
dt_h_sub = -sp.Derivative(h * coeffs_u[0], x).doit()
cont1_alg = sp.expand(cont1_actual.subs(sp.Derivative(h, t), dt_h_sub))

U_0, U_1 = coeffs_u[0], coeffs_u[1]
W_0 = coeffs_w[0]
I_1 = (h * sp.Derivative(U_0, x).doit()
       + sp.Rational(1, 3) * sp.Derivative(h * U_1, x).doit()
       + sp.Rational(1, 3) * U_1 * sp.Derivative(h, x).doit()
       + 2 * (W_0 - U_0 * sp.Derivative(b, x).doit()))
diff_cont1 = sp.simplify(sp.expand(cont1_alg - I_1))

print("Expected (Escalante I_1):")
sp.pprint(I_1)
print()
print("Actual (continuity.test_1, after ∂_t h substitution):")
sp.pprint(sp.expand(cont1_alg))
print()
print("Diff (actual − I_1):")
sp.pprint(diff_cont1)
# -


# ## 16. Discussion
#
# **Status (resolved).**  After the ``System.doit()`` step in §11
# (which distributes constant factors out of compound
# ``Derivative(c·b, t)`` atoms produced by InterfaceKBC's surface
# substitution), and the consequent
# ``{∂_t b: 0}`` substitution, the residuals close fully:
#
#     mass diff     = 0
#     cont1 - I_1   = 0
#
# The original redesign-notes claim of a spurious
# ``-2 U_1 ∂_x b - U_1 ∂_x h`` in mass was an artefact of the
# ``∂_t b`` substitution being applied **before** the compound
# atoms had been collapsed by ``EvaluateIntegrals``'s
# basis-resolution and the subsequent ``.doit()``.  Reorder the
# pipeline (``EvaluateIntegrals → simplify → doit → ∂_t b → 0``)
# and the chain primitives produce Escalante eq. (4) bit-for-bit.
#
# **Where the chain still misbehaves.**  Running the *momentum*
# branch with ``φ_u_k((z-b)/h)`` test arguments (as opposed to the
# z-only ``φ_u_k(z)`` form) leaves a missing
# ``-∫ φ_u_j' · u_z · ζ ∂_t h dζ``-style term in ``xmom_jk`` —
# isolated and pinpointed in
# [tutorials/vam/dt_u_term_isolation.py](dt_u_term_isolation.py).
# That issue is upstream of the cont fix shown here: it is rooted
# in ``Integrate.method='auto'``'s coeff-independence check
# refusing to fire Leibniz on ``∂_t`` when the test coefficient
# depends on ``b(t,x)`` / ``h(t,x)`` — and the safe pattern in
# ``vam_galerkin.derive_model`` is therefore the **mixed-test-arg
# convention**: ``φ_p_k((z-b)/h)`` for continuity (where the
# chain-rule volume term at j ≥ 1 is needed and the ``∂_t``
# integrand is absent), ``φ_u_k(z)`` / ``φ_w_k(z)`` for momentum
# (where Leibniz on ``∂_t`` must fire).
