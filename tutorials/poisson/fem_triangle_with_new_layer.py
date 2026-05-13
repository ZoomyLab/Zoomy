# ---
# title: "Poisson on a triangle — symbolic Galerkin chain via .apply()"
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

# # Scalar Poisson on a triangle — symbolic Galerkin chain
#
# This notebook re-derives the local stiffness integrand for the
# scalar Poisson problem on an arbitrary triangle ``K``, using the
# same pattern as ``vam_get_pde_walkthrough``: build the residual,
# then drive each step via a single ``.apply(...)`` call on an
# ``Expression``, ending with ``.describe()``.
#
# Strong form (sign convention chosen for clean IBP):
#
# $$
# \Delta u + f = 0
# $$
#
# Steps:
# 1. Build the strong-form residual.
# 2. Multiply by the test function ``φ``.
# 3. Integrate over ``K``.
# 4. Apply the divergence theorem.
# 5. Map ``K`` to the reference simplex ``T̂``.

# +
import sympy as sp

from zoomy_core.symbolic.domains import Simplex
from zoomy_core.model.models.ins_generator import Expression, Multiply
from zoomy_core.model.models.integrate_over_domain import IntegrateOverDomain
from zoomy_core.model.models.divergence_theorem import DivergenceTheorem
from zoomy_core.model.models.map_to_reference import MapToReferenceElement
# -

# ## Setup
#
# Symbolic vertices ``V₀, V₁, V₂`` for the triangle.  Coordinates
# ``(x, y)``.  Field ``u``, source ``f``, test function ``φ``.

# +
x, y = sp.symbols("x y", real=True)
u   = sp.Function("u",   real=True)(x, y)
f   = sp.Function("f",   real=True)(x, y)
phi = sp.Function("phi", real=True)(x, y)

x0, y0, x1, y1, x2, y2 = sp.symbols("x0 y0 x1 y1 x2 y2", real=True)
K = Simplex([(x0, y0), (x1, y1), (x2, y2)], coords=(x, y), name="K")
F = (sp.diff(u, x), sp.diff(u, y))
K
# -

# ## 1. Strong residual
#
# Wrap ``Δu + f`` in an ``Expression``.  ``Expression`` provides
# ``.apply(Operation)`` and ``.describe()`` — the same surface that
# ``vam_get_pde_walkthrough`` uses.

sys = Expression(sp.diff(u, x, 2) + sp.diff(u, y, 2) + f, name="poisson")
sys.describe()

# ## 2. Multiply by the test function

sys = sys.apply(Multiply(phi))
sys.describe()

# ## 3. Integrate over K
#
# ``IntegrateOverDomain(K)`` wraps the leaf expression in a sympy
# ``Integral`` whose integration variables are exactly ``K.coords``.
# The Domain identity is implicit (the next operation re-discovers
# it from the limit-vars).

sys = sys.apply(IntegrateOverDomain(K))
sys.describe()

# ## 4. Divergence theorem
#
# The user supplies ``φ`` and the flux ``F = ∇u``.
# ``DivergenceTheorem`` walks the expression, decomposes each
# integrand on ``K`` into additive components, and IBPs every
# component matching ``φ · ∂_i F[i]``.  Components that don't match
# (e.g. the source ``φ·f``) stay inside their own ``Integral``.
# Boundary contributions are summed into a single ``BoundaryIntegral``.

sys = sys.apply(DivergenceTheorem(K, phi=phi, F=F, form="weighted"))
sys.describe()

# ## 5. Affine map to the reference simplex T̂
#
# ``MapToReferenceElement(K)``:
#
# - Substitutes ``(x, y) → V₀ + B·(ξ₀, ξ₁)`` everywhere.
# - Chain-rules each first-order ``Derivative(f, x_i)`` via
#   ``∇_x = B⁻ᵀ ∇_ξ``.
# - Multiplies every volume integrand by ``|det B|``.
# - Updates the ``BoundaryIntegral`` domain to ``∂T̂``.

sys = sys.apply(MapToReferenceElement(K))
sys.describe()

# ## Verification — match the textbook local-stiffness integrand
#
# The classical FEM local-stiffness integrand on the reference
# simplex is ``(B⁻ᵀ ∇_ξ φ)·(B⁻ᵀ ∇_ξ u) |det B|``.  We assert
# structural equality (under ``.doit()`` to expand the chain rule
# in our held form).

# +
B, V0 = K.affine_map()
BinvT = B.inv().T
ref = K.reference()
xi0, xi1 = ref.coords
ref_image = V0 + B * sp.Matrix([[xi0], [xi1]])
phi_ref = phi.subs({x: ref_image[0, 0], y: ref_image[1, 0]})
u_ref   = u.subs(  {x: ref_image[0, 0], y: ref_image[1, 0]})
grad_phi_xi = sp.Matrix([sp.diff(phi_ref, xi0), sp.diff(phi_ref, xi1)])
grad_u_xi   = sp.Matrix([sp.diff(u_ref,   xi0), sp.diff(u_ref,   xi1)])
expected_inner = ((BinvT * grad_phi_xi).T @ (BinvT * grad_u_xi))[0, 0] \
    * sp.Abs(B.det())

def _has_func(expr, name):
    return any(a.func.__name__ == name for a in expr.atoms(sp.Function))

# Find the laplacian volume integral (contains u but not f).
volume_pieces = [I for I in sys.expr.atoms(sp.Integral)
                 if I.args[1][0] == xi0]
laplace_vol = next(I for I in volume_pieces
                   if _has_func(I.args[0], "u") and not _has_func(I.args[0], "f"))
diff = sp.simplify(sp.expand(
    laplace_vol.args[0].doit() - expected_inner.doit()))
assert diff == 0, "Mapped Poisson volume integrand should match the textbook"
diff
# -

# ## Same derivation, packaged as a Model
#
# The chain above is exactly what ``ScalarPoissonGalerkin.derive_model``
# does internally.  Instantiating the model runs the chain and stores
# every snapshot, so each stage is inspectable.

# +
from zoomy_core.model.models.scalar_poisson_galerkin import (
    ScalarPoissonGalerkin,
)

model = ScalarPoissonGalerkin()
model.describe()
# -

# ### Intermediate stages
#
# Each step in ``derive_model`` is exposed as an attribute, so we can
# walk through the same derivation without re-running the chain:

model._after_multiply.describe()

# +
model._after_div_thm.describe()
# -

# ## Summary
#
# Compared to ``fem_triangle_experiment.py`` (the original gap
# analysis):
#
# | Item | What it required | Status |
# | --- | --- | --- |
# | 1 | Multi-dim integration domain | ✅ — ``IntegrateOverDomain`` (or sympy native) |
# | 2 | Divergence theorem | ✅ — ``DivergenceTheorem`` |
# | 3 | Matrix Jacobian | ✅ — ``MapToReferenceElement`` |
# | 4 | Gradient transformation ``∇_x = B⁻ᵀ ∇_ξ`` | ✅ — same op, chain-rules each ``Derivative`` |
# | 5 | Multi-dim reference basis | ⏳ — follow-up plan |
#
# Every step is one ``.apply(...)`` call.  The Model class wraps the
# whole chain so it can be reused, parameterised on triangle vertices,
# and described as a single artefact.
