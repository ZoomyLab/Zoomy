# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deriving Shallow Water Models from the INS
#
# This notebook derives both the **SME** (hydrostatic) and **VAM** (non-hydrostatic)
# from the full 3D INS, step by step. Every equation emerges from:
#
# 1. Depth integration with **Leibniz rule** and **fundamental theorem**
# 2. **Kinematic BCs** at surface and bottom
# 3. **Galerkin projection** onto a polynomial basis
#
# Nothing is hardcoded — the pipeline is fully symbolic.

# %%
import sympy as sp
from sympy import Symbol, Function, Derivative, Integral, simplify, expand, Rational
from IPython.display import display, Math
sp.init_printing()

def show_eq(lhs, rhs):
    display(Math(lhs + " = " + sp.latex(rhs)))

# %% [markdown]
# ---
# ## 1. Starting Point: the 3D INS

# %%
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
    Newtonian, Inviscid,
)

state = StateSpace(dimension=2)  # xz plane
ins = FullINS(state)
z = state.z
b, H, eta = state.b, state.H, state.eta

print("Full 3D INS (xz plane):")
for eq in ins.equations:
    display(Math(r"\text{" + eq.name + r": }\quad " + sp.latex(eq.expr) + " = 0"))

# %% [markdown]
# ---
# ## 2. Mass Conservation: Continuity → $\partial H/\partial t + \partial(h\bar{u})/\partial x = 0$
#
# Depth-integrate $\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z} = 0$
# using Leibniz rule and fundamental theorem, then apply kinematic BCs.

# %%
kbc_b = KinematicBCBottom(state)
kbc_s = KinematicBCSurface(state)

# One call does everything: depth-integrate each term + apply BCs + simplify
mass_eq = ins.continuity.map_with_bcs(
    lambda t: t.depth_integrate(b, eta, z),
    bcs=[kbc_s, kbc_b],
)

print("Depth-integrated mass conservation:")
display(Math(sp.latex(mass_eq.expr) + " = 0"))

# %% [markdown]
# ### Project mass equation onto Legendre L2 basis
#
# Substitute $u(\zeta) = \alpha_0 + \alpha_1(1-2\zeta) + \alpha_2\varphi_2(\zeta)$.
# The integral $\int u\,dz$ becomes $H \cdot \alpha_0$ (since $c = [1,0,0]$).

# %%
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted

alpha = [Symbol(f"alpha_{k}") for k in range(3)]

mass_projected = mass_eq.project_onto_basis(
    Legendre_shifted, level=2,
    field_map={"u": alpha},
    z_var=z,
)
print("Mass equation after basis projection:")
display(Math(sp.latex(mass_projected.expr) + " = 0"))

print("\nThis is dH/dt + d(H·alpha_0)/dx = 0  (standard shallow water mass conservation)")

# %% [markdown]
# ---
# ## 3. SME: Hydrostatic x-Momentum
#
# Apply Newtonian material + hydrostatic pressure, then depth-integrate.

# %%
newton = Newtonian(state)
hydro = HydrostaticPressure(state)

# Apply assumptions
xm_hydro = ins.x_momentum.apply(newton, hydro)

print(f"x-momentum after Newtonian + hydrostatic ({len(xm_hydro)} terms):")
display(Math(sp.latex(xm_hydro.expr) + " = 0"))

# %%
# Depth-integrate
xm_di = xm_hydro.map_with_bcs(
    lambda t: t.depth_integrate(b, eta, z),
    bcs=[kbc_s, kbc_b],
)

print(f"Depth-integrated x-momentum ({len(xm_di)} terms):")

# Classify
classes = xm_di.classify(t=state.t, x=state.x, z=state.z)
for role, expr in classes.items():
    print(f"\n  {role} ({len(expr)} terms):")
    display(expr)

# %% [markdown]
# ### Project x-momentum onto Legendre L2

# %%
# Project the advective flux integral: ∫u² dz
# Without test function: scalar (for mass flux row)
# With test_mode=k: projected onto mode k (for momentum rows)

print("Key integrals after projection:\n")

# The integral of u² (advective flux)
int_uu = Expression(Integral(state.u**2, (z, b, eta)), "int_uu")

print("∫u² dz (advective flux, no test function):")
proj_uu = int_uu.project_onto_basis(Legendre_shifted, 2, {"u": alpha}, z)
show_eq(r"\int u^2\,dz", simplify(proj_uu.expr))

print("\n∫u²·φ₀ dz (Galerkin mode 0 — goes into F[hu₀]):")
proj_uu_m0 = int_uu.project_onto_basis(Legendre_shifted, 2, {"u": alpha}, z, test_mode=0)
show_eq(r"\int u^2 \varphi_0\,dz", simplify(proj_uu_m0.expr))

print("\n∫u²·φ₁ dz (Galerkin mode 1 — goes into F[hu₁]):")
proj_uu_m1 = int_uu.project_onto_basis(Legendre_shifted, 2, {"u": alpha}, z, test_mode=1)
show_eq(r"\int u^2 \varphi_1\,dz", simplify(proj_uu_m1.expr))

print("\n∫u²·φ₂ dz (Galerkin mode 2 — goes into F[hu₂]):")
proj_uu_m2 = int_uu.project_onto_basis(Legendre_shifted, 2, {"u": alpha}, z, test_mode=2)
show_eq(r"\int u^2 \varphi_2\,dz", simplify(proj_uu_m2.expr))

# %% [markdown]
# After $M^{-1}$ (with $M = \text{diag}(1, 1/3, 1/5)$ for Legendre):
#
# $$F_0 = \alpha_0^2 + \frac{\alpha_1^2}{3} + \frac{\alpha_2^2}{5}$$
# $$F_1 = 2\alpha_0\alpha_1 + \frac{4\alpha_1\alpha_2}{5}$$
# $$F_2 = 2\alpha_0\alpha_2 + \frac{2\alpha_1^2}{3} + \frac{2\alpha_2^2}{7}$$
#
# These match the `ProjectedModel.flux()` output exactly.

# %%
# Verify against ProjectedModel
from zoomy_core.model.models.model_derivation import derive_shallow_moments
from zoomy_core.model.models.projected_model import ProjectedModel, clear_matrix_cache

clear_matrix_cache()
pre = derive_shallow_moments(state, material=Newtonian(state))
ref = ProjectedModel(pre, basis_type=Legendre_shifted, level=2, eigenvalue_mode='numerical')
F_ref = ref.flux()

print("Verification: ProjectedModel flux at L2")
Minv = [1, 3, 5]
for mode in range(3):
    projected = int_uu.project_onto_basis(Legendre_shifted, 2, {"u": alpha}, z, test_mode=mode)
    after_Minv = simplify(Minv[mode] * projected.expr)
    ref_val = simplify(F_ref[2 + mode, 0])
    # Substitute alpha_k = q_{k+2}/h to compare
    h = ref.variables[1]
    subs = {alpha[k]: ref.variables[2+k] / h for k in range(3)}
    after_Minv_q = simplify(after_Minv.subs(Symbol("H(t, x)"), h).subs(subs))
    print(f"  Mode {mode}: M⁻¹·∫u²·φ_{mode} = {simplify(Minv[mode] * projected.expr / Symbol('H(t, x)'))}")
    print(f"           ProjectedModel  = {ref_val}")

# %% [markdown]
# ---
# ## 4. VAM: Non-Hydrostatic Derivation
#
# Same flow, but **without** hydrostatic assumption. Keep z-momentum and pressure.

# %%
inv = Inviscid(state)

# x-momentum (inviscid, non-hydrostatic)
xm_nh = ins.x_momentum.apply(inv)
xm_nh_di = xm_nh.map_with_bcs(
    lambda t: t.depth_integrate(b, eta, z),
    bcs=[kbc_s, kbc_b],
)
print(f"VAM x-momentum: {len(xm_nh_di)} terms after depth integration")
classes_xm = xm_nh_di.classify(t=state.t, x=state.x, z=state.z)
for role, expr in classes_xm.items():
    print(f"  {role}: {len(expr)} terms")

# z-momentum (inviscid)
zm = ins.z_momentum.apply(inv)
zm_di = zm.map_with_bcs(
    lambda t: t.depth_integrate(b, eta, z),
    bcs=[kbc_s, kbc_b],
)
print(f"\nVAM z-momentum: {len(zm_di)} terms after depth integration")
classes_zm = zm_di.classify(t=state.t, x=state.x, z=state.z)
for role, expr in classes_zm.items():
    print(f"  {role}: {len(expr)} terms")

# %% [markdown]
# ### VAM cross-momentum projection
#
# The z-momentum flux has $\int u \cdot w\,dz$ — the **cross-momentum** term.
# After basis expansion for both $u$ and $w$:

# %%
gamma = [Symbol(f"gamma_{k}") for k in range(3)]

int_uw = Expression(Integral(state.u * state.w, (z, b, eta)), "int_uw")

print("Cross-momentum ∫u·w dz (mode 0):")
proj_uw_m0 = int_uw.project_onto_basis(
    Legendre_shifted, 2,
    {"u": alpha, "w": gamma},
    z, test_mode=0,
)
show_eq(r"\int u\,w\,\varphi_0\,dz", simplify(proj_uw_m0.expr))

print("\nCross-momentum ∫u·w dz (mode 1):")
proj_uw_m1 = int_uw.project_onto_basis(
    Legendre_shifted, 2,
    {"u": alpha, "w": gamma},
    z, test_mode=1,
)
show_eq(r"\int u\,w\,\varphi_1\,dz", simplify(proj_uw_m1.expr))

# %% [markdown]
# These cross-momentum integrals use the **same** triple-product matrix $A_{lij}$
# but with mixed velocity indices ($\alpha_i \gamma_j$ instead of $\alpha_i \alpha_j$).

# %% [markdown]
# ---
# ## 5. Summary: The Symbolic Derivation Pipeline
#
# ```
# INS equations
#     │
#     │  expr.apply(material, assumptions)
#     ▼
# Simplified PDE (still in physical z)
#     │
#     │  expr.map_with_bcs(depth_integrate, bcs)
#     ▼
# Depth-integrated equations (with ∫...dz)
#     │
#     │  expr.classify(t, x, z)
#     ▼
# Classified terms (temporal, convective, source)
#     │
#     │  expr.project_onto_basis(basis, level, field_map)
#     ▼
# Projected PDE with basis matrices (M, A, D)
#     │
#     │  Apply M⁻¹
#     ▼
# Final model: flux(Q), source(Q), NC(Q)
# ```
#
# Every step uses `Expression` methods — no hardcoded equations.
