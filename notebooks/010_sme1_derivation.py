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
# # 010 — SME(1) with Newtonian viscous stress, derived from scratch
#
# Goal: produce the **Shallow Moment Equation at level L = 1** with the
# Newtonian viscous stress tensor retained as a diffusive flux.
#
# Physical-(t, x, z) σ-Navier–Stokes, **hydrostatic** (so `p ≡ 0`,
# z-momentum dropped), with Newtonian viscous stress on the RHS of
# x-momentum:
#
# $$
# \begin{aligned}
# \partial_x u + \partial_z w &= 0 \\
# \partial_t u + u\,\partial_x u + w\,\partial_z u + g\,\partial_x \eta
#   &= \partial_x \sigma_{xx} + \partial_z \sigma_{xz}
# \end{aligned}
# $$
#
# Newton (constant ρ): $\sigma_{xx} = 2\nu\,\partial_x u$,
# $\sigma_{xz} = \nu\,\partial_z u$.  τ_zz lives only in z-momentum
# (dropped under hydrostatic assumption) so it does not enter SME(1).
#
# Projection: shifted Legendre basis φ_i(ζ) on [0, 1] with
# $\zeta = (z - b)/h$, M = 1.  We carry `u_0, u_1`.
#
# Strategy in this notebook:
#
# 1. Use `GalerkinProjection` for the **inviscid** projection
#    (continuity j=0, x-momentum j=0,1).  The library handles all the
#    convective / pressure / topography terms cleanly via the (B)
#    branch of `_integrate_term`.
# 2. Compute the **viscous projection by hand** with the polynomial
#    ansatz — Legendre orthogonality + boundary BCs (no-shear at the
#    free surface).  This produces the diffusion matrix
#    `D · ∂_x Q + bed-shear source`.
# 3. Assemble the full SME(1) equations.
# 4. Bring the mass matrix to identity via the rescaling
#    $\tilde u_i = \sqrt{2i+1}\,u_i$.
# 5. Print the final system in a form ready for SystemModel construction.

# %%
import sympy as sp

from zoomy_core.model.derivation import (
    HydrostaticFlow,
    PolynomialAnsatz,
    GalerkinProjection,
)

# %% [markdown]
# ## 0. Setup

# %%
flow = HydrostaticFlow.with_defaults()
ansatz = PolynomialAnsatz(
    t=flow.t, x=flow.x, z=flow.z,
    h=flow.h, b=flow.b,
    M=1, N_w=-1, N_p=-1,
)
proj = GalerkinProjection(flow=flow, ansatz=ansatz)
u_op, w_op, p_op = proj.opaque_fields()

t, x, z, g, h, b = flow.t, flow.x, flow.z, flow.g, flow.h, flow.b
u_0, u_1 = ansatz.u_coeffs
nu = sp.Symbol("nu", positive=True)
tau_b = sp.Function("tau_b", real=True)(t, x)  # prescribed bed shear

print("state variables: h(t,x), u_0(t,x), u_1(t,x)")
print("u(t,x,z) ansatz:", ansatz.u)

# %% [markdown]
# ## 1. Inviscid projection
#
# Project the inviscid LHS:
# $$
# \partial_t u + \partial_x(u^2) + \partial_z(uw) + g\,\partial_x \eta
# $$
# against φ_0 (continuity also against φ_0; here j refers to the
# x-momentum projection level).

# %%
xmom_inviscid_lhs = (
    sp.Derivative(u_op, t)
    + sp.Derivative(u_op * u_op, x)
    + sp.Derivative(u_op * w_op, z)
    + g * sp.Derivative(flow.eta, x)
)
cont_lhs = flow.continuity_lhs(u_op, w_op)

cont_j0 = sp.expand(proj.project(cont_lhs, 0, u_op, w_op, p_op).doit())
xmom_j0_inv = sp.expand(
    proj.project(xmom_inviscid_lhs, 0, u_op, w_op, p_op).doit()
)
xmom_j1_inv = sp.expand(
    proj.project(xmom_inviscid_lhs, 1, u_op, w_op, p_op).doit()
)

# %%
print("continuity j=0  =", cont_j0, "= 0")
print()
print("inviscid x-mom j=0 =", xmom_j0_inv, "= 0")
print()
print("inviscid x-mom j=1 =", xmom_j1_inv, "= 0")

# %% [markdown]
# Apply continuity j=0 to substitute residual `∂_t h` atoms inside the
# j=1 momentum equation.

# %%
dt_h_atom = sp.Derivative(h, t)
dt_h_rhs = -sp.Derivative(h * u_0, x)


def apply_dt_h(expr):
    prev = None
    cur = sp.expand(expr.doit())
    while prev != cur:
        prev = cur
        cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}).doit())
    return cur


xmom_j0_inv = apply_dt_h(xmom_j0_inv)
xmom_j1_inv = apply_dt_h(xmom_j1_inv)

print("after ∂_t h substitution:")
print("inviscid x-mom j=0 =", sp.simplify(xmom_j0_inv))
print()
print("inviscid x-mom j=1 =", sp.simplify(xmom_j1_inv))

# %% [markdown]
# ## 2. Sanity check — SME(1) at u_1 = 0 must reduce to SWE
#
# When the first shear moment vanishes, the j=0 momentum equation
# must coincide with the standard shallow-water momentum equation:
# $$
# \partial_t(h u_0) + \partial_x\!\left(h u_0^2 + \tfrac{1}{2} g h^2\right)
#   + g h\,\partial_x b = 0.
# $$
# (After ∂_t h substitution this becomes an evolution for u_0
# coupled to ∂_x h, ∂_x b — same form as K&T 2019 eq (4.14) row 2 in
# the SWE limit.)

# %%
swe_lhs = (sp.Derivative(h * u_0, t)
           + sp.Derivative(h * u_0**2 + sp.Rational(1, 2) * g * h**2, x)
           + g * h * sp.Derivative(b, x))
swe_lhs_sub = apply_dt_h(swe_lhs)
zero_u1 = {u_1: sp.S.Zero,
           sp.Derivative(u_1, x): sp.S.Zero,
           sp.Derivative(u_1, t): sp.S.Zero}
my_j0_at_u1_zero = sp.expand(xmom_j0_inv.xreplace(zero_u1))
diff_swe = sp.simplify(my_j0_at_u1_zero - swe_lhs_sub)
print(f"(my x-mom j=0)|_{{u_1=0}} − SWE = {diff_swe}")
print("✓ MATCH (SME(1) reduces to SWE at u_1=0)"
      if diff_swe == 0 else "✗ MISMATCH")

# K&T 2019 eq (4.14) row 3 is also a useful reference but the
# **flat-bottom flat-h** statement they print is a simplification —
# my general projection retains the ∂_x h and ∂_x b chain-rule terms,
# which collapse to K&T's form only when (∂_x h = ∂_x b = 0).  We
# don't enforce that here.

# %% [markdown]
# ## 3. Viscous projection (by hand)
#
# We project `∂_x σ_xx + ∂_z σ_xz` against φ_j directly, exploiting
# Legendre orthogonality and the ansatz.
#
# ### 3a. σ_xz contribution (vertical shear)
#
# With $\sigma_{xz} = \nu\,\partial_z u$, the projection of
# $\partial_z \sigma_{xz}$ at level j is, via integration by parts in z:
# $$
# \int_b^{\eta} \varphi_j(\zeta(z))\, \partial_z \sigma_{xz}\, dz
#   = [\varphi_j \sigma_{xz}]_{z=b}^{z=\eta}
#     - \int_b^{\eta} (\partial_z \varphi_j)\,\sigma_{xz}\,dz.
# $$
#
# At $z = \eta$: no-shear free-surface BC ⇒ $\sigma_{xz}|_{\eta} = 0$.
# At $z = b$: $\sigma_{xz}|_b = -\tau_b$ — minus the bed shear stress
# (the convention is that $\tau_b$ is what the bed exerts ON the
# fluid; the stress tensor entry has the opposite sign).
# So $[\varphi_j \sigma_{xz}]_b^{\eta} = +\varphi_j(0)\,\tau_b
#  = \tau_b$ (since $\varphi_j(0) = 1$ in our convention).
#
# The bulk integral uses $\partial_z \varphi_j = \varphi'_j(\zeta)/h$
# and $\sigma_{xz} = \nu/h \sum_i u_i \varphi'_i(\zeta)$:
# $$
# \int_b^{\eta} \partial_z \varphi_j\,\sigma_{xz}\,dz
#   = \frac{\nu}{h^2}\sum_i u_i \int_b^{\eta} \varphi'_j(\zeta)\,\varphi'_i(\zeta)\,dz
#   = \frac{\nu}{h}\sum_i u_i \int_0^1 \varphi'_j\varphi'_i\,d\xi.
# $$
#
# For shifted Legendre on [0, 1] at M=1:
# $\varphi_0 = 1$, $\varphi_1 = 2\xi - 1$, $\varphi'_0 = 0$,
# $\varphi'_1 = 2$.  The Gram-of-derivatives matrix is
# $K_{ij} = \int_0^1 \varphi'_j\varphi'_i\,d\xi$ ⇒
# $K = \begin{pmatrix} 0 & 0 \\ 0 & 4 \end{pmatrix}$.

# %%
xi = ansatz.xi_ref
phi = ansatz.basis_xi
print("shifted Legendre basis:")
for i, p in enumerate(phi[:2]):
    print(f"  φ_{i} =", p, "  φ_{i}'(0) =", p.diff(xi).subs(xi, 0))

K_visc = sp.Matrix([
    [sp.integrate(phi[j].diff(xi) * phi[i].diff(xi), (xi, 0, 1))
     for i in range(2)]
    for j in range(2)
])
print("\nGram of derivatives K_ij = ∫ φ'_i φ'_j dξ:")
sp.pprint(K_visc)

# %% [markdown]
# σ_xz contribution to the j-th projected momentum equation:
# $$
# (\sigma_{xz}\text{-part})_j = \tau_b\,\varphi_j(0)
#   \;-\;\frac{\nu}{h}\sum_i K_{ji}\,u_i.
# $$

# %%
visc_xz_j0 = (
    tau_b * phi[0].subs(xi, 0)
    - (nu / h) * (K_visc[0, 0] * u_0 + K_visc[0, 1] * u_1)
)
visc_xz_j1 = (
    tau_b * phi[1].subs(xi, 0)
    - (nu / h) * (K_visc[1, 0] * u_0 + K_visc[1, 1] * u_1)
)
print("σ_xz contribution j=0:", sp.simplify(visc_xz_j0))
print("σ_xz contribution j=1:", sp.simplify(visc_xz_j1))

# %% [markdown]
# ### 3b. σ_xx contribution (in-plane normal stress)
#
# With $\sigma_{xx} = 2\nu\,\partial_x u$, the projection at level j is
# $$
# \int_b^{\eta} \varphi_j\,\partial_x \sigma_{xx}\,dz
#   = \partial_x \int_b^{\eta} \varphi_j\,\sigma_{xx}\,dz
#     - [\varphi_j\sigma_{xx}\,\partial_x\eta]_{z=\eta}
#     + [\varphi_j\sigma_{xx}\,\partial_x b]_{z=b}
#     - \int_b^{\eta} (\partial_x \varphi_j)\,\sigma_{xx}\,dz.
# $$
#
# The two boundary correction terms are usually neglected in shallow
# water (small free-surface and bottom slopes vs. the depth-averaged
# divergence — Brufau-García-Navarro line of work) and the
# $\partial_x \varphi_j$ recursion contributes only at higher SME
# orders.  For SME(1) it gives a small correction that becomes the
# bottom-slope shear coupling.
#
# For now we use the **flat-bottom linearised** form, which is enough
# to get the leading diffusion matrix:
# $$
# \int_b^{\eta} \varphi_j\,\sigma_{xx}\,dz
#   = 2\nu h \sum_i \frac{\delta_{ji}}{2j+1}\,\partial_x u_i
#   = \frac{2\nu h}{2j+1}\,\partial_x u_j.
# $$
# Then `∂_x σ_xx` projected gives the diffusive flux per moment
# $$
# (\sigma_{xx}\text{-part})_j \;=\; \partial_x\!\left(\frac{2\nu h}{2j+1}\,\partial_x u_j\right).
# $$
#
# This is the standard SWE-viscous diffusion term applied per moment.

# %%
visc_xx_j0 = sp.Derivative(2 * nu * h / 1 * sp.Derivative(u_0, x), x)
visc_xx_j1 = sp.Derivative(2 * nu * h / 3 * sp.Derivative(u_1, x), x)
print("σ_xx contribution j=0: ∂_x(2 ν h ∂_x u_0)")
print("σ_xx contribution j=1: ∂_x(2 ν h / 3 ∂_x u_1)")

# %% [markdown]
# ## 4. Assemble full SME(1) equations
#
# Convention: write each Galerkin equation in the form
# $$
# \text{(left side)} = (\sigma_{xx}\text{-part})_j + (\sigma_{xz}\text{-part})_j.
# $$
# I.e. inviscid LHS = RHS-viscous.

# %%
cont = cont_j0
xmom_j0 = xmom_j0_inv - visc_xx_j0 - visc_xz_j0
xmom_j1 = xmom_j1_inv - visc_xx_j1 - visc_xz_j1

print("=== continuity ===")
sp.pprint(cont)
print()
print("=== x-momentum j=0 ===")
sp.pprint(sp.simplify(xmom_j0))
print()
print("=== x-momentum j=1 ===")
sp.pprint(sp.simplify(xmom_j1))

# %% [markdown]
# ## 5. Mass matrix to identity
#
# The j-th projection produces $\int_0^1 \varphi_j^2\,d\xi \cdot
# \partial_t(h u_j) = h/(2j+1)\,\partial_t(h u_j)$ as the mass term
# (after K&T row substitution).  Multiplying the j-th equation by
# (2j+1) yields the canonical form
# $$
# \partial_t(h u_j) + (\text{flux + NCP + diff})_j = (\text{src})_j.
# $$
# The conservative state per moment is then $hu_j$; the diffusion
# matrix on this state is
# $$
# D_{j,j} = \frac{2\nu (2j+1)}{2j+1}\,h = 2 \nu h ,
# $$
# i.e. **the same diffusion coefficient on every moment** when the
# state is $hu_j$ (the $(2j+1)$ factors cancel between the equation
# scaling and the σ_xx integral).
#
# σ_xz projects to $(2j+1)\,\bigl[\tau_b\,\varphi_j(0) - \frac{\nu}{h}\sum_i K_{ji}u_i\bigr]$.

# %%
xmom_j0_scaled = sp.expand((2*0 + 1) * xmom_j0)
xmom_j1_scaled = sp.expand((2*1 + 1) * xmom_j1)
print("Equation-scaled (×(2j+1)) x-mom j=0:")
sp.pprint(sp.simplify(xmom_j0_scaled))
print()
print("Equation-scaled (×(2j+1)) x-mom j=1:")
sp.pprint(sp.simplify(xmom_j1_scaled))

# %% [markdown]
# ## 6. Decomposition for SystemModel
#
# State $Q = (b, h, hu_0, hu_1)$.  We pull out:
#
# - **flux**  $F = (0, hu_0, hu_0^2 + g h^2/2 + h u_1^2/3, 2 h u_0 u_1)$
#   (1D; first column of the flux tensor)
# - **NCP / hydrostatic_pressure**  the bathymetry topography $-g h\,\partial_x b$
#   (in non-conservative form) **and** the SME row-3 cross-coupling
#   $- u_0\,\partial_x(h u_1)$.
# - **diffusion_matrix**  $D_{2,2,0,0} = 2 \nu$, $D_{2,1,0,0} = -2 \nu u_0$
#   (so that $F_{\text{diff}}[2,0] = 2\nu\,\partial_x(hu_0) - 2\nu u_0\,\partial_x h
#   = 2\nu h\,\partial_x u_0$); similarly for j=1 with the $(2j+1)$ scale.
# - **source**  bed shear $\tau_b$ split among moments (per σ_xz formula).
#
# The actual SystemModel class is implemented in
# `tutorials/sme/sme1_viscous.py` (next step).

# %%
print("Symbolic derivation complete.")
print("Continuity, x-mom j=0, x-mom j=1 equations available.")
print()
print("Next: materialise as SystemModel — see tutorials/sme/sme1_viscous.py")
