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
# # PDE Generator: Full Report
#
# Two layers of the framework:
#
# 1. **Composable algebra** (`ins_generator.py`): `Expression`, `FullINS`, `materials`, `assumptions`
# 2. **Integration engine** (`pde_generator.py`): `PiecewiseIntegrator`, `LayeredAnsatz`, `InterfaceRouter`
#
# This notebook walks through both.

# %%
import sympy as sp
from sympy import Symbol, Function, Derivative, S, Rational, Integral, Heaviside, DiracDelta
sp.init_printing()

# %% [markdown]
# ---
# ## Part 1: Composable Algebra
#
# ### 1.1 FullINS — the starting point

# %%
from zoomy_core.model.models.ins_generator import (
    FullINS, Expression, integrate_by_parts,
    materials, assumptions,
)

ins = FullINS(dimension=1)
for eq in ins.equations:
    print(f"\n--- {eq.name} ({len(eq)} terms) ---")
    display(eq.expr)

# %% [markdown]
# ### 1.2 Material models
#
# Applied via `.apply()`. Works on full equations or individual terms.

# %%
newton = materials.newtonian(ins)
print("Newtonian substitution replaces abstract tau_ij with mu*(du_i/dx_j + du_j/dx_i)")
print(f"  nu parameter: {newton.nu}")

xm_newton = ins.x_momentum.apply(newton)
display(xm_newton.simplify().expr)

# %%
# Inviscid: all tau = 0
xm_inviscid = ins.x_momentum.apply(materials.inviscid(ins))
display(xm_inviscid.expr)

# %% [markdown]
# ### 1.3 Assumptions
#
# Kinematic BCs and hydrostatic pressure as substitution objects.

# %%
kbc_bot = assumptions.kinematic_bc_bottom(ins)
kbc_top = assumptions.kinematic_bc_surface(ins)
hydro = assumptions.hydrostatic_pressure(ins)

print("Kinematic BC (bottom):")
for old, new in kbc_bot.subs_map.items():
    display(sp.Eq(old, new))

print("\nHydrostatic pressure:")
for old, new in hydro.subs_map.items():
    display(sp.Eq(old, new))

# %% [markdown]
# ### 1.4 Projection and IBP

# %%
phi_0 = S.One
b, eta = ins.b, ins.eta

# IBP on d(uw)/dz
duwdz = ins.x_momentum[4]
print(f"Term: {duwdz.expr}")

ibp = duwdz.ibp(var=ins.z, test_weight=phi_0, domain=(b, eta))
print(f"\n  integrate:       {ibp.integrate.expr}")
print(f"  boundary_upper:  {ibp.boundary_upper.expr}")
print(f"  boundary_lower:  {ibp.boundary_lower.expr}")

# Apply kinematic BCs
ibp_bc = ibp.apply_bcs(bc_lower=kbc_bot.subs_map, bc_upper=kbc_top.subs_map)
print(f"\nAfter kinematic BCs:")
print(f"  upper: {ibp_bc.boundary_upper.expr}")
print(f"  lower: {ibp_bc.boundary_lower.expr}")

# %% [markdown]
# ---
# ## Part 2: Integration Engine
#
# ### 2.1 PiecewiseIntegrator — handles Heaviside without AST explosion

# %%
from zoomy_core.model.models.pde_generator import PiecewiseIntegrator

z = Symbol("z")
integrator = PiecewiseIntegrator(z, [S.Zero, Rational(1, 2), S.One])

# Smooth: Heaviside-windowed polynomial
W0 = Heaviside(z) - Heaviside(z - Rational(1, 2))
result = integrator.integrate(z**2 * W0)
print(f"integral z^2 * W0 = {result}  (expect 1/24)")

# Delta: sifting property
result = integrator.integrate((3*z + 1) * DiracDelta(z - Rational(1, 2)))
print(f"integral (3z+1)*delta(z-1/2) = {result}  (expect 5/2)")

# %% [markdown]
# ### 2.2 LayeredAnsatz — Heaviside windowing for multi-layer

# %%
from zoomy_core.model.models.pde_generator import LayeredAnsatz
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Monomials

la = LayeredAnsatz(n_layers=2, basis=Monomials(level=0), dimension=1)
zeta = la.zeta

print(f"Interfaces: {la.layer_interfaces}")
print(f"DOFs: {la.get_all_dof_symbols('u')}")
print(f"\nFull ansatz:")
display(la.full_ansatz("u"))

# %% [markdown]
# ### 2.3 Derivatives produce DiracDelta at interfaces

# %%
deriv = la.derivative_ansatz("u")
print("d/dzeta [ansatz] =")
display(sp.expand(deriv))

# %% [markdown]
# ### 2.4 InterfaceRouter — boundary vs internal classification

# %%
from zoomy_core.model.models.pde_generator import InterfaceRouter

router = InterfaceRouter(la.layer_interfaces, zeta)
classified = router.classify_delta_terms(deriv)

print(f"Boundary deltas: {classified.n_boundary}")
for j in classified.boundary:
    print(f"  {j}")
print(f"\nInternal deltas: {classified.n_internal}")
for j in classified.internal:
    print(f"  {j}")

# %% [markdown]
# ---
# ## Part 3: Generated Model (old pipeline)
#
# The `GeneratedShallowModel` assembles everything into a `Model`-compatible class.
# Verified algebraically against `ShallowMomentsTopo`.

# %%
from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel

m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
print(f"Variables: {list(m.variables.values())}")
print(f"\nFlux:")
display(m.flux().tomatrix())
print(f"\nHydrostatic pressure:")
display(m.hydrostatic_pressure().tomatrix())
print(f"\nEigenvalues:")
for ev in m.eigenvalues():
    display(ev)

# %% [markdown]
# ### Verification: matches ShallowMomentsTopo

# %%
from zoomy_core.model.models.shallow_moments_topo import ShallowMomentsTopo

gen = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
ref = ShallowMomentsTopo(level=1, dimension=1)

subs = {}
for i in range(gen.n_variables):
    gv, rv = gen.variables[i], ref.variables[i]
    if gv != rv:
        subs[gv] = rv
for k in gen.parameters.keys():
    gp = gen.parameters[k]
    if ref.parameters.contains(k):
        rp = ref.parameters[k]
        if gp != rp:
            subs[gp] = rp

print("Flux difference (generated - reference):")
F_gen, F_ref = gen.flux(), ref.flux()
all_zero = True
for i in range(gen.n_variables):
    diff = sp.simplify(sp.expand(F_gen[i, 0].subs(subs) - F_ref[i, 0]))
    print(f"  Row {i}: {'OK' if diff == 0 else diff}")
    if diff != 0:
        all_zero = False
if all_zero:
    print("\nAll match.")
