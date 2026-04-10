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
# # PDE Generator: Architecture Report
#
# | Building block | What it does |
# |---------------|-------------|
# | `StateSpace` | Shared symbols: coordinates, fields, stress tensor, parameters |
# | `Expression` | PDE wrapper with `.terms`, `[i]`, `.project()`, `.ibp()`, `.apply()` |
# | `Relation` | Base for substitution rules (lhs = rhs), with LaTeX display |
# | `Assumption(Relation)` | Physical conditions (kinematic BCs, hydrostatic) |
# | `Material(Relation)` | Constitutive models (Newtonian, inviscid) |
# | `FullINS(state)` | All 4 INS equations, built from a StateSpace |

# %%
import sympy as sp
from sympy import Symbol, Function, Derivative, S, Rational
sp.init_printing()

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, integrate_by_parts,
    materials, assumptions,
)

# %% [markdown]
# ---
# ## 1. StateSpace: shared symbols
#
# Everything — equations, materials, assumptions — references the same `StateSpace`.

# %%
state = StateSpace(dimension=2)
print(f"Coordinates: t={state.t}, x={state.x}, z={state.z}")
print(f"Fields: u={state.u}, w={state.w}, p={state.p}")
print(f"Stress: tau_xz={state.tau['xz']}")
print(f"Bathymetry: b={state.b}, H={state.H}, eta={state.eta}")
print(f"Parameters: rho={state.rho}, g={state.g}")

# %% [markdown]
# ---
# ## 2. FullINS: equations from the state

# %%
ins = FullINS(state)
for eq in ins.equations:
    print(f"\n--- {eq.name} ({len(eq)} terms) ---")
    display(eq.expr)

# %% [markdown]
# ### Browsing and dropping terms manually

# %%
xm = ins.x_momentum
print(f"X-momentum: {len(xm)} terms\n")
for i, term in enumerate(xm):
    display(sp.Eq(Symbol(f"T_{i}"), term.expr))

# %% [markdown]
# Drop the z-inertia from z-momentum manually to get hydrostatic balance:

# %%
zm = ins.z_momentum
print("Full z-momentum:")
display(zm.expr)

# Keep only pressure + gravity + stress (drop dw/dt, d(wu)/dx, d(ww)/dz)
zm_hydro = Expression(S.Zero, "z_hydrostatic")
for term in zm:
    has_w_inertia = (
        term.has(Derivative(state.w, state.t))
        or term.has(Derivative(state.w * state.u, state.x))
        or term.has(Derivative(state.w * state.w, state.z))
    )
    if not has_w_inertia:
        zm_hydro = zm_hydro + term

print("\nHydrostatic balance:")
display(zm_hydro.expr)

# %% [markdown]
# ---
# ## 3. Materials: constitutive models
#
# Materials take the `StateSpace`, not the equations. They're independent.

# %%
newton = materials.newtonian(state)
print(f"Type: {type(newton).__name__}")
print(f"nu = {newton.nu}")
print(f"Number of substitution rules: {len(newton)}")
display(newton)  # renders as aligned equations in notebook

# %%
# Apply to full equation
xm_newton = ins.x_momentum.apply(newton)
print("X-momentum with Newtonian stress:")
display(xm_newton.simplify().expr)

# %%
# Apply to single term
stress_term = ins.x_momentum[2]
print(f"Original: {stress_term.expr}")
print(f"After Newtonian: {stress_term.apply(newton).simplify().expr}")

# %%
# Inviscid
inv = materials.inviscid(state)
display(inv)

xm_inviscid = ins.x_momentum.apply(inv)
print("Inviscid x-momentum:")
display(xm_inviscid.expr)

# %% [markdown]
# ---
# ## 4. Assumptions: physical conditions
#
# Also take `StateSpace`, not equations.

# %%
kbc_bot = assumptions.kinematic_bc_bottom(state)
kbc_top = assumptions.kinematic_bc_surface(state)
hydro = assumptions.hydrostatic_pressure(state)

print("Kinematic BC (bottom):")
display(kbc_bot)

print("\nKinematic BC (surface):")
display(kbc_top)

print("\nHydrostatic pressure:")
display(hydro)

# %% [markdown]
# ### Chaining: apply multiple in one call

# %%
xm_simple = ins.x_momentum.apply(hydro, inv)
print("Hydrostatic + inviscid x-momentum:")
display(xm_simple.simplify().expr)

# %% [markdown]
# ---
# ## 5. Projection and Integration by Parts

# %%
phi_0 = S.One
b, eta = state.b, state.eta

# Standard projection
cont_projected = ins.continuity.project(phi_0, state.z, domain=(b, eta))
print("Projected continuity:")
display(cont_projected.expr)

# %% [markdown]
# ### IBP on the $\partial_z(uw)$ term

# %%
duwdz = ins.x_momentum[4]
print(f"Term: {duwdz.expr}")

ibp = duwdz.ibp(var=state.z, test_weight=phi_0, domain=(b, eta))
print(f"\n  integrate:  {ibp.integrate.expr}")
print(f"  upper:      {ibp.boundary_upper.expr}")
print(f"  lower:      {ibp.boundary_lower.expr}")

# %% [markdown]
# ### Apply kinematic BCs to boundary terms

# %%
ibp_bc = ibp.apply_bcs(bc_lower=kbc_bot, bc_upper=kbc_top)
print("After kinematic BCs:")
print(f"  upper: {ibp_bc.boundary_upper.expr}")
print(f"  lower: {ibp_bc.boundary_lower.expr}")

assembled = ibp_bc.assemble()
print(f"\nAssembled: {assembled.expr}")

# %% [markdown]
# ### Stress boundary conditions

# %%
tau_b, tau_s = Symbol("tau_b"), Symbol("tau_s")
stress_bc_bot = {state.tau["xz"].subs(state.z, b): tau_b}
stress_bc_top = {state.tau["xz"].subs(state.z, eta): tau_s}

stress_term = ins.x_momentum[2]
print(f"Stress term: {stress_term.expr}")

stress_ibp = stress_term.ibp(var=state.z, test_weight=phi_0, domain=(b, eta))
stress_with_bcs = stress_ibp.apply_bcs(bc_lower=stress_bc_bot, bc_upper=stress_bc_top)
print(f"\nAfter BCs:")
print(f"  integrate: {stress_with_bcs.integrate.expr}")
print(f"  upper:     {stress_with_bcs.boundary_upper.expr}")
print(f"  lower:     {stress_with_bcs.boundary_lower.expr}")

# %% [markdown]
# ---
# ## Class Hierarchy
#
# ```
# SymbolicBase (name + display)
# ├── Expression   — wraps a SymPy expr, adds .terms, .project(), .ibp(), .apply()
# └── Relation     — wraps {lhs: rhs} substitutions, adds .apply_to(), LaTeX display
#     ├── Assumption  — kinematic BCs, hydrostatic pressure, ...
#     └── Material    — Newtonian, inviscid, ...
#
# StateSpace       — coordinates, fields, stress tensor, parameters (shared by all)
# FullINS(state)   — builds Expression objects for each INS equation
# ```
#
# Key design: `materials.newtonian(state)` and `assumptions.hydrostatic_pressure(state)`
# take `StateSpace`, NOT `FullINS`. They're independent of the equation system.
