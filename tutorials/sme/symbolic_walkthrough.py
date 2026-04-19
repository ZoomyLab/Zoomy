# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SWE and SME from scratch — symbolic walkthrough
#
# Derivation of the Shallow-Water Equations (SWE, level=0) and the Shallow
# Moment Equations (SME, level>0) directly from incompressible Navier-Stokes,
# using only base primitives:
#
# * ``Expression.apply(...)`` on an ``_EquationProxy`` (``model.x_momentum.apply(...)``)
#   — returns the proxy so calls chain.
# * ``Integrate(var, lower, upper, method="auto"|"analytical"|"leibniz"|"fundamental_theorem"|"direct")``
#   — unified integration Operation.  Per-term dispatch through
#   ``Expression.depth_integrate`` for auto/leibniz/fundamental; whole-expression
#   ``sympy.integrate`` for analytical.  Boundary terms are kept as
#   ``Subs(f, var, bound)`` and resolved later via plain substitution.
# * ``model.z_momentum.solve_for(state.p)`` — returns an Expression whose
#   ``_as_relation`` is consumed by ``apply`` as a substitution.
# * Substitution dicts for the remaining BC closures
#   (atmospheric pressure at the free surface, zero surface/bottom
#   tangential stress, kinematic BCs on w).
# * ``Newtonian`` Relation kept as a convenience for the stress tensor.
#
# No ``DepthIntegrate``, ``HydrostaticPressure``, ``ApplyKinematicBCs``,
# ``StressFreeSurface``, ``ZeroAtmosphericPressure``, ``SimplifyIntegrals``
# shortcuts.  Every step is explicit.

# %% [markdown]
# ## Imports

# %%
import sympy as sp

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Integrate, Newtonian,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling

# %% [markdown]
# ## Step 1 — Start from the raw Navier-Stokes system

# %%
state = StateSpace(dimension=2)          # (t, x, z)
model = FullINS(state)
model.describe()

# %% [markdown]
# ## Step 2 — Hydrostatic assumption on z-momentum
#
# Set $w = 0$, $\tau_{zz} = \tau_{xz} = \tau_{zx} = 0$ inside z-momentum.
# `.apply()` returns the proxy so `.simplify()` chains.

# %%
model.z_momentum.apply(hydrostatic_scaling(state)).simplify()
model.z_momentum.describe()

# %% [markdown]
# ## Step 3 — Integrate z-momentum analytically to get $p(z)$
#
# Integrate $g + \partial_z p / \rho = 0$ from the current depth $z$ up to the
# free surface $\eta = b + h$ via analytical mode (whole-expression
# ``sympy.integrate``).

# %%
model.z_momentum.apply(
    Integrate(state.z, state.z, state.eta, method="analytical")
)
model.z_momentum.describe()

# %% [markdown]
# ## Step 4 — Atmospheric-pressure BC at the free surface
#
# Close $p(\eta) = 0$ (atmospheric gauge) by a plain substitution dict.

# %%
model.z_momentum.apply({state.p.subs(state.z, state.eta): 0}).simplify()
model.z_momentum.describe()

# %% [markdown]
# ## Step 5 — Substitute the solved $p$ into x-momentum, then drop z-momentum
#
# `model.z_momentum.solve_for(state.p)` returns an Expression that
# ``apply()`` consumes directly as ``{p: solution}``.

# %%
model.x_momentum.apply(model.z_momentum.solve_for(state.p)).simplify()
model.z_momentum.remove()
model.describe()

# %% [markdown]
# ## Step 6 — Newtonian constitutive model
#
# Kept as a convenience Relation.  Substitutes
# $\tau_{ij} = \mu(\partial_j u_i + \partial_i u_j)$ in every equation.

# %%
newton = Newtonian(state)
for name in list(model.equations.keys()):
    model.equations[name] = model.equations[name].apply(newton).simplify()
model.describe()

# %% [markdown]
# ## Step 7 — Depth-integrate continuity and x-momentum from $b$ to $b+h$
#
# One ``Integrate`` call with ``method="auto"`` per equation — per-term dispatch
# picks Leibniz for $\partial_x$ and the fundamental theorem for $\partial_z$.

# %%
for name in list(model.equations.keys()):
    model.equations[name] = model.equations[name].apply(
        Integrate(state.z, state.b, state.eta, method="auto")
    )
model.continuity.describe()

# %%
model.x_momentum.describe()

# %% [markdown]
# ## Step 8 — Resolve $w$ boundary terms via the kinematic BCs
#
# Two substitutions — the kinematic BCs at the bottom and at the surface —
# written as plain dicts.  The `Subs(...)` shapes that Step 7 left in the
# equation are the same expressions these dicts key on, so substitution is
# direct.

# %%
w_at_b = state.w.subs(state.z, state.b)
w_at_eta = state.w.subs(state.z, state.eta)
u_at_b = state.u.subs(state.z, state.b)
u_at_eta = state.u.subs(state.z, state.eta)

kinematic_bcs = {
    w_at_b:   sp.Derivative(state.b, state.t)   + u_at_b   * sp.Derivative(state.b, state.x),
    w_at_eta: sp.Derivative(state.eta, state.t) + u_at_eta * sp.Derivative(state.eta, state.x),
}

for name in list(model.equations.keys()):
    model.equations[name] = model.equations[name].apply(kinematic_bcs).simplify()

model.x_momentum.describe()

# %% [markdown]
# ## Step 9 — Zero tangential stress at surface and bottom
#
# Two plain dicts: stress-free surface ($\tau_{xz}|_\eta = 0$) and
# zero tangential normal stress at both boundaries ($\tau_{xx}|_b = \tau_{xx}|_\eta = 0$).
# Bottom shear stress ($\tau_{xz}|_b$) stays symbolic — close it with a
# Navier-slip or no-slip assumption as a further `.apply({...})` step.

# %%
no_surface_shear = {state.tau["xz"].subs(state.z, state.eta): 0}
no_tangential_normal_stress = {
    state.tau["xx"].subs(state.z, state.b): 0,
    state.tau["xx"].subs(state.z, state.eta): 0,
}

for name in list(model.equations.keys()):
    model.equations[name] = (
        model.equations[name]
        .apply(no_surface_shear)
        .apply(no_tangential_normal_stress)
        .simplify()
    )

model.x_momentum.describe()

# %% [markdown]
# ## Step 10 — Bottom stress closure (Navier slip)
#
# $\tau_{xz}|_b = \rho\,(\lambda/\tau_c)\,u|_b$ — the only non-trivial
# boundary term still present.  Plain substitution dict.
#
# After this step the depth-integrated PDE has only ``u(t,x,z)``, its
# boundary evaluations, and the remaining volume ``Integral(...)`` terms.
# Projection onto a vertical basis (separate step — level=0 for SWE,
# level>0 for SME) produces the final closed equations.

# %%
lamda = sp.Symbol("lamda", positive=True)
tau_c = sp.Symbol("tau_c", positive=True)
friction_closure = {
    state.tau["xz"].subs(state.z, state.b): state.rho * (lamda / tau_c) * u_at_b,
}

for name in list(model.equations.keys()):
    model.equations[name] = model.equations[name].apply(friction_closure).simplify()

model.x_momentum.describe()

# %% [markdown]
# ## What comes next
#
# * **SWE (level=0)** — project with a constant vertical profile: every
#   ``u(t,x,z) → u_mean(t,x)``, boundary evaluations become the same
#   constant, ``∫ u dz = h · u_mean``.  A few `.apply({...})` substitutions
#   complete the reduction.
# * **SME (level≥1)** — project with ``u(t,x,z) = sum_k α_k(t,x) φ_k(ζ)``
#   and Galerkin-test against each basis function.  Today this is
#   ``Expression.project_onto_basis(basis, level, field_map, var, test_mode=l)``;
#   it only rewrites ``Integral`` nodes, so a small companion substitution
#   for surface/bottom ``u`` evaluations is needed (covered in the next
#   file, once `project_onto_basis` is extended).
#
# The notebook above is the derivation spine — clean, reviewable, built
# entirely from `apply` / `Integrate` / substitution dicts / `solve_for` /
# `Newtonian`.  No convenience Relation beyond `Newtonian` and
# `hydrostatic_scaling` is used.
