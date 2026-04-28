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
# Moment Equations (SME, level>0) directly from incompressible Navier-Stokes.
#
# Everything is mutation-based:
#
# * ``model.apply(op)``  — mutates every equation of the system, returns ``self``.
# * ``model.<eq>.apply(op).simplify()`` — chainable in-place mutation on a
#   single equation through the proxy.
# * ``model.<eq>.solve_for(var)``  — returns an Expression that ``apply``
#   consumes directly as ``{var: solution}``.
# * ``model.<eq>.remove()`` — drops an equation from the system.
#
# No ``model.equations[name] = model.equations[name].apply(...)`` pattern.
# No ``DepthIntegrate`` / ``HydrostaticPressure`` / ``ApplyKinematicBCs`` /
# ``StressFreeSurface`` / ``ZeroAtmosphericPressure`` / ``SimplifyIntegrals``
# shortcuts — everything goes through ``Integrate`` and substitution dicts.

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
# $w = 0$, $\tau_{zz} = \tau_{xz} = \tau_{zx} = 0$ inside z-momentum only.
# Chained: ``apply(...)``→proxy, ``.simplify()``→proxy.

# %%
model.z_momentum.apply(hydrostatic_scaling(state)).simplify()
model.z_momentum.describe()

# %% [markdown]
# ## Step 3 — Integrate z-momentum analytically to get $p(z)$
#
# Integrate $g + \partial_z p / \rho = 0$ from the current depth $z$ up to the
# free surface $\eta$.  `method="analytical"` runs ``sympy.integrate`` on the
# whole expression (needed for partial / running integrals).

# %%
model.z_momentum.apply(
    Integrate(state.z, state.z, state.eta, method="analytical")
)
model.z_momentum.describe()

# %% [markdown]
# ## Step 4 — Atmospheric-pressure BC at the free surface
#
# $p(\eta) = 0$ (atmospheric gauge).  Plain substitution dict.

# %%
model.z_momentum.apply({state.p.subs(state.z, state.eta): 0}).simplify()
model.z_momentum.describe()

# %% [markdown]
# ## Step 5 — Substitute the solved $p$ into x-momentum, then drop z-momentum

# %%
model.x_momentum.apply(model.z_momentum.solve_for(state.p)).simplify()
model.z_momentum.remove()
model.describe()

# %% [markdown]
# ## Step 6 — Newtonian constitutive model
#
# System-level mutation: ``model.apply(op)`` runs the op on every equation
# of the system.
#
# ⚠️ ``.simplify()`` is deliberately **not** chained here: sympy's expand
# distributes chain/product rule across conservative derivatives
# (e.g. ``∂_x(u²) → 2u · ∂_x u``), which breaks the Leibniz-rule
# detection that ``Integrate`` uses in Step 7.  Simplification is fine
# **after** integration, once conservative derivatives have been
# consumed.

# %%
model.apply(Newtonian(state))
model.describe()

# %% [markdown]
# ## Step 7 — Depth-integrate continuity and x-momentum from $b$ to $\eta$
#
# One ``Integrate`` call at system level — per-term auto dispatch picks
# Leibniz for $\partial_x$ and the fundamental theorem for $\partial_z$.

# %%
model.apply(Integrate(state.z, state.b, state.eta, method="auto"))
model.continuity.describe()

# %%
model.x_momentum.describe()

# %% [markdown]
# ## Step 8 — Resolve $w$ boundary terms via the kinematic BCs
#
# Two substitutions for the $w$ evaluations at bottom and surface.  System-level
# ``apply(dict)`` applies the dict to every equation and chains into ``simplify``.

# %%
u_at_b = state.u.subs(state.z, state.b)
u_at_eta = state.u.subs(state.z, state.eta)
kinematic_bcs = {
    state.w.subs(state.z, state.b):
        sp.Derivative(state.b, state.t) + u_at_b * sp.Derivative(state.b, state.x),
    state.w.subs(state.z, state.eta):
        sp.Derivative(state.eta, state.t) + u_at_eta * sp.Derivative(state.eta, state.x),
}
model.apply(kinematic_bcs).simplify()
model.x_momentum.describe()

# %% [markdown]
# ## Step 9 — Zero tangential stress at surface and bottom
#
# Stress-free surface ($\tau_{xz}|_\eta = 0$) and zero tangential normal
# stress at both boundaries ($\tau_{xx}|_b = \tau_{xx}|_\eta = 0$).

# %%
stress_free_surface = {state.tau["xz"].subs(state.z, state.eta): 0}
no_tangential_normal_stress = {
    state.tau["xx"].subs(state.z, state.b): 0,
    state.tau["xx"].subs(state.z, state.eta): 0,
}
model.apply(stress_free_surface).apply(no_tangential_normal_stress).simplify()
model.x_momentum.describe()

# %% [markdown]
# ## Step 10 — Bottom stress closure (Navier slip)
#
# $\tau_{xz}|_b = \rho\,(\lambda/\tau_c)\,u|_b$.  Dict + chain.

# %%
lamda = sp.Symbol("lamda", positive=True)
tau_c = sp.Symbol("tau_c", positive=True)
friction_closure = {
    state.tau["xz"].subs(state.z, state.b): state.rho * (lamda / tau_c) * u_at_b,
}
model.apply(friction_closure).simplify()
model.x_momentum.describe()

# %% [markdown]
# ## What's left on the board
#
# At this point every equation is:
#
# * Depth-integrated over $[b,\eta]$.
# * Closed for $w$, the tangential normal stress, and bottom shear.
# * In terms of the velocity field $u(t,x,z)$, its surface/bottom evaluations
#   $u|_b$, $u|_\eta$, and volume integrals $\int_b^\eta f\,dz$.
#
# The remaining step is projection against a vertical basis:
#
# * **SWE (level=0)** — constant vertical profile.  Substitute
#   ``u(t,x,z) → u_mean(t,x)`` (both in the volume integrals and in the
#   surface/bottom evaluations) and evaluate the resulting integrals.
# * **SME (level≥1)** — expand ``u(t,x,z) = Σ α_k(t,x) φ_k(ζ)`` and Galerkin-
#   test against each ``φ_l``.  Today this is ``Expression.project_onto_basis``;
#   it rewrites ``Integral`` nodes and we complement it with explicit
#   ``{u|_b: Σ α_k φ_k(0), u|_η: Σ α_k φ_k(1)}`` substitutions for the
#   boundary evaluations.
#
# Both projections follow the same mutation pattern used above:
# ``model.apply(...)`` + chained ``.simplify()``.
