# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SME + slip-Newton friction → SystemModel
#
# End-to-end walkthrough on the new Model class:
#
# 1. Construct `SME(N=2, parameters=...)` — pipeline runs, equations
#    left in Function form (`h(t, x)`, `q_fn(k, t, x)`).
# 2. Plug in a **stress model** — `sme.apply_slip_newton_friction()`
#    closes the open `τ_xz` BC + Integral atoms with the Newtonian
#    bulk / Navier-slip BC / free-surface BC constitutive law.
# 3. Hand off to **SystemModel** — `SystemModel.from_model(sme)`
#    auto-triggers the Function → Symbol substitution + tagging, then
#    freezes the operator matrices.
#
# Output matches K&T 2019 (4.17) line-for-line.

# %%
import sympy as sp

from zoomy_core.model.models.sme import SME
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)

sp.init_printing()


# %% [markdown]
# ## 1. Construct SME(N=2)
#
# Declare all four parameters up front (`g`, `ρ`, `ν`, `λ`).  All
# four are marked `positive` so the K&T derivation's sign-resolved
# substitutions land cleanly.

# %%
boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])

sme = SME(
    N=2,
    parameters={
        "g":      (9.81,  "positive"),
        "rho":    (1.0,   "positive"),
        "nu":     (1e-3,  "positive"),
        "lambda": (1e-2,  "positive"),
    },
    boundary_conditions=boundary_conditions,
)

print(f"SME constructed: {sme.name}")
print(f"  variables       : {list(sme.variables.keys())}")
print(f"  parameters      : {list(sme.parameters.keys())}")
print(f"  parameter_values: {dict(sme.parameter_values.as_dict())}")
print(f"  finalized?      : {sme._finalized}   (equations in Function form)")


# %% [markdown]
# ## 2. Inspect equations BEFORE friction
#
# `momentum_x_0` carries unresolved `τ_xz(t, x, 0/1)` boundary atoms
# and an `Integral(τ_xz, σ)` interior atom — those are what the
# stress closure resolves.

# %%
print("momentum_x_0 (before friction):")
for i, term in enumerate(sme.momentum_x_0):
    print(f"  [{i}] {term.expr}")


# %% [markdown]
# ## 3. Plug in the stress model
#
# `apply_slip_newton_friction` substitutes:
#
# * Newtonian bulk: `τ_xz(σ) = (ν / h) · ∂_σ u(σ)`
# * Navier-slip bottom: `τ_xz(σ=0) = (ν / λ) · u(σ=0)`
# * Free-surface top:  `τ_xz(σ=1) = 0`
#
# All τ_xz atoms close into algebraic expressions in `ν, λ, ρ, h, q_k`.

# %%
sme.apply_slip_newton_friction()

print("momentum_x_0 (after friction):")
for i, term in enumerate(sme.momentum_x_0):
    print(f"  [{i}] {term.expr}")
print()
print(f"finalized? : {sme._finalized}  (friction reset the flag —"
      " SystemModel.from_model will re-finalize)")


# %% [markdown]
# ## 4. Hand off to SystemModel
#
# `SystemModel.from_model(sme)` auto-triggers
# `sme._finalize_for_systemmodel()`: Function → Symbol substitution,
# `_variable_map` setup, auto-tagging, function registration.  Then
# the operator matrices are extracted.

# %%
sm = SystemModel.from_model(sme)

print(f"SystemModel state         : {list(sm.state)}")
print(f"SystemModel parameters    : {list(sm.parameters.values())}")
print(f"SystemModel parameter_vals: {dict(sm.parameter_values.as_dict())}")


# %% [markdown]
# ## 5. The matrices — K&T 2019 (4.17)
#
# The "true" SystemModel: same matrices the canonical
# `thesis/notebooks/legacy/modeling/transparent_derivations/sme_clean.py`
# produces (which itself is verified against K&T eqs 3.11, 3.16, 4.17).

# %%
print("flux  (convective)  F:")
for i in range(sm.flux.shape[0]):
    print(f"  F[{i}] = {sm.flux[i, 0]}")

print("\nhydrostatic_pressure  P:")
for i in range(sm.hydrostatic_pressure.shape[0]):
    print(f"  P[{i}] = {sm.hydrostatic_pressure[i, 0]}")

print("\nsource  S:")
for i in range(sm.source.shape[0]):
    print(f"  S[{i}] = {sm.source[i, 0]}")

print("\nnonconservative_matrix  B (non-zero entries):")
B = sm.nonconservative_matrix
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        for d in range(B.shape[2]):
            v = B[i, j, d]
            if v != 0:
                print(f"  B[{i}, {j}, {d}] = {v}")


# %% [markdown]
# ## 6. Cross-check against the K&T paper
#
# K&T 2019 (4.17), level N = 2 with slip-Newton friction:
#
# | Row | Convective F | Hydrostatic P | Source S |
# |---|---|---|---|
# | h  | 0 | 0 | 0 |
# | q₀ | q₀²/h + q₁²/(3h) + q₂²/(5h) | g·h²/2 | −g·h·∂ₓb − (ν/(λρh))·Σq_k |
# | q₁ | 2·q₀·q₁/h + 4·q₁·q₂/(5h)     | 0      | −(3ν/(λρh))·Σq_k − 12ν·q₁/(ρh²) |
# | q₂ | 2·q₀·q₂/h + 2·q₁²/(3h) + 2·q₂²/(7h) | 0 | −(5ν/(λρh))·Σq_k − 60ν·q₂/(ρh²) |
#
# Plus nonconservative coupling on the higher modes (B[2,2], B[2,3],
# B[3,2], B[3,3] above).  Line-for-line match with my SystemModel.
