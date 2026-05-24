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
# # VAM — Vertically-Averaged Moments, new Model class
#
# Non-hydrostatic counterpart of [SME](sme_new.py).  Unlike SME, the
# pressure is *not* eliminated via hydrostatic reduction: `w` and `p`
# stay as state variables and the z-momentum equation is
# Galerkin-projected alongside continuity and x-momentum.
#
# State layout (matrix-extraction surface): `[h, q_0..q_N, r_0..r_N,
# p_0..p_N]` with `q_k = h·u_k`, `r_k = h·w_k`.  Pressure modes
# `p_k` are kept symbolic (no closure imposed at this level — the
# Chorin / DAE solver closes them at run time).
#
# ```python
# vam = VAM(N=1, parameters={"g": 9.81, "rho": 1.0},
#           boundary_conditions=bcs)
# sm  = SystemModel.from_model(vam)
# ```

# %%
import sympy as sp

from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)

sp.init_printing()


# %% [markdown]
# ## Construction

# %%
boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])

vam = VAM(
    N=1,
    parameters={"g": 9.81, "rho": 1.0},
    boundary_conditions=boundary_conditions,
)

print(f"VAM class:      {type(vam).__name__}")
print(f"MRO:            {[c.__name__ for c in type(vam).__mro__[:5]]}")
print(f"variables:      {list(vam.variables.keys())}")
print(f"parameters:     {list(vam.parameters.keys())}")
print(f"parameter_vals: {dict(vam.parameter_values.as_dict())}")


# %% [markdown]
# ## Final equation set (named attributes)

# %%
for eq in vam:
    print(f"  {eq.name}:")
    print(f"    {eq.expr}")
    print()


# %% [markdown]
# ## Operator-API matrices

# %%
F = vam.flux()
S = vam.source()
B = vam.nonconservative_matrix()

print(f"flux                    shape: {F.shape}")
print(f"source                  shape: {S.shape}")
print(f"nonconservative_matrix  shape: {B.shape}")

print()
print("flux (non-zero rows):")
for i in range(F.shape[0]):
    if F[i, 0] != 0:
        print(f"  F[{i}, 0] = {F[i, 0]}")

print()
print("source (non-zero rows):")
for i in range(S.shape[0]):
    if S[i] != 0:
        print(f"  S[{i}] = {S[i]}")

print()
print("nonconservative_matrix (non-zero entries):")
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        for d in range(B.shape[2]):
            v = B[i, j, d]
            if v != 0:
                print(f"  B[{i}, {j}, {d}] = {v}")


# %% [markdown]
# ## SystemModel handoff

# %%
sm = SystemModel.from_model(vam)

print(f"SystemModel state         : {list(sm.state)}")
print(f"SystemModel parameters    : {list(sm.parameters.values())}")
print(f"SystemModel parameter_vals: {dict(sm.parameter_values.as_dict())}")

print(f"\nsm.flux  shape: {sm.flux.shape}")
print(f"sm.source shape: {sm.source.shape}")
print(f"sm.nonconservative_matrix shape: {sm.nonconservative_matrix.shape}")


# %% [markdown]
# ## Summary
#
# * Single-line construction: `VAM(N=1, parameters={...}, boundary_conditions=...)`.
# * Named-attribute equations: `vam.continuity_0`, `vam.momentum_x_0`,
#   `vam.momentum_z_0`, etc.
# * Pressure modes `p_0..p_N` remain in the state (auxiliary closures
#   imposed by the downstream solver — not at the symbolic level).
# * `SystemModel.from_model(vam)` works directly.
