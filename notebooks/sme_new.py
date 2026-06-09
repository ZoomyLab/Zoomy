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
# # SME — Shallow Moment Equations, new Model class
#
# Re-creation of the canonical `sme_clean.py` in the new `Model`
# layer.  The full SME(N=2) → SystemModel pipeline is now packaged
# inside the `SME` class — derivation, σ-transform, Galerkin
# projection, KBC closure, σ-integration, basis resolution, integral
# evaluation, gravity self-pair fold, higher-mode w closure, mass-
# matrix inversion, Function → Symbol substitution.  The user-facing
# code is just:
#
# ```python
# sme = SME(N=2, parameters={"g": 9.81, "rho": 1.0},
#           boundary_conditions=bcs)
# sm  = SystemModel.from_model(sme)
# ```
#
# This notebook walks the result end-to-end: the equation set, the
# operator-API matrices, the SystemModel hand-off.
#
# Output equation set matches K&T 2019 eq. (4.17).

# %%
import sympy as sp

from zoomy_core.model.models.sme import SME
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)

sp.init_printing()


# %% [markdown]
# ## Construction
#
# Single-line construction.  Parameters are sympy Symbols on
# `sme.parameters` (used by the equations) and numeric floats on
# `sme.parameter_values` (used by the printer / lambdify boundary).

# %%
boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])

sme = SME(
    N=2,
    parameters={"g": 9.81, "rho": 1.0},
    boundary_conditions=boundary_conditions,
)

print(f"SME class:      {type(sme).__name__}")
print(f"MRO:            {[c.__name__ for c in type(sme).__mro__[:5]]}")
print(f"variables:      {list(sme.variables.keys())}")
print(f"parameters:     {list(sme.parameters.keys())}")
print(f"parameter_vals: {dict(sme.parameter_values.as_dict())}")


# %% [markdown]
# ## The derivation
#
# After construction, `sme.derive_model()` has already run.  The
# derivation history is in `sme.history`; the final equations are
# accessible as **named attributes** (`sme.continuity_0`,
# `sme.momentum_x_0`, etc.).

# %%
print("Final equation set:")
print()
for eq in sme:
    print(f"  {eq.name}:")
    print(f"    {eq.expr}")
    print()


# %% [markdown]
# ## Derivation history
#
# Every `apply(Op(...))` call records a history entry.

# %%
print(f"Derivation history ({len(sme.history)} ops):")
for h in sme.history:
    if h.get("level") == "minor":
        continue
    desc = f" — {h['description']}" if h.get("description") else ""
    print(f"  [{h['op']}] target={h['target']}{desc}")


# %% [markdown]
# ## Operator-API matrices
#
# `sme.flux()`, `sme.source()`, `sme.nonconservative_matrix()`
# auto-extract from the tagged equation graph via
# `tag_extraction.collect_solver_tag`.  These are the same matrices
# `SystemModel.from_model` reads.

# %%
F = sme.flux()
S = sme.source()
B = sme.nonconservative_matrix()

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
#
# `SystemModel.from_model(sme)` freezes the operator API into
# matrices on a sibling object.

# %%
sm = SystemModel.from_model(sme)

print(f"SystemModel state         : {list(sm.state)}")
print(f"SystemModel parameters    : {list(sm.parameters.values())}")
print(f"SystemModel parameter_vals: {dict(sm.parameter_values.as_dict())}")

print(f"\nsm.flux                   shape: {sm.flux.shape}")
print(f"sm.source                 shape: {sm.source.shape}")
print(f"sm.nonconservative_matrix shape: {sm.nonconservative_matrix.shape}")


# %% [markdown]
# ## Sanity check — strict missing-parameter
#
# If the user omits a required parameter, MassMomentum's
# construction surfaces a clear `ValueError` *before* anything in
# the pipeline gets confused.

# %%
import pytest

try:
    SME(N=2, parameters={"rho": 1.0},   # g intentionally omitted
        boundary_conditions=boundary_conditions)
except ValueError as e:
    print(f"OK — caught ValueError: {e}")


# %% [markdown]
# ## Summary
#
# * Single-line construction: `SME(N=2, parameters={...}, boundary_conditions=...)`.
# * Named-attribute equations: `sme.continuity_0`, `sme.momentum_x_0`, etc.
# * Iteration: `for eq in sme: ...`.
# * Operator-API auto-extracted from tagged equation graph.
# * `SystemModel.from_model(sme)` works directly.
# * Parameters: Symbols on `sme.parameters`, floats on
#   `sme.parameter_values`.  No remap needed — parameters are passed
#   into `MassMomentum` at construction so the equations and the
#   Model share the same Symbol identity.
