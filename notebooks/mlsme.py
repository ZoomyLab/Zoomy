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
# # ML-SME (Audusse) → SystemModel
#
# **Multi-Layer Shallow Moment Equations**, derived in the
# *Audusse-Bristeau-Perthame-Sainte-Marie* sense: the water column
# `z ∈ [b, b+H]` is partitioned into `N_layers` *geometric* layers
# with fixed thickness fractions, and one coupled system is derived
# from the 3D Euler / INS equations.
#
# Key facts about this formulation:
#
# 1. **One global continuity** `∂_t H + ∂_x Q = 0` (NOT L per-layer
#    continuities — they all reduce to this under fixed-α).
# 2. **Layer thickness is not state**: `h_ℓ = α_ℓ·H` with `α_ℓ`
#    constant, `Σ α_ℓ = 1`.  Only `H` is a state variable.
# 3. **Interface mass flux** `G_{ℓ+1/2}` is determined by a
#    closed-form recursion — no extra closure needed.
# 4. **Upwinded interface velocity** `u*_{ℓ+1/2}` (Piecewise on the
#    sign of `G`) appears in the momentum-transfer terms; the SHARED
#    `u*` at each interface guarantees **global momentum
#    conservation**.
# 5. **N=0 limit** recovers the Audusse multi-layer SWE; under a
#    uniform velocity assumption (u_ℓ = U for all ℓ), the sum of
#    layer momenta reduces EXACTLY to single-layer SWE.

# %%
import sympy as sp

from zoomy_core.model.models.mlsme import MLSME
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)

sp.init_printing()


# %% [markdown]
# ## 1. Construct ML-SME(N_layers=2, N=2)
#
# 2 layers, K&T moment level N = 2.  Default uniform layer fractions
# `α_1 = α_2 = 1/2`.  State vector has size `1 + L·(N+1) = 1 + 2·3 = 7`.

# %%
boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])

mlsme = MLSME(
    N_layers=2,
    N=2,
    parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
    boundary_conditions=boundary_conditions,
)

print(f"Constructed: {mlsme.name}")
print(f"  N_layers        : {mlsme.N_layers}")
print(f"  N (moment level): {mlsme.N}")
print(f"  α layer fractions: {mlsme._alphas}")
print(f"  state size       : 1 + L·(N+1) = "
      f"{1 + mlsme.N_layers * (mlsme.N + 1)}")
print(f"  variables        : {list(mlsme.variables.keys())}")
print(f"  equation set     : {list(mlsme._equations.keys())}")
print(f"  finalized?       : {mlsme._finalized}  (Function form)")


# %% [markdown]
# ## 2. Inspect the global continuity
#
# This is the Audusse signature: ONE global mass conservation equation,
# `∂_t H + ∂_x Q = 0` with `Q = q_1_0 + q_2_0`.

# %%
print(f"continuity_global: {mlsme.continuity_global.expr}")


# %% [markdown]
# ## 3. Inspect a per-layer momentum equation (before SystemModel)
#
# Layer-1 momentum-0 already carries:
#
# * **Self convective flux**: `∂_x(q_1_0² / h_1)`
# * **Intra-layer hydrostatic**: `∂_x(g·h_1²/2)` (will split into P + B at tag-time)
# * **Inter-layer hydrostatic**: `g·h_1·∂_x h_other` (purely noncon)
# * **Bathymetry source**: `g·h_1·∂_x b`
# * **Open stress atoms**: `τ_xz(σ=0)`, `τ_xz(σ=1)` — interface shear stresses (close with a friction model later)
# * **Mass-exchange transfer**: `± u*_{ℓ±1/2}·G_{ℓ±1/2}` (Piecewise) — coupling to the OTHER layer's q's via the upwinded interface velocity

# %%
print("momentum_x_layer_1_0 (Function form):")
for i, term in enumerate(mlsme.momentum_x_layer_1_0):
    print(f"  [{i}] {term.expr}")


# %% [markdown]
# ## 4. Hand off to SystemModel
#
# `SystemModel.from_model(mlsme)` auto-finalizes (Function → Symbol
# substitution, variable_map setup, auto-tagging) then extracts the
# operator matrices.

# %%
sm = SystemModel.from_model(mlsme)

print(f"SystemModel state ({len(list(sm.state))}): {list(sm.state)}")
print(f"  flux shape:                {sm.flux.shape}")
print(f"  hydrostatic_pressure shape:{sm.hydrostatic_pressure.shape}")
print(f"  source shape:              {sm.source.shape}")
print(f"  nonconservative shape:     {sm.nonconservative_matrix.shape}")


# %% [markdown]
# ## 5. Reference — what the **Audusse paper** says
#
# Audusse-Bristeau-Perthame-Sainte-Marie 2011 (and Fernández-Nieto's
# subsequent extensions) write the multi-layer SWE system on the state
# `(H, q_1, q_2, …, q_L)` with `h_ℓ = α_ℓ·H` as:
#
# $$\partial_t H + \partial_x Q = 0, \qquad Q \equiv \sum_{m=1}^{L} q_m$$
#
# $$\boxed{\;
# \partial_t q_\ell
# \;+\; \partial_x\!\left(\frac{q_\ell^{\,2}}{h_\ell} + \tfrac12 g h_\ell^{\,2}\right)
# \;+\; g\,h_\ell\,\partial_x b
# \;+\; g\,h_\ell\;\partial_x\!\!\!\sum_{m\neq\ell}\!h_m
# \;=\;
# G_{\ell-1/2}\,u^{*}_{\ell-1/2} \;-\; G_{\ell+1/2}\,u^{*}_{\ell+1/2}
# \;}$$
#
# with interface mass flux (closed form under fixed-α)
#
# $$G_{\ell+1/2} = \Bigl(\sum_{m\le\ell}\alpha_m\Bigr)\partial_x Q \;-\; \sum_{m\le\ell}\partial_x q_m,$$
#
# bottom + free-surface impermeable: $G_{1/2}=G_{L+1/2}=0$, and the
# upwinded interface velocity
#
# $$u^{*}_{\ell+1/2}=\begin{cases}u_\ell(\sigma=1) & G_{\ell+1/2}>0\\[2pt] u_{\ell+1}(\sigma=0) & G_{\ell+1/2}\le 0\end{cases}$$
#
# For the **SME generalization to K moments** (this code), the velocity
# inside layer ℓ is the K-mode Legendre expansion
# `u_ℓ(σ) = Σ_k (q_ℓ_k / h_ℓ)·φ_k(σ)`, and the per-layer momentum
# equation is replaced by `K+1` Galerkin-projected moment equations.
# The depth-mean equation (`k = 0`, `φ_0 = 1`) reduces to the form
# above; higher moments add the K&T convective + nonconservative
# coupling within each layer.
#
# For **N=2, N_layers=2**, the expected matrices (with uniform `α=1/2`):
#
# | Slot   | Row | Expected form                                                   |
# |--------|-----|------------------------------------------------------------------|
# | F      | 0   | `q_1_0 + q_2_0`        (= global Q)                              |
# | F      | 1   | `q_1_0²/h_1 + q_1_1²/(3h_1) + q_1_2²/(5h_1) = 2q²/H+2q²/(3H)+…`  |
# | F      | 2   | `2·q_1_0·q_1_1/h_1 + 4·q_1_1·q_1_2/(5h_1)` (K&T higher moment)    |
# | F      | 3   | `2·q_1_0·q_1_2/h_1 + 2·q_1_1²/(3h_1) + 2·q_1_2²/(7h_1)`           |
# | F      | 4–6 | same as 1–3 but layer 2                                          |
# | P      | 1   | `½·g·h_1²` = `g·H²/8` (PART of hydrostatic — rest is in B)        |
# | P      | 4   | `½·g·h_2²` = `g·H²/8`                                            |
# | S      | 1   | `-g·h_1·∂_x b` = `-(g·H/2)·b_x`                                   |
# | S      | 2,3 | open friction Galerkin moments `∫σⁿ·∂_σ τ_xz dσ` (close later)    |
# | S      | 4   | `-g·h_2·∂_x b`                                                   |
# | B[1, 0]| –   | inter-layer hydrostatic `g·h_1·α_2 = g·H/4` × `∂_x H`             |
# | B[1, j]| –   | `±u*_{3/2}·α_1/2` × `∂_x q_{other}` (upwind transfer, Piecewise)  |
# | B[4, 0]| –   | inter-layer hydrostatic `g·h_2·α_1 = g·H/4` × `∂_x H`             |
# | B[2, k], B[3, k] | – | K&T intra-layer higher-mode noncon (q-q couplings)          |
#
# Note the **split** between P and B: the SystemModel auto-tagger
# distributes the conservative pressure `∂_x(g·h_ℓ²/2)` into `P`, and
# the noncon piece `g·h_ℓ·∂_x H_other` (from neighboring layers) into
# `B`.  The PHYSICAL force per layer is the SUM `∂_x P[ℓ] + B[ℓ, ⋅]·∂_x state`,
# which we'll verify against the SWE limit in §7.

# %% [markdown]
# ## 6. The matrices — actual SystemModel output

# %%
print("flux  F:")
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
        v = B[i, j, 0]
        if v != 0:
            print(f"  B[{i}, {j}, 0] = {v}")


# %% [markdown]
# ## 7. ML-SWE limit:  N = 0  →  Audusse multi-layer SWE
#
# Setting N = 0 (piecewise-constant velocity profile per layer)
# produces 2 layers × 1 momentum mode each, plus the global continuity
# — total state size `1 + 2·1 = 3`.
#
# **Audusse paper form for L=2, uniform α=1/2:**
#
# $$\partial_t H + \partial_x(q_1 + q_2) = 0$$
#
# $$\partial_t q_1 + \partial_x\!\left(\tfrac{2 q_1^{\,2}}{H} + \tfrac18 g H^2\right)
# + \tfrac14 g H \,\partial_x H
# + \tfrac12 g H \,\partial_x b
# \;=\; - G_{3/2}\,u^{*}_{3/2}$$
#
# $$\partial_t q_2 + \partial_x\!\left(\tfrac{2 q_2^{\,2}}{H} + \tfrac18 g H^2\right)
# + \tfrac14 g H \,\partial_x H
# + \tfrac12 g H \,\partial_x b
# \;=\; + G_{3/2}\,u^{*}_{3/2}$$
#
# where $G_{3/2} = \tfrac12\,\partial_x(q_2 - q_1)$ and
# $u^{*}_{3/2} = \mathrm{Piecewise}\bigl((u_1, G_{3/2}>0),(u_2,\text{else})\bigr)$.
#
# Note the **opposite-sign** transfer term in the two layers — that's
# exactly how Audusse guarantees global momentum conservation when you
# sum the two equations: the upwind transfer cancels, leaving you with
# the standard single-layer SWE on `Q = q_1 + q_2`.

# %%
mlswe = MLSME(
    N_layers=2,
    N=0,
    parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
    boundary_conditions=boundary_conditions,
)
sm0 = SystemModel.from_model(mlswe)

print(f"ML-SWE state: {list(sm0.state)}")
print(f"\nflux F (size {sm0.flux.shape[0]}):")
for i in range(sm0.flux.shape[0]):
    print(f"  F[{i}] = {sm0.flux[i, 0]}")
print(f"\nhydrostatic P:")
for i in range(sm0.hydrostatic_pressure.shape[0]):
    print(f"  P[{i}] = {sm0.hydrostatic_pressure[i, 0]}")
print(f"\nsource S:")
for i in range(sm0.source.shape[0]):
    print(f"  S[{i}] = {sm0.source[i, 0]}")
print(f"\nnonconservative B (non-zero):")
for i in range(sm0.nonconservative_matrix.shape[0]):
    for j in range(sm0.nonconservative_matrix.shape[1]):
        v = sm0.nonconservative_matrix[i, j, 0]
        if v != 0:
            print(f"  B[{i}, {j}, 0] = {v}")


# %% [markdown]
# ## 8. Verify the SWE reduction symbolically
#
# Under the **uniform-velocity assumption** `u_1 = u_2 = U`
# (equivalent to `q_ℓ = α_ℓ·H·U`, so `q_1 = q_2 = Q/2` with
# `Q = q_1 + q_2 = H·U`), the SUM of the two per-layer momentum
# equations should give the single-layer SWE momentum equation
# exactly:
#
# $$\partial_t Q + \partial_x \!\left(\frac{Q^2}{H} + \tfrac12 g H^2\right) + g H \,\partial_x b \;=\; 0$$
#
# The upwind transfer terms cancel between the two layers (they share
# `u*_{3/2}` with opposite signs — that's how Audusse achieves global
# momentum conservation).

# %%
H = mlswe.variables.H
q1, q2 = mlswe.variables.q_layer_1_0, mlswe.variables.q_layer_2_0
g = mlswe.parameters.g
b_x = sp.Symbol("b_x", real=True)
H_x = sp.Symbol("H_x", real=True)
q1_x = sp.Symbol("q_layer_1_0_x", real=True)
q2_x = sp.Symbol("q_layer_2_0_x", real=True)


def physical_rhs(i):
    """F·∂_x Q + P·∂_x Q + B·∂_x Q − S for matrix row i."""
    F = sm0.flux[i, 0]
    P = sm0.hydrostatic_pressure[i, 0]
    F_x = (sp.diff(F, H) * H_x
           + sp.diff(F, q1) * q1_x
           + sp.diff(F, q2) * q2_x)
    P_x = (sp.diff(P, H) * H_x
           + sp.diff(P, q1) * q1_x
           + sp.diff(P, q2) * q2_x)
    B_term = (sm0.nonconservative_matrix[i, 0, 0] * H_x
              + sm0.nonconservative_matrix[i, 1, 0] * q1_x
              + sm0.nonconservative_matrix[i, 2, 0] * q2_x)
    return F_x + P_x + B_term - sm0.source[i, 0]


# Sum layer 1 + layer 2 momentum RHS.
total_rhs = sp.simplify(physical_rhs(1) + physical_rhs(2))
print("Total physical RHS (layer 1 + layer 2 momentum):")
print(f"  {total_rhs}")

# Substitute uniform-velocity assumption q_1 = q_2 = Q/2.
Q_sym = sp.Symbol("Q", positive=True)
Q_x = sp.Symbol("Q_x", real=True)
uniform = {q1: Q_sym / 2, q2: Q_sym / 2,
           q1_x: Q_x / 2, q2_x: Q_x / 2}
total_uniform = sp.simplify(total_rhs.xreplace(uniform))
print(f"\nUnder u_1 = u_2 = U (uniform velocity):")
print(f"  {total_uniform}")

swe_expected = sp.simplify(
    -Q_sym**2 * H_x / H**2 + 2 * Q_sym * Q_x / H
    + g * H * H_x + g * H * b_x
)
print(f"\nExpected SWE: ∂_x(Q²/H + g·H²/2) + g·H·∂_x b:")
print(f"  {swe_expected}")

residual = sp.simplify(total_uniform - swe_expected)
print(f"\nResidual (should be 0):  {residual}")
assert residual == 0
print("\n✔  ML-SWE reduction verified exactly.")


# %% [markdown]
# ## 9. Custom layer fractions
#
# Use the `alphas=` constructor kwarg to set non-uniform layer
# distributions.  Below: a thin top layer (`α_2 = 0.1`) above a thick
# bottom layer (`α_1 = 0.9`), useful for stratified flow studies.

# %%
mlsme_strat = MLSME(
    N_layers=2,
    N=0,
    alphas=[sp.Rational(9, 10), sp.Rational(1, 10)],
    parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
    boundary_conditions=boundary_conditions,
)
sm_strat = SystemModel.from_model(mlsme_strat)

print("Stratified (α_1 = 9/10, α_2 = 1/10):")
print(f"  F[1] (layer 1 mom)        : {sm_strat.flux[1, 0]}")
print(f"  F[2] (layer 2 mom — thin) : {sm_strat.flux[2, 0]}")
print(f"  P[1]                      : {sm_strat.hydrostatic_pressure[1, 0]}")
print(f"  P[2]                      : {sm_strat.hydrostatic_pressure[2, 0]}")
print(f"  S[1]                      : {sm_strat.source[1, 0]}")
print(f"  S[2]                      : {sm_strat.source[2, 0]}")


# %% [markdown]
# ## 10. The interface-flux machinery (debug view)
#
# These are stashed on the MLSME instance for inspection.  `G_interfaces`
# holds `[G_{1/2}, G_{3/2}, ..., G_{L+1/2}]` (size `L+1`, with the first
# and last entries always zero).  `u_interfaces` holds the corresponding
# upwinded interface velocities.

# %%
print("Interface mass fluxes G_{ℓ+1/2} (size L+1):")
for i, G in enumerate(mlswe._G_interfaces):
    print(f"  G_{{i+1/2 = {2*i + 1}/2}}: {G}")

print("\nInterface velocities u*_{ℓ+1/2}:")
for i, u in enumerate(mlswe._u_interfaces):
    print(f"  u*_{{{2*i + 1}/2}}: {u}")
