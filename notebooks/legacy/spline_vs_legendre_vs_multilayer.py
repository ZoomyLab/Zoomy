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
# # Model Comparison: Spline vs Legendre vs Multi-Layer
#
# Compares the symbolic structure of three approaches for representing
# vertical velocity profiles in shallow water models:
#
# 1. **Legendre L2**: global polynomial basis (1, 1-2z, 6z²-6z+1)
# 2. **Spline L2**: piecewise linear B-splines (1, hat at z=0.5, hat at z=1)
# 3. **Multi-layer (2-layer)**: piecewise constant per layer
#
# All produce 5-variable systems (b, h, + 3 velocity DOFs).

# %%
import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, Markdown
import zoomy_core.misc.misc as misc

from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted, SplineBasis
from zoomy_core.misc.misc import ZArray

sp.init_printing()
os.makedirs(os.path.join(misc.get_main_directory(), "outputs/model_comparison"), exist_ok=True)

# %% [markdown]
# ## 1. Build all three models

# %%
# Legendre L2 (3 moments: alpha_0, alpha_1, alpha_2)
m_leg = GeneratedShallowModel(n_layers=1, level=2, dimension=1,
                                basis_type=Legendre_shifted, eigenvalue_mode="numerical")

# Spline L2 (3 DOFs: constant + 2 hat functions)
m_spl = GeneratedShallowModel(n_layers=1, level=2, dimension=1,
                                basis_type=SplineBasis, eigenvalue_mode="numerical")

# Multi-layer 2-layer L0 (2 layer velocities)
m_ml = GeneratedShallowModel(n_layers=2, level=0, dimension=1,
                               eigenvalue_mode="numerical")

print(f"Legendre L2: {m_leg.n_variables} variables")
print(f"Spline L2:   {m_spl.n_variables} variables")
print(f"2-Layer L0:  {m_ml.n_variables} variables")

# %% [markdown]
# ## 2. Basis functions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

for ax, model, title in [
    (axes[0], m_leg, "Legendre L2"),
    (axes[1], m_spl, "Spline L2"),
    (axes[2], None, "2-Layer L0"),
]:
    if model is not None:
        basis = model.basisfunctions
        z_lo, z_hi = float(basis.bounds()[0]), float(basis.bounds()[1])
        zeta = np.linspace(z_lo, z_hi, 200)
        for k in range(model.level + 1):
            phi_fn = basis.get_lambda(k)
            y = np.array(phi_fn(list(zeta)))
            zeta_norm = (zeta - z_lo) / (z_hi - z_lo)
            ax.plot(y, zeta_norm, linewidth=1.5, label=f"φ_{k}")
    else:
        # Multi-layer: piecewise constant
        ax.fill_betweenx([0, 0.5], [0, 0], [1, 1], alpha=0.3, label="Layer 0 (φ₀=1)")
        ax.fill_betweenx([0.5, 1.0], [0, 0], [1, 1], alpha=0.3, label="Layer 1 (φ₀=1)")
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)

    ax.set_title(title); ax.set_xlabel("φ(ζ)"); ax.set_ylabel("ζ")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

OUT = os.path.join(misc.get_main_directory(), "outputs/model_comparison")
plt.savefig(os.path.join(OUT, "basis_functions.png"), dpi=150)
print("Saved: basis_functions.png")

# %% [markdown]
# ## 3. Basis matrix properties

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    bm = model.basismatrices
    n = model.level + 1
    display(Markdown(f"### {name}"))
    display(Markdown(f"**phib** (φ_k at bottom): {list(bm.phib[:n])}"))

    M_str = sp.Matrix([[bm.M[i, j] for j in range(n)] for i in range(n)])
    display(Markdown("**Mass matrix M:**"))
    display(M_str)

    A_diag = [bm.A[k, k, k] for k in range(n)]
    display(Markdown(f"**A[k,k,k]** (self-advection): {A_diag}"))

    D_diag = [bm.D[k, k] for k in range(n)]
    display(Markdown(f"**D[k,k]** (viscous self-damping): {D_diag}"))
    display(Markdown("---"))

# %% [markdown]
# ## 4. Flux comparison

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    F = model.flux()
    display(Markdown(f"### {name} — Flux F(Q)"))
    F_mat = sp.Matrix([[F[r, 0]] for r in range(model.n_variables)])
    display(F_mat)
    display(Markdown("---"))

# %% [markdown]
# ## 5. Hydrostatic pressure

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    P = model.hydrostatic_pressure()
    display(Markdown(f"### {name} — Pressure P(Q)"))
    P_mat = sp.Matrix([[P[r, 0]] for r in range(model.n_variables)])
    display(P_mat)
    display(Markdown("---"))

# %% [markdown]
# ## 6. Non-conservative matrix B_x

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    nc = model.nonconservative_matrix()
    n = model.n_variables
    B = sp.Matrix([[nc[r, c, 0] for c in range(n)] for r in range(n)])
    display(Markdown(f"### {name} — Non-conservative matrix B_x"))
    display(B)
    display(Markdown("---"))

# %% [markdown]
# ## 7. Source: slip friction

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    slip = model.slip()
    display(Markdown(f"### {name} — Slip friction"))
    for i in range(model.n_variables):
        if slip[i] != 0:
            display(Math(f"S_{{{i}}} = {sp.latex(slip[i])}"))
    if all(slip[i] == 0 for i in range(model.n_variables)):
        display(Markdown("*All zero — basis functions vanish at boundary (phib=0 for k≥1)*"))
    display(Markdown("---"))

# %% [markdown]
# ## 8. Source: Newtonian viscosity

# %%
for model, name in [(m_leg, "Legendre L2"), (m_spl, "Spline L2"), (m_ml, "2-Layer L0")]:
    display(Markdown(f"### {name} — Newtonian viscous source"))
    try:
        visc = model.newtonian()
        for i in range(model.n_variables):
            if visc[i] != 0:
                display(Math(f"S^{{visc}}_{{{i}}} = {sp.latex(visc[i])}"))
        if all(visc[i] == 0 for i in range(model.n_variables)):
            display(Markdown("*All zero — D matrix is zero (constant basis has zero derivatives)*"))
    except AttributeError:
        display(Markdown("*Model does not have 'nu' parameter — add it for viscous terms*"))
    display(Markdown("---"))

# %% [markdown]
# ## 9. Model summary comparison

# %%
print(f"{'Property':<30s} {'Legendre L2':>15s} {'Spline L2':>15s} {'2-Layer L0':>15s}")
print("-" * 78)

for prop, fn in [
    ("n_variables", lambda m: m.n_variables),
    ("n_parameters", lambda m: m.n_parameters),
    ("M diagonal?", lambda m: all(m.basismatrices.M[i, j] == 0
                                   for i in range(m.level + 1)
                                   for j in range(m.level + 1) if i != j)),
    ("phib all = 1?", lambda m: all(float(m.basismatrices.phib[k]) == 1.0
                                      for k in range(m.level + 1))),
    ("Viscous D nonzero?", lambda m: any(m.basismatrices.D[k, k] != 0
                                          for k in range(min(m.level + 1, 3)))),
    ("Slip affects k≥1?", lambda m: any(m.slip()[2 + k] != 0
                                         for k in range(1, m.level + 1))),
]:
    vals = []
    for model in [m_leg, m_spl, m_ml]:
        try:
            vals.append(str(fn(model)))
        except Exception:
            vals.append("error")
    print(f"{prop:<30s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s}")

# %% [markdown]
# ## 10. Observations
#
# | Property | Legendre L2 | Spline L2 | 2-Layer L0 |
# |----------|------------|-----------|------------|
# | Basis type | Global polynomials | Piecewise linear | Piecewise constant |
# | M diagonal | Yes | No | Yes |
# | phib (boundary values) | [1, 1, 1] | [1, 0, 0] | [1, 1] per layer |
# | Friction reaches k≥1 | Yes (all modes) | No (hat functions vanish at z=0) | Yes (per layer) |
# | Viscous damping k≥1 | Yes (D grows with k) | Yes (finite differences) | No (constant per layer) |
# | NC matrix | Present for k≥1 | Present, coupled | Present for inter-layer |
#
# **Key insight**: The spline basis has `phib = [1, 0, 0]` — the hat functions
# at interior knots are zero at the bottom boundary. This means:
# - Standard slip friction cannot excite higher spline modes
# - Need Nitsche penalty or modified friction to couple boundary physics to interior DOFs
# - Similar to the Galerkin basis: BCs are "built in" by the shape function support
#
# **Multi-layer comparison**: The 2-layer model is structurally similar to splines
# with one knot at z=0.5. Each layer has a constant velocity (level=0 per layer).
# But the multi-layer model has separate `phib` per layer (both = 1), so friction
# reaches both layers through the boundary coupling.
