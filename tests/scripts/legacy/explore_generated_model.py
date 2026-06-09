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
# # Generated Shallow Model Explorer
#
# Interactive exploration of the automated PDE generator.
# Tweak `n_layers`, `level`, `dimension`, and `basis_type` to see
# how the generated equations change.

# %%
from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted, Monomials, Chebyshevu
import sympy as sp
sp.init_printing()

# %% [markdown]
# ## Case 1: Classical Shallow Water Equations
# `n_layers=1, level=0` — constant velocity profile, should recover SWE.

# %%
m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
b, h, mu, mv, hinv = m.get_primitives()
display(sp.Symbol("Q"), sp.Matrix(m.variables.values()))

# %%
display(sp.Symbol("F"), m.flux().tomatrix())

# %%
display(sp.Symbol("F_p"), m.hydrostatic_pressure().tomatrix())

# %%
sp.Symbol("eigenvalues"), m.eigenvalues().tomatrix()

# %% [markdown]
# ## Case 2: Shallow Moments (linear velocity profile)
# `level=1` adds a first-order moment — the velocity varies linearly over depth.

# %%
m1 = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
b, h, mu, mv, hinv = m1.get_primitives()
display(sp.Symbol("Q"), sp.Matrix(m1.variables.values()))
display(sp.Symbol("primitives"), sp.Matrix(mu[0]))

# %%
display(sp.Symbol("F"), m1.flux().tomatrix())

# %%
nc = m1.nonconservative_matrix()
nc_x = sp.Matrix([[nc[r, c, 0] for c in range(m1.n_variables)] for r in range(m1.n_variables)])
display(sp.Symbol("B_x"), nc_x)

# %%
display(sp.Symbol("eigenvalues"), m1.eigenvalues().tomatrix())

# %% [markdown]
# ## Case 3: Two-layer SWE
# `n_layers=2, level=0` — two layers, each with constant velocity.

# %%
m2 = GeneratedShallowModel(n_layers=2, level=0, dimension=1)
b, h, mu, mv, hinv = m2.get_primitives()
display(sp.Symbol("Q"), sp.Matrix(m2.variables.values()))
print(f"Layer 0 velocity: {mu[0]}")
print(f"Layer 1 velocity: {mu[1]}")

# %%
display(sp.Symbol("F"), m2.flux().tomatrix())

# %% [markdown]
# ## Case 4: 2D Shallow Moments
# Uncomment to run (eigenvalues are slow for larger systems).

# %%
# m2d = GeneratedShallowModel(n_layers=1, level=1, dimension=2)
# display(sp.Symbol("F"), m2d.flux().tomatrix())

# %% [markdown]
# ## Verification against ShallowMomentsTopo
# Check that the generated flux matches the hand-derived reference.

# %%
from zoomy_core.model.models.shallow_moments_topo import ShallowMomentsTopo

gen = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
ref = ShallowMomentsTopo(level=1, dimension=1)

F_gen = gen.flux()
F_ref = ref.flux()

subs = {}
for i in range(gen.n_variables):
    gv = gen.variables[i]
    rv = ref.variables[i]
    if gv != rv:
        subs[gv] = rv
for k in gen.parameters.keys():
    gp = gen.parameters[k]
    if ref.parameters.contains(k):
        rp = ref.parameters[k]
        if gp != rp:
            subs[gp] = rp

print("Flux difference (should be all zeros):")
for i in range(gen.n_variables):
    diff = sp.simplify(sp.expand(F_gen[i, 0].subs(subs) - F_ref[i, 0]))
    print(f"  Row {i}: {diff}")
