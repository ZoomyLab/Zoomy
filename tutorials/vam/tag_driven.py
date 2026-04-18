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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VAM end-to-end: symbolic derivation → solver tags → numerical operators
#
# Walks the same tag-driven pipeline as the SME notebook, but for the
# non-hydrostatic **Viscous Alignment Model** (`VAMModelTagged`).  The key
# differences from SME:
#
# * z-momentum is kept (non-hydrostatic), so the state vector includes
#   `hw_k` w-moments alongside the `hu_k` horizontal moments.
# * Each `z_momentum_k` equation is also decomposed into solver tags.
# * The pressure splitting path (`source_implicit`) is NOT routed through
#   solver tags — it's inherited verbatim from `VAMModel` and used by the
#   IMEX pressure-correction step, which isn't shown in this notebook.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import sympy as sp

from zoomy_core.model.models.vam_model import VAMModel, VAMModelTagged
from zoomy_core.kernel import Kernel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

# %% [markdown]
# ## Build the model

# %%
LEVEL = 1
model = VAMModelTagged(level=LEVEL)
print("n_variables:         ", model.n_variables)
print("n_aux_variables:     ", model.n_aux_variables)
print("n_u (u moments):     ", model._n_u)
print("n_w (w moments):     ", model._n_w)
print("variables:           ", model.variables)
print("aux variables:       ", model.aux_variables)

# %% [markdown]
# ## Inspect each equation's solver tags
#
# `_operator_system.equations` now includes:
# * `continuity` — scalar (h)
# * `x_momentum_k` — one per u-moment
# * `z_momentum_k` — one per w-moment
#
# Each equation's solver_tags sum to its full LHS.

# %%
for name, eq in model._operator_system.equations.items():
    print(f"\n=== {name} ===")
    for tag, piece in eq.solver_tags.items():
        # Use a length-limited repr so the notebook stays readable for large L.
        s = str(piece)
        if len(s) > 100:
            s = s[:97] + "..."
        print(f"  {tag:22s} = {s}")
    remainder = eq.untagged_remainder()
    print(f"  untagged remainder = {remainder}")

# %% [markdown]
# ## Numerical operators

# %%
F = model.flux()
S = model.source()
NC = model.nonconservative_matrix()
Hp = model.hydrostatic_pressure()
print("flux shape:                   ", F.shape)
print("source shape:                 ", np.array(S).shape)
print("hydrostatic_pressure shape:   ", Hp.shape)
print("nonconservative_matrix shape: ", NC.shape)

# %% [markdown]
# ## Parity with hand-coded VAMModel

# %%
ref = VAMModel(level=LEVEL)

def _matrix_delta(a, b):
    return sp.simplify(sp.Matrix(a) - sp.Matrix(b))

print("flux    diff:", _matrix_delta(ref.flux(),                 F))
print("source  diff:", _matrix_delta(list(ref.source()),         list(S)))
print("hydro   diff:", _matrix_delta(ref.hydrostatic_pressure(), Hp))

# NC rank-3 elementwise
max_diff = 0
for i in range(model.n_variables):
    for j in range(model.n_variables):
        for k in range(model.dimension):
            d = sp.simplify(ref.nonconservative_matrix()[i, j, k] - NC[i, j, k])
            if d != 0:
                max_diff = d
print("NC diff (first nonzero, or 0):", max_diff)

# %% [markdown]
# ## Compile + single-cell runtime evaluation
#
# VAM's production time-stepping uses `IMEXSourceSolver` to handle the
# implicit pressure-correction step (outside the scope of this notebook).
# We still demonstrate that the tag-driven flux and source compile and
# agree bit-for-bit with the reference on a sample state.

# %%
rt_ref = NumpyRuntimeModel(ref,    kernel=Kernel(ref))
rt_tag = NumpyRuntimeModel(model,  kernel=Kernel(model))

Q = np.zeros(model.n_variables)
Q[1] = 0.5      # h
Q[2] = 0.1      # hu_0
Qaux = np.zeros(model.n_aux_variables) if model.n_aux_variables else np.array([])
p = np.asarray(model.parameters.get_list(), dtype=float)

F_num_ref = np.asarray(rt_ref.flux(Q, Qaux, p), dtype=float)
F_num_tag = np.asarray(rt_tag.flux(Q, Qaux, p), dtype=float)
print("max |flux_ref - flux_tag|   at sample state:", np.max(np.abs(F_num_ref - F_num_tag)))

S_num_ref = np.asarray(rt_ref.source(Q, Qaux, p), dtype=float)
S_num_tag = np.asarray(rt_tag.source(Q, Qaux, p), dtype=float)
print("max |source_ref - source_tag| at sample state:", np.max(np.abs(S_num_ref - S_num_tag)))

# %% [markdown]
# ## Summary
#
# `VAMModelTagged` produces bit-identical flux, source, NC, and
# hydrostatic pressure to the hand-coded `VAMModel`.  The `source_implicit`
# pressure-splitting path is inherited untouched.
#
# Routing `source_implicit` through solver tags is a follow-up
# (`VAMNewtonianTagged` is also deferred — see the class docstring).
