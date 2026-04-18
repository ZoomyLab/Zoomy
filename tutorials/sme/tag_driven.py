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
# # SME end-to-end: symbolic derivation → solver tags → numerical solve
#
# This notebook walks the full pipeline for the **tag-driven** Shallow Moment
# Equations (`SMEModelTagged`):
#
# 1. Build the model and inspect its symbolic derivation.
# 2. Inspect the `solver_tag` decomposition of each projected equation.
# 3. Read off the numerical operators (`flux`, `source`,
#    `nonconservative_matrix`, `hydrostatic_pressure`) — all extracted from
#    the tagged equations via `collect_solver_tag`.
# 4. Compile to numpy and run a short simulation with `HyperbolicSolver`.
# 5. Compare with the hand-coded `SMEModel` reference — parity should be
#    bit-identical.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from zoomy_core.model.models.sme_model import SMEModel, SMEModelTagged
from zoomy_core.mesh import BaseMesh, ensure_lsq_mesh
from zoomy_core.kernel import Kernel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC

# %% [markdown]
# ## Build the model and inspect its derivation
#
# `SMEModelTagged` inherits the full `SMEModel` derivation chain
# (FullINS → hydrostatic → depth-integrate → kinematic BCs →
# stress-free surface → zero atmospheric pressure → Newtonian material).

# %%
LEVEL = 1
model = SMEModelTagged(level=LEVEL)
print("n_variables:", model.n_variables)
print("variables:  ", model.variables)

# %% [markdown]
# The symbolic derivation is recoverable via `describe()`:

# %%
print(model.describe(derivation="markdown"))

# %% [markdown]
# ## Inspect the solver tags on each post-projection scalar equation
#
# `_operator_system.equations` holds the scalar equations
# (one per output row). Each equation carries five canonical solver tags
# whose sum equals the full LHS (the `untagged_remainder` invariant).

# %%
for name, eq in model._operator_system.equations.items():
    print(f"\n=== {name} ===")
    print("  expr (LHS = 0):", eq.expr)
    for tag, piece in eq.solver_tags.items():
        print(f"  solver_tag[{tag:22s}] = {piece}")
    print("  untagged remainder:", eq.untagged_remainder())

# %% [markdown]
# ## Numerical operators (read off the tags)
#
# Each body below is a one-liner over `collect_solver_tag`:
# ```python
# def flux(self):              return ZArray(self._collect("flux",     shape_dirs=1))
# def source(self):            return -ZArray(self._collect("source",  shape_dirs=0))
# def nonconservative_matrix(self):  return ZArray(self._collect("nonconservative_flux", shape_dirs=1))
# def hydrostatic_pressure(self):    return ZArray(self._collect("hydrostatic_pressure", shape_dirs=1))
# ```

# %%
F  = model.flux()
S  = model.source()
NC = model.nonconservative_matrix()
Hp = model.hydrostatic_pressure()
print("flux                 :", sp.Matrix(F).T)
print("source               :", sp.Matrix(list(S)).T)
print("hydrostatic_pressure :", sp.Matrix(Hp).T)
print("nonconservative_matrix (rank-3) shape:", NC.shape)

# %% [markdown]
# ## Parity with hand-coded SMEModel (the reference)

# %%
ref = SMEModel(level=LEVEL)

def _matrix_delta(a, b):
    return sp.simplify(sp.Matrix(a) - sp.Matrix(b))

print("flux     diff:", _matrix_delta(ref.flux(),                   F))
print("source   diff:", _matrix_delta(list(ref.source()),           list(S)))
print("hydro    diff:", _matrix_delta(ref.hydrostatic_pressure(),   Hp))
# NC rank-3: check elementwise
nc_diffs = [sp.simplify(ref.nonconservative_matrix()[i, j, k] - NC[i, j, k])
            for i in range(model.n_variables)
            for j in range(model.n_variables)
            for k in range(model.dimension)]
print("NC max diff:", max((str(d) for d in nc_diffs if d != 0), default="0"))

# %% [markdown]
# ## Compile and run
#
# Short 1D test: Gaussian bump in water height, periodic-length domain, 40
# cells, 0.05 s.  Both reference and tagged models are compiled and solved
# — the states at t = 0.05 s should agree to machine precision.

# %%
def _run(cls):
    np.random.seed(1)
    m = cls(level=LEVEL)
    m.boundary_conditions = BC.BoundaryConditions(
        boundary_conditions_list=[BC.Extrapolation(tag=t) for t in ("left", "right")])
    m.initial_conditions = IC.UserFunction(
        function=lambda X: np.array(
            [0.0, 0.5 + 0.1 * np.exp(-((X[0] - 5) ** 2) / 0.5)] + [0.0] * (LEVEL + 1)
        )
    )
    mesh = BaseMesh.create_1d((0, 10), 40)
    Q, _ = HyperbolicSolver(
        time_end=0.05, compute_dt=ts.adaptive(CFL=0.3)
    ).solve(mesh, m, write_output=False)
    return mesh, Q

mesh_ref, Q_ref = _run(SMEModel)
mesh_tag, Q_tag = _run(SMEModelTagged)
print(f"max |ref - tag| over all fields/cells: "
      f"{np.max(np.abs(Q_ref[:, :40] - Q_tag[:, :40])):.2e}")

# %% [markdown]
# ## Plot h and hu_0 side-by-side

# %%
xc = ensure_lsq_mesh(mesh_ref).cell_centers[0, :40]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(xc, Q_ref[1, :40], label="h (reference)")
axes[0].plot(xc, Q_tag[1, :40], "--", label="h (tagged)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("h"); axes[0].legend()
axes[1].plot(xc, Q_ref[2, :40], label="hu0 (reference)")
axes[1].plot(xc, Q_tag[2, :40], "--", label="hu0 (tagged)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("hu0"); axes[1].legend()
fig.suptitle(f"SMEModelTagged(level={LEVEL}) vs SMEModel at t = 0.05 s")
fig.tight_layout()
plt.show()
