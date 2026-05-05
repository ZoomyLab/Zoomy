# ---
# title: "VAM end-to-end: Model → SystemModel → analysis → 2D simulation"
# author: Ingo Steldermann
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# # VAM end-to-end: full pipeline via the class hierarchy
#
# Demonstrates the canonical flow from a `Model` instance through to a
# 2D numerical simulation, using each layer for its job:
#
# 1. **`Model`** — the derivation framework.  Carries the equation
#    tree, history, and the operator-API methods that walk it.
# 2. **`SystemModel`** — the operator-form sibling.  Built from a
#    `Model` via `SystemModel.from_model(m)`; freezes the symbolic
#    flux / NCP / source / mass / hydrostatic-pressure operators.
#    Analysis and transformation consume it.
# 3. **System-level operations** — `apply(InvertMassMatrix())` etc.
#    Mutate the SystemModel's stored matrices.
# 4. **Analysis** — quasilinear matrix evaluated at a base state,
#    eigenvalues directly via sympy.
# 5. **Numerics** — `FSFSplittingSolver` accepts the model class and
#    runs a 2D dam-break.

# +
import numpy as np
import sympy as sp

from zoomy_core.model.models.vam_model import VAMModel
from zoomy_core.model.models.system_model import SystemModel, InvertMassMatrix
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.mesh import BaseMesh
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
# -


# ## 1. Model instantiation
#
# `VAMModel` derives from `INSModel(DerivedModel)` via the class
# hierarchy in `vam_model.py`.  The 1D variant (`ins_dimension=2`) is
# what we'll analyse.  The 2D variant (`ins_dimension=3`) is what
# step 5 evolves.

# +
class VAM1D(VAMModel):
    ins_dimension = 2

class VAM2D(VAMModel):
    ins_dimension = 3

m1d = VAM1D(level=0)
m2d = VAM2D(level=0)
m1d.describe()
# -


# ## 2. From `Model` to `SystemModel`
#
# `SystemModel.from_model(m)` reads the operator API of the model
# once and freezes the result as stored sympy matrices.  The two
# classes have the same operator-API surface — `flux`,
# `nonconservative_matrix`, `source`, `mass_matrix`,
# `hydrostatic_pressure`, `quasilinear_matrix` — but Model walks its
# equation tree on every call while SystemModel returns the cached
# matrices.  Decoupling derivation from the runtime/operator surface.

sm = SystemModel.from_model(m1d)
sm.describe(full=True)


# ## 3. Apply a system-level operation
#
# `sm.apply(InvertMassMatrix())` brings the system to canonical
# `∂_t Q + ∂_x F + B·∂_x Q − S = 0` form.  For VAM(level=0) the mass
# matrix is already identity (the existing operator-API path returns
# canonical form), so this is a no-op — but the call is still useful
# because the resulting SystemModel carries the ``invert_mass_matrix``
# entry in its history, which lets downstream solvers verify the
# canonical-form invariant.

sm.apply(InvertMassMatrix())
sm.describe(full=False)


# ## 4. Dispersion analysis via `SystemModel`
#
# Substitute the symbolic state with a quiescent base state
# `(b₀, h₀, hu₀=0, hw₀=0)`, fix `ez=1` (vertical-gravity gauge), and
# read eigenvalues of the quasilinear matrix in the x-direction.  No
# `PDESystem`, no `linearise`, no hand-rolled symbols — the model's
# own state Symbols flow through to the substitution.

# +
h0 = sp.Symbol("h0", positive=True)
base_state = {
    m1d.variables.b:   sp.Integer(0),
    m1d.variables.h:   h0,
    m1d.variables.hu0: sp.Integer(0),
    m1d.variables.hw0: sp.Integer(0),
}
ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

qm = sm.quasilinear_matrix()
M_x = sp.Matrix(
    sm.n_equations,
    sm.n_equations,
    lambda i, j: sp.simplify(qm[i, j, 0].subs(base_state).subs(ez_param, 1)),
)
eigenvalues = M_x.eigenvals()
M_x, eigenvalues
# -


# ## 5. 2D dam-break with `FSFSplittingSolver`
#
# `FSFSplittingSolver` is the free-surface specialisation of
# `SplittingSolver`: explicit hyperbolic predictor (positive Rusanov
# with hydrostatic reconstruction; b at index 0, h at index 1) plus
# explicit viscous diffusion plus a pressure Poisson correction
# (Chorin projection).  It accepts the model class directly —
# internally it compiles via `NumpyRuntimeModel` to runtime kernels.
# Every substep is ghost-cell-free: BCs are evaluated inline at
# boundary faces by calling the BC's `face_value(...)` method
# directly, no ghost-cell fill.

# +
def dam_break_ic(x, _nv=m2d.n_variables):
    Q = np.zeros(_nv)
    Q[1] = 2.0 if (x[0] < 5.0 and x[1] < 5.0) else 1.0
    return Q

m2d.initial_conditions = IC.UserFunction(function=dam_break_ic)
m2d.boundary_conditions = BC.BoundaryConditions([
    BC.Extrapolation(tag=tag) for tag in ("left", "right", "bottom", "top")
])

mesh = BaseMesh.create_2d((0.0, 10.0, 0.0, 10.0), nx=20, ny=20)
solver = FSFSplittingSolver(time_end=0.05,
                            compute_dt=ts.adaptive(CFL=0.3),
                            viscosity=0.01)
Q, p = solver.solve(mesh, m2d, write_output=False)
nc = mesh.n_inner_cells
diagnostics = {
    "inner cells":      nc,
    "all-finite Q":     bool(np.isfinite(Q[:, :nc]).all()),
    "all-finite p":     bool(np.isfinite(p[:nc]).all()),
    "h_min":            float(Q[1, :nc].min()),
    "h_max":            float(Q[1, :nc].max()),
    "|p|_max":          float(np.abs(p[:nc]).max()),
    "mass":             float(np.sum(Q[1, :nc]) * 100.0 / nc),
    "mass expected":    2.0 * 25.0 + 1.0 * 75.0,
}
diagnostics
# -


# ## Summary
#
# The full chain runs end-to-end:
#
# * `VAMModel(level=0)` → derivation, equation tree, operator-API
#   methods.
# * `SystemModel.from_model(m)` → frozen operator matrices, suitable
#   for analysis and transformation.
# * `sm.apply(InvertMassMatrix())` → canonical-form invariant
#   recorded in history.
# * `sm.quasilinear_matrix().subs(base_state)` → constant matrix,
#   eigenvalues by direct sympy.
# * `FSFSplittingSolver.solve(mesh, m2d)` → 2D dam-break with
#   pressure Poisson correction.
#
# What's not yet rolled out (separate workstreams):
#
# * `VAMModel.derive_model` rebuild around the new chain
#   (`Multiply` / `AffineProjection` / `Expand` /
#   `EvaluateIntegrals`) + author-side term tagging.  The current
#   model uses the existing manual-basis-matrix path, which produces
#   correct operators but doesn't walk a tagged equation tree.
# * `zoomy_core.analysis` package rewrite around `SystemModel`
#   directly (currently still has the `PDESystem`-based linearise /
#   pencil / dispersion routines).
# * Transformation (`NumpyRuntimeModel` etc.) accepting `SystemModel`
#   as input rather than `Model`.
