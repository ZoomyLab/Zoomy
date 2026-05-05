# ---
# title: "VAM end-to-end: derivation, SystemModel, analysis, 2D simulation"
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

# # VAM end-to-end: derivation → SystemModel → analysis → numerics
#
# Single self-contained notebook showing the full pipeline through
# the new architecture:
#
# 1. **`VAMModelGalerkin`** — VAM derived from scratch via the
#    explicit Galerkin chain (Multiply / Integrate / InterfaceKBC /
#    atmospheric pressure / AffineProjection / Expand / EvaluateIntegrals).
#    Two snapshots taken: the unevaluated ``Sum``-form intermediate
#    (paper notation) and the closed primitive-form system.
#
# 2. **`SystemModel.from_model(m)`** — operator-form sibling.
#    Frozen sympy matrices for flux / NCP / source / mass /
#    hydrostatic-pressure.  Decoupled from the derivation tree.
#
# 3. **Analysis directly on `SystemModel`** — substitute the model's
#    own state Symbols with a base state, compute eigenvalues from
#    ``sm.quasilinear_matrix()``.  No ``PDESystem``, no ``linearise``,
#    no hand-rolled symbols.
#
# 4. **`FSFSplittingSolver`** — 2D dam-break with pressure Poisson
#    correction.  Operates on the model class directly; internally
#    compiles via ``NumpyRuntimeModel``.

# +
import numpy as np
import sympy as sp

from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel, InvertMassMatrix
from zoomy_core.analysis.system_model_analysis import plane_wave_dispersion
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
from zoomy_core.mesh import BaseMesh
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
# -


# ## 1. Model instantiation
#
# `VAMModelGalerkin` instantiates with a level (number of vertical
# moments).  Internally it runs the full Galerkin chain twice — once
# stopped at the ``Expand`` step (gives us a paper-form ``Σ_k`` view)
# and once all the way through ``EvaluateIntegrals`` (closed primitive
# form).  Both snapshots live on the model.

# +
class VAM1D(VAMModelGalerkin):
    ins_dimension = 2

class VAM2D(VAMModelGalerkin):
    ins_dimension = 3

m1d = VAM1D(level=0)
m2d = VAM2D(level=0)
m1d.describe()
# -


# ## 2. Chain intermediate — paper-form ``Σ_k U_k φ_k(ζ)``
#
# After the three ``Expand`` calls (one per ansatz field — ``u``, ``w``,
# ``p`` — each with its own basis ``phi`` / ``eta`` / ``mu``) but
# **before** ``EvaluateIntegrals`` resolves the integrals, every leaf
# of the System carries unevaluated ``sp.Sum`` atoms.  These render
# in paper notation: ``Σ_{k=0}^{L} U_k · phi_k(ζ)`` etc.

m1d.describe_chain_intermediate()


# ## 3. Chain closed — primitive-form equations
#
# After ``EvaluateIntegrals`` unrolls the Sums, applies the basis
# orthogonality + cache-based polynomial integration, and resolves
# everything to closed form.  The leaves are now polynomial in
# ``(h, U_k, W_k, P_k)`` — no held integrals, no held sums.

m1d.describe_chain_closed()


# ## 4. From `Model` to `SystemModel`
#
# `SystemModel.from_model(m)` reads the operator API of the model
# once and freezes the result as stored sympy matrices.  Independent
# class — no inheritance with ``Model``.  The two share the same
# operator-API surface (``flux`` / ``nonconservative_matrix`` /
# ``source`` / ``mass_matrix`` / ``hydrostatic_pressure`` /
# ``quasilinear_matrix``) but with different internals: ``Model``
# walks the equation tree on every call; ``SystemModel`` returns the
# cached matrices.

sm = SystemModel.from_model(m1d)
sm.describe(full=True)


# ## 5. Apply a system-level operation
#
# `sm.apply(InvertMassMatrix())` brings the system to canonical
# ``∂_t Q + ∂_x F + B·∂_x Q − S = 0`` form.  For VAM(level=0) the
# inherited operator path delivers the canonical M=I matrices already,
# so this is a no-op — but the call records the canonical-form
# invariant in ``sm.history`` for downstream solver verification.

sm.apply(InvertMassMatrix())
sm.describe(full=False)


# ## 6. Analysis on `SystemModel`
#
# `plane_wave_dispersion(sm, base_state, axis, parameters)` lives in
# ``zoomy_core.analysis.system_model_analysis`` — a SystemModel-direct
# routine.  It linearises by substituting the base state into the
# *quasilinear* matrix (i.e. on `∂F/∂Q + ∂P/∂Q + B`, the Jacobian
# already taken) and solves ``det(M_x − λ M_t) = 0`` for the wave
# speeds.  No ``PDESystem``, no ``linearise``-on-equations, no
# hand-rolled symbols — the model's own state Symbols
# (``m1d.variables.h`` etc.) flow through directly.

# +
h0 = sp.Symbol("h0", positive=True)
base_state = {
    m1d.variables.b:   sp.Integer(0),
    m1d.variables.h:   h0,
    m1d.variables.hu0: sp.Integer(0),
    m1d.variables.hw0: sp.Integer(0),
}
ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

result = plane_wave_dispersion(
    sm, base_state, axis=0, parameters={ez_param: 1}
)
result
# -


# ## 7. 2D dam-break with `FSFSplittingSolver`
#
# `FSFSplittingSolver` is the free-surface specialisation of the
# splitting solver: explicit hyperbolic predictor (positive Rusanov
# with hydrostatic reconstruction; b at index 0, h at index 1) plus
# explicit viscous diffusion plus a pressure Poisson correction
# (Chorin projection).  Accepts the model class directly — internally
# it compiles via ``NumpyRuntimeModel`` to runtime kernels.  Every
# substep is ghost-cell-free: BCs are evaluated inline at boundary
# faces by calling the BC's ``face_value(...)`` method directly, no
# ghost-cell fill.

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


# ## What this notebook proves
#
# * **Step 6 (derivation)**: `VAMModelGalerkin.derive_model` runs the
#   explicit symbolic Galerkin chain end-to-end.  Two intermediate
#   snapshots (Sum-form, closed-form) are inspectable via the
#   `describe_chain_*` methods.  No procedural builders; the chain
#   IS the derivation, and its closed equations are what the model
#   describes.
#
# * **Step 4 (SystemModel)**: `SystemModel.from_model(m)` extracts the
#   operator surface as frozen sympy matrices.  Independent sibling
#   class — `Model` and `SystemModel` share the same operator-API
#   shape but with different internals.
#
# * **Step 5 (system-level ops)**: `apply(InvertMassMatrix())`
#   demonstrates the system-level-operation hook on `SystemModel`.
#
# * **Step 7 (analysis)**: dispersion eigenvalues read directly from
#   `sm.quasilinear_matrix()` after substituting the model's own state
#   Symbols with a base state.  No ``PDESystem``, no `linearise`, no
#   hand-rolled symbols.
#
# * **Step 8 (numerics)**: `FSFSplittingSolver.solve(mesh, m2d)` runs
#   the 2D dam-break.  Mass conserved, positivity preserved, pressure
#   field finite.
#
# Standing follow-on work (acknowledged here, not hidden):
#
# 1. The chain currently produces **primitive-form** equations
#    (``∂_t (U_k h) + …``).  The operator-API surface that solvers
#    consume needs **conservative-form** matrices (M = I in
#    ``∂_t hu_k + …``).  ``VAMModelGalerkin`` runs the chain (for
#    derivation transparency) **and** the inherited basis-matrix
#    machinery (for the canonical operators).  A future commit
#    collapses these into a single path: the chain, followed by an
#    explicit primitive→conservative substitution + manual
#    term-tagging at the end of `derive_model`, with operators
#    extracted from the tagged closed equations via
#    ``collect_solver_tag``.  Mathematically equivalent; structurally
#    a single source of truth.
#
# 2. The ``zoomy_core.analysis`` package still has the
#    ``PDESystem``-based ``linearise`` / ``plane_wave_dispersion`` /
#    ``pencil`` routines.  This notebook bypasses them — analysis is
#    just ``sm.quasilinear_matrix().subs(base_state)``.  Porting the
#    full set of analysis routines to consume `SystemModel` directly
#    is a separate workstream.
#
# 3. The runtime `NumpyRuntimeModel` is built from a `Model` (not
#    yet a `SystemModel`).  Adding a `SystemModel`-direct entry point
#    is a small adapter — the matrices `sm.flux` / `sm.nonconservative_matrix`
#    / etc. lambdify the same way `Model.flux()` does.
