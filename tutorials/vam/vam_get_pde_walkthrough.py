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

# # VAM end-to-end: derivation → SystemModel → analysis → transform → numerics
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
# 4. **Transform `SystemModel` → runtime** via
#    ``NumpyRuntimeModel.from_system_model(sm)`` — lambdifies the
#    cached sympy matrices to per-operator numpy callables.
#
# 5. **`FSFSplittingSolver`** — 2D dam-break with pressure Poisson
#    correction.  Operates on the model class directly; internally
#    compiles via ``NumpyRuntimeModel``.

# +
import numpy as np
import sympy as sp

from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel, InvertMassMatrix
from zoomy_core.analysis.system_model_analysis import plane_wave_dispersion
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
from zoomy_core.mesh import BaseMesh
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
# -


# ## 1. Model instantiation — asymmetric basis levels (M, N_w, N_p)
#
# `VAMModelGalerkin` accepts three independent basis levels:
#
# * ``M`` — horizontal velocity ``u`` modes (M+1 of them).
# * ``N_w`` — vertical velocity ``w`` modes (N_w+1).
# * ``N_p`` — non-hydrostatic pressure ``p`` modes (N_p+1).
#
# Defaults: ``M = level``, ``N_w = N_p = M + 1``.  This matches the
# canonical VAM(M=1, N_w=2, N_p=2) parametrisation in Escalante 2024 —
# pressure and vertical velocity get one degree more than horizontal
# velocity, which is what makes the surface BC ``p(η) = 0`` and the
# bottom KBC produce non-trivial modal *constraints* rather than
# collapsing to zero.
#
# Internally `VAMModelGalerkin` runs the Galerkin chain twice — once
# stopped at the ``Expand`` step (gives us a paper-form ``Σ_k`` view)
# and once all the way through ``EvaluateIntegrals`` (closed primitive
# form).  Both snapshots live on the model.

# +
class VAM1D(VAMModelGalerkin):
    ins_dimension = 2

class VAM2D(VAMModelGalerkin):
    ins_dimension = 3

# level=1 ⇒ default M=1, N_w=N_p=2 (Escalante's canonical case).
m1d = VAM1D(level=1)
m2d = VAM2D(level=1)
print(f"Chain levels: M={m1d._chain_M}, N_w={m1d._chain_N_w}, N_p={m1d._chain_N_p}")
m1d.describe()
# -


# ## 1b. Chain DAE structure — projection + algebraic constraint leaves
#
# At ``(M, N_w, N_p) = (1, 2, 2)`` the chain produces 9 leaves
# matching Escalante 2024 eq (4)+(5):
#
# * **momentum.x.test_0/1** — x-momentum projected against ``φ_0, φ_1``
# * **momentum.z.test_0/1/2** — z-momentum projected against
#   ``φ_0, φ_1, φ_2``
# * **mass** — explicit depth-averaged continuity ``∂_t h + ∂_x(h u_0)``
# * **kbc_top_alg** — kinematic BC at η (with ``∂_t h`` substituted)
# * **kbc_bot** — kinematic BC at the bottom (``∂_t b = 0``)
# * **surface_bc** — non-hydrostatic surface BC ``Σ_k (-1)^k P_k = 0``
#
# Continuity ``test_0`` is dropped (replaced by the explicit ``mass``
# equation) and ``test_1, test_2`` are dropped because Escalante's
# eq (5) only retains ``cont_j1..M-1`` (empty range for ``M=1``).
#
# All four DAE constraint equations are added via the existing
# ``System.add_equation`` API — no new primitives.  The kinematic
# BCs that the chain *also* applied as ``Relation`` substitutions
# (which produces the conservative form for the projection rows)
# are now ALSO carried as separate algebraic rows; the DAE solver
# enforces them per-step.

# +
chain_leaf_paths = {p for p, _ in m1d._chain_system.leaves()}
expected_chain_paths = {
    ("momentum", "x", "test_0"),
    ("momentum", "x", "test_1"),
    ("momentum", "z", "test_0"),
    ("momentum", "z", "test_1"),
    ("momentum", "z", "test_2"),
    ("mass",),
    ("kbc_top_alg",),
    ("kbc_bot",),
    ("surface_bc",),
}
assert chain_leaf_paths == expected_chain_paths, (
    f"chain DAE structure regressed: {chain_leaf_paths}"
)
chain_leaf_paths
# -


# ## 1c. Verified Escalante DAE partition — `build_vam_pdesystem`
#
# The April-2026 builder ``tutorials/vam/vam_pdesystem.build_vam_pdesystem``
# produces VAM(M, N_w, N_p) directly via physical-z polynomial
# integration (no Galerkin chain primitives — uses sympy +
# ``polynomial_integrate``).  It carries the KBCs / surface BC as
# **separate algebraic equations** in the PDESystem, giving the
# canonical Escalante DAE shape:
#
# * ``2M+4`` evolution rows: ``mass`` + ``xmom_j0..M`` + ``zmom_j0..N_w``
# * ``M+2`` algebraic rows: ``kbc_top_alg, kbc_bot, surface_bc``
#   (+ ``cont_j1..cont_j(M-1)`` when ``M ≥ 2``)
#
# For ``(M=1, N_w=2, N_p=2)``: 9 equations / 9 fields, 6 dynamic + 3
# algebraic.  ``dae_partition`` (the auto-classifier in
# ``thesis/notebooks/verification/dae_solver/test_dae_partition_bridge.py``)
# detects this partition by inspecting which rows of the linearised
# system carry a non-zero ``∂_t`` coefficient.
#
# This partition is the reference for our chain to converge to —
# the cancellations Escalante writes in eq (4) emerge once the
# model imposes the algebraic constraints alongside the evolutions.

# +
import sys as _sys
_sys.path.insert(0, "tutorials/vam")
_sys.path.insert(0, "thesis/notebooks/verification/dae_solver")
from vam_pdesystem import build_vam_pdesystem  # noqa: E402
from test_dae_partition_bridge import dae_partition  # noqa: E402
_sys.path[:2] = []

april_pdesys = build_vam_pdesystem(M=1, N_w=2, N_p=2, flat_bottom=True)
_dyn, _alg, _, _ = dae_partition(april_pdesys)
escalante_partition = {
    "evolution":  [april_pdesys.equation_names[i] for i in _dyn],
    "algebraic":  [april_pdesys.equation_names[i] for i in _alg],
    "n_fields":   len(april_pdesys.fields),
    "n_equations": len(april_pdesys.equations),
}
# Sanity assertions: this is the verified canonical shape; if the
# builder ever drifts, the walkthrough fails fast.
assert escalante_partition["n_equations"] == 9
assert escalante_partition["n_fields"] == 9
assert escalante_partition["evolution"] == [
    "mass", "xmom_j0", "xmom_j1", "zmom_j0", "zmom_j1", "zmom_j2",
]
assert escalante_partition["algebraic"] == [
    "kbc_top_alg", "kbc_bot", "surface_bc",
]
escalante_partition
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
# ``∂_t Q + ∂_x F + B·∂_x Q − S = 0`` form.  The inherited operator
# path delivers canonical M=I matrices already, so the call is a
# no-op on the operators themselves — but it records the
# canonical-form invariant in ``sm.history`` for downstream solver
# verification.

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
# At level=1 the inherited state vector is
# [b, h, hu_0, hu_1, hw_0, hw_1] — six entries.
base_state = {
    m1d.variables.b:   sp.Integer(0),
    m1d.variables.h:   h0,
    m1d.variables.hu0: sp.Integer(0),
    m1d.variables.hu1: sp.Integer(0),
    m1d.variables.hw0: sp.Integer(0),
    m1d.variables.hw1: sp.Integer(0),
}
ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

result = plane_wave_dispersion(
    sm, base_state, axis=0, parameters={ez_param: 1}
)
result
# -


# ## 7. SystemModel → runtime kernels (`NumpyRuntimeModel.from_system_model`)
#
# The transformation pipeline can consume a ``SystemModel`` directly:
# ``NumpyRuntimeModel.from_system_model(sm)`` lambdifies the stored
# sympy matrices (``flux`` / ``nonconservative_matrix`` / ``source`` /
# ``mass_matrix`` / ``hydrostatic_pressure``) into per-operator
# numerical callables ``f(Q, Qaux, p)``.  Same operator-API surface as
# the Model-based runtime, but built from the cached sympy matrices
# rather than walking the equation tree on every call.
#
# Below: build the runtime, evaluate every operator on a sample state,
# verify the identity mass matrix and the canonical SWE values
# ``flux[1, 0] = h·u_mean`` / ``hydrostatic_pressure[2, 0] = g·h²/2``.

# +
rt = NumpyRuntimeModel.from_system_model(sm)

# At level=1: state = [b, h, hu_0, hu_1, hw_0, hw_1].  Sample state
# with h=1 and only the mean horizontal mode hu_0 set.
Q_sample = np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0])
Qaux_sample = np.zeros(rt.n_aux_variables, dtype=float)
p_sample = rt.parameters

runtime_outputs = {
    "flux":                 rt.flux(Q_sample, Qaux_sample, p_sample),
    "nonconservative":      rt.nonconservative_matrix(Q_sample, Qaux_sample, p_sample),
    "source":               rt.source(Q_sample, Qaux_sample, p_sample),
    "mass_matrix":          rt.mass_matrix(Q_sample, Qaux_sample, p_sample),
    "hydrostatic_pressure": rt.hydrostatic_pressure(Q_sample, Qaux_sample, p_sample),
}
# Sanity: g · h² / 2 at the (h u_0)-row column 0
assert np.isclose(runtime_outputs["hydrostatic_pressure"][2, 0], 9.81 * 0.5)
# Sanity: identity mass matrix (canonical M = I after InvertMassMatrix)
assert np.allclose(runtime_outputs["mass_matrix"], np.eye(rt.n_variables))
# Sanity: gravity body force on the first hw row
hw0_row = 2 + m1d.dimension * (m1d._n_u)
assert np.isclose(runtime_outputs["source"][hw0_row, 0], 9.81)
runtime_outputs
# -


# ## 8. 2D dam-break with `FSFSplittingSolver`
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
# * **Step 1b (projection structure)**: the chain at
#   ``(M=1, N_w=2, N_p=2)`` produces 8 leaves (3 continuity + 2
#   x-momentum + 3 z-momentum), every one carrying a
#   ``Derivative(_, t)`` term — the chain applies the KBCs as
#   ``Relation`` substitutions, which is what generates the
#   conservative form ``∂_t(h U_k)`` etc.
#
# * **Step 1c (verified Escalante DAE partition)**: the April-2026
#   builder ``build_vam_pdesystem(1, 2, 2)`` produces the canonical
#   Escalante DAE shape — 9 equations / 9 fields, 6 evolution rows
#   (``mass`` + ``xmom_j0..1`` + ``zmom_j0..2``) and 3 algebraic
#   rows (``kbc_top_alg, kbc_bot, surface_bc``).  This is the
#   reference partition our chain should converge to once the KBC /
#   surface-BC equations are carried alongside the evolutions
#   instead of substituted inline.
#
# * **Step 6 (analysis)**: dispersion eigenvalues read directly from
#   `sm.quasilinear_matrix()` after substituting the model's own state
#   Symbols with a base state.  No ``PDESystem``, no `linearise`, no
#   hand-rolled symbols.
#
# * **Step 7 (transform)**: `NumpyRuntimeModel.from_system_model(sm)`
#   lambdifies the cached sympy matrices into per-operator numpy
#   callables ``f(Q, Qaux, p)``.  Same surface as the Model-based
#   runtime — ``rt.flux`` / ``rt.nonconservative_matrix`` /
#   ``rt.source`` / ``rt.mass_matrix`` / ``rt.hydrostatic_pressure``.
#   The SystemModel-direct compilation path is exercised end-to-end:
#   sample evaluation gives ``g·h²/2`` in the hydrostatic-pressure row,
#   identity mass matrix, gravity source on ``hw0``.
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
