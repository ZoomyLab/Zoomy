# ---
# title: "VAM walkthrough: get_pde() → analysis → numerics → 2D simulation"
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

# # VAM walkthrough — full pipeline via the model class
#
# Demonstrates the canonical flow once `Model.get_pde()` exists:
#
# 1. Instantiate a VAM model from the class hierarchy.
# 2. Get the symbolic strong-form PDE via `model.get_pde()`.
# 3. Analyse it with `zoomy_core.analysis` —
#    `linearise → extract_quasilinear_pencil → generalised_eigenvalues`.
# 4. Compile the symbolic model to a numpy runtime via the
#    transformation pipeline.
# 5. Run a small 2D dam-break with `FreeSurfaceFlowSolver`.
#
# No procedural builders, no `sys.path`, no manual sympy symbols —
# every variant of the model is a class in
# `library/zoomy_core/zoomy_core/model/models/`, every analysis routine
# in `zoomy_core.analysis` is fully model-agnostic, every solver in
# `zoomy_core.fvm` accepts the model class directly.

# +
import numpy as np
import sympy as sp

from zoomy_core.model.models.vam_model import VAMModel
from zoomy_core.analysis import (
    linearise, extract_quasilinear_pencil, generalised_eigenvalues,
)
from zoomy_core.analysis.pde_system import PDESystem
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.mesh import BaseMesh
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
# -

# ## 1. Model instantiation
#
# `VAMModel` derives from `INSModel(DerivedModel)` via the class
# hierarchy in `vam_model.py`. The 1D variant (`ins_dimension=2`) is
# the cleanest for a dispersion read; the 2D variant
# (`ins_dimension=3`) is what we'll evolve in step 5.

# +
class VAM1D(VAMModel):
    ins_dimension = 2

class VAM2D(VAMModel):
    ins_dimension = 3

m1d = VAM1D(level=0)
m2d = VAM2D(level=0)

print("VAM 1D, level=0:")
print("  variables    =", list(m1d.variables.keys()))
print("  aux          =", list(m1d.aux_variables.keys()))
print("  parameters   =", list(m1d.parameters.keys()))
print()
print("VAM 2D, level=0:")
print("  variables    =", list(m2d.variables.keys()))
print("  aux          =", list(m2d.aux_variables.keys()))
print("  dimension    =", m2d.dimension)
# -

# ## 2. Symbolic PDE in strong form
#
# `get_pde()` assembles
#
#     ∂_t Q + ∂_d (F + P)[:,d] + B[:,:,d] · ∂_d Q − S = 0
#
# from `flux()`, `hydrostatic_pressure()`, `nonconservative_matrix()`,
# `source()`, plus any algebraic aux closures in `aux_equations`. It
# returns a `zoomy_core.analysis.PDESystem` — the same struct every
# analysis routine consumes.

pde1d = m1d.get_pde()
print(pde1d)
print("fields:", pde1d.fields)
print("aux_fields:", pde1d.aux_fields)
print("parameters:", {k: v for k, v in pde1d.parameters.items()})
print()
print("Equation 1 (continuity):")
sp.pprint(sp.expand(pde1d.equations[1]))
print()
print("Equation 2 (x-momentum):")
sp.pprint(sp.expand(pde1d.equations[2]))

# Rationals like `hu0²/h` come out as-is — that is the strong form,
# no model-side regularisation has been applied.

# ## 3. Dispersion via the single model-agnostic analysis path
#
# `linearise(pde, base_state)` substitutes `Q → Q₀ + ε δQ`, expands to
# first order in ε, and returns a linearised `PDESystem` whose fields
# are the perturbations. For equations with rationals the routine
# falls back to a `sp.series` Taylor expansion automatically.
#
# `extract_quasilinear_pencil(linearised)` then reads `M_t`, `M_xa`,
# `M_0` purely by matching coefficients of `Derivative(δq, t)`,
# `Derivative(δq, x)`, and `δq` — no model knowledge required. This is
# what makes the path **fully model-agnostic**: every matrix comes
# from the same coefficient-extraction operation, regardless of where
# the PDE came from.

# +
b_f, h_f, hu_f, hw_f = pde1d.fields
h0, u0, w0 = sp.symbols("h0 u0 w0", positive=True)

# Drop the trivial ∂_t b = 0 row by substituting b ≡ 0 into all
# equations (flat-bottom assumption for the dispersion read).
pde_red = PDESystem(
    equations=[eq.xreplace({b_f: 0}) for eq in pde1d.equations[1:]],
    fields=[h_f, hu_f, hw_f],
    time=pde1d.time,
    space=pde1d.space,
    parameters=pde1d.parameters,
    aux_fields=pde1d.aux_fields,
)
base = {h_f: h0, hu_f: h0 * u0, hw_f: h0 * w0}
lin = linearise(pde_red, base)
M_t, M_xa, M_0 = extract_quasilinear_pencil(lin)
ez_par = next(s for s in pde1d.parameters if str(s) == "ez")
lambdas = [sp.simplify(s.subs({u0: 0, w0: 0, ez_par: 1}))
           for s in generalised_eigenvalues(M_xa[0], M_t)]
print("Eigenvalues at quiescent base state (u₀ = w₀ = 0, ez = 1):")
for s in lambdas:
    print(f"  λ = {s}")
# -

# ## 4. Code transformation — symbolic → numpy runtime
#
# `NumpyRuntimeModel(model)` walks every function registered on the
# model (`flux`, `nonconservative_matrix`, `source`, `eigenvalues`,
# BC kernels, …) and lambdifies each into a vectorised numpy
# callable. This is the second of the two-step pipeline: the model
# carries equations symbolically; the transformation compiles them to
# numerical kernels.

rt = NumpyRuntimeModel(m2d)
print("Compiled runtime functions:",
      sorted(rt.runtime_functions.keys())[:8], "…")

# ## 5. 2D dam-break with `FSFSplittingSolver`
#
# `FSFSplittingSolver` is the free-surface specialization of
# `SplittingSolver`: explicit hyperbolic predictor (positive Rusanov
# with hydrostatic reconstruction; b at index 0, h at index 1) plus
# explicit viscous diffusion plus a pressure Poisson correction
# (Chorin projection). This is the right solver for non-hydrostatic
# VAM. It accepts the model class directly — internally it calls the
# `NumpyRuntimeModel` runtime from step 4. Every substep is
# ghost-cell-free: BCs are evaluated inline at boundary faces by
# calling the BC's `face_value(...)` directly, no ghost-cell fill.

# +
def dam_break_ic(x, _nv=m2d.n_variables):
    Q = np.zeros(_nv)
    Q[1] = 2.0 if (x[0] < 5.0 and x[1] < 5.0) else 1.0
    return Q

m2d.initial_conditions = IC.UserFunction(function=dam_break_ic)
m2d.boundary_conditions = BC.BoundaryConditions([
    BC.Extrapolation(tag=t) for t in ("left", "right", "bottom", "top")
])

mesh = BaseMesh.create_2d((0.0, 10.0, 0.0, 10.0), nx=20, ny=20)
solver = FSFSplittingSolver(time_end=0.05,
                            compute_dt=ts.adaptive(CFL=0.3),
                            viscosity=0.01)
Q, p = solver.solve(mesh, m2d, write_output=False)
nc = mesh.n_inner_cells
h = Q[1, :nc]
hu = Q[2, :nc]
hv = Q[3, :nc]
hw = Q[4, :nc]

print(f"Inner cells     : {nc}")
print(f"All-finite Q    : {np.isfinite(Q[:, :nc]).all()}")
print(f"All-finite p    : {np.isfinite(p[:nc]).all()}")
print(f"Positivity h≥0  : {(h >= -1e-10).all()}")
print(f"h range         : [{h.min():.4f}, {h.max():.4f}]")
print(f"|u_max|         : {(np.abs(hu / h)).max():.4f}")
print(f"|v_max|         : {(np.abs(hv / h)).max():.4f}")
print(f"|w_max|         : {(np.abs(hw / h)).max():.4f}")
print(f"|p|max          : {np.abs(p[:nc]).max():.4f}")
mass = float(np.sum(h) * (10.0 * 10.0) / nc)
mass_expected = 2.0 * 25.0 + 1.0 * 75.0
print(f"mass {mass:.3f} (expected {mass_expected})")
# -

# ## Summary
#
# The full chain `Model class → get_pde() → analysis → transformation
# → solver` runs end-to-end via a single model-agnostic path. Every
# variant (level, dimension, regularization) is just a constructor or
# subclass of `DerivedModel`. The analysis package never inspects
# model attributes — it only consumes a `PDESystem` and extracts
# pencil matrices via coefficient matching.
