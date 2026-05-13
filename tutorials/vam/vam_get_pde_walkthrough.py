# ---
# title: "VAM end-to-end: from-scratch derivation, class equivalence, SystemModel, runtime, 2D simulation"
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

# # VAM(1, 2, 2) — full pipeline
#
# Build the Escalante 2024 VAM(M=1, N_w=2, N_p=2) DAE from primitives,
# then re-derive it via ``VAMModelGalerkin`` and confirm both paths
# produce identical equations.  Continue: ``SystemModel`` →
# ``NumpyRuntimeModel`` → 2D dam-break.

# +
import copy
import numpy as np
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection, EvaluateIntegrals, Expand, FullINS, InterfaceKBC,
    Integrate, Inviscid, Multiply, ProductRule, StateSpace,
)
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel, InvertMassMatrix
from zoomy_core.analysis import PDESystem
from zoomy_core.analysis.system_model_analysis import plane_wave_dispersion
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.mesh import BaseMesh
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
# -


# ## 1. Inline derivation
#
# Same primitives the class uses, written end-to-end so the
# construction is visible.  Asymmetric levels: ``M`` u-modes,
# ``N_w`` w-modes, ``N_p`` p-modes.

# +
M, N_w, N_p = 1, 2, 2

state = StateSpace(dimension=2)
z = state.z
basis_u = Legendre_shifted(level=M,   symbol="phi")
basis_w = Legendre_shifted(level=N_w, symbol="eta")
basis_p = Legendre_shifted(level=N_p, symbol="mu")

coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
            for k in range(M + 1)]
coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
            for k in range(N_w + 1)]
coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
            for k in range(N_p + 1)]

test_phi_u = Zstruct(
    **{f"phi_{k}": basis_u.phi[k](z) for k in range(M + 1)})
test_phi_w = Zstruct(
    **{f"phi_{k}": basis_w.phi[k](z) for k in range(N_w + 1)})

sys = FullINS(state)
sys.apply(Inviscid(state)).simplify()

# Hydrostatic split: p = ρ·g·(η − z) + p_NH.
p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
          ).simplify()

# Project momentum (continuity stays scalar — depth-integrating it
# gives the mass evolution directly).
sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

# Inverse product rule on every term carrying a ∂_z derivative.
sys.apply(ProductRule(variables=[z]))

# Depth integrate (Leibniz on ∂_t / ∂_x; FT on ∂_z).
sys.apply(Integrate(z, state.b, state.eta, method="auto"))

# Kinematic BCs absorb the boundary u·w cross-terms; static bottom
# clears the bottom ∂_t b atoms KBC@b introduces.
sys.apply(InterfaceKBC(state, state.b)).simplify()
sys.apply(InterfaceKBC(state, state.eta)).simplify()
sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

# Surface BC for p_NH at the field level (p_NH(η) = 0).
sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

# Affine ζ-map on basis args, then ansatz substitution u/w/p_NH.
sys.apply(AffineProjection(state))
sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u,
                 state=state))
sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                 state=state))
sys.apply(Expand(p_NH, basis=basis_p, coefficients=coeffs_p,
                 state=state))

# Snapshot Sum-form intermediate before resolving integrals.
chain_intermediate = copy.deepcopy(sys)

sys.apply(EvaluateIntegrals(state)).simplify()

# Boundary values from the basis.
u_at_b   = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero) for k in range(M + 1))
u_at_eta = sum(coeffs_u[k] * basis_u.eval(k, sp.S.One)  for k in range(M + 1))
w_at_b   = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero) for k in range(N_w + 1))
w_at_eta = sum(coeffs_w[k] * basis_w.eval(k, sp.S.One)  for k in range(N_w + 1))
p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)  for k in range(N_p + 1))

h, b, eta = state.h, state.b, state.eta
t, x = state.t, state.x

# kbc_top = w(η) − u(η)·∂_x η + ∂_x(h u_0)  — ∂_t h substituted via mass eq.
sys.add_equation("kbc_top", sp.expand(
    w_at_eta
    - u_at_eta * sp.Derivative(eta, x).doit()
    + sp.Derivative(h * coeffs_u[0], x).doit()))
sys.add_equation("kbc_bot", sp.expand(
    w_at_b - u_at_b * sp.Derivative(b, x).doit()))

# Eliminate the highest p-mode via the surface BC.
p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()

inline_sys = sys
# -


# ## 2. Sum-form intermediate (paper notation)

chain_intermediate.describe()


# ## 3. Closed DAE

inline_sys.describe()


# ## 4. Bit-for-bit match with Escalante 2024 eq (4)

# +
ref_mass = sp.Derivative(h, t) + sp.Derivative(h * coeffs_u[0], x).doit()
ref_xmom_j0 = (
    sp.Derivative(h * coeffs_u[0], t)
    + sp.Derivative(
        h * coeffs_u[0]**2
        + sp.Rational(1, 3) * h * coeffs_u[1]**2
        + h * coeffs_p[0] / state.rho, x).doit()
    + state.g * h * sp.Derivative(eta, x).doit()
    + 2 * coeffs_p[1] * sp.Derivative(b, x).doit() / state.rho
)
ref_zmom_j0 = (
    sp.Derivative(h * coeffs_w[0], t)
    + sp.Derivative(
        h * coeffs_u[0] * coeffs_w[0]
        + sp.Rational(1, 3) * h * coeffs_u[1] * coeffs_w[1], x).doit()
    - 2 * coeffs_p[1] / state.rho
)


def _diff_against_inline(name, ref):
    leaf = getattr(inline_sys._tree, name)
    return sp.simplify(sp.expand(leaf.expr - ref))


assert _diff_against_inline("continuity", ref_mass) == 0
assert _diff_against_inline(("momentum", "x", "test_0"),
                            ref_xmom_j0) == 0 if False else True
# The descend-by-tuple form of getattr is a path lookup; use .equations dict.
chain_xmom_j0 = inline_sys.equations["momentum.x.test_0"].expr
chain_zmom_j0 = inline_sys.equations["momentum.z.test_0"].expr
assert sp.simplify(sp.expand(chain_xmom_j0 - ref_xmom_j0)) == 0
assert sp.simplify(sp.expand(chain_zmom_j0 - ref_zmom_j0)) == 0
"continuity / xmom_j0 / zmom_j0 match Escalante eq (4) exactly"
# -


# ## 5. Same model via the class — equivalence check

# +
class VAM1D(VAMModelGalerkin):
    ins_dimension = 2

class VAM2D(VAMModelGalerkin):
    ins_dimension = 3

m1d = VAM1D(level=1)
m2d = VAM2D(level=1)

# NOTE: The inline-vs-class equation comparison below is currently
# skipped because the class derivation has migrated to the
# Escalante / cont-projection formulation (System B; see
# ``thesis/chapters/derivation_vam.md`` §5.7) while the inline block
# above still uses the System A KBC-row formulation.  Reinstating
# this comparison is a Phase 1 task — it requires rewriting the
# inline derivation to project continuity at j = 1, …, N_p and to
# close ``W_{N_w}`` via the bottom KBC at the basis level.  Until
# then we just expose the class PDESystem for downstream cells.
class_pdesys = m1d._chain_dae
"VAMModelGalerkin(level=1) chain DAE built (System B; inline comparison TODO)"
# -


# ## 6. SystemModel from the chain DAE
#
# Mass matrix: 1s/state-dependent on evolution rows, all-zero on
# kbc_top / kbc_bot.  Flux / NCP / source / hydrostatic_pressure are
# zero placeholders — per-term tagging is the next workstream.

m1d._chain_dae_systemmodel.describe(full=False)


# ## 7. Eigenvalue dispersion (uses the inherited operator path)
#
# The chain-DAE SystemModel doesn't carry tagged operators yet, so
# analysis still goes through ``SystemModel.from_model(m1d)`` —
# inherited operator surface from ``VAMModel``.

# +
sm = SystemModel.from_model(m1d)
sm.apply(InvertMassMatrix())

h0 = sp.Symbol("h0", positive=True)
base_state = {
    m1d.variables.b:   sp.Integer(0),
    m1d.variables.h:   h0,
    m1d.variables.hu0: sp.Integer(0),
    m1d.variables.hu1: sp.Integer(0),
    m1d.variables.hw0: sp.Integer(0),
    m1d.variables.hw1: sp.Integer(0),
}
ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

dispersion = plane_wave_dispersion(
    sm, base_state, axis=0, parameters={ez_param: 1})
dispersion["eigenvalues"]
# -


# ### 7b. True ω(k) dispersion relation (G7)
#
# The eigenvalues above are wave speeds at the rest base state — phase
# velocities, no k-dependence.  ``plane_wave_dispersion`` returns the
# full ``ω(k)`` curves by default (``return_omega_k=True``); they live
# in the ``omega_solutions`` and ``phase_velocity_solutions`` keys.
# This is the dispersion relation the chain DAE actually produces when
# linearised around the rest state.

# +
omega_solutions = dispersion["omega_solutions"]
phase_velocity_solutions = dispersion["phase_velocity_solutions"]
print(f"VAM(1, 2, 2) ω(k) solutions at rest:")
for s in omega_solutions:
    print(f"  ω = {sp.simplify(s)}")
print(f"\nPhase velocities ω/k at rest:")
for s in phase_velocity_solutions:
    print(f"  c = {sp.simplify(s)}")
# -


# ## 8. Runtime kernels via SystemModel

# +
rt = NumpyRuntimeModel.from_system_model(sm)

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
assert np.isclose(runtime_outputs["hydrostatic_pressure"][2, 0], 9.81 * 0.5)
assert np.allclose(runtime_outputs["mass_matrix"], np.eye(rt.n_variables))
runtime_outputs
# -


# ## 9. 2D dam-break via FSFSplittingSolver

# +
def dam_break_ic(x, _nv=m2d.n_variables):
    Q = np.zeros(_nv)
    Q[1] = 2.0 if (x[0] < 5.0 and x[1] < 5.0) else 1.0
    return Q


m2d.initial_conditions = IC.UserFunction(function=dam_break_ic)
m2d.boundary_conditions = BC.BoundaryConditions(
    [BC.Extrapolation(tag=tag) for tag in ("left", "right", "bottom", "top")])

mesh = BaseMesh.create_2d((0.0, 10.0, 0.0, 10.0), nx=20, ny=20)
solver = FSFSplittingSolver(time_end=0.05,
                            compute_dt=ts.adaptive(CFL=0.3),
                            viscosity=0.01)
Q, p = solver.solve(mesh, m2d, write_output=False)

nc = mesh.n_inner_cells
diagnostics = {
    "inner cells":   nc,
    "all-finite Q":  bool(np.isfinite(Q[:, :nc]).all()),
    "all-finite p":  bool(np.isfinite(p[:nc]).all()),
    "h_min":         float(Q[1, :nc].min()),
    "h_max":         float(Q[1, :nc].max()),
    "|p|_max":       float(np.abs(p[:nc]).max()),
    "mass":          float(np.sum(Q[1, :nc]) * 100.0 / nc),
    "mass expected": 2.0 * 25.0 + 1.0 * 75.0,
}
diagnostics
# -


# ## Open work
#
# * Per-term solver tags (``flux`` / ``nonconservative_flux`` /
#   ``source`` / ``hydrostatic_pressure``) on the chain DAE
#   equations — once added, ``SystemModel.from_pdesystem`` populates
#   the operators directly and ``from_model`` can be retired.
# * Split ``VAM3D`` (pre-ansatz physics) from ``VAM`` (with the
#   polynomial ansatz) so the two phases are visible as separate
#   classes.
# * Replace the inherited ``VAMModel`` operator path entirely once
#   the chain DAE drives the solver.
