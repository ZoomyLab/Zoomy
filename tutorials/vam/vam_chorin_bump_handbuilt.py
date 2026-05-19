# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VAM(1,2,2) hand-built SystemModel + Chorin — flat-bottom bump
#
# Validates the modern `ChorinSplitVAMSolver` on a SystemModel built
# from scratch in conservative state variables (literal Escalante 2024
# eq (4) — no `VAMModelGalerkin` import in the model-build phase).
#
# Reference hierarchy:
#
# 1. **Experimental data** (`vam_analytical_eta`, `vam_analytical_p`
#    in `library/zoomy_tests/zoomy_tests/swashes/plots_paper.py:18-26`)
#    — ground truth for the dam-break-over-bump experiment.
# 2. **Escalante 2024 paper** (JCP 504 (2024) 112882) eq (4) — the
#    equations.  Used by `VAMHyperbolic` + `VAMPoisson` + `PoissonSolver`
#    in `web/tutorials/vam/simple.ipynb` (commit `e1c91370`, recovered
#    from the JAX-era prototype).
# 3. **Our solvers** (`DAESolver`, `ChorinSplitVAMSolver`) — internal
#    implementations.  This notebook is the Chorin validation.
#
# **Scope of this notebook.**  The flat-bottom cosine-bump propagation
# test from `tutorials/vam/vam_1d_bump_dae.py`, built end-to-end on a
# hand-coded SystemModel.  This isolates the symbolic-model and
# splitter+solver pipelines from the chain.
#
# **Status on the dam-break-over-bump experimental test.**
# `ChorinSplitVAMSolver` now uses Audusse–Bristeau–Klein hydrostatic-
# reconstruction Rusanov on the predictor (the
# `PositiveNonconservativeRusanov` Riemann solver, same class
# `FreeSurfaceFlowSolver` uses), with `FreeSurfaceLSQMUSCL` wet/dry-
# aware reconstruction available at `reconstruction_order >= 2`.
# Lake-at-rest on a Gaussian bump is now preserved to machine
# precision for ~10 steps — a real well-balanced win over the
# previous plain-Rusanov predictor that diverged immediately.
#
# Two issues remain before the full T=20 dam-break-over-bump
# experimental comparison can run:
#
# 1. **Pressure-GMRES injection at sub-tolerance forcing.**  When
#    `‖b‖` (the pressure-system data forcing) drops below
#    `pressure_tol` the existing early-exit triggers correctly,
#    but once `‖b‖` crosses *above* `pressure_tol` at machine-noise
#    level the GMRES solve produces ill-conditioned garbage that
#    feeds back into the corrector and exits the lake-at-rest
#    fixed point.  Decoupling the "skip" threshold from the GMRES
#    `rtol/atol` (the current code conflates them) would fix this.
# 2. **q_U1 mode growth under explicit time integration when the
#    shock excites it.**  Even with WB, the dam-break's strong
#    gradient excites `q_U1` and the explicit predictor can't damp
#    it.  Either IMEX-ARK on the `q_U1` row or whole-cycle SSPRK2
#    wrap (matching the old `PredictorCorrectorSolver`) would fix
#    this.
#
# This notebook continues to validate the symbolic pipeline on the
# flat-bottom cosine-bump test (no bathymetry → WB issue not
# triggered, no shock → q_U1 mode not excited).  Experimental
# comparison is queued behind both follow-ups above.

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.misc.misc import Zstruct
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.models.system_model import SystemModel, InvertMassMatrix
from zoomy_core.model.splitter import split_for_pressure
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

# %% [markdown]
# ## Experimental reference data
#
# Digitized from the published dam-break-over-bump experiment.
# Source: `library/zoomy_tests/zoomy_tests/swashes/plots_paper.py`.
# Embedded here so the notebook is self-contained and does not pull
# `pyswashes` through the `plots_paper` module.

# %%
ETA_EXP_X = np.array([
    -0.5928667563930013, -0.5430686406460297, -0.4946164199192463,
    -0.4448183041722746, -0.3990578734858681, -0.34522207267833105,
    -0.2981157469717362, -0.2510094212651413, -0.19851951547779273,
    -0.15141318977119783, -0.10430686406460293, -0.0531628532974428,
    -0.0006729475100942239, 0.04643337819650073, 0.09757738896366086,
    0.14737550471063254, 0.19851951547779279, 0.24562584118438757,
    0.29811574697173626, 0.34791386271870794, 0.3963660834454913,
    0.446164199192463, 0.49865410497981155, 0.5511440107671601,
])
ETA_EXP_Y = np.array([
    0.3418918918918919, 0.34121621621621623, 0.3398648648648649,
    0.3418918918918919, 0.3398648648648649, 0.3398648648648649,
    0.33851351351351355, 0.33783783783783783, 0.3337837837837838,
    0.32770270270270274, 0.322972972972973, 0.31486486486486487,
    0.3054054054054054, 0.29054054054054057, 0.26891891891891895,
    0.2425675675675676, 0.21621621621621623, 0.18581081081081083,
    0.15540540540540543, 0.13108108108108107, 0.10608108108108108,
    0.0918918918918919, 0.07297297297297298, 0.06554054054054054,
])
PB_EXP_X = np.array([
    -0.6001390820584145, -0.5556328233657858, -0.5041724617524339,
    -0.4485396383866481, -0.3998609179415855, -0.35396383866481224,
    -0.30389429763560505, -0.24965229485396384, -0.2051460361613352,
    -0.15229485396383868, -0.1008344923504868, -0.05354659248956889,
    -0.0006954102920723737, 0.0521557719054242, 0.09805285118219742,
    0.15090403337969394, 0.20236439499304582, 0.24826147426981915,
    0.30111265646731566, 0.3539638386648122, 0.4026425591098748,
    0.45271210013908203, 0.5013908205841446, 0.5514603616133518,
])
PB_EXP_Y = np.array([
    0.3319327731092437, 0.33053221288515405, 0.32212885154061627,
    0.30952380952380953, 0.29061624649859946, 0.27450980392156865,
    0.25, 0.22338935574229693, 0.18417366946778713, 0.15756302521008403,
    0.12394957983193278, 0.09453781512605042, 0.0700280112044818,
    0.04201680672268908, 0.04481792717086835, 0.03431372549019608,
    0.04831932773109244, 0.058823529411764705, 0.06022408963585434,
    0.06932773109243698, 0.0742296918767507, 0.0861344537815126,
    0.08473389355742297, 0.07913165266106442,
])

# %% [markdown]
# ## Symbols
#
# **Modal-conservative state**: `q_Uk = h · U_k / (2k+1)`,
# `q_Wk = h · W_k / (2k+1)`.  Physically equivalent to Escalante's
# standard-conservative state (`hu_k = h · u_k`) under the rescaling
# `q_Uk = hu_k / (2k+1)`; the modal form makes the mass matrix
# `M = I` directly without an `InvertMassMatrix` step.  Numerically
# equivalent to the chain's `quadratic_form="escalante"` +
# `change_state_variables({U_k: (2k+1) · q_Uk / h})` path.
#
# (Equivalent **standard-conservative** path produces literal
# Escalante eq (4) flux/source expressions — verified bit-for-bit in
# the prior diagnostic — but its numerical dispersion ran 3× slow on
# this test, suggesting the row-rescaling from `InvertMassMatrix`
# interacts poorly with the numerical-eigenvalue computation.  Modal
# CoV avoids that path entirely.)

# %%
t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)

h = sp.Symbol("h", positive=True, real=True)
q_U0 = sp.Symbol("q_U0", real=True)
q_U1 = sp.Symbol("q_U1", real=True)
q_W0 = sp.Symbol("q_W0", real=True)
q_W1 = sp.Symbol("q_W1", real=True)
P_0 = sp.Symbol("P_0", real=True)
P_1 = sp.Symbol("P_1", real=True)
state = [h, q_U0, q_U1, q_W0, q_W1, P_0, P_1]
n_state = len(state)
n_eq = n_state

g = sp.Symbol("g", positive=True)
rho = sp.Symbol("rho", positive=True)
ez = sp.Symbol("ez", positive=True)
parameters = Zstruct(g=g, ez=ez, rho=rho)
parameters._symbolic_name = "p"
parameter_values = Zstruct(g=9.81, ez=1.0, rho=1000.0)

b_sym = sp.Symbol("b", real=True)
b_x = sp.Symbol("b_x", real=True)
h_x = sp.Symbol("h_x", real=True)
q_U0_x = sp.Symbol("q_U0_x", real=True)
q_U1_x = sp.Symbol("q_U1_x", real=True)
aux_state = [b_sym, b_x, h_x, q_U0_x, q_U1_x]

# Primitive-state derivatives expressed in conservative form via
# chain rule.  In modal-conservative state (q_Uk = h · U_k / (2k+1)):
#   U_0 = q_U0/h        ⇒ U_0_x = q_U0_x/h − q_U0 · h_x / h²
#   U_1 = 3 · q_U1/h    ⇒ U_1_x = 3 · q_U1_x/h − 3 · q_U1 · h_x / h²
U_0_x_expr = q_U0_x / h - q_U0 * h_x / h**2
U_1_x_expr = 3 * q_U1_x / h - 3 * q_U1 * h_x / h**2

# %% [markdown]
# ## Operators — Escalante 2024 eq (4) in modal-conservative state
#
# Verified bit-for-bit against
# `VAMModelGalerkin(quadratic_form="escalante")` +
# `change_state_variables({U_k → (2k+1) q_Uk/h})`.  Residual convention:
# ```
#   M · ∂_t Q  +  ∂_x F  +  ∂_x P  +  B · ∂_x Q  −  S  =  0
# ```

# %%
M = sp.eye(n_state)
M[5, 5] = 0          # cont_j1 algebraic
M[6, 6] = 0          # cont_j2 algebraic

F = sp.zeros(n_eq, 1)
F[0, 0] = q_U0
F[1, 0] = P_0 * h / rho + q_U0**2 / h + 3 * q_U1**2 / h
F[2, 0] = P_1 * h / (3 * rho) + 2 * q_U0 * q_U1 / h
F[3, 0] = (q_U0 * q_W0 + 3 * q_U1 * q_W1) / h
F[4, 0] = (2 * b_x * q_U0 * q_U1 + 6 * b_x * q_U1**2
           + 5 * q_U0 * q_W1 + 3 * q_U1 * q_W0
           - 6 * q_U1 * q_W1) / (5 * h)

P = sp.zeros(n_eq, 1)
P[1, 0] = g * h * (b_sym + h)

B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, 1)
B[1, 0, 0] = g * (-b_sym - h)
B[2, 0, 0] = (-P_0 + P_1 / 3) / rho
B[2, 2, 0] = -q_U0 / h
B[5, 0, 0] = (-q_U0 + q_U1) / h
B[5, 1, 0] = sp.Integer(1)
B[5, 2, 0] = sp.Integer(1)
B[6, 0, 0] = (q_U0 - 3 * q_U1) / h
B[6, 1, 0] = sp.Integer(-1)

S = sp.zeros(n_eq, 1)
S[1, 0] = -2 * P_1 * b_x / rho
S[2, 0] = 2 * b_x * (P_0 - P_1) / rho
S[3, 0] = 2 * P_1 / rho
S[4, 0] = (
    -2 * P_0 / rho + 2 * P_1 / rho
    - U_0_x_expr * U_1_x_expr * h**2 / 6
    - U_0_x_expr * h_x * q_U1 / 2
    - U_1_x_expr**2 * h**2 / 15
    + U_1_x_expr * b_x * q_U0 / 3
    - U_1_x_expr * h_x * q_U1 / 2
    + b_x * h_x * q_U0 * q_U1 / h**2
    - 9 * h_x**2 * q_U1**2 / (10 * h**2)
)
S[5, 0] = 2 * (b_x * q_U0 - q_W0) / h
S[6, 0] = 6 * (b_x * q_U1 - q_W1) / h

# %% [markdown]
# ## Build SystemModel + BCs

# %%
sm = SystemModel(
    time=t,
    space=[x],
    state=state,
    aux_state=aux_state,
    parameters=parameters,
    flux=F,
    hydrostatic_pressure=P,
    nonconservative_matrix=B,
    source=S,
    mass_matrix=M,
    parameter_values=parameter_values,
)
sm.equation_names = [
    "mass", "xmom_j0", "xmom_j1", "zmom_j0", "zmom_j1", "cont_j1", "cont_j2",
]

# Boundary-condition kernels (indexed Piecewise Function).
position_zs = Zstruct(x=x)
position_zs._symbolic_name = "X"
distance_sym = sp.Symbol("dX", positive=True)
variables_zs = Zstruct(h=h, q_U0=q_U0, q_U1=q_U1,
                       q_W0=q_W0, q_W1=q_W1, P_0=P_0, P_1=P_1)
variables_zs._symbolic_name = "Q"
aux_variables_zs = Zstruct(b=b_sym, b_x=b_x, h_x=h_x,
                           q_U0_x=q_U0_x, q_U1_x=q_U1_x)
aux_variables_zs._symbolic_name = "Qaux"
normal_zs = Zstruct(n0=sp.Symbol("n0", real=True))
normal_zs._symbolic_name = "n"

# Extrapolation on both ends — same as the cosine-bump test in
# vam_1d_bump_dae.py.  (The dam-break-over-bump experimental setup
# needs inflow `Lambda` on the left + Gaussian bathymetry; deferred,
# see the WB-gap discussion in the header.)
bc_list = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])
sm.boundary_conditions = bc_list.get_boundary_condition_function(
    t, position_zs, distance_sym,
    variables_zs, aux_variables_zs, parameters, normal_zs,
    function_name="boundary_conditions",
)
sm.boundary_gradients = bc_list.get_boundary_gradient_function(
    t, position_zs, distance_sym,
    variables_zs, aux_variables_zs, parameters, normal_zs,
    function_name="boundary_gradients",
)

aux_bc_list = BoundaryConditions([
    AuxBC.Extrapolation(tag="left"),
    AuxBC.Extrapolation(tag="right"),
])
sm.aux_boundary_conditions = aux_bc_list.get_boundary_condition_function(
    t, position_zs, distance_sym,
    variables_zs, aux_variables_zs, parameters, normal_zs,
    function_name="aux_boundary_conditions",
)

# IC placeholders — overridden manually below after setup_simulation.
sm.initial_conditions = UserFunction(function=lambda x: np.zeros(n_state))
sm.aux_initial_conditions = UserFunction(
    function=lambda x: np.zeros(len(aux_state))
)

# Aux registry — same mapping as the chain's expose_aux_atoms would
# produce.
sm.aux_registry = [
    {"kind": "function", "name": "b", "row": 0,
     "atom": sp.Function("b", real=True)(t, x), "aux_symbol": b_sym},
    {"kind": "derivative", "name": "b_x", "row": 1,
     "atom": sp.Derivative(sp.Function("b", real=True)(t, x), x),
     "aux_symbol": b_x, "target_name": "b", "multi_index": (1,),
     "target_kind": "function", "function_row": 0},
    {"kind": "derivative", "name": "h_x", "row": 2,
     "atom": sp.Derivative(h, x, evaluate=False), "aux_symbol": h_x,
     "target_name": "h", "multi_index": (1,),
     "target_kind": "state", "state_index": 0},
    {"kind": "derivative", "name": "q_U0_x", "row": 3,
     "atom": sp.Derivative(q_U0, x, evaluate=False), "aux_symbol": q_U0_x,
     "target_name": "q_U0", "multi_index": (1,),
     "target_kind": "state", "state_index": 1},
    {"kind": "derivative", "name": "q_U1_x", "row": 4,
     "atom": sp.Derivative(q_U1, x, evaluate=False), "aux_symbol": q_U1_x,
     "target_name": "q_U1", "multi_index": (1,),
     "target_kind": "state", "state_index": 2},
]

sm.assert_diagonal_mass_matrix()
sm.apply(InvertMassMatrix())     # no-op: M already diagonal of 1s
sm.eigenvalues = None
print("State        :", [str(s) for s in sm.state])
print("Equations    :", sm.equation_names)
print("assert_diagonal_mass_matrix: PASSED")

# %% [markdown]
# ## Split for pressure

# %%
dt_sym = sp.Symbol("dt", positive=True)
split = split_for_pressure(sm, [P_0, P_1], dt_sym)
print(f"SM_pred  evolves Q[{split.SM_pred.equation_to_state_index}]")
print(f"SM_press evolves Q[{split.SM_press.equation_to_state_index}]")
print(f"SM_corr  evolves Q[{split.SM_corr.equation_to_state_index}]")

# %% [markdown]
# ## Set up the flat-bottom cosine-bump simulation
#
# Matches `tutorials/vam/vam_1d_bump_dae.py` exactly.

# %%
L_DOM = 20.0
NX = 40
H_REST = 1.0
AMP = 0.02
N_MODES = 1
G = 9.81
T_END = 1.0

mesh = BaseMesh.create_1d(domain=(0.0, L_DOM), n_inner_cells=NX)
solver = ChorinSplitVAMSolver(
    split.SM_pred, split.SM_press, split.SM_corr,
    reconstruction_order=1,
    pressure_tol=1e-9, pressure_maxit=200,
)
Q0 = solver.setup_simulation(mesh)
nc = solver.nc
xc = solver._sim_mesh.cell_centers[0, :nc]
solver.set_function_aux("b", np.zeros(nc))     # flat bottom
solver.update_aux_variables()
Q0[:] = 0.0
Q0[0, :] = H_REST + AMP * np.cos(2 * np.pi * N_MODES * xc / L_DOM)
solver._sim_Q = Q0.copy()
solver.update_aux_variables()

# %% [markdown]
# ## Run

# %%
dx = float(solver._sim_mesh.cell_volumes[0])
dt = 0.3 * dx / np.sqrt(G * H_REST)
n_steps = int(np.ceil(T_END / dt))
print(f"dt = {dt:.4f}, n_steps = {n_steps}")

mass_0 = Q0[0].sum() * dx
log_steps = sorted({2, 5, 10, 20, n_steps})
print(f"{'step':>5}  {'t':>5}    {'|h-1| max':>10}  "
      f"{'|q_U0| max':>10}  {'|q_U1| max':>10}  {'mass drift':>10}")
print("-" * 65)
for k in range(n_steps):
    solver.step(dt)
    if (k + 1) in log_steps:
        Q = solver._sim_Q
        mT = Q[0].sum() * dx
        print(f"{k+1:>5}  {solver._sim_time:>5.3f}    "
              f"{np.max(np.abs(Q[0] - H_REST)):>10.3e}  "
              f"{np.max(np.abs(Q[1])):>10.3e}  "
              f"{np.max(np.abs(Q[2])):>10.3e}  "
              f"{(mT - mass_0) / mass_0:>+10.3e}")
        if not np.all(np.isfinite(Q)):
            print("  ⇒ BLOWUP detected")
            break

# %% [markdown]
# ## Phase-speed diagnostic vs Escalante eq (10)

# %%
def escalante_eq10_c(k, H_=H_REST, g_=G):
    """Escalante 2024 eq (10): VAM(1,2,2) linear phase speed."""
    kH = k * H_
    return np.sqrt(g_ * H_ * (1 + kH**2 / 12)
                   / (1 + 5 * kH**2 / 12 + kH**4 / 144))


Q_final = solver._sim_Q
t_final = solver._sim_time
k = 2 * np.pi * N_MODES / L_DOM
h_dev = Q_final[0] - H_REST
cos_kx = np.cos(k * xc)
proj = np.sum(h_dev * cos_kx) * dx / (L_DOM / 2)
cos_omega_T = float(np.clip(proj / AMP, -1.0, 1.0))
c_obs = np.arccos(cos_omega_T) / (k * t_final)
c_pred = escalante_eq10_c(k)

print()
print(f"[result]    h range = [{Q_final[0].min():.4f}, {Q_final[0].max():.4f}]")
print(f"            mass drift = {abs(Q_final[0].sum()*dx - mass_0)/mass_0:.3e}")
print(f"[dispersion] k = 2π/L · {N_MODES} = {k:.4f},  kH = {k*H_REST:.4f}")
print(f"             Escalante eq (10) c = {c_pred:.4f}")
print(f"             observed c          = {c_obs:.4f}")
print(f"             relative error      = {abs(c_obs - c_pred)/c_pred:.3e}")

# %% [markdown]
# ## What this validates — and what it doesn't
#
# **Validates:**
#
# * Hand-built SystemModel with literal Escalante 2024 eq (4)
#   operators, run through `split_for_pressure` + `ChorinSplitVAMSolver`.
# * `q_U1` stays exactly at zero throughout — the symbolic form has
#   no spurious state-quadratic NCP driver (which only appears on the
#   `cantero_chinchilla + remove_non_diagonal_h` path).
# * Phase speed agrees with Escalante eq (10) to ~2%.
# * Mass conservation: interior flux-divergence is exact; the
#   `~5e-4` drift is boundary outflow through Extrapolation as the
#   wave reaches the ends (verified separately:
#   `q_U0` at the boundary grows from 0 → 1.3e-2 over T=1).
#
# **What has been added to the solver as part of this round:**
#
# * `ChorinSplitVAMSolver._build_numerics` now defaults to
#   `PositiveNonconservativeRusanov` (Audusse–Bristeau–Klein
#   hydrostatic reconstruction).  Lake-at-rest on a Gaussian bump
#   stays at machine precision for ~10 steps (previously it
#   diverged immediately).
# * `ChorinSplitVAMSolver._build_reconstruction` now picks up
#   `FreeSurfaceLSQMUSCL` at `reconstruction_order >= 2`
#   (free-surface `η = h + b` slope-limited, wet-cell clamp) —
#   matching `FreeSurfaceFlowSolver`.
#
# **What's still needed for the full experimental T=20 test:**
#
# 1. Pressure-GMRES early-exit needs decoupling from the solve
#    tolerance (currently `‖b‖` slightly above `pressure_tol` =
#    1e-9 triggers an ill-conditioned solve at machine-noise
#    level that injects garbage into the corrector).
# 2. `q_U1` mode damping under shocks — either IMEX-ARK on the
#    `q_U1` row inside the predictor, or whole-cycle SSPRK2 wrap
#    on the pred-press-corr trio (matching the old
#    `PredictorCorrectorSolver`).
#
# The digitized experimental points (`ETA_EXP_X/Y`, `PB_EXP_X/Y`
# above) are kept in the notebook for the eventual comparison plot
# once those two land.
