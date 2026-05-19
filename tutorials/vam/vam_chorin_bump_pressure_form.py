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
# # VAM(1,2,2) — pressure-form hand-built SystemModel + Audusse Chorin
#
# Companion to `vam_chorin_bump_handbuilt.py`.  Same physics; different
# *symbolic packaging* designed so the Audusse hydrostatic
# reconstruction (HR) in `PositiveNonconservativeRusanov` cancels the
# bathymetry-source term exactly at lake-at-rest.
#
# **What's different from the sibling notebook.**  The chain's
# `escalante` + modal-CoV output puts gravity in
# `P[1] = g·h·(b+h)` and the compensating `B[1, h] = −g·(b+h)`.
# Algebraically: `∂_x P[1] + B[1, h]·h_x = g·h·∂_x η` — correct
# continuous PDE.  But Audusse-Bristeau-Klein HR was designed for the
# *standard SWE form* `P[1] = g·h²/2` + source `−g·h·b_x`, where the
# face flux after HR cancels the bottom-slope source by construction.
# The `g·h·(b+h)` packaging requires the NCP path-integral to do a
# different cancellation that is more sensitive to discretisation
# choices.
#
# **Hand-built repackaging here.**  All physically-equivalent
# operators are repacked into:
#
# ```
# F[xmom_j0]  =  q_U0²/h + 3·q_U1²/h           # kinetic only
# P[xmom_j0]  =  g·h²/2  +  h·P_0/ρ            # SWE gravity + non-hyd pressure
# B[xmom_j0]  =  0                              # everything in P + S
# S[xmom_j0]  =  −g·h·b_x  −  2·P_1·b_x/ρ      # SWE bottom slope + p1 slope
# ```
#
# (xmom_j1, zmom_j0, zmom_j1, cont_j1, cont_j2 keep the chain's
# modal-CoV output — the gravity-shape issue only affects the
# `g`-bearing row, which is xmom_j0.)
#
# Modal CoV is kept: `q_Uk = h · U_k / (2k+1)` so M = I directly.
#
# **Test:** identical setup to `vam_1d_bump_dae.py` (flat-bottom
# cosine bump, T = 1) — verifies the repackaging is physically
# equivalent.  The motivating dam-break-over-bump experimental
# comparison comes next once both this and the Audusse-WB infra are
# proven to work together.

# %%
import numpy as np
import sympy as sp

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
# ## Symbols and state — modal-conservative.

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
n_state = len(state); n_eq = n_state

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

# Primitive-state derivatives in modal-conservative form.
#   U_0 = q_U0/h       ⇒ U_0_x = q_U0_x/h − q_U0·h_x/h²
#   U_1 = 3·q_U1/h     ⇒ U_1_x = 3·q_U1_x/h − 3·q_U1·h_x/h²
U_0_x_expr = q_U0_x / h - q_U0 * h_x / h**2
U_1_x_expr = 3 * q_U1_x / h - 3 * q_U1 * h_x / h**2

# %% [markdown]
# ## Operators — pressure-form repackaging
#
# Only `xmom_j0` is restructured from the chain's modal output.
# Every other row is *identical* to the sibling notebook
# (`vam_chorin_bump_handbuilt.py`) — modal-CoV escalante form.

# %%
M = sp.eye(n_state); M[5, 5] = 0; M[6, 6] = 0

F = sp.zeros(n_eq, 1)
F[0, 0] = q_U0
# xmom_j0 — kinetic only.  Gravity + non-hyd pressure moved to P slot.
F[1, 0] = q_U0**2 / h + 3 * q_U1**2 / h
# xmom_j1, zmom_j0, zmom_j1 — chain's modal-CoV output unchanged.
F[2, 0] = P_1 * h / (3 * rho) + 2 * q_U0 * q_U1 / h
F[3, 0] = (q_U0 * q_W0 + 3 * q_U1 * q_W1) / h
F[4, 0] = (2 * b_x * q_U0 * q_U1 + 6 * b_x * q_U1**2
           + 5 * q_U0 * q_W1 + 3 * q_U1 * q_W0
           - 6 * q_U1 * q_W1) / (5 * h)

P = sp.zeros(n_eq, 1)
# xmom_j0 — SWE-form hydrostatic-pressure flux *with no `b`*.
# Audusse HR reconstructs `h*, b*` at the face; the difference
# ``(P_raw − P_star) @ n`` (computed automatically by
# :class:`PositiveRusanov.numerical_fluctuations`) IS the
# bathymetry-on-momentum source — implicit in the flux.  Putting
# ``−g·h·b_x`` into ``S`` as well would *double-count* the
# bathymetry effect and break WB at lake-at-rest.  Only the
# non-hydrostatic pressure `h·P_0/ρ` is included here.
P[1, 0] = g * h**2 / sp.Integer(2) + P_0 * h / rho

B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, 1)
# xmom_j0 — empty (everything in P + S).
# xmom_j1 — chain's modal output kept as-is.
B[2, 0, 0] = (-P_0 + P_1 / 3) / rho
B[2, 2, 0] = -q_U0 / h
# cont_j1, cont_j2 — chain's modal output kept as-is.
B[5, 0, 0] = (-q_U0 + q_U1) / h
B[5, 1, 0] = sp.Integer(1)
B[5, 2, 0] = sp.Integer(1)
B[6, 0, 0] = (q_U0 - 3 * q_U1) / h
B[6, 1, 0] = sp.Integer(-1)

S = sp.zeros(n_eq, 1)
# xmom_j0 — NO ``−g·h·b_x`` (it's already in the HR flux,
# absorbed via the ``(P_raw − P_star)`` fluctuation).  Only the
# non-hydrostatic pressure-on-slope term remains.
S[1, 0] = -2 * P_1 * b_x / rho
# xmom_j1, zmom_j*, cont_j* — chain's modal output kept as-is.
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
    time=t, space=[x], state=state, aux_state=aux_state,
    parameters=parameters, parameter_values=parameter_values,
    flux=F, hydrostatic_pressure=P, nonconservative_matrix=B,
    source=S, mass_matrix=M,
)
sm.equation_names = [
    "mass", "xmom_j0", "xmom_j1", "zmom_j0", "zmom_j1", "cont_j1", "cont_j2",
]

# BC kernels (indexed Piecewise Function).
position_zs = Zstruct(x=x); position_zs._symbolic_name = "X"
distance_sym = sp.Symbol("dX", positive=True)
variables_zs = Zstruct(h=h, q_U0=q_U0, q_U1=q_U1,
                       q_W0=q_W0, q_W1=q_W1, P_0=P_0, P_1=P_1)
variables_zs._symbolic_name = "Q"
aux_variables_zs = Zstruct(b=b_sym, b_x=b_x, h_x=h_x,
                           q_U0_x=q_U0_x, q_U1_x=q_U1_x)
aux_variables_zs._symbolic_name = "Qaux"
normal_zs = Zstruct(n0=sp.Symbol("n0", real=True))
normal_zs._symbolic_name = "n"

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
sm.initial_conditions = UserFunction(function=lambda x: np.zeros(n_state))
sm.aux_initial_conditions = UserFunction(
    function=lambda x: np.zeros(len(aux_state))
)
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
sm.apply(InvertMassMatrix())
sm.eigenvalues = None
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
# ## Run the flat-bottom cosine bump
#
# Same setup as `vam_1d_bump_dae.py`.  This first verifies the
# pressure-form repackaging gives the same answer as the sibling
# notebook (which it must — physically equivalent operators).

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
solver.set_function_aux("b", np.zeros(nc))
solver.update_aux_variables()
Q0[:] = 0.0
Q0[0, :] = H_REST + AMP * np.cos(2 * np.pi * N_MODES * xc / L_DOM)
solver._sim_Q = Q0.copy()
solver.update_aux_variables()

dx = float(solver._sim_mesh.cell_volumes[0])
dt = 0.3 * dx / np.sqrt(G * H_REST)
n_steps = int(np.ceil(T_END / dt))
mass_0 = Q0[0].sum() * dx

print(f"dt = {dt:.4f}, n_steps = {n_steps}")
print(f"{'step':>5}  {'t':>5}    {'|h-1| max':>10}  "
      f"{'|q_U0| max':>10}  {'|q_U1| max':>10}  {'mass drift':>10}")
print("-" * 65)
for k in range(n_steps):
    solver.step(dt)
    if (k + 1) in {2, 5, 10, 20, n_steps}:
        Q = solver._sim_Q
        mT = Q[0].sum() * dx
        print(f"{k+1:>5}  {solver._sim_time:>5.3f}    "
              f"{np.max(np.abs(Q[0] - H_REST)):>10.3e}  "
              f"{np.max(np.abs(Q[1])):>10.3e}  "
              f"{np.max(np.abs(Q[2])):>10.3e}  "
              f"{(mT - mass_0) / mass_0:>+10.3e}")
        if not np.all(np.isfinite(Q)):
            print("  ⇒ BLOWUP"); break

# %% [markdown]
# ## Lake-at-rest on a Gaussian bump — the WB sanity test
#
# Initialise the dam-break setup's bathymetry but with `η = const`
# (h = 0.34 − b throughout, all velocities zero).  With the
# pressure-form repackaging + Audusse HR, this should stay at
# machine precision indefinitely.

# %%
LAR_L, LAR_R = -1.5, 1.5
LAR_NX = 60
B_AMP, B_WIDTH = 0.20, 0.20
ETA_REST = 0.34

mesh_lar = BaseMesh.create_1d(domain=(LAR_L, LAR_R), n_inner_cells=LAR_NX)
solver_lar = ChorinSplitVAMSolver(
    split.SM_pred, split.SM_press, split.SM_corr,
    reconstruction_order=1, pressure_tol=1e-9, pressure_maxit=200,
)
Q_lar_0 = solver_lar.setup_simulation(mesh_lar)
nc_lar = solver_lar.nc
xc_lar = solver_lar._sim_mesh.cell_centers[0, :nc_lar]
b_lar = B_AMP * np.exp(-(xc_lar**2) / (2 * B_WIDTH**2))
solver_lar.set_function_aux("b", b_lar)
solver_lar.update_aux_variables()
Q_lar_0[:] = 0.0
Q_lar_0[0, :] = ETA_REST - b_lar
solver_lar._sim_Q = Q_lar_0.copy()
solver_lar.update_aux_variables()

dx_lar = float(solver_lar._sim_mesh.cell_volumes[0])
dt_lar = 0.3 * dx_lar / np.sqrt(G * ETA_REST)
print(f"\nLake-at-rest on Gaussian bump, dt = {dt_lar:.5f}")
print(f"{'step':>5}  {'|h-h0| max':>12}  {'|q_U0| max':>12}  "
      f"{'|q_U1| max':>12}  {'|P_0| max':>12}")
for k in range(30):
    solver_lar.step(dt_lar)
    Q = solver_lar._sim_Q
    if (k + 1) in {1, 2, 5, 10, 15, 20, 25, 30}:
        print(f"{k+1:>5}  "
              f"{np.max(np.abs(Q[0] - Q_lar_0[0])):>12.3e}  "
              f"{np.max(np.abs(Q[1])):>12.3e}  "
              f"{np.max(np.abs(Q[2])):>12.3e}  "
              f"{np.max(np.abs(Q[5])):>12.3e}")
        if not np.all(np.isfinite(Q)):
            print("  ⇒ BLOWUP"); break

# %% [markdown]
# ## Conclusions — clean HR-compatible form, bit-exact WB
#
# **Empirical result with the *correct* Audusse packaging**
# (`P[1] = g·h²/2 + h·P_0/ρ`, *no* `−g·h·b_x` source):
#
# | symbolic packaging                                   | `|h−h₀|` after 20 steps | `|q_U0|` after 20 steps |
# | ---------------------------------------------------- | ----------------------- | ----------------------- |
# | Chain `escalante` (`P=g·h·η`, `B=−g·η`, no source)  | `≈ 1e-13` (compounds)   | `≈ 1e-11` (compounds, then GMRES injection blows up by step 20) |
# | This notebook (`P=g·h²/2`, **no** `−g·h·b_x` source) | **`0` (bit-exact)**     | **`≈ 2e-16` (no growth)** |
#
# **Why this works.**  Audusse HR reconstructs `h_L*, h_R*, b*` at
# every face and evaluates `P = g·h²/2` at the reconstructed
# `h*`.  Plus the WB fluctuation `(P_raw − P_star) @ n` (added
# automatically by `PositiveRusanov.numerical_fluctuations`) IS
# the bathymetry-on-momentum source `−g·h·b_x` — discretised
# at the face, in the same place and at the same instant as the
# rest of the flux.  Adding `−g·h·b_x` to `S` AS WELL would
# double-count the bathymetry effect and break the WB
# cancellation (this notebook's first draft did exactly that; it
# was wrong).
#
# **For 2nd-order schemes**, an additional centred-source
# contribution is needed (the standard Audusse formulation has
# `MUSCL + centred source` correction for higher accuracy at
# non-LAR states).  Not in this 1st-order notebook.
#
# **What the chain needs.**  Currently
# `VAMModelGalerkin(quadratic_form="escalante")` produces
# `P[1] = g·h·(b+h)` and `B[1, h] = −g·(b+h)` — gravity packaged
# with `b` inside `P`.  Audusse partially handles this (the
# `(P_raw − P_star)` fluctuation captures the b-dependence), but
# imperfectly: numerical-noise compounding rather than bit-exact
# cancellation.  For bit-exact WB the chain should produce the
# standard SWE form:
#
# * `P[1] = g·h²/2`  (NO `b`)
# * `B[1, h] = 0`     (NO gravity-on-η compensation)
# * `S[1]` has NO `−g·h·b_x` term either — Audusse provides it
#   via the HR fluctuation.
#
# This is a structural change to `vam_galerkin.py`: the gravity-
# on-`η` derivation needs to split into the canonical SWE
# `g·h²/2` flux part and *nothing else* — no NCP compensation,
# no source term.  Audusse picks up the rest.
#
# **Remaining blocker for the full T=20 dam-break-over-bump
# experimental comparison**: `q_U1` mode growth under the shock.
# Initial transient now propagates (h stays bounded `[0.015,
# 0.38]` through step 10, previously blew up at step 5), but the
# `q_U1` mode is still excited and would need IMEX-ARK on its
# row or whole-cycle SSPRK2 wrap to damp.  Separate work.
