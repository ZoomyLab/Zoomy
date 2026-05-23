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
# # VAM(1,2,2) 8-state SystemModel — **NCP-only Riemann route, no HR**
#
# Companion to ``vam_chorin_bump_8state_chain.py``.  Same chain-
# derived 8-state ``SystemModel``, same dam-break-over-bump test,
# same DG(0) reconstruction — but a **different Riemann route**:
#
# ```
#   chain (no HydrostaticReconstruction step!)
#         │
#         ▼
#   SystemModel (8-state, M = I, gravity still in NCP)
#         │
#         ▼
#   ChorinSplitVAMSolver(riemann_solver="ncp", ...)
#         │
#         ▼
#   _NCPRusanovLARBalanced  ← inline subclass of NonconservativeRusanov
#                             with Id[h, b] = 1 added to the fluctuation
#                             dissipation; LAR cancellation comes from
#                             dh + db = 0 inside the path-integral.
# ```
#
# The two key differences from the HR notebook:
#
# 1. **``HydrostaticReconstruction`` is not applied.**  The chain
#    leaves ``g·h·∂_x(b + h)`` symbolically in NCP (auto-tagger routes
#    the bed and surface derivatives into ``B[xmom_j0, h, x] = g·h``
#    and ``B[xmom_j0, b, x] = g·h``).  Without HR the symbolic
#    pressure tensor stays at the primitive form.
# 2. **No Audusse face-state reconstruction.**  The solver swaps the
#    default ``PositiveNonconservativeRusanov`` for a plain
#    ``NonconservativeRusanov`` whose ``get_viscosity_identity_
#    fluctuations`` carries the LAR-balance entry ``Id[h, b] = 1``.
#    For lake-at-rest the h-row dissipation becomes
#    ``s_max · (dh + db) = 0`` — cancellation by symbolic identity,
#    not by face-state engineering.
#
# This is the **second DG(0) baseline** of the dam-break-over-bump
# test.  Once both routes pass the validation integration test
# (``tests/integration/zoomy_core/test_vam_dam_break_validation.py``,
# tolerance set by Ingo's eyeball check on these two PNGs) the
# higher-order ``PrimitiveReconstruction`` wrapper can be revisited.

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.misc.misc import Zstruct
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, Lambda,
)
from zoomy_core.model.initial_conditions import UserFunction
from zoomy_core.model.models.system_model import (
    SystemModel,
    InvertMassMatrix,
)
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.splitter import split_simple
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

# %% [markdown]
# ## Build the 8-state SystemModel via the chain — **without HR**
#
# Note the absence of ``sm.apply(HydrostaticReconstruction())``.  The
# gravity stays in NCP as a symbolic identity; the NCP-Riemann route
# below handles the bed-slope force via path-integral fluctuations.

# %%
m = VAMModelGalerkin(level=1, dimension=2, quadratic_form="escalante")
m.parameters.g = 9.81
m.boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])
sm = SystemModel.from_model(m)

# Modal-conservative change of variables.  ``q_Uk = h · U_k · μ_k``
# with ``μ_k = 1/(2k+1)``, so ``U_k = (2k+1)·q_Uk / h``.
h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
q_U0, q_U1 = sp.symbols("q_U0 q_U1", real=True)
q_W0, q_W1 = sp.symbols("q_W0 q_W1", real=True)
sm.change_state_variables(
    new_state=[h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1],
    transform={U_0: q_U0 / h, U_1: q_U1 / h,
               W_0: q_W0 / h, W_1: q_W1 / h},
)
sm.apply(InvertMassMatrix())
# Intentionally NO HydrostaticReconstruction — gravity stays in NCP.
sm.eigenvalues = None

print("STATE         :", [str(s) for s in sm.state])
print("AUX           :", [str(s) for s in sm.aux_state])
print("EQUATIONS     :", sm.equation_names)
print("M diag        :", [sp.simplify(sm.mass_matrix[i, sm.equation_to_state_index[i]])
                            for i in range(sm.n_equations)])

# %% [markdown]
# ## Attach BCs
#
# Same configuration as the HR notebook: prescribed inflow on the
# left (``q_U0``, ``q_U1``, ``q_W0``, ``q_W1``, ``b``); extrapolated
# on the right.

# %%
b_state_idx = [str(s) for s in sm.state].index("b")
P_indices = [i for i, s in enumerate(sm.state)
             if str(s).startswith("P_")]

b_inflow_value = 0.20 * np.exp(-(-1.5)**2 / (2 * 0.20**2))
bc_list = BoundaryConditions([
    Lambda(tag="left", prescribe_fields={
        1: lambda *a: 0.11197,     # q_U0
        2: lambda *a: 0.0,         # q_U1
        3: lambda *a: 0.0,         # q_W0
        4: lambda *a: 0.0,         # q_W1
        b_state_idx: lambda *a: b_inflow_value,
    }),
    Extrapolation(tag="right"),
])

# Rebuild indexed-BC kernels for the 8-state layout.
t, x = sm.time, sm.space[0]
position_zs = Zstruct(x=x); position_zs._symbolic_name = "X"
dX = sp.Symbol("dX", positive=True)
variables_zs = Zstruct(**{str(s): s for s in sm.state})
variables_zs._symbolic_name = "Q"
aux_variables_zs = Zstruct(**{str(s): s for s in sm.aux_state})
aux_variables_zs._symbolic_name = "Qaux"
normal_zs = Zstruct(n0=sp.Symbol("n0", real=True))
normal_zs._symbolic_name = "n"

sm.boundary_conditions = bc_list.get_boundary_condition_function(
    t, position_zs, dX, variables_zs, aux_variables_zs,
    sm.parameters, normal_zs, function_name="boundary_conditions")
sm.boundary_gradients = bc_list.get_boundary_gradient_function(
    t, position_zs, dX, variables_zs, aux_variables_zs,
    sm.parameters, normal_zs, function_name="boundary_gradients")
aux_bc_list = BoundaryConditions([
    AuxBC.Extrapolation(tag="left"),
    AuxBC.Extrapolation(tag="right"),
])
sm.aux_boundary_conditions = aux_bc_list.get_boundary_condition_function(
    t, position_zs, dX, variables_zs, aux_variables_zs,
    sm.parameters, normal_zs, function_name="aux_boundary_conditions")

# Initial-condition placeholders — the user overrides Q0 manually.
sm.initial_conditions = UserFunction(function=lambda x: np.zeros(sm.n_state))
sm.aux_initial_conditions = UserFunction(
    function=lambda x: np.zeros(len(sm.aux_state)))

# %% [markdown]
# ## Solve — DG(0), NCP Riemann route
#
# The single difference from the HR notebook is the
# ``riemann_solver="ncp"`` flag, which makes the solver swap its
# face-flux numerics for a plain ``NonconservativeRusanov`` with
# ``Id[h, b] = 1`` added to the fluctuation dissipation.  Everything
# else — the SystemModel, the predictor / pressure / corrector split,
# the BCs, the initial condition, the mesh, the CFL — is identical.

# %%
split = split_simple(sm, [sm.state[i] for i in P_indices],
                     sp.Symbol("dt", positive=True))


def run(T_end=20.0):
    """End-to-end DG(0) Chorin-NCP run on the chain-derived 8-state SM."""
    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=60)
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        riemann_solver="ncp",
        time_order=1,
        pressure_tol=1e-9, pressure_maxit=200,
    )
    Q0 = solver.setup_simulation(mesh)
    xc = solver._sim_mesh.cell_centers[0, :solver.nc]
    b_vals = 0.20 * np.exp(-(xc**2) / (2 * 0.20**2))
    Q0[:] = 0
    Q0[0, :] = np.maximum(np.where(xc < 1.0, 0.34 - b_vals, 0.015), 0.015)
    Q0[b_state_idx, :] = b_vals
    solver._sim_Q = Q0.copy()
    solver.update_aux_variables()
    dx = float(solver._sim_mesh.cell_volumes[0])
    cfl = 0.3
    dt = cfl * dx / (np.sqrt(9.81 * 0.34) + 1.0)
    n_steps = int(np.ceil(T_end / dt))
    print(f"\n--- DG(0) NCP: cfl={cfl}, dt={dt:.5f}, n_steps={n_steps} ---")
    print(f"{'step':>5} {'t':>6} {'hmin':>7} {'hmax':>7} "
          f"{'|q_U0|':>10} {'|q_U1|':>10}")
    log_steps = sorted({1, 5, 20, 100, 500, 1000, 2000,
                        n_steps // 2, n_steps})
    for k in range(n_steps):
        solver.step(dt)
        Q = solver._sim_Q
        if (k + 1) in log_steps:
            print(f"{k+1:>5} {solver._sim_time:>6.3f} {Q[0].min():>7.4f} "
                  f"{Q[0].max():>7.4f} {np.max(np.abs(Q[1])):>10.3e} "
                  f"{np.max(np.abs(Q[2])):>10.3e}")
        if not np.all(np.isfinite(Q)):
            print(f"  BLOWUP at step {k+1}")
            return None
    return solver, xc, b_vals


result_ncp = run()

# %% [markdown]
# ## Plot vs experimental data — DG(0) NCP route
#
# Digitized free-surface measurements from the Escalante 2024
# dam-break-over-bump experiment.

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
# Bottom pressure (m-head): ``p_b/g = h + 2·P_1/g``.  Digitized from
# ``library/zoomy_tests/zoomy_tests/swashes/plots_paper.py::vam_analytical_p``.
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
    0.25, 0.22338935574229693, 0.18417366946778713,
    0.15756302521008403, 0.12394957983193278, 0.09453781512605042,
    0.0700280112044818, 0.04201680672268908, 0.04481792717086835,
    0.03431372549019608, 0.04831932773109244, 0.058823529411764705,
    0.06022408963585434, 0.06932773109243698, 0.0742296918767507,
    0.0861344537815126, 0.08473389355742297, 0.07913165266106442,
])
G = 9.81
P_1_state_idx = [str(s) for s in sm.state].index("P_1")

fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
if result_ncp is not None:
    solver, xc, b_vals = result_ncp
    Q = solver._sim_Q
    eta = Q[0] + Q[b_state_idx]
    pb  = Q[0] + 2.0 * Q[P_1_state_idx] / G
    eta_at_0 = eta[len(eta) // 2]
    pb_at_0  = pb[len(pb) // 2]
    ax[0].plot(xc, eta, color="C2", lw=1.5,
               label=f"DG(0) NCP, η(x=0)={eta_at_0:.4f}")
    ax[1].plot(xc, pb, color="C2", lw=1.5,
               label=f"DG(0) NCP, p_b/g(x=0)={pb_at_0:.4f}")
    ax[0].plot(xc, b_vals, "k-", lw=1.0, alpha=0.4, label="bathymetry")
ax[0].plot(ETA_EXP_X, ETA_EXP_Y, "ko", ms=4, label="experiment")
ax[1].plot(PB_EXP_X, PB_EXP_Y, "ko", ms=4, label="experiment")
ax[0].set_ylabel("free surface η  [m]")
ax[0].set_title("dam-break over bump — 8-state chain, DG(0) NCP")
ax[0].legend(fontsize=9); ax[0].grid(True, alpha=0.3)
ax[1].set_xlabel("x  [m]")
ax[1].set_ylabel(r"bottom pressure  $p_b/g$  [m]")
ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dam_break_8state_ncp.png", dpi=100, bbox_inches="tight")
print("\nsaved dam_break_8state_ncp.png")

# %% [markdown]
# ## What this notebook validates
#
# * **NCP-only Riemann route works on the chain-derived 8-state SM**.
#   The well-balancing comes from the symbolic identity
#   ``Id[h, b] = 1`` in the fluctuation dissipation plus the chain's
#   pre-HR NCP entries ``B[xmom_j0, {h, b}, x] = g·h`` carried through
#   the path-integral.  No Audusse face-state reconstruction is
#   needed.
# * **Polymorphism**: the only solver-side change vs the HR notebook
#   is the ``riemann_solver="ncp"`` flag.  The inline
#   ``_NCPRusanovLARBalanced`` subclass lives entirely in
#   :meth:`ChorinSplitVAMSolver._build_numerics`; no top-level
#   ``riemann_solvers.py`` change.
# * **Companion to the HR baseline** — once both this notebook and
#   ``vam_chorin_bump_8state_chain.py`` produce visually acceptable
#   solutions, the validation integration test pins the L∞ tolerance
#   and the higher-order ``PrimitiveReconstruction`` wrapper can be
#   debugged against a locked baseline.
