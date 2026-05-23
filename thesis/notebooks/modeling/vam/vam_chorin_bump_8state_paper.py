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
# # VAM(1,2,2) hand-built 8-state SystemModel — **paper-matching operators**
#
# Bit-for-bit matches the operators in the Escalante 2024 JCP paper /
# the JAX legacy `web/tutorials/vam/simple.ipynb`.  No
# `_eliminate_quadratic_W` substitution, no escalante-form
# `(∂_x Q)²` leak, no μ_k scaling in the state variables.
# State convention:  `q_U_k = h · u_k` directly (no μ_k weight).
#
# State (8):  ``[h, hu0, hu1, hw0, hw1, b, P_0, P_1]``.
# Mass matrix: ``I`` on evolution rows (0..5), ``0`` on cont rows (6, 7).
#
# Operators (ρ ≡ 1 — kinematic-pressure convention, matching legacy):
#
# **Flux F[i]** (depth-integrated convective):
#   F[0] = hu0                                          (mass)
#   F[1] = hu0²/h + hu1²/(3h)                           (xmom_j0)
#   F[2] = 2·hu0·hu1/h                                  (xmom_j1)
#   F[3] = (hu0·hw0)/h + (hu1·hw1)/(3h)                 (zmom_j0)
#   F[4] = (hu0·hw1 + hu1·hw0)/h + (2/5)·(hu1·hw2)/h    (zmom_j1)
#   F[5] = 0                                            (b: static)
#   F[6] = F[7] = 0                                     (cont rows)
#
# **NCP B[i, j]**:
#   B[xmom_j0, h, x]  = g·h          (gravity column h)
#   B[xmom_j0, b, x]  = g·h          (gravity column b)
#   B[xmom_j1, hw0, x] = -hu0/h      (mode-coupling, matches legacy nc[2,3])
#   B[zmom_j1, hu1, x] = (hw2 - 5·hw0)/(5h)  (closure, matches legacy nc[4,2])
#   B[cont_j1, hu0, x] = 1
#   B[cont_j1, hu1, x] = 1/3
#   B[cont_j1, h, x]   = (hu1 − 3·hu0)/(3h)
#   B[cont_j1, b, x]   = -2·hu0/h
#   B[cont_j2, hu0, x] = 1
#   B[cont_j2, h, x]   = (hu1 − hu0)/h
#   B[cont_j2, b, x]   = 2·hu1/h
#
# **Source S[i]** (with SystemModel convention residual = ... − S; legacy
# returns -R from source(), so SystemModel S = -R_legacy):
#   S[mass]    = 0
#   S[xmom_j0] = -dhp0dx − 2·P_1·b_x
#   S[xmom_j1] = -dhp1dx + (3·P_0 − P_1)·h_x + 6·(P_0 − P_1)·b_x
#   S[zmom_j0] = 2·P_1
#   S[zmom_j1] = -6·(P_0 − P_1) = -6·P_0 + 6·P_1
#   S[b]       = 0
#   S[cont_j1] = -2·hw0/h                              (gives residual +2·hw0/h)
#   S[cont_j2] = +2·hw1/h                              (gives residual -2·hw1/h)
#
# Where `dhp0dx = ∂_x(h·P_0)`, `dhp1dx = ∂_x(h·P_1)`, `h_x = ∂_x h`,
# `b_x = ∂_x b`, `hw2` is the closure aux computed in `update_qaux`.

# %%
import sys
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
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.splitter import split_simple
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

# %% [markdown]
# ## State / aux / parameters

# %%
t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
h = sp.Symbol("h", positive=True, real=True)
hu0 = sp.Symbol("q_U0", real=True)
hu1 = sp.Symbol("q_U1", real=True)
hw0 = sp.Symbol("q_W0", real=True)
hw1 = sp.Symbol("q_W1", real=True)
b = sp.Symbol("b", real=True)
P_0 = sp.Symbol("P_0", real=True)
P_1 = sp.Symbol("P_1", real=True)
state = [h, hu0, hu1, hw0, hw1, b, P_0, P_1]
n_state = len(state)
n_eq = n_state

# Aux state (derivative + closure).
b_x = sp.Symbol("b_x", real=True)
h_x = sp.Symbol("h_x", real=True)
hu0_x = sp.Symbol("q_U0_x", real=True)
hu1_x = sp.Symbol("q_U1_x", real=True)
hw2_aux = sp.Symbol("hw2", real=True)
P_0_x = sp.Symbol("P_0_x", real=True)
P_1_x = sp.Symbol("P_1_x", real=True)
aux_state = [hw2_aux, b_x, h_x, hu0_x, hu1_x, P_0_x, P_1_x]

g = sp.Symbol("g", positive=True)
parameters = Zstruct(g=g)
parameter_values = Zstruct(g=9.81)

# %% [markdown]
# ## Operators

# %%
M = sp.eye(n_state)
M[6, 6] = 0           # cont_j1 is algebraic
M[7, 7] = 0           # cont_j2 is algebraic

F = sp.zeros(n_eq, 1)
F[0, 0] = hu0
# Pressure stays in SOURCE via product-rule-expanded ``∂_x(h·P_k)``
# = ``h·P_k_x + P_k·h_x``.  P_k_x are LSQ-refreshed aux entries, so
# the elliptic block's second-derivative-of-P structure is built
# correctly by the chain solver's expose_aux_atoms.
F[1, 0] = hu0**2 / h + hu1**2 / (3 * h)
F[2, 0] = 2 * hu0 * hu1 / h
F[3, 0] = (hu0 * hw0) / h + (hu1 * hw1) / (3 * h)
F[4, 0] = (hu0 * hw1 + hu1 * hw0) / h + (sp.Rational(2, 5)) * (hu1 * hw2_aux) / h
# F[5..7] = 0

P = sp.zeros(n_eq, 1)   # gravity is in NCP, not P_hydro

B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, 1)
# xmom_j0 (i=1): gravity via NCP on h, b columns.
B[1, 0, 0] = g * h
B[1, 5, 0] = g * h
# xmom_j1 (i=2): mode-coupling.  Legacy nc[2, 3] = -u0 → on hw0-column.
B[2, 3, 0] = -hu0 / h
# zmom_j1 (i=4): closure-coupling.  Legacy nc[4, 2] = 1/5 w2 - w0.
B[4, 2, 0] = hw2_aux / (5 * h) - hw0 / h
# cont_j1 (i=6): from C[0] = ∂_x hu0 + (1/3) ∂_x hu1 + ((hu1/3 - hu0)/h)·∂_x h - 2·(hu0/h)·∂_x b + 2·hw0/h
B[6, 1, 0] = sp.Integer(1)            # ∂_x hu0
B[6, 2, 0] = sp.Rational(1, 3)        # ∂_x hu1
B[6, 0, 0] = (hu1 - 3 * hu0) / (3 * h)  # ∂_x h
B[6, 5, 0] = -2 * hu0 / h             # ∂_x b
# cont_j2 (i=7): from C[1] = ∂_x hu0 + ((hu1 - hu0)/h)·∂_x h + 2·(hu1/h)·∂_x b - 2·hw1/h
B[7, 1, 0] = sp.Integer(1)
B[7, 0, 0] = (hu1 - hu0) / h
B[7, 5, 0] = 2 * hu1 / h

S = sp.zeros(n_eq, 1)
# Residual is ∂_t Q + ∂_x F + B·∂_x Q + ∂_x P_hydro − S.  Legacy residual
# in source-only RK1 form: ∂_t Q + ... + R = 0 where R = legacy R_legacy.
# So S = -R_legacy.
S[1, 0] = -(h * P_0_x + P_0 * h_x) - 2 * P_1 * b_x
S[2, 0] = -(h * P_1_x + P_1 * h_x) + (3 * P_0 - P_1) * h_x + 6 * (P_0 - P_1) * b_x
S[3, 0] = 2 * P_1
S[4, 0] = -6 * (P_0 - P_1)
S[6, 0] = -2 * hw0 / h                # gives residual +2·hw0/h
S[7, 0] = 2 * hw1 / h                 # gives residual -2·hw1/h

# %% [markdown]
# ## Build SystemModel + BCs

# %%
def build_paper_sm(bc_list):
    sm = SystemModel(
        time=t, space=[x], state=list(state), aux_state=list(aux_state),
        parameters=parameters, parameter_values=parameter_values,
        flux=F, hydrostatic_pressure=P, nonconservative_matrix=B,
        source=S, mass_matrix=M,
    )
    sm.equation_names = [
        "mass", "xmom_j0", "xmom_j1", "zmom_j0", "zmom_j1",
        "b_eq", "cont_j1", "cont_j2",
    ]
    sm.aux_registry = [
        # hw2 closure — set live by solver hook.
        {"kind": "function", "name": "hw2", "row": 0,
         "atom": sp.Function("hw2", real=True)(t, x), "aux_symbol": hw2_aux},
        # Derivative aux entries — LSQ-refreshed via mesh.compute_derivatives.
        {"kind": "derivative", "name": "b_x", "row": 1,
         "atom": sp.Derivative(b, x, evaluate=False), "aux_symbol": b_x,
         "target_name": "b", "multi_index": (1,),
         "target_kind": "state", "state_index": 5},
        {"kind": "derivative", "name": "h_x", "row": 2,
         "atom": sp.Derivative(h, x, evaluate=False), "aux_symbol": h_x,
         "target_name": "h", "multi_index": (1,),
         "target_kind": "state", "state_index": 0},
        {"kind": "derivative", "name": "q_U0_x", "row": 3,
         "atom": sp.Derivative(hu0, x, evaluate=False), "aux_symbol": hu0_x,
         "target_name": "q_U0", "multi_index": (1,),
         "target_kind": "state", "state_index": 1},
        {"kind": "derivative", "name": "q_U1_x", "row": 4,
         "atom": sp.Derivative(hu1, x, evaluate=False), "aux_symbol": hu1_x,
         "target_name": "q_U1", "multi_index": (1,),
         "target_kind": "state", "state_index": 2},
        {"kind": "derivative", "name": "P_0_x", "row": 5,
         "atom": sp.Derivative(P_0, x, evaluate=False), "aux_symbol": P_0_x,
         "target_name": "P_0", "multi_index": (1,),
         "target_kind": "state", "state_index": 6},
        {"kind": "derivative", "name": "P_1_x", "row": 6,
         "atom": sp.Derivative(P_1, x, evaluate=False), "aux_symbol": P_1_x,
         "target_name": "P_1", "multi_index": (1,),
         "target_kind": "state", "state_index": 7},
    ]

    position_zs = Zstruct(x=x); position_zs._symbolic_name = "X"
    dX = sp.Symbol("dX", positive=True)
    variables_zs = Zstruct(h=h, q_U0=hu0, q_U1=hu1, q_W0=hw0, q_W1=hw1,
                            b=b, P_0=P_0, P_1=P_1)
    variables_zs._symbolic_name = "Q"
    aux_variables_zs = Zstruct(hw2=hw2_aux, b_x=b_x, h_x=h_x,
                                q_U0_x=hu0_x, q_U1_x=hu1_x,
                                P_0_x=P_0_x, P_1_x=P_1_x)
    aux_variables_zs._symbolic_name = "Qaux"
    normal_zs = Zstruct(n0=sp.Symbol("n0", real=True))
    normal_zs._symbolic_name = "n"

    sm.boundary_conditions = bc_list.get_boundary_condition_function(
        t, position_zs, dX, variables_zs, aux_variables_zs,
        parameters, normal_zs, function_name="boundary_conditions")
    sm.boundary_gradients = bc_list.get_boundary_gradient_function(
        t, position_zs, dX, variables_zs, aux_variables_zs,
        parameters, normal_zs, function_name="boundary_gradients")
    aux_bc_list = BoundaryConditions([
        AuxBC.Extrapolation(tag="left"),
        AuxBC.Extrapolation(tag="right"),
    ])
    sm.aux_boundary_conditions = aux_bc_list.get_boundary_condition_function(
        t, position_zs, dX, variables_zs, aux_variables_zs,
        parameters, normal_zs, function_name="aux_boundary_conditions")
    sm.initial_conditions = UserFunction(function=lambda x: np.zeros(n_state))
    sm.aux_initial_conditions = UserFunction(
        function=lambda x: np.zeros(len(aux_state)))
    sm.eigenvalues = None
    return sm


# %% [markdown]
# ## Dam-break-over-bump test setup

# %%
b_inflow = 0.20 * np.exp(-(-1.5)**2 / (2 * 0.20**2))
bc_list = BoundaryConditions([
    Lambda(tag="left", prescribe_fields={
        1: lambda *a: 0.11197, 2: lambda *a: 0.0,
        3: lambda *a: 0.0, 4: lambda *a: 0.0,
        5: lambda *a: b_inflow,
    }),
    Extrapolation(tag="right"),
])

sm = build_paper_sm(bc_list)

# Pressure-mode state indices.
P_indices = [state.index(P_0), state.index(P_1)]
b_state_idx = state.index(b)
dt_sym = sp.Symbol("dt", positive=True)
split = split_simple(sm, [sm.state[i] for i in P_indices], dt_sym)


# %% [markdown]
# ## Run — DG(0) HR Chorin solver

# %%
def run(T_end=20.0, n_inner=60):
    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=n_inner)
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        riemann_solver="ncp", pressure_solver="gmres",
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
    print(f"--- paper-form HR-DG(0): cfl={cfl}, dt={dt:.4e}, n_steps={n_steps} ---")
    print(f"{'step':>5} {'t':>6} {'hmin':>7} {'hmax':>7} {'|hu0|':>10} {'|P0|':>10}")
    log_steps = sorted({1, 5, 20, 100, 500, 1000, 2000, n_steps})
    for k in range(n_steps):
        solver.step(dt)
        Q = solver._sim_Q
        if (k + 1) in log_steps:
            print(f"{k+1:>5} {solver._sim_time:>6.3f} {Q[0].min():>7.4f} "
                  f"{Q[0].max():>7.4f} {np.max(np.abs(Q[1])):>10.3e} "
                  f"{np.max(np.abs(Q[state.index(P_0)])):>10.3e}")
        if not np.all(np.isfinite(Q)):
            print(f"BLOWUP at step {k+1}")
            return None
    return solver, xc, b_vals


result = run()


# %% [markdown]
# ## Plot

# %%
if result is not None:
    solver, xc, b_vals = result
    Q = solver._sim_Q
    h_vals = Q[0]
    eta = h_vals + Q[b_state_idx]
    P_1_vals = Q[state.index(P_1)]
    pb_g = h_vals + 2.0 * P_1_vals / 9.81

    # Escalante 2024 experimental dots.
    eta_x = np.array([
        -0.5929, -0.5431, -0.4946, -0.4448, -0.3991, -0.3452, -0.2981,
        -0.2510, -0.1985, -0.1514, -0.1043, -0.0532, -0.0007, 0.0464,
        0.0976, 0.1474, 0.1985, 0.2456, 0.2981, 0.3479, 0.3964, 0.4462,
        0.4987, 0.5511,
    ])
    eta_y = np.array([
        0.3419, 0.3412, 0.3399, 0.3419, 0.3399, 0.3399, 0.3385, 0.3378,
        0.3338, 0.3277, 0.3230, 0.3149, 0.3054, 0.2905, 0.2689, 0.2426,
        0.2162, 0.1858, 0.1554, 0.1311, 0.1061, 0.0919, 0.0730, 0.0655,
    ])
    pb_x = np.array([
        -0.6001, -0.5556, -0.5042, -0.4485, -0.3999, -0.3540, -0.3039,
        -0.2497, -0.2051, -0.1523, -0.1008, -0.0535, -0.0007, 0.0522,
        0.0981, 0.1509, 0.2024, 0.2483, 0.3011, 0.3540, 0.4026, 0.4527,
        0.5014, 0.5515,
    ])
    pb_y = np.array([
        0.3319, 0.3305, 0.3221, 0.3095, 0.2906, 0.2745, 0.2500, 0.2234,
        0.1842, 0.1576, 0.1239, 0.0945, 0.0700, 0.0420, 0.0448, 0.0343,
        0.0483, 0.0588, 0.0602, 0.0693, 0.0742, 0.0861, 0.0847, 0.0791,
    ])

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(xc, eta, "C0-", lw=1.5, label=f"η = h+b (T={solver._sim_time:.2f})")
    ax[0].plot(xc, b_vals, "k-", lw=1.0, alpha=0.4, label="bathymetry")
    ax[0].plot(eta_x, eta_y, "ko", ms=4, label="experiment")
    ax[0].set_ylabel("η [m]"); ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[0].set_title("Paper-form 8-state SystemModel — chain Chorin HR DG(0)")
    ax[1].plot(xc, pb_g, "C0-", lw=1.5, label="p_b/g = h + 2P_1/g")
    ax[1].plot(pb_x, pb_y, "ko", ms=4, label="experiment")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("p_b/g [m]")
    ax[1].grid(True, alpha=0.3); ax[1].legend()
    plt.tight_layout()
    plt.savefig("dam_break_8state_paper.png", dpi=100, bbox_inches="tight")
    print("saved dam_break_8state_paper.png")
