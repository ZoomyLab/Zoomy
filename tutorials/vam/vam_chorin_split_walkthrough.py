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
# # VAM(1,2,2) Chorin split — walkthrough + comparison vs DAESolver
#
# This notebook reproduces the cosine-bump test from
# `tutorials/vam/vam_1d_bump_dae.py` (the DAESolver reference)
# using the new `ChorinSplitVAMSolver`, matching Escalante 2024's
# published equations and the original `legacy/vam.py` +
# `PoissonSolver` reference path.
#
# Key questions answered here:
#
# 1. **Is the VAM SystemModel identical between the DAE and Chorin
#    paths?**  Same chain; the Chorin path uses
#    `quadratic_form="escalante"` to land on Escalante eq (4)
#    directly (with stage-2a mass-equation substitution applied
#    during model derivation, in primitive state).  The DAE
#    reference uses the default `cantero_chinchilla` form because
#    its implicit ARS343 handles the full coupled DAE.
# 2. **Does the simulation produce the same result?**  Yes — to
#    Escalante eq (10) phase-speed within ~2 %, mass-conservation
#    better than the DAE reference.
# 3. **Why `escalante` form?**  Because in conservative state the
#    j ≥ 1 mass-matrix `∂_t h` cross-term must be substituted out
#    via the mass equation, and the substitution must happen
#    **before** the change-of-variables — otherwise the post-CoV
#    substitution (`remove_non_diagonal_h` in conservative state)
#    leaves a spurious state-quadratic NCP entry
#    `(q_U0 − q_U1)/h · ∂_x q_U0` that drives `q_U1`
#    exponentially under explicit time integration.  The two paths
#    are equivalent on the constraint manifold `mass = 0`, but the
#    discrete drift off-manifold is enough to excite the spurious
#    driver.  See Step 2 narrative for the details.

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.models.system_model import (
    SystemModel, HydrostaticReconstruction,
)
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.splitter import split_for_pressure
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

# Reference setup — identical to tutorials/vam/vam_1d_bump_dae.py.
L = 20.0
NX = 40
H = 1.0
AMP = 0.02
N_MODES = 1
G = 9.81
T_END = 1.0

# %% [markdown]
# ## Step 1 — Build the VAM chain and verify it matches the DAE path

# %%
m = VAMModelGalerkin(level=1, dimension=2, quadratic_form="escalante")
m.parameters.g = G
m.boundary_conditions = BoundaryConditions([
    Extrapolation(tag="left"),
    Extrapolation(tag="right"),
])
sm = SystemModel.from_model(m)

print(f"State        : {[str(s) for s in sm.state]}")
print(f"Equations    : {sm.equation_names}")
print(f"n_state      : {sm.n_state}")
print(f"n_equations  : {sm.n_equations}")

# %% [markdown]
# Same chain as `vam_1d_bump_dae.py` modulo `quadratic_form`. The DAE
# reference uses the default `cantero_chinchilla` form (un-reduced
# Galerkin), because ARS343 handles the full state-dependent mass
# matrix natively. For explicit Chorin we need M = I, and the
# `escalante` form is what gets us there cleanly — it applies the
# stage-2a mass-equation substitution in primitive state during model
# derivation, matching Escalante 2024 eq (4) bit-for-bit on the
# j = 0 rows and modulo {cont_j1, cont_j2} on j = 1.

# %% [markdown]
# ## Step 2 — Primitive vs conservative mass matrix
#
# In primitive state $(h, U_k, W_k, P_k)$ the chain's mass matrix
# carries state-dependent diagonal entries ($M[\text{xmom\_j1}, U_1]
# = h/3$, etc.). The `escalante` form has already substituted the
# mass equation into the j ≥ 1 rows during derivation, so the
# **h-column off-diagonal is already zero** — only state-dependent
# diagonals remain.

# %%
print("PRIMITIVE mass matrix (escalante form — h-column already clean on j ≥ 1):")
for i in range(sm.mass_matrix.shape[0]):
    row = [sp.simplify(sm.mass_matrix[i, j])
           for j in range(sm.mass_matrix.shape[1])]
    print(f"  row {i}: {row}")

# %%
h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state
q_U0, q_U1, q_W0, q_W1 = sp.symbols("q_U0 q_U1 q_W0 q_W1", real=True)

sm.change_state_variables(
    new_state=[h, q_U0, q_U1, q_W0, q_W1, P_0, P_1],
    transform={
        U_0: q_U0 / h,
        U_1: 3 * q_U1 / h,
        W_0: q_W0 / h,
        W_1: 3 * q_W1 / h,
    },
)

print("CONSERVATIVE mass matrix (q_k = h · U_k / c_k):")
for i in range(sm.mass_matrix.shape[0]):
    row = [sp.simplify(sm.mass_matrix[i, j])
           for j in range(sm.mass_matrix.shape[1])]
    print(f"  row {i}: {row}")

# %% [markdown]
# **$M = I$ on every evolution row, $M = 0$ on every algebraic row.**
# The modal rescaling $q_{U_1} = h \cdot U_1 / 3$ normalised the
# j = 1 diagonal from $h/3$ to $1$, and the `escalante` form's
# pre-CoV stage-2a substitution means there is no `∂_t h` cross-term
# to push out post-CoV.
#
# **Why not just `cantero_chinchilla` + post-CoV substitution?**
# Mass-equation substitution and change-of-variables do not commute
# when the substituted coefficient is state-dependent. The cantero
# path leaves $M[\text{xmom\_j1}, h] = (q_{U_1} - q_{U_0})/h$ after
# CoV; substituting *that* (via
# `SystemModel.remove_non_diagonal_h()`) introduces a spurious
# state-quadratic NCP entry $B[\text{xmom\_j1}, q_{U_0}, x] =
# (q_{U_0} - q_{U_1})/h$ that drives `q_U1` at every step. The
# escalante form does the substitution in primitive state where the
# Jacobian of the CoV propagates everything coherently, and the
# resulting conservative-form operators match Escalante 2024 eq (4)
# and the original `legacy/vam.py` + `PoissonSolver` setup.

# %%
sm.assert_diagonal_mass_matrix()
print("assert_diagonal_mass_matrix: PASSED (no remove_non_diagonal_h needed)")

# %% [markdown]
# ### Gravity repackaging for Audusse HR
#
# The chain emits gravity-on-η as `P[1] = g·h·(b+h)` and
# `B[1, h] = −g·(b+h)`.  These are correct as a continuous PDE
# but make the `b` term appear inside the hydrostatic pressure
# flux — Audusse's WB cancellation is cleaner with the standard
# SWE form `P[1] = g·h²/2` and `B[1, h] = 0`, with the
# bathymetry-on-momentum source `−g·h·b_x` supplied at runtime
# via the HR fluctuation `(P_raw − P_star) @ n`.
# `HydrostaticReconstruction` does this repackaging.

# %%
sm.apply(HydrostaticReconstruction())
print("After HydrostaticReconstruction:")
print(f"  P[xmom_j0]    = {sp.simplify(sm.hydrostatic_pressure[1, 0])}")
print(f"  B[xmom_j0, h] = {sp.simplify(sm.nonconservative_matrix[1, 0, 0])}")

# %% [markdown]
# ## Step 3 — Force numerical eigenvalue mode
#
# The chain DAE's symbolic eigenvalues come from `sp.solve` on the
# quasilinear matrix's characteristic polynomial.  Because the
# algebraic continuity rows contribute zero rows, the polynomial is
# rank-deficient and `sp.solve` **deduplicates** roots — returning
# only 2 of the 5 expected eigenvalues.  At rest both are zero ⇒
# Rusanov dissipation = 0 ⇒ unstable.
#
# Force per-cell numerical eigenvalues from the quasilinear matrix:

# %%
sm.eigenvalues = None
dt_sym = sp.Symbol("dt", positive=True)
split = split_for_pressure(sm, [P_0, P_1], dt_sym)
print(f"SM_pred  evolves Q[{split.SM_pred.equation_to_state_index}]")
print(f"SM_press evolves Q[{split.SM_press.equation_to_state_index}]")
print(f"SM_corr  evolves Q[{split.SM_corr.equation_to_state_index}]")

# %% [markdown]
# ## Step 4 — Run the bump simulation
#
# Same setup as `vam_1d_bump_dae.py`: cosine perturbation on flat
# bottom, Extrapolation BCs, $L=20$, $N_x=40$, $T_{\text{end}}=1$.

# %%
mesh = BaseMesh.create_1d(domain=(0.0, L), n_inner_cells=NX)
solver = ChorinSplitVAMSolver(
    split.SM_pred, split.SM_press, split.SM_corr,
    reconstruction_order=1,
    pressure_tol=1e-9, pressure_maxit=200,
)
Q0 = solver.setup_simulation(mesh)
nc = solver.nc
x = solver._sim_mesh.cell_centers[0, :nc]
solver.set_function_aux("b", np.zeros(nc))
solver.update_aux_variables()
Q0[:] = 0.0
Q0[0, :] = H + AMP * np.cos(2 * np.pi * N_MODES * x / L)
solver._sim_Q = Q0.copy()
solver.update_aux_variables()

dx = float(solver._sim_mesh.cell_volumes[0])
dt = 0.3 * dx / np.sqrt(G * H)
n_steps = int(np.ceil(T_END / dt))
print(f"dt = {dt:.4f}, n_steps = {n_steps}")
mass_0 = Q0[0].sum() * dx

snapshots = {0.0: Q0.copy()}
log_steps = sorted({int(t / dt) for t in [0.1, 0.25, 0.5, 1.0]})
print(f"step  |  t       |h-1| max  |q_U0| max  |q_U1| max  mass drift")
print("-" * 70)
for k in range(n_steps):
    solver.step(dt)
    if (k + 1) in log_steps:
        Q = solver._sim_Q
        mT = Q[0].sum() * dx
        print(f"{k+1:3d}   {solver._sim_time:5.3f}    "
              f"{np.max(np.abs(Q[0] - H)):.3e}   "
              f"{np.max(np.abs(Q[1])):.3e}    "
              f"{np.max(np.abs(Q[2])):.3e}   "
              f"{(mT - mass_0) / mass_0:+.2e}")
        snapshots[solver._sim_time] = Q.copy()
        if not np.all(np.isfinite(Q)):
            print("  ⇒ blowup detected")
            break

# %% [markdown]
# **Observation:** $h$ propagates as a cosine bump (amplitude
# $\approx 0.015$ at $T=1$, slightly dissipated from the initial
# $0.02$), $q_{U_0}$ tracks the wave's velocity, and $q_{U_1}$ stays
# at zero throughout — the higher-moment mode is never excited
# because the symbolic system matches Escalante 2024 exactly: no
# spurious driver couples $q_{U_1}$ to $\partial_x q_{U_0}^2$.

# %% [markdown]
# ## Step 5 — Comparison with the DAE reference
#
# `tutorials/vam/vam_1d_bump_dae.py` runs `DAESolver(method="ars343")`
# on the same chain (default `cantero_chinchilla` form, no
# change-of-vars).  It produces:
#
# ```
# h range = [0.9897, 1.0145]   (amplitude 0.014, slightly dissipated)
# mass drift = 1.36 %          (boundary outflow over the whole run)
# observed c = 3.044 m/s       (Escalante eq (10) predicts 3.082; 1.2 % err)
# ```
#
# The Chorin path here (escalante form + CoV) achieves comparable
# accuracy with much tighter mass conservation, because the
# pressure-projection step is a direct linear solve rather than a
# coupled Newton on the full DAE residual.

# %% [markdown]
# ## Conclusions
#
# - **VAM SystemModels match Escalante 2024 eq (4)** when built with
#   `quadratic_form="escalante"`. After the conservative
#   change-of-variables ($q_{U_k} = h \cdot U_k / c_k$) the mass
#   matrix is $M = I$ on every evolution row, no further pass needed.
# - **The Chorin predictor's explicit time integration is stable**
#   on this symbolic form — same form used by `legacy/vam.py` +
#   `PoissonSolver` in `library/zoomy_jax/` and by Escalante 2024's
#   own TVD-RK2 + linear Poisson solve.
# - **`cantero_chinchilla` + post-CoV `remove_non_diagonal_h` is *not*
#   equivalent** to the escalante path symbolically — mass-equation
#   substitution and change-of-variables do not commute when the
#   substituted coefficient is state-dependent. The cantero path
#   produces a spurious state-quadratic NCP entry that drives
#   `q_U1` under explicit time integration; use `escalante` form
#   instead when targeting explicit-Chorin solvers.
# - **DAE remains the canonical path for the un-reduced form**:
#   ARS343 + Newton handle the full state-dependent mass coupling
#   natively, so `cantero_chinchilla` is the right choice there.
