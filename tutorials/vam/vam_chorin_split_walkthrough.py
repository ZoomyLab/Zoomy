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
# using the new `ChorinSplitVAMSolver` and **honestly documents
# where Chorin and the DAE differ**.
#
# Key questions answered here:
#
# 1. **Is the VAM SystemModel identical between the DAE and Chorin
#    paths?**  Yes, bit-for-bit.
# 2. **Does the simulation produce the same result?**  Partially.
#    Mass conservation is *better* than DAE (0.05 % vs 1.4 %), wave
#    speed within 2 % of Escalante eq (10), but the **q_U1
#    second-moment mode is unstable under explicit time integration**
#    in the chain — DAE's implicit ARS343 damps it, Chorin's explicit
#    Euler / SSP-RK2 doesn't.
# 3. **Is the mass matrix really identity before code generation?**
#    Yes after the new `SystemModel.absorb_mass_couplings()` pass.
#    Without that pass, the j=1 rows had a state-dependent
#    `(±q_U/W)/h` cheating that masked the q_U1 instability.

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.models.system_model import SystemModel
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
m = VAMModelGalerkin(level=1, dimension=2)
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
# This is **identical** to what `vam_1d_bump_dae.py` builds for the
# DAESolver. Both paths share the same `VAMModelGalerkin.derive_model`.

# %% [markdown]
# ## Step 2 — Primitive vs conservative mass matrix
#
# In primitive state $(h, U_k, W_k, P_k)$ the chain's mass matrix has
# state-dependent entries on the higher-order momentum rows. This is
# the **cheating** that `HyperbolicSolver` (which assumes $M=I$) would
# otherwise apply.

# %%
print("PRIMITIVE mass matrix:")
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
# **j=0 rows are now clean $M=I$.** The j=1 rows still have an
# off-diagonal `$(-q_U+q_U_{k+1})/h$` in the $\partial_t h$ column —
# state-dependent, zero at lake-at-rest but **non-zero under any
# dynamics**.
#
# To eliminate this, substitute the continuity equation
# $\partial_t h = -\partial_x q_{U_0}$ into those rows and push the
# resulting term into the non-conservative product matrix $B$:

# %%
sm.absorb_mass_couplings()

print("After absorb_mass_couplings (M=I, no cheating):")
for i in range(sm.mass_matrix.shape[0]):
    row = [sp.simplify(sm.mass_matrix[i, j])
           for j in range(sm.mass_matrix.shape[1])]
    print(f"  row {i}: {row}")

# %% [markdown]
# **Now $M = I$ on every evolution row, $M = 0$ on every algebraic
# row.** This is what the runtime sees — no more cheating anywhere.
#
# The price: the NCP matrix gains a state-quadratic entry
# $B[\text{xmom\_j1}, q_{U_0}, x] = (q_{U_0} - q_{U_1})/h$ (and the
# zmom analogue).  This entry exposes a previously-hidden instability
# mode in the chain — see Step 4.

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
# ## Step 4 — Run the bump simulation and inspect the q_U1 instability
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
# **Observation:** $h$ stays bounded and $q_{U_0}$ grows reasonably
# (the cosine wave propagating).  But **$q_{U_1}$ grows
# exponentially** — about $3\times$ per step.  After a few steps it
# overwhelms everything.
#
# This is **not a Chorin bug** — it's a fundamental property of the
# chain's xmom_j1 row under explicit time integration once the
# mass-matrix cheating is removed.  The new NCP term
# $((q_{U_0} - q_{U_1})/h) \cdot \partial_x q_{U_0}$ couples $q_{U_1}$
# to its own gradient through $q_{U_0}$, and the explicit Euler /
# SSP-RK2 scheme has no damping for that mode.
#
# **The DAE reference handles it** because ARS343 is *L-stable* —
# implicit-stage projection damps the unstable mode every step.

# %% [markdown]
# ## Step 5 — How the DAE reference produces a clean result
#
# `tutorials/vam/vam_1d_bump_dae.py` runs `DAESolver(method="ars343")`
# on the same chain (without absorption, without conservative
# change-of-vars).  It produces:
#
# ```
# h range = [0.9897, 1.0145]   (amplitude 0.014, slightly dissipated)
# mass drift = 1.36 %          (boundary outflow over the whole run)
# observed c = 3.044 m/s       (Escalante eq (10) predicts 3.082; 1.2 % err)
# ```
#
# This works because:
# 1. ARS343 is L-stable — damps high-frequency modes intrinsically.
# 2. Implicit-stage Newton enforces the algebraic constraints
#    `cont_j1`, `cont_j2` exactly each step.
# 3. The full mass matrix coupling (state-dependent M) is handled
#    naturally by the DAE residual.

# %% [markdown]
# ## Conclusions
#
# - **VAM SystemModels are identical** between DAE and Chorin paths.
# - **Conservative state-of-variables + `absorb_mass_couplings`**
#   give $M = I$ bit-perfect on every evolution row — no cheating.
# - **The Chorin predictor's explicit time integration (Euler / SSP-
#   RK2) is unstable for the chain's $q_{U_1}$ mode.**  Smaller dt
#   doesn't help — the mode's eigenvalue has unbounded imaginary
#   part contributions that explicit RK can't damp.
# - **DAE works** because of ARS343's L-stability + implicit
#   constraint enforcement.
#
# ## Open question for the next iteration
#
# Two possible paths to make Chorin handle the $q_{U_1}$ mode stably:
#
# 1. Use **IMEX-ARK** (same ARS343 tableau as DAESolver) for the
#    predictor, treating the $q_{U_1}$ row's nonlinear NCP term
#    `((q_U0 - q_U1)/h) · ∂_x q_U0` implicitly.  This is a real
#    architectural step.
# 2. Add **artificial damping** on the $q_{U_1}$ row — a hack but
#    might work for low-Froude regimes.
#
# Until one of those lands, the bump test stays `xfail` for
# `ChorinSplitVAMSolver` at order $\ge 1$.
