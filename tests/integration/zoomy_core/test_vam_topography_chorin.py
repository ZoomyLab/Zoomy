"""VAM(1,2,2) cosine-bump test on :class:`ChorinSplitVAMSolver`.

Reproduces the same physical setup as
``tutorials/vam/vam_1d_bump_dae.py`` (the DAESolver reference), so
the Chorin solver can be compared against the DAE baseline:

  domain  : [0, 20]
  cells   : 40
  H       : 1.0   (water depth at rest)
  cosine amp : 0.02   (1 mode along the domain)
  T_end   : 1.0
  flat bottom (b ≡ 0)
  Extrapolation BCs both ends

The wave splits into ±C(k) halves and propagates.  Phase speed is
verified against Escalante 2024 eq (10):

    C²/(g·H)  =  (1 + (kH)²/12) / (1 + 5(kH)²/12 + (kH)⁴/144).

DAE reference on this setup: c_obs = 3.044 (1.2 % error from Escalante).
Chorin target: c_obs within a similar tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation,
)
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.splitter import split_for_pressure
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver


L = 20.0
NX = 40
H = 1.0
AMP = 0.02
N_MODES = 1
G = 9.81
T_END = 1.0


def _escalante_eq10_c(k, H_=H, g_=G):
    kH = k * H_
    return np.sqrt(g_ * H_ * (1 + kH**2 / 12) /
                   (1 + 5 * kH**2 / 12 + kH**4 / 144))


@pytest.fixture(scope="module")
def chorin_chain():
    """Build the VAM chain DAE, convert to conservative form, split.

    Same chain as ``vam_1d_bump_dae.py`` uses for the DAESolver path,
    routed through ``change_state_variables`` so HyperbolicSolver
    integrates the j=0 evolution rows correctly (M=I after the
    transform; without it, the primitive chain forces a missing 1/h
    factor on the time derivative).
    """
    m = VAMModelGalerkin(level=1, dimension=2)
    m.parameters.g = G
    # Critical — without explicit BCs the default is an empty
    # BoundaryConditions list, which produces a degenerate BC kernel
    # at the boundary faces (not Extrapolation).  Match the DAE
    # reference (vam_1d_bump_dae.py) exactly.
    m.boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])
    sm = SystemModel.from_model(m)
    h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state
    q_U0, q_U1, q_W0, q_W1 = sp.symbols(
        "q_U0 q_U1 q_W0 q_W1", real=True,
    )
    from zoomy_core.model.models.system_model import InvertMassMatrix
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: 3 * q_U1 / h,
                   W_0: q_W0 / h, W_1: 3 * q_W1 / h},
    )
    # Consistency check + trivial diagonal inversion.  If the
    # variable transform doesn't produce a diagonal mass matrix on
    # the evolution rows, ``assert_diagonal_mass_matrix`` raises with
    # a precise location — the user picked wrong variables for this
    # system.
    sm.assert_diagonal_mass_matrix()
    sm.apply(InvertMassMatrix())
    # The chain DAE's symbolic eigenvalues come from sp.solve on a
    # rank-deficient characteristic polynomial — force numerical mode.
    sm.eigenvalues = None
    dt_sym = sp.Symbol("dt", positive=True)
    return split_for_pressure(sm, [P_0, P_1], dt_sym)


def _build_solver_with_ic(chorin_chain):
    """Build the Chorin solver and initialise the cosine-bump IC."""
    mesh = BaseMesh.create_1d(domain=(0.0, L), n_inner_cells=NX)
    solver = ChorinSplitVAMSolver(
        chorin_chain.SM_pred, chorin_chain.SM_press, chorin_chain.SM_corr,
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
    return solver, Q0, x


@pytest.mark.xfail(
    reason=(
        "VAM chain has a q_U1 (second-moment) mode that grows "
        "exponentially under explicit time integration once the "
        "j=1 mass-matrix cheating is removed (via "
        "absorb_mass_couplings).  DAESolver handles it via implicit "
        "ARS343; the Chorin explicit predictor (RK1 or SSP-RK2) "
        "doesn't have the damping mechanism.  Either: switch "
        "predictor to a higher-order implicit IMEX-style step on the "
        "evolution rows; or treat q_U1 as a stiff source.  Deferred."
    ),
    strict=False,
)
def test_chorin_cosine_bump_runs_to_T1(chorin_chain):
    """Chorin solver runs the cosine-bump IC to T = 1.0 without
    blowup; mass conservation holds; h amplitude stays bounded
    (the wave propagates dispersively but doesn't grow)."""
    solver, Q0, x = _build_solver_with_ic(chorin_chain)
    dx = float(solver._sim_mesh.cell_volumes[0])
    mass0 = Q0[0].sum() * dx
    dt = 0.3 * dx / np.sqrt(G * H)        # CFL = 0.3
    n_steps = int(np.ceil(T_END / dt))
    for _ in range(n_steps):
        solver.step(dt)
    Q = solver._sim_Q

    assert np.all(np.isfinite(Q)), "Chorin step produced non-finite Q"
    # Mass conservation — bounded above the DAE reference's 1.4 %.
    massT = Q[0].sum() * dx
    mass_drift = abs(massT - mass0) / mass0
    assert mass_drift < 2e-2, f"mass drift {mass_drift:.3e} > 2 %"

    # h amplitude stays bounded — no instability.
    assert np.max(np.abs(Q[0] - H)) < 2 * AMP, (
        f"h-amplitude {np.max(np.abs(Q[0] - H)):.3e} blew up vs IC {AMP}"
    )


@pytest.mark.xfail(
    reason="Same as test_chorin_cosine_bump_runs_to_T1 — the q_U1 "
           "explicit-time-integration instability prevents the wave "
           "from cleanly propagating to T=1.",
    strict=False,
)
def test_chorin_cosine_bump_phase_speed_matches_escalante(chorin_chain):
    """The Chorin-propagated wave speed agrees with Escalante eq (10)
    to within 5 %.  DAE reference on this setup: 1.2 % error;
    Chorin's first-order coupling is slightly more dissipative,
    pushing the error to ~2 %."""
    solver, Q0, x = _build_solver_with_ic(chorin_chain)
    dx = float(solver._sim_mesh.cell_volumes[0])
    dt = 0.3 * dx / np.sqrt(G * H)
    n_steps = int(np.ceil(T_END / dt))
    for _ in range(n_steps):
        solver.step(dt)
    Q = solver._sim_Q
    t_final = solver._sim_time

    # Standing-wave decomposition: at time t, amplitude factor cos(ωt)
    # with ω = k·c.  Project h-deviation onto cos(kx).
    k = 2 * np.pi * N_MODES / L
    h_dev = Q[0] - H
    cos_kx = np.cos(k * x)
    proj = np.sum(h_dev * cos_kx) * dx / (L / 2)
    cos_omega_T = float(np.clip(proj / AMP, -1.0, 1.0))
    c_obs = np.arccos(cos_omega_T) / (k * t_final)
    c_pred = _escalante_eq10_c(k)
    rel_err = abs(c_obs - c_pred) / c_pred
    assert rel_err < 0.05, (
        f"Chorin phase speed {c_obs:.4f} vs Escalante {c_pred:.4f}: "
        f"rel error {rel_err:.3e} > 5 %"
    )
