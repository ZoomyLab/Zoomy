"""2D SWE smoke test for the JAX hyperbolic solver.

A Gaussian h-pulse on a flat-bed lake — the simplest 2D SWE problem
that exercises both spatial dimensions of the JAX solver pipeline
(flux, MUSCL reconstruction, BCs, RK2). No analytical reference; we
just assert that the simulation runs cleanly, conserves mass to good
precision (closed walls), and the field stays smooth (no NaN, no
positivity wreckage).

This mirrors the title-slide use case the user wants — a 2D SWE
simulation run headlessly to produce a frame sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from loguru import logger
logger.remove()

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

import models


# ── Test fixture ─────────────────────────────────────────────────────────────
# The derived SME(level=0) in two horizontal dimensions (state [b, h, q_0,
# q_1]), wall BCs declared on the Model and attached by
# ``SystemModel.from_model``.

DOMAIN = (0.0, 10.0, 0.0, 10.0)         # (x_min, x_max, y_min, y_max), 10 m × 10 m
NX, NY = 40, 40
H0 = 0.5                                  # background depth
H_PERT = 0.05                             # initial Gaussian peak
SIGMA = 1.0
T_END = 0.5


def _ic(x):
    """Gaussian h-pulse on a flat bed, zero initial velocity."""
    r2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
    return np.array([0.0, H0 + H_PERT * np.exp(-r2 / (2 * SIGMA ** 2)),
                     0.0, 0.0])


def _run(order: int) -> tuple[LSQMesh, np.ndarray, int]:
    mesh = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
    sm = SystemModel.from_model(models.swe(dimension=3, bc="wall"))
    sm.initial_conditions = IC.UserFunction(function=_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    solver = HyperbolicSolver(
        time_end=T_END,
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=2),
        march_log_interval=0,
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return mesh, np.asarray(Q), models.state_index(nsm, "h")


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.jax
def test_swe2d_o1_runs():
    """O1 SWE 2D: simulation completes, h is finite + positive."""
    mesh, Q, ih = _run(order=1)
    n = mesh.n_inner_cells
    h = Q[ih, :n]
    assert np.all(np.isfinite(Q[:, :n])), "Non-finite values in solution"
    assert np.all(h > 0), "h became non-positive somewhere"


@pytest.mark.small
@pytest.mark.jax
def test_swe2d_o2_runs_and_propagates():
    """O2 SWE 2D: simulation completes, mass conserved on closed walls, h spreads."""
    mesh, Q, ih = _run(order=2)
    n = mesh.n_inner_cells
    h = Q[ih, :n]
    assert np.all(np.isfinite(Q[:, :n])), "Non-finite values in O2 solution"
    assert np.all(h > 0), "h became non-positive somewhere"

    # Mass conservation: total volume = sum(h * cell_volume).
    cv = np.asarray(mesh.cell_volumes[:n])
    V_final = float(np.sum(h * cv))
    V_initial = H0 * float(np.sum(cv)) + H_PERT * (2 * np.pi * SIGMA ** 2)
    # We're inside a 10×10 basin so the Gaussian tail is ~exactly captured.
    # Closed walls → mass conserved up to the discrete-IC vs analytic-Gaussian
    # gap; 0.5% is generous.
    assert abs(V_final - V_initial) / V_initial < 5e-3, (
        f"Mass drift {(V_final - V_initial) / V_initial:.3%} > 0.5%"
    )

    # The pulse should have spread out — the *peak* h should drop relative
    # to the initial peak by more than 5% (it's a 2D dispersion problem).
    h0_peak = H0 + H_PERT
    assert h.max() < 0.95 * h0_peak, (
        f"O2 pulse didn't disperse: max h = {h.max():.4f}, initial peak {h0_peak:.4f}"
    )
