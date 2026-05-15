"""``ChorinSplitVAMSolver`` — incremental wiring tests.

The solver consumes three sub-systems from
:func:`split_for_pressure`.  Tests here grow as the implementation
lands:

1. Constructor accepts three :class:`SystemModel` inputs and detects
   the ``dt`` Symbol baked into ``SM_press`` / ``SM_corr``.
2. ``setup_simulation`` builds three runtimes and pre-allocates
   ``Q`` / ``Qaux`` of the right shape.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.splitter import split_for_pressure


@pytest.fixture(scope="module")
def split_122():
    m = VAMModelGalerkin(level=1)
    sm = SystemModel.from_model(m)
    name_to_sym = {str(s): s for s in sm.state}
    dt = sp.Symbol(r"\Delta t", positive=True)
    return split_for_pressure(
        sm, [name_to_sym["P_0"], name_to_sym["P_1"]], dt,
    )


def test_chorin_solver_constructor_takes_three_subsystems(split_122):
    """Constructor accepts three SystemModels; `time_end` & friends
    are class-level params overridable via kwargs."""
    from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        time_end=0.2, reconstruction_order=2,
    )
    assert solver.sm_pred is split_122.SM_pred
    assert solver.sm_press is split_122.SM_press
    assert solver.sm_corr is split_122.SM_corr
    assert solver.time_end == 0.2
    assert solver.reconstruction_order == 2
    # The 7-state must be shared across all three sub-systems.
    assert solver.n_state == 7
    assert [str(s) for s in solver.state] == [
        "h", "U_0", "U_1", "W_0", "W_1", "P_0", "P_1"
    ]


def test_chorin_solver_detects_dt_symbol(split_122):
    """The dt Symbol baked into SM_press/SM_corr by the splitter must
    be auto-detected — we cannot bind it numerically at step time
    without knowing which Symbol to bind."""
    from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
    )
    assert solver._dt_symbol is not None
    assert str(solver._dt_symbol) in {"dt", "Delta t", r"\Delta t"}


def test_chorin_solver_constructor_rejects_non_systemmodel(split_122):
    """Type guard: the three positional args must each be a
    :class:`SystemModel`."""
    from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

    with pytest.raises(TypeError, match="sm_pred"):
        ChorinSplitVAMSolver(
            "not a SystemModel",
            split_122.SM_press, split_122.SM_corr,
        )
    with pytest.raises(TypeError, match="sm_press"):
        ChorinSplitVAMSolver(
            split_122.SM_pred, None, split_122.SM_corr,
        )


def test_chorin_solver_setup_simulation_builds_runtimes(split_122):
    """``setup_simulation`` builds three runtimes and a Q array
    whose shape matches the 7-state chain on the mesh."""
    from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

    mesh = BaseMesh.create_1d(domain=(0.0, 4.0), n_inner_cells=16)
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
    )
    Q = solver.setup_simulation(mesh)

    assert Q.shape == (7, 16)
    assert solver.nc == 16
    # Three runtimes built.
    assert hasattr(solver, "rt_pred")
    assert hasattr(solver, "rt_press")
    assert hasattr(solver, "rt_corr")
    # State-index slices match the sub-systems' equation_to_state_index.
    assert solver._pred_state_idx.tolist() == [0, 1, 2, 3, 4]
    assert solver._press_state_idx.tolist() == [5, 6]
    assert solver._corr_state_idx.tolist() == [1, 2, 3, 4]


def test_chorin_solver_step_runs_without_crash(split_122):
    """One ``step()`` exercises predictor → pressure → corrector
    end-to-end on a non-trivial initial state.  Smoke check that the
    three substeps fire in sequence and ``Q`` advances (state changes,
    no crash, no NaN).  Order-of-accuracy verification comes later."""
    from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver

    mesh = BaseMesh.create_1d(domain=(0.0, 4.0), n_inner_cells=16)
    solver = ChorinSplitVAMSolver(
        split_122.SM_pred, split_122.SM_press, split_122.SM_corr,
        reconstruction_order=1,
    )
    Q0 = solver.setup_simulation(mesh)
    nc = solver.nc
    x = solver._sim_mesh.cell_centers[0, :nc]
    # Non-trivial IC: h has a bump, U_0 has a gradient.  Drives both
    # mass advection (h transport) and pressure response.
    Q0[0, :] = 1.0 + 0.01 * np.cos(2 * np.pi * x / 4.0)
    Q0[1, :] = 0.1 * np.sin(2 * np.pi * x / 4.0)
    solver._sim_Q = Q0.copy()

    solver.step(0.01)
    Q1 = solver._sim_Q

    assert Q1.shape == Q0.shape, "step() must preserve Q shape"
    assert np.all(np.isfinite(Q1)), "step() produced non-finite values"
    # Predictor advected h non-trivially through ∂_x(h·U_0).
    assert np.max(np.abs(Q1[0, :] - Q0[0, :])) > 1e-6
    # Corrector wrote to U_0 (row 1) — at minimum the state_update
    # callable executed and assigned (delta may be small but non-zero
    # if pressure ≠ 0).
    assert np.all(np.isfinite(Q1[1, :]))
