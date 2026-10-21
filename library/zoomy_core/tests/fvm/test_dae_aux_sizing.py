"""``DAESolver`` must allocate Qaux against the SAME model its kernels compile from.

``NumpyRuntimeModel.from_system_model`` routes through ``to_numerical_system_model``,
and ``NumericalSystemModel.derive()`` REGISTERS AUX -- the Kurganov-Petrova
desingularised ``hinv``.  ``DAESolver.setup_simulation`` used to normalise its
input with ``_coerce_to_system_model``, which returns a bare SystemModel
unchanged, so every buffer was sized from a pre-NSM ``aux_state`` that is one
row shorter than the compiled kernels index.  The first ``mass_matrix`` call
then died with ``IndexError: index N is out of bounds for axis 0 with size N``
(steffler_1d, curvilinear double-moment DAE).

The invariant under test is general, not DAE-specific: after setup, the solver's
aux width == the runtime's aux width == ``solver.sm.aux_state`` width, and a
model carrying ``1/h`` gets its ``hinv`` slot.
"""
import numpy as np
import pytest

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_dae_numpy import DAESolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree)
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _sm():
    """A bare SystemModel (NOT promoted) whose flux carries ``1/h``."""
    return SystemModel.from_model(
        SME(level=1, dimension=2, parameters={"g": 9.81},
            closures=[Newtonian(), NavierSlip(), StressFree()]))


def test_dae_setup_aux_width_matches_runtime_kernels():
    sm = _sm()
    n_aux_pre = len(sm.aux_state)
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=12)
    solver = DAESolver(time_end=1e-3, method="ars232",
                       compute_dt=timestepping.adaptive(CFL=0.2))
    solver.setup_simulation(mesh, sm, write_output=False)

    aux_names = [str(a) for a in solver.sm.aux_state]
    # the promotion is what adds ``hinv`` -- a bare SystemModel does not have it
    assert "hinv" in aux_names, aux_names
    assert len(aux_names) > n_aux_pre
    # every buffer and the runtime agree on that width
    assert solver.n_aux_total == len(aux_names)
    assert solver._sim_Qaux.shape[0] == len(aux_names)
    assert solver.rt.n_aux_variables == len(aux_names)
    # and the compiled kernel actually accepts a Qaux of that width
    M = np.asarray(solver.rt.mass_matrix(
        solver._sim_Q[:, 0], solver._sim_Qaux[:, 0],
        solver._parameters_array()), dtype=float)
    assert np.isfinite(M).all()


def test_dae_hinv_tracks_the_depth_across_a_step():
    """The KP ``hinv`` must be REFRESHED, not frozen at its initial value.

    ``Solver._apply_local_aux_formula`` needs the LOWERED
    ``update_aux_variables`` and skips a non-callable one; on a SystemModel/NSM
    that slot is a symbolic ZArray.  ``DAESolver`` keeps the symbolic model in
    ``_sim_model`` (the base solver rebinds it to the runtime), so handing
    ``_sim_model`` to ``update_qaux`` silently skipped the whole local-aux leg:
    ``hinv`` stayed at ``1/h(t=0)`` while ``h`` evolved, and every source term
    carrying ``1/h`` was evaluated at the wrong depth.  ``hinv*h == 1`` is the
    invariant (away from the desingularisation floor, which a wet state is).
    """
    sm = _sm()
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=12)
    solver = DAESolver(time_end=1e-2, method="ars232",
                       compute_dt=timestepping.adaptive(CFL=0.2))
    solver.setup_simulation(mesh, sm, write_output=False)
    # SME state is [b, h, q_0, ...] -- resolve by NAME, never by position
    i_h = [str(s) for s in solver.sm.state].index("h")
    i_hinv = [str(a) for a in solver.sm.aux_state].index("hinv")

    # a non-uniform depth so a frozen hinv is detectable
    solver._sim_Q[i_h, :] = np.linspace(0.7, 1.3, solver.nc)
    solver.step(1e-4)
    h = solver._sim_Q[i_h, :solver.nc]
    hinv = solver._sim_Qaux[i_hinv, :solver.nc]
    assert np.allclose(hinv * h, 1.0, atol=1e-9), np.ptp(hinv * h)


def test_dae_aux_initial_conditions_get_the_full_width():
    """The aux IC writes the WHOLE Qaux (the ``Solver.initialize`` contract), and
    a mis-sized user IC raises instead of silently leaving zeros -- a prescribed
    bathymetry/curvature that never lands is a different problem, not a warning.
    """
    from zoomy_core.model.initial_conditions import UserFunction

    sm = _sm()
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=12)
    seen = {}

    solver = DAESolver(time_end=1e-3, method="ars232",
                       compute_dt=timestepping.adaptive(CFL=0.2))
    # width is only known after promotion, so probe it through the solver
    solver.setup_simulation(mesh, sm, write_output=False)
    n_aux = len(solver.sm.aux_state)

    def aux_ic(xv):
        seen["n"] = n_aux
        return np.full(n_aux, 0.25)

    sm2 = _sm()
    sm2.aux_initial_conditions = UserFunction(function=aux_ic)
    solver2 = DAESolver(time_end=1e-3, method="ars232",
                        compute_dt=timestepping.adaptive(CFL=0.2))
    solver2.setup_simulation(mesh, sm2, write_output=False)
    assert seen["n"] == n_aux
    assert solver2._sim_Qaux.shape[0] == n_aux

    def bad_aux_ic(xv):
        return np.zeros(n_aux - 1)

    sm3 = _sm()
    sm3.aux_initial_conditions = UserFunction(function=bad_aux_ic)
    solver3 = DAESolver(time_end=1e-3, method="ars232",
                        compute_dt=timestepping.adaptive(CFL=0.2))
    with pytest.raises(ValueError):
        solver3.setup_simulation(mesh, sm3, write_output=False)
