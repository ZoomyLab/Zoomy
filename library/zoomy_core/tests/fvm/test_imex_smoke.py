"""The ONE IMEX runtime small (spec v3 delta, §E1 ruling).

The IMEX solver otherwise loses ALL runtime coverage in the reduced suite
(retain_viscous march + imex_global both deleted).  One smoke: a real
``IMEXSolver`` step on SME(1) friction stays finite on BOTH source modes —
including ``source_mode='global'``, whose Newton-GMRES JVP path was completely
dead (TypeError on the first JVP) until REQ-144 threaded the numeric
parameters through.
"""
import numpy as np
import pytest

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_imex_numpy import IMEXSolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree)
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small, pytest.mark.gate]


class _IMEXGlobal(IMEXSolver):
    source_mode = "global"


def test_imex_smoke_default_and_global_source():
    """One real IMEX step each on the default and the 'global' implicit-source
    path (REQ-144): both finite."""
    mesh = BaseMesh.create_2d(domain=(0.0, 10.0, 0.0, 1.0), nx=8, ny=4)
    nsm = NumericalSystemModel.from_model(
        SME(level=1, dimension=2, parameters={"g": 9.81, "n_m": 0.02},
            closures=[Newtonian(), NavierSlip(), StressFree()]))
    for cls in (IMEXSolver, _IMEXGlobal):
        solver = cls(time_end=0.02, compute_dt=timestepping.adaptive(CFL=0.3))
        solver.setup_simulation(mesh, nsm, write_output=False)
        solver.step(1e-3)               # 'global' was TypeError on the 1st JVP
        assert np.isfinite(solver._sim_Q).all(), cls.__name__
