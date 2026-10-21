"""Smoke test of the NumPy backend of the symbolic solvers: the C1 bump of the cross-backend
matrix (SWE, walls, O2 minmod) at 20 x 20 marches without JAX and conserves the mass.
"""
import sys

import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def bump(nx=20):
    """`(mesh, nsm)`: a Gaussian surface bump at rest in a walled 10 x 10 box (matrix C1)."""
    tags = ("left", "right", "bottom", "top")
    model = SME(level=0, dimension=3, closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in tags]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0.0, 0.5 + 0.05 * np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0), 0.0, 0.0]))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


def test_c1_bump_numpy_conserves_mass():
    """C1 at 20 x 20 to t = 2.5: finite, at the end time, the wave has moved, mass to round-off,
    and JAX never imported."""
    mesh, nsm = bump()
    s = SymbolicHyperbolicSolverNumpy(time_end=2.5, cfl=0.9, mood=False)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    vol = np.asarray(s.mesh_np.cell_volumes)[:Q.shape[1]]
    h = [str(v) for v in nsm.state].index("h")
    mass0, mass = float(Q0[h] @ vol), float(Q[h] @ vol)
    assert t == pytest.approx(2.5, abs=1e-12)
    assert np.isfinite(Q).all()
    assert np.abs(Q[h] - Q0[h]).max() > 1e-3
    assert abs(mass - mass0) / mass0 <= 1e-12
    assert "jax" not in sys.modules


def test_restart_is_bitwise(tmp_path):
    """C1 at 20 x 20: 0 -> 2.5 with a checkpoint at 1.25 = 0 -> 1.25, then a restart -> 2.5, bit
    for bit; a misspelt restart file is named."""
    def run(**kw):
        mesh, nsm = bump()
        s = SymbolicHyperbolicSolverNumpy(cfl=0.9, mood=False, **kw)
        Q0, A = s.setup(mesh, nsm)
        return s.run(Q0, A)
    _, Q_full, _ = run(time_end=2.5, dt_checkpoint=1.25, checkpoint=tmp_path / "full.h5")
    run(time_end=1.25, checkpoint=tmp_path / "half.h5")
    t, Q, _ = run(time_end=2.5, restart_from=tmp_path / "half.h5")
    assert t == 2.5
    assert np.array_equal(Q, Q_full)
    with pytest.raises(FileNotFoundError, match="did you mean 'half.h5'"):
        run(time_end=2.5, restart_from=tmp_path / "hal.h5")
