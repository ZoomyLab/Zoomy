"""A periodic seam is an interior face: the run is invariant under moving the seam (NumPy backend).

The seam face's Riemann problem reads the partner cell's aux (`face_aux`), its CFL speed the
partner cell's (`face_wavespeed`), and the reconstruction's ghost there is the partner cell
with its aux (`boundary_states`).  `NonconservativeRusanov` reads the regularised `hinv` of
both cells in its face kernel (the `Positive*` classes rebuild it from the reconstructed
depth): before the fix the 2 seam cells' momentum differed by 1e-4 after one step.
"""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.settings import CflWaveSpeed
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]

L, NX, SHIFT = 10.0, 40, 7


def periodic_wave(level, order, shift, nx=NX, length=L):
    """`(mesh, nsm)`: SME(level) in 1-D, periodic, a smooth non-uniform flow; the physical
    coordinate is `x + shift` (the seam sits at physical `shift`)."""
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left")]
    sm = SystemModel.from_model(SME(level=level, dimension=2, boundary_conditions=BC.BoundaryConditions(bcs),
                                    closures=[Newtonian(), ManningFriction(), StressFree()]))
    names = [str(s) for s in sm.state]

    def ic(x):
        xs = 2 * np.pi * ((x[0] + shift) % L) / L
        q = np.zeros(len(names))
        h = 0.5 + 0.2 * np.sin(xs) + 0.05 * np.cos(3 * xs)
        q[names.index("h")] = h
        q[names.index("q_0")] = h * (0.3 + 0.1 * np.cos(xs))
        if level >= 1:
            q[names.index("q_1")] = 0.05 * h * np.sin(xs)
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=NonconservativeRusanov,
        reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_1d(domain=(0, length), n_inner_cells=nx), nsm


def _run(level, order, shift, **kw):
    mesh, nsm = periodic_wave(level, order, shift)
    s = SymbolicHyperbolicSolverNumpy(mood=False, **kw)
    Q0, A = s.setup(mesh, nsm)
    _, Q, _ = s.run(Q0, A)
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    return np.asarray(Q)[:, np.argsort((x + shift) % L)]


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("order", [1, 2])
def test_seam_shift_constant_dt(level, order):
    """20 steps of a constant dt: the seam moved by 7 cells changes nothing (round-off)."""
    kw = dict(time_end=0.2, dt=0.01, cfl_check_every=0)
    a = _run(level, order, 0.0, **kw)
    b = _run(level, order, SHIFT * L / NX, **kw)
    assert np.abs(a - b).max() <= 1e-14


@pytest.mark.parametrize("speed", [CflWaveSpeed.cells, CflWaveSpeed.faces])
def test_seam_shift_adaptive_dt(speed):
    """Adaptive CFL steps from the cells' or the faces' wave speeds: the seam face's speed
    is an interior face's (the partner cell's), so every step and the end state agree."""
    kw = dict(time_end=0.3, cfl=0.9, cfl_wave_speed=speed)
    a = _run(0, 2, 0.0, **kw)
    b = _run(0, 2, SHIFT * L / NX, **kw)
    assert np.abs(a - b).max() <= 1e-13
