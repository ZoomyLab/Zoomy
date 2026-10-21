"""HLL on a dry bed: the face speed of a dry-dry face is +0, never -0 (ANALYSIS 64).

`face_wave_speed` = `max(-min(s_L, 0), max(s_R, 0))`.  At a dry-dry face `s_L = s_R = 0`, so
`-min(s_L, 0) = -0.0` and `max(s_R, 0) = +0.0`; sympy orders the two arguments
`(max(s_R, 0), -min(s_L, 0))`, and a maximum of `(+0, -0)` is backend-dependent: NumPy's
`maximum` and OpenFOAM's `Foam::max` (`(a > b) ? a : b`) return the second, `-0`; AMReX's
`amrex::max` (`std::max`, `(a < b) ? b : a`) returns the first, `+0`; XLA's max orders
`-0 < +0`.  `-0` makes the CFL step `min(cfl r / S) = -inf`, time `-inf`, and
`dt * (source - divergence)` `0 * inf = NaN` (the step-5 `speed` case: t = NaN on NumPy and
foam, finite on JAX and AMReX).  The `Abs` around the speed makes it `+0` on every backend.
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall
from zoomy_core.model.models import SWE
from zoomy_core.numerics import (NumericalSystemModel, PositiveHLL,
                                 PositiveNonconservativeHLL)
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]

TAGS = ("left", "right", "bottom", "top")


def _swe2d():
    bcs = BoundaryConditions([Wall(tag=t, momentum_field_indices=[[2, 3]],
                                   permeability=0.0, wall_slip=1.0) for t in TAGS])
    return SystemModel.from_model(SWE(dimension=2, parameters={"g": 9.81},
                                      boundary_conditions=bcs))


@pytest.mark.parametrize("riemann", [PositiveHLL, PositiveNonconservativeHLL])
def test_dry_dry_face_speed_is_plus_zero(riemann):
    """The face speed at a dry-dry face (b = h = hu = hv = 0 on both sides), printed by the
    NumPy printer, is +0 for every normal."""
    nsm = NumericalSystemModel.from_system_model(_swe2d(), riemann=riemann)
    num = nsm.riemann(nsm.sm)
    e = num.face_wave_speed()
    syms = sorted(e.free_symbols, key=str)
    f = sp.lambdify(syms, e, modules="numpy")
    for nx, ny in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.6, 0.8)):
        vals = {"g": 9.81, "n0": nx, "n1": ny}
        s = np.float64(f(*[vals.get(str(x), 0.0) for x in syms]))
        assert s == 0.0 and not np.signbit(s), (riemann.__name__, nx, ny, s)


@pytest.mark.parametrize("riemann", [PositiveHLL, PositiveNonconservativeHLL])
def test_hll_dam_break_onto_dry_bed_is_finite(riemann):
    """SWE 2-D, a circular dam onto an exactly dry bed in a closed basin, HLL on the
    hydrostatic reconstruction, order 1, NumPy: the march reaches t_end with a finite state,
    no invalid floating-point operation, mass kept."""
    sm = _swe2d()
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    nsm = NumericalSystemModel.from_system_model(sm, riemann=riemann)
    mesh = LSQMesh.create_2d((0.0, 10.0, 0.0, 10.0), 8, 8)
    s = SymbolicHyperbolicSolverNumpy(time_end=0.02, cfl=0.45, dt_write=0.01)
    Q0, A = s.setup(mesh, nsm)
    n = int(mesh.n_inner_cells)
    xc = np.asarray(mesh.cell_centers)[:, :n]
    Q0 = np.zeros((4, n))
    Q0[1] = np.where(np.hypot(xc[0] - 5.0, xc[1] - 5.0) < 2.0, 2.0, 0.0)
    A = s.initial_aux(Q0, A)
    with np.errstate(invalid="raise", divide="ignore"):
        t, Q, _ = s.run(Q0, A)
    Q = np.asarray(Q)[:, :n]
    assert t == pytest.approx(0.02) and np.isfinite(Q).all()
    assert Q[1].min() >= 0.0 and Q[1].min() == 0.0          # the bed is still dry somewhere
    vol = np.asarray(mesh.cell_volumes)[:n]
    assert abs((Q[1] - Q0[1]) @ vol) <= 1e-12 * (Q0[1] @ vol)
