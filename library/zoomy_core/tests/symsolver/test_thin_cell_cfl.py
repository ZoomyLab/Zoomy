"""The cells CFL rule bounds every speed that moves mass (ANALYSIS 49, item 2): the dt spectrum
`eigenvalues_cfl` is gated to 0 only where h <= 0 (no water: the Audusse face states carry no
mass), not below `wet_dry_eps` -- a thin cell moves mass with its own velocity, and its speed
enters the Rusanov dissipation of its faces.  Without a closed form it is taken of A_n with the
Riemann speeds' 1/max(1e-14, h), not the KP hinv.
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.discretization.time import CflWaveSpeed
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.operations import gate_eigenvalues_dry
from zoomy_core.systemmodel.system_model import SpectralRadius, SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _sm(closed_form=False):
    return SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), StressFree()], symbolic_spectrum=closed_form,
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")])))


@pytest.mark.parametrize("closed_form, eig", [(False, False), (False, True), (True, False)])
def test_thin_cell_speed_is_its_own(closed_form, eig):
    """A thin cell (h = 1e-9 < wet_dry_eps) at u = 300 m/s: the dt speed is |u| + sqrt(g h)
    (closed form) or the Gelfand bound of it (no closed form: at most 1.1 x, ANALYSIS 48; the
    near-defective pair u +- 1e-4 makes it 1.044 x here), a dry cell (h = 0) with a round-off
    q: 0."""
    sm = _sm(closed_form)
    if eig:                            # opt in: the numerical eigenvalues (ANALYSIS 61)
        sm.spectral_radius_method = SpectralRadius.eig
    nsm = NumericalSystemModel.from_system_model(sm)
    assert (nsm.eigenvalues is not None) == closed_form
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0)
    s.setup(LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=2), nsm)
    h = np.array([1e-9, 0.0])
    Q = np.array([[0.0, 0.0], h, [300.0 * 1e-9, 1e-20]])
    A = s.initial_aux(Q, np.zeros((len(nsm.aux_state), 2)))
    with np.errstate(all="ignore"):
        ev = np.abs(np.asarray(s.rt.eigenvalues_cfl(Q, A, s.parameters, np.array([[1.0]]))))
    speed = ev.reshape(-1, 2).max(0)
    exact = 300.0 + np.sqrt(9.81e-9)
    if closed_form or eig:
        assert speed[0] == pytest.approx(exact, rel=1e-9)
    else:
        assert exact <= speed[0] <= 1.1 * exact
    assert speed[1] == 0.0


def test_thin_film_stays_positive_under_the_cells_rule():
    """SME(0) 1-D, order 2 without limiter + MOOD, forward Euler, the cells rule: a 1 mm film at
    300 m/s onto a dry bed (200 cells) keeps h >= 0; the former gate (eps = 1e-8, kept as an
    explicit `gate_eigenvalues_dry(1e-8)`) gives a thin cell (h = 7e-9 m, u = 3e4 m/s) a zero
    speed, the step drains its neighbour by the dissipation and h < 0."""
    def h_min(extra):
        sm = _sm()
        sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
            [0.0, 1e-3 * (x[0] < 0.5), 0.3 * (x[0] < 0.5)]))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.none),
            extra_operations=extra)
        out = []
        s = SymbolicHyperbolicSolverNumpy(
            time_end=1.5e-4, cfl=0.4, mood=True, time_order=1,
            cfl_wave_speed=CflWaveSpeed.cells, dt_write=2.5e-6,
            on_write=lambda t, Q, A: out.append(float(Q[1].min())))
        Q0, A = s.setup(LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=200), nsm)
        with np.errstate(all="ignore"):
            s.run(Q0, A)
        return min(out)
    assert h_min([]) >= 0.0
    assert h_min([gate_eigenvalues_dry(1e-8)]) < -1e-4


def test_spectral_radius_method_is_typed():
    """`spectral_radius_method` takes `SpectralRadius`; a plain string raises TypeError."""
    sm = _sm()
    sm.spectral_radius_method = "eig"
    with pytest.raises(TypeError, match="SpectralRadius.eig"):
        NumericalSystemModel.from_system_model(sm)
