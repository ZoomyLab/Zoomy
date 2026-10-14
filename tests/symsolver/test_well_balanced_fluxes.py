"""Lake at rest for every (model, Riemann class) pair the symbolic solvers offer (NumPy backend).

Plain `SWE` keeps the hydrostatic pressure in its conservative `flux` (`hydrostatic_pressure = 0`),
the derived `SME(0)` in `hydrostatic_pressure`.  The hydrostatic reconstruction's consistency
source restores the flux at rest wherever the model keeps it (`Numerics.rest_flux_definition`),
so each well-balanced class is exact for both; a class whose face terms drop the bed-slope
product raises at setup; `NonconservativeRusanov` / `QuasilinearRusanov` are not well-balanced
by construction (Rusanov dissipation on the depth jump) and are consistent on both models.
"""
import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
import zoomy_core.numerics as N
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]

#: exact at a lake at rest (round-off)
WELL_BALANCED = ("PositiveRusanov", "PositiveHLL", "PositiveNonconservativeRusanov",
                 "PositiveNonconservativeHLL", "PositiveQuasilinearRusanov",
                 "WellBalancedNonconservativeRusanov", "NonconservativeRoe", "QuasilinearRoe")
#: their face terms do not discretize the bed-slope product: setup raises
DROPS_BED = ("Rusanov", "HLL", "HLLC")
#: consistent, not well-balanced by construction
NOT_WELL_BALANCED = ("NonconservativeRusanov", "QuasilinearRusanov")


def _model(kind, dim):
    tags = ("left", "right") if dim == 1 else ("left", "right", "bottom", "top")
    bcs = BoundaryConditions([FromModel(tag=t, definition="wall") for t in tags])
    if kind == "swe":
        return SystemModel.from_model(SWE(dimension=dim, boundary_conditions=bcs))
    return SystemModel.from_model(SME(level=0, dimension=dim + 1, boundary_conditions=bcs,
                                      closures=[Newtonian(), ManningFriction(), StressFree()]))


def _run(kind, dim, riemann, order, flow=0.0, nsteps=20):
    """`(names, Q0, Q)` after `nsteps` steps of 0.01: water at eta = 0.5 over a Gaussian bed
    bump (`flow`: a discharge and a surface hump, a moving state)."""
    sm = _model(kind, dim)
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        r2 = (x[0] - 5.0) ** 2 + ((x[1] - 5.0) ** 2 if dim == 2 else 0.0)
        b = 0.2 * np.exp(-r2 / 2.0)
        q[names.index("b")] = b
        q[names.index("h")] = 0.5 - b + flow * 0.5 * np.exp(-(x[0] - 3.0) ** 2)
        q[2] = flow
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=getattr(N, riemann),
        reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    mesh = (LSQMesh.create_1d(domain=(0, 10), n_inner_cells=40) if dim == 1
            else LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=10, ny=10))
    s = SymbolicHyperbolicSolverNumpy(time_end=nsteps * 0.01, dt=0.01, mood=False,
                                      cfl_check_every=0)
    Q0, A = s.setup(mesh, nsm)
    _, Q, _ = s.run(Q0, A)
    return names, Q0, np.asarray(Q)


def test_classification_is_exhaustive():
    """Every offered Riemann class has a lake-at-rest verdict here."""
    assert sorted(WELL_BALANCED + DROPS_BED + NOT_WELL_BALANCED) == sorted(N.RIEMANN)


@pytest.mark.parametrize("riemann", WELL_BALANCED)
@pytest.mark.parametrize("kind", ["swe", "sme"])
@pytest.mark.parametrize("order", [1, 2])
def test_lake_at_rest_1d(riemann, kind, order):
    """Pressure in the flux (SWE) or split (SME(0)): the lake at rest to round-off, O1 / O2.
    Before the fix SWE moved with PositiveRusanov / PositiveHLL / PositiveNonconservative* at
    O1 (|hu| 8e-2), PositiveRusanov / PositiveHLL at O2 on both models (no in-cell bed term)
    and PositiveQuasilinearRusanov on SME(0) (the pressure jump counted twice)."""
    names, Q0, Q = _run(kind, 1, riemann, order)
    h, b = names.index("h"), names.index("b")
    assert np.abs(Q[2]).max() <= 1e-13
    assert np.abs(Q[h] + Q[b] - 0.5).max() <= 1e-13


@pytest.mark.parametrize("riemann", ["PositiveHLL", "PositiveRusanov",
                                     "PositiveNonconservativeHLL", "PositiveQuasilinearRusanov"])
def test_lake_at_rest_2d_swe(riemann):
    """2-D plain SWE, O2: both momentum rows stay at rest."""
    names, Q0, Q = _run("swe", 2, riemann, 2)
    assert np.abs(Q[[names.index("hu"), names.index("hv")]]).max() <= 1e-13


@pytest.mark.parametrize("riemann", DROPS_BED)
@pytest.mark.parametrize("kind", ["swe", "sme"])
def test_dropped_bed_slope_raises(riemann, kind):
    """A conservative flux on a model with a bed slope raises at setup, naming the flux and
    the dropped column."""
    with pytest.raises(ValueError, match=rf"riemann={riemann} .*columns \['b'\]"):
        _run(kind, 1, riemann, 1, nsteps=1)


@pytest.mark.parametrize("riemann", ["PositiveHLL", "PositiveRusanov"])
def test_moment_coupling_dropped_raises(riemann):
    """`PositiveRusanov` / `PositiveHLL` discretize the bed column only: on SME(1), whose
    nonconservative matrix couples the moments, they raise."""
    sm = SystemModel.from_model(SME(
        level=1, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")])))
    nsm = NumericalSystemModel.from_system_model(sm, riemann=getattr(N, riemann))
    s = SymbolicHyperbolicSolverNumpy(time_end=0.01, dt=0.01, mood=False)
    with pytest.raises(ValueError, match=rf"riemann={riemann} "):
        s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=10), nsm)


@pytest.mark.parametrize("riemann", NOT_WELL_BALANCED)
def test_not_well_balanced_by_construction_runs(riemann):
    """The non-well-balanced path-conservative Rusanov runs and treats both pressure
    placements alike (they differ only by SME(0)'s Gelfand wave-speed bound)."""
    _, _, a = _run("swe", 1, riemann, 1)
    _, _, b = _run("sme", 1, riemann, 1)
    assert 1e-4 < np.abs(a[2]).max() < 1e-2
    assert np.abs(a - b).max() <= 1e-2 * np.abs(a[2]).max()


@pytest.mark.parametrize("riemann", ["PositiveHLL", "PositiveNonconservativeRusanov",
                                     "PositiveQuasilinearRusanov", "NonconservativeRoe"])
@pytest.mark.parametrize("order", [1, 2])
def test_moving_flow_pressure_placement(riemann, order):
    """A flow over the bump: SWE (pressure in F) = SME(0) (pressure in P).  HLL (Davis speeds
    of the spectrum, a bound of it for SME(0)) and the exact Roe matrix agree to round-off;
    the Rusanov family up to its dissipation speed (SME(0) has no closed-form spectrum: the
    Gelfand bound, 1.7e-6 / 3.2e-6).  Before the fix PositiveHLL differed by O(0.1)."""
    _, _, a = _run("swe", 1, riemann, order, flow=0.1, nsteps=40)
    _, _, b = _run("sme", 1, riemann, order, flow=0.1, nsteps=40)
    bound = 1e-14 if riemann == "NonconservativeRoe" else (
        1e-12 if "HLL" in riemann else 1e-5)
    assert np.abs(a - b).max() <= bound
