"""A lake at rest in a partially wet pit stays at rest at order 2 (NumPy backend).

The order-2 hydrostatic reconstruction is well balanced only if the reconstruction maps the
lake at rest to a flat free surface in every wet cell.  A dry bank above the lake carries
``eta = b`` into the stencil of the shore cells; the Venkatakrishnan limiter does not remove
that slope (Barth-Jespersen / minmod do: the shore cell is a local minimum of ``eta``), and the
lake accelerated from rest (0.40 m/s within 1 s here; the clamp-saturated pits of the Malpasset
thesis run).  The hydrostatic extension of the free surface in the stencil
(``reconstruction.shore_extension``) keeps it at rest to round-off for every limiter.
"""
import re

import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import (Limiter, NumericalSystemModel, PositiveNonconservativeHLL,
                                 ReconstructionSpec, SourceTreatment)
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]

TAGS = ("left", "right", "bottom", "top")


def pit(eta0, limiter, tilt=0.0, front_tol=0.5):
    """`(mesh, nsm)`: SME(0) at rest in a Gaussian pit `b = 1 - exp(-r^2/2) + tilt (x - 5)`
    filled to `eta0`, walls, the Malpasset thesis numerics (HLL + hydrostatic reconstruction,
    order 2 on (h, eta) with every row tied to eta, the front detector, the point-implicit
    source), 20 x 20 quads on [0, 10]^2."""
    model = SME(level=0, dimension=3, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions(
                    [FromModel(tag=t, definition="wall") for t in TAGS]))
    sm = SystemModel.from_model(model)

    def ic(x):
        b = 1.0 - np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.3) ** 2) / 2.0) + tilt * (x[0] - 5.0)
        return np.array([b, max(eta0 - b, 0.0), 0.0, 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)
    names = [str(s) for s in sm.state]
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=2, limiter=limiter, front_tol=front_tol,
                                          ties={n: "h" for n in names if n != "h"}),
        source_treatment=SourceTreatment.linearized)
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=20, ny=20), nsm


@pytest.mark.parametrize("limiter, eta0, tilt", [
    (Limiter.venkatakrishnan, 0.5, 0.0),      # partially wet: the shore cells see dry banks
    (Limiter.venkatakrishnan, 0.5, 0.05),     # an asymmetric pit
    (Limiter.minmod, 0.5, 0.0),
    (Limiter.venkatakrishnan, 2.0, 0.0),      # fully wet
])
def test_pit_lake_stays_at_rest(limiter, eta0, tilt):
    mesh, nsm = pit(eta0, limiter, tilt)
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0, cfl=0.4, mood=True)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    h = Q[1]
    wet = h > 1e-3
    u = np.hypot(Q[2], Q[3])[wet] / h[wet]
    eta = (Q[0] + h)[wet]
    assert t == pytest.approx(1.0, abs=1e-12)
    assert (0 < wet.sum() < h.size) if eta0 < 1.0 else wet.all()
    assert u.max() < 1e-12, f"the lake moved: max |u| = {u.max():.3e}"
    assert np.ptp(eta) < 1e-12


def test_shore_rule_follows_the_free_surface_roles():
    """The dry-neighbour rule is free-surface specific (ANALYSIS 50, 62): active only where the
    model declares the depth and bed roles and the reconstruction is the pair (h, eta = b + h)
    with the bed derived, its rows the roles' (never fixed indices); every other system (Euler,
    a conservative reconstruction, a reconstructed bed) prints the plain gradient and limiter
    blocks."""
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.numerics import BedReconstruction, ReconstructedVariables
    from zoomy_core.symsolver.discretization.reconstruction import (bed_rows, maps,
                                                                    shore_extension)
    from zoomy_core.systemmodel.euler import compressible_euler

    def nsm_of(sm, **spec):
        return NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod, **spec))

    def rows_of(nsm):                  # as `reconstruct`
        return bed_rows(nsm) if maps(nsm)[0] is not None else None

    for level in (0, 1):
        sm = SystemModel.from_model(SME(level=level, dimension=3,
                                        closures=[Newtonian(), StressFree()]))
        nsm = nsm_of(sm)
        rows = rows_of(nsm)
        assert rows == (sm.roles.bed, sm.roles.depth)
        (eta, h), eps = shore_extension(nsm, rows)
        assert (eta, h) == (sm.roles.depth, sm.roles.bed)
        for spec in (dict(variables=ReconstructedVariables.conservative),
                     dict(bed=BedReconstruction.reconstructed)):
            other = nsm_of(sm, **spec)
            assert shore_extension(other, rows_of(other)) == ()
    euler = compressible_euler(2, 1.4, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in TAGS]))
    nsm = nsm_of(euler)
    assert euler.roles.depth is None and rows_of(nsm) is None
    s = SymbolicHyperbolicSolverNumpy(time_end=0.01, cfl=0.45)
    s.setup(LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=4, ny=4), nsm)
    blocks = [ln.strip() for ln in s.source.splitlines()
              if "gradient(" in ln or "minmod(" in ln]
    plain = (r"\w+ = gradient\(\w+, \w+, mesh\)",
             r"\w+ = minmod\(\([\d, ]+\), \w+, \w+, \w+, mesh\)")
    assert blocks and all(any(re.fullmatch(f, ln) for f in plain) for ln in blocks), blocks
