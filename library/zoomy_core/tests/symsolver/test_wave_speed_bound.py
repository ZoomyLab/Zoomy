"""The CFL step's positivity guarantee: the wave speed of a model without closed-form
eigenvalues is an upper bound of rho(A) (`gelfand_bound`), and under the faces rule every MOOD
repair takes at most the step of its own Riemann problems (ANALYSIS 48).
"""
import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.userfunctions import gelfand_bound
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import (Limiter, NumericalSystemModel, PositiveNonconservativeHLL,
                                 ReconstructionSpec)
from zoomy_core.symsolver.discretization.time import CflWaveSpeed
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _flat(A):
    n = A.shape[-1]
    return [A[:, i, j] for i in range(n) for j in range(n)]


def test_gelfand_bound_is_an_upper_bound():
    """bound >= max|eig| on Gaussian, non-normal triangular, defective, nilpotent matrices over
    12 decades; tight on rotations; 0 for the zero matrix, inf for a non-finite entry."""
    rng = np.random.default_rng(48)
    m, j, off = 2000, SystemModel.spectral_squarings, SystemModel.spectral_cw_offset
    for n in (2, 3, 5, 8):
        d = rng.uniform(-1, 1, (m, n))
        sets = [rng.normal(size=(m, n, n)),
                np.triu(rng.normal(0, 1e3, (m, n, n)), 1) + d[:, :, None] * np.eye(n),
                np.eye(n) * rng.uniform(-1, 1, (m, 1, 1)) + np.eye(n, k=1) * 10.0,
                np.triu(rng.normal(size=(m, n, n)), 1)]
        for A in sets:
            A = A * 10.0 ** rng.uniform(-6, 6, (m, 1, 1))
            rho = np.abs(np.linalg.eigvals(A)).max(-1)
            assert np.all(gelfand_bound(j, off, *_flat(A)) >= rho * (1 - 1e-12))
    th = rng.uniform(0, 2 * np.pi, m)
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                  np.stack([np.sin(th), np.cos(th)], -1)], -2)
    assert np.allclose(gelfand_bound(j, off, *_flat(R)), 1.0, rtol=2e-2)
    Z = np.zeros((1, 3, 3))
    assert gelfand_bound(j, off, *_flat(Z))[0] == 0.0
    Z[0, 1, 2] = np.nan
    assert gelfand_bound(j, off, *_flat(Z))[0] == np.inf


# Found by a random search over 8-cell states (bed steps; dry, thin and deep cells): under the
# faces rule the order-2 candidate's step is 5.6x the first-order step of the Riemann problems
# MOOD's fallback solves.  Repaired with the candidate's step, the fallback drained a cell to
# h = -4.7e-2 (h = -8.9e-3 at t = 0.01).
_B = [0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0]
_H = [0.0, 0.0, 0.0, 1.6708862480570887, 1.9262666353430017e-04, 5.8382781462709676e-01,
      3.4479773158151740e-03, 1.1315922745493967e-04]
_Q = [0.0, 0.0, 0.0, 4.344260887673245, -4.862527115560378e-03, 5.704882873866848e-01,
      -1.266965718181803e-01, 4.005592882274292e-03]


def test_mood_repair_takes_its_own_faces_step():
    """SME(0) 1-D, HLL with the hydrostatic reconstruction, order 2 (Venkatakrishnan) + MOOD,
    the faces rule: the march from the state above to t = 0.01 (an uncapped first step) keeps
    h >= 0."""
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    assert [str(s) for s in sm.state] == ["b", "h", "q_0"]
    n = len(_H)
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [v[min(int(x[0] * n), n - 1)] for v in (_B, _H, _Q)]))
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=2, limiter=Limiter.venkatakrishnan))
    h_min = []
    s = SymbolicHyperbolicSolverNumpy(
        time_end=0.01, cfl=0.4, mood=True, cfl_wave_speed=CflWaveSpeed.faces, dt_write=0.01,
        on_write=lambda t, Q, A: h_min.append(float(np.min(Q[1]))))
    Q0, A = s.setup(LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n), nsm)
    assert np.array_equal(Q0, np.array([_B, _H, _Q]))
    t, Q, _ = s.run(Q0, A)
    assert t == pytest.approx(0.01, abs=1e-12)
    assert np.isfinite(Q).all()
    assert len(h_min) == 2 and min(h_min) >= 0.0


def test_faces_rule_order2_bounds_the_cells_own_speeds():
    """SME(1) 1-D, order 2 with the edge source (the NCP inside each cell): the faces rule's
    step is at most the CFL step of the cells' own speeds, also where the order-2 traces are
    slower (an Audusse-dry face carries speed 0)."""
    import sympy as sp

    from zoomy_core.symsolver import PyPrinter
    from zoomy_core.symsolver.core import unpack
    from zoomy_core.symsolver.discretization.reconstruction import reconstruct
    from zoomy_core.symsolver.discretization.time import cfl_dt, face_dt, face_rule
    from zoomy_core.symsolver.numpy_backend import runtime as R
    from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
    from zoomy_core.symsolver.vocabulary import boundary_traces, face_aux, faces

    model = SME(level=1, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=PositiveNonconservativeHLL,
        reconstruction=ReconstructionSpec(order=2, limiter=Limiter.venkatakrishnan))
    n, cfl = 12, 0.4
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0, cfl=cfl, mood=False,
                                      cfl_wave_speed=CflWaveSpeed.faces)
    Q0, A0 = s.setup(LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n), nsm)
    nsm = s.nsm
    assert nsm.call.numerical_cell_edge_source is not None
    tpl = HyperbolicTemplate(nsm, cfl=cfl, dimension=1, wave_speed=CflWaveSpeed.faces)

    def proc(Q, Qaux, p, time, mesh):
        cap = sp.Float(1e30)
        *_, dt = tpl.step(Q, Qaux, p, time, None, mesh,
                          dt_rule=face_rule(nsm, mesh, cfl, 1, None, time, cap))
        QL, QR = reconstruct(nsm, Q, Qaux, p, time, 2, sp.S.false, mesh)
        QR = boundary_traces(nsm.call.boundary_conditions, Q, QL, QR, Qaux, p, time, mesh)
        AL, AR = unpack(face_aux(Qaux, mesh), 2)
        S = unpack(faces(nsm.call.numerical_face_sides_speed, QL, QR, AL, AR, p, mesh), 3)[2]
        return dt, face_dt(nsm, S, mesh, cfl, 1), cfl_dt(nsm, Q, Qaux, p, mesh, cfl, 1)

    fn = R.build(PyPrinter().procedure("proc", sp.symbols("Q Qaux p time mesh"), proc), "proc",
                 s.module)
    rng = np.random.default_rng(1)
    slower = 0
    for _ in range(200):
        Q = np.zeros_like(Q0)
        Q[0] = rng.choice([0.0, 0.3, 1.0], n)
        Q[1] = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(0.05, 2.0, n))
        Q[2] = Q[1] * rng.uniform(-3, 3, n)
        Q[3] = Q[1] * rng.uniform(-20, 20, n)
        dt, dt_riemann, dt_cells = (float(v) for v in fn(Q, s.initial_aux(Q, A0), s.parameters,
                                                          0.0, s.mesh, 0)[:-1])
        assert dt <= dt_cells * (1 + 1e-12) and dt <= dt_riemann * (1 + 1e-12)
        slower += dt_riemann > 1.01 * dt_cells
    assert slower > 0          # the cells' speeds bind on some of the states
