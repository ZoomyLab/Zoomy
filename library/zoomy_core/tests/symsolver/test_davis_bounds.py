"""HLL without a closed-form spectrum: Davis speeds from Gelfand bounds of the signed extremes,
s_R = min(G(A + a I) - a, a) >= max Re lambda, s_L = max(a - G(a I - A), -a) <= min Re lambda,
a = G(A) (ANALYSIS 49, item 3; the full verification is gate 50).
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.riemann_solvers import HLL
from zoomy_core.fvm.userfunctions import gelfand_bound
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, PositiveNonconservativeHLL
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _sme(level, n_tags=("left", "right")):
    return SystemModel.from_model(SME(
        level=level, dimension=2, closures=[Newtonian(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in n_tags])))


@pytest.mark.parametrize("level", [0, 1])
def test_davis_pair_brackets_the_spectrum(level):
    """The production pair of SME(level) 1-D (no closed form) on random wet / thin states:
    s_L <= min eig, s_R >= max eig (eig of the same matrix), and narrower than (-a, a)."""
    nsm = NumericalSystemModel.from_system_model(_sme(level))
    assert nsm.eigenvalues is None
    num = HLL(nsm.sm)
    Q, A, P, N = (list(num.variables), list(num.aux_variables), list(num.parameters),
                  list(num.normal))
    sL, sR = num.wave_speed_bounds(Q, Q, A, A, P, N)
    mods = [{"gelfand_bound": gelfand_bound,
             "conditional": lambda c, x, y: np.where(c, x, y)}, "numpy"]
    f = sp.lambdify([*Q, *A, *P, *N], [sL, sR], modules=mods)
    a_expr = num.local_max_abs_eigenvalue(Q, A, P, N)
    fa = sp.lambdify([*Q, *A, *P, *N], a_expr, modules=mods)
    hinv = [i for i, s in enumerate(A) if str(s) == "hinv"]
    rng = np.random.default_rng(3)
    names = [str(q) for q in Q]
    m = 2000
    h = 10 ** rng.uniform(-6, 1, m)
    Qv = np.array([h if n == "h" else rng.uniform(0, 1, m) if n == "b"
                   else h * rng.uniform(-5, 5, m) for n in names])
    Av = np.zeros((len(A), m))
    Av[hinv] = 1.0 / h
    pv = [float(getattr(nsm.sm.parameter_values, str(p))) for p in P]
    lo, hi = (np.broadcast_to(v, (m,)) for v in f(*Qv, *Av, *pv, 1.0))
    a = np.broadcast_to(fa(*Qv, *Av, *pv, 1.0), (m,))
    # the spectrum of the same matrix A_n (with 1/h): from the quasilinear matrix, b dropped
    qm = nsm.quasilinear_matrix
    Mf = sp.lambdify([*Q, *A, *P], [[qm[i, j, 0] for j in range(len(Q))]
                                    for i in range(len(Q))], modules=mods)
    M = np.moveaxis(np.array([[np.broadcast_to(np.asarray(x, float), (m,)) for x in row]
                              for row in Mf(*Qv, *Av, *pv)]), -1, 0)
    ev = np.linalg.eigvals(M[:, 1:, 1:]).real                       # b's row and column are 0
    assert np.all(lo <= ev.min(-1) + 1e-12 * a) and np.all(hi >= ev.max(-1) - 1e-12 * a)
    assert np.all(lo >= -a) and np.all(hi <= a) and np.median((hi - lo) / (2 * a)) < 0.9


def test_hll_order1_dambreak_stays_positive():
    """SME(0) 1-D, HLL with the hydrostatic reconstruction (no closed form: the Davis bounds),
    order 1, a dam break onto a dry bed with a bed step: h >= 0 at every write, mass kept."""
    sm = _sme(0)
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0.2 * (x[0] > 0.6), 1.0 * (x[0] < 0.4), 0.0]))
    nsm = NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeHLL)
    h_min = []
    s = SymbolicHyperbolicSolverNumpy(time_end=0.2, cfl=0.45, dt_write=0.02,
                                      on_write=lambda t, Q, A: h_min.append(float(Q[1].min())))
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=100)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    assert t == pytest.approx(0.2) and min(h_min) >= 0.0
    vol = np.asarray(mesh.cell_volumes)[:Q.shape[1]]
    assert abs((Q[1] - Q0[1]) @ vol) <= 1e-12 * (Q0[1] @ vol)


def test_wall_pair_is_symmetric_at_an_oblique_wall():
    """SME(1) 2-D: at a face with normal (cos 0.7, sin 0.7) and the reflected state on the
    outside (a wall), s_L = -s_R to round-off (the shifted matrices are bounded in the face's
    normal frame), so the HLL mass flux through the wall vanishes."""
    sm = SystemModel.from_model(SME(level=1, dimension=3, closures=[Newtonian(), StressFree()]))
    nsm = NumericalSystemModel.from_system_model(sm)
    num = HLL(nsm.sm)
    Q, A, P, N = (list(num.variables), list(num.aux_variables), list(num.parameters),
                  list(num.normal))
    QR = list(num.variables_plus)
    sL, sR = num.wave_speed_bounds(Q, QR, A, A, P, N)
    mods = [{"gelfand_bound": gelfand_bound,
             "conditional": lambda c, x, y: np.where(c, x, y)}, "numpy"]
    f = sp.lambdify([*Q, *QR, *A, *P, *N], [sL, sR], modules=mods)
    names = [str(q) for q in Q]
    rng = np.random.default_rng(5)
    nrm = np.array([np.cos(0.7), np.sin(0.7)])
    pv = [float(getattr(nsm.sm.parameter_values, str(p))) for p in P]
    for _ in range(50):
        qL = {k: rng.uniform(-2, 2) for k in names}
        qL["h"], qL["b"] = rng.uniform(0.1, 2.0), 0.0
        qR = dict(qL)
        for lvl in ("0", "1"):
            u = np.array([qL[f"q_x_{lvl}"], qL[f"q_y_{lvl}"]])
            qR[f"q_x_{lvl}"], qR[f"q_y_{lvl}"] = u - 2 * (u @ nrm) * nrm
        aux = [1.0 / qL["h"] if str(a) == "hinv" else 0.0 for a in A]
        lo, hi = f(*[qL[k] for k in names], *[qR[k] for k in names], *aux, *pv, *nrm)
        assert abs(lo + hi) <= 1e-12 * abs(hi)
