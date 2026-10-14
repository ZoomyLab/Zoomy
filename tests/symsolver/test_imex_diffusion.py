"""IMEX-SSP2(2,2,2) with the implicit diffusion in its stages (NumPy backend; gate 50).

u_t + a u_x = nu u_xx on [0, 1], u = 0 at both ends (Dirichlet: mirror ghosts), a
SystemModel written directly (the diffusion tensor in the implicit slot `diffusion_matrix`):

* the solver equals the scheme written in matrix form (Pareschi, Russo 2005, Table II) from the
  semi-discrete operators L_a (advection) and L_d (diffusion), stiff (nu dt / dx^2 = 10);
* the time order against the semi-discrete exact solution expm(t L) u0 is 2 (pure stiff
  diffusion, nu dt / dx^2 up to 12.8).
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import Zstruct
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.discretization.time import IMEX_RK
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.settings import Imex
from zoomy_core.systemmodel.system_model import SystemModel, face_normal_symbols

pytestmark = [pytest.mark.solver, pytest.mark.small]

NX, T = 32, 0.05
TIGHT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=200)


def advdiff(nu, a):
    """`(mesh, nsm)`: u_t + a u_x = nu u_xx, u = 0 at both ends, u0 = sin(pi x)."""
    x, u = sp.Symbol("x", real=True), sp.Symbol("u", real=True)
    pnu, pa = sp.Symbol("nu", positive=True), sp.Symbol("a", real=True)
    params = Zstruct(nu=pnu, a=pa)
    params._symbolic_name = "p"
    A = sp.MutableDenseNDimArray.zeros(1, 1, 1, 1)
    A[0, 0, 0, 0] = pnu
    sm = SystemModel(
        time=sp.Symbol("t", real=True), space=[x], state=[u], aux_state=[], parameters=params,
        flux=sp.Matrix([[pa * u]]), hydrostatic_pressure=sp.zeros(1, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, 1), source=sp.zeros(1, 1),
        mass_matrix=sp.eye(1), eigenvalues=sp.Matrix([pa * face_normal_symbols(1)[0]]),
        parameter_values=Zstruct(nu=nu, a=a), diffusion_matrix=A)
    sm.declare_roles(depth=None, momentum=[])
    sm.declare_admissible([])
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Dirichlet(t, value=0.0) for t in ("left", "right")]))
    sm.initial_conditions = IC.UserFunction(function=lambda X: np.array([np.sin(np.pi * X[0])]))
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))
    return LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=NX), nsm


def operator(nu, a, dt=1e-3):
    """The semi-discrete matrix L (u' = L u): one explicit Euler step per unit vector."""
    s = SymbolicHyperbolicSolverNumpy(time_end=dt, dt=dt, cfl_check_every=0)
    Q, A = s.setup(*advdiff(nu, a))
    L = np.empty((NX, NX))
    for j in range(NX):
        e = np.zeros((1, NX))
        e[0, j] = 1.0
        L[:, j] = (s.run(e, s.initial_aux(e, A))[1][0] - e[0]) / dt
    return L


def march(nu, a, dt):
    """`(u(T), u0)` by IMEX-SSP2 at the constant step `dt`."""
    s = SymbolicHyperbolicSolverNumpy(time_end=T, dt=dt, cfl_check_every=0, imex=Imex.ssp2,
                                      **TIGHT)
    Q0, A = s.setup(*advdiff(nu, a))
    _, Q, _ = s.run(Q0, A)
    return np.asarray(Q)[0], np.asarray(Q0)[0]


def test_ssp2_diffusion_is_the_tableau():
    """Stiff (nu dt / dx^2 = 10.2): the solver equals IMEX-SSP2 in matrix form, the advection
    explicit and the diffusion implicit in each stage, to the Newton tolerance."""
    nu, a, dt = 0.4, 1.0, 0.0125
    La = operator(0.0, a)
    Ld = operator(nu, a) - La
    tab = IMEX_RK[Imex.ssp2]
    At = [[float(v) for v in r] for r in tab.At]
    Ai = [[float(v) for v in r] for r in tab.A]
    u, u0 = march(nu, a, dt)
    v, eye = u0.copy(), np.eye(NX)
    for _ in range(round(T / dt)):
        F, G = [], []
        for i in range(2):
            r = v + sum(dt * (At[i][j] * F[j] + Ai[i][j] * G[j]) for j in range(i))
            Y = np.linalg.solve(eye - dt * Ai[i][i] * Ld, r)
            F.append(La @ Y)
            G.append(Ld @ Y)
        v = v + dt * sum(float(wt) * f + float(w) * g for wt, w, f, g in zip(tab.wt, tab.w, F, G))
    assert np.abs(u - v).max() <= 1e-11 * np.abs(v).max()


def test_ssp2_stiff_diffusion_is_second_order():
    """Pure diffusion, nu 1, nu dt / dx^2 = 12.8 / 6.4 / 3.2: order 2 against expm(T L) u0."""
    from scipy.linalg import expm
    L = operator(1.0, 0.0)
    errs = []
    for k in range(3):
        u, u0 = march(1.0, 0.0, 0.0125 / 2 ** k)
        ex = expm(T * L) @ u0
        errs.append(np.abs(u - ex).max() / np.abs(ex).max())
    assert np.log2(errs[1] / errs[2]) >= 1.9, errs
