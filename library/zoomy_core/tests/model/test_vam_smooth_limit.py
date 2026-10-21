"""VAM(1) → SME(1) in the smooth long-wave limit.

A gentle Gaussian hump (h0=0.5, amp=0.05, sigma=3 ⇒ dispersion parameter
mu = (h0/L)² ≈ 0.028) marched with identical friction must give the SAME
solution from the non-hydrostatic VAM (Chorin split) and the hydrostatic
SME up to O(mu):

* relative L2 differences of h (vs the hump signal), q_0, q_1 stay at the
  few-percent level,
* the SHEAR FIELD q_1 correlates ≈ 1 (direction and shape, everywhere —
  this is the regression for the inverted-profile splitter bug, which had
  corr ≈ −1 while all symbolic checks passed),
* the non-hydrostatic pressure is a ~1% correction of the hydrostatic
  scale g·h0·amp.

This pins the RUNTIME chain (predictor extraction, elliptic solve,
corrector) in a regime with an analytic expectation — complementary to
the symbolic term-by-term reference tests.
"""
import numpy as np
import pytest

from zoomy_core.model.models import SME, VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver
from zoomy_core.fvm import timestepping
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.model, pytest.mark.solver]

NC, XMAX, T_END, CFL = 100, 20.0, 1.5, 0.9
PAR = {"lambda_s": 0.5, "nu": 1e-3}
H0, AMP, SIG = 0.5, 0.05, 3.0


def _bcs():
    return BoundaryConditions([Extrapolation(tag="left"),
                               Extrapolation(tag="right")])


def _ic(n):
    def ic(xv):
        out = np.zeros(n)
        out[1] = H0 + AMP * np.exp(-((float(xv[0]) - 10.0) ** 2)
                                   / (2 * SIG ** 2))
        return out
    return ic


def _build_sme(mesh):
    sme = SME(closures=[Newtonian(), NavierSlip(), StressFree()], level=1, parameters=dict(PAR), boundary_conditions=_bcs())
    sm_s = SystemModel.from_model(sme)
    sm_s.initial_conditions = IC.UserFunction(function=_ic(len(sm_s.state)))
    sm_s.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    nsm = NumericalSystemModel.from_system_model(
        sm_s, reconstruction=ReconstructionSpec(order=1))
    return nsm, [str(s) for s in sm_s.state]


def _build_vam(mesh):
    vam = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1, parameters=dict(PAR), boundary_conditions=_bcs())
    sm_v = SystemModel.from_model(vam)
    sm_v.initial_conditions = IC.UserFunction(function=_ic(len(sm_v.state)))
    sm_v.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = _bcs()
    sm_v.attach_boundary_conditions(bcs)
    split = vam.chorin_split(system_model=sm_v)
    split.SM_pred.attach_boundary_conditions(bcs)
    sol = ChorinSplitVAMSolver(split.SM_pred, split.SM_press, split.SM_corr,
                               pressure_solver="gmres", pressure_tol=1e-8,
                               riemann_solver="hr")
    sol.setup_simulation(mesh)
    return sol, [str(s) for s in sm_v.state]


def _vam_dt(sol):
    cdt = timestepping.adaptive(CFL=CFL, dimension=1)
    return min(float(cdt(sol._sim_Q, sol._sim_Qaux, sol._sim_parameters,
                         sol._sim_face_inradius,
                         sol._sim_compute_max_abs_eigenvalue)), 5e-3)


@pytest.mark.small
@pytest.mark.gate
def test_vam1_sme1_long_wave_one_step_twin(one_hyperbolic_step):
    """Default-tier canary: identical smooth-hump SME + VAM(Chorin) setup, ONE
    step each.  The O(mu) agreement holds only after real evolution, so the
    twin asserts the cheap invariants (finite, positive depth, bounded
    non-hydrostatic pressure) — the regression guard for a broken split."""
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm, ns = _build_sme(mesh)
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.adaptive(CFL=CFL, dimension=1))
    Qs = one_hyperbolic_step(solver, mesh, nsm)[:, :NC]
    sol, nv = _build_vam(mesh)
    sol.step(_vam_dt(sol))
    Qv = np.asarray(sol._sim_Q[:, :NC], float)
    assert np.all(np.isfinite(Qs)) and np.all(np.isfinite(Qv))
    assert Qs[ns.index("h")].min() > 0.0 and Qv[nv.index("h")].min() > 0.0
    p_scale = 9.81 * H0 * AMP
    assert np.abs(Qv[nv.index("P_0")]).max() < p_scale
    assert np.abs(Qv[nv.index("P_1")]).max() < p_scale


@pytest.mark.large
def test_vam1_matches_sme1_in_long_wave_limit():
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)

    nsm, ns = _build_sme(mesh)
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.adaptive(CFL=CFL, dimension=1))
    Qs, _ = solver.solve(mesh, nsm, write_output=False)
    Qs = np.asarray(Qs[:, :NC], float)

    sol, nv = _build_vam(mesh)
    t_now = 0.0
    while t_now < T_END - 1e-12:
        dt = min(_vam_dt(sol), T_END - t_now)
        sol.step(dt)
        t_now += dt
    Qv = np.asarray(sol._sim_Q[:, :NC], float)

    hs, q0s, q1s = (Qs[ns.index(k)] for k in ("h", "q_0", "q_1"))
    hv, q0v, q1v = (Qv[nv.index(k)] for k in ("h", "q_0", "q_1"))
    P0 = Qv[nv.index("P_0")]
    P1 = Qv[nv.index("P_1")]

    def rel(a, b, scale):
        return np.linalg.norm(a - b) / (np.linalg.norm(scale) + 1e-14)

    assert rel(hv, hs, hs - H0) < 0.10, "h deviates beyond O(mu)"
    assert rel(q0v, q0s, q0s) < 0.08, "q_0 deviates beyond O(mu)"
    assert rel(q1v, q1s, q1s) < 0.08, "q_1 deviates beyond O(mu)"
    # the shear FIELD must agree in direction/shape everywhere
    assert np.corrcoef(q1v, q1s)[0, 1] > 0.99, "shear field decorrelated"
    # non-hydrostatic pressure is a small correction in this regime
    p_scale = 9.81 * H0 * AMP
    assert np.abs(P0).max() < 0.05 * p_scale
    assert np.abs(P1).max() < 0.05 * p_scale
