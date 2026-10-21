"""E7 — a-posteriori MOOD positivity, numpy backend (REQ-152; spec §1b/§2
rank 7).

HARD INVARIANT (user mandate): no solver / reconstruction path clamps, floors
or truncates the depth ``h``.  Positivity is achieved ONLY by the masked
order-1 re-step (Xing-Zhang lemma on demoted cells; conservation from the
shared face flux).  Trimmed from test_aposteriori_mood_req152.py.
"""
import numpy as np
import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.initial_conditions import UserFunction, Constant
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.operations import gate_eigenvalues_dry

pytestmark = [pytest.mark.solver]


def _make_solver(positivity, nx=8, cfl=0.45, t_end=0.1, ic="slab", H=1.0,
                 gate=True):
    """Closed-box (wall) 2-D SWE dry dam break, order-2, given positivity.

    ``gate=False`` drops :func:`gate_eigenvalues_dry`.  Every test here now runs
    with the gate ON (cid=209): the gate publishes its dry-zeroed spectrum to
    the dt-only ``eigenvalues_cfl`` slot and leaves ``eigenvalues`` — the
    Rusanov dissipation — physical.  While it overwrote the shared slot the
    radial case below reached an inadmissible state at step 1, because zeroing
    ``s_max`` at the wet/dry front removed the very damping the first-order
    positivity argument rests on and the MOOD cascade could not recover.
    """
    sm = SystemModel.from_model(SWE(dimension=2, boundary_conditions=BoundaryConditions(
        [FromModel(tag=t, definition="wall")
         for t in ("left", "right", "bottom", "top")])))
    n_state = len(sm.state)              # [b, h, hu, hv]

    def _ic(x):
        q = np.zeros(n_state)
        if ic == "radial":
            q[1] = H if np.hypot(x[0] - 0.5, x[1] - 0.5) < 0.15 else 0.0
        else:                            # slab dry dam break
            q[1] = H if x[0] < 0.5 else 0.0
        return q

    sm.initial_conditions = UserFunction(function=_ic)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=nx, ny=nx)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, positivity=positivity),
        # REQ-181: the dry eigenvalue gate is opt-in; this wet/dry dam break
        # opts in so the recorded dry-front h_min values still hold.
        extra_operations=[gate_eigenvalues_dry()] if gate else [])
    solver = FreeSurfaceFlowSolver(
        time_end=t_end, compute_dt=timestepping.adaptive(CFL=cfl))
    solver.setup_simulation(mesh, nsm, write_output=False)
    return solver


def _h_index(solver):
    return int(solver._free_surface_h_index)


def _drive(solver):
    """Run to ``time_end``; return (min h over all steps, rel mass drift)."""
    nc = solver._sim_mesh.n_inner_cells
    vol = solver._sim_mesh.cell_volumes[:nc]
    h_idx = _h_index(solver)
    mass0 = float((solver._sim_Q[h_idx, :nc] * vol).sum())
    hmin = np.inf
    t = 0.0
    while t < solver.time_end:
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_face_inradius,
            solver._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(solver.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        solver._sim_time = t
        solver.step(dt)
        hmin = min(hmin, float(solver._sim_Q[h_idx, :nc].min()))
        t += dt
    mass1 = float((solver._sim_Q[h_idx, :nc] * vol).sum())
    return hmin, abs(mass1 - mass0) / abs(mass0)


@pytest.mark.small
@pytest.mark.gate
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_twin_masked_o1_only():
    """positivity='mood' on the 8x8 dry slab: h >= 0 at EVERY step with mass
    conserved to 1e-13 — achieved ONLY by the masked O1 re-step; and a
    ``force_o1`` cell reconstructs to its cell MEAN with any NEGATIVE value
    passing through unchanged (slope kill, NOT a depth clamp)."""
    solver = _make_solver("mood")
    assert solver._mood is True and solver._support_o1 is True
    hmin, drift = _drive(solver)
    assert hmin >= 0.0, f"a-posteriori MOOD still yields h<0 (hmin={hmin})"
    assert drift < 1e-13, f"mass drifted {drift:.2e} (corrector must be conservative)"

    # force_o1 is a slope kill, not a clamp
    from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction
    solver = _make_solver("mood", nx=8, t_end=0.0)
    mesh = solver._sim_mesh
    recon = LSQMUSCLReconstruction(mesh, dim=2)
    nc = mesh.n_inner_cells
    n_vars = 4
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n_vars, nc))       # deliberately includes negatives
    ifaces = recon._interior_faces
    Q[1, int(recon._iA_int[0])] = -0.37         # a clearly-negative depth entry
    bf = np.zeros((n_vars, mesh.n_boundary_faces))

    QL0, _QR0 = recon(Q, bf)                                  # order 2
    mask = np.ones(nc, dtype=bool)
    QLm, QRm = recon(Q, bf, force_o1=mask)                    # all cells demoted
    assert np.array_equal(QLm[:, ifaces], Q[:, recon._iA_int])
    assert np.array_equal(QRm[:, ifaces], Q[:, recon._iB_int])
    assert not np.array_equal(QL0[:, ifaces], Q[:, recon._iA_int])
    assert QLm.min() < 0.0
    assert QLm[1, ifaces[0]] == -0.37           # NOT floored to 0


@pytest.mark.small
@pytest.mark.gate
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_routing_and_negative_control():
    """zhang_shu routes to the a-priori XZS cap (``_mood`` False), mood to the
    plain force_o1-aware LSQ-MUSCL (no deviation cap); and WITHOUT the
    corrector the plain order-2 candidate drives the depth negative on the
    same slab — the non-vacuity proof for the twin above."""
    solver = _make_solver("zhang_shu")
    assert solver._mood is False and solver._support_o1 is True

    from zoomy_core.fvm.reconstruction import (
        LSQMUSCLReconstruction, PositivityPreservingLSQMUSCL)
    solver = _make_solver("mood", nx=8, t_end=0.0)
    mesh = solver._sim_mesh
    sm = solver._get_symbolic_model(solver._sim_model)
    base = FreeSurfaceFlowSolver.__mro__[1]      # base HyperbolicSolver class
    for pos, cls in (("zhang_shu", PositivityPreservingLSQMUSCL),
                     ("mood", LSQMUSCLReconstruction)):
        solver.nsm.reconstruction.positivity = pos
        r = base._build_reconstruction(solver, mesh, sm)
        assert type(r) is cls, f"{pos} -> {type(r).__name__}, expected {cls.__name__}"

    # negative control: plain order-2 goes h<0 (the corrector has real work)
    hmin, _ = _drive(_make_solver(""))
    assert hmin < 0.0, f"plain order-2 stayed non-negative (hmin={hmin}); vacuous"


@pytest.mark.large
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_march_bit_equal_rederivation():
    """[large] Mandate check: at each step of a violent 16x16 radial dry dam
    break the committed depth equals an INDEPENDENT re-derivation of the
    corrector (candidate -> troubled mask -> masked O1 re-step) bit-for-bit —
    no clamp, no floor, no post-processing anywhere."""
    solver = _make_solver("mood", nx=16, t_end=0.06, ic="radial", H=2.0)
    flux = solver._sim_flux_operator
    source = solver._sim_source_operator
    params = solver._sim_parameters
    h_idx = _h_index(solver)
    troubled_ever = False
    t = 0.0
    n = 0
    while t < solver.time_end and n < 20:
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_face_inradius,
            solver._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(solver.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        Q0 = np.array(solver._sim_Q)
        Qaux = solver._sim_Qaux

        def _rhs(tt, Qin, f):
            z = np.zeros_like(Qin)
            # REQ-185: source takes the current time (cell centres captured).
            return (flux(dt, tt, Qin, Qaux, params, z, f)
                    + source(dt, tt, Qin, Qaux, params, z))

        def _rk2(f):
            Q1 = Q0 + dt * _rhs(t, Q0, f)
            Q2 = Q1 + dt * _rhs(t + dt, Q1, f)
            return 0.5 * (Q0 + Q2)

        # Re-derive the corrector INDEPENDENTLY (cid 214): demote the troubled
        # cells, then the whole step if that is not enough.  Two levels, no
        # iteration — the same design the jax backend carries.  The EPD_2
        # stencil cascade this replaces was measured harmful at a wet/dry front
        # (see _mood_cascade's docstring).
        def _bad(Qc):
            return (Qc[h_idx, :] < 0.0) | ~np.isfinite(Qc).all(axis=0)

        cand = _rk2(None)
        troubled = _bad(cand)
        expected = cand
        if troubled.any():
            expected = _rk2(troubled)
            if _bad(expected).any():
                expected = _rk2(np.ones(cand.shape[1], dtype=bool))
        troubled_ever = troubled_ever or bool(troubled.any())

        solver._sim_time = t
        solver.step(dt)
        assert np.array_equal(solver._sim_Q[h_idx, :], expected[h_idx, :]), (
            f"committed depth != masked O1 re-step at step {n}")
        t += dt
        n += 1
    assert troubled_ever, "no troubled step occurred — corrector path not exercised"
