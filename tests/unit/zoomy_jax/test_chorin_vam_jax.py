"""ChorinSplitVAMSolverJax on the modern declarative VAM model.

Ported off the deleted ``VAMModelGalerkin`` (legacy model tree removed in the
restructure) onto ``VAM(level=1).system_model`` + ``VAM.chorin_split`` — the
same pipeline the numpy ``test_vam_dambreak`` drives, but through the JAX
Chorin solver.

Also the regression guard for the Chorin aux unification: the jax solver's
per-pool ``_refresh_aux_for_sm`` now delegates to the shared
``HyperbolicSolver._walk_derivative_aux`` and the formerly module-local LSQ
kernel duplicate was deleted in favour of the shared
``zoomy_jax.mesh.mesh.lsq_gradient_per_field`` (used by both the aux refresh and
the pressure matvec).  A finite dam-break-over-bump run exercises both paths.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax


def _build_solver(nc=60):
    model = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1)
    sm = model.system_model

    def bump_ic(xv):
        xx = float(xv[0])
        bv = 0.20 * np.exp(-(xx ** 2) / (2 * 0.20 ** 2))
        hv = max((0.34 - bv) if xx < 1.0 else 0.015, 0.015)
        out = np.zeros(len(sm.state))
        out[0] = bv          # b
        out[1] = hv          # h
        return out

    sm.initial_conditions = IC.UserFunction(function=bump_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    sm.attach_boundary_conditions(bcs)

    split = model.chorin_split(system_model=sm)
    split.SM_pred.attach_boundary_conditions(bcs)

    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=nc)
    solver = ChorinSplitVAMSolverJax(
        split.SM_pred, split.SM_press, split.SM_corr,
        pressure_tol=1e-6, pressure_maxit=100, time_end=0.5)
    solver.setup_simulation(mesh)
    state_names = [str(s) for s in sm.state]
    return solver, state_names


def test_chorin_vam_jax_dambreak_over_bump():
    """Short dam break over a Gaussian bump through the JAX Chorin solver:
    finite, positive depth, mass conserved, non-hydrostatic pressure active."""
    nc = 60
    solver, state_names = _build_solver(nc)
    ih, ip0 = state_names.index("h"), state_names.index("P_0")

    # update_aux_variables -> _refresh_aux_for_sm -> _walk_derivative_aux
    solver.update_aux_variables()

    dx = float(solver._rt_mesh.cell_volumes[0])
    dt = 0.3 * dx / (float(np.sqrt(9.81 * 0.34)) + 1.0)
    Qaux = solver._sim_Qaux
    time = jnp.asarray(0.0, dtype=jnp.float64)
    mass0 = float(np.asarray(solver._sim_Q)[ih, :nc].sum() * dx)

    for _ in range(30):
        solver._sim_Q = solver.step(jnp.asarray(dt), time, solver._sim_Q, Qaux)
        time = time + dt

    Q = np.asarray(solver._sim_Q)
    h = Q[ih, :nc]
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    mass1 = float(h.sum() * dx)
    assert abs(mass1 - mass0) < 5e-3 * mass0          # mass conserved
    P0 = float(np.abs(Q[ip0, :nc]).max())
    assert 1e-5 < P0 < 5.0                            # NH pressure active, bounded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
