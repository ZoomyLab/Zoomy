r"""Periodic boundary conditions in the JAX FVM (REQ-116).

The jax ``HyperbolicSolver`` used to apply a NON-periodic (extrapolation/wall)
flux at the two periodic-seam cells: ``setup_simulation`` omitted the
``mesh.resolve_periodic_bcs`` call numpy runs, and the boundary-flux loop fed
the reconstructed inner state to the BC kernel for ``Q_R`` (extrapolation)
instead of the opposite-side partner cell.  Effect: periodic waves decayed
(mass leaked at the seam) where numpy grows/conserves them.

This is the definitive discriminator: on a periodic domain an advecting pulse
must **conserve mass** to machine precision — an extrapolation seam is an open
boundary and leaks.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import BaseMesh, ensure_lsq_mesh
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_jax.fvm.solver_jax import HyperbolicSolver

L, NC, U0, H0, EPS = 1.0, 120, 0.8, 1.0, 0.05


def _periodic_swe():
    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
    ])

    def ic(x):
        xx = float(x[0])
        h = H0 + EPS * np.sin(2.0 * np.pi * xx / L)   # smooth periodic pulse
        return np.array([0.0, h, h * U0])             # [b, h, hu], uniform u=U0
    return SWE(dimension=1, boundary_conditions=bcs,
               initial_conditions=IC.UserFunction(function=ic))


@pytest.mark.jax
def test_periodic_advection_conserves_mass():
    """A pulse advected at U0 across the periodic seam conserves total mass to
    ~1e-11 (the extrapolation seam this fix removes leaked ~1e-2)."""
    mesh = BaseMesh.create_1d(domain=(0.0, L), n_inner_cells=NC)
    model = _periodic_swe()
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=1))

    # Cell volumes on the same (lsq-resolved) mesh the solver uses.
    vol = np.asarray(ensure_lsq_mesh(mesh, nsm).cell_volumes)[:NC]

    solver = HyperbolicSolver(
        time_end=2.0 * L / U0,                        # ~2 full wraps
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=1))

    Q0, _ = solver.setup_simulation(mesh, nsm)
    mass0 = float(np.sum(np.asarray(Q0)[1, :NC] * vol))

    Qf, _ = solver.solve(mesh, nsm, write_output=False)
    Qf = np.asarray(Qf)
    massf = float(np.sum(Qf[1, :NC] * vol))

    assert np.isfinite(Qf).all()
    assert (Qf[1, :NC] > 0).all()                     # depth stays positive
    # Mass conserved (periodic) — NOT leaked (old extrapolation seam).
    assert abs(massf - mass0) / mass0 < 1e-11, (
        f"periodic mass not conserved: {abs(massf-mass0)/mass0:.2e}")
    # The pulse did not decay away (amplitude survives the seam crossing).
    assert (Qf[1, :NC].max() - Qf[1, :NC].min()) > 0.2 * EPS
