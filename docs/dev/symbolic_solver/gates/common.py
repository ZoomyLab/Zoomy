"""Shared setups for the symbolic-solver gates (scratch, not repo tests)."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel


def sme(level, dimension, tags, bc):
    import zoomy_core.model.boundary_conditions as BC
    bcs = ([BC.Extrapolation(tag=t) for t in tags] if bc == "extrapolation"
           else [FromModel(tag=t, definition=bc) for t in tags])
    return SME(level=level, dimension=dimension,
               closures=[Newtonian(), ManningFriction(), StressFree()],
               boundary_conditions=BoundaryConditions(bcs))


def bump_2d(order, nx=40, flow=(0.0, 0.0), variables="conservative"):
    """SWE 2-D Gaussian bump, walls (tests/test_swe_2d_jax.py); ``flow`` adds
    a uniform background velocity.  Conservative reconstruction: the old
    solver's default path."""
    def ic(x):
        r2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
        h = 0.5 + 0.05 * np.exp(-r2 / 2.0)
        return np.array([0.0, h, h * flow[0], h * flow[1]])
    sm = SystemModel.from_model(sme(0, 3, ("left", "right", "bottom", "top"), "wall"))
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod",
                                              variables=variables))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


def dry_dambreak_2d(nx=48, ties=None):
    """Radial dry dam break, the model's reconstruction variables (eta = b + h,
    u = hinv q) + MOOD (tests/runtime/test_radial_dry_dambreak.py).  ``ties``:
    "old" = the old eta reconstruction's (b, momenta limited no more than
    eta)."""
    sm = SystemModel.from_model(sme(0, 3, ("left", "right", "bottom", "top"), "wall"))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = 2.0 if np.hypot(x[0] - 0.5, x[1] - 0.5) < 0.15 else 0.0
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    if ties == "old":
        ties = {n: "h" for n in names if n == "b" or n.startswith("q_")}
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod",
                                              ties=ties or {}))
    return LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx), nsm


def dambreak_1d(level, order, domain=(0.0, 1.0), nx=200, x_dam=0.5,
                hl=1.0, hr=0.5, bc="extrapolation", variables="conservative"):
    """SME(level) 1-D dam break with friction (source) and moments."""
    sm = SystemModel.from_model(sme(level, 2, ("left", "right"), bc))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = hl if x[0] < x_dam else hr
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod",
                                              variables=variables))
    return LSQMesh.create_1d(domain=domain, n_inner_cells=nx), nsm


def adaptive(cfl, mesh):
    return timestepping.adaptive(CFL=cfl, dimension=int(mesh.dimension))
