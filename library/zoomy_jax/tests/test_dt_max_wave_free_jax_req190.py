"""REQ-190 (jax adoption): the NSM's ``dt_max`` reaches the jax solver's
``compute_dt``, and a WAVE-FREE domain steps at ``dt_max`` (not ``inf``, not a
magic floor).

Core (zoomy_core@93e9e2e) added ``dt_max`` as a standard ``NumericalSystemModel``
parameter (default 5.0 s) and made the shared ``timestepping.adaptive`` fill its
cap from ``nsm.dt_max`` via ``apply_default_dt_max`` — a wave-free domain (every
gated ``|λ| = 0`` → local CFL limits ``+inf``) then collapses to ``dt_max``.  The
jax solver shares that ``timestepping.adaptive``; this test covers the jax
CALL-SITE fill: ``HyperbolicSolver.setup_simulation`` now calls
``timestepping.apply_default_dt_max(self.compute_dt, nsm.dt_max)``.

Uses a scalar advection model ``∂_t c + a·∂_x c = 0`` (``flux = a·c``,
``eigenvalue = a``) — no ``h``, so a wave-free case (``a = 0``) has NO 0/0, and
the dt behaviour is isolated from wet/dry desingularization.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

import sympy as sp

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

from loguru import logger
logger.remove()


class Advect(StructuredDerivativeModel):
    """``∂_t c + a·∂_x c = 0`` — advection speed ``a`` is the only wave speed, so
    ``a = 0`` gives a wave-free domain (``max|λ| = 0``)."""

    dimension = 1
    variables = ["c"]
    parameters = {"a": (1.0, "real")}

    def _build_function_groups(self):
        return {}

    def flux(self):
        F = sp.Matrix.zeros(1, 1)
        F[0, 0] = self.parameters.a * self.Q.c
        return ZArray(F)

    def eigenvalues(self):
        return ZArray([self.parameters.a])


def _setup(a, dt_max, *, compute_dt=None, nc=20):
    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    model = Advect(
        parameters={"a": (a, "real")},
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=lambda x: np.array([1.0])),
    )
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nc)
    kw = {} if dt_max is None else {"dt_max": dt_max}
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=1), **kw)
    solver = HyperbolicSolver(
        time_end=1.0,
        **({"compute_dt": compute_dt} if compute_dt is not None
           else {"compute_dt": timestepping.adaptive(CFL=0.9, dimension=1)}))
    Q, Qaux = solver.setup_simulation(mesh, nsm)
    return solver, Q, Qaux


@pytest.mark.jax
def test_wave_free_steps_at_dt_max_jax():
    """``a = 0`` → every ``|λ| = 0`` → wave-free → dt is exactly ``dt_max``
    (not ``inf``, not a floor).  Proves the NSM ``dt_max`` reaches jax's
    ``compute_dt`` AND the wave-free branch collapses to it."""
    DTMAX = 0.5
    solver, Q, Qaux = _setup(a=0.0, dt_max=DTMAX)
    dt = float(solver.compute_timestep(Q, Qaux))
    assert np.isfinite(dt), "wave-free dt is inf — dt_max not wired into jax compute_dt"
    assert dt == pytest.approx(DTMAX), (
        f"wave-free dt {dt} != dt_max {DTMAX} — the NSM dt_max did not reach "
        "the jax solver's compute_dt (apply_default_dt_max not called at setup)")


@pytest.mark.jax
def test_nsm_dt_max_clamps_wet_dt_jax():
    """A wet (``a = 2``) case has a finite CFL dt; a tiny NSM ``dt_max`` below it
    clamps the step to ``dt_max`` — the direct proof that ``nsm.dt_max`` is wired
    into the jax ``compute_dt``.  With the default 5.0 s cap the same case is
    unaffected (dt well below 5)."""
    solver_big, Qb, Qab = _setup(a=2.0, dt_max=5.0)
    dt_big = float(solver_big.compute_timestep(Qb, Qab))
    assert np.isfinite(dt_big) and dt_big < 5.0, "wet CFL dt should be < dt_max=5"

    TINY = 1e-4
    solver_tiny, Qt, Qat = _setup(a=2.0, dt_max=TINY)
    dt_tiny = float(solver_tiny.compute_timestep(Qt, Qat))
    assert dt_tiny == pytest.approx(TINY), (
        f"dt {dt_tiny} != tiny dt_max {TINY} — nsm.dt_max not wired into jax compute_dt")


@pytest.mark.jax
def test_explicit_dt_max_wins_over_nsm_jax():
    """An explicit ``adaptive(dt_max=...)`` from the caller is never overwritten
    by the NSM default at setup — explicit wins (mirrors the numpy contract)."""
    explicit = timestepping.adaptive(CFL=0.9, dimension=1, dt_max=0.1)
    solver, Q, Qaux = _setup(a=0.0, dt_max=5.0, compute_dt=explicit)
    dt = float(solver.compute_timestep(Q, Qaux))
    assert dt == pytest.approx(0.1), (
        f"dt {dt} != explicit 0.1 — NSM dt_max wrongly overrode the caller's cap")
