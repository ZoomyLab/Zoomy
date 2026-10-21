"""REQ-185 (jax adoption): time + position threaded into the ``source`` and
``update_aux_variables`` call sites of the jax FVM solver.

Core landed the interface (``source(Q,Qaux,p,t,dt,x)`` /
``update_aux_variables(Q,Qaux,p,t,x)``, zoomy_core@bafd2c8) and the
signature-driven jax lowering (zoomy_jax@d5e8110).  This test covers the jax
CALL-SITE fill: ``HyperbolicSolver.update_qaux`` threads the current ``time``
(and cell positions) into ``JaxRuntime.update_aux_variables``, and
``get_compute_source`` threads ``time``/``dt``/position into ``JaxRuntime.source``
via the timestepping (``ode.RK1(..., dt, time)``).

Acceptance mirrors the zoomy_core RainTracer march: a rain aux
``r_o = Piecewise((rate, t < T_rain), (0, True))`` drives ``∂_t c + a·∂_x c = r_o``.
The accumulated tracer mass after marching PAST ``T_rain`` must equal exactly
``rate·T_rain·area`` — rain turned ON then OFF at ``T_rain``.  If the jax solver
dropped ``t`` in ``update_aux_variables`` the rain would never switch off and the
mass would keep climbing to ``rate·T_end·area`` (here 0.06 vs the correct 0.025) —
so this single number discriminates a bound ``t`` from a dropped one.
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


class RainTracer(StructuredDerivativeModel):
    """``∂_t c + a·∂_x c = r_o``, ``r_o`` active only while ``t < T_rain`` — a
    mirror of the zoomy_core REQ-185 acceptance model, small enough to march
    cheaply on the jax solver."""

    dimension = 1
    variables = ["c"]
    parameters = {
        "a":         (1.0, "real"),          # advection speed (finite CFL dt)
        "rain_rate": (0.5, "nonnegative"),
        "T_rain":    (0.05, "positive"),
    }

    def __init__(self, **kw):
        super().__init__(aux_variables=["r_o"], **kw)

    def _build_function_groups(self):
        return {}

    def flux(self):
        F = sp.Matrix.zeros(1, 1)
        F[0, 0] = self.parameters.a * self.Q.c
        return ZArray(F)

    def eigenvalues(self):
        return ZArray([self.parameters.a])

    def update_aux_variables(self):
        t = sp.Symbol("t", real=True)
        p = self.parameters
        return ZArray([sp.Piecewise((p.rain_rate, t < p.T_rain),
                                    (sp.S.Zero, True))])

    def source(self):
        S = sp.Matrix.zeros(1, 1)
        S[0, 0] = self.aux_variables.r_o     # rain into the tracer equation
        return ZArray(S)


@pytest.mark.jax
def test_rain_mass_matches_rate_times_T_rain_jax():
    """Marching to ``t_end > T_rain``: accumulated mass == ``rate·T_rain·area``.

    This is the jax analogue of ``test_rain_volume_plateaus_after_T_rain`` — it
    proves ``update_qaux`` binds ``t`` (rain turns off at ``T_rain``).  A dropped
    ``t`` would give ``rate·t_end·area`` instead."""
    NC, XMAX = 40, 1.0
    T_RAIN, RATE, T_END = 0.05, 0.5, 0.12

    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
    ])

    def ic(x):
        return np.array([1.0])               # uniform tracer c = 1

    model = RainTracer(
        parameters={"a": (1.0, "real"), "rain_rate": (RATE, "nonnegative"),
                    "T_rain": (T_RAIN, "positive")},
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )
    mesh = LSQMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=1))

    # NOT a CFL-law test: this asserts a TIME INTEGRAL of a source that switches
    # off at T_rain, so its accuracy is set by dt, not by stability.  The step
    # straddling T_rain applies rain over its whole width, making the error
    # O(rate*dt) — at the suite CFL (0.9, dimension=1 => dt = 0.9*dx/a = 0.0225)
    # that is 0.011, four times the 5e-3 tolerance, and the test fails on
    # quadrature error alone while the t-binding it exists to check is perfect.
    # A fixed dt that divides T_rain and t_end exactly (0.05/1e-3 = 50 steps,
    # 0.12/1e-3 = 120) puts the switch on a step boundary and drives the
    # quadrature error to zero, leaving the assertion sensitive only to whether
    # t is bound.  CFL here is 0.04 — stability is not the constraint.
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.constant(dt=1.0e-3))
    Q, _ = solver.solve(mesh, nsm, write_output=False)

    c = np.asarray(Q[0, :NC], dtype=float)
    assert np.all(np.isfinite(c)), "non-finite tracer after rain march"
    dx = XMAX / NC
    v0 = 1.0 * XMAX                           # uniform c=1 over |domain|
    gain = float(c.sum() * dx) - v0

    expected = RATE * T_RAIN * XMAX          # 0.025 — rain ON then OFF at T_rain
    unbound = RATE * T_END * XMAX            # 0.060 — if t were dropped (rain stuck on)
    assert abs(gain - expected) < 5e-3, (
        f"accumulated rain {gain:.4f} != rate·T_rain·area {expected:.4f} "
        f"(if t were unbound it would be rate·t_end·area {unbound:.4f}) — "
        "update_aux_variables did not bind t in the jax update_qaux")


class RainSourceTracer(StructuredDerivativeModel):
    """Same rain, but ``t`` enters the ``source`` DIRECTLY (no aux) — this
    exercises the timestepping→source threading (``ode.RK1(..., dt, time)`` →
    ``get_compute_source`` → ``JaxRuntime.source(..., time=t)``), the leg the
    aux test above does not touch."""

    dimension = 1
    variables = ["c"]
    parameters = {
        "a":         (1.0, "real"),
        "rain_rate": (0.5, "nonnegative"),
        "T_rain":    (0.05, "positive"),
    }

    def _build_function_groups(self):
        return {}

    def flux(self):
        F = sp.Matrix.zeros(1, 1)
        F[0, 0] = self.parameters.a * self.Q.c
        return ZArray(F)

    def eigenvalues(self):
        return ZArray([self.parameters.a])

    def source(self):
        t = sp.Symbol("t", real=True)
        p = self.parameters
        S = sp.Matrix.zeros(1, 1)
        S[0, 0] = sp.Piecewise((p.rain_rate, t < p.T_rain), (sp.S.Zero, True))
        return ZArray(S)


@pytest.mark.jax
def test_source_binds_time_directly_jax():
    """A ``source`` that references ``t`` directly turns off at ``T_rain`` —
    proving the current ``time`` reaches ``JaxRuntime.source`` through the
    timestepping (``ode.RK1(self._rt_source_op, ..., dt, time)``)."""
    NC, XMAX = 40, 1.0
    T_RAIN, RATE, T_END = 0.05, 0.5, 0.12

    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
    ])

    def ic(x):
        return np.array([1.0])

    model = RainSourceTracer(
        parameters={"a": (1.0, "real"), "rain_rate": (RATE, "nonnegative"),
                    "T_rain": (T_RAIN, "positive")},
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )
    mesh = LSQMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=1))

    # Fixed dt for the same reason as the aux test above: this measures a time
    # integral, not stability, and the suite CFL's dt = 0.0225 straddles T_rain
    # badly enough (O(rate*dt) = 0.011) to break a 5e-3 tolerance on quadrature
    # error alone.  dt = 1e-3 divides both T_rain and t_end exactly.
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.constant(dt=1.0e-3))
    Q, _ = solver.solve(mesh, nsm, write_output=False)

    c = np.asarray(Q[0, :NC], dtype=float)
    assert np.all(np.isfinite(c)), "non-finite tracer after rain-source march"
    dx = XMAX / NC
    gain = float(c.sum() * dx) - 1.0 * XMAX

    expected = RATE * T_RAIN * XMAX          # 0.025 — source ON then OFF at T_rain
    unbound = RATE * T_END * XMAX            # 0.060 — if t were dropped in source
    assert abs(gain - expected) < 5e-3, (
        f"accumulated source {gain:.4f} != rate·T_rain·area {expected:.4f} "
        f"(if t were unbound it would be {unbound:.4f}) — the source did not "
        "bind t: time is not threaded through the timestepping into JaxRuntime.source")
