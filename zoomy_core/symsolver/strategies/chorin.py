"""The Chorin projection solver, written once for every backend.

The model's pressure split gives three systems:

* ``pred``: the hyperbolic system without the pressure (padded to the full
  state, so the pressure rows stay frozen in the predictor);
* ``press``: the elliptic constraint, its equations on the pressure rows;
* ``corr``: the correction of the remaining rows by the solved pressure.

One projection stage is a hyperbolic Euler step, a pressure solve and a
correction.  ``time_order=2`` / ``3`` wraps two / three stages in the outer
SSP-RK2 / RK3.

Ground truth: ``zoomy_jax.fvm.solver_chorin_vam_jax.ChorinSplitVAMSolverJax``
(``chorin_stage``, ``chorin_cycle``, ``_step_pressure_pure``,
``_step_corrector_pure``, ``_cfl_dt``).
"""
import sympy as sp

from zoomy_core.symsolver.core import LinSolve, While
from zoomy_core.symsolver.discretization.residual import (
    explicit_step, full_aux, implicit_step, mood_step)
from zoomy_core.symsolver.discretization.time import SSP_RK
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.symsolver.vocabulary import (
    boundary_ghosts, cells, derivative_aux, diffusion, halo, put, reduce_max,
    take, zeros)


class ChorinTemplate(HyperbolicTemplate):
    """The projection solver of a pressure split ``pred`` / ``press`` /
    ``corr``; ``nsm`` is the predictor ``pred``.  ``config`` as in
    :class:`HyperbolicTemplate`, whose ``march``, ``dt`` and ``initial_aux``
    (on the predictor) it inherits; ``step`` is the projection step.  The CFL
    step carries the reference's per-degree factor 1 / (2 order - 1), applied
    to ``cfl`` here.  ``pressure_newton=(tol, maxit)``: a nonlinear pressure
    constraint solved by Newton (see :func:`pressure`)."""

    def __init__(self, pred, press, corr, *, cfl, time_order=1,
                 pressure_newton=None, **config):
        super().__init__(pred, cfl=cfl / (2 * pred.reconstruction.order - 1),
                         time_order=time_order, **config)
        self.press = press
        self.corr = corr
        self.pressure_newton = pressure_newton

    def step(self, Q, Qaux, p, time, dt, mesh, *, dt_rule=None):
        """One time step: the outer SSP-RK over full projection stages.
        Returns ``(Q, Qaux, dt)``; with ``dt_rule`` the first stage picks
        ``dt``."""
        Q0 = Q
        for i, (a, c) in enumerate(SSP_RK[self.time_order]):
            rule = dt_rule if i == 0 else None
            Q_stage, dt = self.stage(Q, Qaux, p, time + c * dt if i else time,
                                     dt, mesh, rule)
            Q = a * Q0 + (1 - a) * Q_stage
        return Q, full_aux(self.nsm, Q, Qaux, p, time + dt, dt, mesh), dt

    def precice_march(self, *args, **kwargs):
        raise NotImplementedError("Chorin with preCICE coupling")

    def stage(self, Q, Qaux, p, time, dt, mesh, dt_rule=None):
        """One projection stage: predictor, pressure solve, correction.
        Returns ``(Q, dt)``.  Under ``dt_rule`` the stage keeps the aux it is
        given (the step's own): ``dt`` is chosen only by the predictor's
        flux."""
        if dt_rule is None:
            Qaux = full_aux(self.nsm, Q, Qaux, p, time, dt, mesh)
        Q, dt = self.predictor(Q, Qaux, p, time, dt, mesh, dt_rule)
        Q = pressure(self.press, Q, p, time, dt, mesh, self.pressure_newton)
        return correct(self.corr, Q, p, time, dt, mesh), dt

    def predictor(self, Q, Qaux, p, time, dt, mesh, dt_rule=None):
        """Euler flux step, then the SystemModel's implicit slots as Lie
        sub-steps: implicit (``imex``) or explicit Euler (the reference).
        Returns ``(Q, dt)``; with ``dt_rule`` the flux step picks ``dt``."""
        pred = self.nsm
        step = mood_step if self.mood else explicit_step
        Q = step(pred, Q, Qaux, p, time, dt, mesh, self.order,
                 implicit_terms=False, rk=1, dt_rule=dt_rule)
        if dt_rule is not None:
            Q, dt = Q
        if self.imex:
            return implicit_step(pred, Q, Qaux, p, time + dt, dt, mesh), dt
        if pred.call.source is not None:
            Q = Q + dt * cells(pred.call.source, Q, Qaux, p, time, dt, mesh)
        if pred.call.diffusion_matrix is not None:
            Q = halo(Q + dt * diffusion(pred.call.diffusion_matrix, Q, Qaux, p,
                                        mesh), mesh)
        return Q, dt


def pressure(press, Q, p, time, dt, mesh, newton=None):
    """Solve ``press`` for its own rows with the rest of ``Q`` frozen.  The
    pressure derivatives see the system's boundary conditions.

    Default: the residual is affine in the pressure, A(P) = R(P) - R(0),
    solved once.  ``newton=(tol, maxit)``: a nonlinear constraint, Newton
    from the current pressure, each step the same linear solve at the
    iterate, until max|R| <= tol."""
    rows = tuple(press.equation_to_state_index)
    Qaux = full_aux(press, Q, zeros(len(press.aux_state), mesh), p, time, dt,
                    mesh)

    def R(P):
        Y = put(Q, rows, halo(P, mesh))
        Yb = boundary_ghosts(press.call.boundary_conditions, Y, Qaux, p, time,
                             mesh)
        Y_aux = derivative_aux(press.call.aux_derivatives, Y, Qaux, mesh, Yb,
                               rows)
        return cells(press.call.source, Y, Y_aux, p, time, dt, mesh)

    if newton is None:
        b = R(sp.S.Zero)
        return put(Q, rows, halo(LinSolve(lambda P: R(P) - b, -b), mesh))
    tol, maxit = newton

    def newton_step(P, r, k):
        P = P + LinSolve(lambda d: R(P + d) - r, -r)
        return P, R(P), k + 1

    P0 = take(Q, rows)
    P, _, _ = While(lambda P, r, k: sp.And(k < maxit, reduce_max(abs(r)) > tol),
                    newton_step, (P0, R(P0), 0))
    return put(Q, rows, halo(P, mesh))


def correct(corr, Q, p, time, dt, mesh):
    """Overwrite ``corr``'s rows by its update formula."""
    Qaux = full_aux(corr, Q, zeros(len(corr.aux_state), mesh), p, time, dt,
                    mesh)
    return halo(put(Q, tuple(corr.equation_to_state_index),
                    cells(corr.call.update_variables, Q, Qaux, p, time, dt,
                          mesh)), mesh)
