"""The Chorin projection solver, written once for every backend: a hyperbolic predictor,
a pressure solve and a correction, from the three systems of the model's pressure
split (``pred``, ``press``, ``corr``). ``time_order`` 2 / 3 wraps stages in SSP-RK2 /
RK3 and ``imex="ssp2"`` uses IMEX-SSP2(2,2,2) with projected stages.
"""
import sympy as sp

from zoomy_core.symsolver.core import LinSolve, While
from zoomy_core.symsolver.discretization.residual import (
    explicit_step, full_aux, imex_step, implicit_step, mood_step)
from zoomy_core.symsolver.discretization.time import IMEX_RK, SSP_RK, Imex
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.symsolver.vocabulary import (
    boundary_ghosts, cells, derivative_aux, diffusion, halo, put, reduce_max,
    take, zeros)


class ChorinTemplate(HyperbolicTemplate):
    """The projection solver of a pressure split ``pred`` / ``press`` / ``corr``, with
    ``nsm`` the predictor ``pred``; ``step`` is the projection step and ``march``,
    ``dt`` and ``initial_aux`` are inherited from :class:`HyperbolicTemplate`.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """

    def __init__(self, pred, press, corr, *, cfl, time_order=1,
                 pressure_newton=None, **config):
        """Set the split and the configuration of :class:`HyperbolicTemplate`; ``cfl``
        is divided by ``2 order - 1``.

        Args:
            pressure_newton: ``(tol, maxit)`` for a nonlinear pressure constraint solved
                by Newton, else ``None``.
        """
        super().__init__(pred, cfl=cfl / (2 * pred.reconstruction.order - 1),
                         time_order=time_order, **config)
        self.press = press
        self.corr = corr
        self.pressure_newton = pressure_newton

    def step(self, Q, Qaux, p, time, dt, mesh, *, dt_rule=None):
        """One time step: the outer SSP-RK over projection stages; with ``imex="ssp2"``
        every stage solves the implicit source and is then projected.

        Args:
            dt_rule: the first stage picks ``dt``.
        Returns:
            ``(Q, Qaux, dt)``.
        """
        # imex="ssp2": the pressure joins the implicit source S in the tableau as the
        # stage correction B_i = A_ii dt G P_i (off the pressure rows); f, S act on
        # the projected stages Y_i = P(R_i):
        #   R_i = Q + sum_{j<i} (At_ij F_j + A_ij (S_j + B_j / A_jj)) + A_ii dt S(R_i)
        #   B_i = Y_i - R_i,  F_i = dt f(Y_i),  S_i = dt S(Y_i)
        #   Q'  = P(Q + sum_i (wt_i F_i + w_i (S_i + B_i / A_ii)))
        if self.imex is Imex.ssp2:
            rows = tuple(self.press.equation_to_state_index)

            def project(Y, t, dt):          # and its update off the pressure rows
                """Project a stage; returns it with its update off the pressure rows.
                """
                Yp = self.project(Y, p, t, dt, mesh)
                return Yp, put(Yp, rows, take(Y, rows)) - Y
            Q = imex_step(self.nsm, Q, Qaux, p, time, dt, mesh, self.order,
                          IMEX_RK[self.imex], self.mood, dt_rule=dt_rule,
                          project=project)
            if dt_rule is not None:
                Q, dt = Q
            return Q, full_aux(self.nsm, Q, Qaux, p, time + dt, dt, mesh), dt
        Q0 = Q
        for i, (a, c) in enumerate(SSP_RK[self.time_order]):
            rule = dt_rule if i == 0 else None
            Q_stage, dt = self.stage(Q, Qaux, p, time + c * dt if i else time,
                                     dt, mesh, rule)
            Q = a * Q0 + (1 - a) * Q_stage
        return Q, full_aux(self.nsm, Q, Qaux, p, time + dt, dt, mesh), dt

    def precice_march(self, *args, **kwargs):
        """Not supported: raises ``NotImplementedError``."""
        raise NotImplementedError("Chorin with preCICE coupling")

    def stage(self, Q, Qaux, p, time, dt, mesh, dt_rule=None):
        """One projection stage: predictor, pressure solve, correction; returns
        ``(Q, dt)``.

        Args:
            dt_rule: ``dt`` is chosen only by the predictor's flux, so the stage keeps
                the aux it is given.
        """
        if dt_rule is None:
            Qaux = full_aux(self.nsm, Q, Qaux, p, time, dt, mesh)
        Q, dt = self.predictor(Q, Qaux, p, time, dt, mesh, dt_rule)
        return self.project(Q, p, time, dt, mesh), dt

    def project(self, Q, p, time, dt, mesh):
        """The pressure solve and the correction of ``Q``."""
        Q = pressure(self.press, Q, p, time, dt, mesh, self.pressure_newton)
        return correct(self.corr, Q, p, time, dt, mesh)

    def predictor(self, Q, Qaux, p, time, dt, mesh, dt_rule=None):
        """Euler flux step, then the implicit slots as Lie sub-steps (implicit with
        ``imex``, else explicit Euler); returns ``(Q, dt)``.
        """
        pred = self.nsm
        step = mood_step if self.mood else explicit_step
        Q = step(pred, Q, Qaux, p, time, dt, mesh, self.order,
                 implicit_terms=False, rk=1, dt_rule=dt_rule)
        if dt_rule is not None:
            Q, dt = Q
        if self.imex is Imex.lie:
            return implicit_step(pred, Q, Qaux, p, time + dt, dt, mesh), dt
        if pred.call.source is not None:
            Q = Q + dt * cells(pred.call.source, Q, Qaux, p, time, dt, mesh)
        if pred.call.diffusion_matrix is not None:
            Q = halo(Q + dt * diffusion(pred.call.diffusion_matrix,
                                        pred.call.boundary_ghost_state, Q,
                                        Qaux, p, time, mesh), mesh)
        return Q, dt


def pressure(press, Q, p, time, dt, mesh, newton=None):
    """Solve ``press`` for its own rows with the rest of ``Q`` frozen: one linear solve
    (GMRES) of the affine residual ``A(P) = R(P) - R(0)``, or Newton from the current
    pressure with ``newton=(tol, maxit)``.

    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
        Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
    """
    rows = tuple(press.equation_to_state_index)
    Qaux = full_aux(press, Q, zeros(len(press.aux_state), mesh), p, time, dt,
                    mesh)

    def R(P):
        """Pressure residual of the constraint at pressure ``P``."""
        Y = put(Q, rows, halo(P, mesh))
        Yg = boundary_ghosts(press.call.boundary_ghost_state, Y, Qaux, p, time,
                             mesh)
        Y_aux = derivative_aux(press.call.aux_derivatives, Y, Qaux, mesh, Yg,
                               rows)
        return cells(press.call.source, Y, Y_aux, p, time, dt, mesh)

    if newton is None:
        b = R(sp.S.Zero)
        return put(Q, rows, halo(LinSolve(lambda P: R(P) - b, -b), mesh))
    tol, maxit = newton

    def newton_step(P, r, k):
        """One Newton update ``P + d`` with ``d`` from a linear solve at the iterate.
        """
        P = P + LinSolve(lambda d: R(P + d) - r, -r)
        return P, R(P), k + 1

    P0 = take(Q, rows)
    P, _, _ = While(lambda P, r, k: sp.And(k < maxit, reduce_max(abs(r)) > tol),
                    newton_step, (P0, R(P0), 0))
    return put(Q, rows, halo(P, mesh))


def correct(corr, Q, p, time, dt, mesh):
    """Overwrite the rows of ``corr`` by its update formula (the correction by the
    solved pressure).
    """
    Qaux = full_aux(corr, Q, zeros(len(corr.aux_state), mesh), p, time, dt,
                    mesh)
    return halo(put(Q, tuple(corr.equation_to_state_index),
                    cells(corr.call.update_variables, Q, Qaux, p, time, dt,
                          mesh)), mesh)
