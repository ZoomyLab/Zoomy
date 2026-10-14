"""The Chorin projection solver, written once for every backend: a hyperbolic predictor,
a pressure solve and a correction, from the three systems of the model's pressure
split (`pred`, `press`, `corr`). `time_order` 2 / 3 wraps stages in SSP-RK2 /
RK3 and `imex=Imex.ssp2` uses IMEX-SSP2(2,2,2) with projected stages.
"""

import sympy as sp

from zoomy_core.misc.options import Option, typed
from zoomy_core.symsolver.core import Cond, LinSolve, While
from zoomy_core.symsolver.discretization.residual import (
    explicit_step, full_aux, imex_step, implicit_step, mood_step)
from zoomy_core.symsolver.discretization.time import IMEX_RK, SSP_RK, Imex, cfl_dt, checked
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate, _write
from zoomy_core.symsolver.vocabulary import (
    boundary_ghosts, cells, couple, derivative_aux, diffusion, halo, io, overlap_cells,
    overlap_fill, overlap_states, put, reduce_max, select, take, zeros)


class Projection(Option):
    """What a projection stage solves for: the pressure, or its increment over a lagged one.
    """
    #: the full pressure each stage (Almgren, Bell, Crutchfield 2000, version (2))
    standard = "standard"
    #: correct with the lagged pressure ``P^n``, solve the same ``press`` system for
    #: ``dP``, store ``P^n + dP`` (their version (1)): the approximate projection's defect
    #: then acts on ``dP = O(dt)`` only, so the scheme keeps its time order on a fixed mesh.
    #: The first step lags the initial state's pressure rows (zero: a standard projection)
    increment = "increment"


class ChorinTemplate(HyperbolicTemplate):
    """The projection solver of a pressure split `pred` / `press` / `corr`, with
    `nsm` the predictor `pred`; `step` is the projection step and `march`,
    `dt` and `initial_aux` are inherited from `HyperbolicTemplate`.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
        Almgren, Bell, Crutchfield 2000, https://doi.org/10.1137/S1064827599357024
    """

    def __init__(self, pred, press, corr, *, cfl, time_order=1,
                 pressure_newton=None, projection=Projection.standard, **config):
        """Set the split and the configuration of `HyperbolicTemplate`; `cfl`
        is divided by `2 order - 1`.

        Args:
            pressure_newton: `(tol, maxit)` for a nonlinear pressure constraint solved
                by Newton, else `None`.
            projection: a `Projection`.
        """
        super().__init__(pred, cfl=cfl / (2 * pred.reconstruction.order - 1),
                         time_order=time_order, **config)
        self.press = press
        self.corr = corr
        self.pressure_newton = pressure_newton
        self.projection = typed(Projection, projection, "projection")

    def step(self, Q, Qaux, p, time, dt, mesh, *, dt_rule=None, dirichlet=None):
        """One time step: the outer SSP-RK over projection stages; with `imex=Imex.ssp2`
        every stage solves the implicit source and is then projected.

        Args:
            dt_rule: the first stage picks `dt`.
            dirichlet: `(mask, P)`: the pressure solves keep the cells `mask` at the step's
                pressure `P` (the overlap of a Schwarz-coupled participant,
                `precice_march`); one projection stage only.
        Returns:
            `(Q, Qaux, dt)`.
        """
        if dirichlet is not None and (self.imex is Imex.ssp2 or self.time_order != 1):
            raise NotImplementedError("Dirichlet pressure data (the coupled march) with "
                                      "several projection stages")
        # Imex.ssp2: the pressure joins the implicit part S (the source and the two-point
        # part of the implicit diffusion, `implicit_stage`) in the tableau as the stage
        # correction B_i = A_ii dt G P_i (off the pressure rows); f, S act on the
        # projected stages Y_i = P(R_i):
        #   R_i = Q + sum_{j<i} (At_ij F_j + A_ij (S_j + B_j / A_jj)) + A_ii dt S(R_i)
        #   B_i = Y_i - R_i,  F_i = dt f(Y_i),  S_i = dt S(Y_i)
        #   Q'  = P(Q + sum_i (wt_i F_i + w_i (S_i + B_i / A_ii)))
        # increment: the lag of stage i (and of the final P) is cl_i P^n, cl_i the projection
        # pressure the stage takes from a flux-driven pressure P^n (DG P^n = D f): stage i
        # removes D R_i / dt = sum_{j<i} (At_ij - A_ij cl_j / A_jj) D f, so cl_i is that sum
        # (SSP2: cl = (0, 1), final 1 - 1/(2 g)); the stored pressure is the step's total
        # P^{n+1} = sum_i w_i P_i / A_ii + P_final.  A source-driven part of the pressure is
        # not lagged (first order in that part only).
        if self.imex is Imex.ssp2:
            rows = tuple(self.press.equation_to_state_index)
            tab, incr = IMEX_RK[self.imex], self.projection is Projection.increment
            cl = []
            for i in range(len(tab.c)):
                cl.append(sum(tab.At[i][j] - tab.A[i][j] * cl[j] / tab.A[j][j]
                              for j in range(i)))
            cl.append(sum(tab.wt) - sum(w * cj / tab.A[j][j]
                                        for j, (w, cj) in enumerate(zip(tab.w, cl))))
            Pn, P_tab = take(Q, rows), []

            def project(Y, t, dt, i=None):  # and its update off the pressure rows
                """Project stage `i` (`None`: the final combination); returns it with
                its update off the pressure rows.
                """
                k = len(tab.c) if i is None else i
                lag = cl[k] * Pn if incr and cl[k] != 0 else None
                Yp = self.project(Y, p, t, dt, mesh, lag)
                if incr and i is not None:
                    P_tab.append(tab.w[i] * take(Yp, rows) / tab.A[i][i])
                elif incr:
                    Yp = put(Yp, rows, halo(take(Yp, rows) + sum(P_tab), mesh))
                return Yp, put(Yp, rows, take(Y, rows)) - Y
            Q = imex_step(self.nsm, Q, Qaux, p, time, dt, mesh, self.order,
                          tab, self.mood, dt_rule=dt_rule, project=project)
            if dt_rule is not None:
                Q, dt = Q
            return Q, full_aux(self.nsm, Q, Qaux, p, time + dt, dt, mesh), dt
        Q0 = Q
        for i, (a, c) in enumerate(SSP_RK[self.time_order]):
            rule = dt_rule if i == 0 else None
            Q_stage, dt = self.stage(Q, Qaux, p, time + c * dt if i else time,
                                     dt, mesh, rule, dirichlet)
            Q = a * Q0 + (1 - a) * Q_stage
        return Q, full_aux(self.nsm, Q, Qaux, p, time + dt, dt, mesh), dt

    def precice_march(self, Q, Qaux, p, time, mesh):
        """The march as one preCICE participant coupled by overlapping Schwarz: the mesh
        reaches across the coupled interface into the peer's domain (the overlap,
        `zoomy_core.mesh.overlap`), one step per time window of an implicit coupling scheme.
        Every coupling iteration takes the whole step from the window start, the overlap
        holding the peer's state at the window start: the predictor and the aux run over the
        overlap too (the overlap must be wide enough that their invalid outer rings do not
        reach the own cells' stencils), the pressure solve keeps the overlap's pressure at the
        peer's of its last iteration (a Dirichlet subdomain solve) and the participant writes
        its mirror cells (`overlap_states`, all rows).  preCICE repeats the window until the
        exchanged data converge; at the fixed point every own-cell equation of the step reads
        the peer's values, the single-domain step (to the coupling and pressure tolerances).
        Constant step = the window, no MOOD, one projection stage, no implicit diffusion
        (a second solve over both domains).

        References:
            Toselli, Widlund 2005 (overlapping Schwarz), https://doi.org/10.1007/b137868
            Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
        """
        nsm, rows = self.nsm, tuple(self.press.equation_to_state_index)
        if self.mood:
            raise NotImplementedError("the coupled Chorin march without MOOD (its "
                                      "demotion is a global decision)")
        if self.imex is Imex.ssp2 or self.time_order != 1:
            raise NotImplementedError("the coupled Chorin march with several projection "
                                      "stages (each needs the peer's stage pressure)")
        if nsm.call.diffusion_matrix is not None and self.imex is not None:
            raise NotImplementedError("the coupled Chorin march with an implicit diffusion "
                                      "(a second solve over both domains)")
        cfl, dimension, every = self.cfl, self.dimension, self.cfl_check_every
        dt_write = self.dt_write

        def window(time, Q, Qaux, going, n_write):
            """One window: coupling iterations of the step from its start until preCICE
            accepts the exchanged data."""
            dt = checked(couple.max_dt(), time, every,
                         lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension))
            Q = overlap_fill(Q, couple.read(), mesh)      # the peer's window start

            def iteration(Q1, Qaux1, going, repeat):
                """The step from the window start, the overlap's pressure the peer's of its
                last iteration; write the mirror, advance."""
                P = take(overlap_fill(Q, couple.read_iterate(), mesh), rows)
                Q1, Qaux1, _ = self.step(Q, Qaux, p, time, dt, mesh,
                                         dirichlet=(overlap_cells(mesh), P))
                couple.write(overlap_states(Q1, mesh))
                going = couple.advance(dt)
                return Q1, Qaux1, going, couple.repeat()

            Q, Qaux, going, _ = While(lambda Q1, Qaux1, going, repeat: repeat, iteration,
                                      (Q, Qaux, going, sp.S.true))
            time = time + dt
            if dt_write is not None:
                n_write = Cond(time >= (n_write - sp.Rational(1, 10**9)) * dt_write,
                               lambda: _write(time, Q, Qaux, n_write), lambda: n_write)
            return time, Q, Qaux, going, n_write

        going = couple.init(overlap_states(Q, mesh))
        if dt_write is not None:
            io.write(time, Q, Qaux)
        time, Q, Qaux, *_ = While(lambda time, Q, Qaux, going, n_write: going, window,
                                  (time, Q, Qaux, going, 1))
        return time, Q, Qaux

    def stage(self, Q, Qaux, p, time, dt, mesh, dt_rule=None, dirichlet=None):
        """One projection stage: predictor, pressure solve, correction; returns
        `(Q, dt)`.

        Args:
            dt_rule: `dt` is chosen only by the predictor's flux, so the stage keeps
                the aux it is given.
            dirichlet: see `step`.
        """
        if dt_rule is None:
            Qaux = full_aux(self.nsm, Q, Qaux, p, time, dt, mesh)
        # increment: the lag is the stage input's pressure (the predictor's flux would
        # diffuse the pressure rows, a wrong lag at the boundaries)
        lag = (take(Q, tuple(self.press.equation_to_state_index))
               if self.projection is Projection.increment else None)
        Q, dt = self.predictor(Q, Qaux, p, time, dt, mesh, dt_rule)
        return self.project(Q, p, time, dt, mesh, lag, dirichlet), dt

    def project(self, Q, p, time, dt, mesh, lag=None, dirichlet=None):
        """The pressure solve and the correction of `Q`.

        Args:
            lag: `None`: solve for the pressure.  Else the lagged pressure `P^n`:
                correct with it, solve `press` on that state for `dP` (the corrector
                is linear in the pressure) and correct `Q` with `P^n + dP`.
            dirichlet: `(mask, P)`: the pressure of the cells `mask` is `P` (with `lag`, the
                increment there is `P - P^n`).
        """
        if lag is None:
            Q = pressure(self.press, Q, p, time, dt, mesh, self.pressure_newton, dirichlet)
            return correct(self.corr, Q, p, time, dt, mesh)
        rows = tuple(self.press.equation_to_state_index)
        Q1 = put(correct(self.corr, put(Q, rows, lag), p, time, dt, mesh), rows,
                 zeros(len(rows), mesh))
        if dirichlet is not None:
            dirichlet = (dirichlet[0], dirichlet[1] - lag)
        dP = take(pressure(self.press, Q1, p, time, dt, mesh, self.pressure_newton,
                           dirichlet), rows)
        return correct(self.corr, put(Q, rows, halo(lag + dP, mesh)), p, time, dt,
                       mesh)

    def predictor(self, Q, Qaux, p, time, dt, mesh, dt_rule=None):
        """Euler flux step, then the implicit slots as Lie sub-steps (implicit with
        `imex`, else explicit Euler); returns `(Q, dt)`.
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


def pressure(press, Q, p, time, dt, mesh, newton=None, dirichlet=None):
    """Solve `press` for its own rows with the rest of `Q` frozen: one linear solve
    (GMRES) of the affine residual `A(P) = R(P) - R(0)`, or Newton from the current
    pressure with `newton=(tol, maxit)`.  `dirichlet=(mask, G)`: the cells `mask` keep the
    pressure `G` (the rows of the cells `mask` are the identity: the subdomain solve of
    overlapping Schwarz, the other cells' equations reading `G` there).

    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
        Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
    """
    rows = tuple(press.equation_to_state_index)
    Qaux = full_aux(press, Q, zeros(len(press.aux_state), mesh), p, time, dt,
                    mesh)

    def R(P):
        """Pressure residual of the constraint at pressure `P`."""
        Y = put(Q, rows, halo(P, mesh))
        Yg = boundary_ghosts(press.call.boundary_ghost_state, Y, Qaux, p, time,
                             mesh)
        Y_aux = derivative_aux(press.call.aux_derivatives, Y, Qaux, mesh, Yg,
                               rows)
        return cells(press.call.source, Y, Y_aux, p, time, dt, mesh)

    # the rows of the held cells: `fixed(their value, the residual's)`
    if dirichlet is None:
        fixed, held, free = (lambda A, B: B), (lambda P: P), (lambda r: r)
    else:
        mask, G = dirichlet
        fixed = lambda A, B: select(mask, A, B)                        # noqa: E731
        held = lambda P: select(mask, G, P)                            # noqa: E731
        free = lambda r: select(mask, sp.S.Zero, r)                    # noqa: E731
    if newton is None:
        b = R(sp.S.Zero)
        rhs = -b if dirichlet is None else fixed(G, -b)
        P = LinSolve(lambda P: fixed(P, R(P) - b), rhs)
        return put(Q, rows, halo(held(P), mesh))
    tol, maxit = newton

    def newton_step(P, r, k):
        """One Newton update `P + d` with `d` from a linear solve at the iterate.
        """
        P = P + LinSolve(lambda d: fixed(d, R(P + d) - r), -r)
        return P, free(R(P)), k + 1

    P0 = held(take(Q, rows))
    P, _, _ = While(lambda P, r, k: sp.And(k < maxit, reduce_max(abs(r)) > tol),
                    newton_step, (P0, free(R(P0)), 0))
    return put(Q, rows, halo(held(P), mesh))


def correct(corr, Q, p, time, dt, mesh):
    """Overwrite the rows of `corr` by its update formula (the correction by the
    solved pressure).
    """
    Qaux = full_aux(corr, Q, zeros(len(corr.aux_state), mesh), p, time, dt,
                    mesh)
    return halo(put(Q, tuple(corr.equation_to_state_index),
                    cells(corr.call.update_variables, Q, Qaux, p, time, dt,
                          mesh)), mesh)


def predictor_nsm(pred, press, corr, reconstruction=None):
    """The predictor NSM of a pressure split on the full state (`pad_to_square`): positive
    Rusanov, the hydrostatic rescaling on every row but `h`, `b` and the pressure rows, `press`
    and `corr` sizing the LSQ stencil; `reconstruction` a `ReconstructionSpec` (default order 1).
    Pass it with `press`, `corr` to a symbolic Chorin solver's `setup`.
    """
    from zoomy_core.numerics import (NumericalSystemModel, PositiveNonconservativeRusanov,
                                     ReconstructionSpec)
    from zoomy_core.systemmodel.padding import pad_to_square
    square = pad_to_square(pred)
    names = [str(s) for s in square.state]
    fixed = {names.index("h"), names.index("b"), *map(int, press.equation_to_state_index)}
    return NumericalSystemModel.from_system_model(
        square, riemann=PositiveNonconservativeRusanov,
        reconstruction=reconstruction or ReconstructionSpec(),
        additional_systems=[press, corr],
        scaled_q_indices=[i for i in range(square.n_variables) if i not in fixed])
