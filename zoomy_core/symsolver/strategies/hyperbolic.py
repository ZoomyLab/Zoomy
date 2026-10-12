"""The explicit hyperbolic solver, written once for every backend.

A :class:`HyperbolicTemplate` holds the configuration of one solver; its
methods are the graphs a backend prints::

    template = HyperbolicTemplate(nsm, cfl=0.9, dimension=2, mood=True)
    PyPrinter().procedure("hyperbolic", (Q, Qaux, p, time, time_end, mesh),
                          template.march)

``nsm`` is the NumericalSystemModel.  ``Q``, ``Qaux``, ``p`` (parameters),
``time``, ``dt`` and ``mesh`` are symbols.  Python ``if`` / ``for`` run at
build time; ``While`` / ``Cond`` run at run time.  The pieces it composes are
the discretization components (:mod:`zoomy_core.symsolver.discretization`).

Ground truth: ``zoomy_jax.fvm.solver_jax.HyperbolicSolver`` (``_advance``,
``_run``).
"""
import sympy as sp

from zoomy_core.symsolver.core import Cond, While
from zoomy_core.symsolver.discretization.residual import (
    explicit_step, full_aux, implicit_step, mood_step, post_step)
from zoomy_core.symsolver.discretization.time import (
    SSP_RK, cfl_dt, checked, face_dt, face_rule, step_size)
from zoomy_core.symsolver.vocabulary import (
    couple, coupled_faces, interface_flux, interface_states, io)


class HyperbolicTemplate:
    """The configuration of the explicit hyperbolic solver.

    ``cfl`` and ``dimension`` set the CFL step; ``dt_constant`` (a number, or
    ``None`` for the adaptive step) a constant step that warns when above it,
    tested every ``cfl_check_every`` steps (1: each step, 0: never).
    ``wave_speed``: the CFL wave speeds from the first stage's Riemann problem
    ("faces": the speeds of the interface states, Audusse et al. 2004 (2.34);
    no eigensolve of its own), or from the cell states of each face ("cells").
    ``mood``: the a-posteriori MOOD guard of the RK step; ``imex``: the
    SystemModel's implicit slots by an implicit step (Lie) instead of in the
    RK step; ``dt_write``: the output interval (``None``: no output).
    ``time_order``: the order of the SSP-RK step (``SSP_RK``); ``None``: the
    reconstruction order."""

    def __init__(self, nsm, *, cfl, dimension, mood=False, dt_write=None,
                 imex=False, cfl_check_every=10, wave_speed="faces",
                 dt_constant=None, time_order=None):
        if time_order is not None and time_order not in SSP_RK:
            raise ValueError(f"time_order {time_order}: SSP-RK of order "
                             f"{sorted(SSP_RK)} only")
        self.nsm = nsm
        self.cfl = cfl
        self.dimension = dimension
        self.mood = mood
        self.dt_write = dt_write
        self.imex = imex
        self.cfl_check_every = cfl_check_every
        self.wave_speed = wave_speed
        self.dt_constant = None if dt_constant is None else sp.Float(dt_constant)
        self.time_order = self.order if time_order is None else time_order

    @property
    def order(self):
        return self.nsm.reconstruction.order

    def step(self, Q, Qaux, p, time, dt, mesh, *, coupled=None, dt_rule=None):
        """One time step ``(Q, Qaux, dt)``.  Explicit: every term in the
        (MOOD-guarded) RK step.  IMEX (Lie): the RK step without the
        SystemModel's implicit slots, then implicit diffusion and implicit
        Euler on the implicit source slot.  With ``dt_rule`` the first stage
        picks ``dt`` from its own face wave speeds.  ``coupled``: the faces
        carrying a shared answer, ``(mask, flux)``."""
        nsm, order, imex = self.nsm, self.order, self.imex
        if self.mood:
            Q = mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled,
                          implicit_terms=not imex, rk=self.time_order,
                          dt_rule=dt_rule)
        else:
            Q = explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order,
                              coupled=coupled, implicit_terms=not imex,
                              rk=self.time_order, dt_rule=dt_rule)
        if dt_rule is not None:
            Q, dt = Q
        if imex:
            Q = implicit_step(nsm, Q, Qaux, p, time + dt, dt, mesh)
        return (*post_step(nsm, Q, Qaux, p, time + dt, dt, mesh), dt)

    def dt(self, Q, Qaux, p, mesh):
        """The CFL step from the cell states of each face."""
        return cfl_dt(self.nsm, Q, Qaux, p, mesh, self.cfl, self.dimension)

    def initial_aux(self, Q, Qaux, p, mesh):
        """The aux of the initial state: pointwise formula + derivative aux."""
        return full_aux(self.nsm, Q, Qaux, p, sp.S.Zero, sp.S.One, mesh)

    def march(self, Q, Qaux, p, time, time_end, mesh):
        """March from ``time`` to ``time_end``."""
        nsm, cfl, dimension = self.nsm, self.cfl, self.dimension
        dt_constant, cfl_check_every = self.dt_constant, self.cfl_check_every

        def step(Q, Qaux, time, cap):
            if self.wave_speed == "faces":
                return self.step(Q, Qaux, p, time, None, mesh,
                                 dt_rule=face_rule(
                                     nsm, mesh, cfl, dimension, dt_constant,
                                     time, cap, cfl_check_every))
            dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                                  dt_constant, time, cfl_check_every), cap)
            return self.step(Q, Qaux, p, time, dt, mesh)

        return _time_loop(Q, Qaux, time, time_end, self.dt_write, step)

    def precice_march(self, Q, Qaux, p, time, mesh, *, side, subcycling=False):
        """The same march as one preCICE participant: preCICE ends the loop,
        the coupled faces carry the shared interface answer on (own, peer).

        subcycling=False: one step per window (dt = window; warns above the CFL
            step).  Both participants pose the same Riemann problem with the
            same dt, so mass is conserved exactly.
        subcycling=True: adaptive steps inside the window, peer state held at
            its window value.  Stable, but the two sides charge different
            fluxes: mass is NOT exactly conserved (that would need
            refluxing)."""
        nsm, cfl, dimension = self.nsm, self.cfl, self.dimension
        dt_constant, cfl_check_every = self.dt_constant, self.cfl_check_every
        dt_write, wave_speed = self.dt_write, self.wave_speed

        def body(time, Q, Qaux, going, n_write):
            window = couple.max_dt()
            if subcycling:                          # adaptive inside the window
                cap = window if dt_write is None else sp.Min(
                    window, n_write * dt_write - time)
                if wave_speed == "faces":
                    rule = face_rule(nsm, mesh, cfl, dimension, dt_constant,
                                     time, cap, cfl_check_every)
                else:
                    dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                                          dt_constant, time, cfl_check_every),
                                cap)
            elif wave_speed == "faces":             # one step per window
                rule = lambda S: checked(window, time, cfl_check_every,
                                         lambda: face_dt(nsm, S, mesh, cfl,
                                                         dimension))
            else:
                dt = checked(window, time, cfl_check_every,
                             lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl,
                                            dimension))
            peer = couple.read()
            Wc = interface_flux(nsm.call.numerical_face_sides,
                                nsm.call.update_aux_variables or sp.S.false,
                                side, Q, Qaux, peer, p, mesh)
            if wave_speed == "faces":
                Q, Qaux, dt = self.step(Q, Qaux, p, time, None, mesh,
                                        coupled=(coupled_faces(mesh), Wc),
                                        dt_rule=rule)
            else:
                Q, Qaux, dt = self.step(Q, Qaux, p, time, dt, mesh,
                                        coupled=(coupled_faces(mesh), Wc))
            couple.write(interface_states(Q, mesh))
            going = couple.advance(dt)
            time = time + dt
            if dt_write is not None:
                n_write = Cond(
                    time >= (n_write - sp.Rational(1, 10**9)) * dt_write,
                    lambda: _write(time, Q, Qaux, n_write),
                    lambda: n_write)
            return time, Q, Qaux, going, n_write

        going = couple.init(interface_states(Q, mesh))
        if dt_write is not None:
            io.write(time, Q, Qaux)
        time, Q, Qaux, *_ = While(
            lambda time, Q, Qaux, going, n_write: going, body,
            (time, Q, Qaux, going, 1))
        return time, Q, Qaux


def _time_loop(Q, Qaux, time, time_end, dt_write, step):
    """The time loop: ``step(Q, Qaux, time, cap) -> (Q, Qaux, dt)`` takes a
    step ``dt <= cap``, ``cap`` the distance to ``time_end`` and to the next
    write instant."""
    def body(time, Q, Qaux, n_write):
        cap = time_end - time
        if dt_write is not None:                    # land on write instants
            cap = sp.Min(cap, n_write * dt_write - time)
        Q, Qaux, dt = step(Q, Qaux, time, cap)
        time = time + dt
        if dt_write is not None:
            n_write = Cond(time >= (n_write - sp.Rational(1, 10**9)) * dt_write,
                           lambda: _write(time, Q, Qaux, n_write),
                           lambda: n_write)
        return time, Q, Qaux, n_write

    if dt_write is not None:
        io.write(time, Q, Qaux)
    time, Q, Qaux, _ = While(lambda time, Q, Qaux, n_write: time < time_end,
                             body, (time, Q, Qaux, 1))
    return time, Q, Qaux


def _write(time, Q, Qaux, n_write):
    io.write(time, Q, Qaux)
    return n_write + 1
