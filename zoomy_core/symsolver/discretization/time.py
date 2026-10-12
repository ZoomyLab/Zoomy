"""Time discretization of the solver templates: the SSP-RK stages and the
CFL step size.

``nsm`` is the NumericalSystemModel; ``Q``, ``Qaux``, ``p`` (parameters),
``time`` and ``mesh`` are symbols.  Python ``if`` / ``for`` run at build time;
``Cond`` runs at run time.
"""
import sympy as sp

from zoomy_core.symsolver.core import Cond
from zoomy_core.symsolver.vocabulary import (
    check, face_inradius, face_wavespeed, minimum, reduce_min)

# Shu-Osher form: Q <- a*Q0 + (1 - a)*(Q + dt*L(Q, t + c*dt)), per stage (a, c);
# c is the time of the stage's input state (RK3: Q0 at t, Q1 at t + dt, Q2 at
# t + dt/2), which time-dependent sources and boundary conditions see.
SSP_RK = {1: ((0, 0),),
          2: ((0, 0), (sp.S.Half, 1)),
          3: ((0, 0), (sp.Rational(3, 4), 1), (sp.Rational(1, 3), sp.S.Half))}


def cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension):
    """dt <= CFL * 2 r_f / (d |lambda|_f), min over faces, capped by dt_max;
    the wave speeds from the cell states of each face."""
    speed = (nsm.call.eigenvalues_cfl if nsm.call.eigenvalues_cfl is not None
             else nsm.call.eigenvalues)
    return face_dt(nsm, face_wavespeed(speed, Q, Qaux, p, mesh), mesh, cfl,
                   dimension)


def face_dt(nsm, speed, mesh, cfl, dimension):
    """The CFL step from face wave speeds."""
    dt = reduce_min(cfl * 2 * face_inradius(mesh) / (dimension * speed))
    return dt if nsm.dt_max is None else minimum(dt, nsm.dt_max)


def step_size(nsm, Q, Qaux, p, mesh, cfl, dimension, dt_constant, time,
              check_every=10):
    """Adaptive CFL step, or a constant step that warns when above it."""
    if dt_constant is None:
        return cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension)
    return checked(dt_constant, time, check_every,
                   lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension))


def face_rule(nsm, mesh, cfl, dimension, dt_constant, time, cap, check_every=10):
    """``dt_rule`` taking the step from the first stage's face wave speeds
    (no eigensolve of its own); capped by ``cap``."""
    if dt_constant is None:
        return lambda S: minimum(face_dt(nsm, S, mesh, cfl, dimension), cap)
    return lambda S: minimum(checked(dt_constant, time, check_every,
                                     lambda: face_dt(nsm, S, mesh, cfl,
                                                     dimension)), cap)


def checked(dt, time, every, dt_cfl):
    """``dt``, warning (once) if it exceeds the CFL step ``dt_cfl()``; tested
    every ``every`` steps of size ``dt`` (1: each step, 0: never)."""
    def check_now():
        check.cfl(dt_cfl(), dt)
        return sp.S.Zero

    if every == 1:
        check_now()
    elif every > 1:
        step = sp.floor(time / dt + sp.S.Half)
        Cond(sp.Eq(sp.Mod(step, every), 0), check_now, lambda: sp.S.Zero)
    return dt
