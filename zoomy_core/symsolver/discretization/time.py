"""Time discretization of the solver templates: the SSP-RK stages, the IMEX Runge-Kutta
tableaux and the CFL step size.
"""
from typing import NamedTuple

import sympy as sp

from zoomy_core.misc.options import Option
from zoomy_core.symsolver.core import Cond
from zoomy_core.symsolver.vocabulary import (
    check, face_inradius, face_wavespeed, minimum, reduce_min)

# Shu-Osher form: Q <- a*Q0 + (1 - a)*(Q + dt*L(Q, t + c*dt)), per stage (a, c);
# c is the time of the stage's input state (RK3: Q0 at t, Q1 at t + dt, Q2 at
# t + dt/2), which time-dependent sources and boundary conditions see.
SSP_RK = {1: ((0, 0),),
          2: ((0, 0), (sp.S.Half, 1)),
          3: ((0, 0), (sp.Rational(3, 4), 1), (sp.Rational(1, 3), sp.S.Half))}


class Imex(Option):
    """How a step treats the SystemModel's implicit slots: `lie` splits them off after
    the SSP-RK step (first order), `ssp2` solves them in every stage of
    IMEX-SSP2(2,2,2).
    """
    #: Lie splitting: the explicit SSP-RK step, then implicit diffusion and
    #: implicit Euler on the source (first order)
    lie = "lie"
    #: IMEX-SSP2(2,2,2) (:data:`IMEX_RK`): the implicit source and the two-point part
    #: of the implicit diffusion in every stage (second order)
    ssp2 = "ssp2"


class CflWaveSpeed(Option):
    """Where the CFL step takes its wave speeds."""
    #: from the first RK stage's Riemann problems; at order 2 with an edge source (the NCP
    #: inside each cell) also from the cell states, which that term moves
    faces = "faces"
    #: from the cell states on each face
    cells = "cells"


class ImexTableau(NamedTuple):
    """Double Butcher tableau: explicit `(At, wt, ct)` and diagonally implicit
    `(A, w, c)`.

    References:
        Pareschi, Russo 2005, https://doi.org/10.1007/s10915-004-4636-4
    """
    At: tuple
    wt: tuple
    ct: tuple
    A: tuple
    w: tuple
    c: tuple


# IMEX-SSP2(2,2,2), Pareschi & Russo 2005, Table II (J. Sci. Comput. 25:129-155,
# doi:10.1007/s10915-004-4636-4), their eqs. (3)-(5):
#   Y_i = Q + dt sum_{j<i} At_ij f(Y_j) + dt sum_{j<=i} A_ij g(Y_j)
#   Q'  = Q + dt sum_i (wt_i f(Y_i) + w_i g(Y_i))
#   explicit (SSP2):  At = [[0, 0], [1, 0]],          wt = (1/2, 1/2), ct = (0, 1)
#   implicit (L-stable DIRK): A = [[g, 0], [1-2g, g]], w = (1/2, 1/2), c = (g, 1-g)
#   g = 1 - 1/sqrt(2)  (a Float: the printers know no sqrt of a constant)
_G = (1 - 1 / sp.sqrt(2)).evalf(20)
IMEX_RK = {Imex.ssp2: ImexTableau(
    At=((0, 0), (1, 0)), wt=(sp.S.Half, sp.S.Half), ct=(0, 1),
    A=((_G, 0), (1 - 2 * _G, _G)), w=(sp.S.Half, sp.S.Half), c=(_G, 1 - _G))}


def cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension):
    """CFL step `cfl * 2 r_f / (d |lambda|_f)`, min over faces, from the wave speeds
    of the cell states of each face and capped by `dt_max`.

    References:
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """
    return face_dt(nsm, cell_wavespeed(nsm, Q, Qaux, p, mesh), mesh, cfl,
                   dimension)


def cell_wavespeed(nsm, Q, Qaux, p, mesh):
    """Max `|eigenvalue|` per face over the cell states of its two cells."""
    speed = (nsm.call.eigenvalues_cfl if nsm.call.eigenvalues_cfl is not None
             else nsm.call.eigenvalues)
    return face_wavespeed(speed, Q, Qaux, p, mesh)


def face_dt(nsm, speed, mesh, cfl, dimension):
    """The CFL step from the face wave speeds `speed`, capped by `dt_max`.

    References:
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """
    dt = reduce_min(cfl * 2 * face_inradius(mesh) / (dimension * speed))
    return dt if nsm.dt_max is None else minimum(dt, nsm.dt_max)


def step_size(nsm, Q, Qaux, p, mesh, cfl, dimension, dt_constant, time,
              check_every=10):
    """Adaptive CFL step, or a constant step that warns when above it."""
    if dt_constant is None:
        return cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension)
    return checked(dt_constant, time, check_every,
                   lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension))


#: The landing tolerance of a step on the next instant (end time, write, checkpoint): a
#: leftover distance below this fraction of the step is merged into the step.  The time
#: is a floating-point sum (`time + dt` per step): after n steps it is off by up to
#: ~n u |time| (u = 2^-53), i.e. n^2 u in units of a constant step dt = time / n, so
#: n dt lands short of n dt by a sliver of ~1e-13 (dt 1e-3, t 2; 1e-17 at dt 5e-3, t 0.05)
#: and a further step of that size is taken: a Chorin projection solves its pressure with
#: 1 / dt^2 in the operator and fails there (ANALYSIS 58).  1e-6 merges every such residue
#: up to n ~ 1e5 steps (worst case) and lengthens the merged step by at most 1e-6 of
#: itself: far below any CFL margin.
LANDING_TOL = 1e-6


def landed(dt, cap):
    """The step `dt` (`<= cap`, the distance to the next instant), or the whole `cap`
    when the leftover `cap - dt` is at most `LANDING_TOL dt`: the last step absorbs a
    floating-point sliver instead of a further step of that size.
    """
    return Cond(cap <= (1 + sp.Float(LANDING_TOL)) * dt, lambda: cap, lambda: dt)


def face_rule(nsm, mesh, cfl, dimension, dt_constant, time, cap, check_every=10):
    """`dt_rule` taking the step from the first stage's face wave speeds, capped by `cap`
    and `landed` on it (`residual` applies it to the cells' speeds too where an edge
    source moves them).
    """
    if dt_constant is None:
        return lambda S: landed(minimum(face_dt(nsm, S, mesh, cfl, dimension), cap), cap)
    return lambda S: landed(minimum(checked(dt_constant, time, check_every,
                                            lambda: face_dt(nsm, S, mesh, cfl,
                                                            dimension)), cap), cap)


def checked(dt, time, every, dt_cfl):
    """`dt`, warning once if it exceeds the CFL step `dt_cfl()`; tested every
    `every` steps (1: each step, 0: never).
    """
    def check_now():
        """Warn if `dt` exceeds the CFL step."""
        check.cfl(dt_cfl(), dt)
        return sp.S.Zero

    if every == 1:
        check_now()
    elif every > 1:
        step = sp.floor(time / dt + sp.S.Half)
        Cond(sp.Eq(sp.Mod(step, every), 0), check_now, lambda: sp.S.Zero)
    return dt
