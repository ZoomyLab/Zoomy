"""The C++ host program of a solver template: ``run`` (the printed march),
``march`` (its call from ``main``), ``initial_aux`` and ``configure`` (the
backend settings).  One text for every C++ runtime (foam ``rt.H``, AMReX
``rt_amrex.H``); a runtime gives each name its meaning.

    source = hyperbolic_program(nsm, dimension=2, cfl=0.9, mood=True)
    source = chorin_program(pred, press, corr, dimension=2, cfl=0.9)
"""
import sympy as sp

from zoomy_core.symsolver import chorin as CH
from zoomy_core.symsolver import hyperbolic as T
from zoomy_core.symsolver.core import CxxPrinter


def _march_kw(cfl, dimension, mood, dt_write, imex, cfl_check_every, wave_speed,
              dt_constant):
    return dict(cfl=cfl, dimension=dimension, mood=mood, dt_write=dt_write,
                imex=imex, cfl_check_every=cfl_check_every, wave_speed=wave_speed,
                dt_constant=None if dt_constant is None else sp.Float(dt_constant))


def _program(nsm, run, implicit, linsolve, lsq_degree):
    Q, Qaux, p, mesh = sp.symbols("Q Qaux p mesh")
    run += ("\n\ninline auto march = [](auto Q, auto Qaux, auto p, auto time_end, "
            "auto mesh) {\n    return run(Q, Qaux, p, 0.0, time_end, mesh, 0L);\n};")
    aux = CxxPrinter().procedure(
        "initial_aux", (Q, Qaux, p, mesh),
        lambda Q, Qaux, p, mesh: (T.full_aux(nsm, Q, Qaux, p, sp.S.Zero, sp.S.One,
                                             mesh),))
    return run + "\n\n" + aux + "\n\n" + configure(lsq_degree, implicit, linsolve) + "\n"


def configure(lsq_degree, implicit, linsolve):
    """``configure()``: LSQ degree, the implicit-solve tolerances."""
    return ("inline void configure()\n{\n"
            f"    settings().lsq_degree = {int(lsq_degree)};\n"
            f"    implicit().tol = {float(implicit['tol'])!r};\n"
            f"    implicit().maxiter = {int(implicit['maxiter'])};\n"
            f"    implicit().ls_tol = {float(linsolve['tol'])!r};\n"
            f"    implicit().ls_restart = {int(linsolve['restart'])};\n"
            f"    implicit().ls_maxiter = {int(linsolve['maxiter'])};\n"
            f"    implicit().ls_accept = {float(linsolve['accept'])!r};\n}}")


def hyperbolic_program(nsm, *, dimension, implicit, linsolve, cfl=0.9, mood=False,
                       dt_write=None, imex=False, cfl_check_every=10,
                       wave_speed="faces", dt_constant=None):
    """:func:`zoomy_core.symsolver.hyperbolic.hyperbolic` as a C++ program."""
    Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
    kw = _march_kw(cfl, dimension, mood, dt_write, imex, cfl_check_every,
                   wave_speed, dt_constant)
    run = CxxPrinter().procedure("run", (Q, Qaux, p, time, time_end, mesh),
                                 lambda *a: T.hyperbolic(nsm, *a, **kw))
    return _program(nsm, run, implicit, linsolve, nsm.resolved_lsq_degree())


def chorin_program(pred, press, corr, *, dimension, implicit, linsolve, cfl=0.9,
                   mood=False, dt_write=None, imex=False, cfl_check_every=10,
                   wave_speed="faces", dt_constant=None, time_order=1,
                   pressure_newton=None):
    """:func:`zoomy_core.symsolver.chorin.chorin` as a C++ program."""
    Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
    kw = _march_kw(cfl, dimension, mood, dt_write, imex, cfl_check_every,
                   wave_speed, dt_constant)
    run = CxxPrinter().procedure(
        "run", (Q, Qaux, p, time, time_end, mesh),
        lambda *a: CH.chorin(pred, press, corr, *a, time_order=time_order,
                             pressure_newton=pressure_newton, **kw))
    degree = max(s.resolved_lsq_degree() for s in (pred, press, corr))
    return _program(pred, run, implicit, linsolve, degree)
