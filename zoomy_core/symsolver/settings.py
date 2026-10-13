"""The settings every symbolic solver shares (section `solver` of a settings file), declared once.

The backends (NumPy, JAX, foam, AMReX) inherit these classes and add their own settings; a
backend that supports only some members of an option narrows it with `OptionParam(members=...)`.
"""
import param

from zoomy_core.misc.options import Configurable, Option, OptionParam
from zoomy_core.solver.constants import ELLIPTIC_RTOL
from zoomy_core.symsolver.discretization.time import Imex
from zoomy_core.symsolver.strategies.chorin import Projection


class CflWaveSpeed(Option):
    """Where the CFL step takes its wave speeds."""
    #: from the first RK stage's Riemann problem
    faces = "faces"
    #: from the cell states on each face
    cells = "cells"


class PressureSolve(Option):
    """How a projection stage solves its pressure constraint."""
    #: the constraint is affine in the pressure: one linear solve
    linear = "linear"
    #: a nonlinear constraint: Newton iterations
    newton = "newton"


class LinearSolver(Option):
    """The linear solver of the pressure (and implicit) systems; each backend supports a subset."""
    #: the backend's best available choice
    auto = "auto"
    #: matrix-free restarted (flexible) GMRES
    gmres = "gmres"
    #: GMRES preconditioned by smoothed-aggregation AMG (PyAMG, CPU)
    pyamg = "pyamg"
    #: AmgX (GPU)
    amgx = "amgx"
    #: hypre FlexGMRES + BoomerAMG on the colour-assembled operator
    hypre = "hypre"
    #: FGMRES with the Schur-complement GAMG preconditioner (foam)
    schur_gamg = "schur-gamg"


class HyperbolicSettings(Configurable):
    """Settings of every symbolic hyperbolic solver: time window, step control, MOOD, IMEX."""
    section = "solver"

    time_end = param.Number(default=None, allow_None=True, doc="End time.")
    cfl = param.Number(default=0.9, bounds=(0, None),
                       doc="CFL number of the adaptive step (and of the check).")
    dt = param.Number(default=None, allow_None=True,
                      doc="Constant time step; None = adaptive CFL step.")
    cfl_check_every = param.Integer(default=10, bounds=(0, None),
                                    doc="Constant dt: test it against the CFL step every this "
                                        "many steps (1: each step, 0: never).")
    cfl_wave_speed = OptionParam(CflWaveSpeed, default=CflWaveSpeed.faces,
                                 doc="Where the CFL step takes its wave speeds.")
    mood = param.Boolean(default=True,
                         doc="A-posteriori MOOD at order >= 2 (cells failing the model's "
                             "admissibility redo at order 1).")
    time_order = param.Integer(default=None, allow_None=True, bounds=(1, 3),
                               doc="Order of the SSP-RK step; None = the reconstruction order.")
    dt_write = param.Number(default=None, allow_None=True,
                            doc="Output interval; None = no intermediate snapshots.")
    imex = OptionParam(Imex, default=False, extra=[False],
                       doc="IMEX: lie = explicit flux step, then implicit diffusion and an "
                           "implicit-Euler source; ssp2 = IMEX-SSP2(2,2,2); False = explicit.")
    implicit_tol = param.Number(default=1e-8, bounds=(0, None),
                                doc="Newton tolerance ||R||_2 (implicit solves).")
    implicit_maxiter = param.Integer(default=6, bounds=(1, None))
    gmres_tol = param.Number(default=1e-7, bounds=(0, None))
    gmres_maxiter = param.Integer(default=30, bounds=(1, None))


class ChorinSettings(HyperbolicSettings):
    """Settings of every symbolic Chorin projection solver: stages and the pressure solve."""
    time_order = param.Integer(default=1, bounds=(1, 3),
                               doc="Outer SSP-RK stages, each a full projection stage.")
    pressure_tol = param.Number(default=ELLIPTIC_RTOL, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None), doc="GMRES restarts.")
    pressure_restart = param.Integer(default=None, allow_None=True, bounds=(1, None),
                                     doc="None: from the block size.")
    pressure_solve = OptionParam(PressureSolve, default=PressureSolve.linear,
                                 doc="Linear solve, or Newton for a nonlinear constraint.")
    pressure_newton_tol = param.Number(default=1e-10, bounds=(0, None),
                                       doc="Newton: stop at max|R| <= tol.")
    pressure_newton_maxit = param.Integer(default=10, bounds=(1, None))
    projection = OptionParam(Projection, default=Projection.standard,
                             doc="standard: each stage solves for the full pressure; increment: "
                                 "for its increment over the lagged one (time order 2 on a "
                                 "fixed mesh; opt-in, ANALYSIS 45e).")
