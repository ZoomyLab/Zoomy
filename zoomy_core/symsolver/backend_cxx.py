"""The C++ backend of the symbolic solvers: what foam and AMReX share.

A :class:`CxxBackend` carries the solver settings and prints one solver
template (:mod:`zoomy_core.symsolver.strategies`) as the C++ host program
``Solver.H``: ``run`` (the printed march), ``march`` (its call from ``main``),
``initial_aux`` and ``configure`` (the backend settings).  One text for every
C++ runtime (foam ``rt.H``, AMReX ``rt_amrex.H``); a runtime gives each name
its meaning.  :class:`CxxChorinBackend` prints the projection template.

    source = SymbolicHyperbolicSolverFoam(cfl=0.9, mood=True)._print(nsm, 2)

A runtime driver supplies the rest: the kernels, the build tree, the case
(mesh, fields, run-time inputs), the launch and the read-back (``build``,
``run``), and the AMR level loop (``_amr_loop``).  Drivers combine with a
strategy by inheritance, driver first::

    class SymbolicHyperbolicSolverFoam(FoamDriver, CxxBackend): ...
    class SymbolicChorinSolverFoam(FoamDriver, CxxChorinBackend): ...
"""
import param
import sympy as sp

from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.solver.constants import ELLIPTIC_RTOL, TPFA_COS_FLOOR, elliptic_constants
from zoomy_core.symsolver.core import CxxPrinter
from zoomy_core.symsolver.discretization import reconstruction as R
from zoomy_core.symsolver.discretization.time import SSP_RK
from zoomy_core.symsolver.strategies.chorin import ChorinTemplate
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.symsolver.vocabulary import clean_round_off, pointwise, put, take


class CxxBackend(param.Parameterized):
    """The settings of a hyperbolic solver on a C++ runtime and the printing
    of its host program (:meth:`_print`).  Subclasses (the drivers) implement
    :meth:`build` and :meth:`run`."""
    time_end = param.Number(default=None, allow_None=True)
    cfl = param.Number(default=0.9, bounds=(0, None))
    time_order = param.Integer(default=None, allow_None=True, bounds=(1, 3),
                               doc="SSP-RK order; None: the reconstruction order.")
    dt = param.Number(default=None, allow_None=True,
                      doc="Constant time step; None = adaptive CFL step.")
    cfl_check_every = param.Integer(default=10, bounds=(0, None),
                                    doc="Constant dt: CFL test every N steps "
                                        "(1: each step, 0: never).")
    cfl_wave_speed = param.Selector(
        default="faces", objects=["faces", "cells"],
        doc="CFL wave speeds: 'faces' from the first RK stage's Riemann "
            "problem, 'cells' from the cell states of each face.")
    mood = param.Boolean(default=True,
                         doc="A-posteriori MOOD at order >= 2 (cells failing "
                             "the model's admissibility redo at order 1).")
    dt_write = param.Number(default=None, allow_None=True,
                            doc="Output interval; None = final state only.")
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(participant=, config=, mesh=, "
                            "write=[..], read=[..], tag=, side=); side 0: this "
                            "domain is the low side of the interface.")
    subcycling = param.Boolean(default=False,
                               doc="preCICE: adaptive steps inside a window.")
    imex = param.Boolean(default=False,
                         doc="IMEX (Lie): implicit source slot by Newton.")
    implicit_tol = param.Number(default=1e-8, bounds=(0, None))
    implicit_maxiter = param.Integer(default=6, bounds=(1, None))
    gmres_tol = param.Number(default=1e-7, bounds=(0, None))
    gmres_restart = param.Integer(default=20, bounds=(1, None))
    gmres_maxiter = param.Integer(default=30, bounds=(1, None))

    # ── the driver ─────────────────────────────────────────────────────────
    def build(self, nsm, dimension):
        """Print (``self.source``), write the build tree, compile (cached).
        Returns the binary."""
        raise NotImplementedError(f"{type(self).__name__}.build")

    def run(self, mesh, model, Q0, workdir):
        """March ``Q0`` (the mesh's order) to ``time_end``: write the case
        into ``workdir``, launch, read back.  Returns ``(time, Q)`` and
        keeps the snapshots in ``self.snapshots`` [(time, Q)].  ``model``:
        the NSM (see :meth:`_prepare`)."""
        raise NotImplementedError(f"{type(self).__name__}.run")

    def _amr_loop(self):
        """Whether the host program carries an AMR level loop's procedures."""
        return False

    # ── printing ───────────────────────────────────────────────────────────
    def _systems(self, nsm):
        """The systems whose kernels the build writes: ``[(name, NSM)]``."""
        return [("nsm", nsm)]

    def _prepare(self, mesh, model):
        """The NumericalSystemModel ``run`` marches (the template's NSM)."""
        return (model if isinstance(model, NumericalSystemModel)
                else NumericalSystemModel.from_system_model(model))

    def _mood(self, nsm):
        return bool(self.mood and nsm.reconstruction.order >= 2
                    and nsm.call.admissible is not None)

    def _linsolve(self, nsm):
        return dict(tol=self.gmres_tol, restart=self.gmres_restart,
                    maxiter=self.gmres_maxiter, accept=-1.0)

    def _degree(self, nsm):
        """The LSQ degree of the derivative stencils."""
        return nsm.resolved_lsq_degree()

    def _config(self, nsm, dimension):
        """The template's settings."""
        return dict(cfl=self.cfl, dimension=dimension, mood=self._mood(nsm),
                    dt_write=self.dt_write, imex=self.imex,
                    cfl_check_every=self.cfl_check_every,
                    wave_speed=self.cfl_wave_speed, dt_constant=self.dt,
                    time_order=self.time_order)

    def _template(self, nsm, dimension):
        return HyperbolicTemplate(nsm, **self._config(nsm, dimension))

    def _print(self, nsm, dimension):
        """The host program of ``nsm``: :meth:`HyperbolicTemplate.march` as
        ``run``; coupled, :meth:`HyperbolicTemplate.precice_march` (one
        preCICE participant; ``time_end`` unused, preCICE ends the loop);
        with an AMR level loop (:meth:`_amr_loop`) also the procedures it calls
        (:func:`_amr_procedures`)."""
        template = self._template(nsm, dimension)
        Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
        if self.couple:
            run = CxxPrinter().procedure(
                "run", (Q, Qaux, p, time, time_end, mesh),
                lambda Q, Qaux, p, time, time_end, mesh: template.precice_march(
                    Q, Qaux, p, time, mesh, side=self.couple["side"],
                    subcycling=self.subcycling))
        else:
            run = CxxPrinter().procedure("run", (Q, Qaux, p, time, time_end, mesh),
                                         template.march)
            if self._amr_loop():
                run += "\n\n" + _amr_procedures(template)
        run += ("\n\ninline auto march = [](auto Q, auto Qaux, auto p, auto time_end, "
                "auto mesh) {\n    return run(Q, Qaux, p, 0.0, time_end, mesh, 0L);\n};")
        aux = CxxPrinter().procedure("initial_aux", (Q, Qaux, p, mesh),
                                     template.initial_aux)
        return run + "\n\n" + aux + "\n\n" + self._configure(nsm) + "\n"

    def _configure(self, nsm):
        """``configure()``: LSQ degree, the two-point flux floor, the
        implicit-solve tolerances."""
        lin = self._linsolve(nsm)
        return ("inline void configure()\n{\n"
                f"    settings().lsq_degree = {int(self._degree(nsm))};\n"
                f"    settings().tpfa_cos_floor = {float(TPFA_COS_FLOOR)!r};\n"
                f"    implicit().tol = {float(self.implicit_tol)!r};\n"
                f"    implicit().maxiter = {int(self.implicit_maxiter)};\n"
                f"    implicit().ls_tol = {float(lin['tol'])!r};\n"
                f"    implicit().ls_restart = {int(lin['restart'])};\n"
                f"    implicit().ls_maxiter = {int(lin['maxiter'])};\n"
                f"    implicit().ls_accept = {float(lin['accept'])!r};\n}}")


class CxxChorinBackend(CxxBackend):
    """The Chorin projection template (:class:`ChorinTemplate`) on a C++
    runtime.  ``run`` takes ``(pred, press, corr)``: ``pred``, the predictor
    NSM on the full state; ``press`` / ``corr``, the pressure and corrector
    systems of the model's split (zoomy's ``dt``)."""
    time_order = param.Integer(default=1, bounds=(1, 3))
    pressure_tol = param.Number(default=ELLIPTIC_RTOL, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None))
    pressure_restart = param.Integer(default=None, allow_None=True, bounds=(1, None))
    pressure_solve = param.Selector(
        default="linear", objects=["linear", "newton"],
        doc="'linear': the constraint is affine in the pressure, one linear "
            "solve.  'newton': a nonlinear constraint, Newton iterations.")
    pressure_newton_tol = param.Number(default=1e-10, bounds=(0, None),
                                       doc="Newton: stop at max|R| <= tol.")
    pressure_newton_maxit = param.Integer(default=10, bounds=(1, None))

    def _prepare(self, mesh, model):
        """Name the split's systems as the template prints them (``pred.``,
        ``press.``, ``corr.``); returns the predictor."""
        pred, press, corr = model
        out = []
        for name, sm in (("pred", pred), ("press", press), ("corr", corr)):
            nsm = (sm if isinstance(sm, NumericalSystemModel)
                   else NumericalSystemModel.from_system_model(sm))
            nsm._symbolic_name = name
            out.append(nsm)
        pred, self.press, self.corr = out
        self._nc = int(mesh.n_inner_cells)
        return pred

    def _systems(self, nsm):
        return [("pred", nsm), ("press", self.press), ("corr", self.corr)]

    def _linsolve(self, nsm):
        n = len(self.press.equation_to_state_index) * self._nc
        ell = elliptic_constants(n_unknowns=n, tol=self.pressure_tol,
                                 restart=self.pressure_restart).values
        tol = float(ell["c_elliptic_tol"])
        return dict(tol=tol, restart=int(ell["c_elliptic_restart"]),
                    maxiter=self.pressure_maxit, accept=max(1e-6, 10.0 * tol))

    def _degree(self, nsm):
        return max(s.resolved_lsq_degree() for s in (nsm, self.press, self.corr))

    def _template(self, nsm, dimension):
        newton = (None if self.pressure_solve == "linear" else
                  (sp.Float(self.pressure_newton_tol), int(self.pressure_newton_maxit)))
        return ChorinTemplate(nsm, self.press, self.corr, pressure_newton=newton,
                              **self._config(nsm, dimension))

    def _print(self, nsm, dimension):
        if self._amr_loop():
            raise NotImplementedError("AMR: the hyperbolic template only")
        return super()._print(nsm, dimension)


def _conservative_rows(nsm):
    """Rows of the state without a nonconservative product."""
    B = nsm.sm.nonconservative_matrix
    n = len(nsm.state)
    return [i for i in range(n)
            if all(sp.sympify(B[i, j, d]) == 0 for j in range(n) for d in range(B.shape[2]))]


def _amr_procedures(template):
    """The template pieces a level loop calls, printed as they are:
    ``step`` = :meth:`HyperbolicTemplate.step` with a given ``dt``, ``dt_of`` =
    :meth:`HyperbolicTemplate.dt`, ``amr_w`` / ``amr_q`` = the reconstruction's
    variables and their inverse (:func:`reconstruction.reconstruct`'s maps:
    coarse data is interpolated in them), ``amr_clean`` = a stage's
    round-off hygiene (for the refluxed state), ``rk_weights`` = the weight of each
    stage's residual in the step (``SSP_RK`` in Butcher form: the time
    integral of the face terms), ``reflux_rows`` = the conservation laws (rows
    without a nonconservative product: only their face terms are refluxed;
    a row with one has no conservation to restore, and the coarse cell keeps
    its own, balanced, face terms)."""
    Q, Qaux, p, time, dt, mesh, W = sp.symbols("Q Qaux p time dt mesh W")
    nsm, order = template.nsm, template.order
    to_w, to_q = R.maps(nsm)
    rows = R.bed_rows(nsm) if to_w is not None else None

    def w_of(Q, Qaux, p, mesh):
        W = Q if to_w is None else pointwise(to_w, Q, Qaux, p)
        if rows is not None:                    # (h, eta): h in the bed's slot
            W = put(W, (rows[0],), take(Q, (rows[1],)))
        return (W,)

    def q_of(W, Qaux, p, mesh):
        if rows is not None:                    # b = eta - h
            W = put(W, (rows[0],), take(W, (rows[1],)) - take(W, (rows[0],)))
        return (W if to_q is None else pointwise(to_q, W, Qaux, p),)

    depth, tol = getattr(nsm.roles, "depth", None), getattr(nsm, "depth_round_off", None)

    def clean(Q, mesh):                         # a stage's hygiene (explicit_step)
        return (Q if depth is None or tol is None else clean_round_off(Q, (depth,), tol),)

    beta = []
    for a, _c in SSP_RK[template.time_order]:
        beta = [(1 - a) * b for b in beta] + [1 - a]
    procs = [
        "#define ZS_AMR_PROCEDURES 1",
        CxxPrinter().procedure("step", (Q, Qaux, p, time, dt, mesh),
                               template.step),
        CxxPrinter().procedure("dt_of", (Q, Qaux, p, mesh), template.dt),
        CxxPrinter().procedure("amr_w", (Q, Qaux, p, mesh), w_of),
        CxxPrinter().procedure("amr_q", (W, Qaux, p, mesh), q_of),
        CxxPrinter().procedure("amr_clean", (Q, mesh), clean),
        "inline std::vector<double> rk_weights() { return {"
        + ", ".join(repr(float(b)) for b in beta) + "}; }",
        "inline std::vector<int> reflux_rows() { return {"
        + ", ".join(str(i) for i in _conservative_rows(nsm)) + "}; }"]
    return "\n\n".join(procs)
