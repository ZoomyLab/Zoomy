"""The C++ host program of a solver template: ``run`` (the printed march),
``march`` (its call from ``main``), ``initial_aux`` and ``configure`` (the
backend settings).  One text for every C++ runtime (foam ``rt.H``, AMReX
``rt_amrex.H``); a runtime gives each name its meaning.

    source = hyperbolic_program(nsm, dimension=2, cfl=0.9, mood=True)
    source = chorin_program(pred, press, corr, dimension=2, cfl=0.9)
"""
import sympy as sp

from zoomy_core.symsolver.core import CxxPrinter
from zoomy_core.symsolver.discretization import reconstruction as R
from zoomy_core.symsolver.discretization.time import SSP_RK
from zoomy_core.symsolver.strategies.chorin import ChorinTemplate
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.symsolver.vocabulary import clean_round_off, pointwise, put, take


def chorin_systems(pred, press, corr):
    """The Chorin split as NSMs named the way the templates print them
    (``pred.``, ``press.``, ``corr.``)."""
    from zoomy_core.numerics import NumericalSystemModel
    out = []
    for name, sm in (("pred", pred), ("press", press), ("corr", corr)):
        nsm = (sm if isinstance(sm, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(sm))
        nsm._symbolic_name = name
        out.append(nsm)
    return out


def _program(template, run, implicit, linsolve, lsq_degree):
    Q, Qaux, p, mesh = sp.symbols("Q Qaux p mesh")
    run += ("\n\ninline auto march = [](auto Q, auto Qaux, auto p, auto time_end, "
            "auto mesh) {\n    return run(Q, Qaux, p, 0.0, time_end, mesh, 0L);\n};")
    aux = CxxPrinter().procedure("initial_aux", (Q, Qaux, p, mesh),
                                 template.initial_aux)
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
                       wave_speed="faces", dt_constant=None, amr=False):
    """:meth:`HyperbolicTemplate.march` as a C++ program; ``amr``: also the
    procedures an AMR level loop calls (:func:`amr_procedures`)."""
    Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
    template = HyperbolicTemplate(
        nsm, cfl=cfl, dimension=dimension, mood=mood, dt_write=dt_write,
        imex=imex, cfl_check_every=cfl_check_every, wave_speed=wave_speed,
        dt_constant=dt_constant)
    run = CxxPrinter().procedure("run", (Q, Qaux, p, time, time_end, mesh),
                                 template.march)
    if amr:
        run += "\n\n" + amr_procedures(template)
    return _program(template, run, implicit, linsolve, nsm.resolved_lsq_degree())


def hyperbolic_precice_program(nsm, *, side, subcycling=False, dimension, implicit,
                               linsolve, cfl=0.9, mood=False, dt_write=None, imex=False,
                               cfl_check_every=10, wave_speed="faces", dt_constant=None):
    """:meth:`HyperbolicTemplate.precice_march` (one preCICE participant) as a
    C++ program; ``run`` ignores ``time_end`` (preCICE ends the loop)."""
    Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
    template = HyperbolicTemplate(
        nsm, cfl=cfl, dimension=dimension, mood=mood, dt_write=dt_write,
        imex=imex, cfl_check_every=cfl_check_every, wave_speed=wave_speed,
        dt_constant=dt_constant)
    run = CxxPrinter().procedure(
        "run", (Q, Qaux, p, time, time_end, mesh),
        lambda Q, Qaux, p, time, time_end, mesh: template.precice_march(
            Q, Qaux, p, time, mesh, side=side, subcycling=subcycling))
    return _program(template, run, implicit, linsolve, nsm.resolved_lsq_degree())


def conservative_rows(nsm):
    """Rows of the state without a nonconservative product."""
    B = nsm.sm.nonconservative_matrix
    n = len(nsm.state)
    return [i for i in range(n)
            if all(sp.sympify(B[i, j, d]) == 0 for j in range(n) for d in range(B.shape[2]))]


def amr_procedures(template):
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
    for a, _c in SSP_RK[order]:
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
        + ", ".join(str(i) for i in conservative_rows(nsm)) + "}; }"]
    return "\n\n".join(procs)


def chorin_program(pred, press, corr, *, dimension, implicit, linsolve, cfl=0.9,
                   mood=False, dt_write=None, imex=False, cfl_check_every=10,
                   wave_speed="faces", dt_constant=None, time_order=1,
                   pressure_newton=None):
    """:meth:`ChorinTemplate.march` as a C++ program."""
    Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
    template = ChorinTemplate(
        pred, press, corr, time_order=time_order,
        pressure_newton=pressure_newton, cfl=cfl, dimension=dimension,
        mood=mood, dt_write=dt_write, imex=imex,
        cfl_check_every=cfl_check_every, wave_speed=wave_speed,
        dt_constant=dt_constant)
    run = CxxPrinter().procedure("run", (Q, Qaux, p, time, time_end, mesh),
                                 template.march)
    degree = max(s.resolved_lsq_degree() for s in (pred, press, corr))
    return _program(template, run, implicit, linsolve, degree)
