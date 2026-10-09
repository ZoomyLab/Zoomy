"""The explicit hyperbolic solver, written once for every backend.

``nsm`` is the NumericalSystemModel: ``nsm.call.<slot>`` is a kernel (the
registrar ``call`` entry; ``None`` if the model has none), every other
attribute is static configuration such as ``nsm.reconstruction.order``.  ``Q``, ``Qaux``, ``p``
(parameters), ``time``, ``dt`` and ``mesh`` are symbols.  Python ``if`` / ``for``
run at build time; ``While`` / ``Cond`` run at run time.

Ground truth: ``zoomy_jax.fvm.solver_jax.HyperbolicSolver`` (``flux_operator``,
``_explicit_hyperbolic_step``, ``post_step``, ``_advance``, ``_run``).
"""
import sympy as sp

from zoomy_core.symsolver.core import Cond, Root, While, unpack
from zoomy_core.symsolver.vocabulary import (
    boundary_states, boundary_traces, cells, check, couple, coupled_faces,
    derivative_aux, diffusion, divergence, edge_source, face_aux, face_inradius,
    face_wavespeed, faces, implicit_diffusion, inadmissible, interface_flux,
    interface_states, io,
    reconstruct, reduce_any, reduce_min, select)

# Shu-Osher form: Q <- a*Q0 + (1 - a)*(Q + dt*L(Q, t + c*dt)), per stage (a, c)
SSP_RK = {1: ((0, 0),), 2: ((0, 0), (sp.S.Half, 1))}


def residual(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
             coupled=None, implicit_terms=True):
    """Semi-discrete right-hand side: -div(face terms) - interior NCP, plus the
    explicit source/diffusion slots, plus the implicit ones unless an implicit
    stage takes them (``implicit_terms=False``).  Which term is stiff is the
    SystemModel's classification, never the solver's."""
    Qb, Qauxb = unpack(boundary_states(nsm.call.boundary_conditions, Q, Qaux, p,
                                       time, mesh), 2)
    QL, QR = unpack(reconstruct(Q, Qaux, Qb, Qauxb, p, order, demote, mesh), 2)
    if order >= 2:              # at first order QL is the cell value: QR is Qb
        QR = boundary_traces(nsm.call.boundary_conditions, Q, QL, QR, Qaux, p,
                             time, mesh)
    AL, AR = unpack(face_aux(Qaux, mesh), 2)
    # one Riemann problem per face, its contribution to each side
    Wm, Wp = unpack(faces(nsm.call.numerical_face_sides, QL, QR, AL, AR, p,
                          mesh), 2)
    if coupled is not None:          # coupled faces carry the shared answer
        mask, Wc = coupled
        Wm = select(mask, Wc, Wm)
    dQ = -divergence(Wm, Wp, mesh)
    if order >= 2 and nsm.call.numerical_cell_edge_source is not None:
        dQ = dQ - edge_source(nsm.call.numerical_cell_edge_source, Q, QL, QR, Qaux,
                              p, mesh)
    sources = [nsm.call.source_explicit]
    diffusions = [nsm.call.diffusion_matrix_explicit]
    if implicit_terms:
        sources.append(nsm.call.source)
        diffusions.append(nsm.call.diffusion_matrix)
    for k in sources:
        if k is not None:
            dQ = dQ + cells(k, Q, Qaux, p, time, dt, mesh)
    for k in diffusions:
        if k is not None:
            dQ = dQ + diffusion(k, Q, Qaux, p, mesh)
    return dQ


def stage_aux(nsm, Q, Qaux, p, time, dt, mesh):
    """Pointwise aux of a stage state (e.g. hinv = 1/h)."""
    if nsm.call.update_aux_variables is None:
        return Qaux
    return cells(nsm.call.update_aux_variables, Q, Qaux, p, time, dt, mesh)


def explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
                  coupled=None, implicit_terms=True, rk=None):
    """One SSP-RK step of the method of lines (RK order ``rk``, default the
    reconstruction order)."""
    Q0, Qaux_stage = Q, Qaux
    for i, (a, c) in enumerate(SSP_RK[rk or order]):
        if i > 0:
            Qaux_stage = stage_aux(nsm, Q, Qaux, p, time + c * dt, dt, mesh)
        L = residual(nsm, Q, Qaux_stage, p, time + c * dt, dt, mesh, order,
                     demote, coupled, implicit_terms)
        Q = a * Q0 + (1 - a) * (Q + dt * L)
    return Q


def mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled=None,
              implicit_terms=True, rk=None):
    """A-posteriori MOOD: order-2 candidate; inadmissible cells redo at first
    order; if that still fails, the whole step redoes at first order."""
    def step(demote):
        return explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote,
                             coupled, implicit_terms, rk)

    def repair():
        Q_cells = step(bad)
        still = inadmissible(nsm.call.admissible, Q_cells, Qaux, p)
        return Cond(reduce_any(still), lambda: step(sp.S.true), lambda: Q_cells)

    Q_cand = step(sp.S.false)
    bad = inadmissible(nsm.call.admissible, Q_cand, Qaux, p)
    return Cond(reduce_any(bad), repair, lambda: Q_cand)


def full_aux(nsm, Q, Qaux, p, time, dt, mesh):
    """Pointwise aux, then the derivative (mesh-gradient) aux."""
    Qaux = stage_aux(nsm, Q, Qaux, p, time, dt, mesh)
    if nsm.call.aux_derivatives is not None:
        Qaux = derivative_aux(nsm.call.aux_derivatives, Q, Qaux, mesh)
    return Qaux


def post_step(nsm, Q, Qaux, p, time, dt, mesh):
    """Model hygiene on the new state, then its aux."""
    if nsm.call.update_variables is not None:
        Q = cells(nsm.call.update_variables, Q, Qaux, p, time, dt, mesh)
    return Q, full_aux(nsm, Q, Qaux, p, time, dt, mesh)


def implicit_source(nsm, Q, Qaux, p, time, dt, mesh):
    """Implicit Euler on the source at ``time``:
    Y - Q - dt S(Y, aux(Y)) = 0.  The pointwise aux follows the iterate, the
    derivative aux is frozen at the explicit predictor ``Q``."""
    Qaux_e = full_aux(nsm, Q, Qaux, p, time, dt, mesh)

    def aux_of(Y):
        return stage_aux(nsm, Y, Qaux_e, p, time, dt, mesh)

    return Root(lambda Y: Y - Q - dt * cells(nsm.call.source, Y, aux_of(Y),
                                             p, time, dt, mesh), Q)


def cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension):
    """dt <= CFL * 2 r_f / (d |lambda|_f), min over faces, capped by dt_max."""
    speed = (nsm.call.eigenvalues_cfl if nsm.call.eigenvalues_cfl is not None
             else nsm.call.eigenvalues)
    lam = face_wavespeed(speed, Q, Qaux, p, mesh)
    dt = reduce_min(cfl * 2 * face_inradius(mesh) / (dimension * lam))
    return dt if nsm.dt_max is None else sp.Min(dt, nsm.dt_max)


def step_size(nsm, Q, Qaux, p, mesh, cfl, dimension, dt_constant, time,
              check_every=10):
    """Adaptive CFL step, or a constant step that warns when above it."""
    if dt_constant is None:
        return cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension)
    return checked(nsm, Q, Qaux, p, mesh, cfl, dimension, dt_constant, time,
                   check_every)


def checked(nsm, Q, Qaux, p, mesh, cfl, dimension, dt, time, every):
    """``dt``, warning (once) if it exceeds the CFL step; tested every
    ``every`` steps of size ``dt`` (1: each step, 0: never)."""
    def check_now():
        check.cfl(cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension), dt)
        return sp.S.Zero

    if every == 1:
        check_now()
    elif every > 1:
        step = sp.floor(time / dt + sp.S.Half)
        Cond(sp.Eq(sp.Mod(step, every), 0), check_now, lambda: sp.S.Zero)
    return dt


def advance(nsm, Q, Qaux, p, time, dt, mesh, order, mood, coupled=None,
            imex=False):
    """One time step.  Explicit: every term in the (MOOD-guarded) RK step.
    IMEX (Lie): the RK step without the SystemModel's implicit slots, then
    implicit diffusion and implicit Euler on the implicit source slot."""
    if mood:
        Q = mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled,
                      implicit_terms=not imex)
    else:
        Q = explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order,
                          coupled=coupled, implicit_terms=not imex)
    if imex:
        Q = implicit_step(nsm, Q, Qaux, p, time + dt, dt, mesh)
    return post_step(nsm, Q, Qaux, p, time + dt, dt, mesh)


def implicit_step(nsm, Q, Qaux, p, time, dt, mesh):
    """The SystemModel's implicit slots: implicit diffusion, then implicit
    Euler on the source."""
    if nsm.call.diffusion_matrix is not None:
        Q = implicit_diffusion(nsm.call.diffusion_matrix, Q, Qaux, p, dt, mesh)
    if nsm.call.source is not None:
        Q = implicit_source(nsm, Q, Qaux, p, time, dt, mesh)
    return Q


def hyperbolic(nsm, Q, Qaux, p, time, time_end, mesh, *, cfl, dimension,
               mood=False, dt_constant=None, dt_write=None, imex=False,
               cfl_check_every=10):
    """March from ``time`` to ``time_end``."""
    order = nsm.reconstruction.order
    return march(
        Q, Qaux, time, time_end, dt_write,
        lambda Q, Qaux, time: step_size(nsm, Q, Qaux, p, mesh, cfl,
                                        dimension, dt_constant, time,
                                        cfl_check_every),
        lambda Q, Qaux, time, dt: advance(nsm, Q, Qaux, p, time, dt, mesh,
                                          order, mood, imex=imex))


def march(Q, Qaux, time, time_end, dt_write, step_size, step):
    """The time loop: ``step_size(Q, Qaux, time)`` clamped to ``time_end`` and the
    write instants, then ``step(Q, Qaux, time, dt) -> (Q, Qaux)``."""
    def body(time, Q, Qaux, n_write):
        dt = sp.Min(step_size(Q, Qaux, time), time_end - time)
        if dt_write is not None:                    # land on write instants
            dt = sp.Min(dt, n_write * dt_write - time)
        Q, Qaux = step(Q, Qaux, time, dt)
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


def hyperbolic_precice(nsm, Q, Qaux, p, time, mesh, *, cfl, dimension, side,
                       subcycling=False, mood=False, dt_constant=None,
                       dt_write=None, imex=False, cfl_check_every=10):
    """The same march as one preCICE participant: preCICE ends the loop, the
    coupled faces carry the shared interface answer on (own, peer).

    subcycling=False: one step per window (dt = window; warns above the CFL step).
        Both participants pose the same Riemann problem with the same dt, so
        mass is conserved exactly.
    subcycling=True: adaptive steps inside the window, peer state held at its
        window value.  Stable, but the two sides charge different fluxes:
        mass is NOT exactly conserved (that would need refluxing)."""
    order = nsm.reconstruction.order

    def body(time, Q, Qaux, going, n_write):
        if subcycling:
            dt = step_size(nsm, Q, Qaux, p, mesh, cfl, dimension, dt_constant,
                           time, cfl_check_every)
            dt = sp.Min(dt, couple.max_dt())
            if dt_write is not None:                # land on write instants
                dt = sp.Min(dt, n_write * dt_write - time)
        else:                                       # one step per window
            dt = checked(nsm, Q, Qaux, p, mesh, cfl, dimension,
                         couple.max_dt(), time, cfl_check_every)
        peer = couple.read()
        Wc = interface_flux(nsm.call.numerical_face_sides,
                            nsm.call.update_aux_variables or sp.S.false,
                            side, Q, Qaux, peer, p, mesh)
        Q, Qaux = advance(nsm, Q, Qaux, p, time, dt, mesh, order, mood,
                          coupled=(coupled_faces(mesh), Wc), imex=imex)
        couple.write(interface_states(Q, mesh))
        going = couple.advance(dt)
        time = time + dt
        if dt_write is not None:
            n_write = Cond(time >= (n_write - sp.Rational(1, 10**9)) * dt_write,
                           lambda: _write(time, Q, Qaux, n_write),
                           lambda: n_write)
        return time, Q, Qaux, going, n_write

    going = couple.init(interface_states(Q, mesh))
    if dt_write is not None:
        io.write(time, Q, Qaux)
    time, Q, Qaux, *_ = While(lambda time, Q, Qaux, going, n_write: going,
                              body, (time, Q, Qaux, going, 1))
    return time, Q, Qaux


def _write(time, Q, Qaux, n_write):
    io.write(time, Q, Qaux)
    return n_write + 1
