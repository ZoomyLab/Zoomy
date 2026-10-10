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
    boundary_states, boundary_traces, cells, check, clean_round_off, couple,
    coupled_faces,
    derivative_aux, diffusion, divergence, edge_source, face_aux, face_inradius,
    face_wavespeed, faces, halo, implicit_diffusion, inadmissible, interface_flux,
    interface_states, io, minimum, reduce_any, reduce_min, select)
from zoomy_core.symsolver.reconstruction import reconstruct

# Shu-Osher form: Q <- a*Q0 + (1 - a)*(Q + dt*L(Q, t + c*dt)), per stage (a, c)
SSP_RK = {1: ((0, 0),), 2: ((0, 0), (sp.S.Half, 1))}


def residual(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
             coupled=None, implicit_terms=True, dt_rule=None):
    """Semi-discrete right-hand side: -div(face terms) - interior NCP, plus the
    explicit source/diffusion slots, plus the implicit ones unless an implicit
    stage takes them (``implicit_terms=False``).  Which term is stiff is the
    SystemModel's classification, never the solver's.

    ``dt_rule``: ``dt`` is not known yet -- the face kernel also returns the
    scheme's wave speed ``S`` and ``dt = dt_rule(S)``; returns ``(dQ, dt)``."""
    Qb, Qauxb = unpack(boundary_states(nsm.call.boundary_conditions, Q, Qaux, p,
                                       time, mesh), 2)
    QL, QR = reconstruct(nsm, Q, Qaux, Qb, Qauxb, p, order, demote, mesh)
    if order >= 2:              # at first order QL is the cell value: QR is Qb
        QR = boundary_traces(nsm.call.boundary_conditions, Q, QL, QR, Qaux, p,
                             time, mesh)
    AL, AR = unpack(face_aux(Qaux, mesh), 2)
    # one Riemann problem per face, its contribution to each side
    if dt_rule is None:
        Wm, Wp = unpack(faces(nsm.call.numerical_face_sides, QL, QR, AL, AR, p,
                              mesh), 2)
    else:                            # ... and its wave speed sets the step
        Wm, Wp, S = unpack(faces(nsm.call.numerical_face_sides_speed, QL, QR,
                                 AL, AR, p, mesh), 3)
        dt = dt_rule(S)
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
    return dQ if dt_rule is None else (dQ, dt)


def stage_aux(nsm, Q, Qaux, p, time, dt, mesh):
    """Pointwise aux of a stage state (e.g. hinv = 1/h)."""
    if nsm.call.update_aux_variables is None:
        return Qaux
    return cells(nsm.call.update_aux_variables, Q, Qaux, p, time, dt, mesh)


def explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
                  coupled=None, implicit_terms=True, rk=None, dt_rule=None):
    """One SSP-RK step of the method of lines (RK order ``rk``, default the
    reconstruction order).  With ``dt_rule`` the first stage picks ``dt``
    from its own face wave speeds; returns ``(Q, dt)``."""
    Q0, Qaux_stage = Q, Qaux
    depth, tol = getattr(nsm.roles, "depth", None), getattr(nsm, "depth_round_off", None)
    for i, (a, c) in enumerate(SSP_RK[rk or order]):
        if i > 0:
            Qaux_stage = stage_aux(nsm, Q, Qaux, p, time + c * dt, dt, mesh)
        if i == 0 and dt_rule is not None:
            L, dt = residual(nsm, Q, Qaux_stage, p, time, None, mesh, order,
                             demote, coupled, implicit_terms, dt_rule)
        else:
            L = residual(nsm, Q, Qaux_stage, p, time + c * dt, dt, mesh, order,
                         demote, coupled, implicit_terms)
        Q = a * Q0 + (1 - a) * (Q + dt * L)
        if depth is not None and tol is not None:   # round-off below 0 is 0
            Q = clean_round_off(Q, (depth,), tol)
        Q = halo(Q, mesh)
    return Q if dt_rule is None else (Q, dt)



def mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled=None,
              implicit_terms=True, rk=None, dt_rule=None):
    """A-posteriori MOOD: order-2 candidate; inadmissible cells redo at first
    order; if that still fails, the whole step redoes at first order.  The
    repairs keep the candidate's ``dt``."""
    def step(demote, dt):
        return explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote,
                             coupled, implicit_terms, rk)

    def repair():
        Q_cells = step(bad, dt)
        still = inadmissible(nsm.call.admissible, Q_cells, Qaux, p)
        return Cond(reduce_any(still), lambda: step(sp.S.true, dt),
                    lambda: Q_cells)

    if dt_rule is None:
        Q_cand = step(sp.S.false, dt)
    else:
        Q_cand, dt = explicit_step(nsm, Q, Qaux, p, time, None, mesh, order,
                                   sp.S.false, coupled, implicit_terms, rk,
                                   dt_rule)
    bad = inadmissible(nsm.call.admissible, Q_cand, Qaux, p)
    Q = Cond(reduce_any(bad), repair, lambda: Q_cand)
    return Q if dt_rule is None else (Q, dt)


def full_aux(nsm, Q, Qaux, p, time, dt, mesh):
    """Pointwise aux, then the derivative (mesh-gradient) aux."""
    Qaux = stage_aux(nsm, Q, Qaux, p, time, dt, mesh)
    if nsm.call.aux_derivatives is not None:
        Qaux = halo(derivative_aux(nsm.call.aux_derivatives, Q, Qaux, mesh), mesh)
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


def advance(nsm, Q, Qaux, p, time, dt, mesh, order, mood, coupled=None,
            imex=False, dt_rule=None):
    """One time step.  Explicit: every term in the (MOOD-guarded) RK step.
    IMEX (Lie): the RK step without the SystemModel's implicit slots, then
    implicit diffusion and implicit Euler on the implicit source slot."""
    if mood:
        Q = mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled,
                      implicit_terms=not imex, dt_rule=dt_rule)
    else:
        Q = explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order,
                          coupled=coupled, implicit_terms=not imex,
                          dt_rule=dt_rule)
    if dt_rule is not None:
        Q, dt = Q
    if imex:
        Q = implicit_step(nsm, Q, Qaux, p, time + dt, dt, mesh)
    return (*post_step(nsm, Q, Qaux, p, time + dt, dt, mesh), dt)


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
               cfl_check_every=10, wave_speed="faces"):
    """March from ``time`` to ``time_end``.  ``wave_speed``: the CFL wave
    speeds from the first stage's Riemann problem ("faces": the speeds of the
    interface states, Audusse et al. 2004 (2.34); no eigensolve of its own),
    or from the cell states of each face ("cells")."""
    order = nsm.reconstruction.order

    def step(Q, Qaux, time, cap):
        if wave_speed == "faces":
            return advance(nsm, Q, Qaux, p, time, None, mesh, order, mood,
                           imex=imex, dt_rule=face_rule(
                               nsm, mesh, cfl, dimension, dt_constant, time,
                               cap, cfl_check_every))
        dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                              dt_constant, time, cfl_check_every), cap)
        return advance(nsm, Q, Qaux, p, time, dt, mesh, order, mood, imex=imex)

    return march(Q, Qaux, time, time_end, dt_write, step)


def march(Q, Qaux, time, time_end, dt_write, step):
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


def hyperbolic_precice(nsm, Q, Qaux, p, time, mesh, *, cfl, dimension, side,
                       subcycling=False, mood=False, dt_constant=None,
                       dt_write=None, imex=False, cfl_check_every=10,
                       wave_speed="faces"):
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
        window = couple.max_dt()
        if subcycling:                              # adaptive inside the window
            cap = window if dt_write is None else sp.Min(
                window, n_write * dt_write - time)
            if wave_speed == "faces":
                rule = face_rule(nsm, mesh, cfl, dimension, dt_constant, time,
                                 cap, cfl_check_every)
            else:
                dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                                      dt_constant, time, cfl_check_every), cap)
        elif wave_speed == "faces":                 # one step per window
            rule = lambda S: checked(window, time, cfl_check_every,
                                     lambda: face_dt(nsm, S, mesh, cfl,
                                                     dimension))
        else:
            dt = checked(window, time, cfl_check_every,
                         lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl, dimension))
        peer = couple.read()
        Wc = interface_flux(nsm.call.numerical_face_sides,
                            nsm.call.update_aux_variables or sp.S.false,
                            side, Q, Qaux, peer, p, mesh)
        if wave_speed == "faces":
            Q, Qaux, dt = advance(nsm, Q, Qaux, p, time, None, mesh, order,
                                  mood, coupled=(coupled_faces(mesh), Wc),
                                  imex=imex, dt_rule=rule)
        else:
            Q, Qaux, dt = advance(nsm, Q, Qaux, p, time, dt, mesh, order, mood,
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
