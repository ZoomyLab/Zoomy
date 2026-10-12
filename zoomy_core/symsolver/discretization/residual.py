"""The semi-discrete residual and the pieces of one time step, written once for
every backend.

``nsm`` is the NumericalSystemModel: ``nsm.call.<slot>`` is a kernel (the
registrar ``call`` entry; ``None`` if the model has none), every other
attribute is static configuration such as ``nsm.reconstruction.order``.  ``Q``,
``Qaux``, ``p`` (parameters), ``time``, ``dt`` and ``mesh`` are symbols.  Python
``if`` / ``for`` run at build time; ``While`` / ``Cond`` run at run time.

Ground truth: ``zoomy_jax.fvm.solver_jax.HyperbolicSolver`` (``flux_operator``,
``_explicit_hyperbolic_step``, ``post_step``).
"""
import sympy as sp

from zoomy_core.symsolver.core import Cond, Root, unpack
from zoomy_core.symsolver.discretization.reconstruction import reconstruct
from zoomy_core.symsolver.discretization.time import SSP_RK
from zoomy_core.symsolver.vocabulary import (
    boundary_states, boundary_traces, cells, clean_round_off, derivative_aux,
    diffusion, divergence, edge_source, face_aux, faces, halo,
    implicit_diffusion, inadmissible, reduce_any, select)


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


def implicit_step(nsm, Q, Qaux, p, time, dt, mesh):
    """The SystemModel's implicit slots: implicit diffusion, then implicit
    Euler on the source."""
    if nsm.call.diffusion_matrix is not None:
        Q = implicit_diffusion(nsm.call.diffusion_matrix, Q, Qaux, p, dt, mesh)
    if nsm.call.source is not None:
        Q = implicit_source(nsm, Q, Qaux, p, time, dt, mesh)
    return Q
