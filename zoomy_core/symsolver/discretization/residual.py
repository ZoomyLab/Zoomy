"""The semi-discrete residual and the pieces of one time step, written once for every
backend: RK stage, MOOD, aux, implicit slots and hygiene. They compose the vocabulary
and are composed by the strategies.
"""
import sympy as sp

from zoomy_core.symsolver.core import Cond, ImplicitSolve, Root, unpack
from zoomy_core.symsolver.discretization.reconstruction import reconstruct
from zoomy_core.symsolver.discretization.time import SSP_RK, cell_wavespeed
from zoomy_core.symsolver.vocabulary import (
    boundary_ghosts, boundary_traces, cells, clean_round_off, coupled_ghosts, derivative_aux,
    diffusion, diffusion_correction, diffusion_tpfa, divergence, edge_source, face_aux, faces,
    halo, implicit_diffusion, inadmissible, minimum, reduce_any, select)


def residual(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
             coupled=None, implicit_terms=True, dt_rule=None, split_diffusion=False):
    """Semi-discrete right-hand side -div(face terms) - interior NCP, plus the explicit
    source and diffusion slots, and the implicit ones unless an implicit stage takes
    them.

    Args:
        demote: MOOD demotion of the reconstruction (see `reconstruct`).
        coupled: `(mask, flux)`, faces carrying a shared answer.
        dt_rule: `dt` is not known yet; the face kernel also returns the wave speed
            `S` and `dt = dt_rule(S)`; with the order-2 edge source at most `dt_rule` of
            the cells' own speeds too, as that term moves the cell states (ANALYSIS 48).
        split_diffusion: without `implicit_terms`, the implicit diffusion's part beyond
            its two-point fluxes (`diffusion_correction`) stays explicit, here (IMEX-RK:
            the two-point part is the stage's implicit term).
    Returns:
        `dQ`, or `(dQ, dt)` with `dt_rule`.
    References:
        LeVeque 2002, https://doi.org/10.1017/CBO9780511791253
        Pares 2006, https://doi.org/10.1137/050628052
    """
    QL, QR = reconstruct(nsm, Q, Qaux, p, time, order, demote, mesh)
    # the outer state of each boundary face's Riemann problem: the BC's face
    # value from the inner trace QL, at the face centre (every order)
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
        if (order >= 2 and nsm.call.numerical_cell_edge_source is not None
                and sp.S(demote) is not sp.S.true):
            # the edge source (the NCP inside each cell) moves the cell states with their
            # own speeds; no Riemann problem carries them where a cell's traces are dry
            dt = minimum(dt, dt_rule(cell_wavespeed(nsm, Q, Qaux, p, mesh)))
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
            dQ = dQ + diffusion(k, nsm.call.boundary_ghost_state, Q, Qaux, p,
                                time, mesh)
    if split_diffusion and not implicit_terms and nsm.call.diffusion_matrix is not None:
        dQ = dQ + diffusion_correction(nsm.call.diffusion_matrix,
                                       nsm.call.boundary_ghost_state, Q, Qaux, p, time,
                                       mesh)
    return dQ if dt_rule is None else (dQ, dt)


def stage_aux(nsm, Q, Qaux, p, time, dt, mesh):
    """Pointwise aux of a stage state (e.g. `hinv = 1/h`)."""
    if nsm.call.update_aux_variables is None:
        return Qaux
    return cells(nsm.call.update_aux_variables, Q, Qaux, p, time, dt, mesh)


def explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote=False,
                  coupled=None, implicit_terms=True, rk=None, dt_rule=None,
                  split_diffusion=False):
    """One SSP-RK step of the method of lines, of order `rk` (default the
    reconstruction order).

    Args:
        dt_rule: the first stage picks `dt` from its own face wave speeds.
        split_diffusion: see `residual`.
    Returns:
        `Q`, or `(Q, dt)` with `dt_rule`.
    References:
        Shu, Osher 1988, https://doi.org/10.1016/0021-9991(88)90177-5
        Gottlieb et al. 2001, https://doi.org/10.1137/S003614450036757X
    """
    Q0, Qaux_stage = Q, Qaux
    depth, tol = getattr(nsm.roles, "depth", None), getattr(nsm, "depth_round_off", None)
    for i, (a, c) in enumerate(SSP_RK[rk or order]):
        if i > 0:
            Qaux_stage = stage_aux(nsm, Q, Qaux, p, time + c * dt, dt, mesh)
        if i == 0 and dt_rule is not None:
            L, dt = residual(nsm, Q, Qaux_stage, p, time, None, mesh, order,
                             demote, coupled, implicit_terms, dt_rule, split_diffusion)
        else:
            L = residual(nsm, Q, Qaux_stage, p, time + c * dt, dt, mesh, order,
                         demote, coupled, implicit_terms,
                         split_diffusion=split_diffusion)
        Q = a * Q0 + (1 - a) * (Q + dt * L)
        if depth is not None and tol is not None:   # round-off below 0 is 0
            Q = clean_round_off(Q, (depth,), tol)
        Q = halo(Q, mesh)
    return Q if dt_rule is None else (Q, dt)


def mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled=None,
              implicit_terms=True, rk=None, dt_rule=None, split_diffusion=False):
    """A-posteriori MOOD: an order-2 candidate step, inadmissible cells redone at first
    order, and the whole step at first order if that still fails.  The repairs keep the
    candidate's `dt`; with `dt_rule` (the step from the face wave speeds) at most that,
    from their own face speeds, as the candidate's are those of the order-2 states and
    do not bound the first-order Riemann problems (ANALYSIS 48).

    References:
        Clain et al. 2011, https://doi.org/10.1016/j.jcp.2011.02.026
    """
    def step(demote, dt):
        """The explicit step with cells `demote` demoted to first order."""
        return explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order, demote,
                             coupled, implicit_terms, rk, split_diffusion=split_diffusion)

    if dt_rule is None:
        def repair():
            """Redo the bad cells at first order, then all cells if some still fail."""
            Q_cells = step(bad, dt)
            still = inadmissible(nsm.call.admissible, Q_cells, Qaux, p)
            return Cond(reduce_any(still), lambda: step(sp.S.true, dt),
                        lambda: Q_cells)

        Q_cand = step(sp.S.false, dt)
        bad = inadmissible(nsm.call.admissible, Q_cand, Qaux, p)
        return Cond(reduce_any(bad), repair, lambda: Q_cand)

    def ruled(demote, cap):
        """`(Q, dt)` of the step with cells `demote` demoted, `dt` at most `cap`."""
        return explicit_step(nsm, Q, Qaux, p, time, None, mesh, order, demote, coupled,
                             implicit_terms, rk, lambda S: minimum(cap, dt_rule(S)),
                             split_diffusion)

    def repair_ruled():
        """`repair` with each repair's own face speeds bounding its `dt`."""
        Q_cells, dt_cells = ruled(bad, dt)
        still = inadmissible(nsm.call.admissible, Q_cells, Qaux, p)
        return Cond(reduce_any(still), lambda: ruled(sp.S.true, dt_cells),
                    lambda: (Q_cells, dt_cells))

    Q_cand, dt = explicit_step(nsm, Q, Qaux, p, time, None, mesh, order, sp.S.false,
                               coupled, implicit_terms, rk, dt_rule, split_diffusion)
    bad = inadmissible(nsm.call.admissible, Q_cand, Qaux, p)
    Q, dt_step = Cond(reduce_any(bad), repair_ruled, lambda: (Q_cand, dt))
    return Q, minimum(dt_step, dt)          # a real-valued node (it enters `time`)


def full_aux(nsm, Q, Qaux, p, time, dt, mesh, coupled=None):
    """Pointwise aux, then the derivative (mesh-gradient) aux, whose boundary samples
    are the BC ghosts at `time`; `coupled`: `(own, face)` per coupled face, whose ghost
    samples are then the mirror images `2 face - own` of the coupling's face states.
    """
    Qaux = stage_aux(nsm, Q, Qaux, p, time, dt, mesh)
    spec = nsm.call.aux_derivatives
    if spec is None:
        return Qaux
    rows, functions = derivative_rows(nsm)
    ghost = nsm.call.boundary_ghost_state
    if functions or ghost is None:
        Qaux = derivative_aux(spec, Q, Qaux, mesh)
    if rows and ghost is not None:
        Qg = boundary_ghosts(ghost, Q, Qaux, p, time, mesh)
        if coupled is not None:
            Qg = coupled_ghosts(Qg, *coupled, mesh)
        Qaux = derivative_aux(spec, Q, Qaux, mesh, Qg, rows)
    return halo(Qaux, mesh)


def derivative_rows(nsm):
    """`(rows, functions)`: the state rows the derivative aux differentiates (static
    tuple) and whether it differentiates aux functions too.
    """
    spec = [e for e in nsm.aux_derivatives if e["kind"] == "derivative"]
    rows = tuple(sorted({int(e["state_index"]) for e in spec
                         if e["target_kind"] == "state"}))
    return rows, any(e["target_kind"] != "state" for e in spec)


def post_step(nsm, Q, Qaux, p, time, dt, mesh):
    """Model hygiene on the new state, then its aux; returns `(Q, Qaux)`."""
    if nsm.call.update_variables is not None:
        Q = cells(nsm.call.update_variables, Q, Qaux, p, time, dt, mesh)
    return Q, full_aux(nsm, Q, Qaux, p, time, dt, mesh)


def implicit_source(nsm, Q, Qaux, p, time, dt, mesh):
    """Implicit Euler on the source at `time`, `Y - Q - dt S(Y, aux(Y)) = 0`, by a
    per-cell root solve with the derivative aux frozen at `Q`.
    """
    Qaux_e = full_aux(nsm, Q, Qaux, p, time, dt, mesh)

    def aux_of(Y):
        """Pointwise aux following the iterate `Y`."""
        return stage_aux(nsm, Y, Qaux_e, p, time, dt, mesh)

    return Root(lambda Y: Y - Q - dt * cells(nsm.call.source, Y, aux_of(Y),
                                             p, time, dt, mesh), Q)


def implicit_step(nsm, Q, Qaux, p, time, dt, mesh):
    """The SystemModel's implicit slots over the step ending at `time`: implicit
    diffusion, then implicit Euler on the source.
    """
    if nsm.call.diffusion_matrix is not None:
        Q = implicit_diffusion(nsm.call.diffusion_matrix,
                               nsm.call.boundary_ghost_state, Q, Qaux, p,
                               time - dt, dt, mesh)
    if nsm.call.source is not None:
        Q = implicit_source(nsm, Q, Qaux, p, time, dt, mesh)
    return Q


def implicit_stage(nsm, Y, Qaux, p, time, dt, mesh):
    """The implicit part of an IMEX-RK stage at `time` over `dt` (`A_ii` times the
    step): `X - Y - dt (D(X) + S(X)) = 0`, `D` the two-point part of the implicit
    diffusion (`diffusion_tpfa`), `S` the implicit source.  Without diffusion a per-cell
    root (`implicit_source`), else one Newton solve over the mesh with the pointwise aux
    following the iterate and the derivative aux frozen at `Y`.

    References:
        Pareschi, Russo 2005, https://doi.org/10.1007/s10915-004-4636-4
        Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
    """
    kdiff, source = nsm.call.diffusion_matrix, nsm.call.source
    if kdiff is None:
        return Y if source is None else implicit_source(nsm, Y, Qaux, p, time, dt, mesh)
    ghost = nsm.call.boundary_ghost_state
    Qaux_e = full_aux(nsm, Y, Qaux, p, time, dt, mesh)

    def R(X):
        """The stage residual at the iterate `X`."""
        X = halo(X, mesh)
        aux = stage_aux(nsm, X, Qaux_e, p, time, dt, mesh)
        r = X - Y - dt * diffusion_tpfa(kdiff, ghost, X, aux, p, time, mesh)
        if source is not None:
            r = r - dt * cells(source, X, aux, p, time, dt, mesh)
        return r
    return halo(ImplicitSolve(R, Y), mesh)


def imex_step(nsm, Q, Qaux, p, time, dt, mesh, order, tableau, mood=False,
              coupled=None, dt_rule=None, project=None):
    """One step of the IMEX Runge-Kutta `tableau`: each stage is an explicit
    (MOOD-guarded) Euler step of the residual without implicit slots plus the
    implicit solve of the stage (`implicit_stage`).  The implicit part `g` is the
    source and the two-point part of the implicit diffusion; the rest of that
    diffusion's flux (`diffusion_correction`, zero in 1-D and on Cartesian faces
    without mixed entries) is in the explicit part `f`.

    Args:
        project: Chorin: `project(Y, time, dt, i) -> (Y', B)` maps stage `i` (the
            final combination: `i` omitted) onto the constraint, `B = Y' - Y` on the
            corrected rows.
        dt_rule: `dt` from the cell states' wave speeds; returns `(Q, dt)`.
    References:
        Pareschi, Russo 2005, https://doi.org/10.1007/s10915-004-4636-4
    """
    if dt_rule is not None:
        dt = dt_rule(cell_wavespeed(nsm, Q, Qaux, p, mesh))
    euler = mood_step if mood else explicit_step
    source, kdiff = nsm.call.source, nsm.call.diffusion_matrix
    F, G = [], []                   # dt f(Y_i); dt (g(Y_i) + G P_i)
    for i, c in enumerate(tableau.c):
        Y = Q + sum(tableau.At[i][j] * F[j] + tableau.A[i][j] * G[j]
                    for j in range(i))
        Y = implicit_stage(nsm, Y, Qaux, p, time + c * dt, tableau.A[i][i] * dt, mesh)
        B = sp.S.Zero
        if project is not None:
            Y, B = project(Y, time + c * dt, dt, i)
        Y_aux = full_aux(nsm, Y, Qaux, p, time + c * dt, dt, mesh)
        F.append(euler(nsm, Y, Y_aux, p, time + tableau.ct[i] * dt, dt, mesh,
                       order, coupled=coupled, implicit_terms=False, rk=1,
                       split_diffusion=True) - Y)
        G.append(B / tableau.A[i][i] + (
            sp.S.Zero if source is None else
            dt * cells(source, Y, Y_aux, p, time + c * dt, dt, mesh)))
        if kdiff is not None:
            G[i] = G[i] + dt * diffusion_tpfa(kdiff, nsm.call.boundary_ghost_state, Y,
                                              Y_aux, p, time + c * dt, mesh)
    Q = Q + sum(wt * f + w * g
                for wt, w, f, g in zip(tableau.wt, tableau.w, F, G))
    if project is not None:
        Q, _ = project(Q, time + dt, dt)
    return Q if dt_rule is None else (Q, dt)
