"""Face reconstruction of the solver templates: order 1 takes cell values of the
reconstruction variables ``W(Q)`` to the faces, order 2 a limited LSQ gradient, and
the inverse map recovers ``Q``. Nothing is clipped: face positivity is the flux
kernel's, cell positivity MOOD's.
"""
import sympy as sp

from zoomy_core.symsolver.core import unpack
from zoomy_core.symsolver.vocabulary import (
    barth_jespersen, boundary_states, cell_traces, extrapolate, face_aux,
    gradient, minimum, minmod, pointwise, put, select, take, venkatakrishnan)

LIMITERS = {"minmod": minmod, "venkatakrishnan": venkatakrishnan,
            "barth_jespersen": barth_jespersen}


def reconstruct(nsm, Q, Qaux, p, time, order, demote, mesh):
    """Face traces ``(QL, QR)`` of order 1 (cell values) or 2 (limited LSQ gradient);
    ``QR`` on a boundary face is a placeholder for the BC value.

    Args:
        demote: ``False`` none, ``True`` all cells, or a cell mask; demoted cells are
            constant in ``W`` (MOOD).
    References:
        van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    demote = sp.S(demote)
    to_w, to_q = maps(nsm)
    rows = bed_rows(nsm) if to_w is not None else None
    W = Q if to_w is None else pointwise(to_w, Q, Qaux, p)
    if rows is not None:               # reconstruct h in the bed's slot
        bed, depth = rows
        W = put(W, (bed,), take(Q, (depth,)))
    if order == 1 or demote is sp.S.true:
        WL, WR = unpack(cell_traces(W, mesh), 2)
    else:
        Qg, Qauxg = unpack(boundary_states(nsm.call.boundary_ghost_state, Q,
                                           Qaux, p, time, mesh), 2)
        Wg = Qg
        if to_w is not None:
            if nsm.call.update_aux_variables is not None:   # the ghost's own aux
                Qauxg = pointwise(nsm.call.update_aux_variables, Qg, Qauxg, p)
            Wg = pointwise(to_w, Qg, Qauxg, p)
        if rows is not None:
            Wg = put(Wg, (bed,), take(Qg, (depth,)))
        G = gradient(W, Wg, mesh)
        phi = limited(nsm, W, Wg, G, mesh)
        if demote is not sp.S.false:
            phi = select(demote, 0, phi)
        WL, WR = unpack(extrapolate(W, G, phi, mesh), 2)
    if rows is not None:               # b_f = eta_f - h_f
        WL, WR = (put(X, (bed,), take(X, (depth,)) - take(X, (bed,))) for X in (WL, WR))
    if to_q is None:
        return WL, WR
    AL, AR = unpack(face_aux(Qaux, mesh), 2)
    return pointwise(to_q, WL, AL, p), pointwise(to_q, WR, AR, p)


def maps(nsm):
    """``(W(Q), Q(W))`` kernels, or ``(None, None)`` when ``Q`` itself is reconstructed.
    """
    what = nsm.reconstruction.variables
    if what == "conservative":
        return None, None
    if what != "model":
        raise ValueError(f"reconstruction variables {what!r}: 'model' or "
                         "'conservative'")
    W = nsm.operator("reconstruction_variables")
    if W is None or [str(w) for w in W.definition] == [str(q) for q in nsm.state]:
        return None, None                       # the model's map is Q
    return nsm.call.reconstruction_variables, nsm.call.state_from_reconstruction


def bed_rows(nsm):
    """``(bed, depth)`` rows when the reconstruction is the pair (h, eta = b + h) with
    ``bed="derived"``, else None.

    References:
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """
    if nsm.reconstruction.bed == "reconstructed":
        return None
    if nsm.reconstruction.bed != "derived":
        raise ValueError(f"reconstruction bed {nsm.reconstruction.bed!r}: "
                         "'derived' or 'reconstructed'")
    cached = nsm.__dict__.get("_bed_rows", False)
    if cached is not False:
        return cached
    roles, rows = nsm.roles, None
    if roles.depth is not None and roles.bed is not None:
        W = nsm.operator("reconstruction_variables").definition
        b, h = nsm.state[roles.bed], nsm.state[roles.depth]
        if (sp.expand(W[roles.bed] - b) == 0
                and sp.expand(W[roles.depth] - b - h) == 0):  # free-surface map
            rows = (roles.bed, roles.depth)
    nsm.__dict__["_bed_rows"] = rows
    return rows


def limited(nsm, W, Wg, G, mesh):
    """Limiter factor per row and cell (the NSM's limiter, per-row overrides, then
    ties), 1 where no row is limited.
    """
    spec, names = nsm.reconstruction, [str(q) for q in nsm.state]
    unknown = (set(spec.limiters) | set(spec.ties) | set(spec.ties.values())) - set(names)
    if unknown:
        raise ValueError(f"reconstruction rows {sorted(unknown)} are not states "
                         f"of the model ({names})")
    by_kind = {}
    for i, name in enumerate(names):
        by_kind.setdefault(spec.limiters.get(name, spec.limiter), []).append(i)
    phi = sp.S.One
    for kind, rows in by_kind.items():
        if kind == "none":
            continue
        if kind not in LIMITERS:
            raise ValueError(f"limiter {kind!r}: one of {sorted(LIMITERS)} or 'none'")
        factor = LIMITERS[kind](tuple(rows), W, Wg, G, mesh)
        phi = factor if phi is sp.S.One else minimum(phi, factor)
    if phi is sp.S.One:
        return phi
    by_lead = {}
    for row, lead in spec.ties.items():
        by_lead.setdefault(names.index(lead), []).append(names.index(row))
    for lead, rows in by_lead.items():          # phi_row <= phi_lead
        rows = tuple(rows)
        phi = put(phi, rows, minimum(take(phi, rows), take(phi, (lead,))))
    return phi
