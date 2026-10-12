"""Face reconstruction, written once for every backend.

What is reconstructed is the NSM's choice (``nsm.reconstruction.variables``):
the model's reconstruction variables ``W(Q)`` with their inverse, or ``Q``
itself.  Order 1 takes the cell values of ``W`` to the faces (so ``Q(W(Q))``
there: the model's map, e.g. a desingularized velocity, applies at every
order).  Order 2 takes the LSQ
gradient of ``W``, limits it row by row (``nsm.reconstruction``: the limiter,
per-row overrides, ties; "none" leaves a row unlimited) and moves each cell's
``W`` to the face centre; the inverse maps the traces back to ``Q``.  Nothing
is clipped: face positivity is the flux kernel's (hydrostatic
reconstruction), cell positivity MOOD's.

Ground truth: ``zoomy_jax.fvm.reconstruction_jax.LSQMUSCLReconstructionJAX``
and ``EmittedWLSQMUSCLJAX`` without its face clamp ``eta_f >= b_f``.
"""
import sympy as sp

from zoomy_core.symsolver.core import unpack
from zoomy_core.symsolver.vocabulary import (
    barth_jespersen, boundary_states, cell_traces, extrapolate, face_aux,
    gradient, minimum, minmod, pointwise, put, select, take, venkatakrishnan)

LIMITERS = {"minmod": minmod, "venkatakrishnan": venkatakrishnan,
            "barth_jespersen": barth_jespersen}


def reconstruct(nsm, Q, Qaux, p, time, order, demote, mesh):
    """Face traces ``(QL, QR)``; ``QR`` on a boundary face is the inner
    trace, a placeholder for the BC's face value (``boundary_traces``).
    ``demote``: ``False`` none, ``True`` all cells, or a cell mask -- demoted
    cells are constant in ``W`` (MOOD).  The gradient and the limiter take
    their boundary samples from the BC GHOSTS at ``time``
    (``nsm.call.boundary_ghost_state``, at the mirror image of the cell
    centre).  With the model's free-surface map the pair is (h, eta) and
    b_f = eta_f - h_f (:func:`bed_rows`)."""
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
    """``(W(Q), Q(W))`` kernels, or ``(None, None)``: ``Q`` itself."""
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
    """``(bed, depth)`` rows when the reconstruction is Audusse's pair
    (h, eta = b + h) -- the NSM's ``bed="derived"`` with a model that declares
    both roles and maps them to (b, b + h) -- else None."""
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
    """Limiter factor per row and cell: the NSM's limiter, its per-row
    overrides, then its ties; 1 where no row is limited.  ``Wg``: the
    boundary ghosts (bounds at the ghost position)."""
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
