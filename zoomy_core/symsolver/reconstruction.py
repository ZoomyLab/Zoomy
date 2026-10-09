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
    barth_jespersen, cell_traces, extrapolate, face_aux, gradient, minimum,
    minmod, pointwise, put, select, take, venkatakrishnan)

LIMITERS = {"minmod": minmod, "venkatakrishnan": venkatakrishnan,
            "barth_jespersen": barth_jespersen}


def reconstruct(nsm, Q, Qaux, Qb, Qauxb, p, order, demote, mesh):
    """Face traces ``(QL, QR)``; ``QR`` on a boundary face is ``Qb``.
    ``demote``: ``False`` none, ``True`` all cells, or a cell mask -- demoted
    cells are constant in ``W`` (MOOD)."""
    demote = sp.S(demote)
    to_w, to_q = maps(nsm)
    W, Wb = Q, Qb
    if to_w is not None:
        if nsm.call.update_aux_variables is not None:   # the ghost's own aux
            Qauxb = pointwise(nsm.call.update_aux_variables, Qb, Qauxb, p)
        W, Wb = pointwise(to_w, Q, Qaux, p), pointwise(to_w, Qb, Qauxb, p)
    if order == 1 or demote is sp.S.true:
        WL, WR = unpack(cell_traces(W, Wb, mesh), 2)
    else:
        G = gradient(W, Wb, mesh)
        phi = limited(nsm, W, Wb, G, mesh)
        if demote is not sp.S.false:
            phi = select(demote, 0, phi)
        WL, WR = unpack(extrapolate(W, Wb, G, phi, mesh), 2)
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


def limited(nsm, W, Wb, G, mesh):
    """Limiter factor per row and cell: the NSM's limiter, its per-row
    overrides, then its ties; 1 where no row is limited."""
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
        factor = LIMITERS[kind](tuple(rows), W, Wb, G, mesh)
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
