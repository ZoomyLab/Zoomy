"""E3 — ``state_from_reconstruction`` MUST be the left inverse of
``reconstruction_variables`` (REQ-180; spec §1b/§2 rank 3).

The WB primitive map limits the free surface ``eta = b + h`` and the velocity
``q/h`` / modal velocities ``alpha_k = q_k/h``.  Its inverse has to ``x-h``
every quotient row back to a conservative momentum.  A stale/duplicate forward
map left the velocity rows as identity, so the round trip returned
``[b, h, hu/h]`` — an O(h) momentum-flux error that detonates into an order-2
dry-bed ``1/h`` blow-up at wet/dry fronts on EVERY backend consuming the emit.
A hurried golden re-bless would accept that again; the semantic left-inverse
cannot.

No depth flooring anywhere in this path (user mandate).
"""

import sympy as sp
import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.reconstruction_inverse import reconstruction_symbols
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.systemmodel, pytest.mark.small, pytest.mark.gate]


def _build(spec):
    kind, kw = spec
    cls = {"SWE": SWE, "SME": SME, "VAM": VAM}[kind]
    if kind != "SWE":
        # A moment model with no stress closure is OPEN — its σ rows read zero
        # and ``SystemModel.from_model`` refuses it.  The reconstruction map is
        # closure-independent, so close them the production way rather than
        # reconstructing a model nobody would run.
        from zoomy_core.model.models.closures import (
            Newtonian, NavierSlip, StressFree)
        kw = dict(kw, closures=[Newtonian(), NavierSlip(), StressFree()])
    return SystemModel.from_model(cls(**kw))


CASES = {
    "swe-dim1":     ("SWE", {"dimension": 1}),
    "swe-dim2":     ("SWE", {"dimension": 2}),
    "sme-level1":   ("SME", {"dimension": 2, "level": 1}),
    "vam-level1":   ("VAM", {"dimension": 2, "level": 1}),
}


@pytest.mark.parametrize("label", list(CASES))
def test_left_inverse(label):
    """simplify(inv(fwd) − state) == 0 slot by slot; a model may opt out of
    primitive WB reconstruction (VAM routes bed slope conservatively, REQ-80)
    — then the invariant is CONSISTENCY: None <-> None."""
    sm = _build(CASES[label])
    fwd = sm.reconstruction_variables
    inv = sm.state_from_reconstruction

    if fwd is None:
        assert inv is None, (
            f"{label}: reconstruction_variables is None but "
            f"state_from_reconstruction is {list(inv)} — inconsistent")
        return

    state = list(sm.state)
    fwd = list(fwd)
    inv = list(inv)
    assert inv is not None and len(inv) == len(state)

    wb = reconstruction_symbols(state)
    subs = {wb[k]: sp.sympify(fwd[k]) for k in range(len(state))}
    for k, s in enumerate(state):
        back = sp.simplify(sp.sympify(inv[k]).xreplace(subs) - s)
        assert back == 0, (
            f"{label}: round trip broken in slot {k} ({s}): "
            f"state_from_reconstruction[{k}] = {inv[k]} feeds back to "
            f"{sp.simplify(sp.sympify(inv[k]).xreplace(subs))}, not {s}")


@pytest.mark.parametrize("label", ["swe-dim1", "swe-dim2", "sme-level1"])
def test_rest_manufactures_no_momentum(label):
    """Well-balancing: a reconstructed still lake (all velocities 0, flat
    surface) inverts to zero momentum — the inverse never manufactures flow."""
    sm = _build(CASES[label])
    state = list(sm.state)
    inv = list(sm.state_from_reconstruction)
    wb = reconstruction_symbols(state)
    rest = {}
    for k, s in enumerate(state):
        nm = str(s)
        rest[wb[k]] = wb[k] if nm in ("b", "h") else sp.Integer(0)
    for k, s in enumerate(state):
        if str(s) in ("b", "h"):
            continue
        val = sp.simplify(sp.sympify(inv[k]).xreplace(rest))
        assert val == 0, f"{label}: slot {s} momentum nonzero at rest: {val}"
