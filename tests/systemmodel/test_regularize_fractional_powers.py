"""``regularize_fractional_powers(eps)`` — the opt-in NSM op that shifts every
fractional-power base by ``eps**2`` so AD through a velocity norm at rest is
finite instead of ``0*inf = NaN``.

Pinned on SME(level=0, dimension=3) + Manning (the derived 2-D SWE, whose
friction source carries ``sqrt(u**2 + v**2)``):

* every fractional-power base in ``nsm.source`` carries the ``eps**2`` term;
* the op is idempotent;
* the source value at a wet moving state moves by < 1e-9 relative;
* ``eps <= 0`` raises.
"""
import pytest
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    ManningFriction, Newtonian, StressFree)
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel.operations import regularize_fractional_powers
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.small, pytest.mark.systemmodel, pytest.mark.nsm]

EPS = 1e-6
E2 = sp.Float(EPS) ** 2


def _sm():
    return SystemModel.from_model(SME(
        level=0, dimension=3,
        closures=[Newtonian(), ManningFriction(), StressFree()]))


@pytest.fixture(scope="module")
def nsms():
    plain = NumericalSystemModel.from_system_model(_sm())
    reg = NumericalSystemModel.from_system_model(
        _sm(), extra_operations=[regularize_fractional_powers(EPS)])
    return plain, reg


def _frac_pows(expr):
    return [p for p in sp.sympify(expr).atoms(sp.Pow)
            if p.exp.is_Rational and not p.exp.is_integer
            and not p.base.is_number]


def test_every_fractional_base_carries_eps2(nsms):
    _, reg = nsms
    pows = [p for e in sp.flatten(reg.source) for p in _frac_pows(e)]
    assert pows, "Manning source should carry fractional powers"
    for p in pows:
        assert isinstance(p.base, sp.Add) and E2 in p.base.args, p


def test_idempotent(nsms):
    _, reg = nsms
    before = list(sp.flatten(reg.source))
    reg.apply(regularize_fractional_powers(EPS))
    assert list(sp.flatten(reg.source)) == before


def _value(nsm):
    h, qx, qy = 1.3, 0.4, -0.2
    vals = {"h": h, "q_x_0": qx, "q_y_0": qy, "hinv": 1.0 / h, "b": 0.0,
            "g": 9.81, "n": 0.03, "e_x": 0.1, "rho": 1000.0, "nu": 1e-3,
            "lambda_s": 0.0}
    out = []
    for e in sp.flatten(nsm.source):
        e = sp.sympify(e)
        sub = {s: vals[s.name] for s in e.free_symbols}
        out.append(float(e.subs(sub)))
    return out


def test_value_change_is_tiny(nsms):
    plain, reg = nsms
    a, b = _value(plain), _value(reg)
    scale = max(abs(x) for x in a)
    assert scale > 0
    for x, y in zip(a, b):
        assert abs(x - y) <= 1e-9 * scale, (x, y)


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_nonpositive_eps_raises(eps):
    with pytest.raises(ValueError):
        regularize_fractional_powers(eps)


def test_derivative_at_rest_is_finite(nsms):
    """The purpose: d(source)/d(q) at q = 0 is finite with the op.  The
    velocity norm's derivative is evaluated as ``0 * inf`` without it."""
    _, reg = nsms
    vals = {"h": 1.3, "q_x_0": 0.0, "q_y_0": 0.0, "hinv": 1.0 / 1.3,
            "b": 0.0, "g": 9.81, "n": 0.03, "e_x": 0.1}
    for e in sp.flatten(reg.source):
        e = sp.sympify(e)
        for q in [s for s in e.free_symbols if s.name in ("q_x_0", "q_y_0")]:
            d = sp.diff(e, q)
            v = complex(d.subs({s: vals[s.name] for s in d.free_symbols}))
            assert v == v and abs(v) < float("inf"), (e, q, v)
