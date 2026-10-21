"""Classification-completeness guard (cid-182 follow-up).

The structural extractor routes every additive residual term into a slot
(``M / F / P / B / A / S``).  A term that matches NO branch is a SILENT DROP —
the class of bug that hid the VAM bed-slope ``∂_x(A·∂_x b)`` for weeks.  The
guard is a TERM-CONSERVATION check: reassemble each row from the slots it was
routed to and subtract the original residual; a nonzero remainder is a dropped
term.  It is generic (balance-law algebra only, no per-system knowledge) and is
enforced two ways:

* a warning on ``from_model`` by default, and
* the opt-in strict :meth:`SystemModel.assert_all_classified`.

These tests pin (a) the guard is SILENT on the healthy families — an empty
remainder is a hard requirement, a trip here is a real finding — and (b) it
CATCHES a deliberately dropped term (a classifier branch made to fail to route),
surfacing it in ``describe(full=True)`` and failing ``assert_all_classified``.
"""
import warnings

import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from zoomy_core.model.derivation import system_extract
from zoomy_core.model.models import SME, VAM, MLVAM, MLSME, MLSWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.systemmodel, pytest.mark.small, pytest.mark.gate]


def _strip(expr):
    """Normalise a residual term to bare-Symbol form so a stored remainder
    (state Symbols) compares against a captured original term (state
    Functions): ``Derivative(f(t,x), x) → Symbol('D_f_x')``, ``f(t,x) → f``."""
    expr = sp.sympify(expr)
    dmap = {}
    for d in expr.atoms(sp.Derivative):
        base = d.args[0]
        nm = base.func.__name__ if isinstance(base, AppliedUndef) else str(base)
        dmap[d] = sp.Symbol("D_%s_%s" % (nm, "_".join(map(str, d.variables))))
    expr = expr.xreplace(dmap)
    fmap = {f: sp.Symbol(f.func.__name__) for f in expr.atoms(AppliedUndef)}
    expr = expr.xreplace(fmap)
    # Drop assumptions so a positive state Symbol (``h``) and a plain one from a
    # lifted Function compare equal.
    smap = {s: sp.Symbol(s.name) for s in expr.free_symbols}
    return sp.expand(expr.xreplace(smap))


_HEALTHY = {
    "SME(0)": lambda: SME(level=0,
                          closures=[Newtonian(), NavierSlip(), StressFree()]),
    "SME(2)": lambda: SME(level=2,
                          closures=[Newtonian(), NavierSlip(), StressFree()]),
    "VAM(1,2)": lambda: VAM(level=1, dimension=2,
                            closures=[Newtonian(), StressFree()]),
    "MLVAM(1,1)": lambda: MLVAM(n_layers=1, level=1, dimension=2,
                                closures=[Newtonian(), StressFree()]),
    "MLVAM(2,1)": lambda: MLVAM(n_layers=2, level=1, dimension=2,
                                closures=[Newtonian(), StressFree()]),
    # Closures on EVERY entry: "healthy" has to mean closed.  These two were
    # the only families built bare, and they carry the same four unbound
    # interface stresses ML-VAM does — silently frictionless internal
    # interfaces, which is not a healthy family.
    "MLSWE(2)": lambda: MLSWE(n_layers=2, dimension=2,
                              closures=[Newtonian(), StressFree()]),
    "MLSME(2,1)": lambda: MLSME(n_layers=2, level=1, dimension=2,
                                closures=[Newtonian(), StressFree()]),
}


@pytest.mark.parametrize("name", list(_HEALTHY))
def test_healthy_families_fully_classified(name):
    """Every declarative family routes every term — no silent drops, no
    warning, and ``assert_all_classified`` returns the model unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # a guard warning would fail
        sm = SystemModel.from_model(_HEALTHY[name]())
    assert sm.unclassified_terms == []
    assert sm.assert_all_classified() is sm


def test_describe_full_shows_clean_section():
    """``describe(full=True)`` carries a clearly-labelled, non-silent
    completeness section even when clean (absence must not read as unchecked)."""
    sm = SystemModel.from_model(
        VAM(level=1, dimension=2, closures=[Newtonian(), StressFree()]))
    txt = str(sm.describe(full=True))
    assert "Unclassified terms" in txt
    assert "none" in txt.split("Unclassified terms", 1)[1][:80]


def test_guard_catches_dropped_term(monkeypatch):
    """A classifier branch that silently fails to route a term is CAUGHT.

    Simulates the historical bed-slope drop: on the momentum row, the first
    spatial-derivative term is removed before classification, so the operators
    lose it while the stored residual keeps it.  The term-conservation guard
    must recover exactly the dropped term, warn on build, list it in
    ``describe(full=True)`` and fail ``assert_all_classified``."""
    real = system_extract._classify_row
    dropped = {}

    def dropping(residual, i, state, state_funcs, t, space, gravity_param,
                 F, P, B, S, M, A):
        if i == 2:                               # a momentum row
            keep = sp.S.Zero
            done = False
            for term in sp.Add.make_args(residual):
                has_spatial = any(
                    v in space for der in term.atoms(sp.Derivative)
                    for v in der.variables)
                if has_spatial and not done:
                    dropped[i] = term
                    done = True
                    continue
                keep += term
            residual = keep
        return real(residual, i, state, state_funcs, t, space, gravity_param,
                    F, P, B, S, M, A)

    monkeypatch.setattr(system_extract, "_classify_row", dropping)

    with pytest.warns(UserWarning, match="classified into NO slot"):
        sm = SystemModel.from_model(
            SME(level=0,
                closures=[Newtonian(), NavierSlip(), StressFree()]))

    assert dropped, "the test did not actually drop a term"
    rows = [i for i, _ in sm.unclassified_terms]
    assert rows == [2]
    # The recovered remainder equals the dropped term exactly.
    (_, remainder), = sm.unclassified_terms
    assert _strip(remainder) == _strip(dropped[2])

    txt = str(sm.describe(full=True))
    assert "Unclassified terms" in txt and "row 2" in txt

    with pytest.raises(ValueError, match="unclassified operator term"):
        sm.assert_all_classified()
