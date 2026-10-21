"""Locking tests for the explicit affine pressure operator (REQ-94).

``SM_press.pressure_operator()`` extracts the elliptic pressure operator that
is otherwise only implicit in ``SM_press.source`` (the matrix-free residual
``A·P − RHS``).  These tests lock the three REQ-94 invariants for every
chorin-split model:

1. **Symbolic residual == 0** — the coefficient fields reproduce
   ``source_k`` exactly on symbolic ``P``.
2. **Probe match** — ``A0`` equals the matrix-free probe
   ``source_k(e_l) − source_k(0)`` at a random concrete cell.
3. **Shapes** — ``A0``/first/second are ``NP×NP``, ``RHS`` length ``NP``,
   ``P_modes`` == the code's pressure ordering ``equation_to_state_index``.

The new option is opt-in and non-breaking: it does NOT re-derive; it
symbolically differentiates the unchanged ``SM_press.source``.
"""
import random

import pytest
import sympy as sp

from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.ml_vam import MLVAM
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree)

#: A moment model with no stress closure is OPEN — its sigma rows read zero
#: and ``SystemModel.from_model`` refuses it.  The Chorin pressure operator
#: is closure-independent, so close them the production way.
_CLOS = lambda: [Newtonian(), NavierSlip(), StressFree()]

# R4 (approved spec §1c): wholesale rederive — operator-vs-matrix-free
# self-consistency, run on re-baseline / by tag, zero default-gate cost.
pytestmark = [pytest.mark.systemmodel, pytest.mark.rederive]


def _row_scalar(smp, i):
    src = smp.source
    try:
        return sp.sympify(src[i, 0])
    except (TypeError, IndexError):
        return sp.sympify(src[i])


def _deriv_aux_map(smp, e2s):
    """(P column l, multi_index) → pressure-derivative aux Symbol."""
    col = {si: l for l, si in enumerate(e2s)}
    reg = list(smp.aux_registry or [])
    reg += list(getattr(smp, "aux_input_registry", None) or [])
    out = {}
    for e in reg:
        if e.get("kind") in ("derivative", "limited_derivative") \
                and e.get("state_index") in col:
            out[(col[e["state_index"]], tuple(e["multi_index"]))] = \
                e["aux_symbol"]
    return out


def _reconstruct(op, smp, aux_of, ndim, k):
    """Rebuild source_k from the operator coefficient fields."""
    rec = -op.RHS[k]
    for l in range(len(op.P_modes)):
        rec += op.A0[k][l] * op.P_modes[l]
    for a, M in op.first_derivative.items():
        mi = tuple(1 if d == a else 0 for d in range(ndim))
        for l in range(len(op.P_modes)):
            s = aux_of.get((l, mi))
            if s is not None:
                rec += M[k][l] * s
    for (a, b), M in op.second_derivative.items():
        mi = [0] * ndim
        mi[a] += 1
        mi[b] += 1
        mi = tuple(mi)
        for l in range(len(op.P_modes)):
            s = aux_of.get((l, mi))
            if s is not None:
                rec += M[k][l] * s
    return rec


def _verify(model, seed=0):
    """Run all three REQ-94 checks for one chorin-split model."""
    smp = model.chorin_split().SM_press
    op = smp.pressure_operator()
    state = list(smp.state)
    e2s = list(smp.equation_to_state_index)
    NP = len(op.P_modes)
    ndim = len(smp.space)

    # ── (3) shapes / ordering ──
    assert op.P_modes == [state[i] for i in e2s]
    assert len(op.A0) == NP and all(len(r) == NP for r in op.A0)
    assert len(op.RHS) == NP
    for M in list(op.first_derivative.values()) \
            + list(op.second_derivative.values()):
        assert len(M) == NP and all(len(r) == NP for r in M)

    aux_of = _deriv_aux_map(smp, e2s)

    rng = random.Random(seed)
    p_syms = set(op.P_modes)
    d_syms = set(aux_of.values())
    all_syms = set().union(*[_row_scalar(smp, k).free_symbols
                             for k in range(NP)])

    # ── (1) reconstruction == source, every row ──
    # Evaluated at a random concrete cell (pressure modes + derivative aux
    # given random NON-zero values too, so the whole affine operator — RHS,
    # A0, first/second-derivative fields — is exercised).  ``sp.simplify(...)
    # == 0`` is NOT a decision procedure and is hash-order dependent for the
    # larger ML_VAM residuals (see check (2) below), so certify the true-zero
    # numerically instead — deterministic and cache-independent.
    probe_all = {s: sp.Rational(rng.randint(2, 9), rng.randint(1, 5))
                 for s in all_syms}
    for k in range(NP):
        rec = _reconstruct(op, smp, aux_of, ndim, k)
        diff = sp.N(sp.expand(_row_scalar(smp, k) - rec).xreplace(probe_all))
        assert abs(complex(diff)) < 1e-9, \
            f"reconstruction residual != 0 on row {k}: {diff}"

    # ── (2) A0 == matrix-free probe at a random concrete cell ──
    frozen = [s for s in all_syms if s not in p_syms and s not in d_syms]
    subs0 = {s: sp.Rational(rng.randint(2, 9), rng.randint(1, 5))
             for s in frozen}
    subs0.update({s: 0 for s in p_syms})
    subs0.update({s: 0 for s in d_syms})
    for k in range(NP):
        base = _row_scalar(smp, k).xreplace(subs0)
        for l in range(NP):
            probe = dict(subs0)
            probe[op.P_modes[l]] = 1
            got = _row_scalar(smp, k).xreplace(probe) - base
            # Numeric zero-test at the concrete cell: ``sp.simplify(...) == 0``
            # is not a decision procedure and fails to certify a true-zero for
            # the larger ML_VAM residuals (hash-order dependent), so evaluate
            # the probe difference to a float and compare with tolerance.
            diff = sp.N(sp.expand(got) - op.A0[k][l].xreplace(subs0))
            assert abs(complex(diff)) < 1e-9, \
                f"A0 probe mismatch at ({k},{l}): {diff}"
    return op


def test_pressure_operator_vam_1_2():
    op = _verify(VAM(level=1, dimension=2, closures=_CLOS()))
    assert len(op.P_modes) == 2
    assert op.Axx is not None          # one horizontal coord → x only
    assert op.Ax is not None
    assert op.Ayy is None


def test_pressure_operator_lazy_cached_and_nonbreaking():
    smp = VAM(level=1, dimension=2,
              closures=_CLOS()).chorin_split().SM_press
    # source is intact / probe path still valid (no re-derivation side effect)
    src_before = sp.sympify(smp.source[0, 0])
    op1 = smp.pressure_operator()
    op2 = smp.pressure_operator()
    assert op1 is op2                                    # cached
    assert sp.sympify(smp.source[0, 0]) == src_before    # source untouched


@pytest.mark.rederive
def test_pressure_operator_vam_1_3():
    op = _verify(VAM(level=1, dimension=3, closures=_CLOS()))
    assert len(op.P_modes) == 2
    assert op.Ax is not None and op.Ay is not None       # x,y horizontal
    assert op.Axx is not None and op.Ayy is not None


@pytest.mark.rederive
def test_pressure_operator_ml_vam():
    op = _verify(MLVAM(n_layers=2, level=1, dimension=2, closures=_CLOS()))
    assert len(op.P_modes) >= 2
