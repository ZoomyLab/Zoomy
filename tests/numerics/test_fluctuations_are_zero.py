"""REQ-209 — ``fluctuations_are_zero`` is a property of the (model x Riemann)
PAIR, decided from the BUILT numerics, and it must reach the C-family headers.

The two cases below are the ones that killed the predecessor detector
(``generic_c._detect_has_nonconservative_product``, which keyed on the model's
``nonconservative_matrix``).  They fail it in OPPOSITE directions, so no
model-only test can pass both:

* 2-D SWE has a genuinely non-zero model NCP (``B[hu][b] = B[hv][b] = g*h``,
  bed slope, because ``b`` is in the state) — yet under ``Rusanov`` / ``HLL`` /
  ``HLLC`` every emitted fluctuation entry is literal zero: they never override
  ``numerical_fluctuations`` and inherit the identically-zero base, regardless
  of the model (the symbolic solvers refuse that pair at setup: it drops the bed
  slope).  A model-only detector blocks the conservative fast path here.
* ``PositiveRusanov`` on the same SWE emits a fluctuation: Audusse's
  consistency source restoring the pressure SWE keeps in its flux (ANALYSIS 50;
  until 2026-10-14 it read only ``hydrostatic_pressure``, empty for SWE, and the
  lake at rest moved).
"""

import sympy as sp

from zoomy_core.fvm.riemann_solvers import (
    HLL,
    HLLC,
    NonconservativeRusanov,
    PositiveNonconservativeHLL,
    PositiveRusanov,
    Rusanov,
)
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel,
    ReconstructionSpec,
)
from zoomy_core.transformation.generic_c import CppModel, CppNumerics

import pytest

# Post-spec keep (REQ-209 landed after the 2026-07-19 refactor spec): the
# fluctuations_are_zero flag is decided from the BUILT numerics and read by
# printers — no golden pins the (model x Riemann) decision table.
pytestmark = [pytest.mark.nsm, pytest.mark.small]


def _nsm(model, riemann):
    return NumericalSystemModel.from_model(
        model, riemann=riemann, reconstruction=ReconstructionSpec(order=1))


def _swe():
    return SWE(dimension=2, parameters={"g": 9.81})


def _sme():
    # Closed: a level-1 SME with no stress closure is OPEN (its σ rows read
    # zero) and ``SystemModel.from_model`` refuses it.  Fluctuation structure
    # is closure-independent, so close it the production way.
    from zoomy_core.model.models.closures import (
        Newtonian, NavierSlip, StressFree)
    return SME(dimension=2, level=1, parameters={"g": 9.81},
               closures=[Newtonian(), NavierSlip(), StressFree()])


def test_swe_model_ncp_is_nonzero():
    """The premise: SWE's model NCP really is non-zero, so the two tests below
    are not vacuous — they genuinely contradict a model-only detector."""
    nsm = _nsm(_swe(), PositiveRusanov)
    entries = [sp.sympify(e) for e in sp.flatten(nsm.nonconservative_matrix)]
    assert any(e.is_zero is not True for e in entries)


def test_positive_rusanov_carries_swe_consistency_source():
    """SWE keeps ``g h^2/2`` in its flux: the consistency source ``S~`` of the
    hydrostatic reconstruction restores it, so a fluctuation is emitted."""
    nsm = _nsm(_swe(), PositiveRusanov)
    assert nsm.fluctuations_are_zero is False


def test_conservative_solvers_never_build_a_fluctuation():
    """Rusanov/HLL/HLLC inherit the identically-zero base for ANY model."""
    for riemann in (Rusanov, HLL, HLLC):
        assert _nsm(_swe(), riemann).fluctuations_are_zero is True
        assert _nsm(_sme(), riemann).fluctuations_are_zero is True


def test_nonconservative_solvers_do_build_one():
    assert _nsm(_swe(), NonconservativeRusanov).fluctuations_are_zero is False
    # PositiveNonconservativeHLL keeps a genuine fluctuation whenever the
    # model has one: SME's moment-coupling NCP survives the Audusse
    # reconstruction (only the bed column is equalised).
    assert _nsm(_sme(), PositiveNonconservativeHLL).fluctuations_are_zero \
        is False


def test_positive_nonconservative_hll_carries_swe_consistency_source():
    """Single-dissipation PosNCHLL on SWE: the reconstruction equalises ``b*``
    (kills the bed-column NCP) and the LF ``s_max·Id·Δq`` lives ONLY in the HLL
    flux, but the consistency source ``S~`` of the pressure SWE keeps in its flux
    remains (ANALYSIS 50)."""
    assert _nsm(_swe(), PositiveNonconservativeHLL).fluctuations_are_zero \
        is False


def test_flag_is_a_property_of_the_pair():
    """One model, two Riemann classes, two answers — the flag reads the built
    expression rather than keying on the model or on the Riemann class."""
    assert _nsm(_swe(), Rusanov).fluctuations_are_zero is True
    assert _nsm(_swe(), PositiveRusanov).fluctuations_are_zero is False
    assert _nsm(_sme(), PositiveRusanov).fluctuations_are_zero is False


def test_flag_is_reread_not_cached():
    """It must track ``apply()``: the numerics is built FIRST, then an
    operation changes the operator the fluctuation is built from.  A cached
    boolean goes stale in both directions."""
    def _zero_pressure(sm):
        P = sm.hydrostatic_pressure
        sm.hydrostatic_pressure = type(P)(
            [0] * len(list(sp.flatten(P)))).reshape(*P.shape)
    _zero_pressure.name = "zero_pressure"

    nsm = _nsm(_sme(), PositiveRusanov)
    nsm.build_numerics()                       # the staleness window
    assert nsm.fluctuations_are_zero is False
    nsm.apply(_zero_pressure)
    assert nsm.fluctuations_are_zero is True


def test_unknown_collapses_to_has_a_fluctuation():
    """Only a PROVEN structural zero licenses the fast path: skipping a real
    fluctuation is silently-wrong physics, a spurious one costs a branch."""
    nsm = _nsm(_swe(), PositiveRusanov)

    def _probe(definition):
        class _P:
            class functions:
                class numerical_fluctuations:
                    pass
        _P.functions.numerical_fluctuations.definition = definition
        nsm._probe_numerics = lambda: _P()
        return nsm.fluctuations_are_zero

    class _NoSlot:
        class functions:
            pass

    nsm._probe_numerics = lambda: _NoSlot()
    assert nsm.fluctuations_are_zero is False          # slot absent
    assert _probe([]) is False                         # nothing built
    assert _probe([sp.Symbol("a")]) is False           # is_zero is None
    assert _probe([sp.Integer(0), sp.Integer(0)]) is True


def test_flag_is_emitted_into_both_headers():
    """REQ-209 defect 2: the constant must actually reach the backends.
    ``Numerics.H`` is its primary home (it is the (model x riemann) artifact);
    ``Model.H`` carries it alongside ``has_diffusion`` / ``dt_max``."""
    for riemann, expected in ((Rusanov, "true"),
                              (NonconservativeRusanov, "false")):
        nsm = _nsm(_swe(), riemann)
        line = f"static constexpr bool fluctuations_are_zero = {expected};"
        assert line in CppModel(nsm).create_code()
        assert line in CppNumerics(nsm.build_numerics()).create_code()


def test_old_model_only_detector_is_gone():
    """It must not come back: it answered a different question than the one
    the drivers need, and having two answer-holders is how they diverged."""
    assert not hasattr(CppModel, "_detect_has_nonconservative_product")
