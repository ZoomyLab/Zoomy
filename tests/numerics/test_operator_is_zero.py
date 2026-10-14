"""Per-operator structural-zero booleans — "may a backend skip this slot?"

The sibling of ``test_fluctuations_are_zero.py`` and deliberately NOT the same
question.  ``fluctuations_are_zero`` reads a BUILT face kernel, so an absent
slot means "cannot tell" (⇒ False); these read MODEL-LEVEL slots, where absent
means the model declares no such operator (⇒ True).  Both collapse UNKNOWN to
the unsafe-to-skip side.

The measured motivation is ``diffusion_matrix_is_zero``: SWE carries a
correctly-SHAPED ``diffusion_matrix`` of literal zeros, and the JAX backend's
shape-only gate therefore built a dense TPFA diffusion operator for it —
2287 s of a 2539 s LowerTriangle run assembling an identically-zero operator.
"""

import sympy as sp

from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    OPERATOR_ZERO_FLAGS,
    NumericalSystemModel,
    ReconstructionSpec,
    operator_is_zero,
    operator_zero_flags,
)
from zoomy_core.transformation.generic_c import CppModel, CppNumerics

import pytest

pytestmark = [pytest.mark.nsm, pytest.mark.small]


def _nsm(model, riemann=Rusanov):
    return NumericalSystemModel.from_model(
        model, riemann=riemann, reconstruction=ReconstructionSpec(order=1))


def _swe():
    return SWE(dimension=2, parameters={"g": 9.81})


# ── the criterion itself ──────────────────────────────────────────────────

def test_criterion_semantics():
    """absent/empty ⇒ True; proven zero ⇒ True; non-zero ⇒ False; and
    UNDECIDABLE ⇒ False, the safety direction (skipping a real operator is
    silently-wrong physics, a spurious evaluation only costs time)."""
    assert operator_is_zero(None) is True                       # absent
    assert operator_is_zero([]) is True                         # empty
    assert operator_is_zero(sp.zeros(2, 3)) is True             # proven zero
    assert operator_is_zero([sp.Integer(0), sp.Integer(0)]) is True
    assert operator_is_zero([sp.Integer(0), sp.Integer(1)]) is False
    # is_zero is None for a bare symbol with no sign assumption.
    a = sp.Symbol("a")
    assert a.is_zero is None
    assert operator_is_zero([sp.Integer(0), a]) is False        # undecidable


def test_absent_disagrees_with_fluctuations_on_purpose():
    """The one row where the two criteria differ, pinned so a future "unify
    them" refactor fails here instead of inverting a physics gate."""
    from zoomy_core.numerics.numerical_system_model import (
        numerics_fluctuations_are_zero)

    class _NoSlot:
        class functions:
            pass

    assert numerics_fluctuations_are_zero(_NoSlot()) is False   # absent ⇒ False
    assert operator_is_zero(None) is True                       # absent ⇒ True


# ── the SWE decision table ────────────────────────────────────────────────

def test_swe_diffusion_is_zero_but_shaped():
    """The measured case: the slot is PRESENT and correctly shaped, so a
    shape-only gate says "has diffusion".  Every entry is zero."""
    nsm = _nsm(_swe())
    dm = nsm.diffusion_matrix
    assert dm is not None and tuple(dm.shape) == (4, 4, 2, 2)
    assert nsm.diffusion_matrix_is_zero is True
    assert nsm.diffusion_matrix_explicit_is_zero is True


def test_swe_transport_operators_are_live():
    nsm = _nsm(_swe())
    assert nsm.flux_is_zero is False
    assert nsm.nonconservative_matrix_is_zero is False   # B[hu][b] = g*h
    assert nsm.eigenvalues_are_zero is False


def test_ncp_boolean_is_not_the_fluctuation_flag():
    """Same model, opposite answers — the NCP slot is live while the EMITTED
    fluctuation under Rusanov is zero (it never builds one; the symbolic solvers
    refuse that pair at setup).  Two different questions; conflating them is how
    the predecessor detector broke."""
    nsm = _nsm(_swe(), Rusanov)
    assert nsm.nonconservative_matrix_is_zero is False
    assert nsm.fluctuations_are_zero is True


def test_model_with_real_diffusion_is_not_zero():
    """Malpasset declares ``diffusion_matrix_explicit`` (depth-averaged eddy
    viscosity), so the explicit slot must read False."""
    nsm = _nsm(MalpassetSWE())
    assert nsm.diffusion_matrix_explicit_is_zero is False


def test_synthetic_live_diffusion_matrix():
    """The implicit slot too: plant a live entry and the boolean flips."""
    nsm = _nsm(_swe())
    assert nsm.diffusion_matrix_is_zero is True
    dm = nsm.diffusion_matrix
    entries = list(dm._array) if hasattr(dm, "_array") else None
    assert entries is not None, "SWE diffusion_matrix is not a ZArray"
    entries[0] = sp.Symbol("nu", positive=True)
    dm._array = entries
    assert nsm.diffusion_matrix_is_zero is False


# ── the mapping + the headers ─────────────────────────────────────────────

def test_mapping_matches_the_individual_properties():
    """One dict so a printer loops; the individual properties are the same
    values by name."""
    nsm = _nsm(_swe())
    flags = nsm.operators_are_zero
    assert set(flags) == set(OPERATOR_ZERO_FLAGS.values())
    assert flags == operator_zero_flags(nsm)
    for name, value in flags.items():
        assert getattr(nsm, name) is value


def test_flags_are_reread_not_cached():
    """They must track ``apply()``, like ``fluctuations_are_zero``."""
    def _zero_flux(sm):
        F = sm.flux
        sm.flux = type(F)([0] * len(list(sp.flatten(F)))).reshape(*F.shape)
    _zero_flux.name = "zero_flux"

    nsm = _nsm(_swe())
    assert nsm.flux_is_zero is False
    nsm.apply(_zero_flux)
    assert nsm.flux_is_zero is True


def test_flags_are_emitted_into_both_headers():
    """A backend can only use them if they reach the code."""
    nsm = _nsm(_swe())
    model_h = CppModel(nsm).create_code()
    numerics_h = CppNumerics(nsm.build_numerics()).create_code()
    for name, value in nsm.operators_are_zero.items():
        line = (f"static constexpr bool {name} = "
                f"{'true' if value else 'false'};")
        assert line in model_h
        assert line in numerics_h
    assert "static constexpr bool diffusion_matrix_is_zero = true;" in model_h
    assert "static constexpr bool flux_is_zero = false;" in model_h
