"""Model → SystemModel refuses to hand back a system with UNCLOSED terms.

A closure that nothing binds is swept into ``aux_state`` with no
``aux_registry`` row, so nothing ever writes it and the term reads ZERO —
absent physics that no error reports.  ML-VAM is the standing case: the
by-parts step DOES derive the internal interface tractions, but
``apply_layer_stress_closures`` binds them only on the top / bed layer, so
every internal interface was silently frictionless.

Pinned here: the open model raises and can be described; the closed one builds;
and an aux the model DECLARED with a definition it cannot evaluate pointwise
(the stay-3D column integrals) is NOT mistaken for a forgotten closure.
"""
import warnings

import pytest

from zoomy_core.systemmodel.system_model import (
    SystemModel, UnclosedClosureError, allow_unclosed)

pytestmark = [pytest.mark.systemmodel, pytest.mark.small]

_SIGMAS = ["\\hat{\\sigma}_1_1", "\\hat{\\sigma}_1_2",
           "\\hat{\\sigma}_2_1", "\\hat{\\sigma}_2_2"]


def _mlvam(closures):
    from zoomy_core.model.models.ml_vam import MLVAM
    return MLVAM(n_layers=2, level=1, dimension=2, closures=closures)


def test_mlvam_without_stress_closure_raises_naming_the_interfaces():
    with pytest.raises(UnclosedClosureError) as exc:
        SystemModel.from_model(_mlvam([]))
    err = exc.value
    assert sorted(err.unclosed) == _SIGMAS
    assert err.model_name == "MLVAM"
    # The message has to be actionable on sight: it names the open terms and
    # says what they do (read zero), not just that something is wrong.
    assert "read ZERO" in str(err)
    for s in _SIGMAS:
        assert s in str(err)
    # …and the open system travels WITH the error, so it can be inspected
    # without rebuilding it.
    assert err.system_model is not None
    assert err.describe() == err.system_model.describe_unclosed()
    assert "UNCLOSED TERMS: 4" in err.describe()


def test_mlvam_with_stress_closures_follows_through():
    from zoomy_core.model.models.closures import (
        Newtonian, NavierSlip, StressFree)
    sm = SystemModel.from_model(
        _mlvam([Newtonian(), NavierSlip(), StressFree()]))
    assert sm._unclosed_closures == []
    assert "The system is CLOSED." in sm.describe_unclosed()


def test_allow_unclosed_returns_the_open_system_with_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with allow_unclosed():
            sm = SystemModel.from_model(_mlvam([]))
    assert any("unclosed term" in str(w.message) for w in caught)
    assert sorted(sm._unclosed_closures) == _SIGMAS
    body = sm.describe_unclosed()
    assert "UNCLOSED TERMS: 4" in body
    for s in _SIGMAS:
        assert s in body
    # The opt-in is scoped — it must not leak past the block.
    with pytest.raises(UnclosedClosureError):
        SystemModel.from_model(_mlvam([]))


def test_declared_column_integral_aux_is_not_reported_unclosed():
    """``U`` / ``ω`` are declared with a definition; only ``tau_xx`` is open.

    A running ζ-quadrature can never be an ``update_aux_variables`` row, so
    these have no registry entry by construction.  Reading only the registry
    therefore flags them — and that false positive is what makes a guard get
    switched off.  The model's own ``add_equation(..., group="aux")`` is the
    declaration; it lives on ``model.derivation``, not on the outer model.
    """
    from zoomy_core.model.models.sigma3d import Sigma3D
    with pytest.raises(UnclosedClosureError) as exc:
        SystemModel.from_model(Sigma3D())
    assert exc.value.unclosed == ["\\tilde{tau_xx}"]
    sm = exc.value.system_model
    assert set(sm.external_aux) == {"U", "omega"}
    assert "backend-supplied" in sm.describe_unclosed()
