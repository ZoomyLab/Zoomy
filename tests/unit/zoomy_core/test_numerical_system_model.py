"""NumericalSystemModel: construction, defaults, LSQ-degree resolution.

NSM is the numerical sibling of SystemModel.  This module pins:

- ``from_system_model`` accepts both a SystemModel and a Model.
- Default Riemann class is ``NonconservativeRusanov``.
- ``resolved_lsq_degree`` reads ``sm.aux_registry`` (richer than
  ``model.derivative_specs``) and picks the max spatial-derivative
  order.  Explicit ``lsq_degree`` overrides.
- ``build_numerics()`` returns an instance of the configured Riemann
  class wrapping ``sm``.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.riemann_solvers import (
    NonconservativeRusanov,
    PositiveNonconservativeRusanov,
)
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import (
    DerivativeSpec, StructuredDerivativeModel,
)
from zoomy_core.model.models.sme_model import SMEModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.numerics import (
    NumericalSystemModel,
    ReconstructionSpec,
    RegularizationSpec,
)


# ── Tiny stub model with a declared 2nd-order x derivative ──────────

class _SecondOrderHModel(StructuredDerivativeModel):
    dimension = 1
    variables = ["h"]
    parameters = {"g": (9.81, "positive")}

    def requested_derivatives(self):
        return [DerivativeSpec(field="h", axes=("x", "x"))]

    def flux(self):
        return ZArray(sp.Matrix([[self.Q.h]]))

    def source(self):
        return ZArray.zeros(self.n_variables)


# ── from_system_model ─────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_from_system_model_accepts_systemmodel():
    sm = SystemModel.from_model(SMEModel(level=0))
    nsm = NumericalSystemModel.from_system_model(sm)
    assert nsm.sm is sm
    assert nsm.riemann is NonconservativeRusanov


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_from_system_model_accepts_model():
    m = SMEModel(level=0)
    nsm = NumericalSystemModel.from_system_model(m)
    # The Model has been promoted to a SystemModel.
    assert isinstance(nsm.sm, SystemModel)
    # Same operator surface as the source model would expose.
    assert nsm.sm.n_equations == m.n_variables


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_default_reconstruction_is_first_order_constant():
    nsm = NumericalSystemModel.from_system_model(SMEModel(level=0))
    assert isinstance(nsm.reconstruction, ReconstructionSpec)
    assert nsm.reconstruction.order == 1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_default_regularization_eigenvalue_eps_nonzero():
    nsm = NumericalSystemModel.from_system_model(SMEModel(level=0))
    assert isinstance(nsm.regularization, RegularizationSpec)
    assert nsm.regularization.eigenvalue_eps > 0


# ── LSQ-degree resolution ────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_default_lsq_degree_for_zero_order_model_is_one():
    nsm = NumericalSystemModel.from_system_model(SMEModel(level=0))
    assert nsm.resolved_lsq_degree() == 1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_lsq_degree_lifts_with_declared_second_order_derivative():
    """A StructuredDerivativeModel with ``D.dxx(h)`` declared should
    push the NSM's resolved LSQ degree to ≥ 2.  Source path: either
    ``sm.aux_registry`` (preferred) or ``sm.derivative_specs``
    fallback."""
    m = _SecondOrderHModel()
    nsm = NumericalSystemModel.from_system_model(m)
    assert nsm.resolved_lsq_degree() >= 2


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_additional_systems_lifts_lsq_degree():
    """Composite-solver case: a degree-1 predictor SystemModel +
    a degree-2 pressure sub-system => resolved degree 2.  This is
    the ChorinSplitVAMSolver path that was silently broken before
    additional_systems landed."""
    pred = SystemModel.from_model(SMEModel(level=0))  # only first derivs
    press = _SecondOrderHModel()  # forces degree 2 via aux_registry
    nsm = NumericalSystemModel.from_system_model(
        pred, additional_systems=[press])
    assert nsm.resolved_lsq_degree() >= 2


# ── Riemann build ─────────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_build_numerics_returns_configured_riemann():
    nsm = NumericalSystemModel.from_system_model(
        SMEModel(level=0), riemann=PositiveNonconservativeRusanov)
    numerics = nsm.build_numerics()
    assert isinstance(numerics, PositiveNonconservativeRusanov)
    # Numerics auto-promotes the input to a SystemModel, but in this
    # case the NSM already holds one — so the Numerics wraps the same
    # SystemModel that the NSM points at.
    assert numerics.model is nsm.sm
