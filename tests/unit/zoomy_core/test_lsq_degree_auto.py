"""Auto-LSQ-degree contract: ensure_lsq_mesh reads model.derivative_specs.

When a model declares spatial derivatives via ``requested_derivatives``,
``ensure_lsq_mesh`` must build the LSQ stencil with a polynomial degree
high enough to evaluate them.  Without a model (or with an empty spec
list) it falls back to ``lsq_degree=1``.
"""
from __future__ import annotations

import pytest
import sympy as sp

from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import (
    DerivativeSpec, StructuredDerivativeModel,
)


class _SecondOrderHModel(StructuredDerivativeModel):
    """1D model that needs ``d^2 h / dx^2`` — forces lsq_degree >= 2."""
    dimension = 1
    variables = ["h"]
    parameters = {"g": (9.81, "positive")}

    def requested_derivatives(self):
        return [DerivativeSpec(field="h", axes=("x", "x"))]

    def flux(self):
        return ZArray(sp.Matrix([[self.Q.h]]))

    def source(self):
        return ZArray.zeros(self.n_variables)


class _FirstOrderHModel(StructuredDerivativeModel):
    """1D model that needs only ``d h / dx`` — lsq_degree == 1."""
    dimension = 1
    variables = ["h"]
    parameters = {"g": (9.81, "positive")}

    def requested_derivatives(self):
        return [DerivativeSpec(field="h", axes=("x",))]

    def flux(self):
        return ZArray(sp.Matrix([[self.Q.h]]))

    def source(self):
        return ZArray.zeros(self.n_variables)


def _max_total_degree(multi_index):
    return max(sum(mi) for mi in multi_index)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_no_model_defaults_to_degree_one():
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    lsq = ensure_lsq_mesh(mesh)
    assert _max_total_degree(lsq.lsq_monomial_multi_index) == 1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_first_order_model_keeps_degree_one():
    m = _FirstOrderHModel()
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    lsq = ensure_lsq_mesh(mesh, m)
    assert _max_total_degree(lsq.lsq_monomial_multi_index) == 1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_second_order_model_lifts_degree_to_two():
    m = _SecondOrderHModel()
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    lsq = ensure_lsq_mesh(mesh, m)
    assert _max_total_degree(lsq.lsq_monomial_multi_index) == 2
    # And the (2,) monomial is actually present.
    assert (2,) in [tuple(mi) for mi in lsq.lsq_monomial_multi_index]


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_explicit_lsq_degree_kwarg_overrides_model():
    m = _FirstOrderHModel()
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    lsq = ensure_lsq_mesh(mesh, m, lsq_degree=2)
    assert _max_total_degree(lsq.lsq_monomial_multi_index) == 2
