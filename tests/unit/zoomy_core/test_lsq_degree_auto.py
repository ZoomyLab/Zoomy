"""Auto-LSQ-degree contract: ``ensure_lsq_mesh`` reads everything from
the model's NumericalSystemModel.  There is no longer a user-facing
``lsq_degree`` knob.

When a model declares spatial derivatives via ``requested_derivatives``,
``ensure_lsq_mesh`` builds the LSQ stencil with a polynomial degree
high enough to evaluate them.  When no derivative info is available
the stencil falls back to degree 1.  Passing no model raises.
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
def test_no_model_raises():
    mesh = FVMMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    with pytest.raises(TypeError, match="requires a model"):
        ensure_lsq_mesh(mesh, None)


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
def test_lsq_stencil_rebuilds_when_existing_degree_too_low():
    """An already-built LSQMesh with degree 1 must be rebuilt to
    degree 2 when ensure_lsq_mesh is called with a model that
    requests d^2/dx^2."""
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    lsq = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    assert _max_total_degree(lsq.lsq_monomial_multi_index) == 1
    m = _SecondOrderHModel()
    out = ensure_lsq_mesh(lsq, m)
    assert out is lsq  # rebuild is in-place
    assert _max_total_degree(out.lsq_monomial_multi_index) == 2
