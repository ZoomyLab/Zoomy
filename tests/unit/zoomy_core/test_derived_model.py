"""Tests for the DerivedModel architecture.

Covers: DerivedModel base, SMEModel, chaining, mesh auto-promotion,
kernel compilation, and NumpyRuntimeModel integration.
"""

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh, FVMMesh, LSQMesh, ensure_lsq_mesh
from zoomy_core.kernel import Kernel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic
from zoomy_core.model.models.derived_model import DerivedModel
from zoomy_core.model.models.sme_model import SMEModel, SMEInviscid, INSModel
from zoomy_core.model.models.ins_generator import (
    StateSpace, Newtonian, HydrostaticPressure, DepthIntegrate,
)


# ── DerivedModel basics ───────────────────────────────────────────────────────

class _IntermediateModel(INSModel):
    """Intermediate: applies hydrostatic + depth integrate, but not Newtonian."""
    def derive_model(self):
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))


class _ChainedModel(_IntermediateModel):
    """Final: adds Newtonian to intermediate."""
    projectable = True
    def derive_model(self):
        super().derive_model()
        self.apply(Newtonian(self.state))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_intermediate_model_not_solver_ready():
    m = _IntermediateModel()
    assert m.projectable is False
    assert m.system is not None
    assert m.n_variables == 0


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_intermediate_describe():
    m = _IntermediateModel()
    desc = str(m.describe())
    assert "IntermediateModel" in desc
    assert "continuity" in desc or "x_momentum" in desc


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_chained_model_is_solver_ready():
    m = _ChainedModel(level=0)
    assert m.projectable is True
    assert m.n_variables == 3
    assert hasattr(m.functions, "flux")


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_chained_model_compiles():
    m = _ChainedModel(level=1)
    rt = NumpyRuntimeModel(m)
    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    assert F.shape == (4, 1)
    assert np.isfinite(F).all()


# ── SMEModel ──────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
@pytest.mark.parametrize("level", [0, 1, 2])
def test_sme_model_levels(level):
    m = SMEModel(level=level)
    expected_vars = 3 + level  # b, h, + (level+1) moments
    assert m.n_variables == expected_vars
    assert m.system.name == "SMEModel"
    flux = m.flux()
    assert flux.shape == (expected_vars, 1)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_inviscid():
    m = SMEInviscid(level=1)
    assert m.n_variables == 4
    assert "inviscid" in m.system.name.lower() or "Inviscid" in str(m.system.assumptions)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_compile_and_evaluate():
    m = SMEModel(level=1)
    rt = NumpyRuntimeModel(m)

    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    S = rt.source(Q, np.array([]), p)
    assert np.isfinite(F).all()
    assert np.isfinite(S).all()
    # flux[1,0] = hu0 = q2 = 0.1
    assert abs(F[1, 0] - 0.1) < 1e-10


# ── Mesh auto-promotion ──────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ensure_lsq_mesh_from_basemesh():
    bm = BaseMesh.create_1d((-1, 1), 20)
    lm = ensure_lsq_mesh(bm)
    assert isinstance(lm, LSQMesh)
    assert lm._lsq_gradQ is not None


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ensure_lsq_mesh_passthrough():
    lm = LSQMesh.create_1d((-1, 1), 20)
    lm2 = ensure_lsq_mesh(lm)
    assert lm2 is lm


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_create_2d_mesh():
    bm = BaseMesh.create_2d((0, 10, 0, 5), nx=5, ny=3)
    assert bm.dimension == 2
    assert bm.n_inner_cells == 15
    assert bm.type == "quad"
    assert set(bm.boundary_conditions_sorted_names) == {"left", "right", "bottom", "top"}


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_create_2d_lsq_mesh_derivatives():
    lm = LSQMesh.create_2d((0, 10, 0, 5), nx=10, ny=10, lsq_degree=1)
    # u = x, check that one derivative component ≈ 1 and the other ≈ 0
    u = lm._cell_centers[0, :lm.n_cells]
    derivs = lm.compute_derivatives(u, degree=1)
    # One of the two monomial components should be ~1 (du/dx), other ~0 (du/dy)
    interior = derivs[50:60, :]  # well away from boundary
    col_means = np.abs(interior.mean(axis=0))
    assert max(col_means) > 0.9, f"Expected one derivative ≈ 1, got {col_means}"
    assert min(col_means) < 0.1, f"Expected one derivative ≈ 0, got {col_means}"


# ── Kernel ────────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_kernel_creation_from_model():
    m = SMEModel(level=0)
    k = Kernel(m)
    assert hasattr(k.functions, "safe_denominator")
    assert hasattr(k.functions, "clamp_positive")
    assert "eps" in k._param_map


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_kernel_compiles_with_numpy():
    m = SMEModel(level=0)
    k = Kernel(m)
    rt = NumpyRuntimeSymbolic(k)
    result = rt.safe_denominator(np.array([0.0, 1.0]), np.float64(1e-8))
    assert np.allclose(result, [1e-8, 1.0 + 1e-8])


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_model_compiles_with_kernel():
    m = SMEModel(level=1)
    k = Kernel(m)
    rt = NumpyRuntimeModel(m, kernel=k)
    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    assert np.isfinite(F).all()


# ── Full pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_full_pipeline_1d():
    """BaseMesh → auto-promote → Kernel → NumpyRuntimeModel → evaluate."""
    model = SMEModel(level=1)
    mesh = BaseMesh.create_1d((0, 10), 20)
    lsq = ensure_lsq_mesh(mesh, model)
    kernel = Kernel(model)
    rt = NumpyRuntimeModel(model, kernel=kernel)

    Q = np.zeros((model.n_variables, lsq.n_cells))
    Q[1, :] = 0.5  # h
    Q[2, :] = 0.1  # hu0
    p = np.array(model.parameter_values, dtype=float)

    for ic in range(lsq.n_inner_cells):
        F = rt.flux(Q[:, ic], np.array([]), p)
        assert np.isfinite(F).all()


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_full_pipeline_2d():
    """2D BaseMesh → auto-promote → Kernel → NumpyRuntimeModel → evaluate."""
    model = SMEModel(level=0)
    mesh = BaseMesh.create_2d((0, 10, 0, 5), nx=5, ny=3)
    lsq = ensure_lsq_mesh(mesh, model)
    kernel = Kernel(model)
    rt = NumpyRuntimeModel(model, kernel=kernel)

    Q = np.zeros((model.n_variables, lsq.n_cells))
    Q[1, :] = 0.5
    Q[2, :] = 0.05
    p = np.array(model.parameter_values, dtype=float)

    for ic in range(min(5, lsq.n_inner_cells)):
        F = rt.flux(Q[:, ic], np.array([]), p)
        assert np.isfinite(F).all()
        assert F.shape == (3, 1)
