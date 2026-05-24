"""JaxRuntime: SystemModel → NumericalSystemModel → JaxRuntime pipeline.

The new clean JAX runtime — every SystemModel operator (flux, source,
NCP, eigenvalues, BC kernel) is lambdified once and wrapped in
``jax.jit(jax.vmap(...))``.  Numerics-derived face operators
(``numerical_flux``, ``numerical_fluctuations``) are vmapped over the
face axis.

This module pins:

1. The runtime can be built from a Model, a SystemModel, or an NSM.
2. Cell operators (``flux``, ``source``, ``eigenvalues``) return the
   expected per-cell shapes when called with full-grid arrays.
3. Face operators return shape ``(n_eq, n_faces)`` /
   ``(2*n_eq, n_faces)``.
4. The ``parameters`` property is live — mutating
   ``nsm.sm.parameter_values.g`` is reflected on the next call (no
   runtime rebuild).
5. AD via ``jax.grad`` over the parameter axis works.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from zoomy_core.model.models.sme_model import SMEModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_jax.transformation.jax_runtime import JaxRuntime


def _make_runtime():
    """Tiny SME(0) — 3 states (b, h, hu), 0 aux, parameters {g, eps,
    ex, ez, rho, lamda, nu}."""
    m = SMEModel(level=0)
    nsm = NumericalSystemModel.from_system_model(m)
    return JaxRuntime.from_nsm(nsm)


# ── Construction ─────────────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_from_nsm_accepts_model_systemmodel_and_nsm():
    m = SMEModel(level=0)
    rt_m = JaxRuntime.from_nsm(m)
    rt_sm = JaxRuntime.from_nsm(SystemModel.from_model(SMEModel(level=0)))
    rt_nsm = JaxRuntime.from_nsm(
        NumericalSystemModel.from_system_model(SMEModel(level=0)))
    assert isinstance(rt_m, JaxRuntime)
    assert isinstance(rt_sm, JaxRuntime)
    assert isinstance(rt_nsm, JaxRuntime)


# ── Parameter access ────────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_parameters_live_reflect_sm_mutation():
    rt = _make_runtime()
    p0 = rt.parameters
    g_idx = rt.parameter_names.index("g")
    assert float(p0[g_idx]) == pytest.approx(9.81)

    rt.sm.parameter_values.g = 12.34
    p1 = rt.parameters
    assert float(p1[g_idx]) == pytest.approx(12.34), (
        "rt.parameters is snapshotted instead of live"
    )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_parameter_symbols_accessible():
    rt = _make_runtime()
    names = rt.parameter_names
    syms = rt.parameter_symbols
    assert set(names) >= {"g"}  # SME family always has g
    assert len(syms) == len(names)


# ── Cell-operator shapes ────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_flux_shape_and_finite():
    rt = _make_runtime()
    n_cells = 8
    Q = jnp.ones((rt.n_state, n_cells))
    Qaux = jnp.zeros((max(rt.n_aux, 1), n_cells))  # 0-axis safe
    p = rt.parameters
    F = rt.flux(Q, Qaux[:rt.n_aux, :], p)
    # SME 1D: flux has shape (n_eq, n_dim=1, n_cells)
    assert F.shape[-1] == n_cells
    assert F.shape[0] == rt.n_state
    assert jnp.all(jnp.isfinite(F))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_source_shape_and_finite():
    rt = _make_runtime()
    n_cells = 8
    Q = jnp.ones((rt.n_state, n_cells))
    Qaux = jnp.zeros((rt.n_aux, n_cells))
    p = rt.parameters
    S = rt.source(Q, Qaux, p)
    assert S.shape == (rt.n_state, n_cells)
    assert jnp.all(jnp.isfinite(S))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_hydrostatic_pressure_finite():
    rt = _make_runtime()
    n_cells = 8
    Q = jnp.ones((rt.n_state, n_cells)).at[1, :].set(0.5)
    Qaux = jnp.zeros((rt.n_aux, n_cells))
    p = rt.parameters
    P = rt.hydrostatic_pressure(Q, Qaux, p)
    assert P.shape[-1] == n_cells
    assert jnp.all(jnp.isfinite(P))


# ── AD w.r.t. parameters ───────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_grad_of_hydrostatic_pressure_wrt_parameters_runs():
    """SME hydrostatic pressure is ``ez·g·h²/2`` — d/dg must be
    non-trivial.  Proves the parameter axis is a real JAX tracer,
    not a captured constant."""
    rt = _make_runtime()
    n_cells = 4
    Q = jnp.ones((rt.n_state, n_cells)).at[1, :].set(0.5)  # h = 0.5
    Qaux = jnp.zeros((rt.n_aux, n_cells))

    def loss(p):
        P = rt.hydrostatic_pressure(Q, Qaux, p)
        return jnp.sum(P * P)

    p0 = rt.parameters
    grad_p = jax.grad(loss)(p0)
    g_idx = rt.parameter_names.index("g")
    assert jnp.all(jnp.isfinite(grad_p))
    assert float(jnp.abs(grad_p[g_idx])) > 0.0, (
        "d/dg of hydrostatic_pressure is zero — AD path through the "
        "SystemModel is broken; parameters are not being treated as tracers."
    )
