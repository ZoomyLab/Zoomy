"""REQ-84 regression: constant-entry rank normalization is a shared core
lowering seam, so JaxRuntime lowers MalpassetSWE without the mixed-rank stack
error.

MalpassetSWE's state is ``[b, h, hu, hv]``; its ``flux`` b-row is *identically
zero* (bathymetry does not flux).  Before REQ-84, ``_lambdify_array`` fed that
constant ``0`` to ``jnp.array`` alongside state-dependent rows, which could not
stack ((n,) scalar row vs (n,) array rows).  ``zoomy_core.transformation.
vectorize.uniform_rank`` (the same helper the numpy printer uses) now wraps the
constant b-row as ``zeros_like(anchor)`` so every row shares the batch rank.

Run (needs jax)::

    JAX_PLATFORMS=cpu pytest test_uniform_rank_malpasset_req84.py -v
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_jax.transformation.jax_runtime import JaxRuntime


def _runtime():
    nsm = NumericalSystemModel.from_system_model(
        MalpassetSWE(), riemann=PositiveNonconservativeHLL)
    return JaxRuntime(nsm)


def _batched_state(rt, n):
    """(n_state, n) batch: b=0, h=1, hu=0.2, hv=0.1 — hinv aux = 1/h."""
    names = [str(s) for s in rt.sm.state]
    Q = np.zeros((rt.n_state, n))
    Q[names.index("h")] = 1.0
    Q[names.index("hu")] = 0.2
    Q[names.index("hv")] = 0.1
    Qaux = np.ones((rt.n_aux, n))  # hinv = 1/h = 1
    return jnp.asarray(Q), jnp.asarray(Qaux)


def test_flux_lowers_and_evaluates_batched():
    """The flux kernel evaluates on a batch without a mixed-rank stack error,
    and the identically-zero b-row stays zero."""
    rt = _runtime()
    n = 5
    Q, Qaux = _batched_state(rt, n)
    F = np.asarray(rt.flux(Q, Qaux, rt.parameters))  # (n_eq, n_dim, n_cells)
    assert np.all(np.isfinite(F)), "non-finite flux"
    names = [str(s) for s in rt.sm.state]
    b_row, h_row = names.index("b"), names.index("h")
    # Constant (state-free) b-row wrapped as zeros_like → stays exactly zero.
    assert np.allclose(F[b_row], 0.0), "bathymetry flux row must be zero"
    # Active mass row untouched by the wrap: F_h = [hu, hv] = [0.2, 0.1].
    assert np.allclose(F[h_row, 0], 0.2) and np.allclose(F[h_row, 1], 0.1), \
        "active mass-flux row perturbed by rank normalization"


def test_fluctuation_kernel_lowers_and_evaluates_batched():
    """The Riemann fluctuation kernel (NCP bed-slope) evaluates per-face on a
    batch without the mixed-rank stack error."""
    rt = _runtime()
    assert rt.numerical_fluctuations is not None, "face operators not built"
    nf = 4
    QL, QauxL = _batched_state(rt, nf)
    QR, QauxR = _batched_state(rt, nf)
    normal = jnp.asarray(np.tile(np.array([[1.0], [0.0]]), (1, nf)))
    Dp_Dm = np.asarray(
        rt.numerical_fluctuations(QL, QR, QauxL, QauxR, rt.parameters, normal))
    assert np.all(np.isfinite(Dp_Dm)), "non-finite fluctuations"
    # Equal L/R states over flat bed → zero fluctuations.
    assert np.allclose(Dp_Dm, 0.0), "identical states must give zero fluctuations"
