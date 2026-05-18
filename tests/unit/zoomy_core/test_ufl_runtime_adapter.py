"""Smoke tests for the UFL runtime adapter (``UFLRuntimeSymbolic``)
and the Riemann ``to_runtime_ufl()`` lowering path.

These tests run whenever ``ufl`` is importable — they do NOT need
Firedrake.  They guard the migration that wires the symbolic
Riemann solvers into the Firedrake backend.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ufl")

from zoomy_core.fvm.riemann_solvers import HLL, HLLC, Rusanov
from zoomy_core.model.models.shallow_water import ShallowWater2D
from zoomy_core.model.models.system_model import SystemModel


def _sm_swe2d():
    model = ShallowWater2D(manning_n=0.03, nu=0.01)
    return SystemModel.from_model(model)


@pytest.mark.parametrize("riemann_cls", [Rusanov, HLL, HLLC])
def test_riemann_lowers_to_ufl(riemann_cls):
    """Every supported Riemann variant must lower to a UFL runtime
    with callable ``numerical_flux`` / ``local_max_abs_eigenvalue``."""
    sm = _sm_swe2d()
    numerics = riemann_cls(sm)
    rt = numerics.to_runtime_ufl()
    assert callable(rt.numerical_flux)
    assert callable(rt.local_max_abs_eigenvalue)


def test_systemmodel_diffusion_matrix_shape():
    """SystemModel.diffusion_matrix is rank-4 ``(n_eq, n_state, n_dim, n_dim)``."""
    sm = _sm_swe2d()
    A = sm.diffusion_matrix
    assert A is not None
    assert tuple(A.shape) == (3, 3, 2, 2)
