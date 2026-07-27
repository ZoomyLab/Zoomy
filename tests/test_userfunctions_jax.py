"""REQ-168 acceptance (jax): the UserFunctions table supplies EVERY opaque
kernel — a missing one must be a RED TEST, not a silent hole.

Covers the three module dicts that exist in this backend:
  * ``jax_runtime._jax_module_base`` — the LIVE table (every solver builds it
    via ``JaxRuntime.from_nsm``),
  * ``to_jax.JaxRuntimeModel`` / ``JaxRuntimeSymbolic`` — legacy, kept working.
All three are now built from the single source ``zoomy_jax.fvm.userfunctions``.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from zoomy_jax.fvm.userfunctions import (
    jax_userfunctions, eigensystem, eigenvalues, solve, pick)
from zoomy_jax.transformation.jax_runtime import _jax_module_base
from zoomy_jax.transformation.to_jax import JaxRuntimeModel, JaxRuntimeSymbolic


def _required_kernels():
    """Core's registry once REQ-168 item 1 lands; else the known contract."""
    try:                                        # pragma: no cover - core-dependent
        from zoomy_core.model import kernel_functions as kf
        names = getattr(kf, "REQUIRED_KERNELS", None)
        if names:
            return set(names)
    except Exception:
        pass
    return {"compute_derivative", "eigensystem", "solve"}


@pytest.mark.jax
@pytest.mark.parametrize("table,name", [
    (jax_userfunctions(), "userfunctions.jax_userfunctions"),
    (_jax_module_base(), "jax_runtime._jax_module_base (LIVE)"),
    (JaxRuntimeModel.module, "to_jax.JaxRuntimeModel (legacy)"),
    (JaxRuntimeSymbolic.module, "to_jax.JaxRuntimeSymbolic (legacy)"),
])
def test_userfunctions_table_supplies_every_kernel(table, name):
    missing = _required_kernels() - set(table)
    assert not missing, f"{name}: UserFunctions table missing {sorted(missing)}"
    # compute_derivative is legitimately None (solver-injected, mesh-bound);
    # the numerical kernels must be real callables.
    for k in ("eigensystem", "solve"):
        if k in _required_kernels():
            assert callable(table[k]), f"{name}: '{k}' present but not callable"


@pytest.mark.jax
def test_eigensystem_matches_numpy_scalar_and_batched():
    """eigensystem(*A_flat) -> the FULL packed [lambda(n), R(n*n), L(n*n)];
    component access is pick(eigensystem(*a), idx)."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    stack = np.asarray(eigensystem(*[float(x) for x in A.reshape(-1)]))
    assert stack.shape == (3 + 2 * 3 * 3,)
    lam = stack[:3]
    assert np.allclose(np.sort(lam), np.sort(np.real(np.linalg.eigvals(A))), atol=1e-10)
    # pick reads a component of the packed stack (== trailing-axis index)
    assert np.allclose(float(pick(eigensystem(*[float(x) for x in A.reshape(-1)]), 0)),
                       stack[0])

    Ab = rng.standard_normal((4, 2, 2))                      # batched over faces
    flat = [jnp.asarray(Ab[:, i, j]) for i in range(2) for j in range(2)]
    lam_b = np.asarray(eigensystem(*flat))[..., :2]          # (4, 2) eigenvalues
    ref = np.sort(np.real(np.linalg.eigvals(Ab)), axis=-1)
    assert np.allclose(np.sort(lam_b, axis=-1), ref, atol=1e-10)


@pytest.mark.jax
def test_solve_matches_numpy_batched():
    """solve(*A_flat, *b) -> the whole A^-1 b vector, batched (REQ-68);
    component access is pick(solve(*a), idx)."""
    rng = np.random.default_rng(2)
    nc, n = 5, 3
    A = rng.standard_normal((nc, n, n)) + n * np.eye(n)
    b = rng.standard_normal((nc, n))
    args = [jnp.asarray(A[:, i, j]) for i in range(n) for j in range(n)] + \
           [jnp.asarray(b[:, i]) for i in range(n)]
    x = np.asarray(solve(*args))                             # (nc, n)
    assert np.allclose(x, np.linalg.solve(A, b[..., None])[..., 0], atol=1e-10)
    assert np.allclose(np.asarray(pick(solve(*args), 1)), x[..., 1], atol=1e-10)


@pytest.mark.jax
def test_eigenvalues_lambda_only_matches_numpy():
    """REQ-168 GAP 1: the λ-only kernel — the whole spectrum vector.

    Must agree with numpy's spectrum, scalar and batched.  Unlike `eigensystem`
    this lowers to `jnp.linalg.eigvals` directly (no pure_callback), so it stays
    inside the jit — the wave speed is evaluated every step inside the fused
    while_loop and a host round-trip there would collapse the fusion.
    """
    rng = np.random.default_rng(3)

    A = rng.standard_normal((3, 3))
    lam = np.asarray(eigenvalues(*[float(x) for x in A.reshape(-1)]))
    assert np.allclose(np.sort(lam), np.sort(np.real(np.linalg.eigvals(A))), atol=1e-10)

    Ab = rng.standard_normal((5, 2, 2))                    # batched over faces
    flat = [jnp.asarray(Ab[:, i, j]) for i in range(2) for j in range(2)]
    lam_b = np.asarray(eigenvalues(*flat))                 # (5, 2)
    ref = np.sort(np.real(np.linalg.eigvals(Ab)), axis=-1)
    assert np.allclose(np.sort(lam_b, axis=-1), ref, atol=1e-10)

    # the CFL bound the wave speed actually wants
    assert np.allclose(np.abs(lam).max(), np.abs(np.real(np.linalg.eigvals(A))).max(),
                       atol=1e-10)
