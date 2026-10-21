r"""Explicit diffusion in the JAX flux operator (REQ-50).

``HyperbolicSolver`` (and the Chorin predictor) now add
``dQ += ∇·(A:∇Q)`` from the model's rank-4 ``diffusion_matrix_explicit``
(else ``diffusion_matrix``), assembled by the SAME dense TPFA divergence the
IMEX dense path uses (:class:`DenseDiffusionOperatorJAX`, REQ-109).  Unlike the
numpy reference's DIAGONAL-only explicit path, this carries cross-variable /
state-dependent tensors.

These tests lock the exact quantity the flux operator adds — the operator's
``_divergence`` — for REQ-50's scenarios:

1. **Constant D on a momentum row** — ``A[m,m,d,d]=D`` (one row) makes
   ``∇·(A:∇Q)`` act as ``D`` × the scalar Laplacian on row ``m`` alone
   (bit-for-bit vs the trusted scalar :class:`DiffusionOperatorJAX`), and
   ZERO on every other row.  Since that scalar operator is a consistent
   Laplacian, this is the exact form of "matches ``D·∂_xx u`` on a smooth
   bump".
2. **Lake-at-rest** — a state-dependent ``A ∝ u`` (the malpasset
   ``A[·,·]∝u`` pattern) evaluated at ``u=0`` gives ``A=0`` ⇒ the diffusion
   contribution is identically zero (well-balancing preserved).
3. **Cross-variable** — an off-diagonal ``A[m,k]`` genuinely couples row ``k``
   into row ``m``'s diffusion (the malpasset chain-rule ``A[2,1]=-D·u`` shape).
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_jax.fvm.reconstruction_jax import (
    DiffusionOperatorJAX, DenseDiffusionOperatorJAX,
)


@pytest.fixture
def mesh():
    return FVMMesh.create_2d((0.0, 1.0, 0.0, 1.0), 6, 5)


def _cells(A_slab, nc):
    return jnp.asarray(np.broadcast_to(
        A_slab[:, :, None, :, :],
        A_slab.shape[:2] + (nc,) + A_slab.shape[2:]))


def test_constant_D_on_momentum_row(mesh):
    """``A[m,m,d,d]=D`` on ONE row ⇒ ``∇·(A:∇Q)`` = ``D``×(scalar Laplacian)
    on row ``m`` alone, and zero on the other rows."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    n_vars, m, D = 3, 2, 0.37          # row 2 = a momentum row
    rng = np.random.default_rng(0)
    Q = jnp.asarray(rng.standard_normal((n_vars, nc)))

    op = DenseDiffusionOperatorJAX(mesh, dim, n_vars)
    scal = DiffusionOperatorJAX(mesh, dim, nu=D)

    A_slab = np.zeros((n_vars, n_vars, dim, dim))
    for d in range(dim):
        A_slab[m, m, d, d] = D
    Dvg = np.asarray(op._divergence(Q, _cells(A_slab, nc)))

    # Row m == D·(L @ Q_m) bit-for-bit; every other row untouched.
    assert np.allclose(Dvg[m], np.asarray(scal.L @ Q[m]), atol=1e-12)
    for k in range(n_vars):
        if k != m:
            assert np.allclose(Dvg[k], 0.0, atol=1e-14)


def test_lake_at_rest_zero_when_u_zero(mesh):
    """A state-dependent ``A ∝ u`` evaluated at ``u=0`` ⇒ zero diffusion
    (well-balancing preserved for A∝u tensors)."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    n_vars = 3
    op = DenseDiffusionOperatorJAX(mesh, dim, n_vars)

    # h = 1 everywhere, momentum rows = 0 (u = q/h = 0): the rest state.
    Q = jnp.zeros((n_vars, nc)).at[0].set(1.0)

    # A[m,m,d,d] = D·q_m  (∝ velocity for h≡1) — vanishes at rest.
    def A_at(Qs):
        A = np.zeros((n_vars, n_vars, nc, dim, dim))
        for d in range(dim):
            A[2, 2, :, d, d] = 0.5 * np.asarray(Qs[2])
        return jnp.asarray(A)

    Dvg = np.asarray(op._divergence(Q, A_at(Q)))
    assert np.allclose(Dvg, 0.0, atol=1e-14)


def test_cross_variable_coupling(mesh):
    """An off-diagonal ``A[2,1]`` couples row 1 into row 2's diffusion (the
    malpasset ``A[2,1]=-D·u`` chain-rule shape) — a dense effect the numpy
    reference's diagonal-only explicit path drops."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    n_vars = 3
    rng = np.random.default_rng(1)
    Q = jnp.asarray(rng.standard_normal((n_vars, nc)))
    op = DenseDiffusionOperatorJAX(mesh, dim, n_vars)

    Adiag = np.zeros((n_vars, n_vars, dim, dim))
    for d in range(dim):
        Adiag[2, 2, d, d] = 1.0
    Aoff = Adiag.copy()
    for d in range(dim):
        Aoff[2, 1, d, d] = -0.6        # row 2 diffuses along ∇(variable 1)

    D_diag = np.asarray(op._divergence(Q, _cells(Adiag, nc)))
    D_off = np.asarray(op._divergence(Q, _cells(Aoff, nc)))
    # The off-diagonal genuinely changes row 2, and equals −0.6·Laplacian(Q_1).
    assert not np.allclose(D_diag[2], D_off[2])
    scal = DiffusionOperatorJAX(mesh, dim, nu=1.0)
    assert np.allclose(D_off[2] - D_diag[2],
                       -0.6 * np.asarray(scal.L @ Q[1]), atol=1e-12)
