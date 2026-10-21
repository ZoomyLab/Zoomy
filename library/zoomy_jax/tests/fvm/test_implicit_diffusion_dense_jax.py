r"""Dense, state-dependent implicit diffusion — JAX mirror (REQ-109).

JAX analogue of ``zoomy_core/tests/fvm/test_implicit_diffusion_dense.py``,
locking :class:`DenseDiffusionOperatorJAX` (``solver_imex_jax`` dense path)
against the same three properties the numpy reference guarantees:

1. **Scalar-ν regression** — a constant diagonal ``A = ν·I·δ_de`` reduces the
   dense two-point divergence to the scalar ``ν·L`` of :class:`DiffusionOperatorJAX`
   bit-for-bit (unit normals ⇒ ``T = ν·δ_ij``), and the Crank–Nicolson update
   matches the scalar operator.
2. **Dense consumption** — a non-zero off-diagonal ``A[0,1]`` cross-couples
   variable 1 into equation 0's diffusion, matching an independent
   hand-computed two-point ``∇·(A:∇Q)``.
3. **State-dependent Newton** — ``A(Q)`` drives the ``jax.jvp``-Jacobian Newton
   loop; the CN residual converges to ~0.

The operator's ``_divergence`` consumes the core per-cell layout
``(n_eq, n_st, nc, d, d)``; ``implicit_solve``/``residual_norm`` take an
``A_fn`` returning the RUNTIME layout ``(n_eq, n_st, d, d, nc)`` (cell axis
last, as the jax runtime emits), which ``_as_cell_tensor`` transposes.
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


def _const_cells(A_slab, nc):
    """Broadcast a constant ``(n_eq, n_st, dim, dim)`` tensor to the per-cell
    ``(n_eq, n_st, nc, dim, dim)`` layout ``_divergence`` consumes."""
    return jnp.asarray(np.broadcast_to(
        A_slab[:, :, None, :, :],
        A_slab.shape[:2] + (nc,) + A_slab.shape[2:]))


def _runtime_layout(A_cells):
    """(n_eq, n_st, nc, d, d) -> (n_eq, n_st, d, d, nc) = what A_fn returns."""
    return jnp.transpose(jnp.asarray(A_cells), (0, 1, 3, 4, 2))


def _independent_divergence(op, Q, A_cells):
    """Reference ``∇·(A:∇Q)`` via an explicit per-face loop — an independent
    re-derivation of ``DenseDiffusionOperatorJAX._divergence`` (interior only)."""
    A_cells = np.asarray(A_cells)
    Q = np.asarray(Q)
    n_eq = A_cells.shape[0]
    ia = np.asarray(op._ia); ib = np.asarray(op._ib)
    gA = np.asarray(op._gA); gB = np.asarray(op._gB)
    n_int = np.asarray(op._n_int)
    D = np.zeros((n_eq, op.nc))
    for f in range(ia.size):
        a, b = int(ia[f]), int(ib[f])
        n = n_int[:, f]
        Aface = 0.5 * (A_cells[:, :, a] + A_cells[:, :, b])
        T = np.einsum('ijde,d,e->ij', Aface, n, n)
        flux = T @ (Q[:, b] - Q[:, a])
        D[:, a] += flux * gA[f]
        D[:, b] -= flux * gB[f]
    return D


def test_scalar_nu_regression(mesh):
    """Constant diagonal ``A = ν·I`` ⇒ dense operator + CN update match the
    scalar :class:`DiffusionOperatorJAX` (interior ``ν·L``) bit-for-bit."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    nu = 0.73
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(nc))

    scal = DiffusionOperatorJAX(mesh, dim, nu=nu)
    dense = DenseDiffusionOperatorJAX(mesh, dim, n_vars=1, state_dependent=False)
    A_slab = np.zeros((1, 1, dim, dim))
    for d in range(dim):
        A_slab[0, 0, d, d] = nu
    A_cells = _const_cells(A_slab, nc)

    # Interior divergence == scalar L @ u.
    got = dense._divergence(u[None], A_cells)[0]
    assert np.allclose(np.asarray(got), np.asarray(scal.L @ u), atol=1e-12)

    # Crank–Nicolson implicit update matches the scalar solve on inner cells.
    dt = 0.04
    ref = scal.implicit_solve(u, dt)[:nc]
    out = dense.implicit_solve(u[None], dt, lambda Qs: _runtime_layout(A_cells),
                               bf_grads=None, tol=1e-12, maxiter=800)[0]
    assert np.allclose(np.asarray(out), np.asarray(ref), atol=1e-9)


def test_dense_off_diagonal_cross_couples(mesh):
    """A non-zero ``A[0,1]`` couples variable 1 into equation 0's diffusion,
    changing the result and matching an independent hand divergence."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    rng = np.random.default_rng(1)
    Q = jnp.asarray(rng.standard_normal((2, nc)))
    op = DenseDiffusionOperatorJAX(mesh, dim, n_vars=2)

    Adiag = np.zeros((2, 2, dim, dim))
    Aoff = None
    for d in range(dim):
        Adiag[0, 0, d, d] = 1.0
        Adiag[1, 1, d, d] = 1.0
    Aoff = Adiag.copy()
    for d in range(dim):
        Aoff[0, 1, d, d] = 0.5           # eq 0 diffuses along ∇(variable 1)

    Adiag_c = _const_cells(Adiag, nc)
    Aoff_c = _const_cells(Aoff, nc)
    D_diag = np.asarray(op._divergence(Q, Adiag_c))
    D_off = np.asarray(op._divergence(Q, Aoff_c))

    # Off-diagonal genuinely changes the update (cross-coupling is live).
    assert not np.allclose(D_diag, D_off)
    # Matches an independent two-point ∇·(A:∇Q).
    assert np.allclose(D_off, _independent_divergence(op, Q, Aoff_c), atol=1e-12)
    # The extra term in equation 0 is exactly 0.5 · Laplacian(variable 1).
    A01 = np.zeros((2, 2, dim, dim))
    for d in range(dim):
        A01[0, 1, d, d] = 1.0
    lap_Q1 = _independent_divergence(op, Q, _const_cells(A01, nc))
    assert np.allclose(D_off[0] - D_diag[0], 0.5 * lap_Q1[0], atol=1e-12)


def test_state_dependent_newton_residual(mesh):
    """A state-dependent ``A(Q)`` drives the jax.jvp Newton loop; the CN
    residual ``Q^{n+1}−Q* − dt/2(𝒟(Q^{n+1})+𝒟(Q*))`` converges to ~0."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    rng = np.random.default_rng(2)
    Q_star = jnp.asarray(0.4 * rng.standard_normal((2, nc)))
    op = DenseDiffusionOperatorJAX(mesh, dim, n_vars=2, state_dependent=True)

    def A_fn(Qs):
        # Runtime layout (n_eq, n_st, d, d, nc), nonlinear + dense in Q.
        A = jnp.zeros((2, 2, dim, dim, Qs.shape[1]))
        for d in range(dim):
            A = A.at[0, 0, d, d, :].set(1.0 + 0.3 * Qs[0] ** 2)
            A = A.at[1, 1, d, d, :].set(0.8)
            A = A.at[0, 1, d, d, :].set(0.2 * jnp.tanh(Qs[1]))
        return A

    dt = 0.02
    Q_np1 = op.implicit_solve(Q_star, dt, A_fn, bf_grads=None,
                              tol=1e-11, maxiter=500,
                              newton_maxiter=20, newton_tol=1e-11)
    assert op.residual_norm(Q_np1, Q_star, dt, A_fn) < 1e-8
    # The solve actually moved the state (diffusion acted).
    assert not np.allclose(np.asarray(Q_np1), np.asarray(Q_star))
