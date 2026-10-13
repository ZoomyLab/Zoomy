"""Mesh-native spectral coupling via the **graph Laplacian** (connectivity, not plane waves).

Two options:

1. **Polynomial filter** `sum_k c_k L_sym^k u` with symmetric normalized Laplacian
   `L_sym = I - D^{-1/2} A D^{-1/2}`. Forward cost **O(K)** dense matmuls per field
   (or O(K·nnz) if `L_sym` is sparse later). No eigendecomposition at run time.

2. **Low-frequency eigenbasis** (graph Fourier): fixed columns `U` from `eigh(L_sym)`
   (computed **once** per mesh offline), forward `u ↦ U ( w ⊙ U^T u )` — **O(n·K)** per
   field; setup **O(n³)** for dense `eigh`, so intended for modest `n` or batched
   offline bases.

These respect **holes and complex boundaries** as long as the adjacency graph does
(only connect cells that share a mesh edge / graph edge).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def symmetric_normalized_laplacian_dense(adj: np.ndarray) -> np.ndarray:
    """Unweighted symmetric adjacency `A` (0/1, zero diagonal). Returns dense `L_sym`."""
    a = np.asarray(adj, dtype=np.float64)
    n = a.shape[0]
    deg = a.sum(axis=1)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    d_inv_sqrt = np.zeros_like(deg)
    m = deg > 1e-14
    d_inv_sqrt[m] = 1.0 / np.sqrt(deg[m])
    dhs = np.diag(d_inv_sqrt)
    return np.eye(n) - dhs @ a @ dhs


def path_graph_adjacency(n: int) -> np.ndarray:
    """Path graph on `n` vertices (line connectivity in cell order)."""
    adj = np.zeros((n, n), dtype=np.float64)
    if n >= 2:
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    return adj


def graph_first_k_eigenvectors(L_sym: np.ndarray, k: int, skip_constant: bool = True) -> np.ndarray:
    """Return `U` with `k` columns (low eigenmodes). Skip trivial constant if requested."""
    n = L_sym.shape[0]
    if k <= 0 or n == 0:
        return np.zeros((n, 0), dtype=np.float64)
    _w, v = np.linalg.eigh(L_sym)
    if skip_constant and n >= 2:
        v = v[:, 1:]
    if v.shape[1] == 0:
        return np.zeros((n, 0), dtype=np.float64)
    take = min(k, v.shape[1])
    return v[:, :take].astype(np.float64)


def apply_graph_polynomial_laplacian_mix(
    q_mp: jnp.ndarray,
    l_sym: jnp.ndarray,
    coeffs: jnp.ndarray,
    blend_logits: jnp.ndarray,
) -> jnp.ndarray:
    """`coeffs` shape `(n_fields, K+1)` for `sum_{k=0}^K c_k L^k u`."""
    n_fields = q_mp.shape[0]
    kmax = coeffs.shape[1] - 1
    l_sym = l_sym.astype(jnp.float64)

    def one_field(u: jnp.ndarray, c: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        acc = c[0] * u
        v = u
        for kk in range(1, kmax + 1):
            v = l_sym @ v
            acc = acc + c[kk] * v
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * acc).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], coeffs[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )


def apply_graph_eigen_diag_mix(
    q_mp: jnp.ndarray,
    u_basis: jnp.ndarray,
    w_eig: jnp.ndarray,
    blend_logits: jnp.ndarray,
) -> jnp.ndarray:
    """`u_basis` `(n, K)`, `w_eig` `(n_fields, K)`."""
    if u_basis.shape[1] == 0:
        return q_mp
    n_fields = q_mp.shape[0]
    u_basis = u_basis.astype(jnp.float64)

    def one_field(u: jnp.ndarray, wf: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        c = u_basis.T @ u
        v = u_basis @ (wf * c)
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * v).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], w_eig[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )
