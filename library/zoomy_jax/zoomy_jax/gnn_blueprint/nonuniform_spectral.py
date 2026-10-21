"""Non-uniform and mesh-agnostic spectral global coupling (NUFFT-style NUDFT + kernel RFF).

**NUDFT** (non-uniform discrete Fourier transform): fixed low-frequency exponentials
`exp(2π i (k·x))` evaluated at node coordinates, analysis `c = Φᴴ u`, learned
per-mode complex weights, synthesis `u' = Re(Φ (w ⊙ c))`. This is a dense
low-mode *type-1 / type-2* NUFFT surrogate (equispaced frequency grid, non-uniform
spatial samples) and is differentiable in JAX.

**RFF kernel mix** (kernel / random Fourier feature view): `cos(2π (ω_m·x) + φ_m)`
with frozen `ω, φ` and learned linear readout + blend — stationary-kernel
approximation on arbitrary point sets.

Coordinates must lie in `[0, 1]` (per dimension) for conditioning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def nudft_phi_matrix_1d(x: jnp.ndarray, n_modes: int) -> jnp.ndarray:
    """`Phi[j,m] = exp(2π i m x_j) / sqrt(n)`, `x` shape `(n,)`."""
    n = x.shape[0]
    m = jnp.arange(n_modes, dtype=jnp.float64)[None, :]
    phase = 2.0 * jnp.pi * x[:, None] * m
    scale = jnp.sqrt(jnp.asarray(max(n, 1), dtype=jnp.float64))
    return jnp.exp(1j * phase) / scale


def nudft_mode_pairs_2d(kmax: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integer wavevectors `(kx, ky)` with `|kx|,|ky| <= kmax`."""
    ks = []
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            ks.append((kx, ky))
    if not ks:
        return jnp.zeros((0,), jnp.int32), jnp.zeros((0,), jnp.int32)
    a = jnp.asarray(ks, dtype=jnp.float64)
    return a[:, 0], a[:, 1]


def nudft_phi_matrix_2d(xy: jnp.ndarray, kx: jnp.ndarray, ky: jnp.ndarray) -> jnp.ndarray:
    """`Phi[j,m] = exp(2π i (kx_m x_j + ky_m y_j)) / sqrt(n)`."""
    n = xy.shape[0]
    phase = 2.0 * jnp.pi * (xy[:, 0:1] * kx[None, :] + xy[:, 1:2] * ky[None, :])
    scale = jnp.sqrt(jnp.asarray(max(n, 1), dtype=jnp.float64))
    return jnp.exp(1j * phase) / scale


def nudft_num_modes_2d(kmax: int) -> int:
    return (2 * kmax + 1) ** 2


def apply_nudft_linear_mix_1d(
    q_mp: jnp.ndarray,
    x: jnp.ndarray,
    w_r: jnp.ndarray,
    w_i: jnp.ndarray,
    blend_logits: jnp.ndarray,
    n_modes: int,
) -> jnp.ndarray:
    """Per-field NUDFT linear map + sigmoid blend with identity (same contract as FFT block)."""
    Phi = nudft_phi_matrix_1d(x, n_modes).astype(jnp.complex128)
    n_fields = q_mp.shape[0]

    def one_field(u: jnp.ndarray, wr: jnp.ndarray, wi: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        c = jnp.conj(Phi).T @ u.astype(jnp.float64)
        k = c.shape[0]
        w = (wr[:k] + 1j * wi[:k]).astype(jnp.complex128)
        v = jnp.real(Phi @ (w * c))
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * v).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], w_r[i], w_i[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )


def apply_nudft_linear_mix_2d(
    q_mp: jnp.ndarray,
    xy: jnp.ndarray,
    w_r: jnp.ndarray,
    w_i: jnp.ndarray,
    blend_logits: jnp.ndarray,
    kmax: int,
) -> jnp.ndarray:
    kx, ky = nudft_mode_pairs_2d(kmax)
    Phi = nudft_phi_matrix_2d(xy.astype(jnp.float64), kx, ky).astype(jnp.complex128)
    n_modes = int(Phi.shape[1])
    n_fields = q_mp.shape[0]

    def one_field(u: jnp.ndarray, wr: jnp.ndarray, wi: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        c = jnp.conj(Phi).T @ u.astype(jnp.float64)
        w = (wr[:n_modes] + 1j * wi[:n_modes]).astype(jnp.complex128)
        v = jnp.real(Phi @ (w * c))
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * v).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], w_r[i], w_i[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )


def apply_rff_kernel_mix_1d(
    q_mp: jnp.ndarray,
    x: jnp.ndarray,
    omega: jnp.ndarray,
    phase: jnp.ndarray,
    w_lin: jnp.ndarray,
    blend_logits: jnp.ndarray,
) -> jnp.ndarray:
    """`x` (n,), `omega` (M,), `phase` (M,), `w_lin` (n_fields, M)."""
    z = 2.0 * jnp.pi * x[:, None] * omega[None, :] + phase[None, :]
    feats = jnp.cos(z)
    n_fields = q_mp.shape[0]

    def one_field(u: jnp.ndarray, wl: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        v = feats @ wl
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * v).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], w_lin[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )


def apply_rff_kernel_mix_2d(
    q_mp: jnp.ndarray,
    xy: jnp.ndarray,
    omega: jnp.ndarray,
    phase: jnp.ndarray,
    w_lin: jnp.ndarray,
    blend_logits: jnp.ndarray,
) -> jnp.ndarray:
    """`xy` (n,2), `omega` (M,2), `phase` (M,), `w_lin` (n_fields, M)."""
    z = 2.0 * jnp.pi * jnp.sum(xy[:, None, :] * omega[None, :, :], axis=-1) + phase[None, :]
    feats = jnp.cos(z)
    n_fields = q_mp.shape[0]

    def one_field(u: jnp.ndarray, wl: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        v = feats @ wl
        b = jax.nn.sigmoid(logit)
        return ((1.0 - b) * u + b * v).astype(q_mp.dtype)

    return jnp.stack(
        [one_field(q_mp[i], w_lin[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )


def max_nudft_modes_1d(n_cells: int) -> int:
    return int(n_cells // 2 + 1)
