"""1D real-FFT linear mixing (FNO-style global coupling) for uniform cell ordering.

For 1D FV lines (e.g. Green–Naghdi): `q` shape `(n_fields, n_cells)`. Each field
uses `rfft`, pointwise complex multiply with learned weights, then `irfft`. A
per-field sigmoid blend mixes the result with the input (identity when blend → 0).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def apply_fourier_linear_mix_1d(
    q_mp: jnp.ndarray,
    w_r: jnp.ndarray,
    w_i: jnp.ndarray,
    blend_logits: jnp.ndarray,
) -> jnp.ndarray:
    """Spectral linear map per field + blend with identity.

    `w_r`, `w_i`: `(n_fields, max_modes)`; only `rfft` length is used.
    `blend_logits`: `(n_fields,)`.
    """
    n_fields, n_cells = q_mp.shape

    def one_field(u: jnp.ndarray, wr: jnp.ndarray, wi: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
        spec = jnp.fft.rfft(u)
        k = spec.shape[0]
        wrk = wr[:k]
        wik = wi[:k]
        w = (wrk + 1j * wik).astype(jnp.complex128)
        v = jnp.fft.irfft(spec * w, n=n_cells)
        b = jax.nn.sigmoid(logit)
        return (1.0 - b) * u + b * v

    out = jnp.stack(
        [one_field(q_mp[i], w_r[i], w_i[i], blend_logits[i]) for i in range(n_fields)],
        axis=0,
    )
    return out.astype(q_mp.dtype)


def max_rfft_modes(n_cells: int) -> int:
    return int(n_cells // 2 + 1)
