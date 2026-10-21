"""Minimal 1D FFT spectral mix demo (uniform cell ordering).

Runs a forward pass of `fourier_1d.apply_fourier_linear_mix_1d` on random
`(n_fields, n_cells)` data. Use this to sanity-check JAX FFT + dtypes on CPU/GPU.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from zoomy_jax.gnn_blueprint.fourier_1d import apply_fourier_linear_mix_1d, max_rfft_modes


def main():
    n_fields, n_cells = 3, 64
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (n_fields, n_cells))
    m = max_rfft_modes(n_cells)
    w_r = jnp.ones((n_fields, m))
    w_i = jnp.zeros((n_fields, m))
    blend = jnp.full((n_fields,), -2.0)
    out = apply_fourier_linear_mix_1d(q, w_r, w_i, blend)
    print("in ", q.shape, "out", out.shape, "max_modes", m)
    print("mean abs delta", float(jnp.mean(jnp.abs(out - q))))


if __name__ == "__main__":
    main()
