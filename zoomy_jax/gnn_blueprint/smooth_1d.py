"""1D box (moving-average) smoothing for cell-centered signals."""

from __future__ import annotations

import jax.numpy as jnp


def box_smooth_1d_vec(x: jnp.ndarray, radius: int) -> jnp.ndarray:
    """1D vector `(n,)` symmetric moving average with `edge` padding.

    `radius` is half-width in cells (kernel size `2*radius+1`).
    """
    r = int(max(radius, 0))
    if r == 0:
        return x
    k = 2 * r + 1
    xp = jnp.pad(x, (r, r), mode="edge")
    acc = jnp.zeros_like(x)
    for j in range(k):
        acc = acc + xp[j : j + x.shape[0]]
    return acc / jnp.asarray(k, dtype=x.dtype)
