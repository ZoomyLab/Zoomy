"""One structured V-cycle: learned graph smoothers + Galerkin coarse + Jacobi coarsest.

Scalar per cell: `n_components=1`, `n_in` in `{3,4}` (optional bump).  Vector stacked
unknown (e.g. $L\\otimes I$): set `n_components=d` and ``init_vcycle_smoothers(...,
use_bump=...)`` so each graph node has a $d$-channel correction; pass `n_components`
through `forward_vcycle*`.  Pass `b_list` (one scalar bump per cell per level) aligned with
cell counts in `edges_list`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from zoomy_jax.gnn_blueprint.graph_mp_multibranch import _mp_layer


def init_smoother(
    key: jax.Array,
    n_in: int,
    hid: int,
    n_mp: int,
    n_out: int = 1,
) -> dict[str, Any]:
    keys = jax.random.split(key, 3 + 3 * n_mp + 2)
    p: dict[str, Any] = {
        "emb_w": 0.1 * jax.random.normal(keys[0], (n_in, hid)),
        "emb_b": jnp.zeros((hid,)),
    }
    for k in range(n_mp):
        base = 1 + 3 * k
        p[f"ws{k}"] = 0.1 * jax.random.normal(keys[base], (hid, hid))
        p[f"wn{k}"] = 0.1 * jax.random.normal(keys[base + 1], (hid, hid))
        p[f"bn{k}"] = jnp.zeros((hid,))
    p["out_w"] = 0.1 * jax.random.normal(keys[-2], (hid, n_out))
    p["out_b"] = jnp.zeros((n_out,))
    return p


def init_vcycle_smoothers(
    key: jax.Array,
    n_levels: int,
    hid: int,
    n_mp: int,
    *,
    n_in: int | None = None,
    n_components: int = 1,
    use_bump: bool = False,
    n_out: int | None = None,
) -> dict[str, Any]:
    """One smoother per level; coarsest level params are unused (Jacobi-only bottom).

    **Legacy (2D / scalar 1D):** pass `n_in` in `{3, 4}` (no bump / bump). Per-node output
    is a single scalar correction.

    **Vector unknowns (`d` components per cell, Poisson on each):** omit `n_in` and set
    `n_components=d` and `use_bump`. Then `n_in = 3*d + (1 if use_bump else 0)` and
    `n_out = d` (one learned update per component at each graph node).
    """
    if n_in is not None:
        in_dim = int(n_in)
        out_dim = int(n_out) if n_out is not None else 1
    else:
        in_dim = 3 * int(n_components) + (1 if use_bump else 0)
        out_dim = int(n_out) if n_out is not None else int(n_components)
    keys = jax.random.split(key, n_levels)
    return {
        "smooth": [init_smoother(keys[i], in_dim, hid, n_mp, out_dim) for i in range(n_levels)]
    }


def smooth_step(
    u: jnp.ndarray,
    f: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    edges: jnp.ndarray,
    p: dict[str, Any],
    n_mp: int,
    hid: int,
    n_components: int = 1,
) -> jnp.ndarray:
    r = f - a @ u
    nin = int(p["emb_w"].shape[0])
    n_out = int(p["out_w"].shape[1])
    d = n_components
    if n_out != d:
        raise ValueError(f"smoother n_out={n_out} != n_components={d}")

    if d == 1:
        if nin == 3:
            x = jnp.stack([u, r, f], axis=-1)
        elif nin == 4:
            x = jnp.stack([u, r, f, b], axis=-1)
        else:
            raise ValueError(f"scalar smoother expects n_in 3 or 4, got {nin}")
    else:
        n_cell = u.shape[0] // d
        if u.shape[0] != n_cell * d:
            raise ValueError(f"u length {u.shape[0]} not divisible by n_components={d}")
        U = u.reshape(n_cell, d)
        Rv = r.reshape(n_cell, d)
        Fv = f.reshape(n_cell, d)
        x = jnp.concatenate([U, Rv, Fv], axis=-1)
        if nin == 3 * d + 1:
            x = jnp.concatenate([x, b.reshape(n_cell, 1)], axis=-1)
        elif nin != 3 * d:
            raise ValueError(
                f"vector smoother expects n_in {3 * d} or {3 * d + 1}, got {nin}"
            )

    h = x @ p["emb_w"] + p["emb_b"]
    for k in range(n_mp):
        h = _mp_layer(h, edges, p[f"ws{k}"], p[f"wn{k}"], p[f"bn{k}"])
    delta = h @ p["out_w"] + p["out_b"]
    if d == 1:
        delta = delta.squeeze(-1)
    else:
        delta = delta.reshape(-1)
    return u + delta


def jacobi_solve(
    u: jnp.ndarray,
    f: jnp.ndarray,
    a: jnp.ndarray,
    n_iter: int,
    omega: float,
) -> jnp.ndarray:
    d = jnp.diag(a)
    d_inv = jnp.where(jnp.abs(d) > 1e-14, 1.0 / d, 0.0)

    def one_step(u_in: jnp.ndarray) -> jnp.ndarray:
        r = f - a @ u_in
        return u_in + omega * d_inv * r

    u_out = u
    for _ in range(n_iter):
        u_out = one_step(u_out)
    return u_out


def vcycle_once(
    u: jnp.ndarray,
    f: jnp.ndarray,
    ell: int,
    params: dict[str, Any],
    a_list: tuple[jnp.ndarray, ...],
    r_list: tuple[jnp.ndarray, ...],
    p_list: tuple[jnp.ndarray, ...],
    edges_list: tuple[jnp.ndarray, ...],
    b_list: tuple[jnp.ndarray, ...],
    n_mp: int,
    hid: int,
    nu1: int,
    nu2: int,
    coarsest_iters: int,
    coarsest_omega: float,
    n_components: int = 1,
) -> jnp.ndarray:
    l_last = len(a_list) - 1
    a = a_list[ell]
    edges = edges_list[ell]
    b_ell = b_list[ell]
    if ell == l_last:
        return jacobi_solve(u, f, a, coarsest_iters, coarsest_omega)

    sp = params["smooth"][ell]
    for _ in range(nu1):
        u = smooth_step(u, f, b_ell, a, edges, sp, n_mp, hid, n_components)

    r = f - a @ u
    rc = r_list[ell] @ r
    nc = rc.shape[0]
    ec = jnp.zeros((nc,), dtype=u.dtype)
    ec = vcycle_once(
        ec,
        rc,
        ell + 1,
        params,
        a_list,
        r_list,
        p_list,
        edges_list,
        b_list,
        n_mp,
        hid,
        nu1,
        nu2,
        coarsest_iters,
        coarsest_omega,
        n_components,
    )
    u = u + p_list[ell] @ ec

    for _ in range(nu2):
        u = smooth_step(u, f, b_ell, a, edges, sp, n_mp, hid, n_components)
    return u


def forward_vcycle(
    f: jnp.ndarray,
    params: dict[str, Any],
    a_list: tuple[jnp.ndarray, ...],
    r_list: tuple[jnp.ndarray, ...],
    p_list: tuple[jnp.ndarray, ...],
    edges_list: tuple[jnp.ndarray, ...],
    b_list: tuple[jnp.ndarray, ...],
    n_mp: int,
    hid: int,
    nu1: int,
    nu2: int,
    coarsest_iters: int,
    coarsest_omega: float,
    n_components: int = 1,
) -> jnp.ndarray:
    n = f.shape[0]
    u0 = jnp.zeros((n,), dtype=f.dtype)
    return vcycle_once(
        u0,
        f,
        0,
        params,
        a_list,
        r_list,
        p_list,
        edges_list,
        b_list,
        n_mp,
        hid,
        nu1,
        nu2,
        coarsest_iters,
        coarsest_omega,
        n_components,
    )


def forward_vcycle_batch(
    f_b: jnp.ndarray,
    params: dict[str, Any],
    a_list: tuple[jnp.ndarray, ...],
    r_list: tuple[jnp.ndarray, ...],
    p_list: tuple[jnp.ndarray, ...],
    edges_list: tuple[jnp.ndarray, ...],
    b_list: tuple[jnp.ndarray, ...],
    n_mp: int,
    hid: int,
    nu1: int,
    nu2: int,
    coarsest_iters: int,
    coarsest_omega: float,
    n_components: int = 1,
) -> jnp.ndarray:
    def one(fv: jnp.ndarray) -> jnp.ndarray:
        return forward_vcycle(
            fv,
            params,
            a_list,
            r_list,
            p_list,
            edges_list,
            b_list,
            n_mp,
            hid,
            nu1,
            nu2,
            coarsest_iters,
            coarsest_omega,
            n_components,
        )

    return jax.vmap(one)(f_b)
