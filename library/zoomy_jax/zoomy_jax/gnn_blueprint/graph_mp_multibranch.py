"""Lightweight graph message passing + 3-branch specialists + per-node gated fusion."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _mp_layer(h: jnp.ndarray, edges: jnp.ndarray, ws: jnp.ndarray, wn: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """One message-passing step; `h` (n, d), `edges` int32 (2, E)."""
    send = edges[0].astype(jnp.int32)
    recv = edges[1].astype(jnp.int32)
    msg = h[send] @ wn
    agg = jnp.zeros_like(h)
    agg = agg.at[recv].add(msg)
    return jnp.tanh(h @ ws + agg + b)


def _branch_forward(
    f_node: jnp.ndarray,
    edges: jnp.ndarray,
    p: dict,
    n_mp: int,
    hid: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Input `f_node` (n,) per sample; returns `out` (n, 1), last hidden (n, hid)."""
    n = f_node.shape[0]
    x = jnp.stack([jnp.zeros((n,), dtype=f_node.dtype), f_node], axis=-1)
    h = x @ p["emb_w"] + p["emb_b"]
    for k in range(n_mp):
        h = _mp_layer(h, edges, p[f"ws{k}"], p[f"wn{k}"], p[f"bn{k}"])
    out = h @ p["out_w"] + p["out_b"]
    return out, h


def _fuse(
    o0: jnp.ndarray,
    o1: jnp.ndarray,
    o2: jnp.ndarray,
    f_node: jnp.ndarray,
    p: dict,
) -> jnp.ndarray:
    """Per-node softmax gates from `[o0,o1,o2,f]`."""
    z = jnp.concatenate([o0, o1, o2, f_node[:, None]], axis=-1)
    logits = z @ p["fuse_w"] + p["fuse_b"]
    g = jax.nn.softmax(logits, axis=-1)
    return g[:, 0:1] * o0 + g[:, 1:2] * o1 + g[:, 2:3] * o2


def forward_multibranch(
    f_b: jnp.ndarray,
    edges: jnp.ndarray,
    p0: dict,
    p1: dict,
    p2: dict,
    pf: dict,
    n_mp: int,
    hid: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """`f_b` (B, n). Returns fused (B,n,1), o0,o1,o2 (B,n,1)."""
    def one(f):
        o0, _ = _branch_forward(f, edges, p0, n_mp, hid)
        o1, _ = _branch_forward(f, edges, p1, n_mp, hid)
        o2, _ = _branch_forward(f, edges, p2, n_mp, hid)
        fu = _fuse(o0, o1, o2, f, pf)
        return fu, o0, o1, o2

    return jax.vmap(one)(f_b)


def forward_z(
    r_b: jnp.ndarray,
    edges: jnp.ndarray,
    p: dict,
    n_mp: int,
    hid: int,
) -> jnp.ndarray:
    """`r_b` (B, n) → `z` (B, n, 1)."""

    def one(r):
        n = r.shape[0]
        x = jnp.stack([jnp.zeros((n,), dtype=r.dtype), r], axis=-1)
        h = x @ p["emb_w"] + p["emb_b"]
        for k in range(n_mp):
            h = _mp_layer(h, edges, p[f"ws{k}"], p[f"wn{k}"], p[f"bn{k}"])
        return h @ p["out_w"] + p["out_b"]

    return jax.vmap(one)(r_b)


def init_branch(key, n_in: int, hid: int, n_mp: int) -> dict:
    keys = jax.random.split(key, 3 + 3 * n_mp + 2)
    p = {
        "emb_w": 0.1 * jax.random.normal(keys[0], (n_in, hid)),
        "emb_b": jnp.zeros((hid,)),
    }
    for k in range(n_mp):
        base = 1 + 3 * k
        p[f"ws{k}"] = 0.1 * jax.random.normal(keys[base], (hid, hid))
        p[f"wn{k}"] = 0.1 * jax.random.normal(keys[base + 1], (hid, hid))
        p[f"bn{k}"] = jnp.zeros((hid,))
    p["out_w"] = 0.1 * jax.random.normal(keys[-2], (hid, 1))
    p["out_b"] = jnp.zeros((1,))
    return p


def init_fuse(key, hid_in: int = 4) -> dict:
    k0, k1 = jax.random.split(key)
    return {
        "fuse_w": 0.15 * jax.random.normal(k0, (hid_in, 3)),
        "fuse_b": jnp.zeros((3,)),
    }
