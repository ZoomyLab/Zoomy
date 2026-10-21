"""Build NPZ datasets for 2D Poisson multibranch + Poisson-z experiments.

**Physics split (linear):** Sample three random forcings `fd, fa, fs`, `f = fd+fa+fs`,
`ud = A^{-1} fd`, etc., `ut = ud+ua+us` (exact). Branches can be supervised on
`ud,ua,us` and fused to match `ut`.

Saves one file per mesh configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zoomy_jax.gnn_blueprint.mesh_2d_poisson import (
    edge_list_from_adjacency,
    edge_list_from_grid,
    laplacian_2d_interior_dense,
    interior_grid_shape,
)
from zoomy_jax.gnn_blueprint.mesh_spectral_geometry import delaunay_centroid_adjacency, morton_z_order_perm


def build_structured(nx: int, ny: int, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = laplacian_2d_interior_dense(nx, ny)
    ni_x, ni_y, n = interior_grid_shape(nx, ny)
    edges = edge_list_from_grid(ni_x, ni_y)
    fd = rng.standard_normal((n_samples, n))
    fa = rng.standard_normal((n_samples, n))
    fs = rng.standard_normal((n_samples, n))
    f = fd + fa + fs
    ud = np.linalg.solve(a, fd.T).T
    ua = np.linalg.solve(a, fa.T).T
    us = np.linalg.solve(a, fs.T).T
    ut = ud + ua + us
    chk = np.linalg.norm(np.linalg.solve(a, f.T).T - ut, axis=1).max()
    if chk > 1e-8:
        raise RuntimeError("linear split consistency failed")
    return {
        "A": a,
        "edges": edges,
        "f": f.astype(np.float64),
        "fd": fd.astype(np.float64),
        "fa": fa.astype(np.float64),
        "fs": fs.astype(np.float64),
        "ud": ud.astype(np.float64),
        "ua": ua.astype(np.float64),
        "us": us.astype(np.float64),
        "ut": ut.astype(np.float64),
        "mesh_kind": np.asarray(["structured"]),
        "nx": np.asarray([nx], dtype=np.int32),
        "ny": np.asarray([ny], dtype=np.int32),
        "n_nodes": np.asarray([n], dtype=np.int32),
        "dataset_seed": np.asarray([seed], dtype=np.int32),
    }


def build_unstructured(n_points: int, ridge: float, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    centroids, adj, _pts, _S = delaunay_centroid_adjacency(n_points, seed)
    perm = morton_z_order_perm(centroids)
    n_c = centroids.shape[0]
    pmat = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j in enumerate(perm):
        pmat[i, j] = 1.0
    adj_p = pmat @ adj @ pmat.T
    deg = adj_p.sum(axis=1)
    l_g = -adj_p + np.diag(deg)
    a = l_g + ridge * np.eye(n_c)
    edges = edge_list_from_adjacency(adj_p)
    n = n_c
    fd = rng.standard_normal((n_samples, n))
    fa = rng.standard_normal((n_samples, n))
    fs = rng.standard_normal((n_samples, n))
    f = fd + fa + fs
    ud = np.linalg.solve(a, fd.T).T
    ua = np.linalg.solve(a, fa.T).T
    us = np.linalg.solve(a, fs.T).T
    ut = ud + ua + us
    return {
        "A": a.astype(np.float64),
        "edges": edges,
        "f": f.astype(np.float64),
        "fd": fd.astype(np.float64),
        "fa": fa.astype(np.float64),
        "fs": fs.astype(np.float64),
        "ud": ud.astype(np.float64),
        "ua": ua.astype(np.float64),
        "us": us.astype(np.float64),
        "ut": ut.astype(np.float64),
        "mesh_kind": np.asarray(["unstructured"]),
        "tri_points": np.asarray([n_points], dtype=np.int32),
        "graph_ridge": np.asarray([ridge], dtype=np.float64),
        "n_nodes": np.asarray([n], dtype=np.int32),
        "dataset_seed": np.asarray([seed], dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gnn_blueprint/dataset_2d_mg"))
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path.cwd()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("struct_s", 10, 10),
        ("struct_m", 14, 14),
        ("struct_l", 18, 18),
    ]
    for name, nx, ny in configs:
        d = build_structured(nx, ny, args.n_samples, args.seed + abs(hash(name)) % 999)
        np.savez_compressed(out_dir / f"{name}.npz", **d)
        print(f"wrote {name} n={d['n_nodes'].reshape(-1)[0]}")

    for name, tp in [("tri_s", 28), ("tri_m", 40), ("tri_l", 52)]:
        d = build_unstructured(tp, ridge=0.03, n_samples=args.n_samples, seed=args.seed + tp)
        np.savez_compressed(out_dir / f"{name}.npz", **d)
        print(f"wrote {name} n={d['n_nodes'].reshape(-1)[0]}")


if __name__ == "__main__":
    main()
