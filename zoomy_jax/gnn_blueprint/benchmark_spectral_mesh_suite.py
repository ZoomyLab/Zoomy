"""GMRES matvec counts for `train_spectral_mesh_suite` checkpoints.

Rebuilds the discrete operator `A` from `mesh_suite` + `dataset_seed` metadata
and compares learned `x0 = pred u` to zero initial guess.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

try:
    from scipy.sparse.linalg import LinearOperator, gmres
except ImportError as e:
    raise SystemExit("scipy required") from e

from zoomy_jax.gnn_blueprint import global_coupling as gc
from zoomy_jax.gnn_blueprint.mesh_spectral_geometry import (
    delaunay_triangle_centroids_laplacian,
    laplacian_1d_nonuniform_vertex,
    morton_z_order_perm,
)
from zoomy_jax.gnn_blueprint.poisson_1d import laplacian_1d_dense
from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned


def _tag(a) -> str:
    v = np.asarray(a).reshape(-1)[0]
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def rebuild_operator(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    mesh = _tag(npz["mesh_suite"])
    seed = int(np.asarray(npz["dataset_seed"]).reshape(-1)[0])
    n = int(np.asarray(npz["n_cells_poisson"]).reshape(-1)[0])
    if mesh == "uniform_1d":
        return np.asarray(laplacian_1d_dense(n), dtype=np.float64)
    if mesh == "nonuniform_1d":
        rng = np.random.default_rng(seed)
        xp = np.sort(rng.uniform(0.06, 0.94, size=n))
        return laplacian_1d_nonuniform_vertex(xp, 0.0, 1.0).astype(np.float64)
    if mesh == "tri2d":
        tp = int(np.asarray(npz["tri_points"]).reshape(-1)[0])
        ridge = float(np.asarray(npz["graph_ridge"]).reshape(-1)[0])
        centroids, a_np, _, _ = delaunay_triangle_centroids_laplacian(tp, seed, ridge=ridge)
        perm = morton_z_order_perm(centroids)
        n_c = centroids.shape[0]
        pmat = np.zeros((n_c, n_c), dtype=np.float64)
        for i, j in enumerate(perm):
            pmat[i, j] = 1.0
        return (pmat @ a_np @ pmat.T).astype(np.float64)
    raise ValueError(f"unknown mesh {mesh}")


def _vec(d, name, default, n_fields: int):
    v = np.asarray(d.get(name, default), dtype=float)
    if v.size < n_fields:
        v = np.pad(v, (0, n_fields - v.size), mode="edge")
    return jnp.asarray(v[:n_fields])


def load_params(weights: Path, n_fields: int, message_steps: int) -> dict:
    d = np.load(weights)
    flow_raw = d.get("flow_mode", np.asarray(["bidir"]))
    flow_mode = str(flow_raw.reshape(-1)[0]) if isinstance(flow_raw, np.ndarray) else str(flow_raw)
    gcm = int(np.asarray(d.get("global_coupling_mode", [gc.MULTIGRID])).reshape(-1)[0])
    out = {
        "w_self": _vec(d, "w_self", np.ones(n_fields) * 0.05, n_fields),
        "w_msg": _vec(d, "w_msg", np.ones(n_fields) * 0.02, n_fields),
        "w_aux": _vec(d, "w_aux", np.ones(n_fields) * 0.01, n_fields),
        "w_coarse": _vec(d, "w_coarse", np.ones(n_fields) * 0.03, n_fields),
        "w_gate": _vec(d, "w_gate", np.ones(n_fields), n_fields),
        "b": _vec(d, "b", np.zeros(n_fields), n_fields),
        "message_steps": int(np.asarray(d.get("message_steps", [message_steps])).reshape(-1)[0]),
        "inner_iters": int(np.asarray(d.get("inner_iters", [1])).reshape(-1)[0]),
        "coarsen_levels": int(np.asarray(d.get("coarsen_levels", [message_steps])).reshape(-1)[0]),
        "flow_mode": flow_mode,
        "global_coupling_mode": gcm,
        "single_layer_mode": int(np.asarray(d.get("single_layer_mode", [0])).reshape(-1)[0]),
    }
    if gcm == gc.FFT_1D:
        mm = int(np.asarray(d.get("max_fft_modes", [1])).reshape(-1)[0])
        fr = np.asarray(d.get("fft_w_r", np.ones((n_fields, mm))), dtype=float)
        fi = np.asarray(d.get("fft_w_i", np.zeros((n_fields, mm))), dtype=float)
        if fr.shape[0] < n_fields:
            fr = np.pad(fr, ((0, n_fields - fr.shape[0]), (0, 0)), mode="edge")
        if fi.shape[0] < n_fields:
            fi = np.pad(fi, ((0, n_fields - fi.shape[0]), (0, 0)), mode="edge")
        out["fft_w_r"] = jnp.asarray(fr[:n_fields])
        out["fft_w_i"] = jnp.asarray(fi[:n_fields])
        out["fft_blend_logit"] = _vec(d, "fft_blend_logit", np.zeros(n_fields), n_fields)
    elif gcm in (gc.NUDFT_1D, gc.NUDFT_2D):
        nm = int(np.asarray(d["spectral_w_r"]).shape[1])
        fr = np.asarray(d["spectral_w_r"], dtype=float)
        fi = np.asarray(d["spectral_w_i"], dtype=float)
        out["spectral_w_r"] = jnp.asarray(fr[:n_fields])
        out["spectral_w_i"] = jnp.asarray(fi[:n_fields])
        out["spectral_blend_logit"] = _vec(d, "spectral_blend_logit", np.zeros(n_fields), n_fields)
        if gcm == gc.NUDFT_1D:
            out["spectral_x"] = jnp.asarray(d["spectral_x"], dtype=jnp.float64)
            out["n_spectral_modes_1d"] = int(np.asarray(d["n_spectral_modes_1d"]).reshape(-1)[0])
        else:
            out["spectral_xy"] = jnp.asarray(d["spectral_xy"], dtype=jnp.float64)
            out["spectral_kmax"] = int(np.asarray(d.get("spectral_kmax", [3])).reshape(-1)[0])
    elif gcm in (gc.RFF_KERNEL_1D, gc.RFF_KERNEL_2D):
        out["rff_omega"] = jnp.asarray(d["rff_omega"], dtype=jnp.float64)
        out["rff_phase"] = jnp.asarray(d["rff_phase"], dtype=jnp.float64)
        out["rff_w_lin"] = jnp.asarray(d["rff_w_lin"], dtype=jnp.float64)
        out["spectral_blend_logit"] = _vec(d, "spectral_blend_logit", np.zeros(n_fields), n_fields)
        if gcm == gc.RFF_KERNEL_1D:
            out["spectral_x"] = jnp.asarray(d["spectral_x"], dtype=jnp.float64)
        else:
            out["spectral_xy"] = jnp.asarray(d["spectral_xy"], dtype=jnp.float64)
    elif gcm == gc.GRAPH_POLY_LAPL:
        out["graph_L_sym"] = jnp.asarray(d["graph_L_sym"], dtype=jnp.float64)
        out["graph_poly_coeff"] = jnp.asarray(d["graph_poly_coeff"], dtype=jnp.float64)
        out["graph_blend_logit"] = _vec(d, "graph_blend_logit", np.zeros(n_fields), n_fields)
    elif gcm == gc.GRAPH_EIGEN_LOW:
        out["graph_eig_U"] = jnp.asarray(d["graph_eig_U"], dtype=jnp.float64)
        out["graph_eig_w"] = jnp.asarray(d["graph_eig_w"], dtype=jnp.float64)
        out["graph_blend_logit"] = _vec(d, "graph_blend_logit", np.zeros(n_fields), n_fields)
    return out


def _gmres_count(a_np: np.ndarray, f: np.ndarray, x0: np.ndarray, rtol: float, maxiter: int):
    n = a_np.shape[0]
    count = [0]

    def mv(v):
        count[0] += 1
        return a_np @ v

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    y, info = gmres(op, f, x0=x0.copy(), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
    rn = float(np.linalg.norm(a_np @ y - f))
    return count[0], rn, int(info)


def run_weights(weights: Path, n_trials: int, seed: int, rtol: float, maxiter: int) -> dict:
    d = np.load(weights)
    a_np = rebuild_operator(d)
    n = a_np.shape[0]
    n_fields = 2
    ms = int(np.asarray(d.get("message_steps", [3])).reshape(-1)[0])
    p = load_params(weights, n_fields, ms)
    rng = np.random.default_rng(seed)
    mvs = []
    for _ in range(n_trials):
        f = rng.standard_normal(n)
        q = np.stack([np.zeros(n), f], axis=0)
        qj = jnp.asarray(q)
        qax = jnp.asarray(f[None, :])
        dq = predict_delta_q_learned(qj, qax, jnp.asarray(1.0), jnp.zeros(n), p, ms)
        x0 = np.asarray(dq[0], dtype=np.float64)
        n_mv, rn, _ = _gmres_count(a_np, f, x0, rtol, maxiter)
        mvs.append(n_mv)
    base = []
    rng2 = np.random.default_rng(seed + 99_001)
    for _ in range(n_trials):
        f = rng2.standard_normal(n)
        n_mv, _, _ = _gmres_count(a_np, f, np.zeros(n), rtol, maxiter)
        base.append(n_mv)
    return {
        "name": weights.parent.name,
        "mean_mv": float(np.mean(mvs)),
        "std_mv": float(np.std(mvs)),
        "base_mean_mv": float(np.mean(base)),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", type=Path, required=True)
    ap.add_argument("--n-trials", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=500)
    args = ap.parse_args()
    root = Path.cwd()
    scan = args.scan_root if args.scan_root.is_absolute() else root / args.scan_root
    rows = []
    for sub in sorted(scan.iterdir()):
        w = sub / "weights_deltaq.npz"
        if w.is_file():
            rows.append(run_weights(w, args.n_trials, args.seed, args.rtol, args.maxiter))
    print("name,n,base_mean_mv,mean_mv,std_mv")
    for r in sorted(rows, key=lambda x: x["mean_mv"]):
        print(f"{r['name']},{r['n']},{r['base_mean_mv']:.2f},{r['mean_mv']:.2f},{r['std_mv']:.2f}")


if __name__ == "__main__":
    main()
