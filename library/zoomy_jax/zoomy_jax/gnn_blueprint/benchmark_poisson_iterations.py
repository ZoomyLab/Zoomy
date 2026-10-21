"""GMRES matvec counts for Poisson 1D with neural initial guesses (architecture ablation).

Expects one `weights_deltaq.npz` per run (from `train_poisson_arch_benchmark`) or
`--scan-root` to aggregate all subfolders.

Uses `scipy.sparse.linalg.gmres` on dense `A u = f` with `x0 = pred[0]` from the
learned predictor (same layout as training).
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
    raise SystemExit("scipy required for GMRES benchmark") from e

try:
    from zoomy_jax.gnn_blueprint import global_coupling as gc
    from zoomy_jax.gnn_blueprint.poisson_1d import laplacian_1d_dense
    from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import global_coupling as gc
    from poisson_1d import laplacian_1d_dense
    from predictor_learned_multilevel import predict_delta_q_learned


def _box_smooth_np(x: np.ndarray, radius: int) -> np.ndarray:
    r = int(max(radius, 0))
    if r == 0:
        return x.copy()
    k = 2 * r + 1
    xp = np.pad(x, (r, r), mode="edge")
    acc = np.zeros_like(x, dtype=np.float64)
    for j in range(k):
        acc += xp[j : j + x.shape[0]]
    return acc / float(k)


def _build_q_np(f_row: np.ndarray, radii: tuple[int, ...]) -> np.ndarray:
    n = f_row.shape[0]
    u = np.zeros(n, dtype=np.float64)
    rows = [u, f_row.copy()]
    for r in radii:
        rows.append(_box_smooth_np(u, int(r)))
        rows.append(_box_smooth_np(f_row, int(r)))
    return np.stack(rows, axis=0)


def _load_p(weights: Path, n_fields: int, message_steps: int) -> dict:
    d = np.load(weights)

    def _vec(name, default):
        v = np.asarray(d.get(name, default), dtype=float)
        if v.size < n_fields:
            v = np.pad(v, (0, n_fields - v.size), mode="edge")
        return jnp.asarray(v[:n_fields])

    flow_raw = d.get("flow_mode", np.asarray(["bidir"]))
    flow_mode = str(flow_raw.reshape(-1)[0]) if isinstance(flow_raw, np.ndarray) else str(flow_raw)
    gcm = int(np.asarray(d.get("global_coupling_mode", [gc.MULTIGRID])).reshape(-1)[0])
    out = {
        "w_self": _vec("w_self", np.ones(n_fields) * 0.05),
        "w_msg": _vec("w_msg", np.ones(n_fields) * 0.02),
        "w_aux": _vec("w_aux", np.ones(n_fields) * 0.01),
        "w_coarse": _vec("w_coarse", np.ones(n_fields) * 0.03),
        "w_gate": _vec("w_gate", np.ones(n_fields)),
        "b": _vec("b", np.zeros(n_fields)),
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
        out["fft_blend_logit"] = _vec("fft_blend_logit", np.zeros(n_fields))
    elif gcm in (gc.NUDFT_1D, gc.NUDFT_2D):
        fr = np.asarray(d["spectral_w_r"], dtype=float)
        fi = np.asarray(d["spectral_w_i"], dtype=float)
        if fr.shape[0] < n_fields:
            fr = np.pad(fr, ((0, n_fields - fr.shape[0]), (0, 0)), mode="edge")
        if fi.shape[0] < n_fields:
            fi = np.pad(fi, ((0, n_fields - fi.shape[0]), (0, 0)), mode="edge")
        out["spectral_w_r"] = jnp.asarray(fr[:n_fields])
        out["spectral_w_i"] = jnp.asarray(fi[:n_fields])
        out["spectral_blend_logit"] = _vec("spectral_blend_logit", np.zeros(n_fields))
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
        out["spectral_blend_logit"] = _vec("spectral_blend_logit", np.zeros(n_fields))
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


def _gmres_matvec_count(a_np: np.ndarray, f: np.ndarray, x0: np.ndarray, rtol: float, maxiter: int) -> tuple[int, float, int]:
    """Return (matvec_count, final_residual_norm, scipy_info)."""
    n = a_np.shape[0]
    count = [0]

    def mv(v):
        count[0] += 1
        return a_np @ v

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    y, info = gmres(op, f, x0=x0.copy(), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
    rn = float(np.linalg.norm(a_np @ y - f))
    return count[0], rn, int(info)


def run_one_weights(
    weights: Path,
    n: int,
    n_trials: int,
    seed: int,
    rtol: float,
    maxiter: int,
    message_steps: int,
) -> dict:
    d = np.load(weights)
    if "n_cells_poisson" in d.files:
        n = int(np.asarray(d["n_cells_poisson"]).reshape(-1)[0])
    if "smooth_radii" in d.files and d["smooth_radii"].size:
        radii = tuple(int(x) for x in np.asarray(d["smooth_radii"]).reshape(-1))
    else:
        radii = ()
    n_fields = 2 + 2 * len(radii)
    p = _load_p(weights, n_fields, message_steps)
    ms = int(p["message_steps"])
    a_np = np.asarray(laplacian_1d_dense(n), dtype=np.float64)
    rng = np.random.default_rng(seed)

    matvecs = []
    resids = []
    for _ in range(n_trials):
        f = rng.standard_normal(n)
        q = _build_q_np(f, radii)
        qj = jnp.asarray(q)
        qax = jnp.asarray(f[None, :])
        dt = jnp.asarray(1.0)
        cls = jnp.zeros((n,), dtype=jnp.float64)
        dq = predict_delta_q_learned(qj, qax, dt, cls, p, ms, return_diagnostics=False)
        x0 = np.asarray(dq[0], dtype=np.float64)
        n_mv, rn, info = _gmres_matvec_count(a_np, f, x0, rtol=rtol, maxiter=maxiter)
        matvecs.append(n_mv)
        resids.append(rn)

    return {
        "weights": str(weights),
        "mean_matvecs": float(np.mean(matvecs)),
        "std_matvecs": float(np.std(matvecs)),
        "mean_resid": float(np.mean(resids)),
        "n_trials": n_trials,
    }


def main():
    ap = argparse.ArgumentParser(description="Poisson 1D GMRES matvec benchmark for learned x0")
    ap.add_argument("--weights", type=Path, default=None, help="Single weights_deltaq.npz")
    ap.add_argument("--scan-root", type=Path, default=None, help="Directory with one subfolder per arch containing weights_deltaq.npz")
    ap.add_argument("--n-cells", type=int, default=64)
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--message-steps", type=int, default=3)
    args = ap.parse_args()
    root = Path.cwd()

    rows = []
    baseline_n = args.n_cells
    if args.scan_root:
        scan = args.scan_root if args.scan_root.is_absolute() else root / args.scan_root
        for sub in sorted(scan.iterdir()):
            w = sub / "weights_deltaq.npz"
            if w.is_file():
                r = run_one_weights(w, args.n_cells, args.n_trials, args.seed, args.rtol, args.maxiter, args.message_steps)
                r["arch"] = sub.name
                rows.append(r)
                if "n_cells_poisson" in np.load(w).files:
                    baseline_n = int(np.asarray(np.load(w)["n_cells_poisson"]).reshape(-1)[0])
    elif args.weights:
        w = args.weights if args.weights.is_absolute() else root / args.weights
        r = run_one_weights(w, args.n_cells, args.n_trials, args.seed, args.rtol, args.maxiter, args.message_steps)
        r["arch"] = w.parent.name
        rows.append(r)
        if "n_cells_poisson" in np.load(w).files:
            baseline_n = int(np.asarray(np.load(w)["n_cells_poisson"]).reshape(-1)[0])
    else:
        raise SystemExit("Provide --weights or --scan-root")

    a_np = np.asarray(np.array(laplacian_1d_dense(baseline_n)), dtype=np.float64)
    rng = np.random.default_rng(args.seed + 10_000)
    base_mvs = []
    for _ in range(args.n_trials):
        f = rng.standard_normal(baseline_n)
        n_mv, _, _ = _gmres_matvec_count(a_np, f, np.zeros(baseline_n), rtol=args.rtol, maxiter=args.maxiter)
        base_mvs.append(n_mv)
    print(
        f"baseline_x0_zero,n={baseline_n},mean_matvecs={float(np.mean(base_mvs)):.2f},"
        f"std={float(np.std(base_mvs)):.2f}"
    )
    print("arch,mean_matvecs,std_matvecs,mean_resid,n_trials")
    for r in sorted(rows, key=lambda x: x["mean_matvecs"]):
        print(
            f"{r['arch']},{r['mean_matvecs']:.2f},{r['std_matvecs']:.2f},"
            f"{r['mean_resid']:.3e},{r['n_trials']}"
        )


if __name__ == "__main__":
    main()
