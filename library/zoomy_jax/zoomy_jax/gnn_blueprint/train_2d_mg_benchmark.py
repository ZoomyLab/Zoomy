"""Train (1) 3-branch graph MP + gated fusion on physics-split Poisson data, and/or
(2) a single-tower `r ↦ z` map for `z ≈ A^{-1} r` on the same mesh.

Evaluates **MSE on test** and **GMRES matvec counts** (`x0 = z(f)` or fused model as `u0` for `A u = f`).

Example:

```
    python dataset_2d_mg_multibranch.py --out-dir outputs/gnn_blueprint/dataset_2d_mg
    python train_2d_mg_benchmark.py --data-dir outputs/gnn_blueprint/dataset_2d_mg \\
        --epochs 120 --n-trials-gmres 32
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

try:
    from scipy.sparse.linalg import LinearOperator, gmres
except ImportError as e:
    raise SystemExit("scipy required") from e

from zoomy_jax.gnn_blueprint.graph_mp_multibranch import (
    forward_multibranch,
    forward_z,
    init_branch,
    init_fuse,
)


def _split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    ix = np.arange(n)
    rng.shuffle(ix)
    n_tr = int(0.75 * n)
    n_va = int(0.1 * n)
    return ix[:n_tr], ix[n_tr : n_tr + n_va], ix[n_tr + n_va :]


def _bundle_init(key, n_mp: int, hid: int):
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    return {
        "b0": init_branch(k0, 2, hid, n_mp),
        "b1": init_branch(k1, 2, hid, n_mp),
        "b2": init_branch(k2, 2, hid, n_mp),
        "fuse": init_fuse(k3),
    }


def train_multibranch(
    data: dict,
    epochs: int,
    batch: int,
    lr: float,
    lambda_branch: float,
    n_mp: int,
    hid: int,
    seed: int,
):
    f = jnp.asarray(data["f"], dtype=jnp.float64)
    ud = jnp.asarray(data["ud"], dtype=jnp.float64)
    ua = jnp.asarray(data["ua"], dtype=jnp.float64)
    us = jnp.asarray(data["us"], dtype=jnp.float64)
    ut = jnp.asarray(data["ut"], dtype=jnp.float64)
    a = jnp.asarray(data["A"], dtype=jnp.float64)
    edges = jnp.asarray(data["edges"], dtype=jnp.int32)
    n_s, n = f.shape
    tr, va, te = _split(n_s, seed)

    key = jax.random.PRNGKey(seed + 31)
    params = _bundle_init(key, n_mp, hid)
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def loss_fn(p, idx):
        fb = f[idx]
        fused, o0, o1, o2 = forward_multibranch(fb, edges, p["b0"], p["b1"], p["b2"], p["fuse"], n_mp, hid)
        ft = ut[idx][..., None]
        fd = ud[idx][..., None]
        fa = ua[idx][..., None]
        fs = us[idx][..., None]
        lt = jnp.mean((fused - ft) ** 2)
        lb = jnp.mean((o0 - fd) ** 2) + jnp.mean((o1 - fa) ** 2) + jnp.mean((o2 - fs) ** 2)
        return lt + lambda_branch * lb, (lt, lb)

    @jax.jit
    def step(p, st, idx):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, idx)
        upd, st = tx.update(grads, st, p)
        p = optax.apply_updates(p, upd)
        return p, st, loss, aux

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(tr)
        losses = []
        for s in range(0, len(tr), batch):
            bi = tr[s : s + batch]
            if bi.shape[0] < 2:
                continue
            params, opt_state, loss, _ = step(params, opt_state, jnp.asarray(bi))
            losses.append(float(loss))
        if ep % max(epochs // 6, 1) == 0 or ep == epochs - 1:
            lt, (ltt, lbb) = loss_fn(params, jnp.asarray(va))
            print(
                f"  [mb] ep={ep:3d} train={float(np.mean(losses)):.4e} "
                f"val_tot={float(ltt):.4e} val_br={float(lbb):.4e}"
            )

    te_idx = jnp.asarray(te)
    lt_te, (ltt_te, lbb_te) = loss_fn(params, te_idx)
    pred_f, _, _, _ = forward_multibranch(f[te_idx], edges, params["b0"], params["b1"], params["b2"], params["fuse"], n_mp, hid)
    mse = float(jnp.mean((pred_f.squeeze(-1) - ut[te_idx]) ** 2))
    return params, {"mse_ut_test": mse, "loss_tot_te": float(ltt_te), "loss_br_te": float(lbb_te)}


def train_z(
    data: dict,
    epochs: int,
    batch: int,
    lr: float,
    beta_star: float,
    n_mp: int,
    hid: int,
    seed: int,
):
    a = jnp.asarray(data["A"], dtype=jnp.float64)
    edges = jnp.asarray(data["edges"], dtype=jnp.int32)
    n_s, n = data["f"].shape
    rng = np.random.default_rng(seed + 404)
    r_all = rng.standard_normal((n_s, n))
    z_star = np.linalg.solve(np.asarray(data["A"]), r_all.T).T
    r_j = jnp.asarray(r_all, dtype=jnp.float64)
    zs_j = jnp.asarray(z_star, dtype=jnp.float64)
    tr, va, te = _split(n_s, seed)

    key = jax.random.PRNGKey(seed + 77)
    params = init_branch(key, 2, hid, n_mp)

    def loss_fn(p, idx):
        rb = r_j[idx]
        zb = forward_z(rb, edges, p, n_mp, hid)
        zt = zs_j[idx][..., None]
        res = jnp.einsum("ij,bj->bi", a, zb.squeeze(-1)) - rb
        l_res = jnp.mean(res**2)
        l_sup = jnp.mean((zb - zt) ** 2)
        return l_res + beta_star * l_sup, (l_res, l_sup)

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(p, st, idx):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, idx)
        upd, st = tx.update(grads, st, p)
        p = optax.apply_updates(p, upd)
        return p, st, loss

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(tr)
        losses = []
        for s in range(0, len(tr), batch):
            bi = tr[s : s + batch]
            if bi.shape[0] < 2:
                continue
            params, opt_state, loss = step(params, opt_state, jnp.asarray(bi))
            losses.append(float(loss))
        if ep % max(epochs // 6, 1) == 0 or ep == epochs - 1:
            _lv, (_lr, _ls) = loss_fn(params, jnp.asarray(va))
            print(f"  [z]  ep={ep:3d} train={float(np.mean(losses)):.4e} val={float(_lv):.4e}")

    te_idx = jnp.asarray(te)
    l_te, _aux = loss_fn(params, te_idx)
    zhat = forward_z(r_j[te_idx], edges, params, n_mp, hid).squeeze(-1)
    mse = float(jnp.mean((zhat - zs_j[te_idx]) ** 2))
    return params, {"mse_z_test": mse, "loss_te": float(l_te)}


def gmres_mean_matvec(
    a_np: np.ndarray,
    z_params: dict,
    f_samples: np.ndarray,
    edges: jnp.ndarray,
    n_mp: int,
    hid: int,
    n_trials: int,
    rtol: float,
    maxiter: int,
) -> tuple[float, float]:
    """`x0 = z(f)` from trained z-model."""
    n = a_np.shape[0]
    counts_z = []
    counts_0 = []
    rng = np.random.default_rng(12345)

    def apply_z(fv):
        z = forward_z(jnp.asarray(fv[None, :], dtype=jnp.float64), edges, z_params, n_mp, hid)
        return np.asarray(z.squeeze(0).squeeze(-1), dtype=np.float64)

    for _ in range(n_trials):
        f = rng.standard_normal(n)
        x0 = apply_z(f)

        def mv(v):
            mv.count += 1
            return a_np @ v

        mv.count = 0
        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        gmres(op, f, x0=x0.copy(), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
        counts_z.append(mv.count)
        mv.count = 0
        gmres(op, f, x0=np.zeros(n), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
        counts_0.append(mv.count)

    return float(np.mean(counts_z)), float(np.mean(counts_0))


def gmres_mean_matvec_multibranch(
    a_np: np.ndarray,
    mb_params: dict,
    f_samples: np.ndarray,
    edges: jnp.ndarray,
    n_mp: int,
    hid: int,
    n_trials: int,
    rtol: float,
    maxiter: int,
) -> tuple[float, float]:
    """`x0 = fused(f)` as initial guess for `A u = f`."""

    def apply_u(fv):
        fu, _, _, _ = forward_multibranch(
            jnp.asarray(fv[None, :], dtype=jnp.float64),
            edges,
            mb_params["b0"],
            mb_params["b1"],
            mb_params["b2"],
            mb_params["fuse"],
            n_mp,
            hid,
        )
        return np.asarray(fu.squeeze(0).squeeze(-1), dtype=np.float64)

    n = a_np.shape[0]
    counts_m = []
    counts_0 = []
    rng = np.random.default_rng(54321)
    for _ in range(n_trials):
        f = rng.standard_normal(n)
        x0 = apply_u(f)

        def mv(v):
            mv.count += 1
            return a_np @ v

        mv.count = 0
        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        gmres(op, f, x0=x0.copy(), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
        counts_m.append(mv.count)
        mv.count = 0
        gmres(op, f, x0=np.zeros(n), rtol=rtol, atol=0.0, maxiter=maxiter, restart=min(50, maxiter))
        counts_0.append(mv.count)

    return float(np.mean(counts_m)), float(np.mean(counts_0))


def run_one_npz(path: Path, args) -> dict:
    d = np.load(path)
    name = path.stem
    mesh = str(np.asarray(d["mesh_kind"]).reshape(-1)[0])
    if isinstance(mesh, bytes):
        mesh = mesh.decode()
    n = int(np.asarray(d["n_nodes"]).reshape(-1)[0])
    print(f"\n=== {name} mesh={mesh} n={n} ===")
    out = {"name": name, "mesh": mesh, "n": n}

    if args.train_multibranch:
        mb_p, met = train_multibranch(
            {k: d[k] for k in d.files},
            epochs=args.epochs,
            batch=args.batch_size,
            lr=args.lr,
            lambda_branch=args.lambda_branch,
            n_mp=args.n_mp_layers,
            hid=args.hidden,
            seed=args.seed,
        )
        out.update({f"mb_{k}": v for k, v in met.items()})
        if args.gmres_trials > 0:
            mz, m0 = gmres_mean_matvec_multibranch(
                np.asarray(d["A"], dtype=np.float64),
                mb_p,
                d["f"],
                jnp.asarray(d["edges"], dtype=jnp.int32),
                args.n_mp_layers,
                args.hidden,
                args.gmres_trials,
                args.rtol,
                args.gmres_maxiter,
            )
            out["gmres_mb"] = mz
            out["gmres_baseline"] = m0

    if args.train_z:
        z_p, met = train_z(
            {k: d[k] for k in d.files},
            epochs=args.epochs,
            batch=args.batch_size,
            lr=args.lr_z,
            beta_star=args.beta_star,
            n_mp=args.n_mp_layers,
            hid=args.hidden,
            seed=args.seed,
        )
        out.update({f"z_{k}": v for k, v in met.items()})
        if args.gmres_trials > 0:
            zz, z0 = gmres_mean_matvec(
                np.asarray(d["A"], dtype=np.float64),
                z_p,
                d["f"],
                jnp.asarray(d["edges"], dtype=jnp.int32),
                args.n_mp_layers,
                args.hidden,
                args.gmres_trials,
                args.rtol,
                args.gmres_maxiter,
            )
            out["gmres_z"] = zz
            if "gmres_baseline" not in out:
                out["gmres_baseline"] = z0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("outputs/gnn_blueprint/dataset_2d_mg"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--lr-z", type=float, default=0.01)
    ap.add_argument("--lambda-branch", type=float, default=0.35)
    ap.add_argument("--beta-star", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-mp-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gmres-trials", type=int, default=40)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--gmres-maxiter", type=int, default=800)
    ap.add_argument("--no-multibranch", action="store_true")
    ap.add_argument("--no-z", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Only these dataset basenames (e.g. struct_s tri_m), without .npz",
    )
    args = ap.parse_args()

    root = Path.cwd()
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    if args.only:
        files = []
        for name in args.only:
            fn = name if name.endswith(".npz") else f"{name}.npz"
            p = data_dir / fn
            if not p.is_file():
                raise SystemExit(f"missing {p}")
            files.append(p)
    else:
        files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No npz in {data_dir}; run dataset_2d_mg_multibranch.py first")

    args.train_multibranch = not args.no_multibranch
    args.train_z = not args.no_z
    rows = []
    for p in files:
        rows.append(run_one_npz(p, args))

    print("\n======== SUMMARY ========")
    keys = ["name", "mesh", "n", "mb_mse_ut_test", "z_mse_z_test", "gmres_baseline", "gmres_mb", "gmres_z"]
    for r in rows:
        line = " ".join(
            f"{k}={r.get(k, float('nan')):.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k, '')}"
            for k in keys
            if k in r or k in ("name", "mesh", "n")
        )
        # cleaner print
        parts = [f"{r['name']}", f"n={r['n']}"]
        if "mb_mse_ut_test" in r:
            parts.append(f"mb_mse={r['mb_mse_ut_test']:.4e}")
        if "z_mse_z_test" in r:
            parts.append(f"z_mse={r['z_mse_z_test']:.4e}")
        if "gmres_baseline" in r:
            parts.append(f"gmres0={r['gmres_baseline']:.1f}")
        if "gmres_mb" in r:
            parts.append(f"gmres_mb={r['gmres_mb']:.1f}")
        if "gmres_z" in r:
            parts.append(f"gmres_z={r['gmres_z']:.1f}")
        print(" | ".join(parts))

    if args.out_json:
        outp = args.out_json if args.out_json.is_absolute() else root / args.out_json
        outp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()
