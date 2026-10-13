"""Train structured V-cycle GNN on 2D Poisson (interior); benchmark GMRES with x0 = one V-cycle.

Requires even interior size (nx-2, ny-2 both even). Uses strict 2x2 coarsening and
Jacobi iterations on the coarsest level (no unstructured meshes).

With `--bump-amplitude > 0`, smoothers see a fourth input channel: a Gaussian bump on the
2D structured interior, restricted to each multigrid level (same `R` as the residual). The
RHS is modulated by that bump so supervision is not independent of bathymetry-like geometry.
This is the V-cycle architecture used for plain Poisson, extended for a 2D bump; wiring into
full Green–Naghdi IMEX waits on a 2D GN JAX residual in this repo.

Example:

```
    python -m zoomy_jax.gnn_blueprint.train_vcycle_structured_poisson --nx 34 --ny 34 \\
        --epochs 80 --gmres-trials 24
```

Bump-conditioned V-cycle:

```
    python -m zoomy_jax.gnn_blueprint.train_vcycle_structured_poisson --nx 34 --ny 34 \\
        --bump-amplitude 0.12 --bump-sigma 0.15 --rhs-bump-scale 0.5
```
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

try:
    from scipy.sparse.linalg import LinearOperator, gmres
except ImportError as e:
    raise SystemExit("scipy required") from e

from zoomy_jax.gnn_blueprint.mesh_2d_poisson import gaussian_bump_interior
from zoomy_jax.gnn_blueprint.mg_structured_hierarchy import (
    build_poisson_hierarchy,
    restrict_field_to_coarser_levels,
)
from zoomy_jax.gnn_blueprint.vcycle_structured_gnn import (
    forward_vcycle,
    forward_vcycle_batch,
    init_vcycle_smoothers,
)


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ix = np.arange(n)
    rng.shuffle(ix)
    n_tr = int(0.75 * n)
    n_va = int(0.1 * n)
    return ix[:n_tr], ix[n_tr : n_tr + n_va], ix[n_tr + n_va :]


def gmres_benchmark_vcycle(
    apply_x0: Callable[[np.ndarray], np.ndarray],
    a_np: np.ndarray,
    n_trials: int,
    rtol: float,
    maxiter: int,
    n_warmup: int,
) -> dict[str, Any]:
    """GMRES stats: matvecs, SciPy `pr_norm` callback counts, and wall times.

    `iter_*` counts residual-norm callbacks from SciPy's GMRES (one per inner step
    in the restarted cycle, up to one matvec each; totals track solver progress).
    """
    n = a_np.shape[0]
    restart = min(50, maxiter)

    rng_w = np.random.default_rng(4242)
    for _ in range(max(0, n_warmup)):
        apply_x0(rng_w.standard_normal(n))

    mat_v: list[int] = []
    mat_0: list[int] = []
    it_v: list[int] = []
    it_0: list[int] = []
    t_x0: list[float] = []
    t_gmres_v: list[float] = []
    t_gmres_0: list[float] = []
    rng = np.random.default_rng(2026)
    for _ in range(n_trials):
        f = rng.standard_normal(n)

        t_a = time.perf_counter()
        x0 = apply_x0(f)
        t_b = time.perf_counter()
        t_x0.append(t_b - t_a)

        def mv(v):
            mv.count += 1
            return a_np @ v

        cb_v = [0]

        def on_pr_v(_: float) -> None:
            cb_v[0] += 1

        mv.count = 0
        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        t0 = time.perf_counter()
        gmres(
            op,
            f,
            x0=x0.copy(),
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            restart=restart,
            callback=on_pr_v,
            callback_type="pr_norm",
        )
        t1 = time.perf_counter()
        mat_v.append(mv.count)
        it_v.append(cb_v[0])
        t_gmres_v.append(t1 - t0)

        cb0 = [0]

        def on_pr_0(_: float) -> None:
            cb0[0] += 1

        mv.count = 0
        t2 = time.perf_counter()
        gmres(
            op,
            f,
            x0=np.zeros(n),
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            restart=restart,
            callback=on_pr_0,
            callback_type="pr_norm",
        )
        t3 = time.perf_counter()
        mat_0.append(mv.count)
        it_0.append(cb0[0])
        t_gmres_0.append(t3 - t2)

    mean_x0 = float(np.mean(t_x0))
    mean_gv = float(np.mean(t_gmres_v))
    mean_g0 = float(np.mean(t_gmres_0))
    return {
        "matvec_mean_v": float(np.mean(mat_v)),
        "matvec_mean_0": float(np.mean(mat_0)),
        "iter_mean_v": float(np.mean(it_v)),
        "iter_mean_0": float(np.mean(it_0)),
        "sec_x0_build_mean": mean_x0,
        "sec_gmres_mean_v": mean_gv,
        "sec_gmres_mean_0": mean_g0,
        "sec_total_mean_v": mean_x0 + mean_gv,
        "sec_total_mean_0": mean_g0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=34)
    ap.add_argument("--ny", type=int, default=34)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-mp-layers", type=int, default=3)
    ap.add_argument("--nu1", type=int, default=1)
    ap.add_argument("--nu2", type=int, default=1)
    ap.add_argument("--coarsest-iters", type=int, default=40)
    ap.add_argument("--coarsest-omega", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gmres-trials", type=int, default=32)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--gmres-maxiter", type=int, default=800)
    ap.add_argument(
        "--no-jit-x0",
        action="store_true",
        help="Disable jax.jit on single-vector V-cycle when forming GMRES x0 (eager).",
    )
    ap.add_argument(
        "--benchmark-warmup",
        type=int,
        default=3,
        help="Number of untimed apply_x0 calls before GMRES timing (JIT compile + cache).",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--bump-amplitude",
        type=float,
        default=0.0,
        help="If >0, train 4-channel smoothers with this Gaussian bump height on the interior grid.",
    )
    ap.add_argument(
        "--bump-sigma",
        type=float,
        default=0.15,
        help="Gaussian bump width in normalized (0,1)^2 cell coordinates.",
    )
    ap.add_argument(
        "--rhs-bump-scale",
        type=float,
        default=0.5,
        help="When bump is on: f = z * (1 + rhs_bump_scale * b) with z ~ N(0,1) per node.",
    )
    args = ap.parse_args()

    a_list_np, r_list_np, p_list_np, edges_list_np, shapes = build_poisson_hierarchy(args.nx, args.ny)
    n = a_list_np[0].shape[0]
    n_levels = len(a_list_np)
    print(f"structured V-cycle: n={n} levels={n_levels} shapes={shapes}")
    if args.bump_amplitude > 0.0:
        print(
            f"  bump mode: amplitude={args.bump_amplitude} sigma={args.bump_sigma} "
            f"rhs_bump_scale={args.rhs_bump_scale} (4-channel smoothers)"
        )

    a_np = a_list_np[0]
    a_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in a_list_np)
    r_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in r_list_np)
    p_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in p_list_np)
    edges_list = tuple(jnp.asarray(e, dtype=jnp.int32) for e in edges_list_np)

    use_bump = args.bump_amplitude > 0.0
    if use_bump:
        b_fine_np = gaussian_bump_interior(
            args.nx,
            args.ny,
            amplitude=args.bump_amplitude,
            sigma=args.bump_sigma,
        )
        b_per_level_np = restrict_field_to_coarser_levels(b_fine_np, list(r_list_np))
        n_in = 4
    else:
        b_fine_np = np.zeros(n, dtype=np.float64)
        b_per_level_np = [np.zeros(a_list_np[i].shape[0], dtype=np.float64) for i in range(n_levels)]
        n_in = 3
    b_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in b_per_level_np)

    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((args.n_samples, n))
    if use_bump:
        f_all = z * (1.0 + args.rhs_bump_scale * b_fine_np.reshape(1, -1))
    else:
        f_all = z
    u_star = np.linalg.solve(a_np, f_all.T).T

    tr, va, te = _split(args.n_samples, args.seed)
    f_j = jnp.asarray(f_all, dtype=jnp.float64)
    u_j = jnp.asarray(u_star, dtype=jnp.float64)

    key = jax.random.PRNGKey(args.seed + 11)
    params = init_vcycle_smoothers(
        key, n_levels, args.hidden, args.n_mp_layers, n_in=n_in
    )
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    def loss_fn(p, idx):
        fb = f_j[idx]
        ub = forward_vcycle_batch(
            fb,
            p,
            a_list,
            r_list,
            p_list,
            edges_list,
            b_list,
            args.n_mp_layers,
            args.hidden,
            args.nu1,
            args.nu2,
            args.coarsest_iters,
            args.coarsest_omega,
        )
        ut = u_j[idx]
        return jnp.mean((ub - ut) ** 2)

    @jax.jit
    def step(p, st, idx):
        loss, grads = jax.value_and_grad(loss_fn)(p, idx)
        upd, st = tx.update(grads, st, p)
        p = optax.apply_updates(p, upd)
        return p, st, loss

    for ep in range(args.epochs):
        rng_ep = np.random.default_rng(args.seed + ep)
        rng_ep.shuffle(tr)
        losses = []
        for s in range(0, len(tr), args.batch_size):
            bi = tr[s : s + args.batch_size]
            if bi.shape[0] < 2:
                continue
            params, opt_state, loss = step(params, opt_state, jnp.asarray(bi))
            losses.append(float(loss))
        if ep % max(args.epochs // 6, 1) == 0 or ep == args.epochs - 1:
            lv = float(loss_fn(params, jnp.asarray(va)))
            print(f"  ep={ep:3d} train_mse={float(np.mean(losses)):.4e} val_mse={lv:.4e}")

    te_idx = jnp.asarray(te)
    mse_te = float(loss_fn(params, te_idx))
    print(f"test MSE (supervised u*): {mse_te:.4e}")

    bench: dict[str, Any] | None = None
    if args.gmres_trials > 0:
        use_jit_x0 = not args.no_jit_x0
        if use_jit_x0:

            def _vcycle_infer(f_jnp: jnp.ndarray, p: dict[str, Any]) -> jnp.ndarray:
                return forward_vcycle(
                    f_jnp,
                    p,
                    a_list,
                    r_list,
                    p_list,
                    edges_list,
                    b_list,
                    args.n_mp_layers,
                    args.hidden,
                    args.nu1,
                    args.nu2,
                    args.coarsest_iters,
                    args.coarsest_omega,
                )

            vcycle_jit = jax.jit(_vcycle_infer)

            def apply_x0(fv: np.ndarray) -> np.ndarray:
                out = vcycle_jit(jnp.asarray(fv, dtype=jnp.float64), params)
                return np.asarray(out, dtype=np.float64)

            print(f"GMRES x0: jax.jit V-cycle (warmup runs={args.benchmark_warmup})")
        else:

            def apply_x0(fv: np.ndarray) -> np.ndarray:
                u_hat = forward_vcycle(
                    jnp.asarray(fv, dtype=jnp.float64),
                    params,
                    a_list,
                    r_list,
                    p_list,
                    edges_list,
                    b_list,
                    args.n_mp_layers,
                    args.hidden,
                    args.nu1,
                    args.nu2,
                    args.coarsest_iters,
                    args.coarsest_omega,
                )
                return np.asarray(u_hat, dtype=np.float64)

            print("GMRES x0: eager V-cycle (no jit)")

        bench = gmres_benchmark_vcycle(
            apply_x0,
            a_np,
            args.gmres_trials,
            args.rtol,
            args.gmres_maxiter,
            args.benchmark_warmup,
        )
        bench["jit_x0"] = bool(use_jit_x0)
        mv_v = bench["matvec_mean_v"]
        mv0 = bench["matvec_mean_0"]
        it_v = bench["iter_mean_v"]
        it0 = bench["iter_mean_0"]
        print(
            f"GMRES mean matvec: x0=V-cycle {mv_v:.1f}  x0=0 {mv0:.1f}  "
            f"(reduction {mv0 - mv_v:+.1f}, "
            f"{(1.0 - mv_v / mv0) * 100.0 if mv0 > 0 else float('nan'):.1f}% fewer)"
        )
        print(
            f"GMRES mean pr_norm steps: x0=V-cycle {it_v:.1f}  x0=0 {it0:.1f}  "
            f"(reduction {it0 - it_v:+.1f}, "
            f"{(1.0 - it_v / it0) * 100.0 if it0 > 0 else float('nan'):.1f}% fewer)"
        )
        tot_v = bench["sec_total_mean_v"]
        tot0 = bench["sec_total_mean_0"]
        gv = bench["sec_gmres_mean_v"]
        g0 = bench["sec_gmres_mean_0"]
        pct_tot = (1.0 - tot_v / tot0) * 100.0 if tot0 > 0 else float("nan")
        pct_g = (1.0 - gv / g0) * 100.0 if g0 > 0 else float("nan")
        print(
            f"Wall clock mean GMRES only: with x0 {gv * 1000.0:.2f} ms  zero x0 {g0 * 1000.0:.2f} ms  "
            f"({pct_g:+.1f}% vs zero-x0 GMRES; positive => faster Krylov phase)"
        )
        print(
            f"Wall clock mean end-to-end (x0 build + GMRES vs GMRES only): "
            f"V-cycle path {tot_v * 1000.0:.2f} ms  zero-x0 path {tot0 * 1000.0:.2f} ms  "
            f"({pct_tot:+.1f}% vs zero-x0 total; often negative if x0 build is expensive)"
        )
        print(
            f"  (breakdown: x0 build {bench['sec_x0_build_mean'] * 1000.0:.2f} ms, "
            f"GMRES with x0 {gv * 1000.0:.2f} ms, GMRES zero x0 {g0 * 1000.0:.2f} ms)"
        )

    if args.out_json:
        root = Path.cwd()
        outp = args.out_json if args.out_json.is_absolute() else root / args.out_json
        row = {
            "nx": args.nx,
            "ny": args.ny,
            "n": n,
            "n_levels": n_levels,
            "shapes": [list(t) for t in shapes],
            "mse_te": mse_te,
            "bump_amplitude": args.bump_amplitude,
            "bump_sigma": args.bump_sigma,
            "rhs_bump_scale": args.rhs_bump_scale if use_bump else None,
            "smoother_n_in": n_in,
        }
        if args.gmres_trials > 0 and bench is not None:
            row["gmres_benchmark"] = bench
        outp.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()
