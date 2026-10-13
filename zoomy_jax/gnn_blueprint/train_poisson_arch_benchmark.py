"""Train Poisson 1D predictors for architecture ablation (multilevel / FFT / smooth inputs).

Supervised target: first channel predicts interior solution `u` (`u_init=0`).
Optional loss term: discrete Poisson residual `||A u_pred - f||^2`.

Saves `weights_deltaq.npz` per architecture for `benchmark_poisson_iterations.run_gmres_benchmark`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

try:
    from zoomy_jax.gnn_blueprint import global_coupling as gc
    from zoomy_jax.gnn_blueprint.fourier_1d import max_rfft_modes
    from zoomy_jax.gnn_blueprint.poisson_1d import laplacian_1d_dense
    from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import global_coupling as gc
    from fourier_1d import max_rfft_modes
    from poisson_1d import laplacian_1d_dense
    from predictor_learned_multilevel import predict_delta_q_learned


def _split_indices(n, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_val], idx[n_tr + n_val :]


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
    """Channels: `u(0), f, su1, sf1, su2, sf2, ...`."""
    n = f_row.shape[0]
    u = np.zeros(n, dtype=np.float64)
    rows = [u, f_row.copy()]
    for r in radii:
        rows.append(_box_smooth_np(u, int(r)))
        rows.append(_box_smooth_np(f_row, int(r)))
    return np.stack(rows, axis=0)


def _make_dataset(n_samples: int, n: int, radii: tuple[int, ...], seed: int):
    rng = np.random.default_rng(seed)
    a_np = np.asarray(np.array(laplacian_1d_dense(n)), dtype=np.float64)
    f_all = rng.standard_normal((n_samples, n))
    u_all = np.linalg.solve(a_np, f_all.T).T
    q_list = []
    dq_list = []
    for i in range(n_samples):
        q = _build_q_np(f_all[i], radii)
        dq = np.zeros_like(q)
        dq[0] = u_all[i]
        q_list.append(q)
        dq_list.append(dq)
    q = np.stack(q_list, axis=0)
    dq = np.stack(dq_list, axis=0)
    cls = np.zeros((n_samples, n), dtype=np.float64)
    dt = np.ones((n_samples,), dtype=np.float64)
    qaux = np.stack([f_all[i][None, :] for i in range(n_samples)], axis=0)
    return q, dq, cls, dt, qaux, a_np


def _init_state(key, n_fields: int, n_cells: int, use_fft: bool):
    k0, k1 = jax.random.split(key)
    m = max_rfft_modes(n_cells)
    state = {
        "w_self": jnp.linspace(0.05, 0.09, n_fields),
        "w_msg": jnp.linspace(0.02, 0.04, n_fields),
        "w_aux": jnp.linspace(0.015, 0.025, n_fields),
        "w_coarse": jnp.linspace(0.02, 0.05, n_fields),
        "w_gate": jnp.linspace(0.8, 1.2, n_fields),
        "b": jnp.linspace(0.005, 0.008, n_fields),
    }
    if use_fft:
        state["fft_w_r"] = jnp.ones((n_fields, m)) + 0.01 * jax.random.normal(k0, (n_fields, m))
        state["fft_w_i"] = 0.01 * jax.random.normal(k1, (n_fields, m))
        state["fft_blend_logit"] = jnp.full((n_fields,), -1.0)
    return state


def _params_from_state(
    state: dict,
    n_fields: int,
    n_cells: int,
    flow_mode: str,
    message_steps: int,
    inner_iters: int,
    coarsen_levels: int,
    use_fft: bool,
    single_layer: int,
):
    p = {
        "w_self": state["w_self"],
        "w_msg": state["w_msg"],
        "w_aux": state["w_aux"],
        "w_coarse": state["w_coarse"],
        "w_gate": state["w_gate"],
        "b": state["b"],
        "message_steps": message_steps,
        "inner_iters": inner_iters,
        "coarsen_levels": coarsen_levels,
        "flow_mode": flow_mode,
        "global_coupling_mode": gc.FFT_1D if use_fft else gc.MULTIGRID,
        "single_layer_mode": int(single_layer),
    }
    if use_fft:
        p["fft_w_r"] = state["fft_w_r"]
        p["fft_w_i"] = state["fft_w_i"]
        p["fft_blend_logit"] = state["fft_blend_logit"]
    return p


def train_one_arch(
    name: str,
    n: int,
    n_samples: int,
    epochs: int,
    lr: float,
    beta_res: float,
    flow_mode: str,
    message_steps: int,
    inner_iters: int,
    coarsen_levels: int,
    use_fft: bool,
    single_layer: int,
    radii: tuple[int, ...],
    out_dir: Path,
    seed: int,
    batch_size: int = 64,
    patience: int = 60,
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    diverge_factor: float = 5e3,
    no_early_stop: bool = False,
):
    n_fields = 2 + 2 * len(radii)
    q, dq, cls, dt, qaux, a_np = _make_dataset(n_samples, n, radii, seed)
    a_mat = jnp.asarray(a_np)
    tr, va, te = _split_indices(n_samples, seed=seed)

    qn = jnp.asarray(q)
    dqn = jnp.asarray(dq)
    cls_j = jnp.asarray(cls)
    dt_j = jnp.asarray(dt)
    qax_j = jnp.asarray(qaux)

    key = jax.random.PRNGKey(seed + 17)
    state = _init_state(key, n_fields, n, use_fft)
    tr_list = np.asarray(tr, dtype=np.int32)
    n_tr = tr_list.shape[0]
    bs = int(max(min(batch_size, n_tr), 1))
    steps_per_epoch = max(n_tr // bs, 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    warm_steps = min(int(warmup_epochs) * steps_per_epoch, max(total_steps // 4, 1))
    decay_steps = max(total_steps - warm_steps, 1)
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(0.0, float(lr), warm_steps),
            optax.cosine_decay_schedule(float(lr), decay_steps, alpha=0.05),
        ],
        [warm_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(float(grad_clip)),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = tx.init(state)

    def loss_fn(st, idx):
        p = _params_from_state(
            st,
            n_fields,
            n,
            flow_mode,
            message_steps,
            inner_iters,
            coarsen_levels,
            use_fft,
            single_layer,
        )
        pred = jax.vmap(
            lambda qq, qax, dti, clsi: predict_delta_q_learned(
                qq, qax, dti, clsi, p, message_steps, return_diagnostics=False
            )
        )(qn[idx], qax_j[idx], dt_j[idx], cls_j[idx])
        tgt = dqn[idx]
        l_sup = jnp.mean((pred[:, 0] - tgt[:, 0]) ** 2)
        u_pred = pred[:, 0]
        f_b = qn[idx, 1]
        res = jax.vmap(lambda up, fv: a_mat @ up - fv)(u_pred, f_b)
        l_r = jnp.mean(res**2)
        return l_sup + beta_res * l_r, (l_sup, l_r)

    @jax.jit
    def step(st, opt_s, idx):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(st, idx)
        updates, opt_s = tx.update(grads, opt_s, st)
        st = optax.apply_updates(st, updates)
        return st, opt_s, loss, aux

    tr_idx = jnp.asarray(tr)
    va_idx = jnp.asarray(va)
    te_idx = jnp.asarray(te)

    init_tr, _ = loss_fn(state, tr_idx)
    init_tr_f = float(init_tr)
    rng_ep = np.random.default_rng(seed + 901)

    best_state = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
    best_val = float("inf")
    patience_left = int(patience)
    stopped = "max_epochs"
    diverged = False

    for ep in range(epochs):
        perm = tr_list.copy()
        rng_ep.shuffle(perm)
        ep_losses = []
        ep_sup = []
        ep_res = []
        for start in range(0, n_tr, bs):
            batch = perm[start : start + bs]
            if batch.shape[0] < bs:
                continue
            bidx = jnp.asarray(batch)
            state, opt_state, tr_loss, (a0, a1) = step(state, opt_state, bidx)
            ep_losses.append(float(tr_loss))
            ep_sup.append(float(a0))
            ep_res.append(float(a1))
        tr_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        va_loss, (va_sup, va_res) = loss_fn(state, va_idx)
        va_f = float(va_loss)

        if (not np.isfinite(tr_mean)) or tr_mean > diverge_factor * max(init_tr_f, 1e-30):
            stopped = "diverged"
            diverged = True
            print(f"[{name}] abort: non-finite or diverging train loss (epoch {ep})")
            state = best_state
            break

        improved = va_f < best_val - 1e-12
        if improved:
            best_val = va_f
            best_state = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
            patience_left = int(patience)
        elif not no_early_stop:
            patience_left -= 1

        if ep % 20 == 0 or ep == epochs - 1:
            print(
                f"[{name}] epoch={ep:4d} train={tr_mean:.6e} val={va_f:.6e} "
                f"sup={float(va_sup):.3e} res={float(va_res):.3e} best_val={best_val:.6e} pat={patience_left}"
            )

        if (not no_early_stop) and patience_left <= 0 and ep >= warmup_epochs:
            stopped = "early_stop"
            state = best_state
            print(f"[{name}] early stop at epoch {ep} (best val={best_val:.6e})")
            break

    if stopped == "max_epochs":
        state = best_state

    final_te, _ = loss_fn(state, te_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kw = {
        "w_self": np.asarray(state["w_self"]),
        "w_msg": np.asarray(state["w_msg"]),
        "w_aux": np.asarray(state["w_aux"]),
        "w_coarse": np.asarray(state["w_coarse"]),
        "w_gate": np.asarray(state["w_gate"]),
        "b": np.asarray(state["b"]),
        "variant": np.asarray([f"poisson_{name}"]),
        "flow_mode": np.asarray([flow_mode]),
        "message_steps": np.asarray([message_steps], dtype=int),
        "inner_iters": np.asarray([inner_iters], dtype=int),
        "coarsen_levels": np.asarray([coarsen_levels], dtype=int),
        "global_coupling_mode": np.asarray([gc.FFT_1D if use_fft else gc.MULTIGRID], dtype=int),
        "single_layer_mode": np.asarray([single_layer], dtype=int),
        "smooth_radii": np.asarray(radii, dtype=int) if radii else np.zeros((0,), dtype=int),
        "n_cells_poisson": np.asarray([n], dtype=int),
        "test_loss": np.asarray([float(final_te)]),
    }
    if use_fft:
        mm = max_rfft_modes(n)
        save_kw["fft_w_r"] = np.asarray(state["fft_w_r"])
        save_kw["fft_w_i"] = np.asarray(state["fft_w_i"])
        save_kw["fft_blend_logit"] = np.asarray(state["fft_blend_logit"])
        save_kw["max_fft_modes"] = np.asarray([mm], dtype=int)
    np.savez_compressed(out_dir / "weights_deltaq.npz", **save_kw)
    summary = (
        f"name={name}\n"
        f"stopped={stopped}\ndiverged={diverged}\n"
        f"best_val_loss={best_val}\ntest_loss={float(final_te)}\n"
        f"batch_size={bs} steps_per_epoch={steps_per_epoch} patience={patience}\n"
    )
    (out_dir / "training_summary.txt").write_text(summary, encoding="utf-8")
    print(f"[{name}] saved {out_dir / 'weights_deltaq.npz'} test_loss={float(final_te):.6e} ({stopped})")
    return float(final_te)


ARCH_PRESETS: dict[str, dict] = {
    "classic_ml": {
        "single_layer": 0,
        "use_fft": False,
        "radii": (),
        "message_steps": 3,
        "coarsen_levels": 3,
        "inner_iters": 1,
    },
    "classic_ml_fft": {
        "single_layer": 0,
        "use_fft": True,
        "radii": (),
        "message_steps": 3,
        "coarsen_levels": 3,
        "inner_iters": 1,
    },
    "single_layer": {
        "single_layer": 1,
        "use_fft": False,
        "radii": (),
        "message_steps": 2,
        "coarsen_levels": 1,
        "inner_iters": 1,
    },
    "single_layer_smooth": {
        "single_layer": 1,
        "use_fft": False,
        "radii": (2, 6),
        "message_steps": 2,
        "coarsen_levels": 1,
        "inner_iters": 1,
    },
    "ml_smooth": {
        "single_layer": 0,
        "use_fft": False,
        "radii": (2, 6),
        "message_steps": 3,
        "coarsen_levels": 3,
        "inner_iters": 1,
    },
}


def main():
    p = argparse.ArgumentParser(description="Train Poisson 1D architecture ablation")
    p.add_argument("--out-root", type=Path, default=Path("outputs/gnn_blueprint/poisson_arch"))
    p.add_argument("--n-cells", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400, help="Max epochs (early stopping may finish sooner)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=70, help="Stop if val loss does not improve for this many epochs")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--diverge-factor", type=float, default=5000.0, help="Abort if train loss exceeds this x initial")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--beta-res", type=float, default=0.25)
    p.add_argument("--flow-mode", type=str, default="bidir")
    p.add_argument("--arch", type=str, default="all", help="all or one of: " + ",".join(ARCH_PRESETS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path.cwd()
    out_root = args.out_root if args.out_root.is_absolute() else root / args.out_root
    names = list(ARCH_PRESETS.keys()) if args.arch == "all" else [args.arch]
    for name in names:
        if name not in ARCH_PRESETS:
            raise SystemExit(f"Unknown arch {name}")
        cfg = ARCH_PRESETS[name]
        train_one_arch(
            name=name,
            n=args.n_cells,
            n_samples=args.n_samples,
            epochs=args.epochs,
            lr=args.lr,
            beta_res=args.beta_res,
            flow_mode=args.flow_mode,
            message_steps=cfg["message_steps"],
            inner_iters=cfg["inner_iters"],
            coarsen_levels=cfg["coarsen_levels"],
            use_fft=cfg["use_fft"],
            single_layer=cfg["single_layer"],
            radii=cfg["radii"],
            out_dir=out_root / name,
            seed=args.seed,
            batch_size=args.batch_size,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            grad_clip=args.grad_clip,
            diverge_factor=args.diverge_factor,
            no_early_stop=args.no_early_stop,
        )


if __name__ == "__main__":
    main()
