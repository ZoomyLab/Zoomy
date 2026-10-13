"""Train multilevel learned `deltaQ` with optional 1D FFT coupling + physics losses.

Loss = supervised MSE + `beta_impl` * implicit residual proxy + `beta_pois` *
discrete Laplacian penalty on a pressure proxy (depth channel `index 1`).

Writes `weights_deltaq.npz` for the learned predictor
when `global_coupling_mode=FFT_1D` (see `zoomy_jax.gnn_blueprint.global_coupling`).
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
    from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import global_coupling as gc
    from fourier_1d import max_rfft_modes
    from predictor_learned_multilevel import predict_delta_q_learned


def _split_indices(n, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_val], idx[n_tr + n_val :]


def _normalize(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    if std is None:
        std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def _lap1d_fields(u):
    left = jnp.pad(u[:, :-1], ((0, 0), (1, 0)), mode="edge")
    right = jnp.pad(u[:, 1:], ((0, 0), (0, 1)), mode="edge")
    return left - 2.0 * u + right


def _implicit_residual_proxy(q_old, dq_pred, dt):
    q_guess = q_old + dq_pred
    s = 0.08 * q_guess + 0.04 * _lap1d_fields(q_guess)
    return dq_pred - dt * s


def _poisson_proxy_loss(q_old, dq_pred, pressure_field: int = 1):
    if q_old.shape[0] <= pressure_field:
        return jnp.asarray(0.0, dtype=q_old.dtype)
    h = q_old[pressure_field] + dq_pred[pressure_field]
    lap = _lap1d_fields(h[None, :])[0]
    return jnp.mean(lap**2)


def _params_from_state(state: dict, n_fields: int, n_cells: int, flow_mode: str, message_steps: int, inner_iters: int, coarsen_levels: int, use_fft: bool):
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
    }
    if use_fft:
        p["fft_w_r"] = state["fft_w_r"]
        p["fft_w_i"] = state["fft_w_i"]
        p["fft_blend_logit"] = state["fft_blend_logit"]
    return p


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


def train(
    dataset_path: Path,
    out_dir: Path,
    n_epochs: int,
    lr: float,
    flow_mode: str,
    message_steps: int,
    inner_iters: int,
    coarsen_levels: int,
    use_fft: bool,
    beta_impl: float,
    beta_pois: float,
    seed: int = 42,
):
    data = np.load(dataset_path)
    q = data["q"]
    dt = data["dt"]
    dq = data["delta_q"]
    class_id = (
        data["class_id"]
        if "class_id" in data.files
        else np.zeros((q.shape[0], q.shape[2]), dtype=np.float64)
    )
    qaux = (
        data["qaux"]
        if "qaux" in data.files
        else np.zeros((q.shape[0], 1, q.shape[2]), dtype=np.float64)
    )

    n_samples, n_fields, n_cells = q.shape
    tr, va, te = _split_indices(n_samples, seed=seed)

    _, q_mean, q_std = _normalize(q[tr])
    _, dq_mean, dq_std = _normalize(dq[tr])
    _, qaux_mean, qaux_std = _normalize(qaux[tr])

    qn = jnp.asarray((q - q_mean) / q_std)
    dqn = jnp.asarray((dq - dq_mean) / dq_std)
    qauxn = jnp.asarray((qaux - qaux_mean) / qaux_std)
    dt = jnp.asarray(dt)
    class_id = jnp.asarray(class_id)

    key = jax.random.PRNGKey(seed)
    state = _init_state(key, n_fields, n_cells, use_fft)
    tx = optax.adam(lr)
    opt_state = tx.init(state)

    def loss_fn(st, idx):
        p = _params_from_state(
            st, n_fields, n_cells, flow_mode, message_steps, inner_iters, coarsen_levels, use_fft
        )
        pred = jax.vmap(
            lambda qq, qax, dti, clsi: predict_delta_q_learned(
                qq, qax, dti, clsi, p, message_steps, return_diagnostics=False
            )
        )(qn[idx], qauxn[idx], dt[idx], class_id[idx])
        target = dqn[idx]
        l_sup = jnp.mean((pred - target) ** 2)
        l_impl = jnp.mean(jax.vmap(_implicit_residual_proxy)(qn[idx], pred, dt[idx]) ** 2)
        l_pois = jnp.mean(jax.vmap(_poisson_proxy_loss)(qn[idx], pred))
        total = l_sup + beta_impl * l_impl + beta_pois * l_pois
        return total, (l_sup, l_impl, l_pois)

    @jax.jit
    def step(st, opt_s, idx):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(st, idx)
        updates, opt_s = tx.update(grads, opt_s, st)
        st = optax.apply_updates(st, updates)
        return st, opt_s, loss, aux

    tr_idx = jnp.asarray(tr)
    va_idx = jnp.asarray(va)
    te_idx = jnp.asarray(te)

    for ep in range(n_epochs):
        state, opt_state, tr_loss, (a, b, c) = step(state, opt_state, tr_idx)
        if ep % 20 == 0 or ep == n_epochs - 1:
            va_loss, va_aux = loss_fn(state, va_idx)
            print(
                f"epoch={ep:4d} train={float(tr_loss):.6e} val={float(va_loss):.6e} "
                f"sup={float(a):.3e} impl={float(b):.3e} pois={float(c):.3e}"
            )

    final_te, te_aux = loss_fn(state, te_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "normalization.npz",
        q_mean=np.asarray(q_mean),
        q_std=np.asarray(q_std),
        dq_mean=np.asarray(dq_mean),
        dq_std=np.asarray(dq_std),
        qaux_mean=np.asarray(qaux_mean),
        qaux_std=np.asarray(qaux_std),
    )

    save_kw = {
        "w_self": np.asarray(state["w_self"]),
        "w_msg": np.asarray(state["w_msg"]),
        "w_aux": np.asarray(state["w_aux"]),
        "w_coarse": np.asarray(state["w_coarse"]),
        "w_gate": np.asarray(state["w_gate"]),
        "b": np.asarray(state["b"]),
        "variant": np.asarray(["multilevel_fft1d" if use_fft else "multilevel"]),
        "flow_mode": np.asarray([flow_mode]),
        "message_steps": np.asarray([message_steps], dtype=int),
        "inner_iters": np.asarray([inner_iters], dtype=int),
        "coarsen_levels": np.asarray([coarsen_levels], dtype=int),
        "global_coupling_mode": np.asarray([gc.FFT_1D if use_fft else gc.MULTIGRID], dtype=int),
        "test_loss": np.asarray([float(final_te)]),
    }
    if use_fft:
        mm = max_rfft_modes(n_cells)
        save_kw["fft_w_r"] = np.asarray(state["fft_w_r"])
        save_kw["fft_w_i"] = np.asarray(state["fft_w_i"])
        save_kw["fft_blend_logit"] = np.asarray(state["fft_blend_logit"])
        save_kw["max_fft_modes"] = np.asarray([mm], dtype=int)
    np.savez_compressed(out_dir / "weights_deltaq.npz", **save_kw)
    print(f"Saved to {out_dir} test_loss={float(final_te):.6e}")
    return float(final_te)


def main():
    p = argparse.ArgumentParser(description="Train multilevel + optional 1D FFT + Poisson/implicit losses")
    p.add_argument("--dataset", type=Path, default=Path("outputs/gnn_blueprint/dataset_deltaq_small.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/gnn_blueprint/model_multilevel_fft1d"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--flow-mode", type=str, default="bidir", choices=["tb", "bt", "bidir", "alternating"])
    p.add_argument("--message-steps", type=int, default=3)
    p.add_argument("--inner-iters", type=int, default=1)
    p.add_argument("--coarsen-levels", type=int, default=2)
    p.add_argument("--no-fft", action="store_true", help="Multigrid only (no spectral block)")
    p.add_argument("--beta-impl", type=float, default=0.5)
    p.add_argument("--beta-pois", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path.cwd()
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    out = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    train(
        dataset,
        out,
        args.epochs,
        args.lr,
        args.flow_mode,
        args.message_steps,
        args.inner_iters,
        args.coarsen_levels,
        use_fft=not args.no_fft,
        beta_impl=args.beta_impl,
        beta_pois=args.beta_pois,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
