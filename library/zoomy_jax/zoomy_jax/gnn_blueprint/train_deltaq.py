import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

try:
    from .models_deltaq import DeltaQGNN
except ImportError:
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from models_deltaq import DeltaQGNN


_FLOW_IDS = {
    "tb": 0,
    "bt": 1,
    "bidir": 2,
    "alternating": 3,
}


def _split_indices(n, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]


def _normalize(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    if std is None:
        std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def _restrict_pairwise(q):
    if q.shape[1] % 2 == 1:
        q = jnp.concatenate([q, q[:, -1:]], axis=1)
    return 0.5 * (q[:, 0::2] + q[:, 1::2])


def _prolong_repeat(qc, n_fine):
    return jnp.repeat(qc, 2, axis=1)[:, :n_fine]


def _flow_messages(q, flow_mode, step_i):
    q_left = jnp.pad(q[:, :-1], ((0, 0), (1, 0)), mode="edge")
    q_right = jnp.pad(q[:, 1:], ((0, 0), (0, 1)), mode="edge")
    msg_tb = q_right - q
    msg_bt = q_left - q

    if flow_mode == "tb":
        return msg_tb
    if flow_mode == "bt":
        return msg_bt
    if flow_mode == "alternating":
        return jnp.where((step_i % 2) == 0, msg_tb, msg_bt)
    return 0.5 * (msg_tb + msg_bt)


def _predict_one(model, q, qaux, dt, cls, variant, message_steps, inner_iters, coarsen_levels, flow_mode):
    cls_scale = jnp.clip(1.0 / (1.0 + 0.2 * cls), 0.5, 1.0).astype(q.dtype)
    aux0 = qaux[0] if qaux.shape[0] > 0 else jnp.zeros((q.shape[1],), dtype=q.dtype)

    if variant in ("linear", "linear_msg", "linear_msg_edge"):
        if variant == "linear":
            rhs = model.w_self[:, None] * q + model.b[:, None]
        elif variant == "linear_msg":
            mean_q = jnp.mean(q, axis=1, keepdims=True)
            rhs = model.w_self[:, None] * q + model.w_msg[:, None] * mean_q + model.b[:, None]
        else:
            mean_q = jnp.mean(q, axis=1, keepdims=True)
            std_q = jnp.std(q, axis=1, keepdims=True)
            rhs = model.w_self[:, None] * q + model.w_msg[:, None] * std_q + model.w_edge[:, None] * mean_q + model.b[:, None]
        return dt * rhs * cls_scale[None, :]

    # multilevel-flow predictor with iterative fixed-point style updates
    q_levels = [q]
    for _ in range(max(int(coarsen_levels) - 1, 0)):
        qf = q_levels[-1]
        if qf.shape[1] <= 4:
            break
        q_levels.append(_restrict_pairwise(qf))

    q_levels[-1] = q_levels[-1] + model.w_aux[:, None] * aux0[None, :q_levels[-1].shape[1]]

    for _ms in range(max(int(message_steps), 1)):
        for li in range(len(q_levels) - 1, -1, -1):
            ql = q_levels[li]
            n = ql.shape[1]
            coarse_ctx = jnp.zeros_like(ql)
            if li < len(q_levels) - 1:
                coarse_ctx = _prolong_repeat(q_levels[li + 1], n)

            for it in range(max(int(inner_iters), 1)):
                msg = _flow_messages(ql, flow_mode, it)
                gate = jax.nn.sigmoid(model.w_gate)[:, None]
                rhs = (
                    model.w_self[:, None] * ql
                    + model.w_msg[:, None] * msg
                    + model.w_aux[:, None] * aux0[None, :n]
                    + model.w_coarse[:, None] * coarse_ctx
                    + model.b[:, None]
                )
                ql = ql + gate * rhs
            q_levels[li] = ql

    q_mp = q_levels[0]
    dq = dt * q_mp * cls_scale[None, :]
    return dq


def _predict_deltaq(model, q, qaux, dt, class_id, variant="linear", message_steps=2, inner_iters=1, coarsen_levels=2, flow_mode="bidir"):
    preds = []
    for i in range(q.shape[0]):
        preds.append(_predict_one(model, q[i], qaux[i], dt[i], class_id[i], variant, message_steps, inner_iters, coarsen_levels, flow_mode))
    return jnp.stack(preds, axis=0)


def train(
    dataset_path: Path,
    out_dir: Path,
    n_epochs: int,
    lr: float,
    variant: str,
    seed: int = 42,
    message_steps: int = 2,
    inner_iters: int = 1,
    coarsen_levels: int = 2,
    flow_mode: str = "bidir",
):
    data = np.load(dataset_path)
    q = data['q']
    dt = data['dt']
    dq = data['delta_q']
    class_id = data['class_id'] if 'class_id' in data.files else np.zeros((q.shape[0], q.shape[2]), dtype=np.float64)
    qaux = data['qaux'] if 'qaux' in data.files else np.zeros((q.shape[0], 1, q.shape[2]), dtype=np.float64)

    n_samples, n_fields, _ = q.shape
    tr, va, te = _split_indices(n_samples, seed=seed)

    _, q_mean, q_std = _normalize(q[tr])
    _, dq_mean, dq_std = _normalize(dq[tr])
    _, qaux_mean, qaux_std = _normalize(qaux[tr])

    qn = (q - q_mean) / q_std
    dqn = (dq - dq_mean) / dq_std
    qauxn = (qaux - qaux_mean) / qaux_std

    qn = jnp.asarray(qn)
    dqn = jnp.asarray(dqn)
    qauxn = jnp.asarray(qauxn)
    dt = jnp.asarray(dt)
    class_id = jnp.asarray(class_id)

    model = DeltaQGNN(n_fields=n_fields)
    tx = optax.adam(lr)
    opt_state = tx.init(model)

    def loss_fn(m, idx):
        pred = _predict_deltaq(
            m, qn[idx], qauxn[idx], dt[idx], class_id[idx],
            variant=variant,
            message_steps=message_steps,
            inner_iters=inner_iters,
            coarsen_levels=coarsen_levels,
            flow_mode=flow_mode,
        )
        target = dqn[idx]
        return jnp.mean((pred - target) ** 2)

    @jax.jit
    def step(m, opt_s, idx):
        loss, grads = jax.value_and_grad(loss_fn)(m, idx)
        updates, opt_s = tx.update(grads, opt_s, m)
        m = optax.apply_updates(m, updates)
        return m, opt_s, loss

    tr_idx = jnp.asarray(tr)
    va_idx = jnp.asarray(va)
    te_idx = jnp.asarray(te)

    hist = []
    for ep in range(n_epochs):
        model, opt_state, tr_loss = step(model, opt_state, tr_idx)
        va_loss = loss_fn(model, va_idx)
        te_loss = loss_fn(model, te_idx)
        hist.append((ep, float(tr_loss), float(va_loss), float(te_loss)))
        if ep % 20 == 0 or ep == n_epochs - 1:
            print(f"epoch={ep:4d} train={float(tr_loss):.6e} val={float(va_loss):.6e} test={float(te_loss):.6e}")

    final_te = float(loss_fn(model, te_idx))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / 'normalization.npz',
        q_mean=np.asarray(q_mean), q_std=np.asarray(q_std),
        dq_mean=np.asarray(dq_mean), dq_std=np.asarray(dq_std),
        qaux_mean=np.asarray(qaux_mean), qaux_std=np.asarray(qaux_std),
    )
    np.savez_compressed(
        out_dir / 'weights_deltaq.npz',
        w_self=np.asarray(model.w_self),
        w_msg=np.asarray(model.w_msg),
        w_edge=np.asarray(model.w_edge),
        w_aux=np.asarray(model.w_aux),
        w_coarse=np.asarray(model.w_coarse),
        w_gate=np.asarray(model.w_gate),
        b=np.asarray(model.b),
        variant=np.asarray([variant]),
        flow_mode=np.asarray([flow_mode]),
        message_steps=np.asarray([message_steps], dtype=int),
        inner_iters=np.asarray([inner_iters], dtype=int),
        coarsen_levels=np.asarray([coarsen_levels], dtype=int),
        flow_mode_id=np.asarray([_FLOW_IDS.get(flow_mode, 2)], dtype=int),
        test_loss=np.asarray([final_te]),
    )
    with (out_dir / 'loss_history.csv').open('w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,test_loss\n')
        for ep, trl, val, tes in hist:
            f.write(f"{ep},{trl},{val},{tes}\n")

    print(f"Saved model bundle to: {out_dir}")
    print(f"test_loss={final_te:.6e}")
    return final_te


def main():
    parser = argparse.ArgumentParser(description='Train deltaQ model (auto split/normalize + loss history)')
    parser.add_argument('--dataset', type=Path, default=Path('outputs/gnn_blueprint/dataset_deltaq.npz'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/gnn_blueprint/model_deltaq'))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--variant', type=str, default='multilevel_flow', choices=['linear', 'linear_msg', 'linear_msg_edge', 'multilevel_flow'])
    parser.add_argument('--message-steps', type=int, default=3)
    parser.add_argument('--inner-iters', type=int, default=2)
    parser.add_argument('--coarsen-levels', type=int, default=3)
    parser.add_argument('--flow-mode', type=str, default='bidir', choices=['tb', 'bt', 'bidir', 'alternating'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    train(
        dataset, out_dir, args.epochs, args.lr, args.variant, args.seed,
        message_steps=args.message_steps,
        inner_iters=args.inner_iters,
        coarsen_levels=args.coarsen_levels,
        flow_mode=args.flow_mode,
    )


if __name__ == '__main__':
    main()
