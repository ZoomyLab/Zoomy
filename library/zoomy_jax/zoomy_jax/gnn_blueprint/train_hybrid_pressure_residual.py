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
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from models_deltaq import DeltaQGNN


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


def _lap1d(u):
    left = jnp.pad(u[:, :-1], ((0, 0), (1, 0)), mode='edge')
    right = jnp.pad(u[:, 1:], ((0, 0), (0, 1)), mode='edge')
    return left - 2.0 * u + right


def _grad1d(v):
    left = jnp.pad(v[:-1], ((1, 0)), mode='edge')
    right = jnp.pad(v[1:], ((0, 1)), mode='edge')
    return 0.5 * (right - left)


def _corr(a, b):
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    da = jnp.linalg.norm(a)
    db = jnp.linalg.norm(b)
    return jnp.where((da > 1e-14) & (db > 1e-14), jnp.sum(a * b) / (da * db), 0.0)


def _predict_deltaq(model, q, dt, class_id):
    preds = []
    for i in range(q.shape[0]):
        qq = q[i]
        cls = class_id[i]
        cls_scale = jnp.clip(1.0 / (1.0 + 0.2 * cls), 0.5, 1.0)
        mean_q = jnp.mean(qq, axis=1, keepdims=True)
        rhs = model.w_self[:, None] * qq + model.w_msg[:, None] * mean_q + model.b[:, None]
        preds.append(dt[i] * rhs * cls_scale[None, :])
    return jnp.stack(preds, axis=0)


def _implicit_residual_proxy(q_old, dq_pred, dt):
    q_guess = q_old + dq_pred
    s = 0.08 * q_guess + 0.04 * _lap1d(q_guess)
    return dq_pred - dt * s


def _pressure_alignment_loss(q_old, dq_pred):
    # expect fields [b,h,u] for GN-topo-like datasets
    if q_old.shape[0] < 3:
        return jnp.asarray(0.0, dtype=q_old.dtype)
    b = q_old[0]
    h = q_old[1]
    du = dq_pred[2]
    p = 9.81 * (b + h)
    dpdx = _grad1d(p)
    c = _corr(du, -dpdx)
    # maximize correlation => minimize (1-c)
    return 1.0 - c


def train(dataset_path: Path, out_dir: Path, n_epochs: int, lr: float, beta_res: float, beta_p: float, seed: int = 42):
    d = np.load(dataset_path)
    q = d['q']
    dt = d['dt']
    dq = d['delta_q']
    cls = d['class_id'] if 'class_id' in d.files else np.zeros((q.shape[0], q.shape[2]), dtype=float)

    n_samples, n_fields, _ = q.shape
    tr, va, te = _split_indices(n_samples, seed=seed)

    _, q_mean, q_std = _normalize(q[tr])
    _, dq_mean, dq_std = _normalize(dq[tr])

    qn = (q - q_mean) / q_std
    dqn = (dq - dq_mean) / dq_std

    qn = jnp.asarray(qn)
    dqn = jnp.asarray(dqn)
    dt = jnp.asarray(dt)
    cls = jnp.asarray(cls)

    model = DeltaQGNN(n_fields=n_fields)
    tx = optax.adam(lr)
    opt_state = tx.init(model)

    def loss_fn(m, idx):
        pred = _predict_deltaq(m, qn[idx], dt[idx], cls[idx])
        target = dqn[idx]
        l_super = jnp.mean((pred - target) ** 2)

        # residual proxy term
        res = jax.vmap(_implicit_residual_proxy)(qn[idx], pred, dt[idx])
        l_res = jnp.mean(res**2)

        # pressure alignment term (batch mean)
        lp = jax.vmap(_pressure_alignment_loss)(qn[idx], pred)
        l_press = jnp.mean(lp)

        total = l_super + beta_res * l_res + beta_p * l_press
        return total, (l_super, l_res, l_press)

    @jax.jit
    def step(m, opt_s, idx):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(m, idx)
        updates, opt_s = tx.update(grads, opt_s, m)
        m = optax.apply_updates(m, updates)
        return m, opt_s, loss, aux

    tr_idx = jnp.asarray(tr)
    va_idx = jnp.asarray(va)
    te_idx = jnp.asarray(te)

    hist = []
    for ep in range(n_epochs):
        model, opt_state, tr_loss, (tr_sup, tr_res, tr_pre) = step(model, opt_state, tr_idx)
        va_loss, (va_sup, va_res, va_pre) = loss_fn(model, va_idx)
        te_loss, (te_sup, te_res, te_pre) = loss_fn(model, te_idx)
        hist.append((
            ep, float(tr_loss), float(va_loss), float(te_loss),
            float(tr_sup), float(tr_res), float(tr_pre),
            float(va_sup), float(va_res), float(va_pre),
            float(te_sup), float(te_res), float(te_pre),
        ))
        if ep % 20 == 0 or ep == n_epochs - 1:
            print(
                f"epoch={ep:4d} train={float(tr_loss):.6e} val={float(va_loss):.6e} test={float(te_loss):.6e} "
                f"super={float(tr_sup):.3e} res={float(tr_res):.3e} press={float(tr_pre):.3e}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / 'weights_deltaq_hybrid.npz',
        w_self=np.asarray(model.w_self),
        w_msg=np.asarray(model.w_msg),
        w_edge=np.asarray(model.w_edge),
        b=np.asarray(model.b),
        beta_res=np.asarray([beta_res]),
        beta_p=np.asarray([beta_p]),
    )
    with (out_dir / 'loss_history_hybrid.csv').open('w', encoding='utf-8') as f:
        f.write('epoch,train,val,test,tr_super,tr_res,tr_press,va_super,va_res,va_press,te_super,te_res,te_press\n')
        for r in hist:
            f.write(','.join(map(str, r)) + '\n')

    print(f"Saved hybrid model to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train deltaQ with hybrid residual+pressure loss')
    parser.add_argument('--dataset', type=Path, default=Path('outputs/gnn_blueprint/dataset_deltaq_small.npz'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/gnn_blueprint/model_deltaq_hybrid'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--beta-res', type=float, default=0.5)
    parser.add_argument('--beta-press', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    out = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    train(dataset, out, args.epochs, args.lr, args.beta_res, args.beta_press, args.seed)


if __name__ == '__main__':
    main()
