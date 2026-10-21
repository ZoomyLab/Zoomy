"""Train spectral global-coupling predictors on three mesh families:

1. **1D uniform** — standard second-difference Laplacian, `x` equispaced in `[0,1]`.
2. **1D non-uniform** — vertex Laplacian on random interior nodes, true `x` for NUDFT/RFF.
3. **2D triangular** — Delaunay triangles, graph Laplacian on centroids + ridge, Morton-ordered
   columns for the existing 1D multilevel smoother; 2D NUDFT/RFF use centroid `(x,y)`.

Architectures compared per mesh (where valid): `FFT_1D`, `NUDFT` / `RFF` on coordinates,
`graph_poly` (polynomial of symmetric graph Laplacian — mesh-native, cheap forward),
`graph_eigen` (low eigenvector basis — graph Fourier prototype; `eigh` once per mesh).

Outputs under `--out-root/<mesh>_<arch>/weights_deltaq.npz`. Use
`benchmark_spectral_mesh_suite` to compare GMRES matvecs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from zoomy_jax.gnn_blueprint import global_coupling as gc
from zoomy_jax.gnn_blueprint.fourier_1d import max_rfft_modes
from zoomy_jax.gnn_blueprint.graph_laplacian_spectral import (
    graph_first_k_eigenvectors,
    path_graph_adjacency,
    symmetric_normalized_laplacian_dense,
)
from zoomy_jax.gnn_blueprint.mesh_spectral_geometry import (
    delaunay_centroid_adjacency,
    laplacian_1d_nonuniform_vertex,
    morton_z_order_perm,
)
from zoomy_jax.gnn_blueprint.nonuniform_spectral import max_nudft_modes_1d, nudft_num_modes_2d
from zoomy_jax.gnn_blueprint.poisson_1d import laplacian_1d_dense
from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned


def _split_indices(n, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_val], idx[n_tr + n_val :]


def _build_q_1d(f_row: np.ndarray) -> np.ndarray:
    n = f_row.shape[0]
    return np.stack([np.zeros(n, dtype=np.float64), f_row.copy()], axis=0)


def dataset_uniform_1d(n_samples: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    a_np = np.asarray(laplacian_1d_dense(n), dtype=np.float64)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    f_all = rng.standard_normal((n_samples, n))
    u_all = np.linalg.solve(a_np, f_all.T).T
    q = np.stack([_build_q_1d(f_all[i]) for i in range(n_samples)], axis=0)
    dq = np.zeros_like(q)
    dq[:, 0] = u_all
    cls = np.zeros((n_samples, n), dtype=np.float64)
    dt = np.ones((n_samples,), dtype=np.float64)
    qaux = np.stack([f_all[i][None, :] for i in range(n_samples)], axis=0)
    g_ls = symmetric_normalized_laplacian_dense(path_graph_adjacency(n))
    meta = {"spectral_x": x, "spectral_xy": None, "dim": 1, "graph_l_sym": g_ls}
    return q, dq, cls, dt, qaux, a_np, meta


def dataset_nonuniform_1d(n_samples: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    xp = np.sort(rng.uniform(0.06, 0.94, size=n))
    a_np = laplacian_1d_nonuniform_vertex(xp, 0.0, 1.0)
    x_norm = (xp - xp.min()) / (xp.max() - xp.min() + 1e-15)
    f_all = rng.standard_normal((n_samples, n))
    u_all = np.linalg.solve(a_np, f_all.T).T
    q = np.stack([_build_q_1d(f_all[i]) for i in range(n_samples)], axis=0)
    dq = np.zeros_like(q)
    dq[:, 0] = u_all
    cls = np.zeros((n_samples, n), dtype=np.float64)
    dt = np.ones((n_samples,), dtype=np.float64)
    qaux = np.stack([f_all[i][None, :] for i in range(n_samples)], axis=0)
    g_ls = symmetric_normalized_laplacian_dense(path_graph_adjacency(n))
    meta = {"spectral_x": x_norm.astype(np.float64), "spectral_xy": None, "dim": 1, "graph_l_sym": g_ls}
    return q, dq, cls, dt, qaux, a_np, meta


def dataset_tri2d(n_samples: int, n_points: int, seed: int, ridge: float):
    rng = np.random.default_rng(seed)
    centroids, adj, _pts, _S = delaunay_centroid_adjacency(n_points, seed)
    perm = morton_z_order_perm(centroids)
    n_c = centroids.shape[0]
    pmat = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j in enumerate(perm):
        pmat[i, j] = 1.0
    adj_perm = pmat @ adj @ pmat.T
    deg = adj_perm.sum(axis=1)
    l_g = -adj_perm + np.diag(deg)
    a_perm = l_g + ridge * np.eye(n_c)
    g_ls = symmetric_normalized_laplacian_dense(adj_perm)
    xy_w = centroids[perm].astype(np.float64)
    f_all = rng.standard_normal((n_samples, n_c))
    u_all = np.linalg.solve(a_perm, f_all.T).T
    q = np.stack([_build_q_1d(f_all[i]) for i in range(n_samples)], axis=0)
    dq = np.zeros_like(q)
    dq[:, 0] = u_all
    cls = np.zeros((n_samples, n_c), dtype=np.float64)
    dt = np.ones((n_samples,), dtype=np.float64)
    qaux = np.stack([f_all[i][None, :] for i in range(n_samples)], axis=0)
    meta = {"spectral_x": None, "spectral_xy": xy_w, "dim": 2, "graph_l_sym": g_ls}
    return q, dq, cls, dt, qaux, a_perm, meta


def _init_state(
    key,
    n_fields: int,
    n_cells: int,
    spec: str,
    spectral_kmax: int,
    rff_m: int,
    graph_poly_order: int,
    graph_eig_columns: int,
):
    k0, k1, k2 = jax.random.split(key, 3)
    st = {
        "w_self": jnp.linspace(0.05, 0.09, n_fields),
        "w_msg": jnp.linspace(0.02, 0.04, n_fields),
        "w_aux": jnp.linspace(0.015, 0.025, n_fields),
        "w_coarse": jnp.linspace(0.02, 0.05, n_fields),
        "w_gate": jnp.linspace(0.8, 1.2, n_fields),
        "b": jnp.linspace(0.005, 0.008, n_fields),
    }
    if spec == "fft":
        m = max_rfft_modes(n_cells)
        st["fft_w_r"] = jnp.ones((n_fields, m)) + 0.01 * jax.random.normal(k0, (n_fields, m))
        st["fft_w_i"] = 0.01 * jax.random.normal(k1, (n_fields, m))
        st["fft_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "nudft_1d":
        nm = max_nudft_modes_1d(n_cells)
        st["spectral_w_r"] = jnp.ones((n_fields, nm)) + 0.01 * jax.random.normal(k0, (n_fields, nm))
        st["spectral_w_i"] = 0.01 * jax.random.normal(k1, (n_fields, nm))
        st["spectral_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "nudft_2d":
        nm = nudft_num_modes_2d(spectral_kmax)
        st["spectral_w_r"] = jnp.ones((n_fields, nm)) + 0.01 * jax.random.normal(k0, (n_fields, nm))
        st["spectral_w_i"] = 0.01 * jax.random.normal(k1, (n_fields, nm))
        st["spectral_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "rff_1d":
        st["rff_omega"] = jax.random.normal(k0, (rff_m,)) * 4.0
        st["rff_phase"] = jax.random.uniform(k1, (rff_m,), minval=0.0, maxval=2 * jnp.pi)
        st["rff_w_lin"] = 0.02 * jax.random.normal(k2, (n_fields, rff_m))
        st["spectral_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "rff_2d":
        st["rff_omega"] = jax.random.normal(k0, (rff_m, 2)) * 4.0
        st["rff_phase"] = jax.random.uniform(k1, (rff_m,), minval=0.0, maxval=2 * jnp.pi)
        st["rff_w_lin"] = 0.02 * jax.random.normal(k2, (n_fields, rff_m))
        st["spectral_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "graph_poly":
        kc = int(graph_poly_order) + 1
        st["graph_poly_coeff"] = 0.05 * jax.random.normal(k0, (n_fields, kc))
        st["graph_blend_logit"] = jnp.full((n_fields,), -1.0)
    elif spec == "graph_eigen":
        ke = max(int(graph_eig_columns), 0)
        st["graph_eig_w"] = (
            0.05 * jax.random.normal(k0, (n_fields, ke)) if ke > 0 else jnp.zeros((n_fields, 0))
        )
        st["graph_blend_logit"] = jnp.full((n_fields,), -1.0)
    else:
        raise ValueError(spec)
    return st


def _params_from_state(
    state: dict,
    spec: str,
    gcm: int,
    n_fields: int,
    n_cells: int,
    flow_mode: str,
    message_steps: int,
    inner_iters: int,
    coarsen_levels: int,
    single_layer: int,
    spectral_x: jnp.ndarray | None,
    spectral_xy: jnp.ndarray | None,
    spectral_kmax: int,
    n_spectral_modes_1d: int,
    graph_l_sym_j: jnp.ndarray | None = None,
    graph_eig_u_j: jnp.ndarray | None = None,
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
        "global_coupling_mode": gcm,
        "single_layer_mode": int(single_layer),
    }
    if spec == "fft":
        p["fft_w_r"] = state["fft_w_r"]
        p["fft_w_i"] = state["fft_w_i"]
        p["fft_blend_logit"] = state["fft_blend_logit"]
    elif spec in ("nudft_1d", "nudft_2d"):
        p["spectral_w_r"] = state["spectral_w_r"]
        p["spectral_w_i"] = state["spectral_w_i"]
        p["spectral_blend_logit"] = state["spectral_blend_logit"]
        if spec == "nudft_1d":
            p["spectral_x"] = spectral_x
            p["n_spectral_modes_1d"] = n_spectral_modes_1d
        else:
            p["spectral_xy"] = spectral_xy
            p["spectral_kmax"] = spectral_kmax
    elif spec in ("rff_1d", "rff_2d"):
        p["rff_omega"] = state["rff_omega"]
        p["rff_phase"] = state["rff_phase"]
        p["rff_w_lin"] = state["rff_w_lin"]
        p["spectral_blend_logit"] = state["spectral_blend_logit"]
        if spec == "rff_1d":
            p["spectral_x"] = spectral_x
        else:
            p["spectral_xy"] = spectral_xy
    elif spec == "graph_poly":
        p["graph_L_sym"] = graph_l_sym_j
        p["graph_poly_coeff"] = state["graph_poly_coeff"]
        p["graph_blend_logit"] = state["graph_blend_logit"]
    elif spec == "graph_eigen":
        p["graph_eig_U"] = graph_eig_u_j
        p["graph_eig_w"] = state["graph_eig_w"]
        p["graph_blend_logit"] = state["graph_blend_logit"]
    return p


def train_one(
    name: str,
    mesh: str,
    arch: str,
    out_dir: Path,
    n_cells: int,
    n_samples: int,
    epochs: int,
    lr: float,
    beta_res: float,
    seed: int,
    batch_size: int,
    patience: int,
    warmup_epochs: int,
    grad_clip: float,
    tri_points: int,
    ridge: float,
    spectral_kmax: int,
    rff_m: int,
    graph_poly_order: int,
    graph_eig_k: int,
):
    flow_mode = "bidir"
    message_steps = 3
    inner_iters = 1
    coarsen_levels = 3
    single_layer = 0
    n_fields = 2

    if mesh == "uniform_1d":
        q, dq, cls, dt, qaux, a_np, meta = dataset_uniform_1d(n_samples, n_cells, seed)
        n_loc = n_cells
    elif mesh == "nonuniform_1d":
        q, dq, cls, dt, qaux, a_np, meta = dataset_nonuniform_1d(n_samples, n_cells, seed)
        n_loc = n_cells
    elif mesh == "tri2d":
        q, dq, cls, dt, qaux, a_np, meta = dataset_tri2d(n_samples, tri_points, seed, ridge)
        n_loc = q.shape[2]
    else:
        raise SystemExit(f"unknown mesh {mesh}")

    if arch == "fft":
        spec, gcm = "fft", gc.FFT_1D
        if mesh != "uniform_1d":
            raise SystemExit("FFT_1D is only wired for uniform 1D indexing in this suite; use nudft/rff on other meshes.")
    elif arch == "nudft":
        spec = "nudft_1d" if meta["dim"] == 1 else "nudft_2d"
        gcm = gc.NUDFT_1D if meta["dim"] == 1 else gc.NUDFT_2D
    elif arch == "rff":
        spec = "rff_1d" if meta["dim"] == 1 else "rff_2d"
        gcm = gc.RFF_KERNEL_1D if meta["dim"] == 1 else gc.RFF_KERNEL_2D
    elif arch == "graph_poly":
        spec, gcm = "graph_poly", gc.GRAPH_POLY_LAPL
    elif arch == "graph_eigen":
        spec, gcm = "graph_eigen", gc.GRAPH_EIGEN_LOW
    else:
        raise SystemExit(f"unknown arch {arch}")

    spectral_x_np = meta["spectral_x"]
    spectral_xy_np = meta["spectral_xy"]
    spectral_x_j = None if spectral_x_np is None else jnp.asarray(spectral_x_np, dtype=jnp.float64)
    spectral_xy_j = None if spectral_xy_np is None else jnp.asarray(spectral_xy_np, dtype=jnp.float64)
    nm1d = max_nudft_modes_1d(n_loc)
    ls_np = meta["graph_l_sym"]
    graph_l_sym_j = jnp.asarray(ls_np, dtype=jnp.float64)
    u_np = graph_first_k_eigenvectors(ls_np, graph_eig_k) if arch == "graph_eigen" else None
    graph_eig_u_j = None if u_np is None else jnp.asarray(u_np, dtype=jnp.float64)
    n_eig_cols = int(u_np.shape[1]) if u_np is not None else 0

    a_mat = jnp.asarray(a_np)
    tr, va, te = _split_indices(n_samples, seed=seed)
    qn = jnp.asarray(q)
    dqn = jnp.asarray(dq)
    cls_j = jnp.asarray(cls)
    dt_j = jnp.asarray(dt)
    qax_j = jnp.asarray(qaux)

    key = jax.random.PRNGKey(seed + 120_217)
    state = _init_state(
        key,
        n_fields,
        n_loc,
        spec,
        spectral_kmax,
        rff_m,
        graph_poly_order,
        n_eig_cols,
    )
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
    tx = optax.chain(optax.clip_by_global_norm(float(grad_clip)), optax.adam(learning_rate=lr_schedule))
    opt_state = tx.init(state)

    def loss_fn(st, idx):
        p = _params_from_state(
            st,
            spec,
            gcm,
            n_fields,
            n_loc,
            flow_mode,
            message_steps,
            inner_iters,
            coarsen_levels,
            single_layer,
            spectral_x_j,
            spectral_xy_j,
            spectral_kmax,
            nm1d,
            graph_l_sym_j,
            graph_eig_u_j,
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
    rng_ep = np.random.default_rng(seed + 55_021)
    best_state = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
    best_val = float("inf")
    patience_left = int(patience)
    stopped = "max_epochs"

    for ep in range(epochs):
        perm = tr_list.copy()
        rng_ep.shuffle(perm)
        ep_losses = []
        for start in range(0, n_tr, bs):
            batch = perm[start : start + bs]
            if batch.shape[0] < bs:
                continue
            state, opt_state, tr_loss, _ = step(state, opt_state, jnp.asarray(batch))
            ep_losses.append(float(tr_loss))
        tr_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        va_loss, (va_sup, va_res) = loss_fn(state, va_idx)
        va_f = float(va_loss)

        if (not np.isfinite(tr_mean)) or tr_mean > 5000.0 * max(init_tr_f, 1e-30):
            stopped = "diverged"
            state = best_state
            break

        if va_f < best_val - 1e-12:
            best_val = va_f
            best_state = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
            patience_left = int(patience)
        else:
            patience_left -= 1

        if ep % 25 == 0 or ep == epochs - 1:
            print(f"[{name}] ep={ep:4d} train={tr_mean:.4e} val={va_f:.4e} sup={float(va_sup):.3e} res={float(va_res):.3e}")

        if patience_left <= 0 and ep >= warmup_epochs:
            stopped = "early_stop"
            state = best_state
            print(f"[{name}] early_stop ep={ep} best_val={best_val:.4e}")
            break

    if stopped == "max_epochs":
        state = best_state

    final_te, _ = loss_fn(state, te_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_final = _params_from_state(
        state,
        spec,
        gcm,
        n_fields,
        n_loc,
        flow_mode,
        message_steps,
        inner_iters,
        coarsen_levels,
        single_layer,
        spectral_x_j,
        spectral_xy_j,
        spectral_kmax,
        nm1d,
        graph_l_sym_j,
        graph_eig_u_j,
    )

    save = {
        "w_self": np.asarray(state["w_self"]),
        "w_msg": np.asarray(state["w_msg"]),
        "w_aux": np.asarray(state["w_aux"]),
        "w_coarse": np.asarray(state["w_coarse"]),
        "w_gate": np.asarray(state["w_gate"]),
        "b": np.asarray(state["b"]),
        "variant": np.asarray([name]),
        "flow_mode": np.asarray([flow_mode]),
        "message_steps": np.asarray([message_steps], dtype=int),
        "inner_iters": np.asarray([inner_iters], dtype=int),
        "coarsen_levels": np.asarray([coarsen_levels], dtype=int),
        "global_coupling_mode": np.asarray([gcm], dtype=int),
        "single_layer_mode": np.asarray([single_layer], dtype=int),
        "smooth_radii": np.zeros((0,), dtype=int),
        "n_cells_poisson": np.asarray([n_loc], dtype=int),
        "mesh_suite": np.asarray([mesh]),
        "arch_suite": np.asarray([arch]),
        "test_loss": np.asarray([float(final_te)]),
        "spectral_kmax": np.asarray([spectral_kmax], dtype=int),
        "rff_m": np.asarray([rff_m], dtype=int),
        "dataset_seed": np.asarray([seed], dtype=int),
        "tri_points": np.asarray([tri_points], dtype=int),
        "graph_ridge": np.asarray([ridge], dtype=np.float64),
        "graph_poly_order": np.asarray([graph_poly_order], dtype=int),
        "graph_eig_k": np.asarray([graph_eig_k], dtype=int),
    }
    save["graph_L_sym"] = np.asarray(ls_np, dtype=np.float64)
    if spec == "graph_eigen" and u_np is not None:
        save["graph_eig_U"] = np.asarray(u_np, dtype=np.float64)
    if spec == "fft":
        mm = max_rfft_modes(n_loc)
        save["fft_w_r"] = np.asarray(state["fft_w_r"])
        save["fft_w_i"] = np.asarray(state["fft_w_i"])
        save["fft_blend_logit"] = np.asarray(state["fft_blend_logit"])
        save["max_fft_modes"] = np.asarray([mm], dtype=int)
    elif spec in ("nudft_1d", "nudft_2d"):
        save["spectral_w_r"] = np.asarray(state["spectral_w_r"])
        save["spectral_w_i"] = np.asarray(state["spectral_w_i"])
        save["spectral_blend_logit"] = np.asarray(state["spectral_blend_logit"])
        if spec == "nudft_1d":
            save["spectral_x"] = np.asarray(spectral_x_np)
            save["n_spectral_modes_1d"] = np.asarray([nm1d], dtype=int)
        else:
            save["spectral_xy"] = np.asarray(spectral_xy_np)
    elif spec in ("rff_1d", "rff_2d"):
        save["rff_omega"] = np.asarray(state["rff_omega"])
        save["rff_phase"] = np.asarray(state["rff_phase"])
        save["rff_w_lin"] = np.asarray(state["rff_w_lin"])
        save["spectral_blend_logit"] = np.asarray(state["spectral_blend_logit"])
        if spec == "rff_1d":
            save["spectral_x"] = np.asarray(spectral_x_np)
        else:
            save["spectral_xy"] = np.asarray(spectral_xy_np)
    elif spec == "graph_poly":
        save["graph_poly_coeff"] = np.asarray(state["graph_poly_coeff"])
        save["graph_blend_logit"] = np.asarray(state["graph_blend_logit"])
    elif spec == "graph_eigen":
        save["graph_eig_w"] = np.asarray(state["graph_eig_w"])
        save["graph_blend_logit"] = np.asarray(state["graph_blend_logit"])

    np.savez_compressed(out_dir / "weights_deltaq.npz", **save)
    (out_dir / "training_summary.txt").write_text(
        f"name={name}\nmesh={mesh}\narch={arch}\nstopped={stopped}\n"
        f"best_val={best_val}\ntest_loss={float(final_te)}\n",
        encoding="utf-8",
    )
    print(f"[{name}] saved {out_dir} test_loss={float(final_te):.4e} ({stopped})")
    return float(final_te)


def main():
    ap = argparse.ArgumentParser(description="Train spectral predictors on uniform 1D / nonuniform 1D / 2D tri mesh")
    ap.add_argument("--out-root", type=Path, default=Path("outputs/gnn_blueprint/spectral_mesh_suite"))
    ap.add_argument("--mesh", type=str, default="all", help="uniform_1d, nonuniform_1d, tri2d, or all")
    ap.add_argument(
        "--arch",
        type=str,
        default="all",
        help="fft, nudft, rff, graph_poly, graph_eigen, or all (fft only with uniform_1d)",
    )
    ap.add_argument("--n-cells", type=int, default=64)
    ap.add_argument("--tri-points", type=int, default=48, help="Delaunay sites in [0,1]^2 (triangle count varies)")
    ap.add_argument("--ridge", type=float, default=0.03, help="SPD shift for graph Laplacian on triangles")
    ap.add_argument("--n-samples", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--beta-res", type=float, default=0.28)
    ap.add_argument("--patience", type=int, default=75)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--spectral-kmax", type=int, default=3, help="Integer waveband for 2D NUDFT: (2kmax+1)^2 modes")
    ap.add_argument("--rff-m", type=int, default=40, help="RFF feature count")
    ap.add_argument("--graph-poly-order", type=int, default=4, help="Polynomial degree K for graph_poly (K+1 coefficients)")
    ap.add_argument("--graph-eig-k", type=int, default=12, help="Low eigenvectors for graph_eigen (excludes constant mode)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()
    out_root = args.out_root if args.out_root.is_absolute() else root / args.out_root
    meshes = ["uniform_1d", "nonuniform_1d", "tri2d"] if args.mesh == "all" else [args.mesh]
    archs = (
        ["fft", "nudft", "rff", "graph_poly", "graph_eigen"]
        if args.arch == "all"
        else [args.arch]
    )

    for mesh in meshes:
        for arch in archs:
            if arch == "fft" and mesh != "uniform_1d":
                print(f"skip {mesh}/{arch} (FFT only defined for uniform 1D line index)")
                continue
            name = f"{mesh}_{arch}"
            train_one(
                name=name,
                mesh=mesh,
                arch=arch,
                out_dir=out_root / name,
                n_cells=args.n_cells,
                n_samples=args.n_samples,
                epochs=args.epochs,
                lr=args.lr,
                beta_res=args.beta_res,
                seed=args.seed,
                batch_size=args.batch_size,
                patience=args.patience,
                warmup_epochs=args.warmup_epochs,
                grad_clip=args.grad_clip,
                tri_points=args.tri_points,
                ridge=args.ridge,
                spectral_kmax=args.spectral_kmax,
                rff_m=args.rff_m,
                graph_poly_order=args.graph_poly_order,
                graph_eig_k=args.graph_eig_k,
            )


if __name__ == "__main__":
    main()
