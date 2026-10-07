"""Shared multilevel learned ``deltaQ`` predictor (optional 1D FFT global coupling).

Shared by the training scripts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from zoomy_jax.gnn_blueprint import global_coupling as gc
from zoomy_jax.gnn_blueprint.fourier_1d import apply_fourier_linear_mix_1d
from zoomy_jax.gnn_blueprint.graph_laplacian_spectral import (
    apply_graph_eigen_diag_mix,
    apply_graph_polynomial_laplacian_mix,
)
from zoomy_jax.gnn_blueprint.nonuniform_spectral import (
    apply_nudft_linear_mix_1d,
    apply_nudft_linear_mix_2d,
    apply_rff_kernel_mix_1d,
    apply_rff_kernel_mix_2d,
    max_nudft_modes_1d,
)


def predict_delta_q_learned(
    Qc: jnp.ndarray,
    Qauxold: jnp.ndarray,
    dt: jnp.ndarray,
    class_id: jnp.ndarray,
    params: dict | None,
    message_steps: int,
    return_diagnostics: bool = False,
):
    """Predict ``deltaQ`` on inner cells; optional FFT_1D mix after multilevel state.

    ``params`` matches bundles from ``train_deltaq.py`` / ``train_multilevel_fft1d.py``.
    """
    n_fields, _n_cells = Qc.shape
    n_inner = class_id.shape[0]

    if params is None:
        p = {
            "w_self": jnp.ones((n_fields,), dtype=Qc.dtype) * 0.05,
            "w_msg": jnp.ones((n_fields,), dtype=Qc.dtype) * 0.02,
            "w_aux": jnp.ones((n_fields,), dtype=Qc.dtype) * 0.01,
            "w_coarse": jnp.ones((n_fields,), dtype=Qc.dtype) * 0.03,
            "w_gate": jnp.ones((n_fields,), dtype=Qc.dtype),
            "b": jnp.zeros((n_fields,), dtype=Qc.dtype),
            "message_steps": int(message_steps),
            "inner_iters": 1,
            "coarsen_levels": int(message_steps),
            "flow_mode": "bidir",
            "global_coupling_mode": gc.MULTIGRID,
            "single_layer_mode": 0,
        }
    else:
        p = params

    w_self = p["w_self"].astype(Qc.dtype)
    w_msg = p["w_msg"].astype(Qc.dtype)
    w_aux = p["w_aux"].astype(Qc.dtype)
    w_coarse = p["w_coarse"].astype(Qc.dtype)
    w_gate = p["w_gate"].astype(Qc.dtype)
    b = p["b"].astype(Qc.dtype)
    flow_mode = p.get("flow_mode", "bidir")
    k = max(int(p.get("message_steps", message_steps)), 1)
    inner_iters = max(int(p.get("inner_iters", 1)), 1)
    coarsen_levels = max(int(p.get("coarsen_levels", k)), 1)
    single_layer = int(p.get("single_layer_mode", 0))
    if single_layer:
        coarsen_levels = 1
    gcm = int(p.get("global_coupling_mode", gc.MULTIGRID))
    if gcm == gc.NUFFT_STUB:
        gcm = gc.MULTIGRID

    q_in = Qc[:, :n_inner]

    if Qauxold.shape[0] > 0:
        aux0 = Qauxold[0, :n_inner]
    else:
        aux0 = jnp.zeros((n_inner,), dtype=Qc.dtype)

    cls_scale = 1.0 / (1.0 + 0.2 * class_id.astype(Qc.dtype))
    cls_scale = jnp.clip(cls_scale, 0.5, 1.0)

    def _restrict_pairwise(q):
        if q.shape[1] % 2 == 1:
            q = jnp.concatenate([q, q[:, -1:]], axis=1)
        return 0.5 * (q[:, 0::2] + q[:, 1::2])

    def _flow_msg(q, step_i):
        q_left = jnp.pad(q[:, :-1], ((0, 0), (1, 0)), mode="edge")
        q_right = jnp.pad(q[:, 1:], ((0, 0), (0, 1)), mode="edge")
        tb = q_right - q
        bt = q_left - q
        if flow_mode == "tb":
            return tb
        if flow_mode == "bt":
            return bt
        if flow_mode == "alternating":
            return jnp.where((step_i % 2) == 0, tb, bt)
        return 0.5 * (tb + bt)

    def _smooth(q_local, coarse_ctx):
        qk = q_local
        for it in range(inner_iters):
            msg = _flow_msg(qk, it)
            gate = jax.nn.sigmoid(w_gate)[:, None]
            rhs = (
                w_self[:, None] * qk
                + w_msg[:, None] * msg
                + w_aux[:, None] * aux0[None, :qk.shape[1]]
                + w_coarse[:, None] * coarse_ctx
                + b[:, None]
            )
            qk = qk + gate * rhs
        return qk

    levels = [q_in]
    if not single_layer:
        for _ in range(coarsen_levels - 1):
            qf = levels[-1]
            if qf.shape[1] <= 4:
                break
            levels.append(_restrict_pairwise(qf))

    levels[-1] = _smooth(levels[-1], jnp.zeros_like(levels[-1]))

    coarse_norm_sum = jnp.asarray(0.0, dtype=Qc.dtype)
    if single_layer:
        for _ in range(k):
            q_fine = levels[0]
            levels[0] = _smooth(q_fine, jnp.zeros_like(q_fine))
    else:
        for _ in range(k):
            for li in range(len(levels) - 2, -1, -1):
                q_fine = levels[li]
                nf = q_fine.shape[1]
                coarse_up = jnp.repeat(levels[li + 1], 2, axis=1)[:, :nf]
                levels[li] = _smooth(q_fine, coarse_up)
                coarse_norm_sum = coarse_norm_sum + jnp.linalg.norm(coarse_up) / jnp.sqrt(
                    jnp.asarray(coarse_up.size, dtype=Qc.dtype)
                )

    q_mp = levels[0]

    if gcm == gc.FFT_1D:
        fft_w_r = p["fft_w_r"].astype(Qc.dtype)
        fft_w_i = p["fft_w_i"].astype(Qc.dtype)
        fft_blend = p["fft_blend_logit"].astype(Qc.dtype)
        q_mp = apply_fourier_linear_mix_1d(q_mp, fft_w_r, fft_w_i, fft_blend)
    elif gcm == gc.NUDFT_1D:
        sx = p.get("spectral_x")
        if sx is None:
            nloc = q_mp.shape[1]
            sx = jnp.linspace(0.0, 1.0, nloc, dtype=q_mp.dtype)
        else:
            sx = jnp.asarray(sx).astype(q_mp.dtype).reshape((q_mp.shape[1],))
        nm = int(p.get("n_spectral_modes_1d", max_nudft_modes_1d(int(q_mp.shape[1]))))
        wr = p["spectral_w_r"].astype(Qc.dtype)
        wi = p["spectral_w_i"].astype(Qc.dtype)
        bl = p["spectral_blend_logit"].astype(Qc.dtype)
        q_mp = apply_nudft_linear_mix_1d(q_mp, sx, wr, wi, bl, nm)
    elif gcm == gc.NUDFT_2D:
        sxy = jnp.asarray(p["spectral_xy"]).astype(q_mp.dtype)
        sxy = sxy.reshape((q_mp.shape[1], 2))
        kmax = int(p.get("spectral_kmax", 3))
        wr = p["spectral_w_r"].astype(Qc.dtype)
        wi = p["spectral_w_i"].astype(Qc.dtype)
        bl = p["spectral_blend_logit"].astype(Qc.dtype)
        q_mp = apply_nudft_linear_mix_2d(q_mp, sxy, wr, wi, bl, kmax)
    elif gcm == gc.RFF_KERNEL_1D:
        sx = p.get("spectral_x")
        if sx is None:
            nloc = q_mp.shape[1]
            sx = jnp.linspace(0.0, 1.0, nloc, dtype=q_mp.dtype)
        else:
            sx = jnp.asarray(sx).astype(q_mp.dtype).reshape((q_mp.shape[1],))
        om = jnp.asarray(p["rff_omega"]).astype(q_mp.dtype)
        ph = jnp.asarray(p["rff_phase"]).astype(q_mp.dtype)
        wl = p["rff_w_lin"].astype(Qc.dtype)
        bl = p["spectral_blend_logit"].astype(Qc.dtype)
        q_mp = apply_rff_kernel_mix_1d(q_mp, sx, om, ph, wl, bl)
    elif gcm == gc.RFF_KERNEL_2D:
        sxy = jnp.asarray(p["spectral_xy"]).astype(q_mp.dtype).reshape((q_mp.shape[1], 2))
        om = jnp.asarray(p["rff_omega"]).astype(q_mp.dtype)
        ph = jnp.asarray(p["rff_phase"]).astype(q_mp.dtype)
        wl = p["rff_w_lin"].astype(Qc.dtype)
        bl = p["spectral_blend_logit"].astype(Qc.dtype)
        q_mp = apply_rff_kernel_mix_2d(q_mp, sxy, om, ph, wl, bl)
    elif gcm == gc.GRAPH_POLY_LAPL:
        l_sym = jnp.asarray(p["graph_L_sym"]).astype(q_mp.dtype)
        coeff = p["graph_poly_coeff"].astype(q_mp.dtype)
        bl = p["graph_blend_logit"].astype(q_mp.dtype)
        q_mp = apply_graph_polynomial_laplacian_mix(q_mp, l_sym, coeff, bl)
    elif gcm == gc.GRAPH_EIGEN_LOW:
        uu = jnp.asarray(p["graph_eig_U"]).astype(q_mp.dtype)
        ww = p["graph_eig_w"].astype(q_mp.dtype)
        bl = p["graph_blend_logit"].astype(q_mp.dtype)
        q_mp = apply_graph_eigen_diag_mix(q_mp, uu, ww, bl)

    dq_in = []
    for i in range(n_fields):
        rhs = w_self[i] * q_mp[i] + b[i] + w_aux[i] * aux0
        dq_in.append(dt * rhs * cls_scale)
    dq_in = jnp.stack(dq_in, axis=0)

    dq = jnp.zeros_like(Qc)
    dq = dq.at[:, :n_inner].set(dq_in)
    if return_diagnostics:
        n_levels = len(levels) - 1
        denom = jnp.asarray(max(n_levels * k, 1), dtype=Qc.dtype)
        coarse_norm_avg = coarse_norm_sum / denom
        return dq, coarse_norm_avg
    return dq
