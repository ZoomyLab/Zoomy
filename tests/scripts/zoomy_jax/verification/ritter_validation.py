"""Ritter 1D dam break validation harness.

Classical analytical solution for the wet/dry dam break on a flat dry
bed (Ritter 1892).  ``h_L`` upstream of the dam, ``h_R = 0`` downstream,
no friction, no viscosity, no bathymetry.  At time ``t > 0``::

      ┌  h_L                                              if  ξ ≤ -c₀
  h = │  (2 c₀ - ξ)² / (9 g)                              if -c₀ ≤ ξ ≤ 2c₀
      └  0                                                if  ξ ≥ 2 c₀

with ``ξ = (x - x_dam) / t`` and ``c₀ = √(g h_L)``.

The harness runs the scheme to several probe times, samples ``h`` on a
1D x-axis (averaging over the channel width), compares against Ritter,
and reports L¹/L∞ errors and the front position error::

    front_num  = max x such that h(x) > h_front_threshold
    front_exact = x_dam + 2 c₀ t

A scheme that gives **shock-like front and rarefaction tail matching
Ritter** is correct.  A scheme that gives **diffusive spread** has
either a buggy reconstruction or wave-speed estimate.

Compare schemes via env vars::

    RITTER_T_PROBE=1.0 RITTER_SCHEMES="o1 o2_cons o2_eta" \\
        python tests/scripts/zoomy_jax/ritter_validation.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wetdry_dambreak_probe import (  # noqa: E402
    setup_solver, set_dambreak_ic, build_swe_sm,
)
from zoomy_jax.mesh.mesh import convert_mesh_to_jax  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402


# Domain: [0, 40] m, dam at x = 10, channel ~thin (400 cells along x).
# Channel must be long enough that the right wall doesn't reflect into
# the rarefaction in the time window we probe.  Front advances at
# 2 c₀ ≈ 6.26 m/s, so for t ≤ 4s we need x_right - x_dam ≥ 25m.
LX = 40.0
LY = 0.1
NX = 400
NY = 2
X_DAM = 10.0
H_L = 1.0
H_R = 0.0
G = 9.81
C0 = float(np.sqrt(G * H_L))


def ritter_h(x, t, h_L=H_L, g=G, x_dam=X_DAM):
    """Analytical Ritter ``h(x, t)``.  Returns an array of the same
    shape as ``x``."""
    if t == 0.0:
        return np.where(x < x_dam, h_L, 0.0).astype(float)
    c0 = float(np.sqrt(g * h_L))
    xi = (np.asarray(x, dtype=float) - x_dam) / t
    h = np.zeros_like(xi)
    rarefaction = (xi > -c0) & (xi < 2.0 * c0)
    h = np.where(xi <= -c0, h_L, h)
    h = np.where(rarefaction, (2.0 * c0 - xi) ** 2 / (9.0 * g), h)
    return h


def ritter_u(x, t, h_L=H_L, g=G, x_dam=X_DAM):
    """Analytical Ritter ``u(x, t)``."""
    if t == 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    c0 = float(np.sqrt(g * h_L))
    xi = (np.asarray(x, dtype=float) - x_dam) / t
    rarefaction = (xi > -c0) & (xi < 2.0 * c0)
    u = np.where(rarefaction, (2.0 / 3.0) * (xi + c0), 0.0)
    return u


def sample_1d(mesh, Q, n_bins=200):
    """Average h, hu over y so we get a clean 1D profile."""
    nc = int(mesh.n_inner_cells)
    xc = np.asarray(mesh.cell_centers)[0, :nc]
    h_cell = np.asarray(Q)[1, :nc]
    hu_cell = np.asarray(Q)[2, :nc]
    bin_edges = np.linspace(0.0, LX, n_bins + 1)
    bin_idx = np.clip(np.digitize(xc, bin_edges) - 1, 0, n_bins - 1)
    h_bin = np.zeros(n_bins)
    hu_bin = np.zeros(n_bins)
    n_bin = np.zeros(n_bins)
    np.add.at(h_bin, bin_idx, h_cell)
    np.add.at(hu_bin, bin_idx, hu_cell)
    np.add.at(n_bin, bin_idx, 1)
    n_bin = np.maximum(n_bin, 1)
    return (0.5 * (bin_edges[:-1] + bin_edges[1:]),
            h_bin / n_bin, hu_bin / n_bin)


def run_to_time(solver, Q, Qaux, t_end, cfl, eig_refresh=1):
    mesh_jax = convert_mesh_to_jax(solver._rt_mesh)
    runtime = solver._rt_model
    flux_op = solver.get_flux_operator(mesh_jax, runtime)
    eig_op = solver.get_compute_max_abs_eigenvalue(mesh_jax, runtime)
    parameters = solver._rt_parameters
    n_owned = int(solver._rt_mesh.n_inner_cells)
    min_inradius = float(np.min(np.asarray(mesh_jax.cell_inradius)))
    update_var = runtime.update_variables

    @jax.jit
    def step(dt, t, Q, Qaux):
        dQ = flux_op(dt, t, Q, Qaux, parameters, jnp.zeros_like(Q))
        Q1 = Q.at[:, :n_owned].add(dt * dQ[:, :n_owned])
        if update_var is not None:
            Q1 = Q1.at[:, :n_owned].set(
                update_var(Q1[:, :n_owned], Qaux[:, :n_owned], parameters))
        return Q1

    # Warmup.
    me = float(eig_op(Q, Qaux, parameters))
    dt0 = jnp.asarray(cfl * min_inradius / max(me, 1e-12), dtype=Q.dtype)
    Q = step(dt0, jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux)
    Q.block_until_ready()

    t_phys = 0.0
    cached_dt = 0.0
    steps_to_eig = 0
    n_steps = 0
    while t_phys < t_end:
        if steps_to_eig == 0:
            me = float(eig_op(Q, Qaux, parameters))
            cached_dt = cfl * min_inradius / max(me, 1e-12)
            steps_to_eig = eig_refresh
        dt = min(cached_dt, t_end - t_phys)
        Q = step(jnp.asarray(dt, dtype=Q.dtype),
                 jnp.asarray(t_phys, dtype=Q.dtype), Q, Qaux)
        t_phys += dt
        steps_to_eig -= 1
        n_steps += 1
    Q.block_until_ready()
    return Q, t_phys, n_steps


def front_position(x_bins, h_bins, threshold=1e-3):
    """Rightmost x with h > threshold."""
    above = np.where(h_bins > threshold)[0]
    return float(x_bins[above[-1]]) if len(above) else float("nan")


def evaluate_scheme(label, order, mode, cfl, t_probes, eig_refresh=1,
                    limiter="minmod"):
    mesh = LSQMesh.create_2d((0.0, LX, 0.0, LY), NX, NY)
    solver, Q, Qaux, sm = setup_solver(mesh, order=order,
                                       reconstruction_mode=mode,
                                       limiter=limiter)
    Q = set_dambreak_ic(solver, Q, x_dam=X_DAM, h_L=H_L, h_R=H_R,
                        b_slope=0.0)
    results = []
    Q_now = Q
    t_now = 0.0
    for t_target in t_probes:
        Q_now, t_now, n_steps = run_to_time(solver, Q_now, Qaux,
                                            t_end=t_target, cfl=cfl,
                                            eig_refresh=eig_refresh)
        x_b, h_b, hu_b = sample_1d(solver._rt_mesh, Q_now)
        h_exact = ritter_h(x_b, t_now)
        L1 = np.sum(np.abs(h_b - h_exact)) * (LX / len(x_b))
        Linf = np.max(np.abs(h_b - h_exact))
        front_num = front_position(x_b, h_b)
        front_exact = X_DAM + 2.0 * C0 * t_now
        # Width of the rarefaction → wave back position
        back_num = LX - front_position(x_b[::-1], (H_L - h_b)[::-1])
        # (lower-side undisturbed-region edge — h still ≈ h_L)
        # Simpler: find the point where h drops below 0.99 * h_L
        below = np.where(h_b < 0.99 * H_L)[0]
        back_num = float(x_b[below[0]]) if len(below) else 0.0
        back_exact = X_DAM - C0 * t_now
        results.append(dict(
            t=t_now, n_steps=n_steps,
            L1=L1, Linf=Linf,
            front_num=front_num, front_exact=front_exact,
            front_err=front_num - front_exact,
            back_num=back_num, back_exact=back_exact,
            x_bins=x_b, h_num=h_b, h_exact=h_exact, hu_num=hu_b,
        ))
    return label, results


def main():
    t_probes = [
        float(t) for t in
        os.environ.get("RITTER_T_PROBES", "0.5,1.0,1.5").split(",")
    ]
    schemes_arg = os.environ.get("RITTER_SCHEMES",
                                 "o1 o2_cons o2_eta")
    schemes_keys = schemes_arg.split()
    cfl_o1 = float(os.environ.get("RITTER_CFL_O1", "0.5"))
    cfl_o2 = float(os.environ.get("RITTER_CFL_O2", "0.25"))
    eig_refresh = int(os.environ.get("RITTER_EIG_REFRESH", "1"))

    scheme_defs = {
        # key:           (label,                  order, mode,          cfl,    limiter)
        "o1":          ("O1 (constant)",          1,    "conservative", cfl_o1, "minmod"),
        "o2_cons":     ("O2 cons (minmod)",       2,    "conservative", cfl_o2, "minmod"),
        "o2_xz":       ("O2 + Xing-Zhang (mm)",   2,    "xz",           cfl_o2, "minmod"),
        "o2_eta":      ("O2 + η-MUSCL (mm)",      2,    "eta",          cfl_o2, "minmod"),
        "o2_eta_vk":   ("O2 + η-MUSCL (VK)",      2,    "eta",          cfl_o2, "venkatakrishnan"),
        "o2_cons_vk":  ("O2 cons (VK)",           2,    "conservative", cfl_o2, "venkatakrishnan"),
    }

    print(f"Ritter 1D dam break  (channel [0,{LX}]×[0,{LY}], "
          f"NX={NX}, NY={NY}, dam at {X_DAM}, h_L={H_L}, h_R={H_R})")
    print(f"Analytical:  c₀ = √(g h_L) = {C0:.3f} m/s   "
          f"front speed 2c₀ = {2*C0:.3f} m/s")
    print(f"Probe times: {t_probes}\n")

    all_results = {}
    for key in schemes_keys:
        if key not in scheme_defs:
            print(f"  unknown scheme {key}")
            continue
        label, order, mode, cfl, limiter = scheme_defs[key]
        print(f"=== {label}  (cfl={cfl}, order={order}, mode={mode}, "
              f"limiter={limiter}) ===")
        all_results[key] = evaluate_scheme(label, order, mode, cfl,
                                            t_probes,
                                            eig_refresh=eig_refresh,
                                            limiter=limiter)
        _, runs = all_results[key]
        for r in runs:
            print(f"  t = {r['t']:5.3f}s  steps={r['n_steps']:5d}  "
                  f"L1={r['L1']:.4f}  Linf={r['Linf']:.4f}  "
                  f"front: num={r['front_num']:6.3f} "
                  f"exact={r['front_exact']:6.3f} "
                  f"err={r['front_err']:+6.3f}  "
                  f"back: num={r['back_num']:6.3f} "
                  f"exact={r['back_exact']:6.3f}")
        print()

    # Side-by-side comparison at the final probe time.
    if not all_results:
        return
    t_final = t_probes[-1]
    print(f"\nh(x) profile at t = {t_final} s — exact + each scheme")
    print(f"  {'x':>5}  {'exact':>8}", end="")
    for key in schemes_keys:
        print(f"  {scheme_defs[key][0][:14]:>14}", end="")
    print()
    # All schemes share the same x bins; take from the first scheme.
    sample = all_results[schemes_keys[0]][1][-1]
    for j in range(0, len(sample["x_bins"]), 4):
        x = sample["x_bins"][j]
        print(f"  {x:5.2f}  {sample['h_exact'][j]:8.4f}", end="")
        for key in schemes_keys:
            print(f"  {all_results[key][1][-1]['h_num'][j]:14.4f}", end="")
        print()


if __name__ == "__main__":
    main()
