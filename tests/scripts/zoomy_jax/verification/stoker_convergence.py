"""Stoker wet-wet dam break convergence test.

Pure-wet Riemann problem (h_L > h_R > 0, flat bed, no friction).
Stoker 1957 gives a closed-form analytical solution: a left-going
rarefaction (smooth) and a right-going shock (sharp), separated by a
constant intermediate state ``(h*, u*)``.

The intermediate state is found by matching characteristics:

   2(sqrt(g h_L) - sqrt(g h*)) = (h* - h_R) sqrt(g (h* + h_R) / (2 h* h_R))

Newton iteration in ``h*``.  Then::

   u* = u_L + 2 (sqrt(g h_L) - sqrt(g h*))
   shock speed  s = u_R + sqrt(g h* (h* + h_R) / (2 h_R))

The structure ``h(x, t)`` for ``ξ = (x - x_dam)/t``:

* ``ξ < u_L - sqrt(g h_L)``                — undisturbed left, ``h = h_L``
* ``u_L - sqrt(g h_L) ≤ ξ ≤ u* - sqrt(g h*)`` — rarefaction
* ``u* - sqrt(g h*) < ξ < s``               — constant intermediate
* ``ξ ≥ s``                                 — undisturbed right, ``h = h_R``

This is the canonical convergence test for shallow-water schemes.  In
the **smooth rarefaction** region a 2nd-order spatial scheme + 2nd-
order time integration should give ``O(dx²)`` L¹ error.  Around the
shock the error is O(dx) regardless of scheme order (TVD constraint),
so we sample errors away from the shock or use L¹ norms which average
both regions.

Usage::

    python tests/scripts/zoomy_jax/stoker_convergence.py
    STOKER_NX_LIST="50,100,200,400" \\
        STOKER_SCHEMES="o1 o2_cons" \\
        python tests/scripts/zoomy_jax/stoker_convergence.py
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
from wetdry_dambreak_probe import setup_solver, build_swe_sm  # noqa: E402
from zoomy_jax.mesh.mesh import convert_mesh_to_jax  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402


LX = 50.0
LY = 0.1
NY = 2
X_DAM = 25.0
H_L = 1.0
H_R = 0.1
G = 9.81


def _stoker_hstar(h_L, h_R, g=G, tol=1e-12, max_iter=100):
    """Newton iteration for the intermediate depth ``h*`` in the
    Stoker (1957) wet-wet dam break."""
    cL = np.sqrt(g * h_L)
    # Start somewhere between h_L and h_R.
    h = 0.5 * (h_L + h_R)
    for _ in range(max_iter):
        c = np.sqrt(g * h)
        # Rarefaction side: u = u_L + 2(cL - c).  ``u_L = 0`` here.
        u_raref = 2.0 * (cL - c)
        # Shock side: u_R + (h - h_R) sqrt(g (h + h_R) / (2 h h_R)).
        # ``u_R = 0`` here.
        sqrt_arg = g * (h + h_R) / (2.0 * h * h_R)
        u_shock = (h - h_R) * np.sqrt(sqrt_arg)
        f = u_raref - u_shock
        # Derivative df/dh: derivatives of both parts.
        # d(u_raref)/dh = -2 d(sqrt(g h))/dh = -sqrt(g/h).
        df_raref = -np.sqrt(g / h)
        # d(u_shock)/dh — derivative of (h - h_R)*sqrt(g(h+h_R)/(2 h h_R)).
        # = sqrt(...) + (h - h_R) * d/dh sqrt(...)
        ratio = (h + h_R) / (2.0 * h * h_R)
        d_ratio = (2.0 * h * h_R - (h + h_R) * 2.0 * h_R) / (
            (2.0 * h * h_R) ** 2)
        # Simplify: d_ratio = (h_R - h_R) / ... no wait
        # ratio = (h + h_R)/(2 h h_R).
        # d_ratio/dh = [ (1)(2 h h_R) - (h + h_R)(2 h_R) ] / (2 h h_R)^2
        #            = [ 2 h h_R - 2 h h_R - 2 h_R^2 ] / (2 h h_R)^2
        #            = -2 h_R^2 / (4 h^2 h_R^2) = -1/(2 h^2)
        d_ratio = -1.0 / (2.0 * h ** 2)
        df_shock = np.sqrt(sqrt_arg) + (h - h_R) * 0.5 * np.sqrt(g) * (
            d_ratio / np.sqrt(ratio))
        df = df_raref - df_shock
        dh = -f / df
        h = max(h + dh, h_R + 1e-12)
        if abs(dh) < tol:
            return h
    return h


def stoker_solution(x, t, h_L=H_L, h_R=H_R, g=G, x_dam=X_DAM):
    """Analytical Stoker solution ``(h(x,t), u(x,t))``."""
    if t == 0.0:
        h = np.where(x < x_dam, h_L, h_R)
        u = np.zeros_like(np.asarray(x, dtype=float))
        return h, u
    h_star = _stoker_hstar(h_L, h_R, g=g)
    cL = np.sqrt(g * h_L)
    c_star = np.sqrt(g * h_star)
    u_star = 2.0 * (cL - c_star)         # u_L = 0
    # Shock speed.
    shock_speed = (h_star * u_star) / (h_star - h_R)

    xi = (np.asarray(x, dtype=float) - x_dam) / t

    h = np.full_like(xi, h_L)
    u = np.zeros_like(xi)

    # Region 2: rarefaction.
    mask_raref = (xi > -cL) & (xi <= u_star - c_star)
    h_raref = (2.0 * cL - xi) ** 2 / (9.0 * g)
    u_raref = (2.0 / 3.0) * (cL + xi)
    h = np.where(mask_raref, h_raref, h)
    u = np.where(mask_raref, u_raref, u)

    # Region 3: constant intermediate.
    mask_const = (xi > u_star - c_star) & (xi <= shock_speed)
    h = np.where(mask_const, h_star, h)
    u = np.where(mask_const, u_star, u)

    # Region 4: undisturbed right.
    mask_right = xi > shock_speed
    h = np.where(mask_right, h_R, h)
    u = np.where(mask_right, 0.0, u)

    return h, u


def sample_1d(mesh, Q, n_bins):
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


def set_stoker_ic(solver, Q):
    nc = int(solver._rt_mesh.n_inner_cells)
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc]
    h0 = np.where(xc < X_DAM, H_L, H_R).astype(float)
    Q = Q.at[0, :nc].set(0.0)
    Q = Q.at[1, :nc].set(jnp.asarray(h0, dtype=Q.dtype))
    Q = Q.at[2, :nc].set(0.0)
    Q = Q.at[3, :nc].set(0.0)
    return Q


def make_step(solver, mesh_jax, n_owned, time_scheme="fe"):
    """Return a JIT-compiled time step function.

    ``time_scheme``:
      * ``"fe"``      — forward Euler (1st order in time).
      * ``"ssprk2"``  — Shu-Osher 1988 SSP-RK2 (2nd order, TVD).
    """
    flux_op = solver.get_flux_operator(mesh_jax, solver._rt_model)
    parameters = solver._rt_parameters
    update_var = solver._rt_model.update_variables

    def _stage(dt, t, Q, Qaux):
        dQ = flux_op(dt, t, Q, Qaux, parameters, jnp.zeros_like(Q))
        Q1 = Q.at[:, :n_owned].add(dt * dQ[:, :n_owned])
        if update_var is not None:
            Q1 = Q1.at[:, :n_owned].set(
                update_var(Q1[:, :n_owned], Qaux[:, :n_owned], parameters))
        return Q1

    if time_scheme == "fe":
        @jax.jit
        def step(dt, t, Q, Qaux):
            return _stage(dt, t, Q, Qaux)
        return step

    if time_scheme == "ssprk2":
        # SSP-RK2 / Heun:
        #   Q1 = Q + dt F(Q, t)
        #   Q2 = Q1 + dt F(Q1, t+dt)
        #   Q^{n+1} = (Q + Q2) / 2
        @jax.jit
        def step(dt, t, Q, Qaux):
            Q1 = _stage(dt, t, Q, Qaux)
            Q2 = _stage(dt, t + dt, Q1, Qaux)
            return 0.5 * (Q + Q2)
        return step
    raise ValueError(time_scheme)


def run_to_time(solver, Q, Qaux, t_end, cfl, time_scheme):
    mesh_jax = convert_mesh_to_jax(solver._rt_mesh)
    n_owned = int(solver._rt_mesh.n_inner_cells)
    min_inradius = float(np.min(np.asarray(mesh_jax.cell_inradius)))
    eig_op = solver.get_compute_max_abs_eigenvalue(
        mesh_jax, solver._rt_model)
    parameters = solver._rt_parameters
    step = make_step(solver, mesh_jax, n_owned, time_scheme=time_scheme)

    # Warmup
    me = float(eig_op(Q, Qaux, parameters))
    dt0 = jnp.asarray(cfl * min_inradius / max(me, 1e-12), dtype=Q.dtype)
    Q = step(dt0, jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux)
    Q.block_until_ready()

    t_phys = 0.0
    n_steps = 0
    while t_phys < t_end:
        me = float(eig_op(Q, Qaux, parameters))
        dt = cfl * min_inradius / max(me, 1e-12)
        dt = min(dt, t_end - t_phys)
        Q = step(jnp.asarray(dt, dtype=Q.dtype),
                 jnp.asarray(t_phys, dtype=Q.dtype), Q, Qaux)
        t_phys += dt
        n_steps += 1
    Q.block_until_ready()
    return Q, t_phys, n_steps


def run_one_case(nx, order, mode, cfl, t_end, time_scheme):
    mesh = LSQMesh.create_2d((0.0, LX, 0.0, LY), nx, NY)
    solver, Q, Qaux, sm = setup_solver(mesh, order=order,
                                        reconstruction_mode=mode,
                                        limiter="minmod")
    Q = set_stoker_ic(solver, Q)
    t0 = time.perf_counter()
    Q, t_now, n_steps = run_to_time(solver, Q, Qaux, t_end=t_end,
                                     cfl=cfl, time_scheme=time_scheme)
    wall = time.perf_counter() - t0

    # Cell-based L¹ norm: ∫|h_num - h_exact| dx ≈ Σ_cells |h_c - h_ex(x_c)| * (cell_vol / LY).
    # (cell volumes have units m² for our 2D thin channel; dividing by
    # LY collapses them to per-x-length contributions, which is what
    # ``∫|·| dx`` is.)  Avoids the empty-bin-reads-zero artifact of
    # histogram-based sampling on irregular triangle meshes.
    nc_owned = int(solver._rt_mesh.n_inner_cells)
    xc = np.asarray(solver._rt_mesh.cell_centers)[0, :nc_owned]
    h_cell = np.asarray(Q)[1, :nc_owned]
    cell_vols = np.asarray(solver._rt_mesh.cell_volumes)[:nc_owned]
    h_exact_cell, _ = stoker_solution(xc, t_now)
    err_cell = np.abs(h_cell - h_exact_cell)
    L1 = float(np.sum(err_cell * cell_vols) / LY)            # ∫|err| dx
    Linf = float(np.max(err_cell))

    # Rarefaction-only L¹ (smooth region, where O2 should beat O1).
    h_star = _stoker_hstar(H_L, H_R)
    c_star = np.sqrt(G * h_star)
    u_star = 2.0 * (np.sqrt(G * H_L) - c_star)
    raref_left = X_DAM + (-np.sqrt(G * H_L)) * t_now
    raref_right = X_DAM + (u_star - c_star) * t_now
    in_raref = (xc > raref_left + 0.5) & (xc < raref_right - 0.5)
    if in_raref.any():
        L1_raref = float(np.sum(
            err_cell[in_raref] * cell_vols[in_raref]) / LY)
    else:
        L1_raref = float("nan")

    # Also build a 1-D x-binned profile for diagnostic print (uses bins
    # large enough that every bin has at least one cell).
    n_bins = max(nx, 40)
    x_b, h_b, hu_b = sample_1d(solver._rt_mesh, Q, n_bins=n_bins)

    return dict(
        nx=nx, dx=LX/nx, n_steps=n_steps, wall=wall,
        L1=L1, Linf=Linf, L1_raref=L1_raref, t_end=t_now,
        x_b=x_b, h_num=h_b,
    )


def main():
    nx_list = [int(x) for x in
               os.environ.get("STOKER_NX_LIST", "50,100,200,400").split(",")]
    schemes_arg = os.environ.get(
        "STOKER_SCHEMES",
        "o1_fe o1_rk2 o2_cons_fe o2_cons_rk2 o2_eta_fe o2_eta_rk2")
    schemes = schemes_arg.split()
    t_end = float(os.environ.get("STOKER_T_END", "1.0"))
    cfl_o1 = float(os.environ.get("STOKER_CFL_O1", "0.5"))
    cfl_o2 = float(os.environ.get("STOKER_CFL_O2", "0.25"))

    scheme_defs = {
        # key:           label,            order, mode,            cfl,    time
        "o1_fe":       ("O1 FE",           1,     "conservative",  cfl_o1, "fe"),
        "o1_rk2":      ("O1 RK2",          1,     "conservative",  cfl_o1, "ssprk2"),
        "o2_cons_fe":  ("O2 cons FE",      2,     "conservative",  cfl_o2, "fe"),
        "o2_cons_rk2": ("O2 cons RK2",     2,     "conservative",  cfl_o2, "ssprk2"),
        "o2_eta_fe":   ("O2 η-MUSCL FE",   2,     "eta",           cfl_o2, "fe"),
        "o2_eta_rk2":  ("O2 η-MUSCL RK2",  2,     "eta",           cfl_o2, "ssprk2"),
    }

    h_star = _stoker_hstar(H_L, H_R)
    u_star = 2.0 * (np.sqrt(G * H_L) - np.sqrt(G * h_star))
    shock_speed = (h_star * u_star) / (h_star - H_R)
    print(f"Stoker wet-wet dam break  (channel [0,{LX}], dam at "
          f"{X_DAM}, h_L={H_L}, h_R={H_R})")
    print(f"  intermediate state h* = {h_star:.6f}  u* = {u_star:.6f}")
    print(f"  shock speed s = {shock_speed:.6f}")
    print(f"  t_end = {t_end}\n")

    for key in schemes:
        if key not in scheme_defs:
            print(f"  unknown scheme {key}")
            continue
        label, order, mode, cfl, ts = scheme_defs[key]
        print(f"=== {label}  (order={order}, mode={mode}, cfl={cfl}, "
              f"time={ts}) ===")
        results = []
        for nx in nx_list:
            r = run_one_case(nx, order, mode, cfl, t_end, ts)
            results.append(r)
            print(f"  nx={r['nx']:>5}  dx={r['dx']:.4f}  steps={r['n_steps']:>5}  "
                  f"L1={r['L1']:.4e}  Linf={r['Linf']:.4e}  "
                  f"L1_raref={r['L1_raref']:.4e}  wall={r['wall']:.1f}s")
        # Convergence rates (between successive resolutions).
        print(f"  {'(rates)':>12}                              ", end="")
        for i in range(1, len(results)):
            r_prev = results[i - 1]
            r_curr = results[i]
            if r_curr["L1"] > 0 and r_prev["L1"] > 0:
                rate_L1 = np.log(r_prev["L1"] / r_curr["L1"]) / np.log(2.0)
                rate_raref = (
                    np.log(r_prev["L1_raref"] / r_curr["L1_raref"])
                    / np.log(2.0))
                print(f"  L1={rate_L1:+.2f} raref={rate_raref:+.2f}", end="")
        print()
        print()


if __name__ == "__main__":
    main()
