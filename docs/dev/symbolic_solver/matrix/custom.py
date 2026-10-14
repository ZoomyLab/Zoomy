"""Runs that are not "march Q0 to t_end": AD (C15), preCICE coupling (C16), restart (C35),
AMR (C17; with diffusion: gate 53) and the wrapped gates 40 / 41 / 43 / 44 / 39 / 45 (C27-C30, C34).  Each
returns {metric: value} (rank 0; None on the other ranks)."""
import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from zoomy_core.numerics import Limiter
from zoomy_core.symsolver.settings import Imex, LinearSolver, Projection
from zoomy_core.systemmodel.system_model import SpectralRadius

GATES = Path(__file__).resolve().parent.parent / "gates"


# ── C15: AD (JAX) ───────────────────────────────────────────────────────────
def _ad_solver(kind, ad_mode, comm):
    import common as C
    from zoomy_jax.symsolver.solver import (SymbolicChorinSolverJax,
                                            SymbolicHyperbolicSolverJax)
    kw = dict(ad_mode=ad_mode, ad_checkpoints=16, comm=comm)
    if kind == "chorin":
        from cases import vam_wave, wave_q0
        import jax.numpy as jnp
        mesh, (pred, press, corr) = vam_wave(1, 30)
        # the Gelfand bound, not eig: the eig tangent at a defective dominant
        # eigenvalue is ~1e15 (the VAM wave here hits one); the CPU's exact tie
        # cancels it, the GPU's split tie does not (NaN; ANALYSIS 44)
        pred.spectral_radius_method, pred._call = SpectralRadius.gelfand, None
        s = SymbolicChorinSolverJax(time_end=0.05, pressure_linear_solver=LinearSolver.gmres, **kw)
        Q, A = s.setup(mesh, pred, press, corr)
        names = [str(v) for v in s.nsm.state]
        Q = jnp.asarray(wave_q0(names, np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
        return s, Q, s.initial_aux(Q, A)
    mesh, nsm = C.bump_2d(2, nx=12, flow=(0.1, 0.05))
    nsm.sm.parameter_values.n = 0.5 if kind == "imex" else 0.03
    nsm.spectral_radius_method, nsm._call = SpectralRadius.eig, None   # as cases.prepare
    s = SymbolicHyperbolicSolverJax(time_end=0.2, imex=Imex.lie if kind == "imex" else False, **kw)
    Q, A = s.setup(mesh, nsm)
    return s, Q, A


def ad(kind, part, comm):
    """``part`` "fwd": J, dJ/dg by jvp and by central FD; "rev": dJ/dg by
    jax.grad.  J = sum over owned cells of h^2 vol; distributed: the ranks'
    shares summed (gate 17)."""
    import jax
    import jax.numpy as jnp
    from zoomy_jax.symsolver.solver import AdMode
    s, Q0, A0 = _ad_solver(kind, AdMode.forward if part == "fwd" else AdMode.reverse, comm)
    h = s.nsm.sm.roles.depth
    n = s.partition.n_own if s.partition else Q0.shape[1]
    total = (lambda x: x) if s.partition is None else (lambda x: comm.allreduce(x))

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[h, :n] ** 2 * s.mesh.cell_volumes[:n])
    p0 = jnp.asarray(s.parameters)
    e = jnp.zeros_like(p0).at[list(s.rt.parameter_names).index("g")].set(1.0)
    if part == "rev":
        g = np.asarray(jax.grad(J)(p0))
        if s.partition is not None:
            g = s.partition.sum_to_global(g, None, None)
        return dict(rev=float(g @ np.asarray(e)))
    Jf = jax.jit(J)
    fwd = total(float(jax.jvp(Jf, (p0,), (e,))[1]))
    fd = [total(float((Jf(p0 + eps * e) - Jf(p0 - eps * e)) / (2 * eps)))
          for eps in (1e-3, 1e-4, 1e-5)]
    return dict(J=total(float(Jf(p0))), fwd=fwd,
                fd_rel=min(abs(x - fwd) for x in fd) / abs(fwd))


def ad3d(variant, part):
    """C22 (gate 3d-e): Euler 3-D wall box 10^3, off-centre pulse, J = sum
    rho^2 |V| at t 0.1, dJ/dgamma; variant o1_dt, o1_adaptive, o2_dt."""
    import jax
    import jax.numpy as jnp
    from cases import pulse
    from zoomy_jax.symsolver.solver import AdMode, SymbolicHyperbolicSolverJax
    order, dt = {"o1_dt": (1, 5e-3), "o1_adaptive": (1, None), "o2_dt": (2, 5e-3)}[variant]
    mesh, nsm = pulse(order, 10, 0.4)
    nsm.spectral_radius_method, nsm._call = SpectralRadius.eig, None
    s = SymbolicHyperbolicSolverJax(time_end=0.1, cfl=0.45, dt=dt, ad_checkpoints=16,
                                    ad_mode=AdMode.forward if part == "fwd" else AdMode.reverse)
    Q0, A0 = s.setup(mesh, nsm)
    V = s.mesh.cell_volumes[:Q0.shape[1]]

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[0] ** 2 * V)
    p0 = jnp.asarray(s.parameters)
    i = list(s.rt.parameter_names).index("gamma")
    e = jnp.zeros_like(p0).at[i].set(1.0)
    if part == "rev":
        return dict(rev=float(jax.grad(J)(p0)[i]))
    Jf = jax.jit(J)
    fwd = float(jax.jvp(Jf, (p0,), (e,))[1])
    fd = [float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e)) / (2 * r * p0[i]))
          for r in (1e-3, 1e-4, 1e-5, 1e-6)]
    return dict(J=float(Jf(p0)), fwd=fwd, fd_rel=min(abs(x - fwd) for x in fd) / abs(fwd))


# ── C16: preCICE (the gate scripts, as subprocesses) ────────────────────────
def _gate(args, timeout=280, **extra):
    env = dict(os.environ, PYTHONPATH=f"{GATES}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
               **extra)
    r = subprocess.run([sys.executable, *args], cwd=GATES, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def couple(pair, ranks):
    """jax|jax (gate 28), foam|foam (gate 26), jax|foam (gate 10, serial),
    amrex|amrex and amrex|jax (gate 38): max|coupled - single domain| and the
    mass drift, O1, window 1e-3."""
    if pair == "amrex_amrex":
        rc, out = _gate(["gate38_amrex_precice.py", "aa", str(ranks), str(ranks)])
    elif pair == "amrex_jax":
        rc, out = _gate(["gate38_amrex_precice.py", "aj", str(ranks)])
    elif pair == "jax_foam":
        if ranks > 1:
            raise NotImplementedError("no jax<->foam driver with MPI ranks (gate 10 is serial)")
        rc, out = _gate(["gate10_xcouple.py", "jax"])
    elif pair == "foam_foam":
        rc, out = _gate(["gate26_foam_couple_parallel.py", str(ranks), str(ranks)])
    else:
        rc, out = _gate(["gate28_jax_couple_parallel.py", "1d", "1", "0.001", "window",
                         str(ranks), str(ranks)])
        if rc == 0:
            from couple_side import vs_single
            return vs_single(np.load(GATES / f"Q_1d_o1_0.001_window_L{ranks}_R{ranks}.npy"))
    m = re.search(r"max\|coupled ?- ?single\| ([\d.e+-]+), mass drift ([\d.e+-]+)", out)
    if rc != 0 or m is None:
        raise RuntimeError(f"gate failed (rc {rc}):\n{out[-2000:]}")
    return dict(coupled_vs_single=float(m.group(1)), mass=float(m.group(2)))


# ── C17: AMReX AMR (gate 37) ────────────────────────────────────────────────
AMR_CONFIG = {"serial": "cpu1", "mpi2": "cpu2", "gpu1": "gpu1"}


def amr(case, config, work):
    """One gate-37 AMR run; its measure (as gate 37's table) and, off cpu1,
    max rel vs the cpu1 run.  The fine single-level reference (cover, wave)
    is made with the cpu1 run."""
    import gate37_amrex_amr as G
    from euler_common import wave_exact
    # one directory for every config: off cpu1 a run is compared with the cpu1 one
    out = Path(work).parents[2] / "serial" / Path(work).parent.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = AMR_CONFIG[config]
    G.run(out, case, cfg)
    r = np.load(out / f"{case}_{cfg}.npz")
    res = {}
    if cfg != "cpu1":
        c1 = np.load(out / f"{case}_cpu1.npz")
        res["vs_cpu1"] = float(G.rel(r["Q"], c1["Q"]))
        return res
    n = G.base_n(case)
    if case.startswith(("cover", "wave")):
        G.ref(out, case)
        f = np.load(out / f"ref_{case}.npz")
    if case.startswith("cover"):
        res["fine_vs_single"] = float(G.rel(r["L1"], f["Qa"]))
    elif case.startswith("wave"):
        t = float(r["t"])
        res["L1"] = float(G.composite(case, n, r["Q"], r["L1"],
                                      lambda Q, x: np.abs(Q[0] - wave_exact(x, t, G.VEL))))
    elif case.startswith("box"):
        res["mass"] = float(abs(r["Q"][0].sum() / r["I0"][0].sum() - 1))
        res["energy"] = float(abs(r["Q"][-1].sum() / r["I0"][-1].sum() - 1))
    elif case == "lake":
        res["u"] = float(max(np.nanmax(np.abs(Q[2:4]) / Q[1]) for Q in (r["Q"], r["L1"])))
    else:
        res["mass"] = float(abs(r["Q"][1].sum() / r["I0"][1].sum() - 1))
        res["h_min"] = float(min(np.nanmin(r["Q"][1]), np.nanmin(r["L1"][1])))
    return res


def amr_diffusion(bc, path, work):
    """Gate 53 (AMReX AMR, the diffusive fluxes refluxed): gate 41's heat equation with BC `bc`
    (neu, per) on path `path` (expl, impl, ssp2): the covering fine level vs the fine single
    level, the composite's mass change with reflux and without (`mass_nr`), and
    L1(AMR - fine single) / L1(coarse single - fine single); `bc` shear: C32's viscous SME,
    the covering fine level vs the fine single level."""
    import gate53_amrex_amr_diffusion as G
    out = Path(work)
    out.mkdir(parents=True, exist_ok=True)
    if bc == "shear":                   # viscous SME (Lie): the covering fine level only
        return dict(fine_vs_single=G.shear_cover(out))
    return G.measures(out, bc, path)


@contextlib.contextmanager
def on_backend(backend, G):
    """Run gate module `G`'s JAX hyperbolic solver as `backend`: jax unchanged, numpy the
    NumPy class (and numpy for its `jnp`)."""
    if backend != "numpy":
        yield
        return
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    old = G.SymbolicHyperbolicSolverJax, getattr(G, "jnp", None)
    G.SymbolicHyperbolicSolverJax = SymbolicHyperbolicSolverNumpy
    if old[1] is not None:
        G.jnp = np
    try:
        yield
    finally:
        G.SymbolicHyperbolicSolverJax = old[0]
        if old[1] is not None:
            G.jnp = old[1]


# ── C27: SSP-RK time order (gate 40) ────────────────────────────────────────
ODE_STEPS = (16, 32, 64)            # gate 40 ``ode``: T = 1, dt = T / n
WAVE_T, WAVE_DT0 = 0.04, 2e-3       # gate 40 ``wave``: dt0 / 2^k, k = 0, 1, 2 vs k = 4


def _to_solver(backend, to, T, dt):
    if backend == "foam":
        from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam as S
    else:
        from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex as S
    return S(time_end=T, dt=dt, mood=False, time_order=to, cfl_check_every=0)


def time_order(backend, part, to, work):
    """Gate 40's forced SWE (``forced_swe``: dh/dt = a cos(w t + 0.3) h, 1-D
    walls, first-order space) marched at constant dt with time_order ``to``;
    ``part`` "ode": 8 cells, T 1, dt T/16, T/32, T/64, error vs the exact
    h(T); "wave": 100 cells, T 0.04, dt 2e-3 / 2^k, error vs k = 4.  JAX runs
    the gate's own ``run``; foam / AMReX the same NSM and mesh through their
    drivers.  Returns the errors and the observed orders (gate 40)."""
    import gate40_time_order as G
    from zoomy_core.mesh import ensure_lsq_mesh
    if part == "ode":
        mesh, nsm, ih = G.forced_swe(1, Limiter.minmod, 8, (0.0, 1.0), lambda x: 1.0)
        T, ks = 1.0, ODE_STEPS
        dts = [T / n for n in ks]
    else:
        mesh, nsm, ih = G.forced_swe(1, Limiter.minmod, 100, (0.0, 1.0),
                                     lambda x: 1.0 + 0.1 * np.exp(-((x - 0.5) / 0.1) ** 2))
        T, ks = WAVE_T, (0, 1, 2, 4)
        dts = [WAVE_DT0 / 2 ** k for k in ks]
    if backend not in ("jax", "numpy"):
        nsm.spectral_radius_method, nsm._call = SpectralRadius.eig, None   # as cases.prepare
        mesh = ensure_lsq_mesh(mesh, nsm)
        names = [str(s) for s in nsm.state]
        xc = np.asarray(mesh.cell_centers)[:, :int(mesh.n_inner_cells)]
        Q0 = np.asarray(nsm.sm.initial_conditions.apply(xc, np.zeros((len(names), xc.shape[1]))),
                        dtype=float)

    def solve(i, dt):
        if backend in ("jax", "numpy"):
            with on_backend(backend, G):
                return G.run(mesh, nsm, T, dt, to)
        return np.asarray(_to_solver(backend, to, T, dt).run(mesh, nsm, Q0, Path(work) / f"s{i}")[1])
    Q = [solve(i, dt) for i, dt in enumerate(dts)]
    extra = {}
    if backend == "numpy":                     # numpy = JAX on every dt
        Qj = [G.run(mesh, nsm, T, dt, to) for dt in dts]
        extra["rel_vs_jax"] = max(float(np.abs(q - j).max() / np.abs(j).max()) for q, j in zip(Q, Qj))
    if part == "ode":
        exact = np.exp(G.A_F * (np.sin(G.W_F * T + G.PHASE) - np.sin(G.PHASE)) / G.W_F)
        errs = [float(abs(float(q[ih, 0]) - exact)) for q in Q]
    else:
        errs = [float(np.abs(q - Q[-1]).max()) for q in Q[:-1]]
    return dict(err=errs, orders=G.orders(errs), order=G.orders(errs)[-1], **extra)


def time_order_control():
    """Gate 40's negative control (JAX): RK3 with the last stage evaluated at
    the wrong time (c = 1, not 1/2) loses the order of the ODE."""
    import gate40_time_order as G
    mesh, nsm, ih = G.forced_swe(1, Limiter.minmod, 8, (0.0, 1.0), lambda x: 1.0)
    exact = np.exp(G.A_F * (np.sin(G.W_F * 1.0 + G.PHASE) - np.sin(G.PHASE)) / G.W_F)
    good = G.rk_time.SSP_RK[3]
    G.rk_time.SSP_RK[3] = (good[0], good[1], (good[2][0], 1))
    try:
        errs = [float(abs(float(G.run(mesh, nsm, 1.0, 1.0 / n, 3)[ih, 0]) - exact)) for n in ODE_STEPS]
    finally:
        G.rk_time.SSP_RK[3] = good
    return dict(err=errs, order=G.orders(errs)[-1])


# ── C28: diffusion blocks at the boundaries (gates 41 and 43) ───────────────
DIFF_CASES = ("dir0", "neu", "dirL", "per")


def diffusion_orders(kind, path, backend="jax"):
    """Gate 41 (JAX, numpy): the L_inf error of the heat equation at nx 16 / 32 / 64
    for dir0, neu, dirL, per on ``kind`` (1d, quad, tri), ``path`` (expl,
    impl), and the steady linear Dirichlet field's residual."""
    import gate41_diffusion_bc as G
    out = {c: [float(G.error(kind, c, path, nx, backend)[0]) for nx in G.NXS] for c in DIFF_CASES}
    out["steady"] = float(G.error(kind, "steady", path, 16, backend)[0])
    return out


def diffusion_run(backend, kind, path, case, work):
    """Gate 43 on ``backend`` alone: gate 41's heat equation, nx 16, the same Q0 on every
    backend; the final field ``u`` (the table compares it with JAX's run) and the L_inf
    error against the exact solution (for ``steady``: the residual max|u - L|)."""
    import gate43_cxx_diffusion as G
    G.WORK = str(work)
    nx = 16
    t, u = G.solve(backend, kind, case, path, nx)
    xc = np.asarray(G.setup(kind, case, path, nx)[0].cell_centers)[:, :len(u)]
    dim = 1 if kind == "1d" else 2
    return dict(err=float(np.abs(u - G.exact(case, dim, xc, t)).max()), t=t, u=u.tolist())


# ── C29: LSQ boundary closure (gate 44) ─────────────────────────────────────
def lsq_closure(p, bc, backend="jax"):
    """Gate 44 on the quads: the max error of the derivatives of order
    k = 1..p over the boundary and the interior cells at nx 12 / 24 / 48, with
    the boundary closure ("closure") and the ghost as a fit sample ("ghost",
    for comparison); degree ``p``, BC kind ``bc`` (dir, neu)."""
    import gate44_lsq_closure as G
    with on_backend(backend, G):
        E = [G.errors("quad", bc, p, nx) for nx in G.NX]
    return {f"{method}_{where}_d{k}": [float(e[(method, k, where)]) for e in E]
            for method in ("closure", "ghost") for where in ("boundary", "interior")
            for k in range(1, p + 1)}


# ── C34: in-plane viscous operator vs the exact div(h tau') (gate 45) ────────
def inplane_operator(backend, nx, work):
    """Gate 45 at ``nx``: the discrete in-plane viscous operator of SME(0) +
    Newtonian with stress=Stress.full (one step nu = 1 minus nu = 0, over dt) against the exact
    div(h tau'), tau' = tau - tau_zz I, the full SystemModel operator
    (mixed-index entries; = div(h tau') symbolically) and its diagonal-index
    part, max rel per momentum row; foam / AMReX also max rel |L - L_JAX|."""
    import gate45_inplane_viscous_operator as G
    L, xc = G.operator(backend, nx, str(work))
    out = G.errors(L, xc, G.sm_operator())
    if backend != "jax":
        Lj, _ = G.operator("jax", nx)
        out["rel_vs_jax"] = max(float(np.abs(L[k] - Lj[k]).max() / np.abs(Lj[k]).max())
                                for k in G.ROWS)
    return out


# ── C30: polynomial reproduction of the LSQ fit (gate 39) ───────────────────
def lsq_polynomials(backend="jax"):
    """Gate 39 as a subprocess: seeded random polynomials of degree p = 1..3
    reproduced to 1e-9 by the LSQ fit (and its boundary closure) on 1-D, quads,
    jittered quads, triangles, hexahedra.  Its SUMMARY line."""
    rc, out = _gate(["gate39_lsq_polynomials.py"], GATE39_BACKEND=backend)
    m = re.search(r"SUMMARY gate39: PASS=(\d+) FAIL=(\d+) \(tol ([\d.e+-]+)\).*inexact in (\d+) cases", out)
    if m is None:
        raise RuntimeError(f"gate 39 gave no summary (rc {rc}):\n{out[-2000:]}")
    return dict(passes=int(m.group(1)), fails=int(m.group(2)), tol=float(m.group(3)),
                sample_inexact=int(m.group(4)))


# ── C35: restart (dt_checkpoint, restart_from) ──────────────────────────────
#: scenario -> (case, run, extra settings); c8: the increment projection (its lagged pressure)
RESTART = {"c1": ("C1", "o2_n40", {}), "c8": ("C8", "vam", {"projection": Projection.increment})}
#: where the half run leaves its state (a work-relative path), per backend
RESTART_STATE = {"jax": "state.h5", "numpy": "state.h5", "foam": ".", "amrex": "out"}


def restart(scenario, part, backend, config, work):
    """C35, one run of the matched pair: `full` 0 -> T with a checkpoint at T/2 (landed on),
    `half` 0 -> T/2, `restart` from the half run's state -> T (max abs(Q - Q_full)), `typo` a
    misspelt `restart_from` (whether it fails at once, naming it)."""
    import dataclasses
    import time
    from backends import BACKENDS
    from cases import CASES, prepare
    case, key, extra = RESTART[scenario]
    r = next(x for x in CASES[case][1] if x.key == key)
    work, T = Path(work), r.t_end
    half = work.parent / f"{scenario}_half"
    kw = dict(extra)
    if backend in ("jax", "numpy"):           # no output file: the restart file named
        kw["checkpoint"] = str(work / "state.h5")
    if part == "full":
        kw["dt_checkpoint"] = T / 2
    elif part == "half":
        T = T / 2
    elif part == "restart":
        kw["restart_from"] = str(half / RESTART_STATE[backend])
    else:                                     # typo: a letter dropped
        kw["restart_from"] = str(half / "stat.h5" if backend in ("jax", "numpy")
                                 else half.with_name(f"{scenario}_hlf"))
    r = dataclasses.replace(r, t_end=T, solver={**(r.solver or {}), **kw})
    mesh, systems, Q0, _ = prepare(r)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        out = BACKENDS[backend]().run(r, mesh, systems, Q0, config, work)
    except FileNotFoundError as e:
        if part != "typo":
            raise
        return {"clear": int("restart_from" in str(e)), "seconds": time.perf_counter() - t0,
                "message": str(e)[:300]}
    if part == "typo":
        return {"clear": 0, "seconds": time.perf_counter() - t0, "message": "no error"}
    if out is None:                           # non-root MPI rank
        return None
    t, Q = out
    np.save(work / "final.npy", Q)
    m = {"t": float(t)}
    if part == "restart":
        m["max_abs"] = float(np.abs(Q - np.load(half.with_name(f"{scenario}_full") / "final.npy"))
                             .max())
    return m
