"""ITEM 1: Select / Guard / Callback / observer on the REAL emitted 1-D SWE
step; whole march traced (jit + eqxi checkpointed While); AD; bitwise vs the
host march (EmittedMarchSolver.march) and the reference (MarchSolver.march).
"""
import logging
import sys
import time as _time

from common import *  # noqa
import numpy as np
import jax
import jax.numpy as jnp
from zoomy_core.mesh import LSQMesh
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.march_emitted import EmittedMarchSolver
from march_ext import build_callables, traced_bodies
from p0_probe_mood3 import radial1d_nsm

# ── compile counter ─────────────────────────────────────────────────────────
COMPILES = []


class _H(logging.Handler):
    def emit(self, rec):
        msg = rec.getMessage()
        if msg.startswith("Compiling "):
            COMPILES.append(msg.split(" ")[1])


jax.config.update("jax_log_compiles", True)
for name in ("jax._src.interpreters.pxla", "jax._src.dispatch", "jax"):
    lg = logging.getLogger(name)
    lg.addHandler(_H())
    lg.setLevel(logging.WARNING)
    lg.propagate = False


def setup(case):
    name = case["name"]
    if name == "radial":
        def mk(cls):
            nsm = radial1d_nsm(2, case["limiter"])
            mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=case["n"])
            s = cls(time_end=case["t_end"], mood_redo=True)
            S = s.setup_march(mesh, nsm, CFL=CFL)
            return s, S, nsm, mesh
    else:
        def mk(cls):
            return make_solver(cls, name, case["order"], n_cells=case["n"],
                               t_end=case["t_end"], recon="conservative")
    return mk


def radial_t_end(n, limiter, n_ok):
    """time after n_ok-1 steps + half of step n_ok (last step clamped)."""
    nsm = radial1d_nsm(2, limiter)
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
    s = MarchSolver(time_end=1.0, mood_redo=True)
    S = s.setup_march(mesh, nsm, CFL=CFL)
    p = s._rt_parameters
    ts = [0.0]
    for _ in range(n_ok):
        lo, hi, dt = s.step_head(S, p, 1.0 - S.time, None)
        S, _ = s.hyperbolic_step(S, float(dt), p)
        ts.append(float(S.time))
    return 0.5 * (ts[-2] + ts[-1])


def run_case(case):
    print(f"\n=== {case['label']} ===", flush=True)
    mk = setup(case)
    t_end, wi = case["t_end"], case["wi"]
    # reference (hand-written solver2 march)
    ref, S, nsm, mesh = mk(MarchSolver)
    nc = mesh.n_inner_cells
    dx = float(np.asarray(mesh.cell_volumes)[0])
    snaps_ref = []
    Sr, st = ref.march(S, write_interval=wi,
                       on_write=lambda S_: snaps_ref.append(
                           (float(S_.time), np.asarray(S_.Q))))
    # host emitted march
    em, S2, _, _ = mk(EmittedMarchSolver)
    snaps_em = []
    Se, ste = em.march(S2, write_interval=wi,
                       on_write=lambda S_: snaps_em.append(
                           (float(S_.time), np.asarray(S_.Q))))
    h_i = [str(s) for s in nsm.state].index("h")
    gauge = int(case["gauge_x"] / dx)
    Q0, Qaux0, p0 = S2.Q, S2.Qaux, em._rt_parameters
    order = int(em.Ops.order)
    mood = bool(em.mood_redo)

    # ── ext lowering, numpy walker (host loop) ─────────────────────────────
    snaps_np = []
    b_np = traced_bodies(em, gauge_cell=gauge, h_index=h_i,
                         on_write=lambda t, i, Q: snaps_np.append((t, np.asarray(Q))))
    march_np, step_proc, march_proc = build_callables(
        em, em.program.constants, em.program.flags, order, mood=mood,
        backend="numpy", bodies=b_np)
    env = march_np(Q=Q0, Qaux=Qaux0, p=p0, time=0.0, iteration=0,
                   i_snapshot=0, t_end=float(t_end), dt_window=None,
                   write_interval=wi, inradius_f=em.MeshRT.inradius_f,
                   n_faces=em.MeshRT.n_faces, obs=0.0)
    Q_np, obs_np = np.asarray(env["Q"]), float(env["obs"])

    # ── ext lowering, jax walker: WHOLE march traced ───────────────────────
    snaps_jx = []
    calls = {"n": 0}

    def rec(t, i, Q):
        calls["n"] += 1
        if rec.on:
            snaps_jx.append((t, np.asarray(Q)))
    rec.on = True
    b_jx = traced_bodies(em, gauge_cell=gauge, h_index=h_i, on_write=rec,
                         safe_dt=SAFE_DT)
    march_rev, _, _ = build_callables(
        em, em.program.constants, em.program.flags, order, mood=mood,
        backend="jax", bodies=b_jx, checkpoints=case.get("ckpt", 4))
    march_fwd, _, _ = build_callables(
        em, em.program.constants, em.program.flags, order, mood=mood,
        backend="jax", bodies=b_jx, while_kind="lax")
    MeshRT = em.MeshRT

    def run(Q, Qaux, p, t_end_, wi_, march_jx=march_rev):
        e = march_jx(Q=Q, Qaux=Qaux, p=p, time=jnp.asarray(0.0),
                     iteration=jnp.asarray(0), i_snapshot=jnp.asarray(0),
                     t_end=t_end_, dt_window=None, write_interval=wi_,
                     inradius_f=MeshRT.inradius_f, n_faces=MeshRT.n_faces,
                     obs=jnp.asarray(0.0))
        return e["Q"], e["time"], e["iteration"], e["obs"]
    run_jit = jax.jit(run)
    n_c0 = len(COMPILES)
    t0 = _time.perf_counter()
    Qj, tj, itj, obsj = jax.block_until_ready(
        run_jit(Q0, Qaux0, p0, jnp.asarray(t_end), jnp.asarray(wi)))
    t_cold = _time.perf_counter() - t0
    calls_fwd = calls["n"]
    rec.on = False
    t0 = _time.perf_counter()
    out2 = jax.block_until_ready(run_jit(Q0, Qaux0, p0, jnp.asarray(t_end),
                                         jnp.asarray(wi)))
    t_warm = _time.perf_counter() - t0
    # different t_end / write_interval: must NOT recompile
    jax.block_until_ready(run_jit(Q0, Qaux0, p0, jnp.asarray(0.5 * t_end),
                                  jnp.asarray(0.5 * wi)))
    n_compiles_run = sum(1 for c in COMPILES[n_c0:] if c == "jit(run)")
    t0 = _time.perf_counter()
    Sx, _ = em.march(S2)
    t_host = _time.perf_counter() - t0
    ref.march(S)                      # warm the reference (compiled once)
    t0 = _time.perf_counter()
    ref.march(S)
    t_ref = _time.perf_counter() - t0

    Qr, Qe = np.asarray(Sr.Q), np.asarray(Se.Q)
    Qj = np.asarray(Qj)
    print(f"steps: ref {st['steps']} host-emitted {ste['steps']} "
          f"ext-np {int(env['iteration'])} ext-jax {int(itj)}; "
          f"MOOD-fired steps (ref) {st['troubled_steps']}")
    print(f"bitwise final Q: jax==host-emitted {bool((Qj == Qe).all())} "
          f"jax==ref {bool((Qj == Qr).all())} np-ext==host {bool((Q_np == Qe).all())} "
          f"max|jax-ref| {np.abs(Qj - Qr).max():.3e}")
    print(f"observer: np {obs_np!r} jax {float(obsj)!r} equal {obs_np == float(obsj)}")
    ok_snap = (len(snaps_jx) == len(snaps_em) == len(snaps_np) == len(snaps_ref)
               and all(a[0] == b[0] and (a[1] == b[1]).all()
                       for a, b in zip(snaps_jx, snaps_em))
               and all(a[0] == b[0] and (a[1] == b[1]).all()
                       for a, b in zip(snaps_jx, snaps_ref)))
    print(f"snapshots: ref {len(snaps_ref)} host {len(snaps_em)} np {len(snaps_np)} "
          f"jax {len(snaps_jx)} times {[round(s[0], 4) for s in snaps_jx]} "
          f"all bitwise {ok_snap}")
    h = Qj[h_i, :nc]
    h0 = np.asarray(Q0)[h_i, :nc]
    print(f"mass drift {(h.sum() - h0.sum()) * dx:.3e}  hmin {h.min():.3e} "
          f"(ref hmin {Qr[h_i, :nc].min():.3e})")
    print(f"compiles of jitted march over 3 calls (2 t_end/wi values): "
          f"{n_compiles_run}; cache_size {run_jit._cache_size()}")
    print(f"wall: jax cold {t_cold:.2f}s warm {t_warm*1e3:.1f}ms  "
          f"host-emitted (re-jits per call) {t_host*1e3:.1f}ms  ref MarchSolver warm {t_ref*1e3:.1f}ms  callback calls fwd {calls_fwd}")

    # ── AD: J(p) = sum h^2 + observer, direction e_g ───────────────────────
    pn = list(em.nsm.parameter_values.as_dict())
    ig = pn.index("g")
    eg = jnp.zeros_like(p0).at[ig].set(1.0)
    T = jnp.asarray(t_end)
    W = jnp.asarray(wi)

    def J(p, mj=march_rev):
        Q, _, _, obs = run(Q0, Qaux0, p, T, W, mj)
        return jnp.sum(Q[h_i, :nc] ** 2) + obs

    Jf = lambda p: J(p, march_fwd)  # noqa: E731
    Jj = jax.jit(Jf)
    Qf = np.asarray(jax.jit(lambda: run(Q0, Qaux0, p0, T, W, march_fwd)[0])())
    print(f"fwd-variant (lax while) final Q bitwise == checkpointed: "
          f"{bool((Qf == Qj).all())}")
    calls["n"] = 0
    _, d = jax.jit(lambda p: jax.jvp(Jf, (p,), (eg,)))(p0)
    c_jvp = calls["n"]
    calls["n"] = 0
    g = jax.jit(jax.grad(J))(p0)
    c_grad = calls["n"]
    res = []
    for eps in (1e-4, 1e-5, 1e-6):
        fd = (Jj(p0 + eps * eg) - Jj(p0 - eps * eg)) / (2 * eps)
        res.append((eps, float(fd), abs(float(fd) - float(d)) / abs(float(d))))
    print(f"AD: jvp {float(d):.12e} grad[g] {float(g[ig]):.12e} "
          f"rel(grad-jvp) {abs(float(g[ig]) - float(d)) / abs(float(d)):.2e}")
    for eps, fd, r in res:
        print(f"    FD eps {eps:.0e}: {fd:.12e} rel {r:.2e}")
    print(f"callback host calls: fwd {calls_fwd} jvp {c_jvp} grad {c_grad} "
          f"(writes per run {len(snaps_jx)})")
    # Q0-direction check (reverse vs forward)
    v = jax.random.normal(jax.random.PRNGKey(0), Q0.shape, dtype=Q0.dtype)
    v = v * (np.asarray(Q0) != 0)   # perturb only where the state is nonzero

    def JQ(q, mj=march_rev):
        Q, _, _, obs = run(q, Qaux0, p0, T, W, mj)
        return jnp.sum(Q[h_i, :nc] ** 2) + obs
    _, dv = jax.jit(lambda q: jax.jvp(lambda x: JQ(x, march_fwd), (q,), (v,)))(Q0)
    gq = jax.jit(jax.grad(JQ))(Q0)
    print(f"AD Q0-dir: jvp {float(dv):.12e} <grad,v> {float(jnp.vdot(gq, v)):.12e} "
          f"rel {abs(float(jnp.vdot(gq, v)) - float(dv)) / abs(float(dv)):.2e}")
    return dict(steps=int(itj))


def guard_tests():
    print("\n=== (b) Guard: FATAL dt guard on the 1-D radial minmod blow-up ===")
    case = dict(name="radial", limiter="minmod", n=48, t_end=0.05)
    mk = setup(case)
    em, S, nsm, mesh = mk(EmittedMarchSolver)
    try:
        em.march(S)
        print("host emitted: NO error (unexpected)")
    except FloatingPointError as e:
        print("host emitted raises:", str(e)[:110])
    h_i = 1
    for backend in ("numpy", "jax"):
        b = traced_bodies(em, gauge_cell=24, h_index=h_i)
        march_fn, _, _ = build_callables(em, em.program.constants,
                                         em.program.flags, 2, mood=True,
                                         backend=backend, bodies=b,
                                         write=False)
        kw = dict(Q=S.Q, Qaux=S.Qaux, p=em._rt_parameters,
                  time=jnp.asarray(0.0), iteration=jnp.asarray(0),
                  i_snapshot=jnp.asarray(0), t_end=jnp.asarray(0.05),
                  dt_window=None, write_interval=1e300,
                  inradius_f=em.MeshRT.inradius_f, n_faces=em.MeshRT.n_faces,
                  obs=jnp.asarray(0.0))
        f = (jax.jit(lambda **k: march_fn(**k)["Q"]) if backend == "jax"
             else (lambda **k: march_fn(**k)["Q"]))
        try:
            jax.block_until_ready(f(**kw))
            print(f"ext {backend}: NO error (unexpected)")
        except Exception as e:  # noqa
            msg = [l for l in str(e).splitlines() if "guard" in l]
            print(f"ext {backend} raises {type(e).__name__}: "
                  f"{(msg[0] if msg else str(e).splitlines()[0])[:140]}")
        if backend == "jax":
            # also under jvp (lax While variant) and grad (checkpointed)
            march_lax, _, _ = build_callables(
                em, em.program.constants, em.program.flags, 2, mood=True,
                backend="jax", bodies=b, write=False, while_kind="lax")

            def J(p, mf=march_fn):
                kw2 = dict(kw, p=p)
                return jnp.sum(mf(**kw2)["Q"][1] ** 2)
            p0 = em._rt_parameters
            for nm, fn in (("jvp", lambda: jax.jvp(lambda q: J(q, march_lax),
                                                   (p0,), (p0,))),
                           ("grad", lambda: jax.grad(J)(p0))):
                try:
                    jax.block_until_ready(jax.jit(fn)())
                    print(f"   under {nm}: NO error (unexpected)")
                except Exception as e:  # noqa
                    print(f"   under {nm}: raises {type(e).__name__}; "
                          f"static msg in exception: {'FATAL dt guard' in str(e)}")

    print("\n=== (b') Guard: progress guard (max_steps=2) on Stoker o1 ===")
    em, S, nsm, mesh = make_solver(EmittedMarchSolver, "stoker", 1,
                                   n_cells=200, t_end=2.0)
    for backend in ("numpy", "jax"):
        b = traced_bodies(em, gauge_cell=120, h_index=1)
        march_fn, _, _ = build_callables(em, em.program.constants,
                                         em.program.flags, 1, mood=False,
                                         backend=backend, bodies=b,
                                         write=False, max_steps=2)
        kw = dict(Q=S.Q, Qaux=S.Qaux, p=em._rt_parameters,
                  time=jnp.asarray(0.0), iteration=jnp.asarray(0),
                  i_snapshot=jnp.asarray(0), t_end=jnp.asarray(2.0),
                  dt_window=None, write_interval=1e300,
                  inradius_f=em.MeshRT.inradius_f, n_faces=em.MeshRT.n_faces,
                  obs=jnp.asarray(0.0))
        f = (jax.jit(lambda **k: march_fn(**k)["Q"]) if backend == "jax"
             else (lambda **k: march_fn(**k)["Q"]))
        try:
            jax.block_until_ready(f(**kw))
            print(f"ext {backend}: NO error (unexpected)")
        except Exception as e:  # noqa
            msg = [l for l in str(e).splitlines() if "guard" in l]
            print(f"ext {backend} raises {type(e).__name__}: "
                  f"{(msg[0] if msg else str(e).splitlines()[0])[:140]}")


SAFE_DT = False

if __name__ == "__main__":
    which = sys.argv[1:] or ["stoker", "ritter", "radial", "guard"]
    if "safe" in which:
        SAFE_DT = True
        print("*** traced reduce_dt body = double-where safe divide ***")
    cases = []
    if "stoker" in which:
        cases.append(dict(label="Stoker o1, 200 cells, t=2", name="stoker",
                          order=1, n=200, t_end=2.0, wi=0.5, gauge_x=6.0))
    if "ritter" in which:
        cases.append(dict(label="Ritter o2 (conservative recon, MOOD), 100 cells, t=1",
                          name="ritter", order=2, n=100, t_end=1.0, wi=0.25,
                          gauge_x=5.5))
    if "radial" in which:
        te = radial_t_end(48, "venkatakrishnan", 3)
        cases.append(dict(label=f"1-D radial dry dam break o2 VK (MOOD), 48 cells, t={te:.5f}",
                          name="radial", limiter="venkatakrishnan", n=48,
                          t_end=te, wi=te / 3, gauge_x=0.6))
    for c in cases:
        run_case(c)
    if "guard" in which:
        guard_tests()
