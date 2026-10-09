"""CFL wave speeds: 'cells' (eigensolve on the cell states of each face, as
before) vs 'faces' (the first RK stage's Riemann wave speed, no eigensolve of
its own).  Same solver otherwise; second run timed, min of 3 alternating.
Steps counted in a separate run (a host callback per step).

    python bench_wave_speed.py [case ...]
"""
import os
import sys
import time as wall

import common as C
import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cases"))
import zoomy_jax.symsolver.runtime as RT  # noqa: E402

REPS = int(os.environ.get("REPS", 3))
_orig = RT.runtime
_count = [0]


def _counting(*a, **k):
    """runtime() whose reduce_min (the CFL step, once per step) ticks a host
    counter."""
    m = _orig(*a, **k)
    rmin = m["reduce_min"]

    def reduce_min(x):
        r = rmin(x)
        jax.debug.callback(lambda: _count.__setitem__(0, _count[0] + 1))
        return r
    m["reduce_min"] = reduce_min
    return m


def hyperbolic(make, t_end, cfl, **kw):
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

    def build(ws):
        mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, cfl_wave_speed=ws, **kw)
        Q, A = s.setup(mesh, nsm)
        return lambda: s.run(Q, A)
    return build


def chorin(dt=True, every=1, time_order=1):
    import gate7_chorin as g7
    import escalante_bump as EB
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

    def build(ws):
        pred, press, corr = EB.split(_DT_SYMBOL)
        s = SymbolicChorinSolverJax(time_end=EB.T_END / 4, dt=EB.DT if dt else None, cfl=0.9,
                                    pressure_tol=1e-9, pressure_maxit=200,
                                    cfl_check_every=every, time_order=time_order,
                                    cfl_wave_speed=ws)
        Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                       g7.pred_nsm(pred, press, corr), press, corr)
        xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
        Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
        A = s.initial_aux(Q, A)
        return lambda: s.run(Q, A)
    return build


def malpasset(imex):
    def make():
        sys.modules.pop("run", None)      # the bump case's run.py shadows it
        import malpasset as MP
        mesh, b = MP.build()
        return mesh, b["nsm"]
    return hyperbolic(make, 0.5, 0.4, imex=imex)


CASES = {
    "sme1d": ("SME(1) 1-D dam break O2, 200 cells, t=0.1",
              hyperbolic(lambda: C.dambreak_1d(1, 2), 0.1, 0.9, mood=False)),
    "swe2d": ("SWE 2-D bump O2, 80x80, t=0.5",
              hyperbolic(lambda: C.bump_2d(2, nx=80), 0.5, 0.9, mood=False)),
    "dry": ("radial dry dam break O2 eta+MOOD, 48x48",
            hyperbolic(C.dry_dambreak_2d, 0.02, 0.45,
                       mood=True)),
    "imex": ("IMEX SME(1) 1-D dam break O2, 100 cells",
             hyperbolic(__import__("gate6_imex").CASES[0][1], 0.1, 0.9, imex=True,
                        mood=False)),
    "chorin_check1": ("Chorin VAM(1) bump, const dt, check every step", chorin(True, 1)),
    "chorin_check10": ("Chorin VAM(1) bump, const dt, check every 10th", chorin(True, 10)),
    "chorin_adaptive": ("Chorin VAM(1) bump, adaptive dt", chorin(False)),
    "chorin_adaptive_rk2": ("Chorin VAM(1) bump, adaptive dt, time_order 2",
                            chorin(False, time_order=2)),
    "malpasset": ("Malpasset SME(1) O1, 26k cells, t=0.5", malpasset(False)),
    "malpasset_imex": ("Malpasset SME(1) O1, IMEX, t=0.5", malpasset(True)),
}


def steps(build, ws):
    RT.runtime = _counting
    try:
        f = build(ws)
        _count[0] = 0
        out = f()
        jax.block_until_ready(out)
        jax.effects_barrier()
        return _count[0], out
    finally:
        RT.runtime = _orig


if __name__ == "__main__":
    print(f"device {jax.devices()[0].platform}")
    for key in sys.argv[1:] or list(CASES):
        name, build = CASES[key]
        runs = {}
        for ws in ("cells", "faces"):
            f = build(ws)
            t0 = wall.perf_counter()
            out = f()
            jax.block_until_ready(out)
            runs[ws] = (f, out, wall.perf_counter() - t0)
        best = {ws: 1e9 for ws in runs}
        for _ in range(REPS):
            for ws, (f, _, _) in runs.items():
                t0 = wall.perf_counter()
                jax.block_until_ready(f())
                best[ws] = min(best[ws], wall.perf_counter() - t0)
        n = {ws: steps(build, ws)[0] for ws in runs}
        Qc, Qf = (np.asarray(runs[ws][1][1]) for ws in ("cells", "faces"))
        tc, tf = (float(runs[ws][1][0]) for ws in ("cells", "faces"))
        d = np.abs(Qc - Qf).max()
        print(f"{name:48s} cells {best['cells']:7.3f}s {n['cells']:5d} st "
              f"{1e3 * best['cells'] / n['cells']:7.3f} ms/st | faces {best['faces']:7.3f}s "
              f"{n['faces']:5d} st {1e3 * best['faces'] / n['faces']:7.3f} ms/st | "
              f"speedup {best['cells'] / best['faces']:5.2f}x per-step "
              f"{(best['cells'] / n['cells']) / (best['faces'] / n['faces']):5.2f}x | "
              f"t {tc:.4g}/{tf:.4g} max|c-f| {d:.1e} (rel {d / np.abs(Qc).max():.1e}) "
              f"| load {os.getloadavg()[0]:.1f}", flush=True)
    print("BENCH_WS_DONE")
