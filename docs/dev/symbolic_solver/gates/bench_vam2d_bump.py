"""Scaling on the CPU: the Escalante flow over a bump (VAM(1), Chorin,
inflow q = Q_IN left, P = 0 right) extruded into a 2-D channel
3 m x 0.3 m with square cells, pressure solves by PyAMG + GMRES ('auto').
Per mesh: wall time of a short run (compile excluded), steps, seconds per
simulated second -> the largest mesh that simulates T_TARGET in 60 s.

    python bench_vam2d_bump.py [T_RUN] [nx ...]
"""
import os
import sys
import time as wall

import common  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np
import zoomy_jax.symsolver.runtime as RT
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
from zoomy_core.model.initial_conditions import Constant
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from gate7_chorin import pred_nsm

H_RES, H_DRY, Q_IN = 0.34, 0.015, 0.11197
BUMP = lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))      # noqa: E731
T_RUN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
NXS = [int(a) for a in sys.argv[2:]] or [60, 120, 240]
SOLVER = os.environ.get("SOLVER", "auto")

_steps = [0]
_orig = RT.runtime


def counting(*a, **k):                    # one reduce_min per step (CFL)
    m = _orig(*a, **k)
    rmin = m["reduce_min"]

    def reduce_min(x):
        jax.debug.callback(lambda: _steps.__setitem__(0, _steps[0] + 1))
        return rmin(x)
    m["reduce_min"] = reduce_min
    return m


RT.runtime = counting
if os.environ.get("SPECTRAL"):            # eig | power | bound
    SystemModel.spectral_radius_method = os.environ["SPECTRAL"]
_amg_t = {"setup": 0.0, "solve": 0.0, "iters": 0, "calls": 0, "setups": 0}


def _timed_pyamg():                       # where the host time goes
    import pyamg

    class P:
        def smoothed_aggregation_solver(self, A, **kw):
            t0 = wall.perf_counter()
            ml = pyamg.smoothed_aggregation_solver(A, **kw)
            _amg_t["setup"] += wall.perf_counter() - t0
            _amg_t["setups"] += 1
            solve = ml.solve

            def timed_solve(b, **kw):
                t0 = wall.perf_counter()
                res = []
                x = solve(b, residuals=res, **kw)
                _amg_t["solve"] += wall.perf_counter() - t0
                _amg_t["iters"] += len(res)
                _amg_t["calls"] += 1
                return x
            ml.solve = timed_solve
            return ml
    return P()


RT._pyamg = _timed_pyamg


def split():
    bcs = ([Dirichlet("left", on=f, value=v) for f, v in
            (("q_x_0", Q_IN), ("q_x_1", 0.0), ("r_0", 0.0), ("r_1", 0.0))]
           + [Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
           + [Extrapolation(tag=t) for t in ("bottom", "top")])
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    return r.SM_pred, r.SM_press, r.SM_corr


print(f"VAM(1) bump channel, {jax.devices()[0].platform}, spectral radius "
      f"{SystemModel.spectral_radius_method}, CPU ({os.cpu_count()} cores visible: "
      f"{len(os.sched_getaffinity(0))}), pressure solver {SOLVER}, run to t = {T_RUN}")
for nx in NXS:
    ny = max(1, nx // 10)
    pred, press, corr = split()
    kw = {k: float(v) if "." in v or "e" in v else int(v) for k, v in
          (a.split("=") for a in os.environ.get("PRESSURE", "").split(",") if a)}
    s = SymbolicChorinSolverJax(time_end=T_RUN, cfl=0.9, pressure_linear_solver=SOLVER, **kw)
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=nx, ny=ny)
    Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    q = np.zeros(Q.shape)
    b = BUMP(xc)
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    Q = jnp.asarray(q)
    A = s.initial_aux(Q, A)
    t0 = wall.perf_counter()
    jax.block_until_ready(s.run(Q, A))
    first = wall.perf_counter() - t0
    _steps[0] = 0
    _amg_t.update(setup=0.0, solve=0.0, iters=0, calls=0, setups=0)
    t0 = wall.perf_counter()
    t, Qe, _ = s.run(Q, A)
    jax.block_until_ready(Qe)
    if os.environ.get("SAVE"):
        f = f"{os.environ['SAVE']}_{nx}.npy"
        if os.path.exists(f):
            ref = np.load(f)
            d = np.abs(np.asarray(Qe) - ref).max(axis=1)
            print("      max|Q - first saved| per row: "
                  + " ".join(f"{nm}={v:.1e}" for nm, v in zip(names, d)))
            print("      max|Q| per row:            "
                  + " ".join(f"{nm}={v:.1e}" for nm, v in zip(names, np.abs(ref).max(axis=1))))
        else:
            np.save(f, np.asarray(Qe))
    jax.effects_barrier()
    sec, n = wall.perf_counter() - t0, _steps[0]
    per_sim = sec / float(t)
    print(f"{nx:4d} x {ny:3d} = {nx * ny:6d} cells: {n:5d} steps, {sec:7.2f} s "
          f"({1e3 * sec / max(n, 1):7.2f} ms/step, {1e6 * sec / max(n, 1) / (nx * ny):6.2f} us/cell/step) "
          f"-> {per_sim:7.1f} s per simulated second; compile {first - sec:5.1f} s; "
          f"finite {bool(np.isfinite(np.asarray(Qe)).all())}, pressure {s._pressure_backend()}",
          flush=True)
    if _amg_t["calls"]:
        print(f"      pyamg: {_amg_t['setups']} setups {_amg_t['setup']:.2f} s, "
              f"{_amg_t['calls']} V-cycles {_amg_t['solve']:.2f} s ({_amg_t['calls'] / max(n, 1):.1f}/step); "
              f"rest {sec - _amg_t['setup'] - _amg_t['solve']:.2f} s", flush=True)
