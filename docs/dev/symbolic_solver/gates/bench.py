"""Speed: symbolic vs existing JAX solvers on the gate cases, each through
its production path (jitted time loop).  Each solver runs twice; the second
run is timed (compile excluded) and its first run reported as compile+run.

    python bench.py [case ...]      cases: sme1d swe2d dry imex chorin malpasset
"""
import os
import sys
import time as wall

import common as C
import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_CHORIN, T_MALP = 1280, 0.5          # every case < 5 min (compile + 2 x 4 runs)
sys.path.insert(0, os.path.join(HERE, "..", "cases"))


def timed(f):
    """First call (compile + run) and a closure that times one more run."""
    t0 = wall.perf_counter()
    out = f()
    jax.block_until_ready(out)
    first = wall.perf_counter() - t0

    def again():
        t = wall.perf_counter()
        jax.block_until_ready(f())
        return wall.perf_counter() - t
    return out, first, again


def best(old_again, new_again, n=3):
    """Alternate old/new runs; the minimum of each (load from other jobs
    only ever adds time)."""
    to, tn = [], []
    for _ in range(n):
        to.append(old_again())
        tn.append(new_again())
    return min(to), min(tn)


def hyperbolic(make, t_end, cfl, old_kw=None, new_kw=None, imex=False):
    from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    mesh, nsm = make()
    Old = IMEXSourceSolverJax if imex else HyperbolicSolver
    o = Old(time_end=t_end, compute_dt=C.adaptive(cfl, mesh), march_log_interval=0,
            **(old_kw or {}))
    Qo, Ao = o.setup_simulation(mesh, nsm)
    old, oc, ot = timed(lambda: o.run_simulation(Qo, Ao, write_output=False))
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, imex=imex,
                                    **{"mood": False, **(new_kw or {})})   # old: no MOOD off eta
    Q, A = s.setup(mesh, nsm)
    new, nc, nt = timed(lambda: s.run(Q, A))
    Qold = np.asarray(old[0] if isinstance(old, tuple) else old)[:, :Q.shape[1]]
    ot, nt = best(ot, nt)
    return np.abs(np.asarray(new[1]) - Qold).max(), oc, ot, nc, nt


def chorin():
    import gate7_chorin as g7
    from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
    import escalante_bump as EB
    import sympy as sp
    from zoomy_core.mesh import LSQMesh
    pred, press, corr = EB.split(sp.Symbol("dt", positive=True))
    o = ChorinSplitVAMSolverJax(pred, press, corr, pressure_tol=1e-9, pressure_maxit=200)
    Q0 = np.array(o.setup_simulation(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS)))
    xc = np.asarray(o._rt_mesh.cell_centers)[0, :o.nc]
    Q0[:, :o.nc] = EB.initial_state(xc, Q0.shape[0])
    n = N_CHORIN
    old, oc, ot = timed(lambda: o.run_jit_steps(jnp.asarray(EB.DT), n, jnp.asarray(Q0)))
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=n * EB.DT, dt=EB.DT, cfl=0.9,
                                pressure_tol=1e-9, pressure_maxit=200)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
    A = s.initial_aux(Q, A)
    new, nc, nt = timed(lambda: s.run(Q, A))
    ot, nt = best(ot, nt)
    return np.abs(np.asarray(new[1]) - np.asarray(old[0])[:, :o.nc]).max(), oc, ot, nc, nt


def malpasset(imex):
    sys.modules.pop("run", None)          # the bump case's run.py shadows it
    import malpasset as MP
    from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    import zoomy_core.fvm.timestepping as ts
    mesh, b = MP.build()
    Old = IMEXSourceSolverJax if imex else HyperbolicSolver
    o = Old(time_end=T_MALP, compute_dt=ts.adaptive(CFL=0.4), march_log_interval=0,
            **b["fs_kwargs"])
    if imex:
        object.__setattr__(o, "source_mode", "local")
    Qo, Ao = o.setup_simulation(mesh, b["nsm"])
    old, oc, ot = timed(lambda: o.run_simulation(Qo, Ao, write_output=False))
    mesh, b = MP.build()
    s = SymbolicHyperbolicSolverJax(time_end=T_MALP, cfl=0.4, 
                                    imex=imex)
    Q, A = s.setup(mesh, b["nsm"])
    new, nc, nt = timed(lambda: s.run(Q, A))
    Qold = np.asarray(old[0] if isinstance(old, tuple) else old)[:, :Q.shape[1]]
    ot, nt = best(ot, nt)
    return np.abs(np.asarray(new[1]) - Qold).max(), oc, ot, nc, nt


CASES = {
    "sme1d": ("SME(1) 1-D dam break O2, 200 cells, t=0.1",
              lambda: hyperbolic(lambda: C.dambreak_1d(1, 2), 0.1, 0.9)),
    "swe2d": ("SWE 2-D bump O2, 40x40, t=0.5",
              lambda: hyperbolic(lambda: C.bump_2d(2), 0.5, 0.9)),
    "swe2d_big": ("SWE 2-D bump O2, 160x160, t=0.5",
                  lambda: hyperbolic(lambda: C.bump_2d(2, nx=160), 0.5, 0.9)),
    "swe2d_big_o1": ("SWE 2-D bump O1, 160x160, t=0.5",
                     lambda: hyperbolic(lambda: C.bump_2d(1, nx=160), 0.5, 0.9)),
    "sme1d_big": ("SME(1) 1-D dam break O2, 4000 cells, t=0.1",
                  lambda: hyperbolic(lambda: C.dambreak_1d(1, 2, nx=4000), 0.1, 0.9)),
    "dry": ("radial dry dam break O2 eta+MOOD, 48x48, t=0.02",
            lambda: hyperbolic(C.dry_dambreak_2d, 0.02, 0.45,
                               __import__("gate1_vs_solver_jax").eta_kw(C.dry_dambreak_2d),
                               dict(mood=True))),
    "imex": ("IMEX SME(1) 1-D dam break O2, 100 cells, t=0.1",
             lambda: hyperbolic(__import__("gate6_imex").CASES[0][1], 0.1, 0.9, imex=True)),
    "chorin": ("Chorin VAM(1) bump, 60 cells, 1280 steps", chorin),
    "malpasset": ("Malpasset SME(1) O1, 26k cells, t=0.5, explicit", lambda: malpasset(False)),
    "malpasset_imex": ("Malpasset SME(1) O1, 26k cells, t=0.5, IMEX", lambda: malpasset(True)),
}

if __name__ == "__main__":
    print(f"device {jax.devices()[0].platform}")
    for key in sys.argv[1:] or [k for k in CASES if not k.endswith("_big")]:
        name, f = CASES[key]
        d, oc, ot, nc, nt = f()
        load = os.getloadavg()[0]
        print(f"{name:52s} old {ot:7.2f}s (first {oc:6.1f}s) | symbolic {nt:7.2f}s "
              f"(first {nc:6.1f}s) | speedup {ot / nt:5.2f}x | max|new-old| {d:.1e} | load {load:.1f}",
              flush=True)
    print("BENCH_DONE")
