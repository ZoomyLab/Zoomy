"""Gate 40: temporal order of the SSP-RK step, time_order 1, 2, 3 (JAX).

    python gate40_time_order.py [ode] [wave] [o2] [chorin] [chorin_orders]
                                                               (default: ode wave)
    mpiexec -n 2 python gate40_time_order.py par

* ode:   SWE at rest on 8 cells, forced by dh/dt = a cos(w t + 0.3) h.  The
         space operator vanishes; the exact solution h(T) = exp(a (sin(w T + 0.3)
         - sin 0.3) / w) is known, so the error is the time error alone.  The
         forcing depends on time AND state: the order is the full RK order, and
         the stage times c of SSP_RK enter.  A wrong stage time (negative
         control) drops the order of time_order 3.
* wave:  the same forcing on a smooth SWE wave (100 cells, walls, first-order
         space) at constant dt, dt/2, dt/4 against dt/16: the same mesh and the
         same space operator in every run, so the difference is the time error.
* o2:    as wave, second-order space (Venkatakrishnan limiter, no MOOD).
* chorin: VAM(1) Chorin (gate 7's channel): one step of time_order 3 = the
         Shu-Osher combination of three time_order 1 steps (the projection
         stage S), to round-off.
* chorin_orders: the same at constant dt, dt/2, dt/4 against dt/16, time_order
         1, 2, 3 (about 5 min).  The projection stage is itself first-order
         accurate, so the outer RK does not raise the order; only a floor is
         asserted.
* par:   time_order 3 on MPI ranks (2-D bump, second order + MOOD) = serial.

Observed order p = log2(e_k / e_(k+1)) of the finest pair; thresholds
0.9 / 1.8 / 2.7 for time_order 1 / 2 / 3.
"""
import sys
import time as clock

import common as C                      # noqa: F401  (x64)
import numpy as np
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.discretization import time as rk_time
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

FLOOR = {1: 0.9, 2: 1.8, 3: 2.7}
A_F, W_F, PHASE = 0.7, 3.0, 0.3          # forcing  dh/dt = A_F cos(W_F t + PHASE) h
failed = []


def check(label, ok):
    print(f"   {label}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        failed.append(label)


def orders(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def forced_swe(order, limiter, nx, domain, ic):
    """SME(0) in 1-D with walls and the forcing A_F cos(W_F t + PHASE) h on the
    mass equation."""
    sm = SystemModel.from_model(C.sme(0, 2, ("left", "right"), "wall"))
    names = [str(s) for s in sm.state]
    ih = names.index("h")
    src = sp.Matrix(sm.source)
    src[ih, 0] += A_F * sp.cos(W_F * sm.time + PHASE) * sm.state[ih]
    sm.source = src

    def q0(x):
        q = np.zeros(len(names))
        q[ih] = ic(x[0])
        return q
    sm.initial_conditions = IC.UserFunction(function=q0)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=limiter))
    return LSQMesh.create_1d(domain=domain, n_inner_cells=nx), nsm, ih


def run(mesh, nsm, time_end, dt, time_order, **kw):
    s = SymbolicHyperbolicSolverJax(time_end=time_end, dt=dt, mood=False,
                                    time_order=time_order, cfl_check_every=0,
                                    **kw)
    Q, A = s.setup(mesh, nsm)
    _, Q, _ = s.run(Q, A)
    return np.asarray(Q)


def ode():
    """Exact solution; the stage times of SSP_RK are checked by a control."""
    T = 1.0
    mesh, nsm, ih = forced_swe(1, "minmod", 8, (0.0, 1.0), lambda x: 1.0)
    exact = np.exp(A_F * (np.sin(W_F * T + PHASE) - np.sin(PHASE)) / W_F)
    steps = (16, 32, 64)

    def errors(to):
        return [abs(run(mesh, nsm, T, T / n, to)[ih, 0] - exact) for n in steps]
    for to in (1, 2, 3):
        t0 = clock.perf_counter()
        errs = errors(to)
        p = orders(errs)
        print(f"ODE dh/dt = a cos(w t + 0.3) h, time_order {to}: max err "
              f"{['%.1e' % e for e in errs]}  orders {['%.2f' % x for x in p]}  "
              f"[{clock.perf_counter() - t0:.0f}s]", flush=True)
        check(f"ODE time_order {to}: order {p[-1]:.2f} >= {FLOOR[to]}",
              p[-1] >= FLOOR[to])
    # control: RK3's last stage evaluated at the wrong time (c = 1, not 1/2)
    good = rk_time.SSP_RK[3]
    rk_time.SSP_RK[3] = (good[0], good[1], (good[2][0], 1))
    try:
        p = orders(errors(3))
    finally:
        rk_time.SSP_RK[3] = good
    print(f"ODE control, RK3 stage time c = 1 for stage 3: max order "
          f"{p[-1]:.2f}", flush=True)
    check("ODE control: a wrong stage time loses the order (< 2.5)", p[-1] < 2.5)


def wave(order, limiter):
    """Smooth forced SWE wave; constant dt, dt/2, dt/4 against dt/16."""
    mesh, nsm, _ = forced_swe(order, limiter, 100, (0.0, 1.0),
                              lambda x: 1.0 + 0.1 * np.exp(-((x - 0.5) / 0.1) ** 2))
    T, dt0 = 0.04, 2e-3
    for to in (1, 2, 3):
        t0 = clock.perf_counter()
        Q = {k: run(mesh, nsm, T, dt0 / 2 ** k, to) for k in (0, 1, 2, 4)}
        errs = [np.abs(Q[k] - Q[4]).max() for k in (0, 1, 2)]
        p = orders(errs)
        print(f"forced SWE wave, space order {order} ({limiter}), time_order "
              f"{to}: max |dt - dt/16| {['%.1e' % e for e in errs]}  orders "
              f"{['%.2f' % x for x in p]}  [{clock.perf_counter() - t0:.0f}s]",
              flush=True)
        check(f"wave O{order} time_order {to}: order {p[-1]:.2f} >= {FLOOR[to]}",
              p[-1] >= FLOOR[to])


def chorin_solver(to, dt, T):
    import jax.numpy as jnp
    import gate7_chorin as g7
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
    pred, press, corr = g7.split(_DT_SYMBOL)
    g7.cases.set_state_width(pred)
    s = SymbolicChorinSolverJax(time_end=T, dt=dt, cfl=0.9, time_order=to)
    Q, Qaux = s.setup(g7.mesh(), g7.pred_nsm(pred, press, corr), press, corr)
    nc = Q.shape[1]
    return s, jnp.asarray(g7.ic(nc, np.asarray(s.mesh_np.cell_centers)[:, :nc])), Qaux


def chorin():
    """The generic stage loop: one step of time_order 3 is the Shu-Osher
    combination of the time_order 1 step (the projection stage S):
    Q1 = S(Q0); Q2 = 3/4 Q0 + 1/4 S(Q1); Q = 1/3 Q0 + 2/3 S(Q2)."""
    import jax.numpy as jnp
    import gate7_chorin as g7
    s3, Q0, Aux = chorin_solver(3, g7.DT, g7.DT)
    s1, _, _ = chorin_solver(1, g7.DT, g7.DT)
    Q0 = np.asarray(Q0)

    def stage(X):
        X = jnp.asarray(X)
        return np.asarray(s1.run(X, s1.initial_aux(X, Aux))[1])
    Q1 = stage(Q0)
    Q2 = 0.75 * Q0 + 0.25 * stage(Q1)
    shu_osher = Q0 / 3 + 2 / 3 * stage(Q2)
    X = jnp.asarray(Q0)
    Q3 = np.asarray(s3.run(X, s3.initial_aux(X, Aux))[1])
    err = np.abs(Q3 - shu_osher).max() / np.abs(shu_osher).max()
    step = np.abs(shu_osher - Q0).max() / np.abs(Q0).max()
    print(f"VAM(1) Chorin, one step of time_order 3 vs the Shu-Osher combination "
          f"of time_order 1 stages: max rel diff {err:.1e}  (step size {step:.1e})",
          flush=True)
    check("Chorin time_order 3 = Shu-Osher over the projection stage (<= 1e-9)",
          err <= 1e-9 and step > 1e3 * err)


def chorin_orders():
    """Temporal order of the Chorin step (about 5 min).  The projection stage
    is first-order accurate (Lie-split sources, pressure at the end of the
    stage), so the outer RK does not raise the order: only a floor is asserted."""
    import gate7_chorin as g7

    def solve(to, dt, T):
        s, Q, Aux = chorin_solver(to, dt, T)
        return np.asarray(s.run(Q, s.initial_aux(Q, Aux))[1])
    T, dt0 = 0.125, g7.DT
    for to in (1, 2, 3):
        t0 = clock.perf_counter()
        ref = solve(to, dt0 / 16, T)
        errs = [np.abs(solve(to, dt0 / 2 ** k, T) - ref).max() for k in (0, 1, 2)]
        p = orders(errs)
        print(f"VAM(1) Chorin, time_order {to}: max |dt - dt/16| "
              f"{['%.1e' % e for e in errs]}  orders {['%.2f' % x for x in p]}  "
              f"[{clock.perf_counter() - t0:.0f}s]", flush=True)
        check(f"Chorin time_order {to}: order {p[-1]:.2f} >= 0.9", p[-1] >= 0.9)


def par():
    """time_order 3 on the ranks = serial (the 2-D bump, O2 + MOOD)."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mesh, nsm = C.bump_2d(2, nx=24, flow=(0.1, 0.05))
    T = 0.2
    s = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.45, time_order=3, comm=comm)
    Q, A = s.setup(mesh, nsm)
    t, Q, A = s.run(Q, A)
    Q.block_until_ready()
    Qg = s.partition.gather(Q) if s.partition else np.asarray(Q)
    if comm.Get_rank() != 0:
        return
    mesh, nsm = C.bump_2d(2, nx=24, flow=(0.1, 0.05))
    r = SymbolicHyperbolicSolverJax(time_end=T, cfl=0.45, time_order=3)
    Qs, As = r.setup(mesh, nsm)
    ts, Qs, _ = r.run(Qs, As)
    Qs = np.asarray(Qs)
    err = np.abs(Qg - Qs).max() / np.abs(Qs).max()
    print(f"SWE 2-D bump O2 + MOOD, time_order 3: ranks {comm.Get_size()}  "
          f"t={float(t):.4g} (serial {float(ts):.4g})  max rel |par-ser| "
          f"{err:.1e}", flush=True)
    check("time_order 3: ranks = serial (<= 1e-10)", err <= 1e-10)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["ode", "wave"]
    for part in parts:
        {"ode": ode, "wave": lambda: wave(1, "minmod"),
         "o2": lambda: wave(2, "venkatakrishnan"), "chorin": chorin,
         "chorin_orders": chorin_orders, "par": par}[part]()
    print("G40_FAIL " + "; ".join(failed) if failed else "G40_DONE", flush=True)
    sys.exit(1 if failed else 0)
