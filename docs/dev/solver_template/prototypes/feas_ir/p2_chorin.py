"""ITEM 2: Chorin as a COMPOSITE STAGE inside a traced emitted-style march.

Per RK stage: [hyperbolic Euler -> elliptic (pressure) -> pointwise
(corrector)], each a Call onto the PRODUCTION ChorinSplitVAMSolverJax
internals (no copy of their scheme logic).  The march is the extended IR
(ir_ext.ProcedureBuilderExt, jax) with the While -> lax (forward) /
eqxi checkpointed (reverse).

Reference: ChorinSplitVAMSolverJax.run_simulation with
timestepping.constant(DT), the Escalante VAM(1) bump of
zoomy_jax/tests/runtime/test_ad_chorin.py (copied, not edited).
"""
import sys
import time as _time

from common import *  # noqa
import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.solver.ir import Assign, Call, Procedure, While

import models
from cases import (ESC_DOMAIN, VAM_PRESSURE_TOL, bump_ic, chorin_split_for,
                   set_state_width)
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
from ir_ext import Guard, ProcedureBuilderExt

N_CELLS, DT, N_STEPS, CKPT = 30, 2.0 ** -5, 4, 3
T_END = N_STEPS * DT


def chorin_step_proc(time_order):
    tableau = ((0.0, 1.0),) if time_order == 1 else ((0.0, 1.0), (0.5, 1.0))
    time, dt, it = sp.symbols("time dt iteration")
    Q0, Q = sp.symbols("Q0 Q")
    stmts = [Call("solver_stage_base", args=("variables",), results=("q0",)),
             Assign("aux_cur", sp.Symbol("Qaux"))]
    for k, (alpha, _beta) in enumerate(tableau):
        stmts += [
            Assign("time_stage", time if k == 0 else time + dt),
            # the COMPOSITE stage: three kinds, one per REQ-173 stage kind
            Call("solver_refresh_pred_aux",
                 args=("variables", "aux_cur", "parameters"),
                 results=("aux_stage",)),
            Call("solver_hyperbolic_euler",
                 args=("dt", "time_stage", "variables", "aux_stage",
                       "parameters"), results=("variables",)),
            Call("solver_elliptic",
                 args=("variables", "dt", "time_stage", "parameters"),
                 results=("variables",)),
            Call("solver_pointwise", args=("variables", "dt", "parameters"),
                 results=("variables",)),
            # the stage's own pool is threaded into the next stage, as
            # chorin_cycle does
            Call("solver_refresh_pred_aux",
                 args=("variables", "aux_stage", "parameters"),
                 results=("aux_cur",)),
        ]
        if alpha != 0.0:
            stmts.append(Assign("Q", sp.Float(alpha) * Q0
                                + sp.Float(1.0 - alpha) * Q, declare=False))
    stmts += [Assign("time", time + dt, declare=False),
              Assign("iteration", it + 1, declare=False, ctype="int")]
    return Procedure(name="solver_chorin_step",
                     args=("variables", "aux_variables", "parameters", "time",
                           "iteration", "dt"), stmts=tuple(stmts))


def chorin_march_proc():
    time, t_end, dt = sp.symbols("time t_end dt")
    body = (
        Call("solver_constant_dt", args=("c_dt", "t_end", "time"),
             results=("dt",)),
        Guard(sp.And(sp.Gt(dt, 0), sp.Lt(dt, sp.oo)),
              "FATAL dt guard: dt = {dt} at t = {time}", on="dt"),
        Call("solver_chorin_step",
             args=("variables", "aux_variables", "parameters", "time",
                   "iteration", "dt"),
             results=("variables", "time", "iteration")),
        # base post_step: predictor pool of the stepped Q, refreshed from the
        # PRE-step pool (exactly HyperbolicSolverJax._advance -> post_step)
        Call("solver_refresh_pred_aux",
             args=("variables", "aux_variables", "parameters"),
             results=("aux_variables",)),
    )
    return Procedure(
        name="solver_march",
        args=("variables", "aux_variables", "parameters", "time", "iteration",
              "t_end"),
        stmts=(While(sp.Lt(time, t_end),
                     carry=("variables", "aux_variables", "time",
                            "iteration"), body=body),))


def bodies_for(s):
    def refresh(Q, Qaux, p):
        return s._refresh_aux_for_sm(Qaux, s.sm_pred, Q, p)

    def elliptic(Q, dt, t, p):
        p_press = s._sub_parameters(s._press_param_src, p, dt)
        Qaux_press = s._refresh_aux_for_sm(s._qaux_press0, s.sm_press, Q,
                                           p_press)
        return s._step_pressure_pure(Q, Qaux_press, dt, t, p_press)

    def pointwise(Q, dt, p):
        p_corr = s._sub_parameters(s._corr_param_src, p, dt)
        Qaux_corr = s._refresh_aux_for_sm(s._qaux_corr0, s.sm_corr, Q, p_corr)
        return s._step_corrector_pure(Q, Qaux_corr, p_corr)

    return {
        "solver_stage_base": lambda Q: Q,
        "solver_refresh_pred_aux": refresh,
        "solver_hyperbolic_euler": s._hyperbolic_stage,
        "solver_elliptic": elliptic,
        "solver_pointwise": pointwise,
        "solver_constant_dt": lambda c, te, t: jnp.minimum(c, te - t),
    }


def main(time_order):
    print(f"\n=== Chorin composite stage, VAM(1) bump, {N_CELLS} cells, "
          f"time_order={time_order}, {N_STEPS} x DT=2^-5 ===", flush=True)
    model = models.vam(level=1, dimension=2, bc="bump")
    triple = chorin_split_for(model, SystemModel.from_model(model))
    set_state_width(triple[0])
    mesh = LSQMesh.create_1d(domain=ESC_DOMAIN, n_inner_cells=N_CELLS)
    s = ChorinSplitVAMSolverJax(
        *triple, time_end=T_END, compute_dt=timestepping.constant(DT),
        march_log_interval=0, ad_checkpoints=CKPT,
        pressure_tol=VAM_PRESSURE_TOL, time_order=time_order)
    Q0 = np.array(s.setup_simulation(mesh))
    xc = np.asarray(s._rt_mesh.cell_centers)[:, :s.nc]
    for j in range(s.nc):
        Q0[:, j] = bump_ic(xc[:, j])
    Q0 = jnp.asarray(Q0)
    p0 = s._rt_parameters
    ig = list(s.nsm.parameter_values.as_dict()).index("g")
    ih = models.state_index(s.nsm, "h")

    # ── production reference ───────────────────────────────────────────────
    s.ad_mode = "forward"
    t0 = _time.perf_counter()
    Qr, Qauxr = jax.block_until_ready(s.run_simulation(Q0, write_output=False))
    t_ref_cold = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    Qr, Qauxr = jax.block_until_ready(s.run_simulation(Q0, write_output=False))
    t_ref = _time.perf_counter() - t0

    # fair per-step number: hand run_simulation a prebuilt pool (its Qaux=None
    # path rebuilds the pool EAGERLY outside the jit on every call)
    Qaux0 = jax.block_until_ready(jax.jit(s._qaux_from_q)(Q0, p0))
    s.run_simulation(Q0, Qaux=Qaux0, write_output=False)
    t0 = _time.perf_counter()
    jax.block_until_ready(s.run_simulation(Q0, Qaux=Qaux0, write_output=False))
    t_ref_pool = _time.perf_counter() - t0

    # ── the composition ────────────────────────────────────────────────────
    b = bodies_for(s)
    step_fn = ProcedureBuilderExt("jax", externals=b).build(
        chorin_step_proc(time_order))

    def step(Q, Qaux, p, time, iteration, dt):
        e = step_fn(Q=Q, Qaux=Qaux, p=p, time=time, iteration=iteration,
                    dt=dt)
        return e["Q"], e["time"], e["iteration"]
    b["solver_chorin_step"] = step

    def build(kind):
        m = ProcedureBuilderExt("jax", externals=b, while_kind=kind,
                                checkpoints=CKPT).build(chorin_march_proc())

        def run(Q, p, t_end=jnp.asarray(T_END)):
            Qaux = s._qaux_from_q(Q, p)
            e = m(Q=Q, Qaux=Qaux, p=p, time=jnp.asarray(0.0),
                  iteration=jnp.asarray(0), t_end=t_end, c_dt=DT)
            return e["Q"], e["Qaux"], e["iteration"]
        return run
    run_fwd, run_rev = build("lax"), build("checkpointed")
    rj = jax.jit(run_rev)
    t0 = _time.perf_counter()
    Qc, Qauxc, nst = jax.block_until_ready(rj(Q0, p0))
    t_c_cold = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    Qc, Qauxc, nst = jax.block_until_ready(rj(Q0, p0))
    t_c = _time.perf_counter() - t0
    Qf = jax.jit(run_fwd)(Q0, p0)[0]
    dQ = float(jnp.abs(Qc - Qr).max())
    dA = float(jnp.abs(Qauxc - Qauxr).max())
    print(f"steps {int(nst)}; |Q - run_simulation|max {dQ:.3e} "
          f"bitwise {bool((Qc == Qr).all())}; |Qaux - ref|max {dA:.3e}; "
          f"fwd(lax) == rev(ckpt) bitwise {bool((Qf == Qc).all())}")
    print(f"wall/step: composition warm {t_c / N_STEPS * 1e3:.2f} ms "
          f"(cold total {t_c_cold:.1f}s); run_simulation warm "
          f"{t_ref / N_STEPS * 1e3:.2f} ms (cold total {t_ref_cold:.1f}s); "
          f"run_simulation(Qaux given) warm {t_ref_pool / N_STEPS * 1e3:.2f} ms")

    # ── AD ─────────────────────────────────────────────────────────────────
    eg = jnp.zeros_like(p0).at[ig].set(1.0)

    def J(p, q, run):
        Q, _, _ = run(q, p)
        return jnp.sum(Q[ih, :s.nc] ** 2)

    _, d = jax.jit(lambda p: jax.jvp(lambda x: J(x, Q0, run_fwd), (p,),
                                     (eg,)))(p0)
    t0 = _time.perf_counter()
    gp, gq = jax.jit(jax.grad(lambda p, q: J(p, q, run_rev),
                              argnums=(0, 1)))(p0, Q0)
    t_grad = _time.perf_counter() - t0
    Jf = jax.jit(lambda p: J(p, Q0, run_fwd))
    print(f"AD g: jvp {float(d):.12e} grad {float(gp[ig]):.12e} rel "
          f"{abs(float(gp[ig]) - float(d)) / abs(float(d)):.2e} "
          f"(grad wall incl. compile {t_grad:.1f}s)")
    for eps in (1e-4, 1e-5, 1e-6):
        fd = (Jf(p0 + eps * eg) - Jf(p0 - eps * eg)) / (2 * eps)
        print(f"    FD eps {eps:.0e}: {float(fd):.12e} rel "
              f"{abs(float(fd) - float(d)) / abs(float(d)):.2e}")
    v = jax.random.normal(jax.random.PRNGKey(1), Q0.shape, dtype=Q0.dtype)
    _, dv = jax.jit(lambda q: jax.jvp(lambda x: J(p0, x, run_fwd), (q,),
                                      (v,)))(Q0)
    print(f"AD Q0-dir: jvp {float(dv):.12e} <grad,v> "
          f"{float(jnp.vdot(gq, v)):.12e} rel "
          f"{abs(float(jnp.vdot(gq, v)) - float(dv)) / abs(float(dv)):.2e}")

    # production reverse for comparison
    s.ad_mode = "reverse"
    gpr = jax.jit(jax.grad(lambda p: jnp.sum(
        s.run_simulation(Q0, write_output=False, parameters=p)[0][ih, :s.nc]
        ** 2)))(p0)
    print(f"production run_simulation reverse grad[g] {float(gpr[ig]):.12e} "
          f"(composition {float(gp[ig]):.12e}, rel "
          f"{abs(float(gpr[ig]) - float(gp[ig])) / abs(float(gp[ig])):.2e})")


if __name__ == "__main__":
    for to in [int(a) for a in sys.argv[1:]] or [1, 2]:
        main(to)
