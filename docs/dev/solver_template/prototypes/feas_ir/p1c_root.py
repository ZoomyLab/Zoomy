"""ITEM 1, Root: point-implicit Manning friction as a Root statement inside the
REAL emitted 1-D SWE step (Stoker o1, 200 cells, t=2, n_imp=0.05).

  jax  : newton_root (custom_root, IFT) inside the traced, checkpointed march
  numpy: plain Newton with a cell-block FD Jacobian; numpy-built step AND
         numpy-built march (fully host, eager bodies)
"""
from common import *  # noqa
import time as _time
import numpy as np
import jax
import jax.numpy as jnp
from zoomy_jax.fvm.march_emitted import EmittedMarchSolver
from zoomy_core.transformation.procedure_python import ProcedureBuilder
from march_ext import build_callables, build_step_ext, build_march_ext, traced_bodies
from ir_ext import ProcedureBuilderExt

N_IMP = 0.05
em, S, nsm, mesh = make_solver(EmittedMarchSolver, "stoker", 1, n_cells=200,
                               t_end=2.0)
nc = mesh.n_inner_cells
h_i, gauge = 1, 120
p0 = em._rt_parameters
ig = list(nsm.parameter_values.as_dict()).index("g")
MeshRT = em.MeshRT

# ── numpy: numpy-built step (Root -> FD Newton) + numpy-built march ─────────
b = traced_bodies(em, gauge_cell=gauge, h_index=h_i, implicit_n=N_IMP)
sw = ProcedureBuilder("numpy", externals=b).build(em.program.should_write)
b["solver_should_write"] = lambda t, dt, i, wi: (
    lambda e: (e["do_write"], int(e["i_snapshot"])))(
        sw(time=t, dt=dt, i_snapshot=i, write_interval=wi))
step_proc = build_step_ext(em.program.constants, em.program.flags, 1,
                           mood=False, implicit_root=True)
step_np = ProcedureBuilderExt("numpy", externals=b).build(step_proc)
iters = []


def step_np_fn(Q, Qaux, p, time, iteration, dt):
    # the BODIES are jax functions (vmapped gathers need jax arrays), so the
    # host walker hands them jax arrays; the numpy Root returns numpy.
    e = step_np(Q=Q, Qaux=Qaux, p=p, time=time, iteration=iteration, dt=dt)
    iters.append(e["Q_iters"])
    return (jnp.asarray(e["Q"]), jnp.asarray(e["Qaux"]), e["time"],
            e["iteration"], e["troubled"])


b_np = dict(b)
b_np["solver_hyperbolic_step"] = step_np_fn
b_np["solver_write_fields"] = lambda *a: None
march_np = ProcedureBuilderExt("numpy", externals=b_np).build(
    build_march_ext(em.program.constants))
t0 = _time.perf_counter()
e_np = march_np(Q=S.Q, Qaux=S.Qaux, p=p0,
                time=0.0, iteration=0, i_snapshot=0, t_end=2.0,
                dt_window=None, write_interval=1e300,
                inradius_f=MeshRT.inradius_f, n_faces=MeshRT.n_faces, obs=0.0)
t_np = _time.perf_counter() - t0
Q_np = np.asarray(e_np["Q"])
print(f"numpy march (eager, FD-Newton Root): {int(e_np['iteration'])} steps, "
      f"Newton iters/step {iters}, wall {t_np:.2f}s")

# ── jax: whole march traced, Root -> newton_root ────────────────────────────
bj = traced_bodies(em, gauge_cell=gauge, h_index=h_i, implicit_n=N_IMP,
                   safe_dt=True)
mk = lambda kind: build_callables(  # noqa: E731
    em, em.program.constants, em.program.flags, 1, mood=False,
    backend="jax", bodies=bj, implicit_root=True, write=False,
    while_kind=kind)[0]
m_rev, m_fwd = mk("checkpointed"), mk("lax")


def run(Q, p, mf):
    e = mf(Q=Q, Qaux=S.Qaux, p=p, time=jnp.asarray(0.0),
           iteration=jnp.asarray(0), i_snapshot=jnp.asarray(0),
           t_end=jnp.asarray(2.0), dt_window=None, write_interval=1e300,
           inradius_f=MeshRT.inradius_f, n_faces=MeshRT.n_faces,
           obs=jnp.asarray(0.0))
    return e["Q"], e["iteration"], e["obs"]


rj = jax.jit(lambda Q, p: run(Q, p, m_rev))
Qj, itj, obsj = jax.block_until_ready(rj(S.Q, p0))
t0 = _time.perf_counter()
jax.block_until_ready(rj(S.Q, p0))
t_j = _time.perf_counter() - t0
Qj = np.asarray(Qj)
rel = np.abs(Qj - Q_np).max() / np.abs(Qj).max()
q_noimp = None
print(f"jax traced march (newton_root Root): {int(itj)} steps, warm {t_j*1e3:.1f}ms")
print(f"jax vs numpy lowering: max rel diff {rel:.2e}  (FD-Jacobian Newton vs "
      f"AD Newton; both to tol 1e-13)")
# friction actually acts: compare momentum against the n=0 march
bj0 = traced_bodies(em, gauge_cell=gauge, h_index=h_i, safe_dt=True)
m0 = build_callables(em, em.program.constants, em.program.flags, 1, mood=False,
                     backend="jax", bodies=bj0, write=False)[0]
Q0n = np.asarray(jax.jit(lambda Q, p: run(Q, p, m0)[0])(S.Q, p0))
print(f"friction effect: max|q| no-friction {np.abs(Q0n[2]).max():.4e}  "
      f"implicit n=0.05 {np.abs(Qj[2]).max():.4e}; mass drift "
      f"{(Qj[1, :nc].sum() - np.asarray(S.Q)[1, :nc].sum()) * 0.05:.2e}")

# ── AD through the Root ─────────────────────────────────────────────────────
eg = jnp.zeros_like(p0).at[ig].set(1.0)


def J(p, mf):
    Q, _, obs = run(S.Q, p, mf)
    return jnp.sum(Q[h_i, :nc] ** 2) + jnp.sum(Q[2, :nc] ** 2) + obs


_, d = jax.jit(lambda p: jax.jvp(lambda x: J(x, m_fwd), (p,), (eg,)))(p0)
g = jax.jit(jax.grad(lambda p: J(p, m_rev)))(p0)
Jf = jax.jit(lambda p: J(p, m_fwd))
print(f"AD: jvp {float(d):.12e} grad {float(g[ig]):.12e} rel "
      f"{abs(float(g[ig]) - float(d)) / abs(float(d)):.2e}")
for eps in (1e-4, 1e-5, 1e-6):
    fd = (Jf(p0 + eps * eg) - Jf(p0 - eps * eg)) / (2 * eps)
    print(f"    FD eps {eps:.0e}: {float(fd):.12e} rel "
          f"{abs(float(fd) - float(d)) / abs(float(d)):.2e}")
