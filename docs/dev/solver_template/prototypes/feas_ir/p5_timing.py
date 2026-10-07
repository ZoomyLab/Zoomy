"""Clean wall-time comparison (sequential, min of 10): whole-march traced
(lax While / checkpointed While) vs reference MarchSolver.march (host loop,
two jitted dispatches per step)."""
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


def tmin(f, n=10):
    best = np.inf
    for _ in range(n):
        t0 = _time.perf_counter()
        jax.block_until_ready(f())
        best = min(best, _time.perf_counter() - t0)
    return best


def case(label, mk, t_end, gauge):
    ref, S, nsm, mesh = mk(MarchSolver)
    em, S2, _, _ = mk(EmittedMarchSolver)
    st = ref.march(S)[1]
    nsteps = st["steps"]
    t_ref = tmin(lambda: ref.march(S)[0].Q)
    b = traced_bodies(em, gauge_cell=gauge, h_index=1, safe_dt=True)
    out = {}
    for kind in ("lax", "checkpointed"):
        m = build_callables(em, em.program.constants, em.program.flags,
                            int(em.Ops.order), mood=bool(em.mood_redo),
                            backend="jax", bodies=b, while_kind=kind,
                            write=False)[0]
        f = jax.jit(lambda Q, Qaux, p, m=m: m(
            Q=Q, Qaux=Qaux, p=p, time=jnp.asarray(0.0),
            iteration=jnp.asarray(0), i_snapshot=jnp.asarray(0),
            t_end=jnp.asarray(t_end), dt_window=None, write_interval=1e300,
            inradius_f=em.MeshRT.inradius_f, n_faces=em.MeshRT.n_faces,
            obs=jnp.asarray(0.0))["Q"])
        f(S2.Q, S2.Qaux, em._rt_parameters)
        out[kind] = tmin(lambda: f(S2.Q, S2.Qaux, em._rt_parameters))
    print(f"{label}: {nsteps} steps | per step: ref host loop "
          f"{t_ref / nsteps * 1e3:.2f} ms, traced lax {out['lax'] / nsteps * 1e3:.2f}"
          f" ms, traced checkpointed {out['checkpointed'] / nsteps * 1e3:.2f} ms")


case("Stoker o1 200c t=2",
     lambda cls: make_solver(cls, "stoker", 1, n_cells=200, t_end=2.0), 2.0, 120)
case("Stoker o2 200c t=2",
     lambda cls: make_solver(cls, "stoker", 2, n_cells=200, t_end=2.0), 2.0, 120)


def mk_rad(cls, te=0.00989):
    nsm = radial1d_nsm(2, "venkatakrishnan")
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=48)
    s = cls(time_end=te, mood_redo=True)
    return s, s.setup_march(mesh, nsm, CFL=CFL), nsm, mesh


case("radial VK o2 MOOD 48c", mk_rad, 0.00989, 28)
