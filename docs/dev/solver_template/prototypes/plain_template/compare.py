"""Plain-Python template vs the IR template: same numbers?  AD works?"""
import os
import time as _time
from functools import partial
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from zoomy_core.solver.template import Solver
from zoomy_core.solver.template.solver import Setup
from zoomy_core.solver.template.plugin import Output
from zoomy_core.solver.template.backend_numpy import NumpyBackend
from zoomy_core.solver.template.toy import advection_bodies, advection_state
from zoomy_jax.template.backend import JaxBackend

import plain

T_END, CFL, N, SNAP = 0.25, 0.9, 40, 5


# ── IR template (as in the 1a-1 gate) ───────────────────────────────────────
class Adv(Solver):
    name = "adv"

    def __init__(self, record, **kw):
        super().__init__(**kw)
        self.record = record

    def setup_bodies(self, mesh, nsm, backend):
        xp = backend.xp
        Q, Qaux = advection_state(xp, N)
        return Setup(bodies=advection_bodies(xp, order=2, record=self.record),
                     flags={"cell_ncp": False}, geom=None, Q=Q, Qaux=Qaux,
                     parameters=xp.asarray([1.0, 0.0]), dimension=1, degree=1, order=2)


def run_ir(B):
    rec = []
    s = Adv(rec, backend=B, time_end=T_END, cfl=CFL, plugins=[Output(snapshots=SNAP)])
    Q0, A0 = s.setup_simulation(None, None)
    Q, A = s.run_simulation(Q0, A0)
    o = s.last_outputs
    return np.asarray(Q), int(o["iteration"]), [(int(i), float(t)) for i, t, _ in rec]


# ── plain template ───────────────────────────────────────────────────────────
def kernels(xp, record=None):
    return SimpleNamespace(**{**advection_bodies(xp, order=2, record=record), "cell_ncp": None})


def plain_march(B, p, time_end, record=None, snapshots=SNAP):
    xp = B.xp
    k = kernels(xp)
    Q, Qaux = advection_state(xp, N)
    step = partial(plain.mood_step, B, k, plain.SSPRK2, plain.explicit_stage)
    write = (lambda Q, A, t, i: record.append((int(i), float(t)))) if record is not None else None
    return plain.march(B, k, step, Q, Qaux, p, None, 0.0, time_end, cfl=CFL,
                       dimension=1, degree=1, snapshots=snapshots, write=write)


rows = []
for name, B in [("numpy", NumpyBackend()), ("jax-fwd", JaxBackend("forward")),
                ("jax-rev", JaxBackend("reverse"))]:
    Qi, iti, wi = run_ir(B)
    rec = []
    f = B.jit(lambda p, te: plain_march(B, p, te, record=rec))
    Qp, Ap, tp, itp, isp = f(B.xp.asarray([1.0, 0.0]), T_END)
    Qp = np.asarray(Qp)
    rows.append((name, np.array_equal(Qi, Qp), np.max(np.abs(Qi - Qp)), iti, int(itp),
                 wi == rec, float(tp)))
print(f"{'backend':8s} {'bitwise':8s} {'max|dQ|':9s} {'it IR':6s} {'it plain':8s} {'writes eq':9s} t_end")
for r in rows:
    print(f"{r[0]:8s} {str(r[1]):8s} {r[2]:9.1e} {r[3]:<6d} {r[4]:<8d} {str(r[5]):9s} {r[6]}")

# single compile across time_end values (traced)
B = JaxBackend("forward")
f = jax.jit(lambda p, te: plain_march(B, p, te, snapshots=0)[0])
for te in (0.1, 0.2, 0.25, 0.3):
    f(jnp.asarray([1.0, 0.0]), te)
print("jax compiles across 4 time_end values:", f._cache_size())

# AD: d/da sum(Q^2) at t_end, a = advection speed (dt depends on a)
def J(B, a, snapshots):
    p = jnp.stack([a, 0.0])
    Q = plain_march(B, p, T_END, snapshots=snapshots)[0]
    return jnp.sum(Q ** 2)

a0 = 1.0
Jf = jax.jit(lambda a: J(JaxBackend("forward"), a, 0))
_, jvp = jax.jvp(Jf, (a0,), (1.0,))
h = 1e-6
fd = (Jf(a0 + h) - Jf(a0 - h)) / (2 * h)
Jr = jax.jit(lambda a: J(JaxBackend("reverse", checkpoints=8), a, 3))
g = jax.grad(Jr)(a0)
print(f"AD: |jvp-FD|/|jvp| = {abs(jvp - fd) / abs(jvp):.1e}   |grad-jvp|/|jvp| = {abs(g - jvp) / abs(jvp):.1e}")

# timing per step on jax-fwd (after compile)
fw = jax.jit(lambda p, te: plain_march(JaxBackend("forward"), p, te, snapshots=0))
out = fw(jnp.asarray([1.0, 0.0]), T_END); jax.block_until_ready(out)
t0 = _time.perf_counter(); out = fw(jnp.asarray([1.0, 0.0]), T_END); jax.block_until_ready(out)
print(f"plain jax-fwd run: {(_time.perf_counter() - t0) * 1e3:.2f} ms for {int(out[3])} steps")
