"""Attribute the reverse-mode NaN on the DRY cases: is it the IR lowering or a
kernel (sqrt(g h) at h = 0 under a masked min / where)?  Pure reference
blocks, ONE step, no loop, no IR."""
from common import *  # noqa
import numpy as np
import jax
import jax.numpy as jnp
from zoomy_core.mesh import LSQMesh
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.solver2 import blocks as B
from p0_probe_mood3 import radial1d_nsm

nsm = radial1d_nsm(2, "venkatakrishnan")
mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=48)
s = MarchSolver(time_end=1.0, mood_redo=True)
S = s.setup_march(mesh, nsm, CFL=CFL)
p0 = s._rt_parameters
ig = list(nsm.parameter_values.as_dict()).index("g")
M, O = s.MeshRT, s.Ops


def head_dt(p):
    lo, hi = B.dt_pass(S.Q, S.Qaux, p, M, O)
    return B.reduce_dt(lo, hi, M.inradius_f, 5.0, CFL=0.9, dimension=1.0)


def step_fixed_dt(p):
    S2, _ = s.hyperbolic_step(S, 0.001, p)
    return jnp.sum(S2.Q[1] ** 2)


def step_full(p):
    dt = head_dt(p)
    S2, _ = s.hyperbolic_step(S, dt, p)
    return jnp.sum(S2.Q[1] ** 2)


for nm, f in [("dt head only", head_dt), ("step at fixed dt", step_fixed_dt),
              ("head + step", step_full)]:
    g = jax.grad(f)(p0)
    _, d = jax.jvp(f, (p0,), (jnp.zeros_like(p0).at[ig].set(1.0),))
    print(f"{nm:20s} grad[g] {float(g[ig])!r:24} jvp {float(d)!r}")

lo, hi = B.dt_pass(S.Q, S.Qaux, p0, M, O)
_, vjp = jax.vjp(lambda p: B.dt_pass(S.Q, S.Qaux, p, M, O), p0)
ct = (jnp.zeros_like(lo).at[0].set(1.0), jnp.zeros_like(hi))
print("vjp of dt_pass with a one-hot cotangent on face 0:", vjp(ct)[0][ig])
ev = O.eigenvalues(S.Q[:, :3], S.Qaux[:, :3], p0, M.face_normals[:, :3])
print("eigenvalues at 3 dry cells:", np.asarray(ev).ravel()[:6])
