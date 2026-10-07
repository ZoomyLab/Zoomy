"""Option-B feasibility: ONE template (jax's _compile_time_loop, de-jaxed through a
6-member backend object) driving BOTH the numpy and the jax production steps.
Pass = bitwise equal to each backend's own run_simulation."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.expanduser("~/git/Zoomy/library/zoomy_jax/tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feas_ir"))
import numpy as np
from common import nsm_for, DOMAIN
from zoomy_core.mesh import LSQMesh

class NumpyBackend:                      # never imports jax
    jit = staticmethod(lambda f: f)
    def while_loop(self, cond, body, init):
        v = init
        while cond(v): v = body(v)
        return v
    cond = staticmethod(lambda p, a, b: a() if p else b())
    where = staticmethod(np.where)

class JaxBackend:
    def __init__(self):
        import jax, jax.numpy as jnp
        self.jit, self.while_loop, self.where = jax.jit, jax.lax.while_loop, jnp.where
        self.cond = lambda p, a, b: jax.lax.cond(p, a, b)

def make_time_loop(B, advance, proceed, time_end, observer=None):
    """The template: structure + hook calls only (output omitted: no-op save)."""
    @B.jit
    def time_loop(time, Q, Qaux, parameters):
        acc = () if observer is None else observer.init(Q, Qaux, parameters)
        def body(v):
            t, it, Q, Qaux, acc = v
            t_new, dt, Q, Qaux = advance(t, Q, Qaux, time_end=time_end, parameters=parameters)
            if observer is not None:
                acc = observer.update(acc, t_new, dt, Q, Qaux, parameters)
            return (t_new, it + 1, Q, Qaux, acc)
        v = B.while_loop(lambda v: proceed(v[0], time_end), body, (time, 0, Q, Qaux, acc))
        return v[2], v[3], v[1]
    return time_loop

T_END, N = 6.0, 200
import zoomy_core; print(nsm_for("stoker",2).dt_max if hasattr(nsm_for("stoker",2),"dt_max") else None)
mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N)

# ---- numpy: production step(dt) wrapped as a functional _advance (the compat facade) ----
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NpSolver
def np_solver():
    s = NpSolver(time_end=T_END); s.settings.output.snapshots = 2
    s.setup_simulation(mesh, nsm_for("stoker", 2), write_output=False); return s
ref = np_solver(); Qref, _ = ref.run_simulation()
s = np_solver()
def np_advance(t, Q, Qaux, *, time_end, parameters):
    s._sim_Q, s._sim_Qaux, s._sim_time = Q, Qaux, t
    dt = s.compute_dt(Q, Qaux, s._sim_parameters, s._sim_face_inradius, s._sim_compute_max_abs_eigenvalue)
    dt = min(float(dt), float(time_end - t))
    s.step(dt)
    return t + dt, dt, s._sim_Q, s._sim_Qaux
loop = make_time_loop(NumpyBackend(), np_advance, lambda t, te: t < te, T_END)
Qn, _, it_n = loop(0.0, s._sim_Q, s._sim_Qaux, s._sim_parameters)
print(f"numpy : template vs run_simulation max|dQ| = {np.max(np.abs(Qn - Qref)):.3e}  steps={it_n}")

# ---- jax: production _advance unchanged ----
import jax.numpy as jnp
from zoomy_jax.fvm.solver_jax import HyperbolicSolver as JxSolver
j = JxSolver(time_end=T_END, march_log_interval=0)
Q0, A0 = j.setup_simulation(mesh, nsm_for("stoker", 2))
Qjref, _ = j.run_simulation(Q0, A0, write_output=False)
loop = make_time_loop(JaxBackend(), j._advance, j._couple_proceed, T_END)
Qj, _, it_j = loop(0.0, Q0, A0, j._rt_parameters)
print(f"jax   : template vs run_simulation max|dQ| = {float(jnp.max(jnp.abs(Qj - Qjref))):.3e}  steps={int(it_j)}")
print(f"numpy vs jax (same template, different bodies) max|dQ| = {np.max(np.abs(np.asarray(Qj) - Qn)):.3e}")
