"""Inject the same fault (a shape bug in the flux body) into (i) the production
class march and (ii) the v6 IR-emitted march; compare what a student sees."""
import os, sys, traceback, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.expanduser("~/git/Zoomy/library/zoomy_jax/tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feas_ir"))
import numpy as np, jax, jax.numpy as jnp
from common import nsm_for, make_solver, DOMAIN
from zoomy_core.mesh import LSQMesh

def report(tag, exc):
    tb = traceback.extract_tb(exc.__traceback__)
    walker = [f for f in tb if "procedure_python" in f.filename]
    jaxint = [f for f in tb if "/jax/" in f.filename or "/jax/_src" in f.filename]
    print(f"[{tag}] {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
    print(f"   frames={len(tb)} walker_frames={len(walker)} jax_internal={len(jaxint)}")
    user = [f for f in tb if "/jax/" not in f.filename and "site-packages" not in f.filename]
    for f in user[-8:]:
        print(f"     {os.path.basename(f.filename)}:{f.lineno} in {f.name}")

# (i) production class march
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
nsm = nsm_for("stoker", 2)
mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=50)
s = HyperbolicSolver(time_end=0.05)
Q, Qaux = s.setup_simulation(mesh, nsm)
orig = s._rt_flux_op
def broken(dt, time, Q, Qaux, p, dQ, *a, **k):
    out = orig(dt, time, Q, Qaux, p, dQ, *a, **k)
    return out[:, :-1]            # injected shape bug
s._rt_flux_op = broken
try:
    s.run_simulation(Q, Qaux, write_output=False)
except Exception as e:
    report("class/jax", e)

# (ii) v6 IR-emitted march
try:
    from zoomy_jax.fvm.march_emitted import EmittedMarchSolver
    from zoomy_jax.fvm.solver2 import blocks as B
    solver, S, nsm2, mesh2 = make_solver(EmittedMarchSolver, "stoker", 2, n_cells=50, t_end=0.05)
    origB = B.gather_update
    def bad_gather(*a, **k):
        out = origB(*a, **k)
        return (out[0][:, :-1],) + tuple(out[1:])
    B.gather_update = bad_gather
    try:
        print("has run:", hasattr(solver,"run")); out = solver.march(S); print("IR march returned without error", type(out))
    except Exception as e:
        report("IR/emitted", e)
        tb = traceback.extract_tb(e.__traceback__)
        for f in tb[-14:]:
            print("       ", os.path.basename(f.filename), f.lineno, f.name)
    finally:
        B.gather_update = origB
except Exception as e:
    report("IR/emitted-setup", e)
