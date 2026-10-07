"""1-D dam break through the single-source body, NUMPY ONLY (no jax import).
Runs under CPython or Pyodide; asserts jax never enters sys.modules."""
import sys, time, numpy as np
from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import LSQMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
import shared_body as SB

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 400
NSTEP = 20
sm = SystemModel.from_model(SWE(dimension=1, boundary_conditions=BC.BoundaryConditions(
    [BC.Wall(tag="left"), BC.Wall(tag="right")])))
sm.initial_conditions = IC.UserFunction(function=lambda x: np.array([0.0, 2.0 if x[0] < 0.5 else 1.0, 0.0]))
sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"))
ref = HyperbolicSolver(time_end=1e9, compute_dt=timestepping.adaptive(CFL=0.4))
ref.setup_simulation(LSQMesh.create_1d((0.0, 1.0), N), nsm, write_output=False)
Q, A, p = ref._sim_Q.copy(), ref._sim_Qaux.copy(), ref._sim_parameters
dt = float(ref.compute_dt(Q, A, p, ref._sim_face_inradius, ref._sim_compute_max_abs_eigenvalue))
step, _ = SB.make_step(np, SB.NumpyOps(), SB.Geo(ref), SB.NumpyKernels(ref), p)
t = 0.0; ts = []
for _ in range(NSTEP):
    t0 = time.perf_counter(); Q, A = step(Q, A, t, dt); ts.append(time.perf_counter() - t0)
    ref._sim_time = t; ref.step(dt); t += dt
print(f"N={N} steps={NSTEP} max|Q_shared - Q_ref| = {np.abs(Q - ref._sim_Q).max():.3e}  "
      f"bitwise={np.array_equal(Q, ref._sim_Q)}  median {1e3*np.median(ts):.2f} ms/step")
print("jax in sys.modules:", "jax" in sys.modules, "| jaxlib:", "jaxlib" in sys.modules)
assert "jax" not in sys.modules
