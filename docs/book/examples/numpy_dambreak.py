"""The 1-D dam break on the NumPy backend: no JAX, no compiler."""
from dambreak import dambreak
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy

mesh, nsm = dambreak(order=2, n=300)
solver = SymbolicHyperbolicSolverNumpy(time_end=0.5, cfl=0.9)
Q, Qaux = solver.setup(mesh, nsm)
time, Q, Qaux = solver.run(Q, Qaux)

h = Q[1]
print(f"t = {time:.3f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}], "
      f"mass = {h.sum() * 10.0 / h.size:.6f}")
