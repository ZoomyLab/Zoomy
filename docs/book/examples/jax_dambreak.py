"""The same dam break on JAX and on NumPy: one NSM, one printed template, two runtimes."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from dambreak import dambreak  # noqa: E402
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax  # noqa: E402

mesh, nsm = dambreak(order=2, n=300)

jax_solver = SymbolicHyperbolicSolverJax(time_end=0.5, cfl=0.9)
Q, Qaux = jax_solver.setup(mesh, nsm)             # prints the template, compiles
time, Q, Qaux = jax_solver.run(Q, Qaux)

numpy_solver = SymbolicHyperbolicSolverNumpy(time_end=0.5, cfl=0.9)
Qn, Qauxn = numpy_solver.setup(mesh, nsm)
_, Qn, _ = numpy_solver.run(Qn, Qauxn)

h = np.asarray(Q)[1]
print(f"t = {float(time):.3f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
print(f"max |h_jax - h_numpy| = {np.abs(h - Qn[1]).max():.1e}")
