"""Quick start: the C1 bump on JAX (CPU or GPU)."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from bump import bump  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax  # noqa: E402

mesh, nsm = bump(order=2, nx=40)
solver = SymbolicHyperbolicSolverJax(time_end=2.5, cfl=0.9, mood=False)
Q, Qaux = solver.setup(mesh, nsm)            # prints the template, compiles
time, Q, Qaux = solver.run(Q, Qaux)

h = np.asarray(Q)[1]
print(f"t = {float(time):.3f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
np.save("quickstart_jax.npy", np.asarray(Q))
