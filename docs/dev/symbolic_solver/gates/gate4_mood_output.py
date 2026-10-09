"""Gate 4: MOOD really engages on the dry front; the io.write effect fires at
the requested instants (forward mode)."""
import numpy as np
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

res = {}
for mood in (True, False):
    mesh, nsm = C.dry_dambreak_2d()
    s = SymbolicHyperbolicSolverJax(time_end=0.02, cfl=0.45, reconstruction_variables="eta", mood=mood)
    Q0, A0 = s.setup(mesh, nsm)
    _, Q, _ = s.run(Q0, A0)
    res[mood] = np.asarray(Q)
h = 1
print(f"MOOD on: h_min = {res[True][h].min():.3e}; MOOD off: h_min = "
      f"{res[False][h].min():.3e}; max|on-off| = {np.abs(res[True] - res[False]).max():.2e}")

writes = []
mesh, nsm = C.bump_2d(2, nx=20)
s = SymbolicHyperbolicSolverJax(time_end=0.5, cfl=0.9, dt_write=0.1,
                                on_write=lambda t, Q, A: writes.append(float(t)))
Q0, A0 = s.setup(mesh, nsm)
s.run(Q0, A0)
import jax; jax.effects_barrier()
print("io.write at t =", [round(t, 4) for t in writes])
