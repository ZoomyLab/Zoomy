"""One preCICE participant running the symbolic hyperbolic_precice march.
    python participant.py NAME ORDER OUT.npy {window|subcycle} [DT_WRITE]"""
import sys
import numpy as np
import common as C
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

name, order, out, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
dt_write = float(sys.argv[5]) if len(sys.argv) > 5 else None
left = name == "Left"
mesh, nsm = C.dambreak_1d(0, order, domain=(0.0, 0.5) if left else (0.5, 1.0),
                          nx=100, x_dam=1.0 if left else 0.0)
tag, side = ("right", 0) if left else ("left", 1)
me, peer = ("L", "R") if left else ("R", "L")
adapter = R.PreciceAdapter(name, "precice-config.xml", f"{name}-Mesh",
                           [f"{me}{i}" for i in range(3)],
                           [f"{peer}{i}" for i in range(3)])
frames = []
s = SymbolicHyperbolicSolverJax(
    cfl=0.9, subcycling=(mode == "subcycle"), dt_write=dt_write,
    on_write=lambda t, Q, A: frames.append((float(t), np.asarray(Q))),
    couple=dict(adapter=adapter, tag=tag, side=side))
Q, Qaux = s.setup(mesh, nsm)
t, Q, Qaux = s.run(Q, Qaux)
adapter.finalize()
np.save(out, np.asarray(Q))
if frames:
    import jax
    jax.effects_barrier()
    np.savez(out.replace(".npy", "_frames.npz"), t=[f[0] for f in frames],
             Q=np.stack([f[1] for f in frames]))
import jax
print(f"{name}: t = {float(t):.12g} on {jax.devices()[0].platform}")
