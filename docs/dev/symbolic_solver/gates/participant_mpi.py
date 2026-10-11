"""gates/participant.py on R ranks (the runtime registers each rank's own
coupling vertices).  CASE 1d: gate 3's dam break halves; CASE 2d: a 2-D
dam break, interface x = 0.5 (ny vertices).
    [mpiexec -n R] python participant_mpi.py NAME ORDER OUT.npy {window|subcycle} CASE"""
import sys
import numpy as np
from mpi4py import MPI
import common as C
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

name, order, out, mode, case = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
comm = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
left = name == "Left"
if case == "1d":
    mesh, nsm = C.dambreak_1d(0, order, domain=(0.0, 0.5) if left else (0.5, 1.0),
                              nx=100, x_dam=1.0 if left else 0.0)
else:
    sm = SystemModel.from_model(C.sme(0, 3, ("left", "right", "bottom", "top"), "extrapolation"))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = (1.0 if x[0] < 0.5 else 0.5) + 0.1 * np.sin(2 * np.pi * x[1] / 0.2) * (x[0] < 0.5)
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    mesh = LSQMesh.create_2d(domain=(0.0, 0.5, 0.0, 0.2) if left else (0.5, 1.0, 0.0, 0.2),
                             nx=50, ny=20)
n_state = len(nsm.sm.state)
tag, side = ("right", 0) if left else ("left", 1)
me, peer = ("L", "R") if left else ("R", "L")
adapter = R.PreciceAdapter(name, "precice-config.xml", f"{name}-Mesh",
                           [f"{me}{i}" for i in range(n_state)],
                           [f"{peer}{i}" for i in range(n_state)], comm=comm)
import os
if os.environ.get("TRACE"):
    import mpi4jax, jax
    from jax.experimental import io_callback
    rk0 = 0 if comm is None else comm.Get_rank()
    def wrapop(opname):
        orig = getattr(mpi4jax, opname)
        def w(x, *a, **k):
            x = io_callback(lambda v: (print(f"[{name}{rk0}] {opname} > {v.shape}", flush=True), v)[1],
                            jax.ShapeDtypeStruct(x.shape, x.dtype), x)
            y = orig(x, *a, **k)
            return io_callback(lambda v: (print(f"[{name}{rk0}] {opname} <", flush=True), v)[1],
                               jax.ShapeDtypeStruct(y.shape, y.dtype), y)
        setattr(mpi4jax, opname, w)
    wrapop("allreduce"); wrapop("alltoall")
    rk = 0 if comm is None else comm.Get_rank()
    for meth in ("init", "max_dt", "read", "write", "advance"):
        f = getattr(adapter, meth)
        def wrap(*a, _f=f, _m=meth):
            print(f"[{name}{rk}] {_m} >", flush=True); r = _f(*a); print(f"[{name}{rk}] {_m} <", flush=True); return r
        setattr(adapter, meth, wrap)
s = SymbolicHyperbolicSolverJax(cfl=0.9, subcycling=(mode == "subcycle"), comm=comm,
                                couple=dict(adapter=adapter, tag=tag, side=side))
Q, Qaux = s.setup(mesh, nsm)
print(f"{name} rank {0 if comm is None else comm.Get_rank()}: "
      f"{len(adapter.ids)} coupling vertices", flush=True)
t, Q, Qaux = s.run(Q, Qaux)
adapter.finalize()
Qg = s.partition.gather(Q) if s.partition is not None else np.asarray(Q)
if comm is None or comm.Get_rank() == 0:
    np.save(out, np.asarray(Qg))
    print(f"{name}: t = {float(t):.12g}", flush=True)
