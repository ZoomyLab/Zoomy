import numpy as np, inspect, traceback
from common import *
m = SWE(dimension=2, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in ("left","right","bottom","top")] ))
nsm = build_nsm(m, order=1)
rt = NumpyRuntimeModel.from_system_model(nsm.sm)
fn = rt.boundary_conditions
n = 5
try:
    fn(np.array([0,1,2,3,0]), 0.0, np.zeros((3,n)), np.ones(n), np.ones((4,n)), np.ones((1,n)), np.array([9.81,0,0,0,1e-12,0.01]), np.vstack([np.ones(n), np.zeros(n)]))
except Exception:
    traceback.print_exc()
# find the compiled lambdified source
cl = fn.__closure__
for c in cl:
    v = c.cell_contents
    if callable(v) and getattr(v, "__name__", "") == "_lambdifygenerated":
        print(inspect.getsource(v)[:3000])
