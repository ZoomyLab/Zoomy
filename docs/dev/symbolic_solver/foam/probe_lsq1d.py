import sys
sys.path.insert(0, "/Users/adam-obbpb5az1dhsjzf/git/Zoomy-solver-template/docs/dev/symbolic_solver/gates")
import common, numpy as np
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.geometry_xp import build_topology
mesh, nsm = common.dambreak_1d(0, 2, nx=10)
mesh = ensure_lsq_mesh(mesh, nsm)
topo = build_topology(mesh, nsm=nsm)
print("SME0 degree", topo.lsq_degree, "mon", topo.lsq_monomial_multi_index, "scale", topo.lsq_scale_factors)
for c in (0, 1, 5, 9):
    print(" cell", c, "nbrs", topo.lsq_neighbors[c].tolist(), "bdy", topo.lsq_boundary_face_neighbors[c].tolist(), "mesh nbrs", np.asarray(mesh.lsq_neighbors[c]).tolist())
import gate7_chorin as g7, sympy as sp
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
pred, press, corr = g7.split(_DT_SYMBOL)
g7.cases.set_state_width(pred)
nsm2 = g7.pred_nsm(pred, press, corr)
m2 = ensure_lsq_mesh(g7.mesh(), nsm2)
t2 = build_topology(m2, nsm=nsm2)
print("VAM pred degree", t2.lsq_degree, "mon", t2.lsq_monomial_multi_index, "scale", t2.lsq_scale_factors)
for c in (0, 1, 30, 59):
    print(" cell", c, "nbrs", t2.lsq_neighbors[c].tolist(), "bdy", t2.lsq_boundary_face_neighbors[c].tolist())
