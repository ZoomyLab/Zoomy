import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../docs/dev/symbolic_solver/gates"))
import common, numpy as np
from zoomy_core.mesh import ensure_lsq_mesh
mesh, nsm = common.bump_2d(2, nx=40)
mesh = ensure_lsq_mesh(mesh, nsm)
print("degree", nsm.resolved_lsq_degree(), "monomials", np.asarray(mesh.lsq_monomial_multi_index).tolist())
nb = np.asarray(mesh.lsq_neighbors); bn = np.asarray(mesh.lsq_boundary_face_neighbors)
cc = np.asarray(mesh.cell_centers)
for c in (0, 1, 41, 820):
    print("cell", c, "xc", cc[:2, c].round(3), "nbrs", nb[c].tolist(), "bdy", bn[c].tolist())
    print("   nbr offsets", [tuple((cc[:2, n] - cc[:2, c]).round(3)) for n in nb[c] if n < mesh.n_inner_cells])
