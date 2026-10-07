"""F15c: effect of jax's 2*bf-u bounds lift on a GHOST-convention kernel (Wall):
wall-normal momentum q with a boundary-cell extremum; bf = Wall ghost = -q_cell."""
import numpy as np
from common import *
import jax.numpy as jnp
from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction
from zoomy_jax.fvm.reconstruction_jax import LSQMUSCLReconstructionJAX
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
for n in (20, 80):
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
    nc = n; x = np.asarray(mesh.cell_centers)[0, :nc]
    q = np.sin(3 * np.pi * x) + 0.3          # nonzero at walls, interior extrema
    geo = bf_geometry(mesh, 1)
    ghost = -q[geo["cells"]]                  # Wall kernel output (verified GHOST in F15)
    Q = q[None, :]; bf = ghost[None, :]
    rn = LSQMUSCLReconstruction(mesh, 1, limiter="venkatakrishnan")
    g = rn._compute_gradients(Q, 1, bf_face_values=bf); p = rn._compute_phi(Q, 1, bf, g)
    lo_n, hi_n = rn._neighbor_bounds(q, ghost)
    rj = LSQMUSCLReconstructionJAX(convert_mesh_to_jax(mesh), 1, limiter="venkatakrishnan")
    gj = np.asarray(rj._compute_gradients(jnp.asarray(Q), 1, jnp.asarray(bf)))
    pj = np.asarray(rj._compute_phi(jnp.asarray(Q), 1, jnp.asarray(bf), jnp.asarray(gj)))
    lo_j, hi_j = [np.asarray(v) for v in rj._neighbor_bounds(jnp.asarray(q), jnp.asarray(ghost))]
    c = geo["cells"]
    print(f"n={n}: wall cells {c}: q={q[c]}, ghost={ghost}")
    print(f"   numpy bounds min={lo_n[c]} max={hi_n[c]}  phi={p[0][c]}")
    print(f"   jax   bounds min={lo_j[c]} max={hi_j[c]}  phi={pj[0][c]}   (fixed VK)")
    print(f"   |grad_np-grad_jx| max={np.abs(g-gj).max():.2e}; reconstructed wall-face q: numpy={q[c]+p[0][c]*g[0,0,c]*(geo['fc'][:,0]-x[c])}, jax={q[c]+pj[0][c]*gj[0,0,c]*(geo['fc'][:,0]-x[c])}")
