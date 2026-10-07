"""F15b: which convention do the numpy / jax LSQ gradient + limiter bounds assume?
Feed a linear field u = a + b.x with bf values in FACE convention (what Dirichlet /
InflowOutflow / Lambda return) and in GHOST convention (what Wall / FromData return)."""
import numpy as np
from common import *
import jax.numpy as jnp
from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction
from zoomy_jax.fvm.reconstruction_jax import LSQMUSCLReconstructionJAX
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
import vk_bug

def run(dim, n):
    if dim == 1:
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
        bvec = np.array([0.7]); a = 1.0
    else:
        mesh = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=n, ny=n)
        bvec = np.array([0.7, -0.4]); a = 1.0
    nc = mesh.n_inner_cells
    cc = np.asarray(mesh.cell_centers)[:dim, :nc]
    geo = bf_geometry(mesh, dim)
    u = a + bvec @ cc
    uf = a + geo["fc"][:, :dim] @ bvec
    ucell_bf = u[geo["cells"]]
    ug = 2 * uf - ucell_bf
    Q = np.vstack([u, u])                    # row0 <- FACE bf, row1 <- GHOST bf
    bf = np.vstack([uf, ug])
    bcells = np.unique(geo["cells"])
    out = {}
    rn = LSQMUSCLReconstruction(mesh, dim, limiter="venkatakrishnan")
    g = rn._compute_gradients(Q, 2, bf_face_values=bf)            # (2, dim, nc)
    phi = rn._compute_phi(Q, 2, bf, g)
    out["numpy"] = (g, phi)
    jm = convert_mesh_to_jax(mesh)
    for tag, bug in (("jax(fixed VK, on disk)", False), ("jax(HEAD buggy VK)", True)):
        vk_bug.set_buggy(bug)
        rj = LSQMUSCLReconstructionJAX(jm, dim, limiter="venkatakrishnan")
        gj = np.asarray(rj._compute_gradients(jnp.asarray(Q), 2, jnp.asarray(bf)))
        pj = np.asarray(rj._compute_phi(jnp.asarray(Q), 2, jnp.asarray(bf), jnp.asarray(gj)))
        out[tag] = (gj, pj)
    vk_bug.set_buggy(False)
    inner = np.setdiff1d(np.arange(nc), bcells)
    print(f"\n--- create_{dim}d n={n}: exact grad = {bvec}; boundary cells = {len(bcells)} ---")
    print(f"{'backend':26s} {'bf conv':6s} {'max|grad-b| bnd':>16s} {'max|grad-b| int':>16s} {'min phi bnd':>12s} {'min phi int':>12s}")
    for k, (g, p) in out.items():
        for r, conv in ((0, "FACE"), (1, "GHOST")):
            eb = np.abs(g[r][:, bcells] - bvec[:, None]).max()
            ei = np.abs(g[r][:, inner] - bvec[:, None]).max() if len(inner) else 0
            print(f"{k:26s} {conv:6s} {eb:16.3e} {ei:16.3e} {p[r][bcells].min():12.6f} {p[r][inner].min() if len(inner) else 1:12.6f}")

for dim, n in ((1, 20), (1, 80), (2, 10), (2, 40)):
    run(dim, n)
