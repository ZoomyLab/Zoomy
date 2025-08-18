import jax.numpy as jnp
import os
import numpy as np

import library.mesh.mesh as petscMesh
from library.mesh.mesh import compute_derivatives
from library.mesh.mesh import convert_mesh_to_jax

def test_1d():
    N = 10
    def custom_ic(x):
        Q = jnp.zeros((1, N+2), dtype=float)
        Q = Q.at[0].set(x[0]*x[0])
        return Q

    
    mesh = petscMesh.Mesh.create_1d((0, 1), N, lsq_degree=2)
    mesh = convert_mesh_to_jax(mesh)
    X = mesh.cell_centers
    Q = custom_ic(X)
    d2Q_dx2 = compute_derivatives(Q[0], mesh, derivatives_multi_index=[[2]])
    assert(np.allclose(d2Q_dx2, 2. * np.ones_like(d2Q_dx2)))
    # print(d2Q_dx2)
    
def test_2d():
    main_dir = os.getenv("ZOOMY_DIR")
    mesh = petscMesh.Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"), lsq_degree=2
    )
    def custom_ic(x):
        Q = jnp.zeros((1, mesh.cell_centers.shape[1]), dtype=float)
        Q = Q.at[0].set(x[1]*x[1])
        return Q

    
    mesh = convert_mesh_to_jax(mesh)
    X = mesh.cell_centers
    Q = custom_ic(X)
    # print(Q)
    d2Q_d2y = compute_derivatives(Q[0], mesh, derivatives_multi_index=[[0, 2]])
    assert(np.allclose(d2Q_d2y, 2. * np.ones_like(d2Q_d2y)))
    
    

if __name__ == "__main__":
    test_1d()
    test_2d()



