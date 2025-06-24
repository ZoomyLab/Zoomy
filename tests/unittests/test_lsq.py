import jax.numpy as jnp

import library.mesh.mesh as petscMesh
from library.mesh.mesh import compute_derivatives
from library.mesh.mesh import convert_mesh_to_jax

def test_1d_linear():
    N = 10
    def custom_ic(x):
        Q = jnp.zeros((1, N+2), dtype=float)
        Q = Q.at[0].set(x[0]**2)
        return Q

    
    mesh = petscMesh.Mesh.create_1d((0, 1), N)
    mesh = convert_mesh_to_jax(mesh)
    X = mesh.cell_centers
    Q = custom_ic(X)
    print(compute_derivatives(Q[0], mesh))

if __name__ == "__main__":
    test_1d_linear()



