import numpy as np
import pytest

import zoomy_core.mesh.mesh as petscMesh


pytest.importorskip("jax")
import jax.numpy as jnp
from zoomy_jax.mesh.mesh import compute_derivatives, convert_mesh_to_jax


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_jax_mesh_derivative_of_quadratic():
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80, lsq_degree=2)
    jmesh = convert_mesh_to_jax(mesh)

    x = np.asarray(mesh.cell_centers[0, :], dtype=float)
    u = x * x + 2.0 * x + 1.0
    exact = 2.0 * x + 2.0

    d = np.asarray(
        compute_derivatives(jnp.asarray(u), jmesh, derivatives_multi_index=[[1]])[:, 0],
        dtype=float,
    )
    n = mesh.n_inner_cells
    err = np.max(np.abs(d[:n] - exact[:n]))
    assert np.isfinite(err)
    assert err < 2e-2

