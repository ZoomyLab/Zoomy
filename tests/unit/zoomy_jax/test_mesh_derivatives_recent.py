"""LSQ derivatives on the jax mesh, pinned to core's ACTUAL boundary contract.

The stencil places its boundary sample at the **ghost-cell** position (the
symmetric image of the inner cell through the boundary face), not at the face.
So `u_bf` must carry the value at that ghost point:

  * Neumann-zero / zeroGradient ("extrapolation"): ghost = inner cell value,
    which is what `u_bf=None` gives you.
  * prescribed face value (Dirichlet): `u_ghost = 2*u_face - u_cell`.  Passing
    the BARE face value puts a face-valued sample at a ghost-positioned point
    and caps the boundary gradient at 1st order -- measured: max err 4.16 vs
    0.024 for the correct form, on the quadratic below.

See `zoomy_core.mesh.lsq_reconstruction._resolve_u_boundary_face`, whose
docstring is the source of truth (and which RAISES on None -- core forces
callers to state the BC; the jax kernels default to Neumann-zero instead).

The old version of this test asked for `err < 2e-2` from a QUADRATIC with no
boundary data at all. That is unachievable twice over: Neumann-zero is simply
the wrong BC for x^2+2x+1 (boundary slope 2 and 22, not 0), and even with the
correct ghost value a linear extrapolation cannot represent a quadratic at the
boundary (residual 2.4e-2). It was red for a different reason -- compute_derivatives
fed only lsq_neighbors to a stencil built over neighbours+boundary faces, so the
matmul contracted 5 against 4 and raised TypeError.
"""
import numpy as np
import pytest

from zoomy_core.mesh import LSQMesh


pytest.importorskip("jax")
import jax.numpy as jnp
from zoomy_jax.mesh.mesh import compute_derivatives, convert_mesh_to_jax


def _mesh():
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80)
    mesh._build_lsq_stencil(2)  # standalone mesh test — no NSM in scope
    return mesh, convert_mesh_to_jax(mesh)


def _ghost_values(mesh, f, u):
    """Core's convention: the stencil's boundary slot sits at the ghost point,
    so a prescribed face value enters as ``2*u_face - u_cell``."""
    xb = np.asarray(mesh.face_centers)[np.asarray(mesh.boundary_face_face_indices), 0]
    return 2.0 * f(xb) - u[np.asarray(mesh.boundary_face_cells)]


def _dx(u, jmesh, u_bf=None):
    return np.asarray(
        compute_derivatives(jnp.asarray(u), jmesh, derivatives_multi_index=[[1]],
                            u_bf=None if u_bf is None else jnp.asarray(u_bf))[:, 0],
        dtype=float,
    )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_jax_mesh_derivative_of_linear_is_exact_including_boundary():
    """A linear field is the case the ghost convention reproduces EXACTLY --
    boundary cells included. This is the test that actually exercises the
    boundary path end to end."""
    mesh, jmesh = _mesh()
    x = np.asarray(mesh.cell_centers[0, :], dtype=float)
    f = lambda t: 2.0 * t + 1.0
    u = f(x)
    d = _dx(u, jmesh, _ghost_values(mesh, f, u))
    n = mesh.n_inner_cells
    err = np.abs(d[:n] - 2.0)
    assert err.max() < 1e-6, f"linear must be exact everywhere, got {err.max():.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_jax_mesh_derivative_of_quadratic_is_exact_in_the_interior():
    """Order-2 stencil ⇒ a quadratic is reproduced exactly where the stencil is
    interior. The two boundary-adjacent cells are excluded on purpose: there the
    accuracy is set by the ghost extrapolation, not by this kernel (see module
    docstring). Interior previously CRASHED (5-vs-4 contraction)."""
    mesh, jmesh = _mesh()
    x = np.asarray(mesh.cell_centers[0, :], dtype=float)
    f = lambda t: t * t + 2.0 * t + 1.0
    u = f(x)
    d = _dx(u, jmesh, _ghost_values(mesh, f, u))
    n = mesh.n_inner_cells
    err = np.abs(d[:n] - (2.0 * x[:n] + 2.0))
    assert np.isfinite(err).all()
    assert err[2:n - 2].max() < 1e-4, f"interior must be exact, got {err[2:n-2].max():.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_bare_face_value_is_worse_than_the_ghost_form():
    """Pin the trap: passing the BARE face value where the stencil wants a
    ghost-position value degrades the boundary gradient. Documented by core and
    measured here, so the next caller (I was one) doesn't rediscover it."""
    mesh, jmesh = _mesh()
    x = np.asarray(mesh.cell_centers[0, :], dtype=float)
    f = lambda t: t * t + 2.0 * t + 1.0
    u = f(x)
    n = mesh.n_inner_cells
    exact = 2.0 * x[:n] + 2.0
    xb = np.asarray(mesh.face_centers)[np.asarray(mesh.boundary_face_face_indices), 0]

    err_bare = np.abs(_dx(u, jmesh, f(xb))[:n] - exact).max()          # WRONG usage
    err_ghost = np.abs(_dx(u, jmesh, _ghost_values(mesh, f, u))[:n] - exact).max()
    assert err_ghost < err_bare / 10.0, (
        f"ghost form {err_ghost:.3e} should be far better than bare face "
        f"{err_bare:.3e}")
