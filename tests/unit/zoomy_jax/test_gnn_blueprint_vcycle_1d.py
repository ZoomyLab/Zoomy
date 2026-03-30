"""1D structured Poisson V-cycle hierarchy and Jacobi coarsest (negative diagonal)."""

from __future__ import annotations

import numpy as np
import pytest

from zoomy_jax.gnn_blueprint.mesh_1d_poisson import laplacian_1d_interior_dense
from zoomy_jax.gnn_blueprint.mg_structured_hierarchy_1d import (
    build_poisson_hierarchy_1d,
    build_poisson_hierarchy_1d_vector,
    restriction_prolongation_1d_pair,
)


def test_restriction_prolongation_rp_identity() -> None:
    r, p = restriction_prolongation_1d_pair(10)
    assert r.shape == (5, 10) and p.shape == (10, 5)
    np.testing.assert_allclose(r @ p, np.eye(5), atol=1e-12)


def test_build_hierarchy_shapes_end_at_one() -> None:
    _, rl, pl, _, sh = build_poisson_hierarchy_1d(16)
    assert sh[0] == 16 and sh[-1] == 1
    assert len(rl) == len(pl) == len(sh) - 1
    for a in build_poisson_hierarchy_1d(16)[0]:
        assert np.all(np.isfinite(a))


def test_vector_kron_rp_identity() -> None:
    _, rv, pv, _, sh = build_poisson_hierarchy_1d_vector(16, 3)
    assert sh[0] == 16 and sh[-1] == 1
    for i in range(len(rv)):
        n = rv[i].shape[0]
        np.testing.assert_allclose(rv[i] @ pv[i], np.eye(n), atol=1e-10)
    assert build_poisson_hierarchy_1d_vector(8, 3)[0][0].shape == (24, 24)


def test_gn_var_major_poisson_layout_roundtrip() -> None:
    from zoomy_jax.gnn_blueprint.vcycle_imex_bridge import (
        inner_var_major_to_poisson_flat,
        poisson_flat_to_inner_var_major,
    )

    n_in, nv = 8, 3
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_in * nv)
    p = inner_var_major_to_poisson_flat(v, n_in, nv)
    v2 = poisson_flat_to_inner_var_major(p, n_in, nv)
    np.testing.assert_allclose(v, v2, atol=1e-12)


def test_forward_vcycle_vector_finite() -> None:
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from zoomy_jax.gnn_blueprint.mg_structured_hierarchy_1d import build_poisson_hierarchy_1d_vector
    from zoomy_jax.gnn_blueprint.vcycle_structured_gnn import forward_vcycle, init_vcycle_smoothers

    d = 3
    a_list_np, r_list_np, p_list_np, edges_list_np, shapes = build_poisson_hierarchy_1d_vector(8, d)
    n_levels = len(a_list_np)
    n_state = a_list_np[0].shape[0]
    a_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in a_list_np)
    r_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in r_list_np)
    p_list = tuple(jnp.asarray(x, dtype=jnp.float64) for x in p_list_np)
    edges_list = tuple(jnp.asarray(e, dtype=jnp.int32) for e in edges_list_np)
    b_list = tuple(jnp.zeros(shapes[i], dtype=jnp.float64) for i in range(n_levels))
    key = jax.random.PRNGKey(1)
    params = init_vcycle_smoothers(key, n_levels, 16, 2, n_components=d, use_bump=False)
    f = jnp.ones((n_state,), dtype=jnp.float64)
    out = forward_vcycle(
        f,
        params,
        a_list,
        r_list,
        p_list,
        edges_list,
        b_list,
        2,
        16,
        1,
        1,
        40,
        0.5,
        d,
    )
    assert bool(jnp.isfinite(out).all())


def test_jacobi_coarsest_negative_diagonal_finite() -> None:
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from zoomy_jax.gnn_blueprint.vcycle_structured_gnn import jacobi_solve

    for n in (1, 2):
        a = jnp.asarray(laplacian_1d_interior_dense(n), dtype=jnp.float64)
        f = jnp.ones((n,), dtype=jnp.float64)
        u0 = jnp.zeros((n,), dtype=jnp.float64)
        u = jacobi_solve(u0, f, a, 40, 0.5)
        assert bool(jnp.isfinite(u).all()), f"Jacobi produced non-finite u for n={n}"
