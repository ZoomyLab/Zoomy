"""Tests for ColumnStructure: vertical integration on extruded meshes.

Verifies:
- Column cell indexing matches extrusion convention
- Numerical integration accuracy for polynomial profiles
- Partial (running) integration
- scatter/gather round-trip
"""

import numpy as np
import pytest

from zoomy_core.mesh.column_structure import ColumnStructure


# ── Column indexing ──────────────────────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_column_cell_indexing():
    """Column cells follow cell_3d = iz * n_horizontal + i_horizontal."""
    n_h, n_z = 5, 3
    cs = ColumnStructure(n_h, n_z)

    assert cs.column_cells.shape == (n_h, n_z)
    for i_h in range(n_h):
        for iz in range(n_z):
            assert cs.column_cells[i_h, iz] == iz * n_h + i_h


@pytest.mark.core
@pytest.mark.numpy
def test_zeta_midpoints():
    """Layer midpoints are centered in each layer."""
    cs = ColumnStructure(3, 4)
    expected = np.array([0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(cs.zeta_midpoints, expected)


# ── Full vertical integration ────────────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_integrate_constant():
    """integral_0^1 c dzeta = c for constant profile."""
    n_h, n_z = 4, 10
    cs = ColumnStructure(n_h, n_z)

    Q = 3.0 * np.ones((1, n_h * n_z))
    result = cs.integrate(Q)

    assert result.shape == (1, n_h)
    np.testing.assert_allclose(result, 3.0)


@pytest.mark.core
@pytest.mark.numpy
def test_integrate_linear():
    """integral_0^1 zeta dzeta = 0.5 (midpoint rule exact for linear)."""
    n_h, n_z = 3, 20
    cs = ColumnStructure(n_h, n_z)

    # Fill Q with zeta values at layer midpoints
    Q = np.zeros((1, n_h * n_z))
    for iz in range(n_z):
        zeta = cs.zeta_midpoints[iz]
        cells = cs.column_cells[:, iz]
        Q[0, cells] = zeta

    result = cs.integrate(Q)
    np.testing.assert_allclose(result, 0.5, atol=1e-14)


@pytest.mark.core
@pytest.mark.numpy
def test_integrate_quadratic_convergence():
    """Midpoint rule is O(h^2) for smooth functions.

    integral_0^1 zeta^2 dzeta = 1/3.
    Check that error decreases as O(1/Nz^2).
    """
    n_h = 2
    errors = []
    Nzs = [10, 20, 40, 80]

    for n_z in Nzs:
        cs = ColumnStructure(n_h, n_z)
        Q = np.zeros((1, n_h * n_z))
        for iz in range(n_z):
            zeta = cs.zeta_midpoints[iz]
            cells = cs.column_cells[:, iz]
            Q[0, cells] = zeta ** 2
        result = cs.integrate(Q)
        err = abs(result[0, 0] - 1.0 / 3.0)
        errors.append(err)

    # Check O(h^2) convergence
    for i in range(1, len(errors)):
        rate = np.log(errors[i - 1] / errors[i]) / np.log(Nzs[i] / Nzs[i - 1])
        assert rate > 1.9, f"Integration rate {rate:.2f} < 1.9 at Nz={Nzs[i]}"


@pytest.mark.core
@pytest.mark.numpy
def test_integrate_multivariable():
    """Integration works with multiple variables."""
    n_h, n_z = 3, 10
    cs = ColumnStructure(n_h, n_z)

    Q = np.zeros((2, n_h * n_z))
    for iz in range(n_z):
        zeta = cs.zeta_midpoints[iz]
        cells = cs.column_cells[:, iz]
        Q[0, cells] = 1.0       # integral = 1
        Q[1, cells] = 2 * zeta  # integral = 1

    result = cs.integrate(Q)
    assert result.shape == (2, n_h)
    np.testing.assert_allclose(result[0, :], 1.0, atol=1e-14)
    np.testing.assert_allclose(result[1, :], 1.0, atol=1e-14)


# ── Partial (running) integration ────────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_partial_integrate_constant_from_top():
    """Running integral of constant from top: integral_zeta^1 c dzeta' = c(1-zeta)."""
    n_h, n_z = 2, 10
    cs = ColumnStructure(n_h, n_z)
    c = 2.0

    Q = c * np.ones((1, n_h * n_z))
    result = cs.partial_integrate(Q, from_top=True)

    # At layer iz, the running integral from top should be c * (1 - zeta_bottom)
    # where zeta_bottom = iz / n_z (bottom of layer iz)
    # After accumulating from top to layer iz: c * (n_z - iz) * dzeta
    for iz in range(n_z):
        cells = cs.column_cells[:, iz]
        expected = c * (n_z - iz) * cs.d_zeta
        np.testing.assert_allclose(result[0, cells], expected, atol=1e-14)


@pytest.mark.core
@pytest.mark.numpy
def test_partial_integrate_bottom_equals_full():
    """Partial integral from bottom at the top layer = full integral."""
    n_h, n_z = 3, 15
    cs = ColumnStructure(n_h, n_z)

    Q = np.random.RandomState(42).randn(1, n_h * n_z)
    partial = cs.partial_integrate(Q, from_top=False)
    full = cs.integrate(Q)

    # At top layer (iz = n_z - 1), partial should equal full integral
    top_cells = cs.column_cells[:, n_z - 1]
    np.testing.assert_allclose(partial[0, top_cells], full[0, :], atol=1e-14)


# ── Scatter / Gather ─────────────────────────────────────────────────────────


@pytest.mark.core
@pytest.mark.numpy
def test_scatter_gather_roundtrip():
    """Scatter horizontal values to column, gather back at any layer."""
    n_h, n_z = 4, 5
    cs = ColumnStructure(n_h, n_z)

    Q_horiz = np.array([[1.0, 2.0, 3.0, 4.0]])
    Q_3d = cs.scatter_to_column(Q_horiz)

    assert Q_3d.shape == (1, n_h * n_z)
    for iz in range(n_z):
        gathered = cs.gather_layer(Q_3d, iz)
        np.testing.assert_allclose(gathered, Q_horiz)


@pytest.mark.core
@pytest.mark.numpy
def test_integrate_scattered_constant():
    """Integrating a scattered constant gives back the constant."""
    n_h, n_z = 3, 8
    cs = ColumnStructure(n_h, n_z)

    Q_horiz = np.array([[5.0, 7.0, 9.0]])
    Q_3d = cs.scatter_to_column(Q_horiz)
    result = cs.integrate(Q_3d)

    np.testing.assert_allclose(result, Q_horiz, atol=1e-14)
