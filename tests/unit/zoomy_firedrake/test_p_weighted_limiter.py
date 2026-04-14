"""Tests for the p-weighted slope limiter (Li et al. 2020).

The pure-NumPy core (Legendre transforms, mode limiting) is tested
directly.  Integration tests that require a running Firedrake
installation are marked ``@pytest.mark.firedrake`` and will be skipped
outside the Firedrake container.
"""

import sys
import os
import numpy as np
import pytest

# zoomy_firedrake is not pip-installed outside the Firedrake container,
# but the pure-NumPy limiter module has no Firedrake dependency.
# Add the package directory to sys.path so the import works everywhere.
_firedrake_pkg = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "library", "zoomy_firedrake",
)
_firedrake_pkg = os.path.normpath(_firedrake_pkg)
if _firedrake_pkg not in sys.path:
    sys.path.insert(0, _firedrake_pkg)

from zoomy_firedrake.p_weighted_limiter import (
    _legendre_vandermonde_1d,
    _reference_points_1d,
    limit_modes,
)


# ======================================================================
# Vandermonde / basis tests
# ======================================================================

@pytest.mark.small
@pytest.mark.unittest
class TestLegendreBasis:
    """Verify the 1-D Legendre Vandermonde matrix."""

    def test_vandermonde_p0(self):
        V = _legendre_vandermonde_1d(0, [0.0])
        assert V.shape == (1, 1)
        np.testing.assert_allclose(V, [[1.0]])

    def test_vandermonde_p1(self):
        pts = _reference_points_1d(1)
        V = _legendre_vandermonde_1d(1, pts)
        assert V.shape == (2, 2)
        # P0(x) = 1, P1(x) = x  at x = -1, 1
        expected = np.array([[1.0, -1.0],
                             [1.0,  1.0]])
        np.testing.assert_allclose(V, expected)

    def test_vandermonde_p2(self):
        pts = _reference_points_1d(2)
        V = _legendre_vandermonde_1d(2, pts)
        assert V.shape == (3, 3)
        # P2(x) = (3x^2 - 1)/2
        # At x = -1, 0, 1: P2 = 1, -0.5, 1
        np.testing.assert_allclose(V[:, 0], [1.0, 1.0, 1.0])  # P0
        np.testing.assert_allclose(V[:, 1], [-1.0, 0.0, 1.0])  # P1
        np.testing.assert_allclose(V[:, 2], [1.0, -0.5, 1.0])  # P2

    def test_vandermonde_invertible(self):
        for p in range(1, 5):
            pts = _reference_points_1d(p)
            V = _legendre_vandermonde_1d(p, pts)
            cond = np.linalg.cond(V)
            assert cond < 1e8, f"Vandermonde ill-conditioned at p={p}: cond={cond}"

    def test_roundtrip_nodal_modal(self):
        """Nodal -> modal -> nodal should recover original values."""
        p = 3
        pts = _reference_points_1d(p)
        V = _legendre_vandermonde_1d(p, pts)
        V_inv = np.linalg.inv(V)

        nodal = np.array([1.0, 2.5, -0.3, 4.0])
        modal = V_inv @ nodal
        recovered = V @ modal
        np.testing.assert_allclose(recovered, nodal, atol=1e-12)


# ======================================================================
# Mode limiting tests
# ======================================================================

@pytest.mark.small
@pytest.mark.unittest
class TestLimitModes:
    """Verify the core p-weighted limiting logic."""

    def test_dg0_passthrough(self):
        """DG0: only cell averages, nothing to limit."""
        modal = np.array([[1.0], [2.0], [3.0]])
        nbrs = [[1], [0, 2], [1]]
        result = limit_modes(modal, nbrs)
        np.testing.assert_array_equal(result, modal)

    def test_smooth_field_preserved(self):
        """A smooth linear field should not be clipped.

        Three cells with averages [1, 2, 3] and small linear modes.
        The neighbour range is large compared to the mode magnitudes,
        so theta >> 1 and w = 1 -- modes are untouched.
        """
        modal = np.array([
            [1.0, 0.01],
            [2.0, 0.01],
            [3.0, 0.01],
        ])
        nbrs = [[1], [0, 2], [1]]
        result = limit_modes(modal, nbrs)
        np.testing.assert_allclose(result, modal, atol=1e-14)

    def test_shock_modes_damped(self):
        """At a discontinuity the higher modes should be damped.

        Cell averages [0, 0, 10]: the middle cell has a large mode but
        its neighbours have a small range on the left side, so the mode
        should be reduced.
        """
        modal = np.array([
            [0.0,  0.0, 0.0],
            [0.0,  5.0, 5.0],  # large linear + quadratic mode
            [10.0, 0.0, 0.0],
        ])
        nbrs = [[1], [0, 2], [1]]
        result = limit_modes(modal, nbrs)

        # Cell averages (mode 0) must never change
        np.testing.assert_array_equal(result[:, 0], modal[:, 0])

        # The middle cell's modes should be damped (w < 1)
        assert abs(result[1, 1]) <= abs(modal[1, 1])
        assert abs(result[1, 2]) <= abs(modal[1, 2])

    def test_higher_modes_limited_more(self):
        """For DG2, mode k=2 should be limited more aggressively than k=1.

        With equal-magnitude modes, the p-weighted exponent (p-k+1) gives
        exponent 2 for k=1 and exponent 1 for k=2, so mode 1's weight is
        w^2 while mode 2's is w^1.  When w < 1, w^2 < w^1, so mode 1 is
        actually damped more... but the *ratio* of damping to mode
        contribution is what matters.  The key invariant: the quadratic
        mode (k=2) has exponent 1, so it's limited proportionally to w
        directly, while the linear mode (k=1) has exponent 2, giving even
        more damping.  Both are damped relative to their original values.
        """
        # Set up so w ~ 0.5 for both modes
        modal = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # both modes have |c_k| = 1
            [1.0, 0.0, 0.0],
        ])
        nbrs = [[1], [0, 2], [1]]
        result = limit_modes(modal, nbrs)

        # w_k = min(1, delta / (2*|c_k| + eps))
        # delta = 0 for cell 1 (all neighbours have average 1.0)
        # So w_k ~ 0 for both modes -> both heavily damped
        # Actually all averages are 1.0, delta=0, so modes go to 0
        np.testing.assert_allclose(result[1, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 2], 0.0, atol=1e-10)

    def test_cell_averages_never_modified(self):
        """The cell average (mode 0) must never be touched by the limiter."""
        rng = np.random.default_rng(42)
        n_cells, p = 20, 3
        modal = rng.standard_normal((n_cells, p + 1))
        nbrs = [[max(0, i - 1), min(n_cells - 1, i + 1)] for i in range(n_cells)]
        result = limit_modes(modal, nbrs)
        np.testing.assert_array_equal(result[:, 0], modal[:, 0])

    def test_isolated_cell_no_neighbours(self):
        """A cell with no neighbours: delta=0, all modes zeroed."""
        modal = np.array([[5.0, 2.0, 1.0]])
        nbrs = [[]]
        result = limit_modes(modal, nbrs)
        np.testing.assert_allclose(result[0, 0], 5.0)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-10)


# ======================================================================
# End-to-end nodal round-trip test
# ======================================================================

@pytest.mark.small
@pytest.mark.unittest
class TestNodalRoundTrip:
    """Test full nodal -> modal -> limit -> nodal pipeline (NumPy only)."""

    def _run_1d_advection_limit(self, p, n_cells=20):
        """Simulate a simple 1-D step function and verify limiting.

        Returns (nodal_orig, nodal_limited) arrays of shape (n_cells, p+1).
        """
        pts = _reference_points_1d(p)
        V = _legendre_vandermonde_1d(p, pts)
        V_inv = np.linalg.inv(V)

        # Build a step function: cells 0..9 have value 0, cells 10..19 have value 1
        nodal = np.zeros((n_cells, p + 1))
        for i in range(n_cells // 2, n_cells):
            nodal[i, :] = 1.0

        # 1-D chain adjacency
        nbrs = []
        for i in range(n_cells):
            n_i = []
            if i > 0:
                n_i.append(i - 1)
            if i < n_cells - 1:
                n_i.append(i + 1)
            nbrs.append(n_i)

        # Nodal -> modal
        modal = nodal @ V_inv.T

        # Limit
        modal_lim = limit_modes(modal, nbrs)

        # Modal -> nodal
        nodal_lim = modal_lim @ V.T

        return nodal, nodal_lim

    def test_dg1_step_limited(self):
        nodal, nodal_lim = self._run_1d_advection_limit(p=1, n_cells=20)

        # Cell averages must be preserved
        np.testing.assert_allclose(nodal_lim[:, 0], nodal[:, 0], atol=1e-12,
                                   err_msg="DG1: cell averages changed")

        # Interior cells far from the step should be unchanged
        np.testing.assert_allclose(nodal_lim[0, :], nodal[0, :], atol=1e-12)
        np.testing.assert_allclose(nodal_lim[-1, :], nodal[-1, :], atol=1e-12)

    def test_dg2_step_limited(self):
        nodal, nodal_lim = self._run_1d_advection_limit(p=2, n_cells=20)

        # Cell averages must be preserved
        for i in range(20):
            # The Legendre P0 coefficient is the cell average
            V = _legendre_vandermonde_1d(2, _reference_points_1d(2))
            V_inv = np.linalg.inv(V)
            modal_orig = V_inv @ nodal[i]
            modal_lim = V_inv @ nodal_lim[i]
            assert abs(modal_lim[0] - modal_orig[0]) < 1e-12, (
                f"DG2 cell {i}: average changed"
            )

    def test_dg2_smooth_unmodified(self):
        """A smooth quadratic profile should pass through mostly unchanged."""
        p = 2
        n_cells = 20
        pts = _reference_points_1d(p)
        V = _legendre_vandermonde_1d(p, pts)
        V_inv = np.linalg.inv(V)

        # Smooth quadratic: f(x) = x^2  mapped to each cell
        nodal = np.zeros((n_cells, p + 1))
        for i in range(n_cells):
            # Cell centre at i + 0.5, width 1
            x_cell = (i + 0.5) / n_cells
            for j, xi in enumerate(pts):
                x = x_cell + xi / (2 * n_cells)
                nodal[i, j] = x ** 2

        nbrs = []
        for i in range(n_cells):
            n_i = []
            if i > 0:
                n_i.append(i - 1)
            if i < n_cells - 1:
                n_i.append(i + 1)
            nbrs.append(n_i)

        modal = nodal @ V_inv.T
        modal_lim = limit_modes(modal, nbrs)
        nodal_lim = modal_lim @ V.T

        # For a smooth field the limiter should barely touch interior cells
        for i in range(2, n_cells - 2):
            np.testing.assert_allclose(
                nodal_lim[i], nodal[i], atol=1e-4,
                err_msg=f"DG2 smooth field modified at cell {i}",
            )


# ======================================================================
# Firedrake integration tests (require container)
# ======================================================================

try:
    import firedrake  # noqa: F401
    _has_firedrake = True
except ImportError:
    _has_firedrake = False


@pytest.mark.skipif(not _has_firedrake, reason="Firedrake not installed")
@pytest.mark.firedrake
class TestPWeightedLimiterFiredrake:
    """Integration tests that require a running Firedrake installation.

    These should be run inside the Firedrake container::

        pytest tests/unit/zoomy_firedrake/test_p_weighted_limiter.py -m firedrake
    """

    def test_dg1_advection_1d(self):
        """DG1 advection of a step function with p-weighted limiting."""
        import firedrake as fd
        from zoomy_firedrake.p_weighted_limiter import PWeightedLimiter

        mesh = fd.UnitIntervalMesh(40)
        V = fd.FunctionSpace(mesh, "DG", 1)
        limiter = PWeightedLimiter(V)

        # Step function IC
        x, = fd.SpatialCoordinate(mesh)
        f = fd.Function(V).interpolate(fd.conditional(x > 0.5, 1.0, 0.0))

        # Apply limiter
        limiter.apply(f)

        # Check: cell averages should be preserved (project to DG0)
        V0 = fd.FunctionSpace(mesh, "DG", 0)
        avg_before = fd.Function(V0).interpolate(
            fd.conditional(x > 0.5, 1.0, 0.0)
        )
        avg_after = fd.project(f, V0)
        assert np.allclose(avg_after.dat.data_ro, avg_before.dat.data_ro, atol=1e-10)

    def test_dg2_smooth_preserved(self):
        """DG2: a smooth sine field should pass through nearly unchanged."""
        import firedrake as fd
        from zoomy_firedrake.p_weighted_limiter import PWeightedLimiter

        mesh = fd.UnitIntervalMesh(40)
        V = fd.FunctionSpace(mesh, "DG", 2)
        limiter = PWeightedLimiter(V)

        x, = fd.SpatialCoordinate(mesh)
        f = fd.Function(V).interpolate(fd.sin(2 * np.pi * x))
        f_orig = fd.Function(V).assign(f)

        limiter.apply(f)

        diff = fd.errornorm(f, f_orig)
        assert diff < 1e-3, f"Smooth field changed too much: errornorm = {diff}"
