"""Tests for ``DivergenceTheorem`` — multi-dim integration by parts."""
import pytest
import sympy as sp

from zoomy_core.symbolic.domains import (
    BoundaryIntegral, NormalVector, Simplex,
)
from zoomy_core.model.models.divergence_theorem import DivergenceTheorem


@pytest.fixture
def setup_2d():
    x, y = sp.symbols("x y", real=True)
    phi = sp.Function("phi", real=True)(x, y)
    u = sp.Function("u", real=True)(x, y)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    return x, y, phi, u, K


@pytest.mark.small
@pytest.mark.unittest
def test_bare_divergence_2d_triangle(setup_2d):
    x, y, phi, u, K = setup_2d
    F = (sp.diff(u, x), sp.diff(u, y))
    expr = sp.Integral(sp.diff(u, x, 2) + sp.diff(u, y, 2), x, y)
    out = DivergenceTheorem(K, F=F, form="bare")._leaf_sp(expr)
    n = NormalVector(K.boundary())
    expected = BoundaryIntegral(
        sp.diff(u, x) * n(0) + sp.diff(u, y) * n(1), K.boundary())
    assert out == expected


@pytest.mark.small
@pytest.mark.unittest
def test_weighted_poisson_ibp(setup_2d):
    x, y, phi, u, K = setup_2d
    F = (sp.diff(u, x), sp.diff(u, y))
    expr = sp.Integral(phi * (sp.diff(u, x, 2) + sp.diff(u, y, 2)), x, y)
    out = DivergenceTheorem(K, phi=phi, F=F, form="weighted")._leaf_sp(expr)
    n = NormalVector(K.boundary())
    expected_volume = -sp.Integral(
        sp.diff(phi, x) * sp.diff(u, x) + sp.diff(phi, y) * sp.diff(u, y),
        (x,), (y,))
    # Boundary atom is built term-by-term as ``phi · F[i] · n(i)`` and
    # summed; Mul commutativity normalises the order but no factoring.
    expected_boundary = BoundaryIntegral(
        phi * sp.diff(u, x) * n(0) + phi * sp.diff(u, y) * n(1),
        K.boundary())
    assert out == expected_volume + expected_boundary


@pytest.mark.small
@pytest.mark.unittest
def test_mismatched_F_passes_through(setup_2d):
    """A wrong F leaves the expression unchanged (silent no-op).
    Validation has to be soft because ``Expression.apply`` invokes the
    op per additive term, and source-only terms (e.g. ``φ·f``) are
    legitimate non-matches that must pass through."""
    x, y, phi, u, K = setup_2d
    expr = sp.Integral(phi * (sp.diff(u, x, 2) + sp.diff(u, y, 2)), x, y)
    out = DivergenceTheorem(K, phi=phi, F=(u, u), form="weighted")._leaf_sp(expr)
    assert out == expr


@pytest.mark.small
@pytest.mark.unittest
def test_no_matching_integral_passes_through(setup_2d):
    x, y, phi, u, K = setup_2d
    z = sp.Symbol("z", real=True)
    expr = sp.Integral(phi, z)
    out = DivergenceTheorem(K, phi=phi, F=(u, u), form="weighted")._leaf_sp(expr)
    assert out == expr


@pytest.mark.small
@pytest.mark.unittest
def test_form_validation():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    u = sp.Function("u")(x, y)
    F = (sp.diff(u, x), sp.diff(u, y))
    with pytest.raises(ValueError, match="form must be"):
        DivergenceTheorem(K, F=F, form="bogus")
    with pytest.raises(ValueError, match="phi"):
        DivergenceTheorem(K, F=F, form="weighted")  # missing phi
    with pytest.raises(ValueError, match="F=..."):
        DivergenceTheorem(K, form="bare")
    with pytest.raises(ValueError, match="domain.dim"):
        DivergenceTheorem(K, F=(u,), form="bare")


@pytest.mark.small
@pytest.mark.unittest
def test_mixed_integrand_with_residual_term(setup_2d):
    """φ·Δu + φ·f integrand: only the laplacian piece is IBP'd, the
    source term stays inside the original Integral."""
    x, y, phi, u, K = setup_2d
    f = sp.Function("f")(x, y)
    F = (sp.diff(u, x), sp.diff(u, y))
    # Pre-distributed form: phi·∂²u/∂x² + phi·∂²u/∂y² + phi·f.
    integrand = (phi * sp.diff(u, x, 2)
                 + phi * sp.diff(u, y, 2)
                 + phi * f)
    expr = sp.Integral(integrand, x, y)
    out = DivergenceTheorem(K, phi=phi, F=F, form="weighted")._leaf_sp(expr)
    # Result must contain: residual integral with phi·f, IBP volume
    # integral with -∇φ·∇u, and a single BoundaryIntegral with both
    # flux components combined.  Negation is kept *outside* the volume
    # Integral so the structural form matches the textbook.
    n = NormalVector(K.boundary())
    expected_volume = -sp.Integral(
        sp.diff(phi, x) * sp.diff(u, x) + sp.diff(phi, y) * sp.diff(u, y),
        (x,), (y,))
    expected_residual = sp.Integral(phi * f, (x,), (y,))
    expected_boundary = BoundaryIntegral(
        phi * sp.diff(u, x) * n(0) + phi * sp.diff(u, y) * n(1),
        K.boundary())
    assert out == expected_volume + expected_residual + expected_boundary
