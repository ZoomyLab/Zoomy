"""Tests for the symbolic Domain / BoundaryIntegral / NormalVector layer."""
import pytest
import sympy as sp

from zoomy_core.symbolic.domains import (
    BoundaryIntegral, Box, Interval, NormalVector, PointSet, Simplex,
    SimplexBoundary,
)


@pytest.mark.small
@pytest.mark.unittest
def test_interval_basic():
    x = sp.Symbol("x", real=True)
    I = Interval(x, 0, 1, name="I")
    assert I.coords == (x,)
    assert I.dim == 1
    B, V0 = I.affine_map()
    assert B == sp.Matrix([[1]])
    assert V0 == sp.Matrix([[0]])


@pytest.mark.small
@pytest.mark.unittest
def test_interval_boundary_caches():
    x = sp.Symbol("x", real=True)
    I = Interval(x, 0, 1, name="I")
    # boundary() must return the same instance on repeat calls so the
    # manufactured Function classes (BoundaryIntegral / NormalVector)
    # stay structurally equal across the codebase.
    assert I.boundary() is I.boundary()
    assert I.reference() is I.reference()


@pytest.mark.small
@pytest.mark.unittest
def test_simplex_2d_affine_map():
    x, y = sp.symbols("x y", real=True)
    x0, y0, x1, y1, x2, y2 = sp.symbols("x0 y0 x1 y1 x2 y2", real=True)
    K = Simplex([(x0, y0), (x1, y1), (x2, y2)], coords=(x, y), name="K")
    B, V0 = K.affine_map()
    assert B == sp.Matrix([[x1 - x0, x2 - x0],
                           [y1 - y0, y2 - y0]])
    assert V0 == sp.Matrix([[x0], [y0]])


@pytest.mark.small
@pytest.mark.unittest
def test_simplex_reference_canonical_vertices():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    ref = K.reference()
    assert ref.dim == 2
    assert ref.vertices[0] == sp.Matrix([[0], [0]])
    assert ref.vertices[1] == sp.Matrix([[1], [0]])
    assert ref.vertices[2] == sp.Matrix([[0], [1]])


@pytest.mark.small
@pytest.mark.unittest
def test_simplex_wrong_vertex_count_raises():
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="needs 3 vertices"):
        Simplex([(0, 0), (1, 0)], coords=(x, y))


@pytest.mark.small
@pytest.mark.unittest
def test_box_affine_map_is_diagonal():
    x, y = sp.symbols("x y", real=True)
    Ix = Interval(x, 0, 2, name="Ix")
    Iy = Interval(y, 0, 3, name="Iy")
    R = Box([Ix, Iy], name="R")
    B, V0 = R.affine_map()
    assert B == sp.diag(2, 3)
    assert V0 == sp.Matrix([[0], [0]])


@pytest.mark.small
@pytest.mark.unittest
def test_boundary_of_boundary_raises():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    with pytest.raises(NotImplementedError,
                       match="boundary-of-boundary"):
        K.boundary().boundary()


@pytest.mark.small
@pytest.mark.unittest
def test_volume_domain_has_no_normal():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    with pytest.raises(NotImplementedError,
                       match="not a boundary"):
        K.normal()


@pytest.mark.small
@pytest.mark.unittest
def test_boundary_integral_atom_round_trip():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    f = sp.Function("f")(x, y)
    bi1 = BoundaryIntegral(f, K.boundary())
    bi2 = BoundaryIntegral(f, K.boundary())
    # Two factory calls with the same domain produce the same atom
    # (caching makes the Function class identical).
    assert bi1 == bi2
    # Free-symbol propagation: the integrand's free symbols are visible.
    assert {x, y} <= bi1.free_symbols
    # Survives xreplace and subs.
    assert bi1.xreplace({f: f**2}).args[0] == f**2
    assert bi1.subs(x, 2*x).args[0] == f.subs(x, 2*x)


@pytest.mark.small
@pytest.mark.unittest
def test_normal_vector_per_domain_distinct_classes():
    x, y = sp.symbols("x y", real=True)
    K1 = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K1")
    K2 = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K2")
    n1 = NormalVector(K1.boundary())
    n2 = NormalVector(K2.boundary())
    # Different domains → distinct sympy classes (no accidental sharing).
    assert n1 is not n2
    assert n1(0) != n2(0)


@pytest.mark.small
@pytest.mark.unittest
def test_pointset_dim_zero():
    x = sp.Symbol("x", real=True)
    I = Interval(x, 0, 1, name="I")
    pts = I.boundary()
    assert isinstance(pts, PointSet)
    assert pts.dim == 0
    assert pts.points == (sp.S.Zero, sp.S.One)


@pytest.mark.small
@pytest.mark.unittest
def test_simplex_boundary_returns_normal():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    boundary = K.boundary()
    assert isinstance(boundary, SimplexBoundary)
    n_cls = boundary.normal()
    # The normal Function class carries _domain back to the boundary.
    assert n_cls._domain is boundary
