"""Tests for ``MapToReferenceElement`` — multi-dim affine change-of-variable."""
import pytest
import sympy as sp

from zoomy_core.symbolic.domains import (
    BoundaryIntegral, NormalVector, Simplex,
)
from zoomy_core.model.models.divergence_theorem import DivergenceTheorem
from zoomy_core.model.models.map_to_reference import MapToReferenceElement


@pytest.fixture
def setup_2d_symbolic():
    x, y = sp.symbols("x y", real=True)
    x0, y0, x1, y1, x2, y2 = sp.symbols("x0 y0 x1 y1 x2 y2", real=True)
    K = Simplex([(x0, y0), (x1, y1), (x2, y2)], coords=(x, y), name="K")
    return x, y, K


@pytest.mark.small
@pytest.mark.unittest
def test_volume_only_integral_picks_up_jacobian(setup_2d_symbolic):
    x, y, K = setup_2d_symbolic
    g = sp.Function("g")(x, y)
    expr = sp.Integral(g, x, y)
    out = MapToReferenceElement(K)._leaf_sp(expr)

    B, V0 = K.affine_map()
    ref = K.reference()
    xi0, xi1 = ref.coords
    ref_image = V0 + B * sp.Matrix([[xi0], [xi1]])
    expected = sp.Integral(
        g.subs({x: ref_image[0, 0], y: ref_image[1, 0]}) * sp.Abs(B.det()),
        (xi0,), (xi1,))
    assert sp.simplify(sp.expand(out - expected)) == 0


@pytest.mark.small
@pytest.mark.unittest
def test_first_order_derivative_chain_rule_structure(setup_2d_symbolic):
    x, y, K = setup_2d_symbolic
    u = sp.Function("u")(x, y)
    expr = sp.Integral(sp.diff(u, x), x, y)
    out = MapToReferenceElement(K)._leaf_sp(expr)

    B, V0 = K.affine_map()
    Binv = B.inv()
    ref = K.reference()
    xi0, xi1 = ref.coords
    ref_image = V0 + B * sp.Matrix([[xi0], [xi1]])
    u_ref = u.subs({x: ref_image[0, 0], y: ref_image[1, 0]})
    expected_inner = (Binv[0, 0] * sp.Derivative(u_ref, xi0)
                      + Binv[1, 0] * sp.Derivative(u_ref, xi1))
    expected = sp.Integral(expected_inner * sp.Abs(B.det()),
                           (xi0,), (xi1,))
    assert sp.simplify(sp.expand(out - expected)) == 0


@pytest.mark.small
@pytest.mark.unittest
def test_higher_order_derivative_raises(setup_2d_symbolic):
    x, y, K = setup_2d_symbolic
    u = sp.Function("u")(x, y)
    expr = sp.Integral(sp.diff(u, x, 2), x, y)
    with pytest.raises(NotImplementedError,
                       match="higher-order Derivative"):
        MapToReferenceElement(K)._leaf_sp(expr)


@pytest.mark.small
@pytest.mark.unittest
def test_poisson_ibp_then_map_matches_textbook():
    """End-to-end: volume integrand after IBP + mapping == (B⁻ᵀ ∇_ξ φ)·(B⁻ᵀ ∇_ξ u) |det B|."""
    x, y = sp.symbols("x y", real=True)
    x0, y0, x1, y1, x2, y2 = sp.symbols("x0 y0 x1 y1 x2 y2", real=True)
    K = Simplex([(x0, y0), (x1, y1), (x2, y2)], coords=(x, y), name="K")
    phi = sp.Function("phi")(x, y)
    u = sp.Function("u")(x, y)
    F = (sp.diff(u, x), sp.diff(u, y))
    expr = sp.Integral(phi * (sp.diff(u, x, 2) + sp.diff(u, y, 2)), x, y)

    ibp = DivergenceTheorem(K, phi=phi, F=F, form="weighted")._leaf_sp(expr)
    mapped = MapToReferenceElement(K)._leaf_sp(ibp)

    B, V0 = K.affine_map()
    BinvT = B.inv().T
    ref = K.reference()
    xi0, xi1 = ref.coords
    ref_image = V0 + B * sp.Matrix([[xi0], [xi1]])
    phi_ref = phi.subs({x: ref_image[0, 0], y: ref_image[1, 0]})
    u_ref = u.subs({x: ref_image[0, 0], y: ref_image[1, 0]})
    grad_phi_xi = sp.Matrix([sp.diff(phi_ref, xi0), sp.diff(phi_ref, xi1)])
    grad_u_xi = sp.Matrix([sp.diff(u_ref, xi0), sp.diff(u_ref, xi1)])
    expected_inner = ((BinvT * grad_phi_xi).T @ (BinvT * grad_u_xi))[0, 0] \
        * sp.Abs(B.det())

    def _has_func(expr, name):
        return any(a.func.__name__ == name
                   for a in expr.atoms(sp.Function))

    volume_pieces = [I for I in mapped.atoms(sp.Integral)
                     if I.args[1][0] == xi0]
    laplace_vol = next(I for I in volume_pieces
                       if _has_func(I.args[0], "u"))
    # The mapped integrand keeps Derivative held; the textbook
    # expected uses sp.diff which auto-applies the chain rule into
    # Subs form.  They agree under .doit() (which expands held
    # Derivatives the same way sp.diff does).
    diff = sp.simplify(sp.expand(
        laplace_vol.args[0].doit() - expected_inner.doit()))
    assert diff == 0


@pytest.mark.small
@pytest.mark.unittest
def test_boundary_integral_is_transported_to_ref_boundary():
    x, y = sp.symbols("x y", real=True)
    K = Simplex([(0, 0), (1, 0), (0, 1)], coords=(x, y), name="K")
    u = sp.Function("u")(x, y)
    n = NormalVector(K.boundary())
    bi = BoundaryIntegral(sp.diff(u, x) * n(0), K.boundary())
    out = MapToReferenceElement(K)._leaf_sp(bi)
    # The result is a BoundaryIntegral on the reference boundary.
    assert hasattr(out.func, "_domain")
    assert out.func._domain is K.reference().boundary()
