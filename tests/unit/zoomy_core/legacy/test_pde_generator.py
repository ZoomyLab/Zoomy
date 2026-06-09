"""
Tests for the automated PDE generator (Phases 1–3).

Verifies:
  1.1 — INS base equations are well-formed
  1.2 — Galerkin projection produces abstract integrals with correct structure
  1.3 — Kinematic and dynamic BCs are applied correctly
  2.x — Basis injection with Heaviside windowing
  3.x — Custom integration engine for piecewise expressions
"""

import pytest
import sympy as sp
from sympy import (
    Function, Symbol, Derivative, Heaviside, DiracDelta,
    Rational, integrate, S, symbols,
)

from zoomy_core.model.models.pde_generator import (
    INSBase,
    GalerkinProjection,
    BoundaryConditions,
    ProjectedEquation,
    PiecewiseIntegrator,
    LayeredAnsatz,
)
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted, Monomials


# ---------------------------------------------------------------------------
# Phase 1.1: INS Base
# ---------------------------------------------------------------------------

class TestINSBase:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ins_1d_fields_exist(self):
        ins = INSBase(dimension=1)
        assert ins.u is not None
        assert ins.w is not None
        assert ins.p is not None
        assert ins.v is None
        assert ins.dimension == 1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ins_2d_fields_exist(self):
        ins = INSBase(dimension=2)
        assert ins.u is not None
        assert ins.v is not None
        assert ins.w is not None
        assert ins.p is not None
        assert ins.dimension == 2

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_continuity_1d_structure(self):
        ins = INSBase(dimension=1)
        cont = ins.continuity()
        assert cont.has(Derivative)
        free = cont.free_symbols
        assert ins.x in free or ins.z in free

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_continuity_2d_has_three_terms(self):
        ins = INSBase(dimension=2)
        cont = ins.continuity()
        terms = sp.Add.make_args(cont)
        assert len(terms) == 3

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_x_momentum_has_pressure(self):
        ins = INSBase(dimension=1)
        mom = ins.x_momentum()
        assert mom.has(ins.rho)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_y_momentum_only_in_2d(self):
        ins1 = INSBase(dimension=1)
        assert ins1.y_momentum() is None
        ins2 = INSBase(dimension=2)
        assert ins2.y_momentum() is not None

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_hydrostatic_pressure(self):
        ins = INSBase(dimension=1)
        hp = ins.hydrostatic_pressure()
        assert hp.has(ins.rho)
        assert hp.has(ins.g)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_momentum_equations_count(self):
        ins1 = INSBase(dimension=1)
        assert len(ins1.momentum_equations()) == 1
        ins2 = INSBase(dimension=2)
        assert len(ins2.momentum_equations()) == 2


# ---------------------------------------------------------------------------
# Phase 1.2: Galerkin Projection
# ---------------------------------------------------------------------------

class TestGalerkinProjection:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projection_creates_projected_equations(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eqs = proj.project_all()
        assert len(eqs) == 2
        assert all(isinstance(eq, ProjectedEquation) for eq in eqs)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projection_2d_creates_three_equations(self):
        ins = INSBase(dimension=2)
        proj = GalerkinProjection(ins)
        eqs = proj.project_all()
        assert len(eqs) == 3

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projected_continuity_has_boundary_terms(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eq = proj.project_continuity()
        assert eq.boundary_top != 0
        assert eq.boundary_bottom != 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projected_momentum_has_boundary_terms(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eq = proj.project_momentum(component_index=0)
        assert eq.boundary_top != 0
        assert eq.boundary_bottom != 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projected_continuity_has_volume_integral(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eq = proj.project_continuity()
        assert eq.volume != 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_zeta_coordinate_mapping(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        z_expr = proj.z_of_zeta()
        assert z_expr.has(proj.b)
        assert z_expr.has(proj.H)
        assert z_expr.has(proj.zeta)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_weight_function_appears_in_projection(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eq = proj.project_continuity()
        full = eq.full_equation()
        assert full.has(proj.c_weight)


# ---------------------------------------------------------------------------
# Phase 1.3: Boundary Conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_kinematic_bottom_1d(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        wb = bc.kinematic_bottom()
        assert wb.has(Derivative)
        assert wb.has(proj.b)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_kinematic_surface_1d(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        ws = bc.kinematic_surface()
        assert ws.has(Derivative)
        assert ws.has(proj.H)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_kinematic_bottom_2d_has_y_terms(self):
        ins = INSBase(dimension=2)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        wb = bc.kinematic_bottom()
        assert wb.has(ins.y)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_hydrostatic_pressure_gradient(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        dpx = bc.hydrostatic_pressure_gradient(ins.x)
        assert dpx.has(ins.g)
        assert dpx.has(proj.H) or dpx.has(proj.b)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_to_continuity_replaces_w(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        eq_raw = proj.project_continuity()
        eq_bc = bc.apply_to_continuity(eq_raw)
        assert "_with_kbc" in eq_bc.name
        full = eq_bc.boundary_top + eq_bc.boundary_bottom
        assert not full.has(Function("w_s"))
        assert not full.has(Function("w_b"))

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_to_momentum_adds_stress(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        eq_raw = proj.project_momentum(component_index=0)
        eq_bc = bc.apply_to_momentum(eq_raw, component_index=0)
        assert eq_bc.pressure_gradient is not None
        full_boundary = eq_bc.boundary_top + eq_bc.boundary_bottom
        assert full_boundary.has(bc.tau_sx) or full_boundary.has(bc.tau_bx)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_all_1d(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        raw_eqs = proj.project_all()
        bc_eqs = bc.apply_all(raw_eqs)
        assert len(bc_eqs) == 2
        assert "continuity" in bc_eqs[0].name
        assert "momentum" in bc_eqs[1].name

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_all_2d(self):
        ins = INSBase(dimension=2)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        raw_eqs = proj.project_all()
        bc_eqs = bc.apply_all(raw_eqs)
        assert len(bc_eqs) == 3
        assert bc_eqs[2].pressure_gradient is not None


# ---------------------------------------------------------------------------
# Integration: Full Phase 1 Pipeline
# ---------------------------------------------------------------------------

class TestPhase1Pipeline:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_full_pipeline_1d(self):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        raw_eqs = proj.project_all()
        final_eqs = bc.apply_all(raw_eqs)

        for eq in final_eqs:
            full = eq.full_equation()
            assert full is not None
            assert full != 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_full_pipeline_2d(self):
        ins = INSBase(dimension=2)
        proj = GalerkinProjection(ins)
        bc = BoundaryConditions(proj)
        raw_eqs = proj.project_all()
        final_eqs = bc.apply_all(raw_eqs)

        assert len(final_eqs) == 3
        for eq in final_eqs:
            full = eq.full_equation()
            assert full is not None
            assert full != 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_projected_equation_pprint_runs(self, capsys):
        ins = INSBase(dimension=1)
        proj = GalerkinProjection(ins)
        eq = proj.project_continuity()
        eq.pprint()
        captured = capsys.readouterr()
        assert "continuity" in captured.out


# ===========================================================================
# Phase 3: Custom Integration Engine
# ===========================================================================

class TestPiecewiseIntegrator:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_smooth_polynomial_no_heaviside(self):
        """Plain polynomial integration should just work."""
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, S.One])
        result = integrator.integrate(z**2)
        assert result == Rational(1, 3)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_single_heaviside_window(self):
        """integral_0^1 z * [H(z) - H(z - 1/2)] dz should be integral_0^{1/2} z dz = 1/8."""
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, Rational(1, 2), S.One])
        expr = z * (Heaviside(z) - Heaviside(z - Rational(1, 2)))
        result = integrator.integrate(expr)
        assert sp.simplify(result - Rational(1, 8)) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_two_layer_heaviside_partition(self):
        """
        Two windows covering [0,1] should sum to full integral.
        W0 = H(z) - H(z-1/2),  W1 = H(z-1/2) - H(z-1)
        integral_0^1 z*(W0 + W1) dz = integral_0^1 z dz = 1/2
        """
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, Rational(1, 2), S.One])
        W0 = Heaviside(z) - Heaviside(z - Rational(1, 2))
        W1 = Heaviside(z - Rational(1, 2)) - Heaviside(z - S.One)
        expr = z * (W0 + W1)
        result = integrator.integrate(expr)
        assert sp.simplify(result - Rational(1, 2)) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_dirac_delta_sifting(self):
        """integral_0^1 f(z) * delta(z - 1/2) dz = f(1/2)."""
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, Rational(1, 2), S.One])
        f = 3 * z**2 + 1
        expr = f * DiracDelta(z - Rational(1, 2))
        result = integrator.integrate(expr)
        expected = f.subs(z, Rational(1, 2))
        assert sp.simplify(result - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_dirac_outside_domain_ignored(self):
        """DiracDelta at z=2 should contribute nothing on [0,1]."""
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, S.One])
        expr = z * DiracDelta(z - 2)
        result = integrator.integrate(expr)
        assert result == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_mixed_smooth_and_delta(self):
        """Smooth part + delta part should combine correctly."""
        z = Symbol("z")
        integrator = PiecewiseIntegrator(z, [S.Zero, Rational(1, 2), S.One])
        W0 = Heaviside(z) - Heaviside(z - Rational(1, 2))
        smooth = z * W0  # integral = 1/8
        delta = 5 * DiracDelta(z - Rational(1, 2))  # contributes 5
        result = integrator.integrate(smooth + delta)
        expected = Rational(1, 8) + 5
        assert sp.simplify(result - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_three_uniform_layers(self):
        """Three layers [0,1/3], [1/3,2/3], [2/3,1] with constant 1 per window."""
        z = Symbol("z")
        interfaces = [Rational(k, 3) for k in range(4)]
        integrator = PiecewiseIntegrator(z, interfaces)
        total = S.Zero
        for k in range(3):
            W = Heaviside(z - interfaces[k]) - Heaviside(z - interfaces[k + 1])
            total += W
        result = integrator.integrate(total)
        assert sp.simplify(result - 1) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_dirac_delta_at_internal_interface(self):
        """Delta at z=1/3 (internal interface) should be picked up."""
        z = Symbol("z")
        interfaces = [Rational(k, 3) for k in range(4)]
        integrator = PiecewiseIntegrator(z, interfaces)
        expr = (2 * z + 1) * DiracDelta(z - Rational(1, 3))
        result = integrator.integrate(expr)
        expected = (2 * Rational(1, 3) + 1)
        assert sp.simplify(result - expected) == 0


# ===========================================================================
# Phase 2: Basis Injection & Layered Ansatz
# ===========================================================================

class TestLayeredAnsatz:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_uniform_interfaces_2_layers(self):
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        assert la.layer_interfaces == [S.Zero, Rational(1, 2), S.One]

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_uniform_interfaces_3_layers(self):
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=3, basis=basis, dimension=1)
        assert la.layer_interfaces == [Rational(k, 3) for k in range(4)]

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_dof_creation_1d(self):
        basis = Legendre_shifted(level=1)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        assert "u" in la.layer_dofs
        assert "v" not in la.layer_dofs
        assert len(la.layer_dofs["u"]) == 2  # 2 layers
        assert len(la.layer_dofs["u"][0]) == 2  # level+1 = 2 basis funcs

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_dof_creation_2d(self):
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=2)
        assert "u" in la.layer_dofs
        assert "v" in la.layer_dofs

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_window_function_partition_of_unity(self):
        """Sum of window functions should be 1 everywhere in (0,1)."""
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=3, basis=basis, dimension=1)
        zeta = la.zeta
        total = sum(la.window_function(k) for k in range(3))
        test_points = [Rational(1, 6), Rational(1, 2), Rational(5, 6)]
        for pt in test_points:
            val = total.subs(zeta, pt)
            assert val == 1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ansatz_single_layer_p0_is_constant(self):
        """1 layer, P0 basis (level=0, monomial=1): ansatz should be u_0_0."""
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=1, basis=basis, dimension=1)
        full = la.full_ansatz("u")
        u00 = la.layer_dofs["u"][0][0]
        zeta = la.zeta
        # Inside [0,1], the window is 1, so ansatz = u_0_0 * 1
        val = full.subs(zeta, Rational(1, 2))
        assert sp.simplify(val - u00) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ansatz_two_layers_p0_piecewise(self):
        """2 layers, P0: left half = u_0_0, right half = u_1_0."""
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        full = la.full_ansatz("u")
        zeta = la.zeta
        u00 = la.layer_dofs["u"][0][0]
        u10 = la.layer_dofs["u"][1][0]
        val_left = full.subs(zeta, Rational(1, 4))
        val_right = full.subs(zeta, Rational(3, 4))
        assert sp.simplify(val_left - u00) == 0
        assert sp.simplify(val_right - u10) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ansatz_integration_over_domain(self):
        """
        2-layer P0 ansatz integrated over [0,1] with PiecewiseIntegrator
        should give (u_0_0 + u_1_0) / 2.
        """
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        full = la.full_ansatz("u")
        integrator = PiecewiseIntegrator(la.zeta, la.layer_interfaces)
        result = integrator.integrate(full)
        u00 = la.layer_dofs["u"][0][0]
        u10 = la.layer_dofs["u"][1][0]
        expected = (u00 + u10) / 2
        assert sp.simplify(result - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_derivative_produces_delta_at_interface(self):
        """
        d/d_zeta of a 2-layer P0 ansatz should produce DiracDelta at
        the interface z=1/2 (from the Heaviside derivative).
        """
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        deriv = la.derivative_ansatz("u")
        assert deriv.has(DiracDelta)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_get_all_dofs(self):
        basis = Legendre_shifted(level=1)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        dofs = la.get_all_dof_symbols()
        assert len(dofs) == 4  # 2 layers * 2 basis funcs
        dofs_u = la.get_all_dof_symbols("u")
        assert len(dofs_u) == 4

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_legendre_ansatz_integration(self):
        """
        1-layer, Legendre level=1 ansatz:
          u(zeta) = u_0_0 * P0(2z-1) + u_0_1 * P1(2z-1)
        Integral over [0,1] should give u_0_0 (since P1 integrates to 0).
        """
        basis = Legendre_shifted(level=1)
        la = LayeredAnsatz(n_layers=1, basis=basis, dimension=1)
        full = la.full_ansatz("u")
        integrator = PiecewiseIntegrator(la.zeta, la.layer_interfaces)
        result = integrator.integrate(full)
        u00 = la.layer_dofs["u"][0][0]
        u01 = la.layer_dofs["u"][0][1]
        result_simplified = sp.simplify(result)
        # P0 integrates to 1, P1 (shifted Legendre) integrates to 0
        assert result_simplified.coeff(u00) != 0
        # The P1 coefficient in the integral should vanish
        assert sp.simplify(result_simplified.coeff(u01)) == 0


# ===========================================================================
# Combined Phase 2+3: Integration of ansatz products
# ===========================================================================

class TestAnsatzIntegration:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_mass_matrix_2layer_p0(self):
        """
        For 2-layer P0 ansatz, the "mass matrix" integral
        integral u^2 dz should produce u_0_0^2/2 + u_1_0^2/2.
        """
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        u = la.full_ansatz("u")
        integrator = PiecewiseIntegrator(la.zeta, la.layer_interfaces)
        result = integrator.integrate(u * u)
        u00 = la.layer_dofs["u"][0][0]
        u10 = la.layer_dofs["u"][1][0]
        expected = u00**2 / 2 + u10**2 / 2
        assert sp.simplify(result - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_delta_from_derivative_integration(self):
        """
        The DiracDelta from differentiating the 2-layer P0 ansatz,
        multiplied by a smooth test function, should give the jump at z=1/2.

        d/dz[ansatz] = u00*delta(z) - (u00-u10)*delta(z-1/2) - u10*delta(z-1)

        When multiplied by zeta (vanishes at z=0, nonzero at z=1/2),
        the internal jump term survives.
        """
        basis = Monomials(level=0)
        la = LayeredAnsatz(n_layers=2, basis=basis, dimension=1)
        deriv = la.derivative_ansatz("u")
        zeta = la.zeta
        integrator = PiecewiseIntegrator(zeta, la.layer_interfaces)
        result = integrator.integrate(deriv * zeta)
        u00 = la.layer_dofs["u"][0][0]
        u10 = la.layer_dofs["u"][1][0]
        result = sp.expand(result)
        assert result.has(u00) or result.has(u10)
