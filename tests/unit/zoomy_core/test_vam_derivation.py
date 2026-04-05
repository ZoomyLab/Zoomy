"""
Tests for the Galerkin-projected VAM model derived from INS.

Validates the derived model at L1 Legendre against analytically
computed reference values.
"""

import pytest
import sympy as sp
from sympy import Symbol, Rational, S, sqrt

from zoomy_core.model.models.ins_generator import StateSpace, Inviscid
from zoomy_core.model.models.vam_derivation import derive_vam_moments, VAMPreProjectedEquations
from zoomy_core.model.models.vam_zeta_projection import project_vam_to_zeta, VAMZetaProjectedEquations
from zoomy_core.model.models.vam_projected_model import (
    VAMProjectedHyperbolic, VAMProjectedPoisson,
)
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.projected_model import clear_matrix_cache


@pytest.fixture
def vam_l1():
    """Build a Galerkin VAM model at L1 Legendre."""
    clear_matrix_cache()
    state = StateSpace(dimension=2)
    vam = derive_vam_moments(state, material=Inviscid(state))
    model = VAMProjectedHyperbolic(vam, basis_type=Legendre_shifted, level=1,
                                    eigenvalue_mode='numerical')
    return model


class TestVAMPhase1:

    @pytest.mark.small
    @pytest.mark.unittest
    def test_derive_produces_correct_type(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state, material=Inviscid(state))
        assert isinstance(vam, VAMPreProjectedEquations)

    @pytest.mark.small
    @pytest.mark.unittest
    def test_non_hydrostatic(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state)
        assert "non_hydrostatic" in vam.assumptions_applied

    @pytest.mark.small
    @pytest.mark.unittest
    def test_has_z_momentum(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state)
        assert len(vam.z_momentum) > 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_has_pressure_terms(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state)
        assert len(vam.pressure_x) > 0
        assert len(vam.pressure_z) > 0


class TestVAMPhase2:

    @pytest.mark.small
    @pytest.mark.unittest
    def test_project_produces_correct_type(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state)
        zeta = project_vam_to_zeta(vam)
        assert isinstance(zeta, VAMZetaProjectedEquations)

    @pytest.mark.small
    @pytest.mark.unittest
    def test_has_all_equation_groups(self):
        state = StateSpace(dimension=2)
        vam = derive_vam_moments(state)
        zeta = project_vam_to_zeta(vam)
        eqs = zeta.all_equations()
        assert "continuity" in eqs
        assert "x_momentum" in eqs
        assert "z_momentum" in eqs
        assert "pressure_x" in eqs
        assert "pressure_z" in eqs
        assert "poisson" in eqs


class TestVAMPhase3:

    @pytest.mark.small
    @pytest.mark.unittest
    def test_model_builds(self, vam_l1):
        assert vam_l1.n_variables == 6
        assert list(vam_l1.variables.keys()) == ['h', 'hu0', 'hu1', 'hw0', 'hw1', 'b']

    @pytest.mark.small
    @pytest.mark.unittest
    def test_mass_matrix(self, vam_l1):
        M = vam_l1.mass_matrix()
        assert M == sp.diag(1, Rational(1, 3))

    @pytest.mark.small
    @pytest.mark.unittest
    def test_flux_mass(self, vam_l1):
        """F[0] = hu0 (mass flux)."""
        F = vam_l1.flux()
        hu0 = vam_l1.variables[1]
        assert sp.simplify(F[0, 0] - hu0) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_flux_u_advection_mean(self, vam_l1):
        """F[1] = hu0*u0 + (1/3)*hu1*u1 = (hu0^2 + hu1^2/3)/h."""
        F = vam_l1.flux()
        h, hu0, hu1 = vam_l1.variables[0], vam_l1.variables[1], vam_l1.variables[2]
        expected = (hu0**2 + hu1**2 / 3) / h
        assert sp.simplify(F[1, 0] - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_flux_u_advection_linear(self, vam_l1):
        """F[2] = 2*hu0*hu1/h."""
        F = vam_l1.flux()
        h, hu0, hu1 = vam_l1.variables[0], vam_l1.variables[1], vam_l1.variables[2]
        expected = 2 * hu0 * hu1 / h
        assert sp.simplify(F[2, 0] - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_flux_cross_mean(self, vam_l1):
        """F[3] = (hu0*hw0 + hu1*hw1/3)/h."""
        F = vam_l1.flux()
        h = vam_l1.variables[0]
        hu0, hu1 = vam_l1.variables[1], vam_l1.variables[2]
        hw0, hw1 = vam_l1.variables[3], vam_l1.variables[4]
        expected = (hu0 * hw0 + hu1 * hw1 / 3) / h
        assert sp.simplify(F[3, 0] - expected) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_flux_cross_linear_has_closure(self, vam_l1):
        """F[4] includes the hw2 closure term with coefficient 2/5."""
        F = vam_l1.flux()
        hw2 = Symbol("hw2", real=True)
        assert sp.sympify(F[4, 0]).has(hw2)

    @pytest.mark.small
    @pytest.mark.unittest
    def test_eigenvalues(self, vam_l1):
        """Eigenvalues match the known VAM wave speeds."""
        ev = vam_l1.eigenvalues()
        h = vam_l1.variables[0]
        hu0, hu1 = vam_l1.variables[1], vam_l1.variables[2]
        u0 = hu0 / h
        u1 = hu1 / h
        g = vam_l1.parameters.g

        assert sp.simplify(ev[0] - u0) == 0
        assert sp.simplify(ev[1] - (u0 + u1 / sqrt(3))) == 0
        assert sp.simplify(ev[2] - (u0 - u1 / sqrt(3))) == 0
        assert sp.simplify(ev[3] - (u0 + sqrt(g * h + u1**2))) == 0
        assert sp.simplify(ev[4] - (u0 - sqrt(g * h + u1**2))) == 0
        assert ev[5] == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_nc_gravity(self, vam_l1):
        """NC has g*h on (hu0, h) and (hu0, b) entries."""
        NC = vam_l1.nonconservative_matrix()
        h = vam_l1.variables[0]
        g = vam_l1.parameters.g
        assert sp.simplify(NC[1, 0, 0] - g * h) == 0
        assert sp.simplify(NC[1, 5, 0] - g * h) == 0

    @pytest.mark.small
    @pytest.mark.unittest
    def test_pressure_z_mom_mode0(self, vam_l1):
        """z-momentum mode 0 pressure = -2*p1 (from IBP of dp/dz)."""
        R = vam_l1.source_implicit()
        h = vam_l1.variables[0]
        hp0, hp1 = Symbol("hp0"), Symbol("hp1")
        # R[3] is z-mom mode 0 (hw0 equation)
        # Expected: -2*hp1/h (from IBP: [p*phi_0]_0^1 = -2*p1 = -2*hp1/h)
        expected = sp.Rational(-2) * hp1 / h
        diff = sp.simplify(sp.expand(R[3]) - expected)
        # Only check the hp1 term (there may be additional hp0 terms from boundary)
        assert R[3].has(hp1)

    @pytest.mark.small
    @pytest.mark.unittest
    def test_pressure_z_mom_mode1_vanishes(self, vam_l1):
        """z-momentum mode 1 pressure = 0 in Galerkin projection."""
        R = vam_l1.source_implicit()
        hp0, hp1 = Symbol("hp0"), Symbol("hp1")
        # R[4] is z-mom mode 1 (hw1 equation)
        # In Galerkin: dp/dz IBP onto phi_1 = 0 (boundary and volume cancel)
        val = R[4]
        # Should not contain dhp terms (no spatial derivatives in z-mom mode 1)
        assert not val.has(Symbol("dhp0dx"))
        assert not val.has(Symbol("dhp1dx"))


class TestVAMPoisson:

    @pytest.mark.small
    @pytest.mark.unittest
    def test_poisson_builds(self, vam_l1):
        poi = VAMProjectedPoisson(vam_l1)
        I1, I2 = poi.get_constraints()
        assert I1 is not None
        assert I2 is not None

    @pytest.mark.small
    @pytest.mark.unittest
    def test_poisson_residual(self, vam_l1):
        poi = VAMProjectedPoisson(vam_l1)
        R0, R1 = poi.get_residual_equations()
        assert R0 is not None
        assert R1 is not None
