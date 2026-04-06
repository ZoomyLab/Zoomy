"""
Verification tests for the GeneratedShallowModel.

Checks that the automatically generated equations match the hand-derived
ShallowMomentsTopo for equivalent configurations.
"""

import pytest
import sympy as sp
from sympy import S, Symbol, Rational, simplify, expand, Matrix

from zoomy_core.model.models.generated_shallow_model import GeneratedShallowModel
from zoomy_core.model.models.shallow_moments_topo import ShallowMomentsTopo
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.misc.misc import ZArray


def _symbolic_difference(expr_a, expr_b):
    """Check if two symbolic expressions are equivalent."""
    diff = expand(expr_a - expr_b)
    return simplify(diff)


def _build_substitution_map(gen_model, ref_model):
    """
    Build a symbol substitution map from GeneratedShallowModel variables
    to ShallowMomentsTopo variables so we can compare expressions.

    Both models have Q = [b, h, h*alpha_0, ..., h*alpha_L, ...]
    with the same layout for n_layers=1.
    """
    subs = {}
    for i in range(gen_model.n_variables):
        gen_var = gen_model.variables[i]
        ref_var = ref_model.variables[i]
        if gen_var != ref_var:
            subs[gen_var] = ref_var

    for i in range(gen_model.n_parameters):
        gen_p = gen_model.parameters[gen_model.parameters.keys()[i]]
        ref_p_key = ref_model.parameters.keys()
        for rk in ref_p_key:
            ref_p = ref_model.parameters[rk]
            if gen_p.name == ref_p.name:
                if gen_p != ref_p:
                    subs[gen_p] = ref_p
    return subs


# ===========================================================================
# Basic model creation
# ===========================================================================

class TestGeneratedModelCreation:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_1d_level0_creation(self):
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        assert m.n_variables == 3  # b, h, hu

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_1d_level1_creation(self):
        m = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
        assert m.n_variables == 4  # b, h, h*alpha0, h*alpha1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2d_level0_creation(self):
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=2)
        assert m.n_variables == 4  # b, h, hu, hv

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2d_level1_creation(self):
        m = GeneratedShallowModel(n_layers=1, level=1, dimension=2)
        assert m.n_variables == 6  # b, h, ha0, ha1, hb0, hb1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_multilayer_creation(self):
        m = GeneratedShallowModel(n_layers=3, level=1, dimension=1)
        # 2 + 1 * 3 * 2 = 8
        assert m.n_variables == 8

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_variable_count_matches_smt(self):
        for dim in [1, 2]:
            for lvl in [0, 1, 2]:
                gen = GeneratedShallowModel(n_layers=1, level=lvl, dimension=dim)
                ref = ShallowMomentsTopo(level=lvl, dimension=dim)
                assert gen.n_variables == ref.n_variables, (
                    f"Mismatch for dim={dim}, level={lvl}: "
                    f"gen={gen.n_variables}, ref={ref.n_variables}"
                )


# ===========================================================================
# Algebraic verification: flux
# ===========================================================================

class TestFluxVerification:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_flux_1d_level0(self):
        """
        n_layers=1, level=0, dim=1 should give classical SWE flux.
        F = [hu, hu^2/h] (advective part, without pressure).
        """
        gen = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        ref = ShallowMomentsTopo(level=0, dimension=1)

        F_gen = gen.flux()
        F_ref = ref.flux()

        subs = _build_substitution_map(gen, ref)

        for i in range(gen.n_variables):
            diff = _symbolic_difference(
                F_gen[i, 0].subs(subs), F_ref[i, 0]
            )
            assert diff == 0, (
                f"Flux mismatch at row {i}: "
                f"gen={F_gen[i, 0].subs(subs)}, ref={F_ref[i, 0]}, diff={diff}"
            )

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_flux_1d_level1(self):
        """Level=1: should match ShallowMomentsTopo(level=1)."""
        gen = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
        ref = ShallowMomentsTopo(level=1, dimension=1)

        F_gen = gen.flux()
        F_ref = ref.flux()
        subs = _build_substitution_map(gen, ref)

        for i in range(gen.n_variables):
            diff = _symbolic_difference(
                F_gen[i, 0].subs(subs), F_ref[i, 0]
            )
            assert diff == 0, (
                f"Flux[{i},0] mismatch: diff={diff}"
            )

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_flux_2d_level0(self):
        """2D SWE flux verification."""
        gen = GeneratedShallowModel(n_layers=1, level=0, dimension=2)
        ref = ShallowMomentsTopo(level=0, dimension=2)

        F_gen = gen.flux()
        F_ref = ref.flux()
        subs = _build_substitution_map(gen, ref)

        for i in range(gen.n_variables):
            for d in range(2):
                diff = _symbolic_difference(
                    F_gen[i, d].subs(subs), F_ref[i, d]
                )
                assert diff == 0, (
                    f"Flux[{i},{d}] mismatch: diff={diff}"
                )

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_flux_2d_level1(self):
        """2D shallow moments flux verification."""
        gen = GeneratedShallowModel(n_layers=1, level=1, dimension=2)
        ref = ShallowMomentsTopo(level=1, dimension=2)

        F_gen = gen.flux()
        F_ref = ref.flux()
        subs = _build_substitution_map(gen, ref)

        for i in range(gen.n_variables):
            for d in range(2):
                diff = _symbolic_difference(
                    F_gen[i, d].subs(subs), F_ref[i, d]
                )
                assert diff == 0, (
                    f"Flux[{i},{d}] mismatch: diff={diff}"
                )


# ===========================================================================
# Algebraic verification: hydrostatic pressure
# ===========================================================================

class TestHydrostaticPressureVerification:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_pressure_1d_level0(self):
        gen = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        ref = ShallowMomentsTopo(level=0, dimension=1)
        subs = _build_substitution_map(gen, ref)

        P_gen = gen.hydrostatic_pressure()
        P_ref = ref.hydrostatic_pressure()

        for i in range(gen.n_variables):
            diff = _symbolic_difference(P_gen[i, 0].subs(subs), P_ref[i, 0])
            assert diff == 0, f"Pressure[{i},0] mismatch: diff={diff}"

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_pressure_2d_level1(self):
        gen = GeneratedShallowModel(n_layers=1, level=1, dimension=2)
        ref = ShallowMomentsTopo(level=1, dimension=2)
        subs = _build_substitution_map(gen, ref)

        P_gen = gen.hydrostatic_pressure()
        P_ref = ref.hydrostatic_pressure()

        for i in range(gen.n_variables):
            for d in range(2):
                diff = _symbolic_difference(P_gen[i, d].subs(subs), P_ref[i, d])
                assert diff == 0, f"Pressure[{i},{d}] mismatch: diff={diff}"


# ===========================================================================
# Algebraic verification: nonconservative matrix
# ===========================================================================

class TestNonconservativeVerification:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_nc_1d_level1(self):
        gen = GeneratedShallowModel(n_layers=1, level=1, dimension=1)
        ref = ShallowMomentsTopo(level=1, dimension=1)
        subs = _build_substitution_map(gen, ref)

        NC_gen = gen.nonconservative_matrix()
        NC_ref = ref.nonconservative_matrix()

        for r in range(gen.n_variables):
            for c in range(gen.n_variables):
                diff = _symbolic_difference(
                    NC_gen[r, c, 0].subs(subs), NC_ref[r, c, 0]
                )
                assert diff == 0, f"NC[{r},{c},0] mismatch: diff={diff}"

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_nc_2d_level1(self):
        gen = GeneratedShallowModel(n_layers=1, level=1, dimension=2)
        ref = ShallowMomentsTopo(level=1, dimension=2)
        subs = _build_substitution_map(gen, ref)

        NC_gen = gen.nonconservative_matrix()
        NC_ref = ref.nonconservative_matrix()

        for r in range(gen.n_variables):
            for c in range(gen.n_variables):
                for d in range(2):
                    diff = _symbolic_difference(
                        NC_gen[r, c, d].subs(subs), NC_ref[r, c, d]
                    )
                    assert diff == 0, f"NC[{r},{c},{d}] mismatch: diff={diff}"

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_nc_1d_level0_is_zero(self):
        """Level=0 has no non-conservative terms (no higher moments)."""
        gen = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        NC = gen.nonconservative_matrix()
        for r in range(gen.n_variables):
            for c in range(gen.n_variables):
                assert NC[r, c, 0] == 0, f"NC[{r},{c},0] should be 0"


# ===========================================================================
# Phase 4: Interface routing tests
# ===========================================================================

class TestInterfaceRouting:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_classify_boundary_deltas(self):
        from zoomy_core.model.models.pde_generator import InterfaceRouter, DiracDelta
        z = Symbol("z")
        interfaces = [S.Zero, Rational(1, 2), S.One]
        router = InterfaceRouter(interfaces, z)

        expr = 3 * DiracDelta(z) + 5 * DiracDelta(z - Rational(1, 2)) + 7 * DiracDelta(z - 1)
        classified = router.classify_delta_terms(expr)

        assert classified.n_boundary == 2
        assert classified.n_internal == 1
        assert classified.internal[0].location == Rational(1, 2)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_physical_bcs(self):
        from zoomy_core.model.models.pde_generator import InterfaceRouter, DiracDelta
        z = Symbol("z")
        tau_b = Symbol("tau_b")
        interfaces = [S.Zero, Rational(1, 2), S.One]
        router = InterfaceRouter(interfaces, z)

        expr = z**2 + 3 * DiracDelta(z) + 5 * DiracDelta(z - 1)
        classified = router.classify_delta_terms(expr)
        result = router.apply_physical_bcs(classified, bc_bottom=tau_b, bc_top=S.Zero)
        assert result.has(tau_b)
        assert result.has(z)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_no_internal_for_single_layer(self):
        from zoomy_core.model.models.pde_generator import InterfaceRouter, DiracDelta
        z = Symbol("z")
        interfaces = [S.Zero, S.One]
        router = InterfaceRouter(interfaces, z)

        expr = DiracDelta(z) + DiracDelta(z - 1)
        classified = router.classify_delta_terms(expr)
        assert classified.n_internal == 0
        assert classified.n_boundary == 2


# ===========================================================================
# Multi-layer tests
# ===========================================================================

class TestMultiLayer:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2layer_level0_variable_count(self):
        m = GeneratedShallowModel(n_layers=2, level=0, dimension=1)
        assert m.n_variables == 4  # b, h, hu_layer0, hu_layer1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2layer_level0_flux_shape(self):
        m = GeneratedShallowModel(n_layers=2, level=0, dimension=1)
        F = m.flux()
        assert F.shape == (4, 1)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2layer_mass_flux_is_sum(self):
        """Mass flux should be sum of layer contributions."""
        m = GeneratedShallowModel(n_layers=2, level=0, dimension=1)
        F = m.flux()
        b, h, mu, mv, hinv = m.get_primitives()
        mass_flux = F[1, 0]
        expected = h * (mu[0][0] * Rational(1, 2) + mu[1][0] * Rational(1, 2))
        diff = simplify(expand(mass_flux - expected))
        assert diff == 0, f"Mass flux mismatch: {diff}"

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_multilayer_flux_nonzero(self):
        m = GeneratedShallowModel(n_layers=3, level=1, dimension=1)
        F = m.flux()
        has_nonzero = any(F[i, 0] != 0 for i in range(m.n_variables))
        assert has_nonzero


# ===========================================================================
# Model interface tests
# ===========================================================================

class TestModelInterface:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_eigenvalues_1d_level0(self):
        """SWE eigenvalues: u +/- sqrt(gh)."""
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        evs = m.eigenvalues()
        assert len(evs) == 3  # b is passive -> eigenvalue 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_quasilinear_matrix_exists(self):
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        Q = m.quasilinear_matrix()
        assert Q.shape == (3, 3, 1)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_print_equations_runs(self, capsys):
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        m.print_equations()
        captured = capsys.readouterr()
        assert "Generated Shallow Model" in captured.out

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_functions_registered(self):
        m = GeneratedShallowModel(n_layers=1, level=0, dimension=1)
        assert "flux" in m.functions.keys()
        assert "eigenvalues" in m.functions.keys()
        assert "source" in m.functions.keys()
