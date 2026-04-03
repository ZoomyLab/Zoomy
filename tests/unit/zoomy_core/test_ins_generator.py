"""
Tests for the composable INS generator (ins_generator.py).
"""

import pytest
import sympy as sp
from sympy import Symbol, Function, Derivative, S, Rational, Integral

from zoomy_core.model.models.ins_generator import (
    StateSpace, Expression, FullINS, IBPResult, Relation, Assumption, Material,
    integrate_by_parts, materials, assumptions,
)


class TestStateSpace:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_1d_creation(self):
        s = StateSpace(dimension=1)
        assert s.dim == 1
        assert s.v == S.Zero

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2d_creation(self):
        s = StateSpace(dimension=2)
        assert s.dim == 2
        assert s.v != S.Zero

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_stress_tensor_complete(self):
        s = StateSpace(dimension=2)
        for i in "xyz":
            for j in "xyz":
                assert i + j in s.tau

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_bathymetry(self):
        s = StateSpace(dimension=1)
        assert s.eta == s.b + s.H


class TestExpression:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_term_access(self):
        x = Symbol("x")
        expr = Expression(x**2 + 3*x + 1)
        assert len(expr) == 3

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_apply_relation_object(self):
        x, y = Symbol("x"), Symbol("y")
        rel = Relation({x: y}, name="swap")
        expr = Expression(x**2 + x)
        result = expr.apply(rel)
        assert result.expr == y**2 + y

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_project_basic(self):
        z = Symbol("z")
        expr = Expression(z**2)
        projected = expr.project(S.One, z, (0, 1))
        assert projected.expr == Integral(z**2, (z, 0, 1))

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ibp_returns_integrate_attribute(self):
        z = Symbol("z")
        f = Function("f")(z)
        expr = Expression(Derivative(f, z))
        result = expr.ibp(var=z, test_weight=S.One, domain=(0, 1))
        assert hasattr(result, "integrate")
        assert not hasattr(result, "volume")

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ibp_raises_if_no_derivative(self):
        z = Symbol("z")
        expr = Expression(z**2)
        with pytest.raises(ValueError):
            expr.ibp(var=z, test_weight=S.One)


class TestRelation:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_relation_from_dict(self):
        x, y = Symbol("x"), Symbol("y")
        r = Relation({x: y}, name="test")
        assert len(r) == 1

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_relation_apply_to(self):
        x, y = Symbol("x"), Symbol("y")
        r = Relation({x: y})
        result = r.apply_to(x**2 + x)
        assert result == y**2 + y

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_relation_display(self):
        x, y = Symbol("x"), Symbol("y")
        r = Relation({x: y}, name="test")
        latex = r._repr_latex_()
        assert "=" in latex

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_assumption_is_relation(self):
        assert issubclass(Assumption, Relation)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_material_is_relation(self):
        assert issubclass(Material, Relation)


class TestFullINS:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_1d_equations_count(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        assert len(ins.equations) == 3

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2d_equations_count(self):
        s = StateSpace(dimension=2)
        ins = FullINS(s)
        assert len(ins.equations) == 4

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_x_momentum_has_stress(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        assert ins.x_momentum.has(s.tau["xz"])

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_z_momentum_has_gravity(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        assert ins.z_momentum.has(s.g)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_z_momentum_has_full_inertia(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        zm = ins.z_momentum
        assert zm.has(Derivative(s.w, s.t))


class TestMaterials:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_newtonian_is_material(self):
        s = StateSpace(dimension=1)
        n = materials.newtonian(s)
        assert isinstance(n, Material)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_newtonian_replaces_tau(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        n = materials.newtonian(s)
        xm = ins.x_momentum.apply(n)
        assert not xm.has(s.tau["xz"])

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_newtonian_introduces_nu(self):
        s = StateSpace(dimension=1)
        n = materials.newtonian(s)
        ins = FullINS(s)
        xm = ins.x_momentum.apply(n)
        assert xm.has(n.nu)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_newtonian_custom_nu(self):
        s = StateSpace(dimension=1)
        my_nu = Symbol("my_viscosity", positive=True)
        n = materials.newtonian(s, nu=my_nu)
        assert n.nu == my_nu

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_inviscid_zeros_all_tau(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        inv = materials.inviscid(s)
        xm = ins.x_momentum.apply(inv)
        for key in s.tau:
            assert not xm.has(s.tau[key])

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_newtonian_displays(self):
        s = StateSpace(dimension=1)
        n = materials.newtonian(s)
        latex = n._repr_latex_()
        assert "=" in latex

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_material_on_single_term(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        n = materials.newtonian(s)
        stress_term = ins.x_momentum[2]
        result = stress_term.apply(n)
        assert not result.has(s.tau["xz"])

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_materials_independent_of_ins(self):
        s = StateSpace(dimension=1)
        n = materials.newtonian(s)
        assert not hasattr(n, "ins")


class TestAssumptions:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_kinematic_bc_bottom_is_assumption(self):
        s = StateSpace(dimension=1)
        kbc = assumptions.kinematic_bc_bottom(s)
        assert isinstance(kbc, Assumption)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_hydrostatic_replaces_p(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        hydro = assumptions.hydrostatic_pressure(s)
        xm = ins.x_momentum.apply(hydro)
        assert not xm.has(s.p)
        assert xm.has(s.g)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_kinematic_bc_applied_to_boundary(self):
        s = StateSpace(dimension=1)
        kbc = assumptions.kinematic_bc_bottom(s)
        w_at_b = s.w.subs(s.z, s.b)
        expr = Expression(w_at_b)
        result = expr.apply(kbc)
        assert result.has(Derivative)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_chain_multiple(self):
        s = StateSpace(dimension=1)
        ins = FullINS(s)
        hydro = assumptions.hydrostatic_pressure(s)
        inv = materials.inviscid(s)
        xm = ins.x_momentum.apply(hydro, inv)
        assert not xm.has(s.p)
        assert not xm.has(s.tau["xz"])

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_assumptions_independent_of_ins(self):
        s = StateSpace(dimension=1)
        kbc = assumptions.kinematic_bc_bottom(s)
        assert not hasattr(kbc, "ins")

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_assumptions_display(self):
        s = StateSpace(dimension=1)
        kbc = assumptions.kinematic_bc_bottom(s)
        latex = kbc._repr_latex_()
        assert "=" in latex

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_2d_kinematic_bc_has_y(self):
        s = StateSpace(dimension=2)
        kbc = assumptions.kinematic_bc_bottom(s)
        for _, rhs in kbc.subs_map.items():
            assert rhs.has(s.y)


class TestIntegrateByParts:

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_basic_ibp(self):
        z = Symbol("z")
        f = Function("f")(z)
        g = Function("g")(z)
        result = integrate_by_parts(f, g, z, domain=(0, 1))
        assert isinstance(result, IBPResult)
        assert result.integrate.has(Integral)

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ibp_boundary_values(self):
        z = Symbol("z")
        result = integrate_by_parts(z**2, S.One, z, domain=(0, 1))
        assert result.boundary_upper.expr == 1
        assert result.boundary_lower.expr == 0

    @pytest.mark.small
    @pytest.mark.unittest
    @pytest.mark.core
    def test_ibp_apply_bcs(self):
        z = Symbol("z")
        tau = Symbol("tau")
        f = Function("f")(z)
        result = integrate_by_parts(f, S.One, z, domain=(0, 1))
        bc_result = result.apply_bcs(bc_lower={f.subs(z, 0): tau})
        assert bc_result.boundary_lower.has(tau)
